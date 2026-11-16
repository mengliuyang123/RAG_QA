# -*- coding: utf-8 -*-
"""模型配置在线修改：字段白名单、.env 原地更新、容器热替换和向量维度保护。"""

import time
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from dotenv import dotenv_values
from fastapi import FastAPI

from app.application.settings.model_settings import ModelSettingsError
from app.application.settings.model_settings import build_settings
from app.application.settings.model_settings import env_values
from app.application.settings.model_settings import normalize_updates
from app.application.settings.model_settings import settings_view
from app.config import Settings
from app.domain.auth import User
from app.infrastructure.settings.env_file import EnvFile
from app.interfaces.http.auth import require_admin
from app.interfaces.http.container_runtime import ContainerLeaseMiddleware
from app.interfaces.http.container_runtime import ContainerRuntime
from app.interfaces.http.routes.settings_admin import router


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "postgresql://u:p@localhost/db",
        "llm_api_key": "sk-old", "llm_model": "old-model", **overrides,
    }
    return Settings(_env_file=None, **values)


class _Embedder:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self.dimensions


class _VectorStore:
    def __init__(self, stored: int | None) -> None:
        self.stored = stored

    def stored_dimensions(self) -> int | None:
        return self.stored


class _Container:
    """只实现配置路由和热替换用到的容器接口。"""

    def __init__(self, settings: Settings, stored: int | None, output: int) -> None:
        self.settings = settings
        self.vector_store = _VectorStore(stored)
        self.embedding_provider = _Embedder(output)
        self.closed = False
        self.worker_started = False

    def close(self) -> None:
        self.closed = True

    def start_ingestion_worker(self) -> None:
        self.worker_started = True


def _app(tmp_path: Path, *, stored: int | None = 1536, output: int = 1536):
    built: list[_Container] = []

    def factory(settings: Settings) -> _Container:
        container = _Container(settings, stored, output)
        built.append(container)
        return container

    application = FastAPI()
    application.state.container = factory(_settings())
    application.state.env_file_path = tmp_path / ".env"
    runtime = ContainerRuntime(application.state, factory)
    application.state.container_runtime = runtime
    application.add_middleware(ContainerLeaseMiddleware, runtime=runtime)
    application.include_router(router, prefix="/api")
    application.dependency_overrides[require_admin] = (
        lambda: User(id=uuid4(), name="管理员", role="admin")
    )
    return application, built


def _wait_until(condition, timeout: float = 2.0) -> None:
    """旧容器在后台线程关闭，等待它完成。"""
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)


def _client(application: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://test")


def test_normalize_rejects_fields_outside_whitelist() -> None:
    """数据库连接等基础设施配置不能经由该接口修改。"""
    with pytest.raises(ModelSettingsError, match="database_url"):
        normalize_updates({"database_url": "postgresql://evil"})


def test_blank_values_restore_defaults_and_are_removed_from_env() -> None:
    """清空有默认值的字段会回到默认值，并从 .env 删除而不是写空串。"""
    updates = normalize_updates({"llm_model": "", "llm_api_key": None, "embedding_dimensions": "256"})
    updated = build_settings(_settings(), updates)
    assert updated.llm_model == Settings.model_fields["llm_model"].default
    assert updated.llm_api_key is None
    assert updated.embedding_dimensions == 256
    assert env_values(updates) == {
        "LLM_MODEL": None, "LLM_API_KEY": None, "EMBEDDING_DIMENSIONS": "256",
    }


def test_invalid_values_report_the_field_label() -> None:
    """违反字段约束时给出可读的字段名。"""
    with pytest.raises(ModelSettingsError, match="向量维度"):
        build_settings(_settings(), normalize_updates({"embedding_dimensions": "0"}))
    with pytest.raises(ModelSettingsError, match="整数"):
        normalize_updates({"embedding_dimensions": "abc"})


def test_view_returns_plaintext_keys_and_flags_environment_overrides() -> None:
    """按产品要求返回明文 Key，并标出会被进程环境变量覆盖的键。"""
    view = settings_view(_settings(), {"llm_model": "x"}, 1536)
    fields = {field["name"]: field for group in view["groups"] for field in group["fields"]}
    assert fields["llm_api_key"]["value"] == "sk-old"
    assert fields["llm_model"]["overridden_by_environment"] is True
    assert fields["llm_api_key"]["overridden_by_environment"] is False
    assert view["stored_embedding_dimensions"] == 1536
    assert "database_url" not in fields


def test_env_file_update_preserves_other_lines_and_round_trips(tmp_path: Path) -> None:
    """只改目标键：保留注释和其他键，去掉重复键，特殊字符能被 dotenv 原样读回。"""
    path = tmp_path / ".env"
    path.write_text("# 注释\nLLM_MODEL=old\nOTHER=1\nllm_model=dup\nOPENAI_API_KEY=a\n", encoding="utf-8")
    tricky = 'sk-a b#c"d\\e=f'
    EnvFile(path).update({"llm_model": "new", "openai_api_key": None, "LLM_API_KEY": tricky})
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# 注释\nLLM_MODEL=new\nOTHER=1\n")
    assert "dup" not in text
    values = dotenv_values(path)
    assert values["LLM_API_KEY"] == tricky
    assert "OPENAI_API_KEY" not in values


def test_env_file_round_trip_keeps_crlf_and_missing_final_newline(tmp_path: Path) -> None:
    # Windows 上的 .env 常是 CRLF 且末尾无换行；只改目标行，其余字节保持不变。
    path = tmp_path / ".env"
    original = b"# note\r\nA=1\r\nLLM_MODEL=old\r\nB=2"
    path.write_bytes(original)

    EnvFile(path).update({"x_test": "1"})
    EnvFile(path).update({"x_test": None})
    assert path.read_bytes() == original

    EnvFile(path).update({"llm_model": "new"})
    assert path.read_bytes() == b"# note\r\nA=1\r\nLLM_MODEL=new\r\nB=2"


def test_env_file_writes_in_place_when_mount_point_cannot_be_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Docker 单文件挂载：rename 到挂载点会 EBUSY，必须原地写入且不留临时文件。
    import errno
    import os

    path = tmp_path / ".env"
    path.write_text("# keep\nLLM_MODEL=old\n", encoding="utf-8")

    def busy(*_args: object) -> None:
        raise OSError(errno.EBUSY, "Device or resource busy")

    monkeypatch.setattr(os, "replace", busy)
    EnvFile(path).update({"llm_model": "new"})

    assert path.read_text(encoding="utf-8") == "# keep\nLLM_MODEL=new\n"
    assert list(tmp_path.iterdir()) == [path]


def test_env_file_writes_in_place_when_directory_is_not_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 非 root 容器用户不能在 /app 下创建临时文件，但能写挂载进来的 .env。
    import tempfile

    path = tmp_path / ".env"
    path.write_text("LLM_MODEL=old\n", encoding="utf-8")

    def denied(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(tempfile, "mkstemp", denied)
    EnvFile(path).update({"llm_model": "new"})

    assert path.read_text(encoding="utf-8") == "LLM_MODEL=new\n"


def test_env_file_still_raises_unrelated_rename_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import errno
    import os

    path = tmp_path / ".env"
    path.write_text("LLM_MODEL=old\n", encoding="utf-8")

    def full(*_args: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "replace", full)
    with pytest.raises(OSError):
        EnvFile(path).update({"llm_model": "new"})
    assert path.read_text(encoding="utf-8") == "LLM_MODEL=old\n"
    assert list(tmp_path.iterdir()) == [path]


def test_runtime_keeps_old_container_open_until_its_requests_finish() -> None:
    """正在进行的请求（例如流式回答）结束前，旧容器不能被关闭。"""
    state = type("State", (), {})()
    old = _Container(_settings(), None, 3)
    new = _Container(_settings(llm_model="new"), None, 3)
    state.container = old
    runtime = ContainerRuntime(state, lambda settings: new)
    runtime.start_ingestion_worker()

    lease = runtime.acquire()
    runtime.swap(new)
    assert runtime.current is new
    assert new.worker_started is True
    assert old.closed is False

    runtime.release(lease)
    _wait_until(lambda: old.closed)
    assert old.closed is True
    assert new.closed is False


@pytest.mark.anyio
async def test_put_writes_env_and_hot_swaps_container(tmp_path: Path) -> None:
    """保存后 .env 更新，之后的请求立即读到新配置，旧容器被关闭。"""
    application, built = _app(tmp_path)
    async with _client(application) as client:
        response = await client.put(
            "/api/admin/settings/models",
            json={"values": {"llm_model": "gpt-4.1", "llm_api_key": "sk-new"}},
        )
        assert response.status_code == 200, response.text
        assert response.headers["Cache-Control"] == "no-store"
        assert len(built) == 2
        assert application.state.container is built[1]
        assert built[1].settings.llm_model == "gpt-4.1"

        current = (await client.get("/api/admin/settings/models")).json()
        fields = {field["name"]: field for group in current["groups"] for field in group["fields"]}
        assert fields["llm_model"]["value"] == "gpt-4.1"
        assert fields["llm_api_key"]["value"] == "sk-new"

    values = dotenv_values(tmp_path / ".env")
    assert values["LLM_MODEL"] == "gpt-4.1"
    assert values["LLM_API_KEY"] == "sk-new"
    _wait_until(lambda: built[0].closed)
    assert built[0].closed is True


@pytest.mark.anyio
async def test_embedding_dimension_mismatch_is_rejected_without_side_effects(tmp_path: Path) -> None:
    """新向量模型维度与已入库向量不同时拒绝保存：不写 .env、不替换容器。"""
    application, built = _app(tmp_path, stored=1536, output=1024)
    async with _client(application) as client:
        response = await client.put(
            "/api/admin/settings/models",
            json={"values": {"embedding_model": "bge-m3"}},
        )
    assert response.status_code == 409
    assert "1024" in response.json()["detail"] and "1536" in response.json()["detail"]
    assert application.state.container is built[0]
    assert built[1].closed is True
    assert not (tmp_path / ".env").exists()


@pytest.mark.anyio
async def test_embedding_change_allowed_when_index_is_empty(tmp_path: Path) -> None:
    """还没有向量入库时，可以自由更换向量模型和维度。"""
    application, built = _app(tmp_path, stored=None, output=1024)
    async with _client(application) as client:
        response = await client.put(
            "/api/admin/settings/models",
            json={"values": {"embedding_model": "bge-m3", "embedding_dimensions": 1024}},
        )
    assert response.status_code == 200, response.text
    assert application.state.container is built[1]
    assert dotenv_values(tmp_path / ".env")["EMBEDDING_DIMENSIONS"] == "1024"


@pytest.mark.anyio
async def test_unknown_field_and_no_op_updates(tmp_path: Path) -> None:
    """白名单外字段返回 422；没有实际变化时不创建新容器、不写文件。"""
    application, built = _app(tmp_path)
    async with _client(application) as client:
        rejected = await client.put("/api/admin/settings/models", json={"values": {"log_level": "DEBUG"}})
        unchanged = await client.put("/api/admin/settings/models", json={"values": {"llm_model": "old-model"}})
    assert rejected.status_code == 422
    assert unchanged.status_code == 200
    assert len(built) == 1
    assert not (tmp_path / ".env").exists()
