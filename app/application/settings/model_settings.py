# -*- coding: utf-8 -*-
"""管理员可在线修改的模型配置：字段白名单、展示视图与新配置校验。

只开放模型相关的配置（API Key、服务地址、模型名、向量维度）。数据库、
认证、限流等基础设施配置不在白名单内，改错会导致服务无法启动。
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from app.config import Settings

FieldKind = Literal["secret", "url", "text", "int"]


class ModelSettingsError(ValueError):
    """提交的配置无法应用；message 可直接展示给管理员。"""


@dataclass(frozen=True)
class SettingField:
    """一个可编辑配置项。

    Attributes:
        name: ``Settings`` 上的属性名，也是 .env 键名的小写形式。
        label: 界面显示名。
        kind: 输入类型，secret 表示 API Key。
        hint: 输入框下方的说明。
    """

    name: str
    label: str
    kind: FieldKind
    hint: str = ""

    @property
    def env_key(self) -> str:
        """返回写入 .env 时使用的大写键名。"""
        return self.name.upper()


@dataclass(frozen=True)
class SettingGroup:
    """界面上的一组配置。"""

    id: str
    title: str
    description: str
    fields: tuple[SettingField, ...]


INHERIT_HINT = "留空时沿用生成模型的配置"

SETTING_GROUPS: tuple[SettingGroup, ...] = (
    SettingGroup(
        "llm", "生成模型", "Agent 规划、检索决策和最终回答使用的主模型。",
        (
            SettingField("llm_api_key", "API Key", "secret",
                         "留空且地址为 OpenAI 官方时，使用下方的 OpenAI 通用 Key"),
            SettingField("llm_base_url", "API 地址", "url",
                         "OpenAI 兼容接口的根地址，例如 https://api.openai.com/v1"),
            SettingField("llm_model", "模型名", "text"),
        ),
    ),
    SettingGroup(
        "understanding", "问题理解模型", "改写问题、判断走哪条处理通道。可以单独配一个更快更便宜的模型。",
        (
            SettingField("understanding_api_key", "API Key", "secret", INHERIT_HINT),
            SettingField("understanding_base_url", "API 地址", "url", INHERIT_HINT),
            SettingField("understanding_model", "模型名", "text", INHERIT_HINT),
        ),
    ),
    SettingGroup(
        "judge", "质量审核模型", "判断检索证据是否充分、审核回答质量。",
        (
            SettingField("judge_api_key", "API Key", "secret", INHERIT_HINT),
            SettingField("judge_base_url", "API 地址", "url", INHERIT_HINT),
            SettingField("judge_model", "模型名", "text", INHERIT_HINT),
        ),
    ),
    SettingGroup(
        "embedding", "向量模型", "文档入库和提问检索时把文本转成向量。更换模型或维度后，已有向量不再兼容。",
        (
            SettingField("embedding_api_key", "API Key", "secret",
                         "留空且地址为 OpenAI 官方时，使用下方的 OpenAI 通用 Key"),
            SettingField("embedding_base_url", "API 地址", "url"),
            SettingField("embedding_model", "模型名", "text"),
            SettingField("embedding_dimensions", "向量维度", "int",
                         "留空使用模型默认维度；只有支持 dimensions 参数的模型（如 text-embedding-3）才能指定"),
        ),
    ),
    SettingGroup(
        "reranker", "重排序模型", "hybrid_rerank 策略下对召回结果精排。",
        (
            SettingField("reranker_api_key", "API Key", "secret", "不配置时不启用重排序"),
            SettingField("reranker_base_url", "API 地址", "url"),
            SettingField("reranker_model", "模型名", "text"),
        ),
    ),
    SettingGroup(
        "shared", "OpenAI 通用 Key", "各模型地址为 OpenAI 官方且没有单独配置 Key 时使用。",
        (SettingField("openai_api_key", "API Key", "secret"),),
    ),
)

FIELDS_BY_NAME: dict[str, SettingField] = {
    field.name: field for group in SETTING_GROUPS for field in group.fields
}

# 这些字段变化后需要重新验证向量维度。
EMBEDDING_FIELDS = frozenset({
    "embedding_api_key", "embedding_base_url", "embedding_model",
    "embedding_dimensions", "openai_api_key",
})


def _default(name: str) -> object:
    return Settings.model_fields[name].default


def normalize_updates(raw: Mapping[str, object]) -> dict[str, object]:
    """把界面提交的值转换成 ``Settings`` 字段值。

    空字符串和 None 都表示"删除这项配置"：可选字段变为 None，
    有默认值的字段回到代码默认值。

    Args:
        raw: 字段名到提交值的映射。

    Returns:
        字段名到规范化后取值的映射。

    Raises:
        ModelSettingsError: 出现白名单外的字段，或数值无法解析。
    """
    updates: dict[str, object] = {}
    for name, value in raw.items():
        field = FIELDS_BY_NAME.get(name)
        if field is None:
            raise ModelSettingsError(f"不支持在线修改的配置项：{name}")
        if value is None or (isinstance(value, str) and not value.strip()):
            updates[name] = _default(name)
            continue
        if field.kind == "int":
            try:
                updates[name] = int(str(value).strip())
            except ValueError as exc:
                raise ModelSettingsError(f"{field.label}必须是整数") from exc
            continue
        if not isinstance(value, str):
            raise ModelSettingsError(f"{field.label}必须是字符串")
        updates[name] = value.strip()
    return updates


def build_settings(current: Settings, updates: Mapping[str, object]) -> Settings:
    """在当前配置上应用修改并完整校验。

    直接用 ``model_validate`` 构造，不重新读取进程环境变量和 .env，
    因此新配置等于"当前运行配置 + 本次修改"。

    Args:
        current: 正在运行的配置。
        updates: ``normalize_updates`` 的结果。

    Returns:
        通过全部字段约束和模型校验器的新配置。

    Raises:
        ModelSettingsError: 新配置不合法。
    """
    data = current.model_dump()
    data.update(updates)
    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first.get("loc", ()))
        label = FIELDS_BY_NAME[location].label if location in FIELDS_BY_NAME else location
        raise ModelSettingsError(f"配置不合法（{label}）：{first.get('msg', '')}") from exc


def env_values(updates: Mapping[str, object]) -> dict[str, str | None]:
    """把规范化后的修改转换成写入 .env 的字符串。

    Args:
        updates: ``normalize_updates`` 的结果。

    Returns:
        大写键名到字符串的映射；等于默认值的项为 None，表示从 .env 删除。
    """
    result: dict[str, str | None] = {}
    for name, value in updates.items():
        key = FIELDS_BY_NAME[name].env_key
        result[key] = None if value == _default(name) or value is None else str(value)
    return result


def changed_fields(current: Settings, updated: Settings) -> list[str]:
    """返回实际发生变化的可编辑字段名。"""
    return [
        name for name in FIELDS_BY_NAME
        if getattr(current, name) != getattr(updated, name)
    ]


def settings_view(
    settings: Settings,
    environment: Mapping[str, str],
    stored_dimensions: int | None,
) -> dict[str, object]:
    """生成管理页面使用的配置视图。

    按产品要求，API Key 以明文返回；接口只对管理员开放并禁止缓存。

    Args:
        settings: 当前运行配置。
        environment: 进程环境变量，用于提示哪些键会在重启后被环境变量覆盖。
        stored_dimensions: 当前向量索引已锁定的维度，空库为 None。

    Returns:
        可直接序列化为 JSON 的视图。
    """
    upper_env = {key.upper() for key in environment}
    groups = []
    for group in SETTING_GROUPS:
        fields = []
        for field in group.fields:
            value = getattr(settings, field.name)
            default = _default(field.name)
            fields.append({
                "name": field.name,
                "env_key": field.env_key,
                "label": field.label,
                "kind": field.kind,
                "hint": field.hint,
                "value": value,
                "default": default,
                "overridden_by_environment": field.env_key in upper_env,
            })
        groups.append({
            "id": group.id, "title": group.title,
            "description": group.description, "fields": fields,
        })
    return {"groups": groups, "stored_embedding_dimensions": stored_dimensions}
