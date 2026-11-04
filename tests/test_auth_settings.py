# -*- coding: utf-8 -*-
"""浏览器认证配置不能用通配来源或不完整的URL绕过来源校验。"""

import pytest
from pydantic import ValidationError

from app.config import Settings


@pytest.mark.parametrize("options", [
    {"cors_allowed_origins": ("*",)},
    {"cors_allowed_origins": ("https://safe.example/path",)},
    {"cors_allowed_origins": ("https://user:password@safe.example",)},
    {"cors_allowed_origins": ("null",)},
    {"http_allowed_hosts": ("*",)},
    {"http_allowed_hosts": ()},
    {"http_allowed_hosts": ("https://safe.example",)},
])
def test_unsafe_browser_auth_configuration_fails_closed(options: dict[str, object]) -> None:
    """错误的显式来源应在启动前失败，不能静默放行。

    Args:
        options: 不安全的来源或Host配置。
    """
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **options)
