# -*- coding: utf-8 -*-
"""使用Argon2id存储口令，不实现自定义密码算法。"""

from argon2 import PasswordHasher
from argon2 import Type
from argon2 import extract_parameters
from argon2.exceptions import InvalidHashError
from argon2.exceptions import VerificationError


class Argon2PasswordHasher:
    """使用固定安全参数，并拒绝损坏记录中的超高计算成本。"""

    def __init__(self) -> None:
        """采用Argon2id最低19MiB、两轮、单并行度的明确参数。

        Returns:
            None。
        """
        self._hasher = PasswordHasher(
            memory_cost=19456, time_cost=2, parallelism=1,
            salt_len=16, hash_len=32, type=Type.ID,
        )

    def hash(self, password: str) -> str:
        """为每次存储生成独立随机盐。

        Args:
            password: 未规范化或截断的完整口令。

        Returns:
            包含算法、参数和盐的编码哈希。
        """
        return self._hasher.hash(password)

    def verify(self, password: str, encoded: str) -> bool:
        """验证原始口令，格式错误或不合理参数均拒绝。

        Args:
            password: 未修改的输入口令。
            encoded: 已存储的编码哈希。

        Returns:
            是否匹配；不向外泄露损坏哈希的细节。
        """
        try:
            parameters = extract_parameters(encoded)
            if parameters.type is not Type.ID or parameters.version != 19:
                return False
            if not 19456 <= parameters.memory_cost <= 262144:
                return False
            if not 2 <= parameters.time_cost <= 10:
                return False
            if not 1 <= parameters.parallelism <= 8:
                return False
            if parameters.salt_len < 16 or parameters.hash_len < 32:
                return False
            return self._hasher.verify(encoded, password)
        except (InvalidHashError, VerificationError, ValueError):
            return False
