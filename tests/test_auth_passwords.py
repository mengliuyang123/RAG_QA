# -*- coding: utf-8 -*-
"""密码存储必须使用带盐Argon2id，错误或损坏记录安全拒绝。"""

from app.infrastructure.auth.passwords import Argon2PasswordHasher


def test_password_hash_is_salted_and_verification_preserves_exact_password() -> None:
    """相同口令不能得到相同哈希，也不能通过去空白改变口令语义。"""
    hasher = Argon2PasswordHasher()
    password = "offline-passphrase-12345"
    first = hasher.hash(password)
    second = hasher.hash(password)
    assert first != second
    assert first.startswith("$argon2id$")
    assert password not in first
    assert hasher.verify(password, first)
    assert not hasher.verify(password + " ", first)
    assert not hasher.verify("incorrect-passphrase", first)


def test_invalid_or_excessively_expensive_hash_is_rejected_before_hashing() -> None:
    """损坏数据库不能让认证抛异常或按恶意参数分配巨量内存。"""
    hasher = Argon2PasswordHasher()
    assert not hasher.verify("password", "not-a-hash")
    valid = hasher.hash("offline-passphrase-12345")
    excessive = valid.replace("m=19456", "m=1048576")
    assert not hasher.verify("offline-passphrase-12345", excessive)
