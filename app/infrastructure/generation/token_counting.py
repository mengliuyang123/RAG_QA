# -*- coding: utf-8 -*-
"""为未声明 tokenizer 的兼容模型提供无需下载的保守输入上界。"""


class Utf8TokenUpperBoundCounter:
    """按 UTF-8 字节保守预算，不冒充实际分词结果或计费 usage。

    适用于当前常见字节／子词兼容模型的文本预算；角色及工具模板开销
    由窗口与外层保留预算覆盖。可通过 TokenCounter 替换模型匹配计数器。
    """

    def count_text(self, text: str) -> int:
        """给出文本 token 的保守字节上界。

        Args:
            text: 原始文本。

        Returns:
            UTF-8 编码长度，不是 provider 的 usage。
        """
        return len(text.encode("utf-8"))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """取完整 Unicode 前缀，避免切断中文或 emoji 字节序列。

        Args:
            text: 助手文本。
            max_tokens: 采用相同保守口径的预算。

        Returns:
            安全解码且不超过预算的前缀。

        Raises:
            ValueError: 预算为负数。
        """
        if max_tokens < 0:
            raise ValueError("截断预算不能为负数")
        return text.encode("utf-8")[:max_tokens].decode("utf-8", errors="ignore")
