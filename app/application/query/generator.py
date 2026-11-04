# -*- coding: utf-8 -*-
"""使用生成端口根据受控提示词生成答案。"""

from collections.abc import Iterator

from app.application.ports.generation import LLMProvider
from app.application.query.context_budget import PromptBudgetGuard
from app.application.query.prompting import PromptBuilder
from app.domain.agent_quality import EvidenceItem
from app.domain.retrieval import RetrievalResult


class AnswerGenerator:
    """组合提示词构建器与可替换的语言模型实现。"""

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        llm_provider: LLMProvider,
        prompt_guard: PromptBudgetGuard | None = None,
    ) -> None:
        """装配生成回答所需的组件。

        Args:
            prompt_builder: 把问题和检索结果转换为聊天消息的
                组件。
            llm_provider: 接收聊天消息并生成回答的端口实现。
            prompt_guard: 完整输入预算检查器；生产组合根注入。
        """
        self._prompt_builder = prompt_builder
        self._llm_provider = llm_provider
        self._prompt_guard = prompt_guard

    def generate(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> str:
        """构建受控消息并调用一次语言模型。

        Args:
            question: 用户问题。
            results: 有序检索上下文。

        Returns:
            语言模型生成的完整回答。
        """
        messages = self._prompt_builder.build(question, results)
        if self._prompt_guard is not None:
            self._prompt_guard.ensure(messages)
        return self._llm_provider.generate(messages)

    def stream(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> Iterator[str]:
        """使用与非流式生成相同的提示词返回文本增量。

        Args:
            question: 用户问题。
            results: 有序检索上下文。

        Yields:
            语言模型返回的非空文本增量。
        """
        messages = self._prompt_builder.build(question, results)
        if self._prompt_guard is not None:
            self._prompt_guard.ensure(messages)
        yield from self._llm_provider.stream(messages)

    def generate_grounded(
        self, question: str, history: list[dict[str, str]],
        evidence: list[EvidenceItem],
    ) -> str:
        """仅用批准证据生成草稿，保留过滤前的引用编号。

        Args:
            question: 当前原始问题。
            history: 用于理解指代的已选历史。
            evidence: 裁判批准的稳定编号证据。

        Returns:
            尚待规则校验的完整回答。
        """
        messages = self._prompt_builder.build_numbered(
            question, history, evidence,
        )
        if self._prompt_guard is not None:
            self._prompt_guard.ensure(messages)
        return self._llm_provider.generate(messages)
