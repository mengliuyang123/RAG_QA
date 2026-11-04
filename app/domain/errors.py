# -*- coding: utf-8 -*-
"""定义违反检索、上下文与执行预算不变量时使用的稳定异常。"""


class RetrievalError(ValueError):
    """表示检索请求不符合领域约束。"""


class InvalidQuestionError(RetrievalError):
    """表示问题为空白。"""


class InvalidTopKError(RetrievalError):
    """表示请求结果数量超出支持范围。"""


class UnsupportedRetrievalStrategyError(RetrievalError):
    """表示请求了尚未实现的检索策略。"""


class ExecutionDeadlineError(RuntimeError):
    """表示执行的共享截止时间已经到达。"""


class RetrievalBudgetExceededError(RuntimeError):
    """表示整轮检索次数已经达到上限。"""


class DuplicateRetrievalError(RuntimeError):
    """表示同一轮重复执行规范化查询及其相同范围。"""


class HistoryContextTooLargeError(ValueError):
    """表示当前用户全文无法放入预留后的模型输入预算。"""


class PromptContextTooLargeError(ValueError):
    """表示完整提示词及协议描述无法放入模型输入预算。"""


class InvalidKnowledgeScopeError(ValueError):
    """表示用户指定了目录中不存在或不允许的知识库范围。"""


class InvalidConversationCursorError(ValueError):
    """表示会话分页参数非法或游标与当前资源、身份不符。"""
