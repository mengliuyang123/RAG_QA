# -*- coding: utf-8 -*-
"""定义用例、适配器和资源操作使用的稳定异常。"""


class InvalidCredentialsError(PermissionError):
    """表示账号不存在、口令不匹配或已禁用，公开响应不能区分。"""


class UnauthenticatedError(PermissionError):
    """表示未登录、会话已失效或已被撤销。"""


class ForbiddenError(PermissionError):
    """表示身份有效但不具有目标操作权限。"""


class CSRFError(PermissionError):
    """表示来源或写操作的CSRF令牌校验失败。"""


class AuthStorageError(RuntimeError):
    """表示认证存储不可用，不能回退为匿名访问。"""


class UserAlreadyExistsError(ValueError):
    """表示管理员创建了重复登录标识。"""


class LoginRateLimitError(RuntimeError):
    """表示登录尝试或口令验证并发超过限制。"""

    def __init__(self, retry_after: int) -> None:
        """保存安全的等待秒数，不携带账号或口令。

        Args:
            retry_after: 最短重试等待秒数。

        Returns:
            None。
        """
        super().__init__("登录尝试过于频繁")
        self.retry_after = max(1, retry_after)


class TraceStorageError(RuntimeError):
    """表示追踪数据库读写失败或内容损坏。"""


class TraceNotFoundError(LookupError):
    """表示追踪不存在或已超过其保留期限。"""


class TurnNotFoundError(LookupError):
    """表示本人可访问的正式轮次不存在，不区分越权与删除。"""


class ShareNotFoundError(LookupError):
    """分享不存在或对应正式会话已删除。"""


class ShareExpiredError(RuntimeError):
    """分享授权已超过明确的有效期。"""


class ShareStorageError(RuntimeError):
    """分享授权或正文读取失败，不能返回部分内容。"""


class EvaluationReportNotFoundError(LookupError):
    """尚未发布完整评估报告。"""


class EvaluationReportStorageError(RuntimeError):
    """报告损坏或不可读，不能回退为无评估。"""


class FeedbackNotFoundError(LookupError):
    """表示当前可处理的点踩记录不存在或已取消。"""


class FeedbackChangedError(ValueError):
    """表示用户已修改反馈，管理员需要刷新后重新分类。"""


class InsightStorageError(RuntimeError):
    """表示后台洞察读取或处理存储不可用，不返回伪造的空数据。"""


class RefusalClustersChangedError(RuntimeError):
    """拒答成员或分组已变化，旧游标不可继续使用。"""


class RefusalClustersBusyError(RuntimeError):
    """同一运行已有聚类请求，避免并发重复向量化。"""


class RefusalClustersTimeoutError(RuntimeError):
    """聚类超过协作式预算，不返回部分统计冒充完整结果。"""


class UnsupportedFileTypeError(ValueError):
    """表示文件扩展名没有对应解析器。"""


class DocumentParsingError(ValueError):
    """表示文档存在但无法解析。"""


class EmbeddingError(RuntimeError):
    """表示向量化响应不完整或不一致。"""


class EmbeddingTimeoutError(EmbeddingError):
    """表示向量服务请求超时。"""


class EmbeddingRateLimitError(EmbeddingError):
    """表示向量服务拒绝过量请求。"""


class EmbeddingAuthenticationError(EmbeddingError):
    """表示向量服务认证失败。"""


class EmbeddingResponseError(EmbeddingError):
    """表示向量服务响应不符合稳定契约。"""


class MissingAPIKeyError(EmbeddingError):
    """表示生产向量客户端缺少 API Key。"""


class LLMError(RuntimeError):
    """表示语言模型调用失败。"""


class LLMMissingAPIKeyError(LLMError):
    """表示生产语言模型客户端没有可用凭据。"""


class LLMTimeoutError(LLMError):
    """表示语言模型请求超时。"""


class LLMRateLimitError(LLMError):
    """表示语言模型服务拒绝过量请求。"""


class LLMAuthenticationError(LLMError):
    """表示语言模型服务认证失败。"""


class LLMGenerationError(LLMError):
    """表示语言模型响应无法生成有效答案。"""


class RerankerUnavailableError(RuntimeError):
    """表示远程重排服务缺少配置、调用失败或响应无效。"""


class VectorStoreError(RuntimeError):
    """表示向量存储输入或响应不符合约定。"""


class MissingKeywordRetrieverError(RuntimeError):
    """表示应用未装配混合检索所需的关键词检索器。"""


class MissingRerankerError(RerankerUnavailableError):
    """表示应用未装配两阶段检索所需的重排器。"""


class UploadError(ValueError):
    """表示上传流程中可映射为客户端错误的失败。"""


class MissingFilenameError(UploadError):
    """表示上传请求缺少有效文件名。"""


class EmptyDocumentError(UploadError):
    """表示上传内容为空或无法提取文本。"""


class FileTooLargeError(UploadError):
    """表示上传内容超过配置上限。"""


class DocumentNotFoundError(LookupError):
    """表示目标文档在向量文本块中不存在。"""


class DocumentRevisionChangedError(RuntimeError):
    """表示分页游标所属的索引元数据修订已改变，必须从首页重读。"""


class DocumentStorageCorruptionError(RuntimeError):
    """表示向量来源元数据无法安全映射到唯一文件。"""


class DocumentIngestionInProgressError(RuntimeError):
    """表示该文档仍有可继续执行的持久任务，删除不代表取消任务。"""


class DocumentDeletePartialError(RuntimeError):
    """表示索引已删除但源文件清理失败。"""

    def __init__(self) -> None:
        """创建需要人工修复的稳定部分失败异常。"""
        super().__init__("文档索引已删除，但原文件清理失败")
        self.repair_required = True


class ConversationNotFoundError(LookupError):
    """表示目标会话不存在。"""


class ConversationBusyError(RuntimeError):
    """表示同一会话已有一轮请求正在运行。"""


class ConversationDeletingError(RuntimeError):
    """表示目标会话正在执行可重试删除。"""


class ConversationStorageError(RuntimeError):
    """表示业务 SQLite 操作失败或数据不满足契约。"""


class ConversationAgentError(RuntimeError):
    """表示 Agent 图或其模型执行失败。"""


class ConversationTimeoutError(ConversationAgentError):
    """表示会话模型调用超时或整轮协作式时间预算耗尽。"""


class ConversationDeleteError(RuntimeError):
    """表示 checkpoint 或业务历史删除未完整完成。"""


class KnowledgeBaseStorageError(RuntimeError):
    """表示知识库目录无法读取或持久化。"""


class KnowledgeBaseNotFoundError(LookupError):
    """表示知识库元数据不存在。"""


class KnowledgeBaseConflictError(ValueError):
    """表示知识库ID已经存在，不能静默覆盖其元数据。"""


class KnowledgeBaseNotEmptyError(RuntimeError):
    """表示非force请求不能删除仍有索引或已登记源文件的知识库。"""


class KnowledgeBaseProtectedError(RuntimeError):
    """表示默认知识库是兼容入口的必需归属，不能移除。"""


class KnowledgeBaseIngestionBusyError(RuntimeError):
    """表示知识库存在可恢复入库任务，强制删除不代表取消。"""


class KnowledgeBaseDeletionInProgressError(RuntimeError):
    """表示已有force删除计划，重试不能省略原批量删除授权。"""


class KnowledgeBaseDeletePartialError(RuntimeError):
    """表示持久计划已经开始但跨存储清理未完成，需要按原请求重试。"""


class KnowledgeContentError(RuntimeError):
    """表示只读文档元数据或当前原文暂时不可用。"""


class ChunkNotFoundError(LookupError):
    """表示请求的文档与块组合不存在，不推断文档曾被删除。"""


class DocumentGoneError(LookupError):
    """表示索引当前已无文档且有可靠的既往删除记录。"""


class IngestionJobStorageError(RuntimeError):
    """表示任务登记不可用；读取失败不能被当作没有待发布文档。"""


class IngestionJobNotFoundError(LookupError):
    """表示持久化入库任务不存在。"""


class IngestionJobConflictError(RuntimeError):
    """表示任务身份、执行租约或状态转换不再匹配。"""


class IngestionQueueFullError(RuntimeError):
    """表示持久化入库队列已达有界容量。"""


class IngestionSourceError(RuntimeError):
    """表示受控源文件不能持久保存或恢复时完整性校验失败。"""


class IngestionWorkerUnavailableError(RuntimeError):
    """表示同一索引已有执行实例或后台执行器尚未就绪。"""


class QuestionRateLimitError(RuntimeError):
    """表示已认证用户的滑动窗口提问额度耗尽。"""

    def __init__(self, retry_after: int) -> None:
        """保存不小于一秒的重试等待值，不包含身份或问题正文。

        Args:
            retry_after: 向上取整的等待秒数。

        Returns:
            None。
        """
        super().__init__("提问过于频繁，请稍后重试")
        self.retry_after = max(1, retry_after)


class QuestionQueueFullError(RuntimeError):
    """表示问答排队容量已满或生命周期已停止接收新提问。"""

    def __init__(
        self, message: str = "问答排队容量已满", *, was_queued: bool = False,
    ) -> None:
        """区分响应头前拒绝和排队后停止，以保留正确SSE错误语义。

        Args:
            message: 服务器固定说明，不得包含用户正文。
            was_queued: 是否已经接受并发布了排队阶段。

        Returns:
            None。
        """
        super().__init__(message)
        self.was_queued = was_queued


class QuestionQueueTimeoutError(RuntimeError):
    """表示只消耗等待预算、未开始模型执行的请求已经超时。"""
