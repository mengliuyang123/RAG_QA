# -*- coding: utf-8 -*-
"""只替代外部理解模型，保留生产理解服务及结构化边界。"""

import json

from app.application.ports.structured_output import ResponseModel
from app.application.query.routing import RuleQueryUnderstanding


class OfflineUnderstandingProvider:
    """通过确定性规则返回结构化数据，禁止测试访问真实模型。"""

    def generate(
        self,
        messages: list[dict[str, object]],
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        """模拟模型输出，不替换生产理解流程、路由或检索。

        Args:
            messages: 实际服务构造的消息。
            response_model: 实际服务要求的领域模型。

        Returns:
            独立查询仍为原问题的保守输出。
        """
        payload = json.loads(str(messages[-1]["content"]))
        output = RuleQueryUnderstanding().understand(payload["question"])
        return response_model.model_validate(output.model_dump())
