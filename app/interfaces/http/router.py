# -*- coding: utf-8 -*-
"""聚合全部稳定 HTTP 路由。"""

from fastapi import APIRouter
from fastapi import Depends
from app.interfaces.http.auth import require_admin
from app.interfaces.http.auth import require_user
from app.interfaces.http.routes.auth import router as auth_router
from app.interfaces.http.routes.feedback import router as feedback_router
from app.interfaces.http.routes.shares import router as shares_router
from app.interfaces.http.routes.evaluation_reports import router as evaluation_router
from app.interfaces.http.routes.traces import router as traces_router
from app.interfaces.http.routes.usage import router as usage_router
from app.interfaces.http.routes.feedback_admin import router as feedback_admin_router
from app.interfaces.http.routes.knowledge_bases import admin_router as knowledge_admin_router
from app.interfaces.http.routes.knowledge_bases import read_router as knowledge_read_router
from app.interfaces.http.routes.knowledge_browse import router as knowledge_browse_router
from app.interfaces.http.routes.suggestions import router as suggestions_admin_router
from app.interfaces.http.routes.ingestion_jobs import router as ingestion_jobs_router
from app.interfaces.http.routes.insights import router as insights_router

from app.interfaces.http.routes.documents import router as documents_router
from app.interfaces.http.routes.document_chunks import router as document_chunks_router
from app.interfaces.http.routes.conversations import (
    router as conversations_router,
)
from app.interfaces.http.routes.health import router as health_router
from app.interfaces.http.routes.queries import router as queries_router
from app.interfaces.http.routes.uploads import router as uploads_router


router = APIRouter(prefix="/api")
router.include_router(health_router)
router.include_router(auth_router)
router.include_router(queries_router, dependencies=[Depends(require_admin)])
router.include_router(uploads_router, dependencies=[Depends(require_admin)])
router.include_router(documents_router, dependencies=[Depends(require_admin)])
router.include_router(document_chunks_router, dependencies=[Depends(require_admin)])
router.include_router(conversations_router, dependencies=[Depends(require_user)])
router.include_router(feedback_router, dependencies=[Depends(require_user)])
router.include_router(shares_router, dependencies=[Depends(require_user)])
router.include_router(evaluation_router, dependencies=[Depends(require_admin)])
router.include_router(traces_router, dependencies=[Depends(require_admin)])
router.include_router(usage_router, dependencies=[Depends(require_admin)])
router.include_router(feedback_admin_router, dependencies=[Depends(require_admin)])
router.include_router(knowledge_read_router, dependencies=[Depends(require_user)])
router.include_router(knowledge_browse_router, dependencies=[Depends(require_user)])
router.include_router(knowledge_admin_router, dependencies=[Depends(require_admin)])
router.include_router(suggestions_admin_router, dependencies=[Depends(require_admin)])
router.include_router(ingestion_jobs_router, dependencies=[Depends(require_admin)])
router.include_router(insights_router, dependencies=[Depends(require_admin)])
