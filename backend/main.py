import os
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.config import settings
from backend.memory.qdrant_client import QdrantMemoryService, get_qdrant_service
from backend.services.llm import LLMService, get_llm_service
from backend.memory.router import router as memory_router
from backend.agents.router import router as agent_router
from backend.omi.router import router as omi_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Voice-first autonomous multi-agent system powered by Omi, Qdrant, and Lyzr.",
)

# Enable CORS for local development and cross-origin fetch
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memory vector store
app.include_router(memory_router, prefix="/api/memory", tags=["Memory"])
# Agent orchestration mesh
app.include_router(agent_router, prefix="/api/agent", tags=["Agent Mesh"])
# Omi voice listener
app.include_router(omi_router, prefix="/api/omi", tags=["Omi Voice"])


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint to verify backend service status."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }


@app.get("/health/qdrant", tags=["Health"])
async def qdrant_health_check(
    service: QdrantMemoryService = Depends(get_qdrant_service),
):
    """Health check endpoint to verify Qdrant vector database connection status."""
    return service.check_connection()


@app.get("/health/lyzr", tags=["Health"])
async def lyzr_health_check(
    llm: LLMService = Depends(get_llm_service),
):
    """Health check endpoint to verify Lyzr Cloud Agent API configuration status."""
    return llm.check_lyzr_status()


@app.get("/health/lyzr/test", tags=["Health"])
async def lyzr_connection_test(
    llm: LLMService = Depends(get_llm_service),
):
    """Perform a live standalone connection test against the Lyzr Cloud API."""
    return llm.test_lyzr_connection()


# Mount frontend static assets if directory exists
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
