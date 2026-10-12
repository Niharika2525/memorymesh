"""Configuration settings for MemoryMesh backend."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # Server configuration
    APP_NAME: str = "MemoryMesh"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"

    # Qdrant Vector Memory Configuration
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "memorymesh_memories"
    QDRANT_VECTOR_SIZE: int = 1536
    QDRANT_DISTANCE: str = "Cosine"

    # Omi Voice Integration Configuration (Placeholder for future phases)
    OMI_APP_ID: str = ""
    OMI_APP_SECRET: str = ""

    # Lyzr Multi-Agent Orchestration & LLM Configuration
    LYZR_API_KEY: str = ""
    LYZR_AGENT_ID: str = ""
    LYZR_API_URL: str = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
    OPENAI_API_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
