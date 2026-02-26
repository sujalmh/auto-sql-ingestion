from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    
    # PostgreSQL Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str
    postgres_user: str
    postgres_password: str
    
    # Milvus Configuration
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: Optional[str] = None
    milvus_password: Optional[str] = None
    milvus_collection: str = "sql_table_signatures"
    milvus_db: str = "default"  # Milvus database name
    
    # Similarity Search Configuration (threshold allows Dec/Jan-style period variants to match)
    similarity_threshold: float = 0.85
    similarity_top_k: int = 5
    # Minimum column-overlap percentage to accept a Milvus match as incremental load
    schema_match_min_percentage: float = 60.0
    # Column-based fallback: minimum IDF-weighted column overlap (0-1) when Milvus
    # returns no match.  Higher than Milvus threshold because it lacks semantic
    # context from embeddings.
    column_fallback_min_overlap: float = 0.7
    
    # OpenAI Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    
    # Application Settings
    log_level: str = "INFO"
    approval_timeout_minutes: int = 30
    
    # File Upload Settings
    max_upload_size_mb: int = 100
    allowed_extensions: list[str] = [".csv", ".xlsx"]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    @property
    def database_url(self) -> str:
        """Construct PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# Global settings instance
settings = Settings()
