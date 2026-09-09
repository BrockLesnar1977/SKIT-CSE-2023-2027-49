from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:4173",  
    ]
    RATE_LIMIT_PER_MINUTE: int = 20

    GCP_PROJECT_ID: str = ""
    GCS_BUCKET_NAME: str = ""

    GROQ_API_KEY: str = ""

    JOB_TTL_SECONDS: int = 3600  
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = get_settings()
