from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Postgres
    db_url: str = Field(alias="SYNAPSE_DB_URL")
    db_pool_min: int = 2
    db_pool_max: int = 10

    # Maintenance loop cadence (summaries + embed). Episodes arrive via the
    # /ingest push hook, not a poll, so this only paces post-ingest work.
    poll_interval_seconds: int = Field(default=300, alias="POLL_INTERVAL_SECONDS")

    # Voyage AI API key (for embeddings)
    voyage_api_key: str = Field(default="", alias="VOYAGE_API_KEY")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
