"""Application configuration for DoMarkX."""
import logging
import pathlib

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings for Domarkx.

    Values are loaded from environment variables and/or a .env file.
    """

    DOMARKX_PROJECT_PATH: str = Field(
        default=".",
        validation_alias="DOMARKX_PROJECT_PATH",
        description="The root directory of the domarkx project.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def project_path(self) -> str:
        """Return the resolved absolute path to the project."""
        return str(pathlib.Path(self.DOMARKX_PROJECT_PATH).resolve())


# Global settings instance, initialized when this module is imported.
# Applications should import this instance.
settings = Settings()
