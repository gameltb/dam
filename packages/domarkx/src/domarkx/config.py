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
    def project_path(self) -> pathlib.Path:
        """Return the resolved absolute path to the project."""
        return pathlib.Path(self.DOMARKX_PROJECT_PATH).resolve()

    @property
    def domarkx_dir(self) -> pathlib.Path:
        """Return the path to the .domarkx directory."""
        return self.project_path / ".domarkx"

    @property
    def snapshots_dir(self) -> pathlib.Path:
        """Return the path to the snapshots directory."""
        return self.domarkx_dir / "snapshots"

    @property
    def git_repo_path(self) -> pathlib.Path:
        """Return the path to the git repository for session storage."""
        return self.project_path / ".domarkx" / "git_store"


# Global settings instance, initialized when this module is imported.
# Applications should import this instance.
settings = Settings()
