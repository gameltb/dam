from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings.
    Values are loaded from environment variables and/or a .env file.
    """

    DATABASE_URL: str = Field("sqlite:///./dam.db", validation_alias="DAM_DATABASE_URL")
    # Example: DAM_DATABASE_URL="postgresql://user:pass@host:port/dbname"

    # For file storage simulation (can be expanded later)
    ASSET_STORAGE_PATH: str = Field("./dam_storage", validation_alias="DAM_ASSET_STORAGE_PATH")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields from .env
    )


# Create a single settings instance to be used throughout the application
settings = Settings()

# You can create a .env file in the project root with lines like:
# DAM_DATABASE_URL="sqlite:///./local_dam_database.db"
# DAM_ASSET_STORAGE_PATH="./my_dam_files"
