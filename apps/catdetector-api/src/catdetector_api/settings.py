from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogFormat(StrEnum):
    JSON = "json"
    TEXT = "text"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_web_dist_dir() -> Path:
    return repo_root() / "apps" / "catdetector-web" / "dist"


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CATDETECTOR_",
        env_file=".env",
        extra="ignore",
    )

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    web_dist_dir: Path = Field(
        default_factory=default_web_dist_dir,
        validation_alias=AliasChoices(
            "CATDETECTOR_WEB_DIST", "CATDETECTOR_WEB_DIST_DIR"
        ),
    )
    checkpoints_dir: Path = Path("checkpoints")
    checkpoint: Path | None = None
    vickie_threshold: float = 0.5
    oka_threshold: float = 0.5
    device: str | None = None
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.JSON
    log_access: bool = True

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {
            "CRITICAL",
            "ERROR",
            "WARNING",
            "WARN",
            "INFO",
            "DEBUG",
            "NOTSET",
        }:
            raise ValueError("Unsupported log level.")
        return normalized

    @field_validator("log_format", mode="before")
    @classmethod
    def normalize_log_format(cls, value: str | LogFormat) -> str | LogFormat:
        if isinstance(value, str):
            return value.lower()
        return value

    @model_validator(mode="after")
    def validate_thresholds(self) -> Self:
        for name, value in {
            "vickie_threshold": self.vickie_threshold,
            "oka_threshold": self.oka_threshold,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        return self
