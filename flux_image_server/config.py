from dataclasses import dataclass
from pathlib import Path
import os

MODEL_ID = "black-forest-labs/FLUX.2-dev"
MODEL_REVISION = "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c"


def secret(name: str) -> str:
    path = os.environ.get(name + "_FILE")
    value = Path(path).read_text(encoding="utf-8").strip() if path else os.environ.get(name, "").strip()
    if len(value) < 32:
        raise ValueError(f"{name} must contain at least 32 characters")
    return value


@dataclass(frozen=True)
class Config:
    data_dir: Path
    api_token: str
    metrics_token: str
    model_dir: Path | None = None
    daily_starts: int = 20
    ttl_seconds: int = 86400
    generation_timeout: int = 600
    min_disk_bytes: int = 5 * 1024**3
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance: float = 4.0
    max_prompt_length: int = 2000

    def __post_init__(self):
        if len(self.api_token) < 32 or len(self.metrics_token) < 32:
            raise ValueError("API and metrics tokens must each contain at least 32 characters")
        if self.api_token == self.metrics_token:
            raise ValueError("API and metrics tokens must be distinct")

    @classmethod
    def from_env(cls):
        return cls(
            data_dir=Path(os.environ.get("FLUX_DATA_DIR", "/opt/flux-runtime")),
            api_token=secret("FLUX_API_TOKEN"), metrics_token=secret("FLUX_METRICS_TOKEN"),
            model_dir=Path(os.environ["FLUX_MODEL_DIR"]) if os.environ.get("FLUX_MODEL_DIR") else None,
        )
