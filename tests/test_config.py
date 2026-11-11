import pytest

from flux_image_server.config import Config


def test_reused_api_metrics_environment_credentials_fail_closed(monkeypatch):
    for name in ("FLUX_API_TOKEN_FILE", "FLUX_METRICS_TOKEN_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FLUX_API_TOKEN", "same-secret" * 4)
    monkeypatch.setenv("FLUX_METRICS_TOKEN", "same-secret" * 4)
    with pytest.raises(ValueError, match="must be distinct"):
        Config.from_env()


def test_equal_file_secret_and_environment_secret_are_rejected(monkeypatch, tmp_path):
    path = tmp_path / "api-token"
    path.write_text("same-secret" * 4)
    monkeypatch.setenv("FLUX_API_TOKEN_FILE", str(path))
    monkeypatch.delenv("FLUX_METRICS_TOKEN_FILE", raising=False)
    monkeypatch.setenv("FLUX_METRICS_TOKEN", "same-secret" * 4)
    with pytest.raises(ValueError, match="must be distinct"):
        Config.from_env()


def test_distinct_secret_files_can_start_configuration(monkeypatch, tmp_path):
    for name, value in (("FLUX_API_TOKEN", "api-credential" * 4), ("FLUX_METRICS_TOKEN", "metrics-credential" * 4)):
        path = tmp_path / name
        path.write_text(value)
        monkeypatch.setenv(name + "_FILE", str(path))
    config = Config.from_env()
    assert config.api_token != config.metrics_token
