from dataclasses import replace
import pytest

from flux_image_server.config import Config
from flux_image_server.store import Store

OWNER = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"


@pytest.fixture
def config(tmp_path):
    return Config(data_dir=tmp_path, api_token="a" * 40, metrics_token="m" * 40, min_disk_bytes=0)


@pytest.fixture
def store(config):
    return Store(config)
