from contextlib import contextmanager
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from flux_image_server.app import create_app
from conftest import OWNER, OTHER


@contextmanager
def client_for(config):
    app = create_app(config, start_worker=False)
    with TestClient(app) as client:
        app.state.worker.ready = True
        yield client, app


def headers(config, key="one", owner=OWNER):
    return {"Authorization": "Bearer " + config.api_token, "X-Chat-User-Id": owner, "Idempotency-Key": key}


def test_auth_identity_and_independent_metrics_credential(config):
    with client_for(config) as (client, app):
        assert client.get("/healthz").status_code == 200
        assert client.get("/capabilities").status_code == 401
        assert client.get("/metrics", headers=headers(config)).status_code == 401
        assert client.get("/capabilities", headers={"Authorization": "Bearer " + config.metrics_token}).status_code == 401
        result = client.post("/v1/image-jobs", headers=headers(config, owner="not-verified"), json={"prompt": "A boat"})
        assert result.status_code == 400
        assert app.state.store.snapshot()[0] == {}
        assert client.get("/metrics", headers={"Authorization": "Bearer " + config.metrics_token}).status_code == 200
        assert client.get("/docs", headers=headers(config)).status_code == 404


def test_owned_job_lifecycle_download_cancel_and_idempotent_submit(config):
    with client_for(config) as (client, app):
        result = client.post("/v1/image-jobs", headers=headers(config), json={"prompt": "A boat"})
        assert result.status_code == 202
        job = result.json()
        path = "/v1/image-jobs/" + job["jobId"]
        assert job["status"] == "generating" and job["progress"]["totalSteps"] == 28
        assert client.get(path, headers=headers(config, owner=OTHER)).status_code == 404
        assert client.get(path + "/image", headers=headers(config)).status_code == 409
        assert client.delete(path, headers=headers(config, owner=OTHER)).status_code == 404
        app.state.store.claim()
        requested = client.delete(path, headers=headers(config))
        assert requested.status_code == 202
        assert requested.json()["cancellationRequested"] is True
        destination = app.state.store.image_path(job["jobId"])
        Image.new("RGB", (1024, 1024), "red").save(destination)
        app.state.store.finish(job["jobId"], "completed")
        completed = client.get(path, headers=headers(config)).json()
        assert completed["image"]["url"] == path + "/image"
        result = client.get(path + "/image", headers=headers(config))
        assert result.content.startswith(b"\x89PNG\r\n\x1a\n")
        assert result.headers["cache-control"] == "no-store"
        assert client.get(path + "/image", headers=headers(config, owner=OTHER)).status_code == 404
        replay = client.post("/v1/image-jobs", headers=headers(config), json={"prompt": "A boat"})
        assert replay.json()["jobId"] == job["jobId"]
        assert replay.json()["status"] == "completed"


@pytest.mark.parametrize("payload", [{}, {"prompt": ""}, {"prompt": 1}, {"prompt": "x" * 2001},
                                      {"prompt": "A boat", "width": 512}, {"prompt": "😀" * 1001}])
def test_schema_rejections_never_reserve_a_job(config, payload):
    with client_for(config) as (client, app):
        assert client.post("/v1/image-jobs", headers=headers(config), json=payload).status_code == 400
        assert app.state.store.snapshot()[0] == {}


def test_oversized_body_and_readiness_fail_closed(config):
    with client_for(config) as (client, app):
        assert client.post("/v1/image-jobs", headers={**headers(config), "Content-Type": "application/json"},
                           content=b" " * 17000).status_code == 413
        app.state.worker.ready = False
        assert client.get("/readyz", headers=headers(config)).status_code == 503
        assert client.get("/capabilities", headers=headers(config)).json()["enabled"] is False
        assert client.post("/v1/image-jobs", headers=headers(config), json={"prompt": "A boat"}).status_code == 503


def test_one_process_per_data_directory(config):
    with client_for(config):
        with pytest.raises(RuntimeError, match="Another FLUX worker"):
            with client_for(config):
                pass
