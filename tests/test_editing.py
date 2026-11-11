import pytest

from flux_image_server.errors import ServiceError, Cancelled
from flux_image_server.metrics import Metrics
from flux_image_server.reference import validate_reference
from flux_image_server.worker import Worker
from flux_image_server.store import envelope
from test_reference import image_url
from test_api import client_for, headers
from test_worker import FakeRunner
from conftest import OWNER


def test_edit_idempotency_includes_decoded_reference_content(store):
    red = validate_reference(image_url(color="red"))
    blue = validate_reference(image_url(color="blue"))
    original = store.submit(OWNER, "same", "Change color", True, reference=red)
    assert store.submit(OWNER, "same", "Change color", True, reference=validate_reference(image_url(color="red")))["id"] == original["id"]
    for changed in (blue, None):
        with pytest.raises(ServiceError) as result:
            store.submit(OWNER, "same", "Change color", True, reference=changed)
        assert result.value.code == "idempotency_conflict"


@pytest.mark.parametrize("outcome", ["completed", "cancelled", "failed"])
def test_reference_memory_is_cleared_for_all_terminal_paths(store, outcome):
    reference = validate_reference(image_url())
    original = store.submit(OWNER, "one", "Change color", True, reference=reference)
    job = store.claim()
    seen = []
    class EditRunner(FakeRunner):
        def generate(self, prompt, destination, seed, progress, *, reference_image=None):
            assert reference_image.mode == "RGB" and reference_image.size == (1024, 1024)
            seen.append(reference_image)
            if outcome == "cancelled":
                raise Cancelled()
            if outcome == "failed":
                raise RuntimeError("private reference contents")
            return super().generate(prompt, destination, seed, progress, reference_image=reference_image)
    Worker(store, EditRunner(), Metrics(store)).execute(job)
    terminal = store.owned(OWNER, original["id"])
    assert terminal["status"] == outcome and terminal["reference"] is None and job["reference"] is None
    assert "reference" not in envelope(terminal, store.config)
    with pytest.raises(ValueError):
        seen[0].getpixel((0, 0))


def test_edit_api_capabilities_validation_and_private_reference(config):
    with client_for(config) as (client, app):
        capability = client.get("/capabilities", headers=headers(config)).json()
        assert capability["referenceEditing"] and capability["imageEditing"]["enabled"]
        assert capability["imageEditing"]["version"] == 1
        payload = {"prompt": "Change color", "referenceImage": image_url()}
        response = client.post("/v1/image-jobs", headers=headers(config), json=payload)
        assert response.status_code == 202 and response.json()["kind"] == "edit"
        assert "referenceImage" not in response.text
        identifier = response.json()["jobId"]
        changed = {**payload, "referenceImage": image_url(color="blue")}
        assert client.post("/v1/image-jobs", headers=headers(config), json=changed).status_code == 409
        app.state.store.finish(identifier, "cancelled")
        bad = {"prompt": "Change color", "referenceImage": "https://example.com/reference.png"}
        assert client.post("/v1/image-jobs", headers=headers(config, key="two"), json=bad).status_code == 400
        assert app.state.store.quota(OWNER) == 1
        app.state.worker.ready = False
        capability = client.get("/capabilities", headers=headers(config)).json()
        assert capability["referenceEditing"] is False and capability["imageEditing"]["enabled"] is False


def test_readiness_uses_one_atomic_memory_snapshot(config):
    with client_for(config) as (client, app):
        app.state.worker.runner.memory = {"stale": True}
        app.state.worker.runner.memory_context = {"kind": "wrong"}
        calls = []
        def snapshot():
            calls.append(True)
            return {"kind": "edit", "outcome": "completed", "gpus": {"0": {"peakAllocatedBytes": 123}}}
        app.state.worker.runner.memory_snapshot = snapshot
        result = client.get("/readyz", headers=headers(config)).json()
        assert calls == [True]
        assert result["memory"] == {"0": {"peakAllocatedBytes": 123}}
        assert result["memoryContext"] == {"kind": "edit", "outcome": "completed"}
