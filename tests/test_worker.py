from PIL import Image

from flux_image_server.metrics import Metrics
from flux_image_server.worker import Worker
from conftest import OWNER


class FakeRunner:
    """Explicit test injection; production has no environment-selected fake runner."""
    def __init__(self, action=None):
        self.action = action

    def load(self):
        pass

    def generate(self, prompt, destination, seed, progress):
        progress(0)
        if self.action:
            self.action()
        progress(1)
        Image.new("RGB", (1024, 1024), "red").save(destination)

    def recover_error(self):
        pass


def test_cancel_between_steps_removes_output_and_keeps_quota(store):
    job = store.submit(OWNER, "one", "A boat", True)
    claimed = store.claim()
    worker = Worker(store, FakeRunner(lambda: store.cancel(OWNER, job["id"])), Metrics(store))
    worker.execute(claimed)
    assert store.owned(OWNER, job["id"])["status"] == "cancelled"
    assert not store.image_path(job["id"]).exists()
    assert store.quota(OWNER) == 1


def test_success_writes_image_and_persists_terminal_metadata(store):
    job = store.submit(OWNER, "one", "A boat", True)
    claimed = store.claim()
    worker = Worker(store, FakeRunner(), Metrics(store))
    worker.execute(claimed)
    assert store.owned(OWNER, job["id"])["status"] == "completed"
    assert store.owned(OWNER, job["id"])["prompt"] is None
    assert store.image_path(job["id"]).exists()


def test_oom_is_classified_without_leaking_exception(store):
    class OutOfMemoryError(Exception):
        pass
    def fail():
        raise OutOfMemoryError("secret prompt")
    job = store.submit(OWNER, "one", "A boat", True)
    worker = Worker(store, FakeRunner(fail), Metrics(store))
    worker.execute(store.claim())
    assert store.owned(OWNER, job["id"])["error_code"] == "oom"
    assert b'secret prompt' not in worker.metrics.render()
