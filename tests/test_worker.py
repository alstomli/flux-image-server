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

    def generate(self, prompt, destination, seed, progress, *, reference_image=None):
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
    output = worker.metrics.render()
    assert b'flux2_oom_total 1.0' in output
    assert b'secret prompt' not in output


def test_oom_counter_starts_at_zero_and_counts_load_or_warmup_failure(store):
    class OutOfMemoryError(Exception):
        pass
    class FailedLoad(FakeRunner):
        phase = "warmup"
        def load(self):
            try:
                raise OutOfMemoryError("private content")
            except OutOfMemoryError as error:
                raise RuntimeError("wrapped failure") from error
    metrics = Metrics(store)
    assert b'flux2_oom_total 0.0' in metrics.render()
    worker = Worker(store, FailedLoad(), metrics)
    worker.run()
    assert not worker.ready and worker.loading_error
    output = metrics.render()
    assert b'flux2_oom_total 1.0' in output
    assert b'private content' not in output
