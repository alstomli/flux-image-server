import logging
import shutil
import threading
import time

from .errors import Cancelled, GenerationTimeout, PromptTooLong, is_oom

log = logging.getLogger(__name__)


class Worker:
    def __init__(self, store, runner, metrics):
        self.store, self.runner, self.metrics = store, runner, metrics
        self.ready = False
        self.loading_error = False
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.thread = None
        self.error_class = None
        runner.metrics = metrics

    def start(self):
        # Python must not tear down native model-loading threads after a listener
        # failure or a graceful stop. A hard process kill still needs no recovery.
        self.thread = threading.Thread(target=self.run, name="flux-worker", daemon=False)
        self.thread.start()

    def shutdown(self):
        self.ready = False
        self.metrics.ready.set(0)
        self.stop.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=30)
            # Do not release the single-process lock while GPU work still runs.
            if self.thread.is_alive():
                return False
        return True

    def run(self):
        try:
            load_started = time.monotonic()
            self.runner.load()
            self.metrics.stage.labels("load").observe(time.monotonic() - load_started)
            if self.stop.is_set():
                return
            self.ready = True
            self.metrics.ready.set(1)
        except Exception as error:
            self.loading_error = True
            if is_oom(error):
                self.metrics.oom.inc()
            self.error_class = type(error).__name__
            log.error("Model loading failed: class=%s phase=%s; readiness remains false.",
                      self.error_class, getattr(self.runner, "phase", "load"))
            return
        cleanup_at = 0
        while not self.stop.is_set():
            try:
                if time.monotonic() >= cleanup_at:
                    self.store.cleanup()
                    cleanup_at = time.monotonic() + 30
                job = self.store.claim() if self.ready else None
                if job:
                    self.execute(job)
                else:
                    self.wake.wait(0.5)
                    self.wake.clear()
            except Exception:
                self.ready = False
                self.metrics.ready.set(0)
                log.error("Image worker failed; restarting the service is required.")
                return

    def accepting(self):
        return self.ready and not self.stop.is_set() and shutil.disk_usage(self.store.config.data_dir).free >= self.store.config.min_disk_bytes

    def execute(self, job):
        started = time.monotonic()
        config = self.store.config
        self.metrics.starts.inc()
        def progress(step):
            if self.store.progress(job["id"], step) or self.stop.is_set():
                raise Cancelled()
            if time.monotonic() - started >= config.generation_timeout:
                raise GenerationTimeout()
        status, error = "completed", None
        reference_image = None
        ran_generation = False
        try:
            progress(0)
            if job.get("reference"):
                reference_image = job["reference"].open_rgb()
            ran_generation = True
            self.runner.generate(job["prompt"], self.store.image_path(job["id"]), job["seed"], progress,
                                 reference_image=reference_image)
        except Cancelled:
            status = "cancelled"
        except PromptTooLong:
            status, error = "failed", "prompt_too_long"
        except GenerationTimeout:
            status, error = "failed", "timeout"
        except Exception as failure:
            status, error = "failed", "oom" if is_oom(failure) else "generation_error"
            if error == "oom":
                self.metrics.oom.inc()
            # Never log a pipeline exception: it may embed the prompt or a credential.
            log.error("Image generation failed; job content omitted.")
            try:
                self.runner.recover_error()
            except Exception:
                self.ready = False
                self.metrics.ready.set(0)
        finally:
            if reference_image is not None:
                reference_image.close()
            job["reference"] = None
        snapshot = self.runner.memory_snapshot() if ran_generation and hasattr(self.runner, "memory_snapshot") else None
        if snapshot is not None and snapshot.get("placementVerified") is False:
            self.ready = False
            self.metrics.ready.set(0)
        self.store.finish(job["id"], status, error, generation_memory=snapshot)
        elapsed = time.monotonic() - started
        self.metrics.generation.observe(elapsed)
