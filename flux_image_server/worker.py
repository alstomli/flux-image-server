import logging
import shutil
import threading
import time

from .errors import Cancelled, GenerationTimeout, PromptTooLong

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
        self.thread = threading.Thread(target=self.run, name="flux-worker", daemon=True)
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
            self.ready = True
            self.metrics.ready.set(1)
        except Exception as error:
            self.loading_error = True
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
        try:
            progress(0)
            self.runner.generate(job["prompt"], self.store.image_path(job["id"]), job["seed"], progress)
        except Cancelled:
            status = "cancelled"
        except PromptTooLong:
            status, error = "failed", "prompt_too_long"
        except GenerationTimeout:
            status, error = "failed", "timeout"
        except Exception as failure:
            status, error = "failed", "oom" if type(failure).__name__ == "OutOfMemoryError" else "generation_error"
            # Never log a pipeline exception: it may embed the prompt or a credential.
            log.error("Image generation failed; job content omitted.")
            try:
                self.runner.recover_error()
            except Exception:
                self.ready = False
                self.metrics.ready.set(0)
        self.store.finish(job["id"], status, error)
        elapsed = time.monotonic() - started
        self.metrics.generation.observe(elapsed)
