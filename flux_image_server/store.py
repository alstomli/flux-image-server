"""Process-local immediate admission. There is no waiting queue or database."""
from datetime import datetime, timezone
from hashlib import sha256
import threading
import time
import uuid

from .config import Config
from .errors import ServiceError


class Store:
    def __init__(self, config: Config, clock=time.time):
        self.config, self.clock = config, clock
        self.on_terminal = None
        self.lock = threading.RLock()
        self.jobs = {}
        self.keys = {}
        self.daily = {}
        self.outputs = config.data_dir / "outputs"
        self.outputs.mkdir(parents=True, exist_ok=True)

    def day(self):
        return datetime.fromtimestamp(self.clock(), timezone.utc).strftime("%Y-%m-%d")

    def quota(self, owner):
        return self.daily.get((owner, self.day()), 0)

    def submit(self, owner: str, key: str, prompt: str, ready: bool, *, reference=None):
        now = self.clock()
        # Length-delimited prompt plus decoded-image hash makes retries depend on
        # the actual reference bytes, not data-URL spelling or a source image ID.
        prompt_bytes = prompt.encode("utf-8")
        fingerprint = sha256(len(prompt_bytes).to_bytes(8, "big") + prompt_bytes +
                             (b"\1" + bytes.fromhex(reference.fingerprint) if reference else b"\0")).hexdigest()
        with self.lock:
            existing_id = self.keys.get((owner, key))
            if existing_id:
                existing = self.jobs[existing_id]
                if existing["fingerprint"] != fingerprint:
                    raise ServiceError("idempotency_conflict", "This request key was already used for another prompt or reference image.", 409)
                self.check_expired(existing)
                return dict(existing)
            if not ready:
                raise ServiceError("unavailable", "The image model is loading or unavailable.", 503, 10)
            if self.quota(owner) >= self.config.daily_starts:
                raise ServiceError("daily_quota", "Your daily image limit has been reached. Try again tomorrow (UTC).", 429)
            active = next((job for job in self.jobs.values() if job["status"] == "generating"), None)
            if active:
                if active["owner"] == owner:
                    raise ServiceError("user_capacity", "Your account already has an image in progress.", 429, 5)
                raise ServiceError("busy", "The image model is busy. Try again when the current image finishes.", 429, 5)
            identifier = str(uuid.uuid4())
            job = {"id": identifier, "owner": owner, "idem": key, "fingerprint": fingerprint,
                   "prompt": prompt, "status": "generating", "created": now, "started": None,
                   "finished": None, "expires": None, "cancel": False, "step": 0,
                   "error_code": None, "seed": uuid.uuid4().int % (2**63 - 1)}
            job.update(reference=reference, kind="edit" if reference else "text")
            self.jobs[identifier] = job
            self.keys[(owner, key)] = identifier
            self.daily[(owner, self.day())] = self.quota(owner) + 1
            return dict(job)

    def check_expired(self, job):
        if job["status"] == "expired" or (job["expires"] is not None and job["expires"] <= self.clock()):
            raise ServiceError("expired", "This image job has expired. Generate a new image.", 410)

    def owned(self, owner: str, identifier: str):
        with self.lock:
            job = self.jobs.get(identifier)
            if not job or job["owner"] != owner:
                raise ServiceError("not_found", "Image job not found. The service may have restarted.", 404)
            self.check_expired(job)
            return dict(job)

    def cancel(self, owner, identifier):
        with self.lock:
            self.owned(owner, identifier)
            job = self.jobs[identifier]
            if job["status"] == "generating":
                job["cancel"] = True
            return dict(job)

    def claim(self):
        with self.lock:
            for job in self.jobs.values():
                if job["status"] == "generating" and job["started"] is None:
                    job["started"] = self.clock()
                    return dict(job)
            return None

    def progress(self, identifier, step):
        with self.lock:
            job = self.jobs.get(identifier)
            if not job:
                return True
            job["step"] = step
            return bool(job["cancel"])

    def finish(self, identifier, status, error=None, *, generation_memory=None):
        with self.lock:
            job = self.jobs[identifier]
            if job["status"] != "generating":
                return job["status"]
            # A validated atomically persisted PNG wins over a late cancellation.
            now = self.clock()
            job.update(status=status, finished=now, expires=now + self.config.ttl_seconds,
                       error_code=error, prompt=None, reference=None)
            if generation_memory is not None:
                job["generation_memory"] = generation_memory
            if self.on_terminal:
                self.on_terminal(status, error, max(0, now - job["created"]))
        if status != "completed":
            self.image_path(identifier).unlink(missing_ok=True)
        return status

    def image_path(self, identifier):
        return self.outputs / (str(uuid.UUID(identifier)) + ".png")

    def recover(self):
        # Records and quotas intentionally do not survive process restart. Remove
        # interrupted temporary writes; orphan completed files expire by mtime.
        for path in self.outputs.glob("*.tmp"):
            path.unlink(missing_ok=True)

    def cleanup(self):
        now = self.clock()
        with self.lock:
            for job in self.jobs.values():
                if job["expires"] is not None and job["expires"] <= now:
                    self.image_path(job["id"]).unlink(missing_ok=True)
                    job["status"] = "expired"
            for identifier, job in list(self.jobs.items()):
                if job["expires"] is not None and job["expires"] + self.config.ttl_seconds <= now:
                    self.keys.pop((job["owner"], job["idem"]), None)
                    del self.jobs[identifier]
            self.daily = {key: value for key, value in self.daily.items() if key[1] == self.day()}
            for path in self.outputs.glob("*.png"):
                try:
                    if path.stat().st_mtime + self.config.ttl_seconds <= now:
                        path.unlink(missing_ok=True)
                except FileNotFoundError:
                    pass

    def snapshot(self):
        with self.lock:
            counts = {}
            for job in self.jobs.values():
                counts[job["status"]] = counts.get(job["status"], 0) + 1
            return counts, 0


def envelope(job, config: Config):
    def date(value):
        return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")
    result = {"jobId": job["id"], "status": job["status"], "kind": job["kind"], "createdAt": date(job["created"]),
              "progress": {"step": job["step"], "totalSteps": config.steps}}
    if job["expires"] is not None:
        result["expiresAt"] = date(job["expires"])
    if job.get("generation_memory") is not None:
        result["generationMemory"] = job["generation_memory"]
    if job["cancel"] and job["status"] == "generating":
        result["cancellationRequested"] = True
    if job["status"] == "completed":
        result["image"] = {"url": f"/v1/image-jobs/{job['id']}/image", "width": config.width, "height": config.height}
    if job["error_code"]:
        messages = {"interrupted": "The service stopped while generating. Generate again.",
                    "prompt_too_long": "This prompt exceeds the model's 512-token limit. Shorten it and try again.",
                    "timeout": "Image generation timed out. Try again."}
        result["error"] = {"code": job["error_code"], "message": messages.get(job["error_code"], "Image generation failed. Try again.")}
    return result
