from contextlib import asynccontextmanager
import hmac
import json
import re

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from .config import Config, MODEL_ID, MODEL_REVISION
from .errors import ServiceError
from .locking import ProcessLock
from .metrics import Metrics
from .runner import FluxRunner
from .reference import MAX_REFERENCE_BYTES, MAX_EDIT_REQUEST_BYTES, validate_reference
from .store import Store, envelope
from .worker import Worker

UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
KEY = re.compile(r"^[a-zA-Z0-9_.:-]{1,128}$")
MAX_REQUEST_BYTES = 16 * 1024


def verify_token(request, token):
    supplied = request.headers.get("authorization", "")
    if not hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
        raise ServiceError("unauthorized", "Authentication required.", 401)


def owner(request):
    values = request.headers.getlist("x-chat-user-id")
    if len(values) != 1 or not UUID.fullmatch(values[0]):
        raise ServiceError("user_identity", "A verified account identity is required.", 400)
    return values[0].lower()


def create_app(config: Config | None = None, *, runner=None, start_worker=True):
    config = config or Config.from_env()
    store = Store(config)
    metrics = Metrics(store)
    worker = Worker(store, runner if runner is not None else FluxRunner(config), metrics)
    process_lock = ProcessLock(config.data_dir / "worker.lock")

    @asynccontextmanager
    async def lifespan(_app):
        process_lock.acquire()
        try:
            store.recover()
            store.cleanup()
            if start_worker:
                worker.start()
            yield
        finally:
            if worker.shutdown():
                process_lock.release()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.worker, app.state.metrics = store, worker, metrics

    @app.exception_handler(ServiceError)
    async def service_error(_request, error):
        headers = {"Cache-Control": "no-store"}
        if error.retry_after is not None:
            headers["Retry-After"] = str(error.retry_after)
        return JSONResponse({"error": {"code": error.code, "message": error.message}}, error.status, headers=headers)

    @app.middleware("http")
    async def protect(request, call_next):
        try:
            if request.url.path != "/healthz":
                verify_token(request, config.metrics_token if request.url.path == "/metrics" else config.api_token)
            response = await call_next(request)
        except ServiceError as error:
            response = await service_error(request, error)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/healthz")
    def health():
        return {"status": "alive", "version": 1}

    @app.get("/readyz")
    def ready():
        available = worker.accepting()
        capture = getattr(worker.runner, "memory_snapshot", None)
        snapshot = capture() if callable(capture) else {
            **getattr(worker.runner, "memory_context", {}), "gpus": getattr(worker.runner, "memory", {})}
        return JSONResponse({"ready": available, "model": MODEL_ID, "revision": MODEL_REVISION,
                             "dtype": "bfloat16", "offloading": False,
                             "placement": getattr(worker.runner, "placement", {}),
                             "memory": snapshot.get("gpus", {}),
                             "memoryContext": {key: value for key, value in snapshot.items() if key != "gpus"},
                             "warmupMemory": getattr(worker.runner, "warmup_memory", {}),
                             "phase": getattr(worker.runner, "phase", "load"), "errorClass": worker.error_class,
                             "state": "failed" if worker.loading_error else "ready" if available else "loading_or_unavailable"},
                            200 if available else 503)

    @app.get("/capabilities")
    def capabilities():
        return {"version": 1, "model": "FLUX.2-dev", "enabled": worker.accepting(),
                "limits": {"width": config.width, "height": config.height, "steps": config.steps,
                           "guidanceScale": config.guidance, "maxPromptLength": config.max_prompt_length,
                           "maxUnfinishedPerAccount": 1, "dailyStarts": config.daily_starts,
                           "queueSize": 0, "resultTtlSeconds": config.ttl_seconds,
                           "maxReferenceBytes": MAX_REFERENCE_BYTES, "maxReferenceImages": 1},
                "persistence": "process", "dailyQuotaPersistence": "process",
                "referenceEditing": worker.ready,
                "imageEditing": {"enabled": worker.ready, "version": 1, "maxReferenceImages": 1,
                                 "mimeTypes": ["image/png", "image/jpeg"], "width": 1024, "height": 1024,
                                 "maxImageBytes": MAX_REFERENCE_BYTES, "maxRequestBytes": MAX_EDIT_REQUEST_BYTES},
                "states": ["generating", "completed", "failed", "cancelled", "expired"]}

    @app.post("/v1/image-jobs", status_code=202)
    async def submit(request: Request):
        principal = owner(request)
        key = request.headers.get("idempotency-key", "")
        if not KEY.fullmatch(key):
            raise ServiceError("invalid_request", "Provide an Idempotency-Key of 1–128 safe ASCII characters.")
        if request.headers.get("content-type", "").split(";")[0].lower().strip() != "application/json":
            raise ServiceError("invalid_request", "Send a JSON image request.")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_EDIT_REQUEST_BYTES:
                raise ServiceError("invalid_request", "The image request is too large.", 413)
            body.extend(chunk)
        try:
            data = json.loads(body.decode("utf-8"))
            if not isinstance(data, dict) or not {"prompt"} <= set(data) <= {"prompt", "referenceImage"} or not isinstance(data["prompt"], str):
                raise ValueError()
            prompt = data["prompt"].strip()
            length = len(prompt.encode("utf-16-le")) // 2
            if not prompt or length > config.max_prompt_length:
                raise ValueError()
        except (ValueError, UnicodeError):
            raise ServiceError("invalid_request", "Enter an image prompt of 1–2,000 characters.")
        if "referenceImage" not in data and len(body) > MAX_REQUEST_BYTES:
            raise ServiceError("invalid_request", "The image request is too large.", 413)
        reference = await run_in_threadpool(validate_reference, data["referenceImage"]) if "referenceImage" in data else None
        try:
            job = store.submit(principal, key, prompt, worker.accepting(), reference=reference)
        except ServiceError as error:
            metrics.rejected.labels(error.code).inc()
            raise
        worker.wake.set()
        return envelope(job, config)

    @app.get("/v1/image-jobs/{identifier}")
    def inspect(identifier: str, request: Request):
        return envelope(store.owned(owner(request), identifier), config)

    @app.delete("/v1/image-jobs/{identifier}", status_code=202)
    def cancel(identifier: str, request: Request):
        result = store.cancel(owner(request), identifier)
        worker.wake.set()
        return envelope(result, config)

    @app.get("/v1/image-jobs/{identifier}/image")
    def image(identifier: str, request: Request):
        job = store.owned(owner(request), identifier)
        if job["status"] != "completed":
            raise ServiceError("not_completed", "The image is not ready.", 409)
        try:
            data = store.image_path(job["id"]).read_bytes()
        except FileNotFoundError:
            raise ServiceError("unavailable_image", "This image is no longer available.", 410)
        return Response(data, media_type="image/png", headers={
            "Content-Disposition": 'attachment; filename="kiddo-generated-image.png"',
            "Cross-Origin-Resource-Policy": "same-origin"})

    @app.get("/metrics")
    def exposition():
        return Response(metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8")

    # Separate metrics listener exposes no job or model routes.
    metrics_app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @metrics_app.get("/metrics")
    def dedicated_metrics(request: Request):
        try:
            verify_token(request, config.metrics_token)
        except ServiceError:
            return Response(status_code=401, headers={"Cache-Control": "no-store"})
        return Response(metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8", headers={"Cache-Control": "no-store"})

    app.state.metrics_app = metrics_app
    return app
