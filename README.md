# FLUX image server

Private text-to-image generation for Kiddo Chat using unquantized BF16
`black-forest-labs/FLUX.2-dev`, revision
`26afe3a78bb242c0a8bb181dcc8937bb16e5c66c`. This public repository contains no
weights or credentials. Qwen chat and image understanding remain separate.

**There is no waiting queue or database.** One image can run globally. A new
request while another image runs receives 429; it is not stored for later.
Account ownership, idempotency, status, and 20-start UTC daily limits live in
process memory and reset on service restart. Accepted jobs are never silently
restarted. Old job IDs return 404 after restart. Browser-saved images remain
available independently of these temporary server records.

## GPU/runtime design

Use exactly two native-BF16 CUDA GPUs with at least 80 GiB each. The transformer
and VAE remain on GPU 0, and the Mistral text encoder remains on GPU 1. Encoding
transfers prompt embeddings between GPUs. There is no quantization, CPU/disk model
offloading, remote encoder, or fake production fallback. Host memory is used for
checkpoint loading, tokenization, and final PNG encoding normally.

Disjoint pipeline wrappers prevent device mismatches and duplicate model weights.
Startup verifies BF16 parameter dtype and exact CUDA placement, checks buffers and
offload hooks, and generates a real 1024×1024/28-step warmup image before readiness.
Authenticated `/readyz` reports model revision, placement, and measured peak GPU
memory. Warmup consumes no account quota and overwrites a single evidence image at
`FLUX_DATA_DIR/evidence/warmup.png`. Save that image with project deployment evidence
under `images-in-chat`; customer images must never be committed.

Use Linux x86-64/Python 3.12 and CUDA 12.8 PyTorch wheels. The container pins its
CUDA base digest, direct dependencies, and the resolved CPU dependency constraints.
Live GPU validation remains necessary. Model load failure leaves readiness false
with a sanitized phase/error class. Exception text, prompts, user identifiers, and
credentials are not logged.

## Storage and configuration

Persistent checkpoint storage should have at least 200 GB free; 250 GB is preferred.
The Diffusers transformer and text encoder alone are about 112.4 GB decimal.
Download only Diffusers components, excluding duplicate standalone checkpoints.
RunPod global/object storage can hold model files. Runtime data stays local at
`/opt/flux-runtime`: an exclusive process lock and temporary PNG outputs, with no
SQLite or filesystem locking dependency on the global volume. Container recreation
may remove those outputs, and restart always loses the in-memory ownership records.

| Variable | Meaning |
| --- | --- |
| `FLUX_API_TOKEN_FILE` / `FLUX_API_TOKEN` | API bearer secret, >=32 characters; file takes precedence |
| `FLUX_METRICS_TOKEN_FILE` / `FLUX_METRICS_TOKEN` | Separate metrics bearer secret, >=32 characters |
| `FLUX_DATA_DIR` | Local runtime/output directory, default `/opt/flux-runtime` |
| `FLUX_MODEL_DIR` | Pinned snapshot; Docker default `/workspace/models/flux2-dev` |
| `HF_HOME` | Download cache; Docker default `/workspace/huggingface` |
| `HF_TOKEN` | Authorized Hugging Face token for the separate fetch command |
| `PORT` / `METRICS_PORT` | API/metrics listeners, default 8001/8889 |

Use restrictive secret files or RunPod secret variables. Never commit tokens,
`.env`, images, databases, or weights. Never use a RunPod account key as the image
service token. The Hugging Face account must accept the
[model access conditions](https://huggingface.co/black-forest-labs/FLUX.2-dev).

## Prepare and start

```sh
python3 -m venv /opt/flux-venv
/opt/flux-venv/bin/python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
/opt/flux-venv/bin/python -m pip install -c constraints-cpu.txt '.[gpu]'
/opt/flux-venv/bin/python -m pip check
# Configure directories and secret files from the table first.
/opt/flux-venv/bin/python scripts/fetch_model.py
/opt/flux-venv/bin/python -m flux_image_server
```

The fetch command writes a revision manifest; startup rejects a mismatched local
manifest. Runtime model loading is local-files-only. CI and container builds need
no model access credentials. Never pass an HF token as a Docker build argument.

```sh
docker buildx build --platform linux/amd64 -t flux-image-server:local --load .
docker run --gpus all --rm --env-file /secure/flux.env \
  --mount type=bind,src=/persistent/workspace,dst=/workspace \
  -p 8001:8001 -p 8889:8889 flux-image-server:local
```

Run one service process and one GPU worker. An exclusive local runtime lock rejects
a second process. The HTTP API remains responsive during model loading. `/healthz`
reports process liveness; authenticated `/readyz` gates real traffic. Graceful
shutdown requests cooperative cancellation. A hard stop loses status and quota
records, and never retries the interrupted image automatically.

## API version 1

Every route except minimal `/healthz` requires `Authorization: Bearer <api-token>`.
Job routes also require one canonical UUID in `X-Chat-User-Id`. The trusted frontend
derives this from verified Supabase authentication and its active-session check,
never browser-supplied identity. The frontend enforces the admins-first rollout
gate; the backend enforces ownership and admission.

| Route | Behavior |
| --- | --- |
| `GET /capabilities` | Version, model, enabled/readiness, generation limits, process-local persistence notice |
| `GET /healthz` | Minimal public liveness |
| `GET /readyz` | 200 after real warmup/placement proof; otherwise 503 |
| `POST /v1/image-jobs` | 202 immediately admitted image; JSON `{"prompt":"..."}` and `Idempotency-Key` |
| `GET /v1/image-jobs/{id}` | Owned image-job envelope; another account gets 404 |
| `DELETE /v1/image-jobs/{id}` | 202 same envelope with cooperative cancellation requested |
| `GET /v1/image-jobs/{id}/image` | Owned completed RGB PNG, attachment, no-store |
| `GET /metrics` | Prometheus data with the separate metrics bearer token |

The dedicated 8889 listener exposes only protected `/metrics`. API credentials
cannot scrape metrics, and metrics credentials cannot access jobs. No CORS or
interactive API docs are enabled. Unknown API routes still require authentication.

Jobs contain `jobId`, `status`, `createdAt`, and `progress: {step,totalSteps}`.
Terminal jobs add `expiresAt`; completed jobs add `image: {url,width,height}`.
The URL is API-relative, never an arbitrary remote URL. Failed jobs add safe
`error: {code,message}`. Status is `generating`, `completed`, `failed`, or `cancelled`;
there is no queued state. Expired jobs return 410. Running cancellation returns
`cancellationRequested: true` until confirmed. A validated PNG atomically persisted
before a late cancellation wins and remains completed. Cancellation is checked
before/after encoding, between denoise steps, and before saving; an individual
long CUDA operation cannot be preempted immediately.

Prompts are nonempty, at most 2,000 UTF-16 characters, with at most 16 KiB JSON.
The actual model-formatted encoder input must also fit 512 tokens; generation
explicitly fails rather than silently truncating it. Only a prompt is accepted:
no reference image, custom dimensions, batch size, negative prompt, or model.
One image is 1024×1024 RGB PNG, 28 steps, guidance 4, at most 4 MiB. A random seed
is assigned once. Reusing owner/key/prompt returns the same job while this process
retains it; a different prompt with the same key returns 409. The frontend uses
a stable message UUID to recover an uncertain HTTP submission without duplicating
work during the same process lifetime.

One accepted unfinished image blocks new admissions. The owning account gets
429 `user_capacity`; other accounts get 429 `busy`, both with `Retry-After: 5`.
Each account gets 20 accepted starts per UTC day, process-local and reset on
restart. Rejected or idempotently replayed requests consume no start; accepted
work that fails, times out, or is cancelled consumes one. Generation has a
cooperative ten-minute deadline. Admissions pause below 5 GiB free local disk.

Terminal metadata/output TTL is 24 hours while the process survives. Cleanup
runs every 30 seconds, retaining expired idempotency tombstones another 24 hours.
Prompts are deleted on terminal transition. Orphan outputs from restart are
removed by file age, without restoring ownership. Kiddo Chat stores downloaded
images in account-scoped IndexedDB until conversation deletion/site-data clearing.
Cloud history holds metadata/placeholders only, never generated image files.

## Monitoring, tests, and deployment

FLUX-specific metrics use `flux2_*`: readiness, running/terminal job counts,
starts/completions/failures/rejections, generation/job duration, free disk, and stage
timings (`load`, `warmup`, `encode`, `denoise`, `decode`). There are no queue metrics.
Shared `gpu_*` and `gpu_exporter_up` names remain compatible with existing dashboards;
failed refresh clears stale GPU series. Labels are bounded reasons/states/stages
and GPU indices, without prompt/user/job/credential labels. All counters and
account quotas reset on process restart.

```sh
python -m pip install -c constraints-cpu.txt -e '.[test]'
python -m pytest
python -m pip check
```

CPU tests explicitly inject a fake runner and exercise API auth/ownership, immediate
busy admission, concurrent idempotency, UTC quota reset and restart loss, request
bounds, cancellation races, expiry, and OOM classification. Production never
selects a fake runner from environment configuration. Tests do not establish GPU
memory, latency, or image quality. Restricted Windows runners may need local
asyncio socket permission and a writable `--basetemp`.

Before activation check real readiness placement/memory proof, generate through
Kiddo Chat, download/reload the result, cancel a running image, confirm busy429,
verify another account cannot retrieve it, and confirm both GPUs and FLUX timings
in Prometheus/Grafana. Record checkpoint revision and image digest. Restore a
previously verified immutable digest to roll back, understanding that a restart
resets temporary jobs and quotas.

Candidate CI publishes commit-addressed GHCR images only from the exact trusted
feature branch. Manual versioned release CI runs only on main and refuses to
overwrite tags. Neither workflow merges a PR, loads weights, or deploys a pod.
Deploy by digest after live verification.

References: [Diffusers FLUX.2](https://huggingface.co/docs/diffusers/en/api/pipelines/flux2),
[pinned pipeline source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/pipelines/flux2/pipeline_flux2.py),
[architecture decision](docs/adr-001-image-jobs.md).
