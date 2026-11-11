# ADR 001: Dedicated BF16 worker with immediate admission

Status: implemented following the user's explicit removal of the waiting queue.

FLUX.2-dev uses a different runtime and large text encoder, so a separate service
avoids coupling Qwen/vLLM dependencies and lifecycle. Keep BF16 transformer/VAE on
GPU 0 and the BF16 text encoder on GPU 1, staging prompt embeddings explicitly.
Retain all weights on CUDA with no quantization or CPU/disk offloading. Real
1024×1024 warmup, parameter/device checks, and measured memory gate readiness.

There is no waiting queue, SQLite, or external database. One process-local
thread-safe reservation admits a single generation immediately; competing new
requests get 429. Owner/key idempotency, ownership, progress, and 20 daily starts
are in memory. Process restart resets quotas and loses records; old IDs return
404 and interrupted work is never silently reexecuted. This limitation is explicit
in capabilities and operator documentation. Keep the single-process lock and
temporary PNGs on local runtime storage, with persistent checkpoints on the
existing global volume. A new pod/storage migration is unnecessary.

Temporary server files/status expire after 24 hours while the process survives.
The frontend stores verified downloads in account-scoped IndexedDB and syncs only
metadata/placeholders. An atomically persisted image wins late cancellation. RGB
dimensions and a 4 MiB output cap keep the authenticated proxy within its payload
limit. A separate metrics token and bounded labels protect operational telemetry.
