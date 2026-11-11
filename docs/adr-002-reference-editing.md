# ADR 002: One bounded reference through the existing BF16 pipeline

Status: implemented; live GPU memory validation remains a rollout gate.

Extend JSON image jobs with one optional `referenceImage` PNG/JPEG data URL. Cap
decoded file bytes at 3 MiB, dimensions at exactly 1024-square, frames at one, and
total JSON at 4 MiB + 16 KiB, below the frontend payload ceiling. Fully verify and
decode before admission; normalize to RGB and strip metadata. There is no source
image ID lookup or reference download endpoint.

Include SHA256 of decoded encoded-file bytes with the prompt in owner/key
idempotency. References stay in process memory and are removed on every terminal
path, including the claimed worker copy and decoded PIL image. No queue, database,
quantization, or CPU/disk model offloading is added.

The pinned Flux2Pipeline routes reference pixels through the existing GPU 0 VAE
and transformer. GPU 1 encodes text only, with the same 512-token rejection. Keep
caption upsampling disabled to avoid a different multimodal prompt-encoder path.

Both text and reference-edit warmups gate readiness and editing capability. Reset
CUDA peaks per generation and capture actual allocated/reserved/peak memory in
finally, including failures/cancellation. Keep startup baselines and owner-scoped
terminal job diagnostics so rollout can verify real memory headroom. Roll back if
the existing two 80-GiB devices cannot keep the complete model in BF16 on CUDA.
