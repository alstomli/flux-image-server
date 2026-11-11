"""The only production runner. No quantization, offloading, or fake-image fallback."""
from pathlib import Path
import gc
import os
import json
import time
import threading
from datetime import datetime, timezone

from .config import MODEL_ID, MODEL_REVISION, Config
from .errors import PromptTooLong, Cancelled


class FluxRunner:
    def __init__(self, config: Config):
        self.config = config
        self.encoder = self.pipeline = None
        self.placement = {}
        self.phase = "not_started"
        self.memory = {}
        self.memory_context = {}
        self.warmup_memory = {}
        self.memory_lock = threading.Lock()
        self.metrics = None

    def load(self):
        import torch
        from diffusers import Flux2Pipeline
        self.phase = "cuda_preflight"
        if torch.cuda.device_count() != 2 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Exactly two BF16-capable CUDA GPUs are required")
        source = str(self.config.model_dir) if self.config.model_dir else MODEL_ID
        if self.config.model_dir:
            manifest = json.loads((self.config.model_dir / "flux-snapshot.json").read_text())
            if manifest != {"repo": MODEL_ID, "revision": MODEL_REVISION}:
                raise RuntimeError("Checkpoint manifest does not match the pinned model revision")
        kwargs = {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True,
                  "local_files_only": True}
        if not self.config.model_dir:
            kwargs["revision"] = MODEL_REVISION
        # Two pipeline wrappers hold disjoint modules, never duplicate model weights.
        # Pipeline.encode_prompt uses its execution device for token inputs; passing
        # a raw prompt to a manually split complete pipeline causes device mismatches.
        self.phase = "encoder_load"
        self.encoder = Flux2Pipeline.from_pretrained(source, transformer=None, vae=None, scheduler=None, **kwargs)
        self.encoder.text_encoder.to(device="cuda:1", dtype=torch.bfloat16).eval()
        self.phase = "denoiser_load"
        self.pipeline = Flux2Pipeline.from_pretrained(source, text_encoder=None, tokenizer=None, **kwargs)
        self.pipeline.transformer.to(device="cuda:0", dtype=torch.bfloat16).eval()
        self.pipeline.vae.to(device="cuda:0", dtype=torch.bfloat16).eval()
        self.verify_placement()
        self.phase = "warmup_text"
        warmup = self.config.data_dir / "evidence" / "warmup.png"
        warmup.parent.mkdir(exist_ok=True)
        warmup_start = time.monotonic()
        try:
            self.generate("A small red wooden toy boat on calm blue water.", warmup, 42, lambda _step: None)
            self.verify_placement()
            self.warmup_memory["text"] = self.memory
            self.phase = "warmup_edit"
            from PIL import Image
            with Image.open(warmup) as source_image:
                reference = source_image.convert("RGB")
            try:
                self.generate("Change the red toy boat to a bright yellow toy boat while keeping the scene layout.",
                              warmup.with_name("warmup-edit.png"), 43, lambda _step: None, reference_image=reference)
                self.verify_placement()
                self.warmup_memory["edit"] = self.memory
            finally:
                reference.close()
            if self.metrics:
                self.metrics.stage.labels("warmup").observe(time.monotonic() - warmup_start)
        finally:
            warmup.with_suffix(".tmp").unlink(missing_ok=True)
        self.phase = "ready"
        gc.collect()

    def verify_placement(self):
        import torch
        records = {}
        for name, model, expected in (
            ("text_encoder", self.encoder.text_encoder, "cuda:1"),
            ("transformer", self.pipeline.transformer, "cuda:0"),
            ("vae", self.pipeline.vae, "cuda:0"),
        ):
            if getattr(model, "is_quantized", False) or getattr(model, "quantization_method", None):
                raise RuntimeError("Quantized model weights are forbidden")
            if getattr(model.config, "quantization_config", None):
                raise RuntimeError("Quantization configuration is forbidden")
            for module in model.modules():
                hook = getattr(module, "_hf_hook", None)
                if hook is not None and getattr(hook, "offload", False):
                    raise RuntimeError("CPU/disk offload hooks are forbidden")
            for parameter in model.parameters():
                if str(parameter.device) != expected or parameter.dtype != torch.bfloat16:
                    raise RuntimeError(f"Invalid BF16 CUDA placement in {name}")
            for buffer in model.buffers():
                if str(buffer.device) != expected:
                    raise RuntimeError(f"Non-CUDA model buffer in {name}")
            device_map = getattr(model, "hf_device_map", {})
            if any(str(device) in ("cpu", "disk", "meta") for device in device_map.values()):
                raise RuntimeError("Non-CUDA device map is forbidden")
            records[name] = {"device": expected, "dtype": "bfloat16", "parameters": sum(p.numel() for p in model.parameters())}
        self.placement = records

    def generate(self, prompt: str, destination: Path, seed: int, progress, *, reference_image=None):
        import torch
        outcome = "failed"
        try:
            for index in range(2):
                torch.cuda.synchronize(index)
                torch.cuda.reset_peak_memory_stats(index)
            self._generate(prompt, destination, seed, progress, reference_image=reference_image)
            outcome = "completed"
        except Cancelled:
            outcome = "cancelled"
            raise
        finally:
            memory = {}
            for index in range(2):
                item = {}
                try:
                    torch.cuda.synchronize(index)
                except Exception as error:
                    item["synchronizeErrorClass"] = type(error).__name__
                try:
                    item.update(allocatedBytes=torch.cuda.memory_allocated(index),
                                reservedBytes=torch.cuda.memory_reserved(index),
                                peakAllocatedBytes=torch.cuda.max_memory_allocated(index),
                                peakReservedBytes=torch.cuda.max_memory_reserved(index),
                                totalBytes=torch.cuda.get_device_properties(index).total_memory)
                except Exception as error:
                    item["captureErrorClass"] = type(error).__name__
                memory[str(index)] = item
            # Publish a complete immutable snapshot after CUDA synchronization.
            context = {"kind": "edit" if reference_image is not None else "text", "outcome": outcome,
                       "capturedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
            with self.memory_lock:
                self.memory = memory
                self.memory_context = context

    def memory_snapshot(self):
        with self.memory_lock:
            return {**self.memory_context, "gpus": self.memory}

    def _generate(self, prompt: str, destination: Path, seed: int, progress, *, reference_image=None):
        import torch
        from diffusers.pipelines.flux2.pipeline_flux2 import format_input
        config = self.config
        # Validate the actual model chat template, including the system prompt,
        # instead of silently truncating a long prompt at the pipeline's boundary.
        inputs = self.encoder.tokenizer.apply_chat_template(
            format_input([prompt], system_message=self.encoder.system_message),
            add_generation_prompt=False, tokenize=True, return_dict=True,
            return_tensors="pt", truncation=False,
        )
        if inputs["input_ids"].shape[-1] > 512:
            raise PromptTooLong()
        with torch.inference_mode():
            progress(0)
            encode_start = time.monotonic()
            embeds, _ = self.encoder.encode_prompt(prompt, device=torch.device("cuda:1"), max_sequence_length=512)
            embeds = embeds.to(device="cuda:0", dtype=torch.bfloat16)
            torch.cuda.synchronize(1)
            if self.metrics:
                self.metrics.stage.labels("encode").observe(time.monotonic() - encode_start)
            progress(0)
            denoise_start = time.monotonic()
            denoise_end = None
            def step_end(_pipeline, step, _timestep, callback_kwargs):
                nonlocal denoise_end
                progress(step + 1)
                if step + 1 == config.steps:
                    torch.cuda.synchronize(0)
                    denoise_end = time.monotonic()
                return callback_kwargs
            image = self.pipeline(
                prompt_embeds=embeds, image=reference_image, height=config.height, width=config.width,
                num_inference_steps=config.steps, guidance_scale=config.guidance,
                num_images_per_prompt=1, max_sequence_length=512,
                generator=torch.Generator(device="cuda:0").manual_seed(seed),
                callback_on_step_end=step_end,
                caption_upsample_temperature=None,
            ).images[0]
            torch.cuda.synchronize(0)
            if self.metrics and denoise_end is not None:
                self.metrics.stage.labels("denoise").observe(denoise_end - denoise_start)
                self.metrics.stage.labels("decode").observe(time.monotonic() - denoise_end)
            progress(config.steps)
            image = image.convert("RGB")
            if image.size != (config.width, config.height):
                raise RuntimeError("Unexpected generated dimensions")
            temporary = destination.with_suffix(".tmp")
            try:
                image.save(temporary, format="PNG")
                if temporary.stat().st_size > 4 * 1024**2:
                    raise RuntimeError("Generated PNG exceeds the 4 MiB output limit")
                with temporary.open("rb") as stream:
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
                if os.name != "nt":
                    directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
            finally:
                temporary.unlink(missing_ok=True)
        self.verify_placement()

    def recover_error(self):
        import torch
        gc.collect()
        for device in range(2):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        self.verify_placement()
