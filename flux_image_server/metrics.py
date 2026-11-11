import csv
import io
import shutil
import subprocess
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class Metrics:
    def __init__(self, store):
        self.store = store
        self.registry = CollectorRegistry()
        def gauge(name, help, labels=()):
            return Gauge("flux2_" + name, help, labels, registry=self.registry)
        def counter(name, help, labels=()):
            return Counter("flux2_" + name, help, labels, registry=self.registry)
        def histogram(name, help):
            return Histogram("flux2_" + name, help, buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1200), registry=self.registry)
        self.ready = gauge("model_ready", "BF16 model loaded and placement verified")
        self.jobs = gauge("jobs", "Current jobs", ("state",))
        self.starts = counter("generation_starts_total", "Generation starts in this process")
        self.completed = counter("jobs_completed_total", "Completed images")
        self.failed = counter("jobs_failed_total", "Failed jobs", ("reason",))
        self.rejected = counter("requests_rejected_total", "Rejected submissions", ("reason",))
        self.generation = histogram("generation_seconds", "Generation wall time including encoding and decode")
        self.duration = histogram("job_duration_seconds", "Creation to terminal wall time")
        self.stage = Histogram("flux2_stage_seconds", "Model load, warmup, encoding, denoising, or decode wall time",
                               ("stage",), buckets=(0.1, 1, 5, 10, 30, 60, 120, 300, 600), registry=self.registry)
        self.disk = gauge("disk_available_bytes", "Available output volume bytes")
        self.gpu_up = Gauge("gpu_exporter_up", "GPU telemetry collection success", registry=self.registry)
        self.gpu = {name: Gauge("gpu_" + name, "NVIDIA GPU " + name, ("gpu",), registry=self.registry) for name in (
            "utilization_percent", "memory_used_bytes", "memory_total_bytes", "temperature_celsius", "power_watts")}
        store.on_terminal = self.terminal

    def terminal(self, status, reason, duration):
        self.duration.observe(duration)
        if status == "completed":
            self.completed.inc()
        elif status == "failed":
            self.failed.labels(reason).inc()

    def render(self):
        counts, _oldest = self.store.snapshot()
        for status in ("generating", "completed", "failed", "cancelled", "expired"):
            self.jobs.labels("running" if status == "generating" else status).set(counts.get(status, 0))
        self.disk.set(shutil.disk_usage(self.store.config.data_dir).free)
        for metric in self.gpu.values():
            metric.clear()
        try:
            fields = "index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
            output = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
                                    check=True, capture_output=True, text=True, timeout=5).stdout
            for row in csv.reader(io.StringIO(output)):
                index = str(int(row[0]))
                for (name, metric), value in zip(self.gpu.items(), row[1:]):
                    number = float(value.strip())
                    metric.labels(index).set(number * 1024**2 if "memory" in name else number)
            self.gpu_up.set(1)
        except (OSError, ValueError, subprocess.SubprocessError):
            self.gpu_up.set(0)
        return generate_latest(self.registry)
