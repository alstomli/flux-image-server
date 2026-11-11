from types import SimpleNamespace
import pytest

from flux_image_server.metrics import Metrics


@pytest.mark.parametrize("bad", ["", "0,10,20,80,30,100\n", "0,10,20,80,30,100\n1,NaN,20,80,30,100\n",
                                  "0,10,20,80,30,100\n0,10,20,80,30,100\n"])
def test_incomplete_or_invalid_telemetry_never_looks_healthy(store, monkeypatch, bad):
    metric = Metrics(store)
    monkeypatch.setattr("flux_image_server.metrics.subprocess.run", lambda *_a, **_k:
                        SimpleNamespace(stdout="0,10,20,80,30,100\n1,10,20,80,30,100\n"))
    assert b'gpu_exporter_up 1.0' in metric.render()
    monkeypatch.setattr("flux_image_server.metrics.subprocess.run", lambda *_a, **_k: SimpleNamespace(stdout=bad))
    output = metric.render()
    assert b'gpu_exporter_up 0.0' in output
    assert b'gpu_memory_used_bytes{gpu=' not in output
