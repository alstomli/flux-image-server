import sys
from types import SimpleNamespace
import pytest

from flux_image_server.runner import FluxRunner
from flux_image_server.errors import Cancelled


@pytest.mark.parametrize("failure", [None, Cancelled(), RuntimeError("private")])
def test_each_generation_resets_peaks_and_publishes_memory_on_all_outcomes(config, monkeypatch, failure):
    resets = []
    cuda = SimpleNamespace(synchronize=lambda _index: None, reset_peak_memory_stats=resets.append,
        memory_allocated=lambda index: 100 + index, memory_reserved=lambda index: 200 + index,
        max_memory_allocated=lambda index: 300 + index, max_memory_reserved=lambda index: 400 + index,
        get_device_properties=lambda _index: SimpleNamespace(total_memory=800))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    runner = FluxRunner(config)
    def generate(*_args, **_kwargs):
        if failure:
            raise failure
    monkeypatch.setattr(runner, "_generate", generate)
    if failure:
        with pytest.raises(type(failure)):
            runner.generate("private prompt", config.data_dir / "test.png", 1, lambda _: None, reference_image=object())
    else:
        runner.generate("private prompt", config.data_dir / "test.png", 1, lambda _: None, reference_image=object())
    assert resets == [0, 1]
    snapshot = runner.memory_snapshot()
    assert snapshot["kind"] == "edit" and snapshot["capturedAt"].endswith("Z")
    assert snapshot["outcome"] == ("completed" if failure is None else "cancelled" if isinstance(failure, Cancelled) else "failed")
    assert snapshot["gpus"]["1"]["peakAllocatedBytes"] == 301
    assert "private" not in str(snapshot)
