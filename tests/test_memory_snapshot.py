import sys
import weakref
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
    verified = []
    monkeypatch.setattr(runner, "verify_placement", lambda: verified.append(True))
    if failure:
        with pytest.raises(type(failure)):
            runner.generate("private prompt", config.data_dir / "test.png", 1, lambda _: None, reference_image=object())
    else:
        runner.generate("private prompt", config.data_dir / "test.png", 1, lambda _: None, reference_image=object())
    assert resets == [0, 1]
    assert verified == [True]
    snapshot = runner.memory_snapshot()
    assert snapshot["kind"] == "edit" and snapshot["capturedAt"].endswith("Z")
    assert snapshot["outcome"] == ("completed" if failure is None else "cancelled" if isinstance(failure, Cancelled) else "failed")
    assert snapshot["gpus"]["1"]["peakAllocatedBytes"] == 301
    assert snapshot["placementVerified"] is True
    assert "private" not in str(snapshot)


def test_cancelled_generation_does_not_retain_transient_traceback_frames(config, monkeypatch):
    cuda = SimpleNamespace(synchronize=lambda _: None, reset_peak_memory_stats=lambda _: None,
        memory_allocated=lambda _: 1, memory_reserved=lambda _: 1,
        max_memory_allocated=lambda _: 1, max_memory_reserved=lambda _: 1,
        get_device_properties=lambda _: SimpleNamespace(total_memory=100))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    runner = FluxRunner(config)
    monkeypatch.setattr(runner, "verify_placement", lambda: None)
    references = []
    class Transient:
        pass
    def cancelled(*_args, **_kwargs):
        transient = Transient()
        references.append(weakref.ref(transient))
        raise Cancelled()
    monkeypatch.setattr(runner, "_generate", cancelled)
    try:
        runner.generate("edit", config.data_dir / "unused.png", 1, lambda _: None)
    except Cancelled:
        pass
    assert references[0]() is None
