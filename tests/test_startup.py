import asyncio
import socket
import pytest

from flux_image_server import __main__ as main


def test_occupied_listener_fails_before_model_lifespan(monkeypatch):
    called = []
    monkeypatch.setattr(main, "create_app", lambda: called.append(True))
    with socket.socket() as occupied:
        occupied.bind(("0.0.0.0", 0))
        occupied.listen(1)
        monkeypatch.setenv("PORT", str(occupied.getsockname()[1]))
        with pytest.raises(OSError):
            asyncio.run(main.serve())
    assert called == []
