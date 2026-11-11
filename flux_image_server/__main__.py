import asyncio
import logging
import os
import signal
import uvicorn

from .app import create_app


async def serve():
    app = create_app()
    api = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8001")),
                                        workers=1, access_log=False, log_level="info"))
    metrics = uvicorn.Server(uvicorn.Config(app.state.metrics_app, host="0.0.0.0",
                                            port=int(os.environ.get("METRICS_PORT", "8889")),
                                            workers=1, access_log=False, log_level="warning"))
    # One signal owner coordinates both servers and the one GPU worker.
    loop = asyncio.get_running_loop()
    def stop(*_args):
        api.should_exit = metrics.should_exit = True
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop)
        except NotImplementedError:
            signal.signal(signum, stop)
    # Use the public serve API with capture_signals replaced by a no-op context;
    # both listeners must receive the same shutdown notification.
    from contextlib import nullcontext
    api.capture_signals = nullcontext
    metrics.capture_signals = nullcontext
    tasks = [asyncio.create_task(server.serve()) for server in (api, metrics)]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop()
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(serve())
