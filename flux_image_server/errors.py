class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, retry_after: int | None = None):
        self.code, self.message, self.status, self.retry_after = code, message, status, retry_after
        super().__init__(message)


class Cancelled(Exception):
    pass


class GenerationTimeout(Exception):
    pass


class PromptTooLong(Exception):
    pass


def is_oom(error):
    """Classify native CUDA/host OOM, including wrapping, without reading messages."""
    pending, visited = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, MemoryError) or type(current).__name__ == "OutOfMemoryError":
            return True
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)
    return False
