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
