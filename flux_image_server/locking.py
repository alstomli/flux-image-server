"""Fail closed when a second service process opens the same local runtime."""
import os


class ProcessLock:
    def __init__(self, path):
        self.path, self.file = path, None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.file.seek(0)
                self.file.write(b"\0")
                self.file.flush()
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.file.close()
            self.file = None
            raise RuntimeError("Another FLUX worker already owns this data directory") from error

    def release(self):
        if self.file:
            self.file.close()
            self.file = None
