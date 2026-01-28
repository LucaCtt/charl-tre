import queue
import threading
from collections.abc import Callable


class AsyncCallbackWorker:
    """Asynchronous worker to run callback functions without blocking the training loop."""

    def __init__(self) -> None:
        """Initialize the asynchronous callback worker."""
        self.__q = queue.Queue()
        self.__worker = threading.Thread(target=self._loop, daemon=True)
        self.__worker.start()

    def _loop(self) -> None:
        while True:
            callback = self.__q.get()
            callback()

    def submit(self, callback: Callable, *args: float, **kwargs: float) -> None:
        """Submit a new task to the worker."""
        self.__q.put(lambda: callback(*args, **kwargs))
