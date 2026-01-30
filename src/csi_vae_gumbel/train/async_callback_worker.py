import queue
import threading
from collections.abc import Callable


class AsyncCallbackWorker(threading.Thread):
    """Asynchronous worker to run callback functions without blocking the training loop."""

    def __init__(self) -> None:
        """Initialize the asynchronous callback worker."""
        super().__init__(target=self._loop, daemon=True)
        self.__q = queue.Queue()
        self.__should_stop = threading.Event()

    def _loop(self) -> None:
        while not self.__should_stop.is_set():
            try:
                callback = self.__q.get(timeout=0.1)
                callback()
            except queue.Empty:
                continue

    def submit(self, callback: Callable, *args: float, **kwargs: float) -> None:
        """Submit a new task to the worker."""
        self.__q.put(lambda: callback(*args, **kwargs))

    def stop(self) -> None:
        """Shutdown the worker thread."""
        self.__should_stop.set()
