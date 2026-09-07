"""Background worker infrastructure.

Provides QThread-based workers that run heavy DSP operations off the GUI
thread, with progress reporting and cancellation support.
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class WorkerSignals(QObject):
    """Signals emitted by a :class:`Worker` during execution."""

    started = Signal()
    progress = Signal(float, str)    # (fraction 0..1, description)
    finished = Signal(object)        # result object
    error = Signal(str)              # error message
    cancelled = Signal()


class Worker(QRunnable):
    """Run a callable in the thread pool with progress reporting.

    Usage::

        def heavy_task(progress_cb):
            for i in range(100):
                progress_cb(i / 100, f"Step {i}")
            return "done"

        worker = Worker(heavy_task)
        worker.signals.finished.connect(on_done)
        WorkerPool.instance().start(worker)
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self._cancelled = False
        self.setAutoDelete(True)

    def cancel(self) -> None:
        """Request cancellation.  The worker function must check
        ``is_cancelled()`` periodically."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    @Slot()
    def run(self) -> None:
        self.signals.started.emit()
        try:
            result = self.fn(
                *self.args,
                progress_cb=self._emit_progress,
                cancel_check=self.is_cancelled,
                **self.kwargs,
            )
            if self._cancelled:
                self.signals.cancelled.emit()
            else:
                self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.error.emit(str(exc))

    def _emit_progress(self, fraction: float, description: str = "") -> None:
        self.signals.progress.emit(fraction, description)


class WorkerPool:
    """Singleton wrapper around :class:`QThreadPool`."""

    _instance: WorkerPool | None = None

    def __init__(self) -> None:
        self._pool = QThreadPool.globalInstance()
        self._pool.setMaxThreadCount(4)

    @classmethod
    def instance(cls) -> "WorkerPool":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self, worker: Worker) -> None:
        """Submit a worker to the thread pool."""
        self._pool.start(worker)

    def set_max_threads(self, n: int) -> None:
        self._pool.setMaxThreadCount(n)

    @property
    def active_count(self) -> int:
        return self._pool.activeThreadCount()
