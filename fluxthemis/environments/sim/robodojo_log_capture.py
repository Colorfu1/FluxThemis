"""File-backed native log capture for RoboDojo's PhysX monitor.

Kit can write to stdout/stderr while holding the GIL. A pipe drained by a
Python thread in the same process therefore deadlocks when it fills. Native
writes here go straight to regular files; scanning and throttled console
output keep using RoboDojo's monitor and its singleton error state.
"""

import atexit
import os
import sys
import threading
from pathlib import Path


class FileLogCapture:
    """Adapt the upstream monitor's I/O without replacing PhysX detection."""

    def __init__(self, monitor, directory):
        module = sys.modules[type(monitor).__module__]
        for name in ('_process_log_segment', '_fatal_watchdog_loop',
                     '_throttle_generation', '_started', '_shutdown'):
            if not hasattr(monitor, name):
                raise RuntimeError('Unsupported RoboDojo PhysX monitor: ' +
                                   name)
        self.monitor = monitor
        self.directory = Path(directory)
        self._throttle_cls = module._ThrottleState
        self._split = module._split_log_segments
        self._reset = monitor.reset
        self._shutdown = monitor.shutdown
        self._streams = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None

    def install(self):
        if self.monitor._started:
            raise RuntimeError('Install file log capture before PhysX starts')
        # EvalEnv imports this same singleton. Its start/reset/shutdown calls
        # must use the adapter too, rather than reinstalling bounded pipes.
        self.monitor.start = self.start
        self.monitor.reset = self.reset
        self.monitor.shutdown = self.shutdown

    def _throttle(self):
        return self._throttle_cls(self.monitor.THROTTLE_INTERVAL_S,
                                  self.monitor.THROTTLE_KEY_CAP)

    def start(self, enabled=True):
        if self.monitor._started:
            return
        if not enabled:
            raise RuntimeError('FluxThemis requires PhysX error monitoring')
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            for target in (1, 2):
                stream = sys.stdout if target == 1 else sys.stderr
                stream.flush()
                path = self.directory / f'fd{target}.log'
                writer = path.open('ab', buffering=0)
                reader = None
                try:
                    reader = path.open('rb', buffering=0)
                    saved = os.dup(target)
                except BaseException:
                    if reader is not None:
                        reader.close()
                    writer.close()
                    raise
                self._streams[target] = dict(
                    writer=writer,
                    reader=reader,
                    saved=saved,
                    pending=b'',
                    throttle=self._throttle(),
                    generation=self.monitor._throttle_generation)
                os.dup2(writer.fileno(), target)
            self.monitor._started = True
            self._thread = threading.Thread(
                target=self._read_loop, name='PhysXMonitor-files', daemon=True)
            self._thread.start()
            watchdog = threading.Thread(
                target=self.monitor._fatal_watchdog_loop,
                name='PhysXMonitor-fatal-watchdog',
                daemon=True)
            watchdog.start()
            atexit.register(self.shutdown)
        except BaseException:
            self.shutdown()
            raise

    def _drain(self, final=False):
        for target, state in self._streams.items():
            # Kit may replace fd 1/2 at reset. Reattach to the regular file,
            # never to a pipe. Duplicated native handles still write safely.
            writer = state['writer']
            if not self._stop.is_set():
                current = os.fstat(target)
                expected = os.fstat(writer.fileno())
                if (current.st_dev, current.st_ino) != (expected.st_dev,
                                                        expected.st_ino):
                    os.dup2(writer.fileno(), target)
            if state['generation'] != self.monitor._throttle_generation:
                state['throttle'] = self._throttle()
                state['generation'] = self.monitor._throttle_generation
            # Bound each pass to bytes already present, even during a storm.
            reader = state['reader']
            remaining = os.fstat(reader.fileno()).st_size - reader.tell()
            while remaining > 0:
                chunk = reader.read(min(remaining, 65536))
                if not chunk:
                    break
                remaining -= len(chunk)
                segments, state['pending'] = self._split(state['pending'] +
                                                         chunk)
                for segment, term in segments:
                    self.monitor._process_log_segment(state['saved'], segment,
                                                      term, state['throttle'])
            if final and state['pending']:
                self.monitor._process_log_segment(state['saved'],
                                                  state['pending'], b'',
                                                  state['throttle'])
                state['pending'] = b''

    def _read_loop(self):
        try:
            while not self._stop.wait(.05):
                with self._lock:
                    self._drain()
        except Exception as exc:
            # Losing detection must fail the worker instead of silently
            # continuing to score potentially broken physics.
            message = f'[PhysXMonitor] file capture failed: {exc!r}\n'
            try:
                os.write(self._streams[2]['saved'], message.encode())
            finally:
                os._exit(99)

    def reset(self):
        with self._lock:
            self._drain(final=True)
            # Preserve upstream stale-timestamp filtering and sticky fatal
            # state. Raw files are retained for diagnostics across episodes.
            self._reset()

    def shutdown(self):
        if self._stop.is_set():
            return
        self._stop.set()
        for target, state in self._streams.items():
            os.dup2(state['saved'], target)
        self._shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                return  # Do not close descriptors underneath a slow reader.
        with self._lock:
            try:
                self._drain(final=True)
            finally:
                for state in self._streams.values():
                    state['reader'].close()
                    state['writer'].close()
                    os.close(state['saved'])
                self._streams.clear()
