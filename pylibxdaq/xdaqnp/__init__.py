"""Neuropixels control and acquisition for XDAQ.

Typical use::

    from pylibxdaq import xdaqnp
    from pylibxdaq.device import list_devices

    devices = list_devices()
    if not devices:
        raise RuntimeError("No usable XDAQ device found")

    with devices[0].with_mode("np").create() as device:
        controller = xdaqnp.Controller(device)
        port = controller.port(0)
        headstage = port.detect_headstage()
        probe = headstage.init_probe()

        probe.set_opmode(xdaqnp.OpMode.RECORDING)
        probe.apply_shift_register_config()
        port.switch_to_data_link()

        with probe.streaming():
            chunk = probe.stream.read(timeout_ms=100)
            print(chunk.ap.shape)  # (packets * 12, 384) int16

Wraps :mod:`xdaqnp_core` handles with stable identity, single-reader guards, and scoped
acquisition. Other attributes are forwarded unchanged; raw bindings remain available
as ``pylibxdaq.xdaqnp.xdaqnp_core``.
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

# On Windows, register the parent package directory for extension DLL dependencies.
if sys.platform == "win32":
    os.add_dll_directory(str(Path(__file__).parent.parent))

from ..device import Device, DeviceInfo  # noqa: E402
from . import xdaqnp_core as _core  # noqa: E402
from .xdaqnp_core import (  # noqa: E402
    CalMode,
    ChannelReference,
    Dock,
    IOChunk,
    NP1Chunk,
    NP2Chunk,
    OpMode,
    ProbeInfo,
)

__all__ = [
    # this layer
    "Controller",
    "IOStream",
    "NP1Headstage",
    "NP2Headstage",
    "Port",
    "Probe",
    "Stream",
    # re-exported from pylibxdaq: the device types the whole package shares
    "Device",
    "DeviceInfo",
    # re-exported from the extension module
    "CalMode",
    "ChannelReference",
    "Dock",
    "IOChunk",
    "NP1Chunk",
    "NP2Chunk",
    "OpMode",
    "ProbeInfo",
]


class _Wrapper:
    """Forwards every attribute it does not define itself to the wrapped core handle."""

    def __init__(self, core: Any) -> None:
        self._core = core

    def __getattr__(self, name: str) -> Any:
        return getattr(self._core, name)

    def __repr__(self) -> str:
        return repr(self._core)


class Stream(_Wrapper):
    """A data-plane handle that rejects concurrent readers.

    ``read()`` releases the GIL. A per-wrapper lock protects the single-consumer queue,
    raising :class:`RuntimeError` on concurrent reads. Sequential reads from different
    threads are allowed; cached wrappers ensure readers share the same guard.

    A stream holds its queue directly, so its handle stays valid after probe teardown.
    Lifecycle resets discard queued samples; copy or retain returned chunks before stopping.
    """

    def __init__(self, core: Any, label: str) -> None:
        super().__init__(core)
        self._label = label
        self._read_lock = threading.Lock()

    def read(self, max_items: int = 0, timeout_ms: float = 0.0) -> Any:
        """Drain up to ``max_items`` queued items (0 = everything available) into a new chunk."""
        if not self._read_lock.acquire(blocking=False):
            raise RuntimeError(
                f"{self._label} is already being read by another thread. The queue is "
                "single-consumer; use one reader thread, or hand the chunks to the other "
                "thread instead."
            )
        try:
            return self._core.read(max_items, timeout_ms)
        finally:
            self._read_lock.release()

    def __repr__(self) -> str:
        return f"<{self._label}>"


class _Streamable:
    """Start/stop plumbing shared by probes and the IO stream."""

    _core: Any
    _controller: "Controller"

    def _init_lifecycle(self) -> None:
        self._lifecycle_lock = threading.Lock()
        self._running = False
        self._scoped = False

    def start(self) -> None:
        """Start this stream independently. Repeated manual starts are no-ops."""
        with self._lifecycle_lock:
            if self._scoped:
                raise RuntimeError("Stream is owned by an active streaming context")
            if not self._running:
                self._controller._core.start_stream(self._core)
                self._running = True

    def stop(self) -> None:
        """Stop delivery and discard queued output after decoder acknowledgement."""
        with self._lifecycle_lock:
            if self._scoped:
                raise RuntimeError("Stream is owned by an active streaming context")
            self._controller._core.stop_stream(self._core)
            self._running = False

    @contextmanager
    def streaming(self) -> Generator[Any, None, None]:
        """Own one inactive stream until exit. Overlapping scopes are rejected.

        Use either this wrapper or the raw core API for lifecycle control, not both.
        """
        with self._lifecycle_lock:
            if self._running or self._scoped:
                raise RuntimeError("Stream is already active")
            self._controller._core.start_stream(self._core)
            self._running = True
            self._scoped = True
        try:
            yield self
        finally:
            with self._lifecycle_lock:
                try:
                    self._controller._core.stop_stream(self._core)
                    self._running = False
                finally:
                    self._scoped = False


class IOStream(Stream, _Streamable):
    """ADC, DAC and digital lines. Current firmware produces IO only while a probe runs.

    Starting IO alone succeeds, but reads remain empty until probe production starts.
    """

    def __init__(self, core: Any, controller: "Controller") -> None:
        Stream.__init__(self, core, "IO stream")
        self._controller = controller
        self._init_lifecycle()


class Probe(_Wrapper, _Streamable):
    """A probe's control plane, plus its cached data stream."""

    def __init__(self, core: Any, controller: "Controller") -> None:
        super().__init__(core)
        self._controller = controller
        self._stream: Optional[Stream] = None
        self._init_lifecycle()
        self._stream_lock = threading.Lock()

    @property
    def native_type(self) -> type:
        """The wrapped handle's concrete xdaqnp_core type; no hardware I/O."""
        return type(self._core)

    @property
    def stream(self) -> Stream:
        # Cache under a lock so all readers share one guard.
        if self._stream is None:
            with self._stream_lock:
                if self._stream is None:
                    self._stream = Stream(self._core.stream, f"{type(self._core).__name__} stream")
        return self._stream


class NP1Headstage(_Wrapper):

    def __init__(self, core: Any, controller: "Controller") -> None:
        super().__init__(core)
        self._controller = controller
        self._probe: Optional[Probe] = None
        self._probe_lock = threading.Lock()

    def _wrap(self, core_probe: Any) -> Optional[Probe]:
        if core_probe is None:
            return None
        if self._probe is None:
            with self._probe_lock:
                if self._probe is None:
                    self._probe = Probe(core_probe, self._controller)
        return self._probe

    @property
    def probe(self) -> Optional[Probe]:
        """The initialized probe, or None."""
        return self._wrap(self._core.probe)

    def init_probe(self) -> Probe:
        """Initialize the probe if needed and return a handle to it."""
        probe = self._wrap(self._core.init_probe())
        assert probe is not None
        return probe


class NP2Headstage(_Wrapper):

    def __init__(self, core: Any, controller: "Controller") -> None:
        super().__init__(core)
        self._controller = controller
        self._probes: dict[Any, Probe] = {}
        self._probes_lock = threading.Lock()

    def _wrap(self, dock: Any, core_probe: Any) -> Optional[Probe]:
        if core_probe is None:
            return None
        probe = self._probes.get(dock)
        if probe is None:
            with self._probes_lock:
                probe = self._probes.get(dock)
                if probe is None:
                    probe = Probe(core_probe, self._controller)
                    self._probes[dock] = probe
        return probe

    def probe(self, dock: Any) -> Optional[Probe]:
        """The initialized probe on `dock`, or None."""
        return self._wrap(dock, self._core.probe(dock))

    def init_probe(self, dock: Any) -> Probe:
        """Initialize the probe on `dock` if needed and return a handle to it."""
        probe = self._wrap(dock, self._core.init_probe(dock))
        assert probe is not None
        return probe


def _wrap_headstage(core_headstage: Any, controller: "Controller") -> Any:
    if isinstance(core_headstage, _core.NP1Headstage):
        return NP1Headstage(core_headstage, controller)
    return NP2Headstage(core_headstage, controller)


class Port(_Wrapper):

    def __init__(self, core: Any, controller: "Controller") -> None:
        super().__init__(core)
        self._controller = controller
        self._headstage: Any = None
        self._headstage_generation: Optional[int] = None
        self._detection_lock = threading.Lock()

    @property
    def headstage(self) -> Any:
        """The headstage from the most recent detection on this port, or None."""
        return self._headstage

    def detect_headstage(self) -> Any:
        """Detect (or re-detect) the headstage on this port.

        Healthy headstages retain their handles and active streams. Replacement retires this
        port's streams and invalidates old handles, even if detection then fails. A replacement
        probe must be initialized, configured, and started explicitly.
        """
        with self._detection_lock:
            try:
                core_headstage = self._controller._core.detect_headstage(self._core)
            finally:
                generation = self._core.headstage_generation
                if generation != self._headstage_generation:
                    self._headstage = None
                    self._headstage_generation = generation
            if self._headstage is None:
                self._headstage = _wrap_headstage(core_headstage, self._controller)
            return self._headstage


class Controller(_Wrapper):
    """Control plane for a Neuropixels-mode XDAQ."""

    def __init__(self, device: Device) -> None:
        # Accept either a device wrapper or a raw extension handle.
        raw = getattr(device, "raw", device)
        if raw is None:
            raise ValueError("Device is already closed")
        super().__init__(_core.Controller(raw))
        self._device = device
        self._ports: dict[int, Port] = {}
        self._ports_lock = threading.Lock()

    @property
    def device(self) -> Device:
        """The device this controller was created from, exactly as it was passed in.

        The controller holds its own share of the underlying device, so closing this object does
        not release the hardware until the controller is gone too.
        """
        return self._device

    def port(self, index: int) -> Port:
        """Open (or re-obtain) the port at `index`. Ports are 0-based."""
        port = self._ports.get(index)
        if port is None:
            with self._ports_lock:
                port = self._ports.get(index)
                if port is None:
                    port = Port(self._core.port(index), self)
                    self._ports[index] = port
        return port

    @property
    def ports(self) -> list[Port]:
        return [self.port(index) for index in range(self._core.num_ports)]

    def detect_headstage(self, port: Port) -> Any:
        return port.detect_headstage()

    def start_stream(self, target: Any) -> None:
        if not isinstance(target, _Streamable) or target._controller is not self:
            raise ValueError("Expected a stream belonging to this controller")
        target.start()

    def stop_stream(self, target: Any) -> None:
        if not isinstance(target, _Streamable) or target._controller is not self:
            raise ValueError("Expected a stream belonging to this controller")
        target.stop()

    def create_io_stream(self, max_queued: int = 1 << 17) -> IOStream:
        """Create the IO frame stream. Only one may exist at a time."""
        return IOStream(self._core.create_io_stream(max_queued), self)

    @contextmanager
    def io_stream(self, max_queued: int = 1 << 17) -> Generator[IOStream, None, None]:
        """Create the IO stream, start it, and stop it on exit."""
        stream = self.create_io_stream(max_queued)
        with stream.streaming():
            yield stream
