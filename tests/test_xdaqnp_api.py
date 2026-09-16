"""Hardware-free tests for the Python wrapper layer in `pylibxdaq.xdaqnp`.

The wrappers are duck-typed over the core handles, so stand-ins are enough to exercise caching,
the single-reader guard and acquisition scoping.
"""

import threading
from types import SimpleNamespace

import pytest

from pylibxdaq import xdaqnp as _api


@pytest.mark.parametrize(
    "native_name",
    ["NP1Probe", "NP2003NP2004Probe", "NP2013NP2014Probe"],
)
def test_probe_native_type(native_name):
    native_type = type(native_name, (), {})
    probe = _api.Probe(native_type(), None)
    assert probe.native_type is native_type
    with pytest.raises(AttributeError):
        probe.native_type = native_type


def test_probe_native_type_does_not_classify_handle():
    probe = _api.Probe(object(), None)
    assert probe.native_type is object


class FakeCoreStream:

    def __init__(self):
        self.calls = []

    def read(self, max_items, timeout_ms):
        self.calls.append((max_items, timeout_ms))
        return "chunk"


class BlockingCoreStream:

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def read(self, max_items, timeout_ms):
        self.entered.set()
        assert self.release.wait(5)
        return "chunk"


class FakeCoreProbe:

    def __init__(self):
        self.streams = 0

    @property
    def stream(self):
        # The real handle returns a fresh object on every access; that is what the wrapper's
        # caching exists to paper over.
        self.streams += 1
        return FakeCoreStream()


class FakeCoreController:

    def __init__(self):
        self.events = []

    def start_stream(self, target):
        self.events.append(("start", target))

    def stop_stream(self, target):
        self.events.append(("stop", target))

    def create_io_stream(self, max_queued):
        self.max_queued = max_queued
        return FakeCoreStream()


class FakeController(_api.Controller):

    def __init__(self):
        _api._Wrapper.__init__(self, FakeCoreController())
        self._ports = {}
        self._ports_lock = threading.Lock()


@pytest.mark.parametrize("body_raises", [False, True])
def test_controller_io_scope_stops_on_exit(body_raises):
    controller = FakeController()

    def acquire():
        with controller.io_stream(max_queued=32) as stream:
            assert isinstance(stream, _api.IOStream)
            assert controller._core.max_queued == 32
            assert stream.read(8, 5.0) == "chunk"
            assert stream._core.calls == [(8, 5.0)]
            assert controller._core.events == [("start", stream._core)]
            with pytest.raises(RuntimeError, match="active streaming context"):
                controller.stop_stream(stream)
            if body_raises:
                raise ValueError("body failed")

    if body_raises:
        with pytest.raises(ValueError, match="body failed"):
            acquire()
    else:
        acquire()
    assert [event for event, target in controller._core.events] == ["start", "stop"]
    assert controller._core.events[0][1] is controller._core.events[1][1]


def test_io_stream_rejects_overlapping_scopes():
    controller = FakeController()
    stream = controller.create_io_stream()
    with stream.streaming():
        with pytest.raises(RuntimeError, match="already active"):
            with stream.streaming():
                pytest.fail("Overlapping IO scope entered")
    assert not stream._running


def test_read_forwards_arguments():
    stream = _api.Stream(FakeCoreStream(), "test stream")
    assert stream.read(max_items=8, timeout_ms=5.0) == "chunk"
    assert stream._core.calls == [(8, 5.0)]


def test_concurrent_read_raises():
    core = BlockingCoreStream()
    stream = _api.Stream(core, "test stream")
    reader = threading.Thread(target=stream.read)
    reader.start()
    try:
        assert core.entered.wait(5)
        with pytest.raises(RuntimeError, match="single-consumer"):
            stream.read()
    finally:
        core.release.set()
        reader.join(5)


def test_sequential_reads_from_different_threads_are_allowed():
    stream = _api.Stream(FakeCoreStream(), "test stream")
    results = []
    for _ in range(2):
        thread = threading.Thread(target=lambda: results.append(stream.read()))
        thread.start()
        thread.join(5)
    assert results == ["chunk", "chunk"]


def test_failed_read_releases_the_guard():

    class Failing:

        def read(self, max_items, timeout_ms):
            raise ValueError("boom")

    stream = _api.Stream(Failing(), "test stream")
    with pytest.raises(ValueError):
        stream.read()
    with pytest.raises(ValueError):
        stream.read()


def test_probe_caches_one_stream():
    core = FakeCoreProbe()
    probe = _api.Probe(core, FakeController())
    assert probe.stream is probe.stream
    assert core.streams == 1


def test_np1_headstage_caches_its_probe():
    core_probe = FakeCoreProbe()

    class FakeCoreHeadstage:

        @property
        def probe(self):
            return core_probe

        def init_probe(self):
            return core_probe

    headstage = _api.NP1Headstage(FakeCoreHeadstage(), FakeController())
    assert headstage.init_probe() is headstage.probe


def test_np1_headstage_reports_missing_probe():

    class FakeCoreHeadstage:
        probe = None

    assert _api.NP1Headstage(FakeCoreHeadstage(), FakeController()).probe is None


def test_np2_headstage_caches_per_dock():

    class FakeCoreHeadstage:

        def probe(self, dock):
            return FakeCoreProbe()

        def init_probe(self, dock):
            return FakeCoreProbe()

    headstage = _api.NP2Headstage(FakeCoreHeadstage(), FakeController())
    front = headstage.init_probe("front")
    back = headstage.init_probe("back")
    assert front is headstage.probe("front")
    assert back is headstage.probe("back")
    assert front is not back


def test_streaming_stops_on_exit():
    controller = FakeController()
    core = FakeCoreProbe()
    probe = _api.Probe(core, controller)

    with probe.streaming() as streamed:
        assert streamed is probe
        assert controller._core.events == [("start", core)]

    assert controller._core.events == [("start", core), ("stop", core)]


def test_streaming_stops_when_body_raises():
    controller = FakeController()
    core = FakeCoreProbe()
    probe = _api.Probe(core, controller)

    with pytest.raises(ValueError):
        with probe.streaming():
            raise ValueError("boom")

    assert controller._core.events[-1] == ("stop", core)


def test_start_stream_unwraps_handles():
    controller = FakeController()
    core = FakeCoreProbe()
    probe = _api.Probe(core, controller)

    controller.start_stream(probe)
    controller.stop_stream(probe)

    assert controller._core.events == [("start", core), ("stop", core)]


def test_independent_scopes_stop_only_their_own_stream():
    controller = FakeController()
    probes = [_api.Probe(FakeCoreProbe(), controller) for _ in range(2)]
    with probes[0].streaming():
        with probes[1].streaming():
            assert controller._core.events == [("start", probe._core) for probe in probes]
        assert controller._core.events[-1] == ("stop", probes[1]._core)
        assert probes[0]._running
    assert controller._core.events[-1] == ("stop", probes[0]._core)


def test_streaming_rejects_overlap_and_manual_control():
    controller = FakeController()
    probe = _api.Probe(FakeCoreProbe(), controller)
    with probe.streaming():
        with pytest.raises(RuntimeError, match="already active"):
            with probe.streaming():
                pytest.fail("Overlapping scope entered")
        for operation in (probe.start, probe.stop, lambda: controller.start_stream(probe),
                          lambda: controller.stop_stream(probe)):
            with pytest.raises(RuntimeError, match="active streaming context"):
                operation()
        assert controller._core.events == [("start", probe._core)]
    assert controller._core.events[-1] == ("stop", probe._core)


def test_manual_start_is_idempotent_and_rejects_scope():
    controller = FakeController()
    probe = _api.Probe(FakeCoreProbe(), controller)
    probe.start()
    probe.start()
    assert controller._core.events == [("start", probe._core)]
    with pytest.raises(RuntimeError, match="already active"):
        with probe.streaming():
            pytest.fail("Already-running stream was claimed")
    probe.stop()
    probe.stop()
    with probe.streaming():
        pass


def test_failed_start_does_not_claim_scope():
    controller = FakeController()
    probe = _api.Probe(FakeCoreProbe(), controller)
    original = controller._core.start_stream

    def fail_start(target):
        raise RuntimeError("start failed")

    controller._core.start_stream = fail_start
    with pytest.raises(RuntimeError, match="start failed"):
        with probe.streaming():
            pytest.fail("Failed start entered body")
    controller._core.start_stream = original
    with probe.streaming():
        pass


def test_failed_scope_stop_allows_manual_retry():
    controller = FakeController()
    probe = _api.Probe(FakeCoreProbe(), controller)
    original = controller._core.stop_stream

    def fail_stop(target):
        raise RuntimeError("stop failed")

    controller._core.stop_stream = fail_stop
    with pytest.raises(RuntimeError, match="stop failed"):
        with probe.streaming():
            pass
    assert probe._running
    controller._core.stop_stream = original
    probe.stop()
    assert not probe._running


def test_controller_rejects_foreign_stream():
    controller = FakeController()
    foreign = _api.Probe(FakeCoreProbe(), FakeController())
    with pytest.raises(ValueError):
        controller.start_stream(foreign)
    with pytest.raises(ValueError):
        controller.stop_stream(foreign)
    assert controller._core.events == []


def test_group_api_removed():
    assert not hasattr(_api.Controller, "start_streams")
    assert not hasattr(_api.Controller, "streaming")
    assert not hasattr(_api.xdaqnp_core.Controller, "start_streams")


def test_attributes_are_forwarded_to_the_core_handle():

    class FakeCoreProbeWithControl(FakeCoreProbe):

        def set_opmode(self, mode):
            return f"opmode={mode}"

    probe = _api.Probe(FakeCoreProbeWithControl(), FakeController())
    assert probe.set_opmode("RECORDING") == "opmode=RECORDING"
    with pytest.raises(AttributeError):
        probe.nonexistent


def test_port_caches_headstage_and_controller_delegates():
    core_headstage = object()

    class FakeCoreControllerWithDetect(FakeCoreController):

        def detect_headstage(self, port):
            self.events.append(("detect", port))
            return core_headstage

    controller = FakeController()
    controller._core = FakeCoreControllerWithDetect()
    core_port = SimpleNamespace(headstage_generation=0)
    port = _api.Port(core_port, controller)

    assert port.headstage is None
    headstage = port.detect_headstage()
    assert isinstance(headstage, _api.NP2Headstage)
    assert port.headstage is headstage
    assert controller.detect_headstage(port) is port.headstage
    assert port.headstage is headstage
    assert controller._core.events == [("detect", core_port), ("detect", core_port)]


def test_healthy_detection_preserves_active_probe(monkeypatch):
    core_probe = FakeCoreProbe()
    core_headstage = SimpleNamespace(init_probe=lambda: core_probe)
    monkeypatch.setattr(_api, "_wrap_headstage", _api.NP1Headstage)
    controller = FakeController()
    controller._core.detect_headstage = lambda port: core_headstage
    port = _api.Port(SimpleNamespace(headstage_generation=0), controller)
    headstage = port.detect_headstage()
    probe = headstage.init_probe()
    with probe.streaming():
        assert port.detect_headstage() is headstage
        assert port.headstage.init_probe() is probe
        assert probe._running
    assert controller._core.events == [("start", core_probe), ("stop", core_probe)]


@pytest.mark.parametrize("replacement_fails", [False, True])
def test_replacement_invalidates_cached_headstage(monkeypatch, replacement_fails):
    monkeypatch.setattr(_api, "_wrap_headstage", _api.NP1Headstage)
    controller = FakeController()
    controller._core.detect_headstage = lambda port: SimpleNamespace()
    core_port = SimpleNamespace(headstage_generation=0)
    port = _api.Port(core_port, controller)
    original = port.detect_headstage()

    def replace(port):
        port.headstage_generation += 1
        if replacement_fails:
            raise RuntimeError("No known headstage detected")
        return SimpleNamespace()

    controller._core.detect_headstage = replace
    if replacement_fails:
        with pytest.raises(RuntimeError, match="No known headstage"):
            port.detect_headstage()
        assert port.headstage is None
        controller._core.detect_headstage = lambda port: SimpleNamespace()
    replacement = port.detect_headstage()
    assert replacement is not original
    assert port.headstage is replacement


def test_retired_context_does_not_stop_replacement_or_other_port(monkeypatch):
    monkeypatch.setattr(_api, "_wrap_headstage", _api.NP1Headstage)
    controller = FakeController()
    active = set()
    core_port = SimpleNamespace(headstage_generation=0)
    original_core = FakeCoreProbe()
    replacement_core = FakeCoreProbe()
    other_core = FakeCoreProbe()
    current = original_core

    def detect(port):
        return SimpleNamespace(init_probe=lambda: current)

    def start(target):
        active.add(target)

    def stop(target):
        if target is original_core and core_port.headstage_generation != 0:
            return
        active.discard(target)

    controller._core.detect_headstage = detect
    controller._core.start_stream = start
    controller._core.stop_stream = stop
    port = _api.Port(core_port, controller)
    original = port.detect_headstage().init_probe()
    other = _api.Probe(other_core, controller)
    with other.streaming():
        with original.streaming():
            active.remove(original_core)
            core_port.headstage_generation += 1
            current = replacement_core
            replacement = port.detect_headstage().init_probe()
            assert replacement is not original
            assert active == {other_core}
            replacement.start()
        assert not original._running
        assert active == {replacement_core, other_core}
        replacement.stop()
        assert active == {other_core}
    assert not active


def test_controller_caches_ports():

    class FakeCorePort:

        def __init__(self, index):
            self.index = index

    class FakeCoreControllerWithPorts(FakeCoreController):
        num_ports = 2

        def port(self, index):
            self.events.append(("port", index))
            return FakeCorePort(index)

    controller = FakeController()
    controller._core = FakeCoreControllerWithPorts()

    assert controller.port(0) is controller.port(0)
    assert [port.index for port in controller.ports] == [0, 1]
    assert controller._core.events == [("port", 0), ("port", 1)]


def _record_core_controller(monkeypatch) -> list:
    """Replace `_core.Controller` with a stand-in, returning the list of handles it was given."""
    seen: list = []

    def fake(raw):
        seen.append(raw)
        return FakeCoreController()

    monkeypatch.setattr(_api._core, "Controller", fake)
    return seen


def test_controller_unwraps_the_device_and_keeps_it(monkeypatch):
    seen = _record_core_controller(monkeypatch)
    handle = object()
    device = SimpleNamespace(raw=handle)

    controller = _api.Controller(device)

    assert seen == [handle]
    assert controller.device is device


def test_controller_accepts_a_bare_handle(monkeypatch):
    seen = _record_core_controller(monkeypatch)
    handle = object()

    assert _api.Controller(handle).device is handle
    assert seen == [handle]


def test_controller_rejects_a_closed_device():
    with pytest.raises(ValueError, match="closed"):
        _api.Controller(SimpleNamespace(raw=None))
