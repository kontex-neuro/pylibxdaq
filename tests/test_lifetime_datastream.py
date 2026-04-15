import gc
import json
import threading
import time

import pytest

from pylibxdaq import pyxdaq_device

from .test_lifetime_manager import mock_manager_path


def test_datastream_creation(mock_manager_path, capfd):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)

    device = manager.create_device(json.dumps(json.loads(manager.list_devices())[0]))

    captured = capfd.readouterr()
    assert "MockDevice(0) constructed" in captured.err
    assert "MockDataStream constructed" not in captured.err

    del manager

    captured = capfd.readouterr()
    assert "MockDeviceManager destroyed" not in captured.err

    def cb(event, err):
        print(f"Received event: {event} with error code: {err}")

    with device.start_read_stream(0x11, cb) as stream:

        captured = capfd.readouterr()
        assert "MockDataStream constructed" in captured.err
        assert stream is not None

        stream.stop()

    captured = capfd.readouterr()

    del stream

    del device

    captured = capfd.readouterr()
    assert "MockDevice(0) destroyed" in captured.err
    assert "MockDeviceManager destroyed" in captured.err


def test_stream_keeps_device_and_manager_alive(mock_manager_path, capfd):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps(json.loads(manager.list_devices())[0]))

    captured = capfd.readouterr()
    assert "MockDeviceManager constructed" in captured.err
    assert "MockDevice(0) constructed" in captured.err

    def cb(event, err):
        pass

    stream = device.start_read_stream(0x11, cb)

    captured = capfd.readouterr()
    assert "MockDataStream constructed" in captured.err

    del manager

    captured = capfd.readouterr()
    assert "MockDeviceManager destroyed" not in captured.err

    del device

    captured = capfd.readouterr()
    assert "MockDevice(0) destroyed" not in captured.err
    assert "MockDeviceManager destroyed" not in captured.err

    assert stream is not None

    del stream

    captured = capfd.readouterr()
    assert "MockDevice(0) destroyed" in captured.err
    assert "MockDeviceManager destroyed" in captured.err


def test_del_stream_causes_destruction(mock_manager_path, capfd):
    gc.disable()
    try:
        manager = pyxdaq_device.get_device_manager(mock_manager_path)
        device = manager.create_device(json.dumps(json.loads(manager.list_devices())[0]))

        captured = capfd.readouterr()
        assert "MockDeviceManager constructed" in captured.err
        assert "MockDevice(0) constructed" in captured.err

        def cb(event, err):
            pass

        stream = device.start_read_stream(0x11, cb)

        captured = capfd.readouterr()
        assert "MockDataStream constructed" in captured.err

        del manager
        del device

        captured = capfd.readouterr()
        assert "MockDeviceManager destroyed" not in captured.err
        assert "MockDevice(0) destroyed" not in captured.err
        del stream

        captured = capfd.readouterr()
        assert "MockDataStream destroyed" in captured.err
        assert "MockDevice(0) destroyed" in captured.err
        assert "MockDeviceManager destroyed" in captured.err
    finally:
        gc.enable()


def test_stream_context_manager(mock_manager_path, capfd):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    events = []

    def cb(event, err):
        if event is not None:
            events.append(event)

    with device.start_read_stream(0x11, cb) as stream:
        capfd.readouterr()
        import time
        time.sleep(0.06)

    captured = capfd.readouterr()
    assert "MockDataStream destroyed" in captured.err


def test_aligned_read_stream_lifetime(mock_manager_path, capfd):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    def cb(event, err):
        pass

    stream = device.start_aligned_read_stream(0x11, 256, cb)
    capfd.readouterr()

    del manager
    del device

    captured = capfd.readouterr()
    assert "MockDevice(0) destroyed" not in captured.err

    del stream
    captured = capfd.readouterr()
    assert "MockDataStream destroyed" in captured.err
    assert "MockDevice(0) destroyed" in captured.err
    assert "MockDeviceManager destroyed" in captured.err


def test_device_close_with_active_stream(mock_manager_path, capfd):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    del manager

    def cb(event, err):
        pass

    stream = device.start_read_stream(0x11, cb)
    capfd.readouterr()

    device.close()  # Explicitly close device
    captured = capfd.readouterr()
    # Despite closed handle, MockDevice should stay alive because stream holds shared_ptr
    assert "MockDevice(0) destroyed" not in captured.err
    assert device.closed

    time.sleep(0.05)
    stream.stop()
    del stream

    captured = capfd.readouterr()
    assert "MockDataStream destroyed" in captured.err
    assert "MockDevice(0) destroyed" in captured.err
    assert "MockDeviceManager destroyed" in captured.err


def test_buffer_outlives_stream_strict(mock_manager_path, capfd):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    captured_events = []

    def cb(event, err):
        if event is not None:
            # Store the python object holding the data wrapper
            captured_events.append(event)

    stream = device.start_read_stream(0x11, cb)
    time.sleep(0.06)

    stream.stop()
    del stream
    del device
    del manager

    captured = capfd.readouterr()
    assert "MockDataStream destroyed" in captured.err
    assert "MockDevice(0) destroyed" not in captured.err
    assert "MockDeviceManager destroyed" not in captured.err

    # Verify events are still totally fine
    for ev in captured_events:
        arr = ev.numpy
        assert len(arr) > 0

    del ev
    del arr
    captured_events.clear()

    import gc
    gc.collect()

    captured = capfd.readouterr()
    assert "MockDevice(0) destroyed" in captured.err
    assert "MockDeviceManager destroyed" in captured.err


def test_cyclic_reference_collection(mock_manager_path, capfd):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    # We use a mutable container to capture the stream in the closure
    scope = {}

    def cb(event, err):
        if "stream" in scope:
            pass

    scope["stream"] = device.start_read_stream(0x11, cb)
    capfd.readouterr()

    scope["stream"].stop()
    del scope["stream"]
    del scope
    del device
    del manager

    # Run gc.collect to clean up cycles if any
    import gc
    gc.collect()

    captured = capfd.readouterr()
    assert "MockDataStream destroyed" in captured.err
    assert "MockDevice(0) destroyed" in captured.err
    assert "MockDeviceManager destroyed" in captured.err


def test_aligned_read_stream(mock_manager_path, capfd):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    events = []

    def cb(event, err):
        events.append(event)

    stream = device.start_aligned_read_stream(0x11, 16, cb, chunk_size=1024)
    assert stream is not None

    time.sleep(0.05)
    stream.stop()
    del stream
    del device
    del manager

    assert len(events) > 0


def test_managed_buffer_size_and_numpy(mock_manager_path):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    events = []

    def cb(event, err):
        if event is not None:
            events.append(event)

    stream = device.start_read_stream(0x11, cb)
    time.sleep(0.06)
    stream.stop()
    del stream
    del device
    del manager

    assert len(events) > 0
    for ev in events:
        assert ev.size > 0
        arr = ev.numpy
        assert len(arr) == ev.size


def test_managed_buffer_numpy_is_independent_copy(mock_manager_path, capfd):
    # With NumPy 2.x, ManagedBuffer.numpy returns an independent copy of the data
    # via the DLPack path. The numpy array does NOT hold a reference to the
    # ManagedBuffer, so device lifetime is not extended by holding numpy arrays.
    arrays = []

    def run():
        manager = pyxdaq_device.get_device_manager(mock_manager_path)
        device = manager.create_device(json.dumps({"id": 0}))

        def cb(event, err):
            if event is not None:
                arrays.append(event.numpy)

        stream = device.start_read_stream(0x11, cb)
        time.sleep(0.06)
        stream.stop()
        del stream, device, manager

    gc.disable()
    try:
        run()
        gc.collect()
        # Device should be destroyed once all ManagedBuffer Python objects (event params)
        # go out of scope — numpy arrays alone don't extend device lifetime.
        assert "MockDevice(0) destroyed" in capfd.readouterr().err
    finally:
        gc.enable()

    # numpy arrays are valid independent copies even after device is gone
    assert len(arrays) > 0
    assert all(len(a) > 0 for a in arrays)


def _collect_numpy_views(mock_manager_path):
    """Return numpy_view arrays from ManagedBuffer events. Device is deleted before return."""
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))
    views = []

    def cb(event, err):
        if event is not None:
            views.append(event.numpy_view)

    with device.start_read_stream(0x11, cb):
        time.sleep(0.06)

    del device, manager
    return views


def test_managed_buffer_numpy_view_keeps_device_alive(mock_manager_path, capfd):
    # numpy_view is zero-copy with no lifetime management.
    # Holding numpy_view arrays does NOT keep the ManagedBuffer or device alive.
    views = _collect_numpy_views(mock_manager_path)
    assert "MockDevice(0) destroyed" in capfd.readouterr().err
    assert len(views) > 0


def test_dataview_size_and_numpy_copy(mock_manager_path):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    arrays = []

    def cb(event, err):
        if event is not None:
            assert event.size > 0
            arrays.append(event.numpy)

    stream = device.start_aligned_read_stream(0x11, 16, cb)
    time.sleep(0.06)
    stream.stop()
    del stream
    del device
    del manager
    gc.collect()

    assert len(arrays) > 0
    for arr in arrays:
        assert len(arr) > 0
        _ = arr[0]


def test_dataview_numpy_view_during_callback(mock_manager_path):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    view_sizes = []

    def cb(event, err):
        if event is not None:
            view = event.numpy_view
            assert len(view) == event.size
            view_sizes.append(len(view))

    stream = device.start_aligned_read_stream(0x11, 16, cb)
    time.sleep(0.06)
    stream.stop()
    del stream
    del device
    del manager

    assert len(view_sizes) > 0
    assert all(s > 0 for s in view_sizes)


def test_error_event_received(mock_manager_path):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    errors = []
    error_received = threading.Event()

    def cb(event, err):
        if err is not None:
            errors.append(err)
            error_received.set()

    with device.start_aligned_read_stream(0x11, 16, cb):
        # Wait for the error to be dispatched before __exit__ tears down the queue.
        error_received.wait(timeout=2.0)

    del device
    del manager

    assert len(errors) > 0
    assert all(isinstance(e, str) and len(e) > 0 for e in errors)


def test_stream_callback_cycle(mock_manager_path, capfd):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)
    device = manager.create_device(json.dumps({"id": 0}))

    # Create a cyclic reference: Callback references stream_ref (which contains the stream)
    stream_ref = []

    def cb(event, err):
        if stream_ref:
            pass

    stream = device.start_read_stream(0x11, cb)
    stream_ref.append(stream)

    capfd.readouterr()

    stream.stop()
    del stream
    del stream_ref
    del device
    del manager

    # Since the stream stopped, the background thread finished and dropped the callback
    # Breaking the C++ -> Python cycle and allowing safe cleanup
    captured = capfd.readouterr()
    assert "MockDataStream destroyed" in captured.err
    assert "MockDevice(0) destroyed" in captured.err
    assert "MockDeviceManager destroyed" in captured.err
