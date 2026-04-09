import gc
import json
import pytest
import time
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
        assert not stream.stopped

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
    assert not stream.stopped

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
        assert not stream.stopped

    captured = capfd.readouterr()
    assert "MockDataStream destroyed" in captured.err
    assert stream.stopped


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
    assert not stream.stopped

    time.sleep(0.05)
    stream.stop()
    del stream
    del device
    del manager

    assert len(events) > 0


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
