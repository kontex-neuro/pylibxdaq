import sys
import threading

sys.path.insert(0, sys.argv[1])
import _test_np_native

control = _test_np_native.Control()
entered = threading.Event()
completed = threading.Event()
worker = threading.Thread(target=lambda: control.hold(entered.set))
worker.start()
assert entered.wait(5)
control.hold(completed.set)
worker.join(5)
assert not worker.is_alive()
assert completed.is_set()

generation = control.redetect()
control.validate(generation)
replacement = control.redetect()
try:
    control.validate(generation)
except RuntimeError as error:
    assert "no longer valid" in str(error)
else:
    raise AssertionError("Old generation remained valid")
control.validate(replacement)
assert _test_np_native.queue_reset()
assert _test_np_native.retired_probe_stop()
try:
    _test_np_native.closed_io_stop()
except RuntimeError as error:
    assert "IO stream is closed" in str(error)
else:
    raise AssertionError("Stop suppressed a non-retirement error")