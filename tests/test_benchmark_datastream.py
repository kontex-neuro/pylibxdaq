import json
import time
import statistics
import threading
import pytest
import logging
from pylibxdaq import pyxdaq_device
from .test_lifetime_manager import mock_manager_path


@pytest.mark.parametrize("rate_ev_s", [10, 100, 1000, 10_000, 100_000, 200_000, 400_000])
def test_datastream_benchmark(mock_manager_path, rate_ev_s):
    manager = pyxdaq_device.get_device_manager(mock_manager_path)

    # Wait for 0.1 seconds worth of events to keep tests snappy, minimum 5 for statistical significance
    target_events = max(5, int(rate_ev_s * 0.1))

    config = json.dumps({"id": 0, "event_rate": float(rate_ev_s), "max_events": target_events * 4})
    device = manager.create_device(config)

    rates = []
    min_runs = 5
    max_runs = 20
    cv_threshold = 0.02  # Expect high stability (2% variance) from this data stream

    for _ in range(max_runs):
        events_received = 0
        done = threading.Event()

        def cb(event, err):
            nonlocal events_received
            if event is not None:
                events_received += 1
                if events_received >= target_events:
                    done.set()
            else:
                done.set()  # stream stopped or errored before target reached

        start_time = time.perf_counter()

        with device.start_read_stream(0x11, cb) as stream:
            try:
                done.wait()
            finally:
                stream.stop()

        elapsed = time.perf_counter() - start_time

        # Guard in case thread scheduling caused elapsed to be anomalously near 0
        if elapsed > 0:
            rates.append(events_received / elapsed)

        # Check if we have gathered enough samples to estimate variance
        if len(rates) >= min_runs:
            mean = statistics.mean(rates)
            std = statistics.stdev(rates)
            # Break early if the relative standard deviation (CV) is consistently below our threshold
            if (std / mean) < cv_threshold:
                break

    mean_rate = statistics.mean(rates)
    std_rate = statistics.stdev(rates) if len(rates) > 1 else 0.0

    logger = logging.getLogger("benchmark")
    logger.info(
        f"\n[Benchmark] Target: {rate_ev_s:6d} ev/s "
        f"| Runs: {len(rates):2d} "
        f"| Rate: {mean_rate:10.1f} ± {std_rate:6.1f} ev/s "
        f"({(std_rate/mean_rate)*100 if mean_rate else 0:.2f}%)"
    )

    assert mean_rate > 0
