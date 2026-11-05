#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <spdlog/spdlog.h>
#include <xdaq/data_streams.h>
#include <xdaq/device.h>
#include <xdaq/device_manager.h>
#include <xdaq/device_scanner.h>

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>

#include "aligned_read_stream.hpp"
#include "pyxdaq_handles.hpp"
#include "queued_read_stream.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace pyxdaq
{

struct ManagedBuffer {
    std::unique_ptr<unsigned char[], void (*)(unsigned char *)> data;
    std::size_t size;
    std::shared_ptr<xdaq::Device> device;  // Keeps the DLL alive until the buffer is safely GC'd
};

struct DataView {
    std::span<unsigned char> data;
};

/// The reader's batch while streaming (Device::DataStream::set_batch): reads of at most
/// max_bytes; with fewer waiting, wait up to max_wait_us for more (0: take what is waiting).
inline void set_stream_batch(xdaq::Device::DataStream *stream, std::size_t max_bytes, std::int64_t max_wait_us)
{
    if (!stream) throw nb::value_error("The stream is stopped (or detached: no device reader)");
    bool changed = false;
    {
        nb::gil_scoped_release release;
        changed = stream->set_batch(max_bytes, std::chrono::microseconds(max_wait_us));
    }
    if (!changed) throw nb::value_error("This device's stream has no adjustable batch");
}

struct DataStreamHandle {
    std::shared_ptr<xdaq::Device::DataStream> stream;

    explicit DataStreamHandle(std::shared_ptr<xdaq::Device::DataStream> s) : stream(std::move(s)) {}

    ~DataStreamHandle()
    {
        nb::gil_scoped_release release;
        stream.reset();
    }

    void check() const
    {
        if (!stream) throw nb::value_error("DataStream is already stopped");
    }

    void stop() noexcept
    {
        if (stream) {
            stream->stop();
        }
    }

    void wait_stop()
    {
        if (stream) {
            stream->wait_stop();
            stream.reset();
        }
    }

    bool is_stopped() const noexcept { return !stream; }
};

/// A read stream whose buffers go into a native queue (QueuedReadStream) that Python drains in
/// batches: no Python callback per buffer. Without a device stream (``QueuedStream.detached``) it is
/// fed by ``inject*``, for tests.
struct QueuedStreamHandle {
    std::shared_ptr<QueuedReadStream> queue;
    std::unique_ptr<xdaq::Device::DataStream> stream;
    std::shared_ptr<xdaq::Device> device;  // keeps the plugin (and its buffers' deleters) loaded

    QueuedStreamHandle(
        std::shared_ptr<QueuedReadStream> q, std::unique_ptr<xdaq::Device::DataStream> s,
        std::shared_ptr<xdaq::Device> d
    )
        : queue(std::move(q)), stream(std::move(s)), device(std::move(d))
    {
    }
    QueuedStreamHandle(QueuedStreamHandle &&) = default;
    QueuedStreamHandle &operator=(QueuedStreamHandle &&) = default;

    void stop() noexcept
    {
        if (stream) stream->stop();
    }

    // Stop the reader thread, then drop what is still queued, then release the device.
    void close()
    {
        if (stream) {
            stream->stop();
            stream->wait_stop();
            stream.reset();
        }
        if (queue) queue->clear();
        device.reset();
    }

    ~QueuedStreamHandle()
    {
        nb::gil_scoped_release release;
        close();
    }
};

}  // namespace pyxdaq

NB_MODULE(pyxdaq_device, m)
{
    m.doc() = "Python binding for XDAQ Device";

    nb::enum_<xdaq::Device::return_t>(m, "ReturnCode")
        .value("Success", xdaq::Device::return_t::Success)
        .value("Failure", xdaq::Device::return_t::Failure);

    nb::class_<pyxdaq::DataStreamHandle>(
        m,
        "DataStream",
        "Handle to an active data stream. Must be used as a context manager to ensure clean "
        "shutdown."
    )
        .def(
            "__repr__",
            [](const pyxdaq::DataStreamHandle &h) {
                if (h.is_stopped()) return std::string("<DataStream stopped>");
                return fmt::format(
                    "<DataStream stream={}>", static_cast<const void *>(h.stream.get())
                );
            }
        )
        .def(
            "set_batch",
            [](pyxdaq::DataStreamHandle &h, std::size_t max_bytes, std::int64_t max_wait_us) {
                pyxdaq::set_stream_batch(h.stream.get(), max_bytes, max_wait_us);
            },
            "max_bytes"_a, "max_wait_us"_a,
            "Change the reader's batch while streaming: reads of at most max_bytes (a multiple of 4096); with "
            "less waiting, wait up to max_wait_us for more, then take what is there (0: take whatever is "
            "waiting at once). Raises ValueError if the device's stream has none."
        )
        .def(
            "stop",
            &pyxdaq::DataStreamHandle::stop,
            nb::call_guard<nb::gil_scoped_release>(),
            "Signal the stream to stop. Non-blocking — use the context manager for clean shutdown."
        )
        .def(
            "__enter__",
            [](pyxdaq::DataStreamHandle &h) -> pyxdaq::DataStreamHandle & {
                h.check();
                return h;
            },
            nb::rv_policy::reference
        )
        .def(
            "__exit__",
            [](pyxdaq::DataStreamHandle &h,
               std::optional<nb::object>,
               std::optional<nb::object>,
               std::optional<nb::object>) {
                h.stop();
                h.wait_stop();
            },
            nb::call_guard<nb::gil_scoped_release>()
        );


    nb::class_<pyxdaq::ManagedBuffer>(
        m, "ManagedBuffer", "Owned data buffer delivered by start_read_stream."
    )
        .def_prop_ro(
            "size",
            [](const pyxdaq::ManagedBuffer &b) { return b.size; },
            "Number of bytes in the buffer."
        )
        .def_prop_ro(
            "numpy",
            [](const pyxdaq::ManagedBuffer &b) {
                // Safe copy: allocates a new buffer owned by the numpy array,
                // so it remains valid even after the ManagedBuffer is destroyed.
                auto *buf = new unsigned char[b.size];
                std::copy(b.data.get(), b.data.get() + b.size, buf);
                nb::capsule owner(buf, [](void *p) noexcept {
                    delete[] static_cast<unsigned char *>(p);
                });
                size_t shape[1] = {b.size};
                return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(buf, 1, shape, owner);
            },
            nb::rv_policy::automatic,
            "Copy the buffer into a new numpy array. Safe to hold past the lifetime of this object."
        )
        .def_prop_ro(
            "numpy_view",
            [](const pyxdaq::ManagedBuffer &b) {
                size_t shape[1] = {b.size};
                return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(b.data.get(), 1, shape);
            },
            nb::rv_policy::reference,
            "Zero-copy view into the buffer. Only safe to use while this object is alive."
        );


    nb::class_<pyxdaq::DataView>(
        m,
        "DataView",
        "Aligned data view delivered by start_aligned_read_stream. Only valid during the callback."
    )
        .def_prop_ro(
            "size",
            [](const pyxdaq::DataView &b) { return b.data.size(); },
            "Number of bytes in the view."
        )
        .def_prop_ro(
            "numpy",
            [](const pyxdaq::DataView &b) {
                // Safe copy: allocates a new buffer owned by the numpy array,
                // so it remains valid even after the DataView is destroyed.
                auto *buf = new unsigned char[b.data.size()];
                std::copy(b.data.begin(), b.data.end(), buf);
                nb::capsule owner(buf, [](void *p) noexcept {
                    delete[] static_cast<unsigned char *>(p);
                });
                size_t shape[1] = {b.data.size()};
                return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(buf, 1, shape, owner);
            },
            nb::rv_policy::automatic,
            "Copy the view into a new numpy array. Safe to hold past the callback."
        )
        .def_prop_ro(
            "numpy_view",
            [](nb::object obj) {
                auto &b = nb::cast<pyxdaq::DataView &>(obj);
                size_t shape[1] = {b.data.size()};
                // Zero-copy view. UNSAFE outside the callback — the span is only valid
                // while the C++ DataView object is alive (i.e. during the callback).
                return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(b.data.data(), 1, shape, obj);
            },
            "Zero-copy view into the data. Only safe to use within the callback; do not store."
        );

    nb::class_<pyxdaq::QueuedStreamHandle>(
        m,
        "QueuedStream",
        "A read stream into a native queue, drained in batches by read(): the device's reader thread "
        "queues each buffer without Python or the GIL. Use as a context manager (or close()) for a "
        "clean stop."
    )
        .def(
            "read",
            [](pyxdaq::QueuedStreamHandle &h, std::size_t max_bytes, std::size_t min_bytes,
               double timeout) -> nb::object {
                std::optional<pyxdaq::QueuedReadStream::Batch> batch;
                {
                    nb::gil_scoped_release release;
                    batch = h.queue->take(max_bytes, min_bytes, timeout);
                }
                if (!batch) return nb::none();
                auto *raw = batch->data.release();
                nb::capsule owner(raw, [](void *p) noexcept {
                    delete[] static_cast<unsigned char *>(p);
                });
                size_t shape[1] = {batch->size};
                auto array = nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(raw, 1, shape, owner);
                return nb::make_tuple(nb::cast(array), batch->received_at, batch->buffers);
            },
            "max_bytes"_a = std::size_t{1} << 20,
            "min_bytes"_a = 0,
            "timeout"_a = 0.0,
            "Take the queued buffers, in order and whole, up to max_bytes (at least one), as one "
            "uint8 array: (array, received_at, buffers), or None when nothing is queued. Waits up "
            "to timeout seconds (GIL released) for min_bytes (at least one buffer); past the "
            "timeout it takes what is there. received_at is the oldest buffer's hand-over time "
            "(time.perf_counter's clock)."
        )
        .def_prop_ro(
            "queued_bytes",
            [](const pyxdaq::QueuedStreamHandle &h) { return h.queue->queued(); },
            "Bytes queued and not yet read."
        )
        .def_prop_ro(
            "queued_buffers",
            [](const pyxdaq::QueuedStreamHandle &h) { return h.queue->queued_count(); },
            "Buffers queued and not yet read."
        )
        .def_prop_ro(
            "error",
            [](const pyxdaq::QueuedStreamHandle &h) { return h.queue->error_message(); },
            "The stream's first error (the device's, or the queue's overflow), or None. Sticky."
        )
        .def_prop_ro(
            "stopped",
            [](const pyxdaq::QueuedStreamHandle &h) { return h.queue->is_stopped(); },
            "True once the device's stream has delivered its final Stop."
        )
        .def(
            "stats",
            [](const pyxdaq::QueuedStreamHandle &h) {
                const auto s = h.queue->snapshot();
                nb::dict d;
                d["buffers"] = s.buffers;
                d["bytes"] = s.bytes;
                d["max_buffer_bytes"] = s.max_buffer_bytes;
                d["taken_bytes"] = s.taken_bytes;
                d["batches"] = s.batches;
                d["max_queued_bytes"] = s.max_queued_bytes;
                d["max_queued_buffers"] = s.max_queued_buffers;
                d["max_arrival_gap"] = s.max_arrival_gap;
                d["arrival_gaps"] = s.arrival_gaps;
                d["arrival_gap_threshold"] = h.queue->arrival_gap_threshold;
                return d;
            },
            "Counters: buffers and bytes handed over, taken bytes and batches, the queue's high-water "
            "marks, and the pacing of the hand-over (longest gap between two buffers, gaps over "
            "arrival_gap_threshold seconds)."
        )
        .def(
            "set_batch",
            [](pyxdaq::QueuedStreamHandle &h, std::size_t max_bytes, std::int64_t max_wait_us) {
                pyxdaq::set_stream_batch(h.stream.get(), max_bytes, max_wait_us);
            },
            "max_bytes"_a, "max_wait_us"_a,
            "Change the reader's batch while streaming: reads of at most max_bytes (a multiple of 4096); with "
            "less waiting, wait up to max_wait_us for more, then take what is there (0: take whatever is "
            "waiting at once). Raises ValueError if the device's stream has none."
        )
        .def(
            "stop",
            &pyxdaq::QueuedStreamHandle::stop,
            nb::call_guard<nb::gil_scoped_release>(),
            "Signal the reader to stop (non-blocking); its final Stop sets `stopped`."
        )
        .def(
            "close",
            &pyxdaq::QueuedStreamHandle::close,
            nb::call_guard<nb::gil_scoped_release>(),
            "Stop the reader, wait for it, drop what is still queued and release the device."
        )
        .def(
            "__enter__",
            [](pyxdaq::QueuedStreamHandle &h) -> pyxdaq::QueuedStreamHandle & { return h; },
            nb::rv_policy::reference
        )
        .def(
            "__exit__",
            [](pyxdaq::QueuedStreamHandle &h,
               std::optional<nb::object>,
               std::optional<nb::object>,
               std::optional<nb::object>) { h.close(); },
            nb::call_guard<nb::gil_scoped_release>()
        )
        .def_static(
            "detached",
            [](std::size_t max_queue_bytes, double arrival_gap_threshold) {
                return pyxdaq::QueuedStreamHandle{
                    std::make_shared<pyxdaq::QueuedReadStream>(max_queue_bytes, arrival_gap_threshold),
                    nullptr,
                    nullptr};
            },
            "max_queue_bytes"_a = std::size_t{64} << 20,
            "arrival_gap_threshold"_a = 0.005,
            "A queue without a device, fed by inject/inject_error/inject_stop (tests)."
        )
        .def(
            "inject",
            [](pyxdaq::QueuedStreamHandle &h, nb::bytes data) {
                xdaq::DataStream::Events::OwnedData owned{
                    .buffer = {new unsigned char[data.size()], [](unsigned char *p) { delete[] p; }},
                    .length = data.size()};
                std::memcpy(owned.data(), data.c_str(), data.size());
                h.queue->push(std::move(owned));
            },
            "data"_a,
            "Queue one buffer as the device's reader would (tests)."
        )
        .def(
            "inject_error",
            [](pyxdaq::QueuedStreamHandle &h, std::string message) { h.queue->push_error(message); },
            "message"_a,
            "Deliver an error event (tests)."
        )
        .def(
            "inject_stop",
            [](pyxdaq::QueuedStreamHandle &h) { h.queue->push_stop(); },
            "Deliver the stop event (tests)."
        );

    nb::class_<pyxdaq::DeviceHandle>(
        m,
        "Device",
        "Handle to an open XDAQ device. Use as a context manager or call close() explicitly."
    )
        .def(
            "__repr__",
            [](const pyxdaq::DeviceHandle &h) {
                if (h.is_closed()) return std::string("<Device closed>");
                return fmt::format("<Device device={}>", static_cast<const void *>(h.device.get()));
            }
        )
        .def_prop_ro(
            "closed", &pyxdaq::DeviceHandle::is_closed, "True if the device has been closed."
        )

        .def(
            "close",
            &pyxdaq::DeviceHandle::close,
            "Close this device handle. The device stays alive until all associated streams and "
            "buffers are also released."
        )

        .def(
            "__enter__",
            [](pyxdaq::DeviceHandle &h) -> pyxdaq::DeviceHandle & {
                h.check();
                return h;
            },
            nb::rv_policy::reference
        )
        .def(
            "__exit__",
            [](pyxdaq::DeviceHandle &h,
               std::optional<nb::object>,
               std::optional<nb::object>,
                    std::optional<nb::object>) { h.close(); }
        )

        .def(
            "set_register_sync",
            [](pyxdaq::DeviceHandle &h,
               xdaq::Device::addr_t addr,
               xdaq::Device::value_t value,
               xdaq::Device::value_t mask) {
                h.check();
                auto device = h.device;
                nb::gil_scoped_release release;
                return device->set_register_sync(addr, value, mask);
            },
            "addr"_a,
            "value"_a,
            "mask"_a = xdaq::Device::value_mask,
            "Write a register value synchronously."
        )
        .def(
            "set_register",
            [](pyxdaq::DeviceHandle &h,
               xdaq::Device::addr_t addr,
               xdaq::Device::value_t value,
               xdaq::Device::value_t mask) {
                h.check();
                return h.device->set_register(addr, value, mask);
            },
            "addr"_a,
            "value"_a,
            "mask"_a = xdaq::Device::value_mask,
            "Set a register value."
        )
        .def(
            "send_registers",
            [](pyxdaq::DeviceHandle &h) {
                h.check();
                return h.device->send_registers();
            },
            "Ensure pending register writes are reflected on the device."
        )
        .def(
            "get_register_sync",
            [](pyxdaq::DeviceHandle &h, xdaq::Device::addr_t addr) {
                h.check();
                return h.device->get_register_sync(addr);
            },
            "addr"_a,
            "Read the current register value directly from the device."
        )
        .def(
            "get_register",
            [](pyxdaq::DeviceHandle &h, xdaq::Device::addr_t addr) {
                h.check();
                return h.device->get_register(addr);
            },
            "addr"_a,
            "Return a register value."
        )
        .def(
            "read_registers",
            [](pyxdaq::DeviceHandle &h) {
                h.check();
                return h.device->read_registers();
            },
            "Ensure register values are up to date from the device."
        )
        .def(
            "trigger",
            [](pyxdaq::DeviceHandle &h, xdaq::Device::addr_t addr, int bit) {
                h.check();
                return h.device->trigger(addr, bit);
            },
            "addr"_a,
            "bit"_a,
            "Set a bit in a register to trigger a device action."
        )
        .def(
            "write",
            [](pyxdaq::DeviceHandle &h,
               std::uint32_t addr,
               nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig>
                   arr) {
                h.check();
                return h.device->write(
                    addr, arr.size(), reinterpret_cast<unsigned char *>(arr.data())
                );
            },
            "addr"_a,
            "data"_a,
            "Write raw bytes to the device at the given address."
        )
        .def(
            "read",
            [](pyxdaq::DeviceHandle &h,
               std::uint32_t addr,
               nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig>
                   arr) {
                h.check();
                return h.device->read(
                    addr, arr.size(), reinterpret_cast<unsigned char *>(arr.data())
                );
            },
            "addr"_a,
            "buf"_a,
            "Read raw bytes from the device into buf at the given address."
        )
        .def(
            "start_read_stream",
            [](pyxdaq::DeviceHandle &h,
               std::uint32_t addr,
               std::function<void(std::optional<pyxdaq::ManagedBuffer>, std::optional<std::string>)>
                   callback,
               std::size_t chunk_size,
               std::size_t max_queue_elements) -> std::optional<pyxdaq::DataStreamHandle> {
                h.check();
                auto stream = [&] {
                    nb::gil_scoped_release release;
                    return h.device->start_read_stream(
                    addr,
                    xdaq::DataStream::queue(
                        [callback = std::move(callback), dev = h.device](auto &&event) {
                            nb::gil_scoped_acquire gil;  // <--- Must hold GIL when background
                                                         // thread calls Python callback!
                            std::visit(
                                [&callback, dev](auto &&event) {
                                    using T = std::decay_t<decltype(event)>;
                                    using namespace xdaq::DataStream;
                                    if constexpr (std::is_same_v<T, Events::DataView>) {
                                        callback(std::nullopt, "Unsupported data type: DataView");
                                    } else if constexpr (std::is_same_v<T, Events::OwnedData>) {
                                        try {
                                            callback(
                                                pyxdaq::ManagedBuffer{
                                                    .data = std::move(event.buffer),
                                                    .size = event.length,
                                                    .device = dev  // <--- Safely capture the device
                                                                   // shared_ptr from the Handle!
                                                },
                                                std::nullopt
                                            );
                                        } catch (const std::exception &e) {
                                            spdlog::error("Error in callback: {}", e.what());
                                        }
                                    } else if constexpr (std::is_same_v<T, Events::Stop>) {
                                        try {
                                            callback(std::nullopt, std::nullopt);
                                        } catch (const std::exception &e) {
                                        }
                                    } else if constexpr (std::is_same_v<T, Events::Error>) {
                                        try {
                                            callback(std::nullopt, event.error);
                                        } catch (const std::exception &e) {
                                        }
                                    } else {
                                        static_assert(
                                            xdaq::always_false_v<T>, "non-exhaustive visitor"
                                        );
                                    }
                                },
                                std::move(event)
                            );
                        },
                        64,
                        max_queue_elements,
                        std::chrono::nanoseconds{0}
                    ),
                    chunk_size
                );
                }();
                if (stream == nullptr) return std::nullopt;
                return pyxdaq::DataStreamHandle{{
                    stream.release(),
                    [dev = h.device](xdaq::Device::DataStream *p) { delete p; },
                }};
            },
            "addr"_a,
            "callback"_a,
            nb::kw_only(),
            "chunk_size"_a = 0,
            "max_queue_elements"_a = 4096,
            "Start a raw read stream. callback(event, error) is called from a background thread "
            "with ManagedBuffer events."
        )
        .def(
            "start_queued_read_stream",
            [](pyxdaq::DeviceHandle &h,
               std::uint32_t addr,
               std::size_t chunk_size,
               std::size_t max_queue_bytes,
               double arrival_gap_threshold) -> std::optional<pyxdaq::QueuedStreamHandle> {
                h.check();
                auto queue =
                    std::make_shared<pyxdaq::QueuedReadStream>(max_queue_bytes, arrival_gap_threshold);
                std::unique_ptr<xdaq::Device::DataStream> stream;
                {
                    nb::gil_scoped_release release;
                    stream = h.device->start_read_stream(
                        addr, pyxdaq::QueuedReadStream::sink(queue), chunk_size
                    );
                }
                if (stream == nullptr) return std::nullopt;
                return pyxdaq::QueuedStreamHandle{std::move(queue), std::move(stream), h.device};
            },
            "addr"_a,
            nb::kw_only(),
            "chunk_size"_a = 0,
            "max_queue_bytes"_a = std::size_t{64} << 20,
            "arrival_gap_threshold"_a = 0.005,
            "Start a raw read stream into a native queue (QueuedStream): the reader thread queues "
            "each buffer without calling Python; read() takes batches. A buffer that would pass "
            "max_queue_bytes sets the error 'RHX host stream queue overflow' and is dropped."
        )
        .def(
            "start_aligned_read_stream",
            [](pyxdaq::DeviceHandle &h,
               std::uint32_t addr,
               std::size_t alignment,
               std::function<void(std::optional<pyxdaq::DataView>, std::optional<std::string>)>
                   callback,
               std::size_t chunk_size,
               std::size_t max_queue_elements,
               bool merged) -> std::optional<pyxdaq::DataStreamHandle> {
                h.check();
                // The sink is the same either way; only how chunks are cut differs.
                xdaq::DataStream::receive_callback sink{[callback = std::move(callback),
                                                         dev = h.device](auto &&event) {
                    nb::gil_scoped_acquire gil;  // <--- Must hold GIL when background
                                                 // thread calls Python callback!
                    std::visit(
                        [&callback, dev](auto &&event) {
                            using T = std::decay_t<decltype(event)>;
                            using namespace xdaq::DataStream;
                            if constexpr (std::is_same_v<T, Events::DataView>) {
                                try {
                                    callback(pyxdaq::DataView{.data = event.data}, std::nullopt);
                                } catch (const std::exception &e) {
                                    spdlog::error("Error in callback: {}", e.what());
                                }
                            } else if constexpr (std::is_same_v<T, Events::OwnedData>) {
                                callback(std::nullopt, "Unsupported data type: OwnedData");
                            } else if constexpr (std::is_same_v<T, Events::Stop>) {
                                try {
                                    callback(std::nullopt, std::nullopt);
                                } catch (const std::exception &e) {
                                }
                            } else if constexpr (std::is_same_v<T, Events::Error>) {
                                try {
                                    callback(std::nullopt, event.error);
                                } catch (const std::exception &e) {
                                }
                            } else {
                                static_assert(xdaq::always_false_v<T>, "non-exhaustive visitor");
                            }
                        },
                        std::move(event)
                    );
                }};
                auto stream = h.device->start_read_stream(
                    addr,
                    xdaq::DataStream::queue(
                        pyxdaq::aligned_read_stream(std::move(sink), alignment, merged),
                        64,
                        max_queue_elements,
                        std::chrono::nanoseconds{0}
                    ),
                    chunk_size
                );
                if (stream == nullptr) return std::nullopt;
                return pyxdaq::DataStreamHandle{{
                    stream.release(),
                    [dev = h.device](xdaq::Device::DataStream *p) { delete p; },
                }};
            },
            "addr"_a,
            "alignment"_a,
            "callback"_a,
            nb::kw_only(),
            "chunk_size"_a = 0,
            "max_queue_elements"_a = 4096,
            "merged"_a = false,
            "Start an alignment-aware read stream. callback(event, error) is called from a "
            "background thread with DataView events aligned to the given boundary. Every "
            "view is a whole multiple of alignment, including the last: a trailing partial "
            "sample is dropped at stop rather than delivered short.\n\n"
            "With merged=True, a chunk that straddles a sample boundary is delivered as one "
            "event instead of two -- the spanning sample is made contiguous with the body in "
            "a scratch buffer, costing one memcpy of the chunk to halve the callback rate. "
            "Latency per sample is unchanged."
        )
        .def(
            "get_status",
            [](pyxdaq::DeviceHandle &h) {
                h.check();
                auto result = h.device->get_status();
                if (!result) throw std::runtime_error(result.error());
                return *result;
            },
            "Return the device status as a JSON string."
        )
        .def(
            "get_info",
            [](pyxdaq::DeviceHandle &h) {
                h.check();
                auto result = h.device->get_info();
                if (!result) throw std::runtime_error(result.error());
                return *result;
            },
            "Return device info as a JSON string."
        );

    nb::class_<xdaq::DeviceManager>(
        m,
        "DeviceManager",
        "Manages discovery and creation of XDAQ devices loaded from a shared library."
    )
        .def(
            "__repr__",
            [](const xdaq::DeviceManager &d) {
                return fmt::format("<DeviceManager at {}>", static_cast<const void *>(&d));
            }
        )
        .def(
            "info",
            [](const xdaq::DeviceManager &d) { return d.info(); },
            "Return manager info as a JSON string."
        )
        .def(
            "location",
            [](const xdaq::DeviceManager &d) { return d.location(); },
            "Return the file system path of the loaded shared library."
        )
        .def(
            "list_devices",
            [](const xdaq::DeviceManager &d) { return d.list_devices(); },
            "Return a JSON string listing available devices."
        )
        .def(
            "get_device_options",
            [](const xdaq::DeviceManager &d) { return d.get_device_options(); },
            "Return the JSON schema describing valid options for create_device()."
        )
        .def(
            "create_device",
            [](std::shared_ptr<xdaq::DeviceManager> d, const std::string &config) {
                auto dev = d->create_device(config);
                // keep dll alive until device destruction by capturing manager's shared_ptr
                return pyxdaq::DeviceHandle{{
                    dev.release(),
                    [d](xdaq::Device *p) { delete p; },
                }};
            },
            "config"_a,
            "Create and open a device using a JSON config string. Returns a Device handle."
        );

    m.def(
        "get_device_manager",
        &xdaq::get_device_manager,
        "path"_a,
        "Load a device manager shared library from the given path."
    );

    nb::class_<xdaq::DeviceFound>(m, "DeviceFound", "A device that is reachable and not in use.")
        .def_prop_ro(
            "device_manager_path",
            [](const xdaq::DeviceFound &d) { return std::filesystem::path(d.device_manager_path); }
        )
        .def_ro("device_config_json", &xdaq::DeviceFound::device_config_json)
        .def_ro("serial_number", &xdaq::DeviceFound::serial_number)
        .def_ro("api_version", &xdaq::DeviceFound::api_version)
        .def_ro("info_json", &xdaq::DeviceFound::info_json)
        .def_ro("status_json", &xdaq::DeviceFound::status_json)
        .def("__repr__", [](const xdaq::DeviceFound &d) {
            return fmt::format("<DeviceFound serial={} api={}>", d.serial_number, d.api_version);
        });

    nb::class_<xdaq::DeviceOccupied>(m, "DeviceOccupied", "A device held open by another process.")
        .def_prop_ro(
            "device_manager_path",
            [](const xdaq::DeviceOccupied &d) {
                return std::filesystem::path(d.device_manager_path);
            }
        )
        .def_ro("device_config_json", &xdaq::DeviceOccupied::device_config_json)
        .def("__repr__", [](const xdaq::DeviceOccupied &d) {
            return fmt::format("<DeviceOccupied manager={}>", d.device_manager_path);
        });

    nb::class_<xdaq::DeviceQueryFailed>(
        m, "DeviceQueryFailed", "A device that is visible but could not be queried."
    )
        .def_prop_ro(
            "device_manager_path",
            [](const xdaq::DeviceQueryFailed &d) {
                return std::filesystem::path(d.device_manager_path);
            }
        )
        .def_ro("device_config_json", &xdaq::DeviceQueryFailed::device_config_json)
        .def_ro("error", &xdaq::DeviceQueryFailed::error)
        .def("__repr__", [](const xdaq::DeviceQueryFailed &d) {
            return fmt::format("<DeviceQueryFailed error={}>", d.error);
        });

    m.def(
        "scan_devices",
        [](const std::vector<std::filesystem::path> &paths) {
            nb::gil_scoped_release release;
            return xdaq::scan_devices(paths);
        },
        "paths"_a,
        "Scan a list of device manager plugin paths. Returns one result per enumerated device."
    );

    m.def(
        "scan_devices_dir",
        [](const std::filesystem::path &dir) {
            nb::gil_scoped_release release;
            return xdaq::scan_devices_dir(dir);
        },
        "dir"_a,
        "Scan a directory for device manager plugins. Returns one result per enumerated device."
    );
}