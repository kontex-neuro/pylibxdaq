#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <spdlog/spdlog.h>
#include <xdaq/data_streams.h>
#include <xdaq/device.h>
#include <xdaq/device_manager.h>

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>

namespace nb = nanobind;
using namespace nb::literals;

namespace pyxdaq
{

struct DeviceHandle {
    std::shared_ptr<xdaq::Device> device;

    explicit DeviceHandle(std::shared_ptr<xdaq::Device> d) : device(std::move(d)) {}

    void check() const
    {
        if (!device) throw nb::value_error("Device is already closed");
    }

    void close() noexcept { device.reset(); }

    bool is_closed() const noexcept { return !device; }
};

struct ManagedBuffer {
    std::unique_ptr<unsigned char[], void (*)(unsigned char *)> data;
    std::size_t size;
    std::shared_ptr<xdaq::Device> device;  // Keeps the DLL alive until the buffer is safely GC'd
};

struct DataView {
    std::span<unsigned char> data;
};

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

}  // namespace pyxdaq

NB_MODULE(pyxdaq_device, m)
{
    m.doc() = "Python binding for XDAQ Device";

    nb::enum_<xdaq::Device::return_t>(m, "ReturnCode")
        .value("Success", xdaq::Device::return_t::Success)
        .value("Failure", xdaq::Device::return_t::Failure);

    nb::class_<pyxdaq::DataStreamHandle>(m, "DataStream")
        .def(
            "__repr__",
            [](const pyxdaq::DataStreamHandle &h) {
                if (h.is_stopped()) return std::string("<DataStream stopped>");
                return fmt::format(
                    "<DataStream stream={}>", static_cast<const void *>(h.stream.get())
                );
            }
        )
        .def_prop_ro("stopped", &pyxdaq::DataStreamHandle::is_stopped)
        .def("stop", &pyxdaq::DataStreamHandle::stop, nb::call_guard<nb::gil_scoped_release>())
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


    nb::class_<pyxdaq::ManagedBuffer>(m, "ManagedBuffer")
        .def_prop_ro("size", [](const pyxdaq::ManagedBuffer &b) { return b.size; })
        .def_prop_ro("numpy", [](nb::object obj) {
            auto &b = nb::cast<pyxdaq::ManagedBuffer &>(obj);
            size_t shape[1] = {b.size};
            // By passing `obj`, nanobind increments the refcount of the Python ManagedBuffer object
            // ensuring the C++ struct (and its shared_ptr to the DLL) stays alive
            // as long as the user holds the NumPy array in Python!
            return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(b.data.get(), 1, shape, obj);
        });


    nb::class_<pyxdaq::DataView>(m, "DataView")
        .def_prop_ro("size", [](const pyxdaq::DataView &b) { return b.data.size(); })
        .def_prop_ro("numpy", [](nb::object obj) {
            auto &b = nb::cast<pyxdaq::DataView &>(obj);
            size_t shape[1] = {b.data.size()};
            // Same here, pins the DataView python object while the numpy array exists.
            return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(b.data.data(), 1, shape, obj);
        });

    nb::class_<pyxdaq::DeviceHandle>(m, "Device")
        .def(
            "__repr__",
            [](const pyxdaq::DeviceHandle &h) {
                if (h.is_closed()) return std::string("<Device closed>");
                return fmt::format("<Device device={}>", static_cast<const void *>(h.device.get()));
            }
        )
        .def_prop_ro("closed", &pyxdaq::DeviceHandle::is_closed)

        .def("close", &pyxdaq::DeviceHandle::close, nb::call_guard<nb::gil_scoped_release>())

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
               std::optional<nb::object>) { h.close(); },
            nb::call_guard<nb::gil_scoped_release>()
        )

        .def(
            "set_register_sync",
            [](pyxdaq::DeviceHandle &h,
               xdaq::Device::addr_t addr,
               xdaq::Device::value_t value,
               xdaq::Device::value_t mask) {
                h.check();
                return h.device->set_register_sync(addr, value, mask);
            },
            "addr"_a,
            "value"_a,
            "mask"_a = xdaq::Device::value_mask
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
            "mask"_a = xdaq::Device::value_mask
        )
        .def(
            "send_registers",
            [](pyxdaq::DeviceHandle &h) {
                h.check();
                return h.device->send_registers();
            }
        )
        .def(
            "get_register_sync",
            [](pyxdaq::DeviceHandle &h, xdaq::Device::addr_t addr) {
                h.check();
                return h.device->get_register_sync(addr);
            },
            "addr"_a
        )
        .def(
            "get_register",
            [](pyxdaq::DeviceHandle &h, xdaq::Device::addr_t addr) {
                h.check();
                return h.device->get_register(addr);
            },
            "addr"_a
        )
        .def(
            "read_registers",
            [](pyxdaq::DeviceHandle &h) {
                h.check();
                return h.device->read_registers();
            }
        )
        .def(
            "trigger",
            [](pyxdaq::DeviceHandle &h, xdaq::Device::addr_t addr, int bit) {
                h.check();
                return h.device->trigger(addr, bit);
            },
            "addr"_a,
            "bit"_a
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
            "data"_a
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
            "buf"_a
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
                auto stream = h.device->start_read_stream(
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
            "max_queue_elements"_a = 4096
        )
        .def(
            "start_aligned_read_stream",
            [](pyxdaq::DeviceHandle &h,
               std::uint32_t addr,
               std::size_t alignment,
               std::function<void(std::optional<pyxdaq::DataView>, std::optional<std::string>)>
                   callback,
               std::size_t chunk_size,
               std::size_t max_queue_elements) -> std::optional<pyxdaq::DataStreamHandle> {
                h.check();
                auto stream = h.device->start_read_stream(
                    addr,
                    xdaq::DataStream::queue(
                        xdaq::DataStream::aligned_read_stream(
                            [callback = std::move(callback), dev = h.device](auto &&event) {
                                nb::gil_scoped_acquire gil;  // <--- Must hold GIL when background
                                                             // thread calls Python callback!
                                std::visit(
                                    [&callback, dev](auto &&event) {
                                        using T = std::decay_t<decltype(event)>;
                                        using namespace xdaq::DataStream;
                                        if constexpr (std::is_same_v<T, Events::DataView>) {
                                            try {
                                                callback(
                                                    pyxdaq::DataView{.data = event.data},
                                                    std::nullopt
                                                );
                                            } catch (const std::exception &e) {
                                                spdlog::error("Error in callback: {}", e.what());
                                            }
                                        } else if constexpr (std::is_same_v<T, Events::OwnedData>) {
                                            callback(
                                                std::nullopt, "Unsupported data type: OwnedData"
                                            );
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
                            alignment
                        ),
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
            "max_queue_elements"_a = 4096
        )
        .def(
            "get_status",
            [](pyxdaq::DeviceHandle &h) {
                h.check();
                auto result = h.device->get_status();
                if (!result) throw std::runtime_error(result.error());
                return *result;
            }
        )
        .def("get_info", [](pyxdaq::DeviceHandle &h) {
            h.check();
            auto result = h.device->get_info();
            if (!result) throw std::runtime_error(result.error());
            return *result;
        });

    nb::class_<xdaq::DeviceManager>(m, "DeviceManager")
        .def(
            "__repr__",
            [](const xdaq::DeviceManager &d) {
                return fmt::format("<DeviceManager at {}>", static_cast<const void *>(&d));
            }
        )
        .def("info", [](const xdaq::DeviceManager &d) { return d.info(); })
        .def("location", [](const xdaq::DeviceManager &d) { return d.location(); })
        .def("list_devices", [](const xdaq::DeviceManager &d) { return d.list_devices(); })
        .def(
            "get_device_options",
            [](const xdaq::DeviceManager &d) { return d.get_device_options(); }
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
            }
        );

    m.def("get_device_manager", &xdaq::get_device_manager);
}