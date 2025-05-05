#include <fmt/format.h>
#include <pybind11/functional.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <spdlog/spdlog.h>
#include <xdaq/data_streams.h>
#include <xdaq/device.h>
#include <xdaq/device_manager.h>

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>


namespace py = pybind11;

using std::string;

namespace pyxdaq
{
class PyDataStream : public xdaq::Device::DataStream
{
public:
    virtual void stop() override { PYBIND11_OVERLOAD(void, xdaq::Device::DataStream, stop); }
};

class PyDevice : public xdaq::Device
{
public:
    using Device = xdaq::Device;

    virtual return_t set_register_sync(
        addr_t addr, value_t value, value_t mask = value_mask
    ) noexcept override
    {
        PYBIND11_OVERLOAD(return_t, Device, set_register_sync, addr, value, mask);
    }
    virtual return_t set_register(
        addr_t addr, value_t value, value_t mask = value_mask
    ) noexcept override
    {
        PYBIND11_OVERLOAD_PURE(return_t, Device, set_register, addr, value, mask);
    }
    virtual return_t send_registers() noexcept override
    {
        PYBIND11_OVERLOAD_PURE(return_t, Device, send_registers);
    }
    virtual std::optional<value_t> get_register_sync(addr_t addr) noexcept override
    {
        PYBIND11_OVERLOAD(std::optional<value_t>, Device, get_register_sync, addr);
    }
    virtual value_t get_register(addr_t addr) noexcept override
    {
        PYBIND11_OVERLOAD_PURE(value_t, Device, get_register, addr);
    }
    virtual return_t read_registers() noexcept override
    {
        PYBIND11_OVERLOAD_PURE(return_t, Device, read_registers);
    }

    virtual return_t trigger(addr_t addr, int bit) noexcept override
    {
        PYBIND11_OVERLOAD_PURE(return_t, Device, trigger, addr, bit);
    }

    virtual std::size_t write(
        addr_t addr, std::size_t length, const unsigned char *data
    ) noexcept override
    {
        PYBIND11_OVERLOAD_PURE(std::size_t, Device, write, addr, length, data);
    }

    virtual std::size_t read(addr_t addr, std::size_t length, unsigned char *data) noexcept override
    {
        PYBIND11_OVERLOAD_PURE(std::size_t, Device, read, addr, length, data);
    }

    virtual std::optional<std::string> get_status() override
    {
        PYBIND11_OVERLOAD(std::optional<std::string>, Device, get_status);
    }

    virtual std::optional<std::string> get_info() override
    {
        PYBIND11_OVERLOAD(std::optional<std::string>, Device, get_info);
    }
};

}  // namespace pyxdaq

PYBIND11_MODULE(pyxdaq_device, m)
{
    m.doc() = "Python binding for XDAQ Device";

    py::enum_<xdaq::Device::return_t>(m, "ReturnCode")
        .value("Success", xdaq::Device::return_t::Success)
        .value("Failure", xdaq::Device::return_t::Failure);

    py::class_<
        xdaq::Device::DataStream,
        pyxdaq::PyDataStream,
        std::unique_ptr<xdaq::Device::DataStream>>(m, "DataStream")
        .def("stop", &xdaq::Device::DataStream::stop, py::call_guard<py::gil_scoped_release>())
        .def("__enter__", [](xdaq::Device::DataStream &s) { return &s; })
        .def(
            "__exit__",
            [](xdaq::Device::DataStream &s,
               const std::optional<pybind11::type> &exc_type,
               const std::optional<pybind11::object> &exc_value,
               const std::optional<pybind11::object> &traceback) { s.stop(); },
            py::call_guard<py::gil_scoped_release>()
        );

    struct ManagedBuffer {
        std::unique_ptr<unsigned char[], void (*)(unsigned char *)> data;
        std::size_t size;
    };

    struct DataView {
        std::span<unsigned char> data;
    };

    py::class_<ManagedBuffer, std::unique_ptr<ManagedBuffer>>(
        m, "ManagedBuffer", py::buffer_protocol()
    )
        .def_buffer([](ManagedBuffer &b) -> py::buffer_info {
            return py::buffer_info(
                b.data.get(),
                sizeof(unsigned char),
                py::format_descriptor<unsigned char>::format(),
                1,
                {b.size},
                {sizeof(unsigned char)}
            );
        });

    py::class_<DataView, std::unique_ptr<DataView>>(m, "DataView", py::buffer_protocol())
        .def_buffer([](DataView &b) -> py::buffer_info {
            return py::buffer_info(
                b.data.data(),
                sizeof(unsigned char),
                py::format_descriptor<unsigned char>::format(),
                1,
                {b.data.size()},
                {sizeof(unsigned char)}
            );
        });


    py::class_<
        xdaq::Device,
        pyxdaq::PyDevice,
        std::unique_ptr<xdaq::Device, xdaq::DeviceManager::device_deleter>>(m, "Device")
        .def(py::init<>())
        .def(
            "__repr__",
            [](const xdaq::Device &d) {
                return fmt::format("<pyxdaq_device.Device at {}>", static_cast<const void *>(&d));
            }
        )
        .def("set_register_sync", &xdaq::Device::set_register_sync)
        .def("set_register", &xdaq::Device::set_register)
        .def("send_registers", &xdaq::Device::send_registers)
        .def("get_register_sync", &xdaq::Device::get_register_sync)
        .def("get_register", &xdaq::Device::get_register)
        .def("read_registers", &xdaq::Device::read_registers)
        .def("trigger", &xdaq::Device::trigger)
        .def(
            "write",
            [](xdaq::Device &d, std::uint32_t addr, py::buffer b) {
                auto info = b.request();
                if (info.ndim != 1) throw std::runtime_error("ndim != 1");
                if (info.format != py::format_descriptor<std::uint8_t>::format())
                    throw std::runtime_error("Unsupported buffer format!");
                if (info.strides[0] != 1) throw std::runtime_error("strides[0] != 1");
                return d.write(addr, info.size, (unsigned char *) info.ptr);
            }
        )
        .def(
            "read",
            [](xdaq::Device &d, std::uint32_t addr, py::buffer b) {
                auto info = b.request();
                if (info.readonly) throw std::runtime_error("buffer is readonly");
                if (info.ndim != 1) throw std::runtime_error("ndim != 1");
                if (info.format != py::format_descriptor<std::uint8_t>::format())
                    throw std::runtime_error("Unsupported buffer format!");
                if (info.strides[0] != 1) throw std::runtime_error("strides[0] != 1");
                return d.read(addr, info.size, (unsigned char *) info.ptr);
            }
        )
        .def(
            "start_read_stream",
            [](xdaq::Device &d,
               std::uint32_t addr,
               std::function<void(std::optional<ManagedBuffer> b, std::optional<std::string> error)>
                   callback,
               std::size_t chunk_size = 0,
               std::size_t max_queue_elements =
                   4096) -> std::optional<std::unique_ptr<xdaq::Device::DataStream>> {
                return d.start_read_stream(
                    addr,
                    xdaq::DataStream::queue(
                        [callback = std::move(callback)](auto &&event) {
                            std::visit(
                                [&callback](auto &&event) {
                                    using T = std::decay_t<decltype(event)>;
                                    using namespace xdaq::DataStream;
                                    if constexpr (std::is_same_v<T, Events::DataView>) {
                                        callback(std::nullopt, "Unsupported data type: DataView");
                                    } else if constexpr (std::is_same_v<T, Events::OwnedData>) {
                                        try {
                                            callback(
                                                ManagedBuffer{
                                                    .data = std::move(event.buffer),
                                                    .size = event.length
                                                },
                                                std::nullopt
                                            );
                                        } catch (const std::exception &e) {
                                            spdlog::error("Error in callback: {}", e.what());
                                        }
                                    } else if constexpr (std::is_same_v<T, Events::Stop>) {
                                        callback(std::nullopt, std::nullopt);
                                    } else if constexpr (std::is_same_v<T, Events::Error>) {
                                        callback(std::nullopt, event.error);
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
            },
            py::arg("addr"),
            py::arg("callback"),
            py::kw_only(),
            py::arg("chunk_size") = 0,
            py::arg("max_queue_elements") = 4096,
            py::return_value_policy::move
        )
        .def(
            "start_aligned_read_stream",
            [](xdaq::Device &d,
               std::uint32_t addr,
               std::size_t alignment,
               std::function<void(std::optional<DataView> b, std::optional<std::string> error)>
                   callback,
               std::size_t chunk_size = 0,
               std::size_t max_queue_elements =
                   4096) -> std::optional<std::unique_ptr<xdaq::Device::DataStream>> {
                return d.start_read_stream(
                    addr,
                    xdaq::DataStream::queue(
                        xdaq::DataStream::aligned_read_stream(
                            [callback = std::move(callback)](auto &&event) {
                                std::visit(
                                    [&callback](auto &&event) {
                                        using T = std::decay_t<decltype(event)>;
                                        using namespace xdaq::DataStream;
                                        if constexpr (std::is_same_v<T, Events::DataView>) {
                                            try {
                                                callback(
                                                    DataView{.data = event.data}, std::nullopt
                                                );
                                            } catch (const std::exception &e) {
                                                spdlog::error("Error in callback: {}", e.what());
                                            }
                                        } else if constexpr (std::is_same_v<T, Events::OwnedData>) {
                                            try {
                                                callback(
                                                    DataView{
                                                        .data = {event.buffer.get(), event.length}
                                                    },
                                                    std::nullopt
                                                );
                                            } catch (const std::exception &e) {
                                                spdlog::error("Error in callback: {}", e.what());
                                            }
                                        } else if constexpr (std::is_same_v<T, Events::Stop>) {
                                            callback(std::nullopt, std::nullopt);
                                        } else if constexpr (std::is_same_v<T, Events::Error>) {
                                            callback(std::nullopt, event.error);
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
            },
            py::arg("addr"),
            py::arg("alignment"),
            py::arg("callback"),
            py::kw_only(),
            py::arg("chunk_size") = 0,
            py::arg("max_queue_elements") = 4096,
            py::return_value_policy::move
        )
        .def("get_status", &xdaq::Device::get_status)
        .def("get_info", &xdaq::Device::get_info);

    py::class_<xdaq::DeviceManager, std::shared_ptr<xdaq::DeviceManager>>(m, "DeviceManager")
        .def("info", [](const xdaq::DeviceManager &d) { return d.info(); })
        .def("location", [](const xdaq::DeviceManager &d) { return d.location(); })
        .def("list_devices", [](const xdaq::DeviceManager &d) { return d.list_devices(); })
        .def(
            "get_device_options",
            [](const xdaq::DeviceManager &d) { return d.get_device_options(); }
        )
        .def("create_device", [](xdaq::DeviceManager &d, const std::string &device) {
            py::gil_scoped_release release;
            return d.create_device(device);
        });

    m.def("get_device_manager", &xdaq::get_device_manager);
}