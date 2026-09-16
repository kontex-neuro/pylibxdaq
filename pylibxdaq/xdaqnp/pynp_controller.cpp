#include <nlohmann/json.hpp>
#include <serdes.hpp>

#include "pynp.hpp"
#include "pyxdaq_handles.hpp"

using namespace nb::literals;
using json = nlohmann::json;

namespace pynp
{

void bind_controller(nb::module_ &m)
{
    nb::class_<pynp::Port>(m, "Port")
        .def_ro("index", &pynp::Port::index)
        .def_prop_ro("headstage_generation", [](pynp::Port &p) {
            auto guard = pynp::acquire_control(*p.control);
            return p.get().headstage_generation;
        })
        .def(
            "switch_to_data_link",
            [](pynp::Port &p) {
                auto &port = p.get();
                auto guard = pynp::acquire_control(*p.control);
                nb::gil_scoped_release release;
                pynp::unwrap(switch_to_data_link(port.serdes));
            },
            "Put the SerDes link into data mode; required before streaming."
        )
        .def("__repr__", [](const pynp::Port &p) {
            return fmt::format("<Port index={}>", p.index);
        });

    auto controller_cls =
        nb::class_<pynp::Controller>(m, "Controller", "Control plane for a Neuropixels-mode XDAQ.");
    controller_cls
        .def(
            "__init__",
            [](pynp::Controller *self, pyxdaq::DeviceHandle &device) {
                device.check();
                auto info = json::parse(pynp::unwrap(device.device->get_info()));
                const int num_ports = info.value("NP", 0);
                // Copy the shared_ptr while the GIL is still held: DeviceHandle::close() can
                // run concurrently from another thread once the GIL is released below, and
                // reading `device.device` at that point would race its reset() on the same
                // shared_ptr instance. `dev` is an independent copy, so the two can't collide.
                auto dev = device.device;
                std::shared_ptr<NPController> controller;
                {
                    nb::gil_scoped_release release;
                    controller = pynp::unwrap(NPController::create(dev));
                }
                new (self) pynp::Controller{std::move(controller), num_ports};
            },
            "device"_a
        )
        .def_ro("num_ports", &pynp::Controller::num_ports)
        .def(
            "port",
            [](pynp::Controller &c, int index) {
                auto &controller = c.get();
                auto control = c.lock_for(index);
                std::shared_ptr<NPPort> port;
                {
                    nb::gil_scoped_release release;
                    port = pynp::unwrap(controller.get_np_port(index));
                }
                return pynp::Port{std::move(port), index, std::move(control)};
            },
            "index"_a,
            "Open (or re-obtain) the port at `index`. Ports are 0-based."
        )
        .def(
            "detect_headstage",
            [](pynp::Controller &c, pynp::Port &p) -> nb::object {
                auto &controller = c.get();
                auto &port = p.get();
                auto guard = pynp::acquire_control(*p.control);
                HeadstageRef headstage = [&] {
                    nb::gil_scoped_release release;
                    return pynp::unwrap(controller.detect_headstage(port));
                }();
                const auto generation = port.headstage_generation;
                std::weak_ptr<NPPort> weak = p.port;
                return headstage |
                       match{
                           [&](std::reference_wrapper<NP1::Headstage>) {
                               return nb::cast(pynp::NP1Headstage{weak, p.control, generation});
                           },
                           [&](std::reference_wrapper<NP2::Headstage>) {
                               return nb::cast(pynp::NP2Headstage{weak, p.control, generation});
                           }
                       };
            },
            "port"_a
        )
        .def(
            "create_io_stream",
            [](pynp::Controller &c, std::size_t max_queued) {
                auto &controller = c.get();
                auto queue = std::make_shared<IOFrameQueue>();
                nb::gil_scoped_release release;
                auto worker = pynp::unwrap(controller.create_io_stream(
                    [queue, max_queued](io::IOFrame frame) {
                        // The consumer polls; drop rather than grow without bound if it stalls.
                        if (queue->size_approx() >= max_queued) return;
                        queue->enqueue(frame);
                    },
                    [queue] { queue->clear(); }
                ));
                return pynp::IOStream{std::move(worker), std::move(queue)};
            },
            "max_queued"_a = 1 << 17,
            "Create the IO frame stream. Only one may exist at a time."
        );

    // start_stream / stop_stream take any probe or IO handle. nanobind needs concrete parameter
    // types to build its signature, so each overload is stamped out from a template rather than
    // a generic lambda.
    const auto bind_stream_control = [&controller_cls]<typename Handle>() {
        controller_cls
            .def(
                "start_stream",
                [](pynp::Controller &c, Handle &handle) {
                    auto &controller = c.get();
                    auto worker = pynp::worker_of(handle);
                    nb::gil_scoped_release release;
                    pynp::unwrap(controller.start_stream(*worker.worker));
                },
                "stream"_a
            )
            .def(
                "stop_stream",
                [](pynp::Controller &c, Handle &handle) {
                    auto &controller = c.get();
                    auto worker = pynp::worker_to_stop(handle);
                    if (!worker.worker) return;
                    nb::gil_scoped_release release;
                    pynp::unwrap(controller.stop_stream(*worker.worker));
                },
                "stream"_a
            );
    };

    bind_stream_control.template operator()<pynp::NP1Probe>();
    bind_stream_control.template operator()<pynp::NP2013Probe>();
    bind_stream_control.template operator()<pynp::NP2003Probe>();
    bind_stream_control.template operator()<pynp::IOStream>();
}

}  // namespace pynp
