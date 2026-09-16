#include "../xdaqnp/pynp.hpp"

#include <chrono>
#include <thread>

struct TestControl : pynp::PortControl {
    std::uint64_t generation = 0;
};

NB_MODULE(_test_np_native, module)
{
    nb::class_<TestControl>(module, "Control")
        .def(nb::init<>())
        .def("hold", [](TestControl &control, nb::callable entered) {
            auto guard = pynp::acquire_control(control);
            entered();
            nb::gil_scoped_release release;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        })
        .def("redetect", [](TestControl &control) {
            auto guard = pynp::acquire_control(control);
            return ++control.generation;
        })
        .def("validate", [](TestControl &control, std::uint64_t generation) {
            auto guard = pynp::acquire_control(control);
            pynp::check_generation(control.generation, generation, "test probe");
        });

    module.def("queue_reset", [] {
        xdaqnp::IOFrameQueue queue;
        xdaqnp::io::IOFrame frame{};
        frame.timestamp = 1;
        queue.enqueue(frame);
        queue.clear();
        if (queue.try_dequeue(frame)) return false;
        frame.timestamp = 2;
        queue.enqueue(frame);
        return queue.try_dequeue(frame) && frame.timestamp == 2;
    });

    module.def("retired_probe_stop", [] {
        pynp::NP1Probe np1{};
        pynp::NP2013Probe np2013{};
        pynp::NP2003Probe np2003{};
        bool start_rejected = false;
        try {
            pynp::worker_of(np1);
        } catch (const pynp::InvalidatedHandle &) {
            start_rejected = true;
        }
        return start_rejected && !pynp::worker_to_stop(np1).worker &&
               !pynp::worker_to_stop(np2013).worker && !pynp::worker_to_stop(np2003).worker;
    });

    module.def("closed_io_stop", [] {
        pynp::IOStream stream{};
        pynp::worker_to_stop(stream);
    });
}