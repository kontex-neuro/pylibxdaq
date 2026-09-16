#pragma once

// Design notes
// ------------
// * Ownership. Only NPController, NPPort, xdaq::Device and the IO worker are reference counted.
//   Headstages and probes are non-movable objects owned *inside* an NPPort, so their handles
//   store a weak_ptr to the port plus enough coordinates (dock) to find themselves again, and
//   re-resolve on every call. A handle whose port was released, or whose headstage was replaced
//   by a later detect_headstage(), raises instead of dereferencing a dangling pointer.
//
// * The device. This module neither binds nor owns a device type. Discovery and opening live in
//   pyxdaq_device; Controller only reads a `Device` handle at construction. NPController takes
//   its own share of the device, so the handle can be closed afterwards without breaking the
//   controller -- but the underlying hardware is only released once both are gone.
//
// * Data plane. Each probe's packet queue is exposed as an opaque stream object with a single
//   read() method that drains whatever is queued into freshly allocated, contiguous numpy
//   arrays. Draining happens with the GIL released; per-packet dequeue never crosses into
//   Python.

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <demux/neuropixels_worker.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <xdaq/device.h>

#include <NP1/np1_headstage.hpp>
#include <NP2/np2_headstage.hpp>
#include <controller.hpp>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <np_port.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <utils.hpp>
#include <vector>

#include "stream_read.hpp"

namespace nb = nanobind;

namespace pynp
{

using namespace xdaqnp;

template <typename T>
T unwrap(std::expected<T, std::string> result)
{
    if (!result) throw std::runtime_error(result.error());
    return std::move(result).value();
}

inline void unwrap(std::expected<void, std::string> result)
{
    if (!result) throw std::runtime_error(result.error());
}

/// Most probe control calls report failure as a bare `false`; turn that into an exception so
/// Python callers cannot silently ignore a half-configured probe.
inline void check(bool ok, std::string_view what)
{
    if (!ok) throw std::runtime_error(fmt::format("{} failed", what));
}

inline void collect(
    std::vector<std::string> &errors, std::string_view name,
    const std::expected<void, std::string> &result
)
{
    if (!result) errors.push_back(fmt::format("{}: {}", name, result.error()));
}

inline void raise_if_any(const std::vector<std::string> &errors, std::string_view what)
{
    if (errors.empty()) return;
    throw std::runtime_error(fmt::format("{} failed -- {}", what, fmt::join(errors, "; ")));
}

struct PortControl : std::mutex {};

inline std::unique_lock<std::mutex> acquire_control(std::mutex &mutex)
{
    nb::gil_scoped_release release;
    return std::unique_lock<std::mutex>(mutex);
}

struct Controller {
    std::shared_ptr<NPController> controller;
    int num_ports = 0;
    // One control lock per port index, created on first use by port(). libxdaqnp itself has no
    // such lock: in C++ a returned Ref is a borrow the caller is responsible for sequencing.
    // Python has no equivalent, and we release the GIL during long uploads, so the binding
    // supplies the serialization instead. Only ever touched with the GIL held.
    std::map<int, std::shared_ptr<PortControl>> port_locks;

    NPController &get() const
    {
        if (!controller) throw std::runtime_error("Controller is closed");
        return *controller;
    }

    std::shared_ptr<PortControl> lock_for(int port_index)
    {
        auto &slot = port_locks[port_index];
        if (!slot) slot = std::make_shared<PortControl>();
        return slot;
    }
};

struct Port {
    std::shared_ptr<NPPort> port;
    int index = 0;
    std::shared_ptr<PortControl> control;

    NPPort &get() const
    {
        if (!port) throw std::runtime_error("Port is closed");
        return *port;
    }
};

/// A resolved borrow of an object living inside an NPPort, for control-plane use.
///
/// Three things keep it valid: the port's shared_ptr (which keeps the controller and its UART
/// alive), the control mutex's shared_ptr, and the lock itself, held for the whole call. Every
/// control-plane entry point in this module resolves through here, so detect_headstage() cannot
/// destroy a headstage or its probes while we are using them, and two configuration sequences
/// cannot interleave.
///
/// Member order matters: `control_guard` is declared after `control` so it unlocks before the
/// mutex it refers to can be released.
template <typename T>
struct Resolved {
    std::shared_ptr<NPPort> port;
    std::shared_ptr<PortControl> control;
    std::unique_lock<std::mutex> control_guard;
    T *ptr;

    T &operator*() const noexcept { return *ptr; }
    T *operator->() const noexcept { return ptr; }
};

struct InvalidatedHandle : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] inline void invalidated(std::string_view what)
{
    throw InvalidatedHandle(fmt::format(
        "{} handle is no longer valid: the port was released or the headstage was re-detected", what
    ));
}

inline void check_generation(
    std::uint64_t current, std::uint64_t generation, std::string_view what
)
{
    if (generation != current) invalidated(what);
}

inline std::shared_ptr<NPPort> lock_port(const std::weak_ptr<NPPort> &port, std::string_view what)
{
    auto locked = port.lock();
    if (!locked) invalidated(what);
    return locked;
}

template <typename HeadstageT>
Resolved<HeadstageT> resolve_headstage(
    const std::weak_ptr<NPPort> &port, const std::shared_ptr<PortControl> &control,
    std::uint64_t generation, std::string_view what
)
{
    auto locked = lock_port(port, what);
    auto guard = acquire_control(*control);
    check_generation(locked->headstage_generation, generation, what);
    if (!locked->headstage) invalidated(what);
    auto *typed = std::get_if<HeadstageT>(&locked->headstage.value());
    if (typed == nullptr) invalidated(what);
    return {std::move(locked), control, std::move(guard), typed};
}

inline NP2::Probes *np2_slot(NP2::Headstage &headstage, NP2::Dock dock)
{
    auto &slot = (dock == NP2::Dock::Front) ? headstage.probe_a : headstage.probe_b;
    return slot.get();
}

struct NP1Headstage {
    std::weak_ptr<NPPort> port;
    std::shared_ptr<PortControl> control;
    std::uint64_t generation;
    Resolved<NP1::Headstage> resolve() const
    {
        return resolve_headstage<NP1::Headstage>(port, control, generation, "NP1 headstage");
    }
};

struct NP2Headstage {
    std::weak_ptr<NPPort> port;
    std::shared_ptr<PortControl> control;
    std::uint64_t generation;
    Resolved<NP2::Headstage> resolve() const
    {
        return resolve_headstage<NP2::Headstage>(port, control, generation, "NP2 headstage");
    }
};

struct NP1Probe {
    std::weak_ptr<NPPort> port;
    std::shared_ptr<PortControl> control;
    std::uint64_t generation;

    Resolved<NP1::Probe> resolve() const
    {
        auto headstage = resolve_headstage<NP1::Headstage>(port, control, generation, "NP1 probe");
        if (!headstage->probe) invalidated("NP1 probe");
        return {
            std::move(headstage.port),
            std::move(headstage.control),
            std::move(headstage.control_guard),
            headstage->probe.get()
        };
    }
};

template <typename ProbeT>
struct NP2Probe {
    std::weak_ptr<NPPort> port;
    std::shared_ptr<PortControl> control;
    NP2::Dock dock;
    std::uint64_t generation;

    Resolved<ProbeT> resolve() const
    {
        auto headstage = resolve_headstage<NP2::Headstage>(port, control, generation, "NP2 probe");
        auto *probes = np2_slot(*headstage, dock);
        if (probes == nullptr) invalidated("NP2 probe");
        auto *probe = std::get_if<ProbeT>(probes);
        if (probe == nullptr) invalidated("NP2 probe");
        return {
            std::move(headstage.port),
            std::move(headstage.control),
            std::move(headstage.control_guard),
            probe
        };
    }

    Resolved<NP2::NP2ProbeBase> resolve_base() const
    {
        auto headstage = resolve_headstage<NP2::Headstage>(port, control, generation, "NP2 probe");
        auto *probes = np2_slot(*headstage, dock);
        if (probes == nullptr) invalidated("NP2 probe");
        auto *base = *probes | match{[](auto &probe) -> NP2::NP2ProbeBase * { return &probe; }};
        return {
            std::move(headstage.port),
            std::move(headstage.control),
            std::move(headstage.control_guard),
            base
        };
    }
};

using NP2013Probe = NP2Probe<NP2::NP2013NP2014>;
using NP2003Probe = NP2Probe<NP2::NP2003NP2004>;

// The demux worker driving a stream lives inside the probe, so it is looked up (and validated)
// on every start/stop. The port control lock is held across the NPController call.
struct WorkerRef {
    std::shared_ptr<NPPort> port;
    std::shared_ptr<PortControl> control;
    std::unique_lock<std::mutex> control_guard;
    demux::NeuropixelsWorker *worker;
};

// Streams hold their queue directly rather than a path back to the probe. The queue is shared
// with the producer, so it stays valid even if the probe is torn down mid-read; read() therefore
// needs neither the port lock nor a live probe, and never blocks control-plane work.
struct NP1Stream {
    std::shared_ptr<NP1::Probe::PacketQueue> queue;
};

struct NP2Stream {
    std::shared_ptr<NP2::NP2ProbeBase::PacketQueue> queue;
};

struct IOStream {
    std::shared_ptr<demux::NeuropixelsWorker> worker;
    std::shared_ptr<IOFrameQueue> queue;
};

inline WorkerRef worker_of(NP1Probe &handle)
{
    auto probe = handle.resolve();
    if (!probe->worker) throw std::runtime_error("NP1 probe has no demux worker");
    return {
        std::move(probe.port),
        std::move(probe.control),
        std::move(probe.control_guard),
        probe->worker.get()
    };
}

template <typename ProbeT>
WorkerRef worker_of(NP2Probe<ProbeT> &handle)
{
    auto probe = handle.resolve_base();
    if (!probe->worker) throw std::runtime_error("NP2 probe has no demux worker");
    return {
        std::move(probe.port),
        std::move(probe.control),
        std::move(probe.control_guard),
        probe->worker.get()
    };
}

inline WorkerRef worker_of(IOStream &stream)
{
    if (!stream.worker) throw std::runtime_error("IO stream is closed");
    return {nullptr, nullptr, {}, stream.worker.get()};
}

template <typename Handle>
WorkerRef worker_to_stop(Handle &handle)
{
    try {
        return worker_of(handle);
    } catch (const InvalidatedHandle &) {
        return {};
    }
}

struct NP1Chunk {
    std::size_t packets = 0;
    std::vector<std::int16_t> ap;
    std::vector<std::int16_t> lfp;
    std::vector<std::uint64_t> timestamps;
    std::vector<std::uint16_t> status;
};

struct NP2Chunk {
    std::size_t samples = 0;
    std::vector<std::int16_t> ap;
    std::vector<std::uint64_t> timestamps;
};

struct IOChunk {
    std::size_t frames = 0;
    std::vector<std::uint64_t> timestamps;
    std::vector<std::uint16_t> adc;
    std::vector<std::uint16_t> dac;
    std::vector<std::uint32_t> di;
    std::vector<std::uint32_t> dout;
};

NP1Chunk read_np1(NP1::Probe::PacketQueue &queue, std::size_t max_packets, double timeout_ms);
NP2Chunk read_np2(
    NP2::NP2ProbeBase::PacketQueue &queue, std::size_t max_samples, double timeout_ms
);
IOChunk read_io(IOFrameQueue &queue, std::size_t max_frames, double timeout_ms);

/// Zero-copy numpy view over a chunk's buffer. Paired with rv_policy::reference_internal so the
/// array keeps the owning chunk alive.
template <typename T>
nb::ndarray<nb::numpy, T> view(std::vector<T> &data, std::initializer_list<std::size_t> shape)
{
    // An empty read is the normal case when polling, and vector::data() may be null then;
    // hand out a valid pointer to a zero-length array instead.
    static T empty_placeholder{};
    T *base = data.empty() ? &empty_placeholder : data.data();
    return nb::ndarray<nb::numpy, T>(base, shape, nb::handle());
}

void bind_data(nb::module_ &m);
void bind_probes(nb::module_ &m);
void bind_headstages(nb::module_ &m);
void bind_controller(nb::module_ &m);

}  // namespace pynp
