#include "pynp.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

using namespace nb::literals;

namespace pynp
{

namespace
{

/// How many items to take this round. Called with the GIL released. `size_approx()` on an SPSC
/// queue only ever undercounts from the consumer side, so the subsequent drain never overruns.
template <typename QueueT>
std::size_t items_to_read(QueueT &queue, std::size_t max_items, double timeout_ms)
{
    auto available = queue.size_approx();
    if (available == 0 && timeout_ms > 0.0) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::duration<double, std::milli>(timeout_ms);
        while (available == 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            available = queue.size_approx();
        }
    }
    return (max_items == 0) ? available : std::min(available, max_items);
}

}  // namespace

NP1Chunk read_np1(NP1::Probe::PacketQueue &queue, std::size_t max_packets, double timeout_ms)
{
    constexpr std::size_t CH = NP1::Probe::PROBE_CHANNEL_COUNT;
    constexpr std::size_t SF = NP1::Probe::PROBE_SUPERFRAMESIZE;

    const auto wanted = items_to_read(queue, max_packets, timeout_ms);

    NP1Chunk chunk;
    chunk.ap.resize(wanted * SF * CH);
    chunk.lfp.resize(wanted * CH);
    chunk.timestamps.resize(wanted * SF);
    chunk.status.resize(wanted * SF);

    chunk.packets =
        read_packets(queue, {chunk.ap, chunk.lfp, chunk.timestamps, chunk.status});

    chunk.ap.resize(chunk.packets * SF * CH);
    chunk.lfp.resize(chunk.packets * CH);
    chunk.timestamps.resize(chunk.packets * SF);
    chunk.status.resize(chunk.packets * SF);
    return chunk;
}

NP2Chunk read_np2(
    NP2::NP2ProbeBase::PacketQueue &queue, std::size_t max_samples, double timeout_ms
)
{
    constexpr std::size_t CH = NP2::APData::CHANNEL_COUNT;

    const auto wanted = items_to_read(queue, max_samples, timeout_ms);

    NP2Chunk chunk;
    chunk.ap.resize(wanted * CH);
    chunk.timestamps.resize(wanted);

    chunk.samples = read_packets(queue, {chunk.ap, chunk.timestamps});

    chunk.ap.resize(chunk.samples * CH);
    chunk.timestamps.resize(chunk.samples);
    return chunk;
}

IOChunk read_io(IOFrameQueue &queue, std::size_t max_frames, double timeout_ms)
{
    const auto wanted = items_to_read(queue, max_frames, timeout_ms);

    IOChunk chunk;
    chunk.timestamps.resize(wanted);
    chunk.adc.resize(wanted * 8);
    chunk.dac.resize(wanted * 8);
    chunk.di.resize(wanted);
    chunk.dout.resize(wanted);

    chunk.frames =
        read_packets(queue, {chunk.timestamps, chunk.adc, chunk.dac, chunk.di, chunk.dout});

    chunk.timestamps.resize(chunk.frames);
    chunk.adc.resize(chunk.frames * 8);
    chunk.dac.resize(chunk.frames * 8);
    chunk.di.resize(chunk.frames);
    chunk.dout.resize(chunk.frames);
    return chunk;
}

void bind_data(nb::module_ &m)
{
    nb::class_<pynp::NP1Chunk>(
        m,
        "NP1Chunk",
        "A batch of NP1 electrode packets. Arrays are views into the chunk and stay valid for as "
        "long as it is referenced."
    )
        .def_ro("packets", &pynp::NP1Chunk::packets)
        .def_prop_ro(
            "ap",
            [](pynp::NP1Chunk &c) {
                return pynp::view(c.ap, {c.packets * 12, 384});
            },
            nb::rv_policy::reference_internal,
            "(packets * 12, 384) int16 AP samples."
        )
        .def_prop_ro(
            "lfp",
            [](pynp::NP1Chunk &c) { return pynp::view(c.lfp, {c.packets, 384}); },
            nb::rv_policy::reference_internal,
            "(packets, 384) int16 LFP samples, one row per AP superframe."
        )
        .def_prop_ro(
            "timestamps",
            [](pynp::NP1Chunk &c) { return pynp::view(c.timestamps, {c.packets * 12}); },
            nb::rv_policy::reference_internal,
            "(packets * 12,) uint64, one per AP row."
        )
        .def_prop_ro(
            "status",
            [](pynp::NP1Chunk &c) { return pynp::view(c.status, {c.packets * 12}); },
            nb::rv_policy::reference_internal,
            "(packets * 12,) uint16, one per AP row."
        )
        .def("__len__", [](const pynp::NP1Chunk &c) { return c.packets; })
        .def("__repr__", [](const pynp::NP1Chunk &c) {
            return fmt::format("<NP1Chunk packets={} ap_rows={}>", c.packets, c.packets * 12);
        });

    nb::class_<pynp::NP2Chunk>(m, "NP2Chunk", "A batch of NP2 AP samples.")
        .def_ro("samples", &pynp::NP2Chunk::samples)
        .def_prop_ro(
            "ap",
            [](pynp::NP2Chunk &c) { return pynp::view(c.ap, {c.samples, 384}); },
            nb::rv_policy::reference_internal,
            "(samples, 384) int16 AP samples."
        )
        .def_prop_ro(
            "timestamps",
            [](pynp::NP2Chunk &c) { return pynp::view(c.timestamps, {c.samples}); },
            nb::rv_policy::reference_internal,
            "(samples,) uint64."
        )
        .def("__len__", [](const pynp::NP2Chunk &c) { return c.samples; })
        .def("__repr__", [](const pynp::NP2Chunk &c) {
            return fmt::format("<NP2Chunk samples={}>", c.samples);
        });

    nb::class_<pynp::IOChunk>(m, "IOChunk", "A batch of XDAQ IO frames.")
        .def_ro("frames", &pynp::IOChunk::frames)
        .def_prop_ro(
            "timestamps",
            [](pynp::IOChunk &c) { return pynp::view(c.timestamps, {c.frames}); },
            nb::rv_policy::reference_internal
        )
        .def_prop_ro(
            "adc",
            [](pynp::IOChunk &c) { return pynp::view(c.adc, {c.frames, 8}); },
            nb::rv_policy::reference_internal
        )
        .def_prop_ro(
            "dac",
            [](pynp::IOChunk &c) { return pynp::view(c.dac, {c.frames, 8}); },
            nb::rv_policy::reference_internal
        )
        .def_prop_ro(
            "di",
            [](pynp::IOChunk &c) { return pynp::view(c.di, {c.frames}); },
            nb::rv_policy::reference_internal
        )
        .def_prop_ro(
            "do",
            [](pynp::IOChunk &c) { return pynp::view(c.dout, {c.frames}); },
            nb::rv_policy::reference_internal
        )
        .def("__len__", [](const pynp::IOChunk &c) { return c.frames; })
        .def("__repr__", [](const pynp::IOChunk &c) {
            return fmt::format("<IOChunk frames={}>", c.frames);
        });

    constexpr auto read_doc =
        "Drain up to `max_items` queued items (0 = everything available) into a new chunk. "
        "Non-blocking unless `timeout_ms` is given, in which case it waits that long for the "
        "first item. Releases the GIL.";

    nb::class_<pynp::NP1Stream>(m, "NP1Stream", "Opaque handle to an NP1 probe's packet queue.")
        .def(
            "read",
            [](pynp::NP1Stream &s, std::size_t max_items, double timeout_ms) {
                nb::gil_scoped_release release;
                return pynp::read_np1(*s.queue, max_items, timeout_ms);
            },
            "max_items"_a = 0,
            "timeout_ms"_a = 0.0,
            read_doc
        );

    nb::class_<pynp::NP2Stream>(m, "NP2Stream", "Opaque handle to an NP2 probe's packet queue.")
        .def(
            "read",
            [](pynp::NP2Stream &s, std::size_t max_items, double timeout_ms) {
                nb::gil_scoped_release release;
                return pynp::read_np2(*s.queue, max_items, timeout_ms);
            },
            "max_items"_a = 0,
            "timeout_ms"_a = 0.0,
            read_doc
        );

    nb::class_<pynp::IOStream>(m, "IOStream", "Opaque handle to the XDAQ IO frame queue.")
        .def(
            "read",
            [](pynp::IOStream &s, std::size_t max_items, double timeout_ms) {
                nb::gil_scoped_release release;
                return pynp::read_io(*s.queue, max_items, timeout_ms);
            },
            "max_items"_a = 0,
            "timeout_ms"_a = 0.0,
            read_doc
        );
}

}  // namespace pynp
