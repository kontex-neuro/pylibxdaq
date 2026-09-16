#include "stream_read.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace xdaqnp
{
namespace
{

/// Smallest packet count that fits in every supplied buffer, given each buffer's stride.
std::size_t capacity_of(std::initializer_list<std::pair<std::size_t, std::size_t>> size_stride)
{
    std::size_t n = std::numeric_limits<std::size_t>::max();
    for (auto [size, stride] : size_stride) n = std::min(n, size / stride);
    return n;
}

}  // namespace

std::size_t read_packets(NP1::Probe::PacketQueue &queue, const NP1PacketBuffers &out)
{
    constexpr std::size_t CH = NP1::Probe::PROBE_CHANNEL_COUNT;
    constexpr std::size_t SF = NP1::Probe::PROBE_SUPERFRAMESIZE;

    const auto capacity = capacity_of(
        {{out.ap.size(), CH * SF},
         {out.lfp.size(), CH},
         {out.timestamps.size(), SF},
         {out.status.size(), SF}}
    );

    electrodePacket packet;
    std::size_t n = 0;
    for (; n < capacity && queue.try_dequeue(packet); ++n) {
        // electrodePacket::apData is [12][384] and already contiguous, so a whole packet's AP
        // block lands as one run of 12 rows.
        std::memcpy(out.ap.data() + n * SF * CH, packet.apData, sizeof(packet.apData));
        std::memcpy(out.lfp.data() + n * CH, packet.lfpData, sizeof(packet.lfpData));
        std::memcpy(out.timestamps.data() + n * SF, packet.timestamp, sizeof(packet.timestamp));
        std::memcpy(out.status.data() + n * SF, packet.Status, sizeof(packet.Status));
    }
    return n;
}

std::size_t read_packets(NP2::NP2ProbeBase::PacketQueue &queue, const NP2PacketBuffers &out)
{
    constexpr std::size_t CH = NP2::APData::CHANNEL_COUNT;

    const auto capacity = capacity_of({{out.ap.size(), CH}, {out.timestamps.size(), 1}});

    NP2::APData sample;
    std::size_t n = 0;
    for (; n < capacity && queue.try_dequeue(sample); ++n) {
        std::memcpy(out.ap.data() + n * CH, sample.data.data(), sizeof(sample.data));
        out.timestamps[n] = sample.timestamp;
    }
    return n;
}

std::size_t read_packets(IOFrameQueue &queue, const IOFrameBuffers &out)
{
    const auto capacity = capacity_of(
        {{out.timestamps.size(), 1},
         {out.adc.size(), 8},
         {out.dac.size(), 8},
         {out.di.size(), 1},
         {out.dout.size(), 1}}
    );

    io::IOFrame frame;
    std::size_t n = 0;
    for (; n < capacity && queue.try_dequeue(frame); ++n) {
        out.timestamps[n] = frame.timestamp;
        std::memcpy(out.adc.data() + n * 8, frame.ADC.data(), sizeof(frame.ADC));
        std::memcpy(out.dac.data() + n * 8, frame.DAC.data(), sizeof(frame.DAC));
        out.di[n] = frame.DI;
        out.dout[n] = frame.DO;
    }
    return n;
}

}  // namespace xdaqnp
