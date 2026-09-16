#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "NP1/np1_probe_control.hpp"
#include "NP2/np2_probe_control.hpp"
#include "io/io_frame.h"
#include "xdaqnp_export.hpp"

namespace xdaqnp
{

// Bulk drain of a probe's packet queue into caller-owned, channel-major contiguous buffers.
//
// Rationale: the queues hold packet structs that interleave samples with per-sample metadata.
// Consumers (array languages, file writers, ring buffers) want each field as one contiguous
// run instead. Draining one packet at a time across a language boundary does not keep up with
// a 30 kHz probe, so the loop lives here and the caller supplies the destination memory.
//
// Every read is non-blocking: it moves whatever is currently queued, up to the capacity implied
// by the smallest buffer, and returns how many packets were actually taken. Buffers beyond that
// count are left untouched.

/// Destination for NP1 electrode packets. `n` below is the packet capacity, i.e. the smallest
/// count any single buffer can hold; each packet carries SUPERFRAME_SIZE (12) AP samples and
/// one LFP sample.
struct NP1PacketBuffers {
    std::span<std::int16_t> ap;           ///< [n * 12][384], row-major AP samples
    std::span<std::int16_t> lfp;          ///< [n][384], one LFP row per packet
    std::span<std::uint64_t> timestamps;  ///< [n * 12], one per AP sample
    std::span<std::uint16_t> status;      ///< [n * 12], one per AP sample
};

/// @return Number of packets moved; multiply by 12 for the number of AP rows written.
XDAQNP_EXPORT std::size_t read_packets(NP1::Probe::PacketQueue &queue, const NP1PacketBuffers &out);

/// Destination for NP2 AP samples. `n` is the sample capacity.
struct NP2PacketBuffers {
    std::span<std::int16_t> ap;           ///< [n][384], row-major AP samples
    std::span<std::uint64_t> timestamps;  ///< [n], one per sample
};

/// @return Number of samples moved.
XDAQNP_EXPORT std::size_t read_packets(
    NP2::NP2ProbeBase::PacketQueue &queue, const NP2PacketBuffers &out
);

/// Destination for XDAQ IO frames. `n` is the frame capacity.
struct IOFrameBuffers {
    std::span<std::uint64_t> timestamps;  ///< [n]
    std::span<std::uint16_t> adc;         ///< [n][8]
    std::span<std::uint16_t> dac;         ///< [n][8]
    std::span<std::uint32_t> di;          ///< [n]
    std::span<std::uint32_t> dout;        ///< [n]
};

using IOFrameQueue = demux::ResettableQueue<io::IOFrame>;
/// @return Number of frames moved.
XDAQNP_EXPORT std::size_t read_packets(IOFrameQueue &queue, const IOFrameBuffers &out);

}  // namespace xdaqnp
