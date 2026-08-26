#pragma once

#include <xdaq/data_streams.h>
#include <xdaq/device.h>

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace pyxdaq
{

/**
 * @brief Alignment adapter, replacing `xdaq::DataStream::aligned_read_stream`.
 *
 * Local to pylibxdaq so that the two fixes below need only a rebuild rather than a
 * re-export of the libxdaq conan package. Intended to move upstream once soaked.
 *
 * Differs from `xdaq::DataStream::aligned_read_stream` in three ways:
 *
 * `merge`
 *     Upstream never copies the body of a chunk: when a chunk straddles a sample
 *     boundary it assembles the spanning sample in a side buffer, emits it alone,
 *     then emits the remainder in place. A straddling chunk therefore costs two
 *     downstream events, and two thirds of chunks straddle at the chunk sizes used
 *     for low-latency streaming. That is the right trade when the consumer is
 *     cheap, and the wrong one when each event costs a GIL acquire and a Python
 *     dispatch. With `merge`, the spanning sample and the body are made contiguous
 *     in a reused scratch buffer instead: one memcpy of the chunk, one event.
 *     Latency is unchanged -- both events were already available at the same
 *     instant -- but the event rate halves, and that is what limits the
 *     small-chunk end.
 *
 * `Stop`
 *     Upstream flushes whatever partial sample remains as a short final view.
 *     A fraction of a sample cannot be parsed, so every consumer has to recognize
 *     and discard it, and one that does not reports a spurious parse failure at
 *     every shutdown. Dropped here instead.
 *
 * `Error`
 *     Upstream forwards the error and keeps its partial sample. The dropped bytes
 *     leave a hole of unknown size, so gluing the next chunk onto that partial
 *     misframes every sample from then on. The partial is discarded here, which
 *     resynchronizes on the next chunk boundary.
 *
 * @param on_receive Callback to receive the aligned data chunks.
 * @param alignment The alignment boundary, in bytes.
 * @param merge Deliver one event per upstream chunk rather than two.
 *
 * @note Every delivered chunk is a whole multiple of `alignment`, including the
 * last -- there is no short final view.
 */
[[nodiscard]] inline xdaq::DataStream::receive_callback aligned_read_stream(
    xdaq::DataStream::receive_callback &&on_receive, std::size_t alignment, bool merge = false
)
{
    if (alignment == 0) throw std::runtime_error("alignment must be greater than zero");

    struct Ctx {
        std::vector<unsigned char> leftover;  // < alignment bytes, carried across events
        std::vector<unsigned char> scratch;   // assembly area; only ever grows
    };

    return {[ctx = Ctx{}, on_receive = std::move(on_receive), alignment, merge](
                xdaq::DataStream::Event &&event
            ) mutable {
        std::visit(
            [&](auto &&event) {
                using T = std::decay_t<decltype(event)>;
                using namespace xdaq::DataStream;
                std::span<unsigned char> data;
                if constexpr (std::is_same_v<T, Events::DataView>) {
                    data = event.data;
                } else if constexpr (std::is_same_v<T, Events::OwnedData>) {
                    data = std::span<unsigned char>(event);
                } else if constexpr (std::is_same_v<T, Events::Error>) {
                    ctx.leftover.clear();  // resync; see the note above
                    on_receive(std::move(event));
                    return;
                } else if constexpr (std::is_same_v<T, Events::Stop>) {
                    on_receive(std::move(event));  // a partial sample is not useful
                    return;
                } else {
                    static_assert(xdaq::always_false_v<T>, "non-exhaustive visitor");
                }

                if (ctx.leftover.empty()) {
                    // No boundary to repair, so the chunk goes out as a view of the
                    // upstream buffer with no copy at all, merged or not.
                    const auto n_full = data.size() - data.size() % alignment;
                    ctx.leftover.assign(data.begin() + n_full, data.end());
                    if (n_full > 0) on_receive(Events::DataView{.data = data.first(n_full)});
                    return;
                }

                const auto total = ctx.leftover.size() + data.size();
                if (total < alignment) {  // upstream handed over less than one sample
                    ctx.leftover.insert(ctx.leftover.end(), data.begin(), data.end());
                    return;
                }

                if (merge) {
                    if (ctx.scratch.size() < total) ctx.scratch.resize(total);
                    std::ranges::copy(
                        data, std::ranges::copy(ctx.leftover, ctx.scratch.begin()).out
                    );
                    const auto n_full = total - total % alignment;
                    ctx.leftover.assign(ctx.scratch.begin() + n_full, ctx.scratch.begin() + total);
                    on_receive(Events::DataView{.data = {ctx.scratch.data(), n_full}});
                    return;
                }

                // Two events, but the body is never copied: assemble only the sample
                // that spans the boundary, then hand over the rest where it lies.
                const auto fill = alignment - ctx.leftover.size();
                if (ctx.scratch.size() < alignment) ctx.scratch.resize(alignment);
                std::ranges::copy(
                    data.first(fill), std::ranges::copy(ctx.leftover, ctx.scratch.begin()).out
                );
                on_receive(Events::DataView{.data = {ctx.scratch.data(), alignment}});
                const auto rest = data.size() - fill;
                const auto tail = rest % alignment;
                ctx.leftover.assign(data.end() - tail, data.end());
                if (rest - tail > 0)
                    on_receive(Events::DataView{.data = data.subspan(fill, rest - tail)});
            },
            std::move(event)
        );
    }};
}

}  // namespace pyxdaq
