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
 * @brief Cuts a byte stream into chunks that are whole multiples of `alignment`.
 *
 * Replaces `xdaq::DataStream::aligned_read_stream`, differing from it in three ways:
 *
 *  - `merge` delivers one event per upstream chunk instead of two. Without it, a
 *    chunk straddling a sample boundary is emitted as the spanning sample followed
 *    by the remainder; with it, the two are made contiguous in a reused scratch
 *    buffer. Costs one memcpy of the chunk, halves the event rate, and lowers
 *    latency by roughly one callback's overhead.
 *  - A trailing partial sample is dropped at `Stop` rather than emitted short, so
 *    every delivered view is a whole multiple of `alignment`.
 *  - `Error` discards the partial sample, resynchronizing on the next chunk. The
 *    dropped bytes leave a hole, so keeping it would misframe everything after.
 *
 * @param on_receive Callback to receive the aligned data chunks.
 * @param alignment The alignment boundary, in bytes.
 * @param merge Deliver one event per upstream chunk rather than two.
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
