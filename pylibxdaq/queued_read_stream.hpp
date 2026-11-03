#pragma once

// A native queue between a device read stream and Python.
//
// The device's reader thread hands every buffer to `push` (no Python, no GIL). Python takes
// batches with `take`: one call, and one GIL acquisition, per batch instead of one Python callback
// per buffer. A batch is the queued buffers concatenated, in order, up to a byte limit; a batch
// always holds whole buffers. Errors and the stop are kept as state, as the Python callback saw
// them: an error is sticky, and `stopped` is set by the stream's final Stop event.

#include <xdaq/device.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace pyxdaq
{

struct QueuedReadStream {
    using clock = std::chrono::steady_clock;
    using OwnedData = xdaq::DataStream::Events::OwnedData;

    struct Item {
        OwnedData data;
        double received_at;  // steady_clock seconds (CLOCK_MONOTONIC: Python's time.perf_counter)
    };

    struct Batch {
        std::unique_ptr<unsigned char[]> data;
        std::size_t size = 0;
        std::size_t buffers = 0;
        double received_at = 0.0;  // the oldest buffer's
    };

    struct Stats {
        std::size_t buffers = 0;            // handed over by the device
        std::size_t bytes = 0;
        std::size_t max_buffer_bytes = 0;
        std::size_t taken_bytes = 0;        // taken by the reader
        std::size_t batches = 0;
        std::size_t max_queued_bytes = 0;
        std::size_t max_queued_buffers = 0;
        double max_arrival_gap = 0.0;       // seconds between two buffers handed over
        std::size_t arrival_gaps = 0;       // gaps over arrival_gap_threshold
    };

    explicit QueuedReadStream(std::size_t max_queue_bytes, double arrival_gap_threshold = 0.005)
        : max_queue_bytes(max_queue_bytes), arrival_gap_threshold(arrival_gap_threshold)
    {
    }

    static double now() noexcept
    {
        return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
    }

    // Producer side (the device's reader thread) -------------------------------------------------

    void push(OwnedData &&data)
    {
        const auto at = now();
        bool wake = false;
        {
            std::lock_guard lock(mutex);
            if (last_arrival) {
                const auto gap = at - *last_arrival;
                if (gap > stats.max_arrival_gap) stats.max_arrival_gap = gap;
                if (gap > arrival_gap_threshold) ++stats.arrival_gaps;
            }
            last_arrival = at;
            if (error) return;  // after an error nothing more is queued (the reader raises first)
            if (queued_bytes + data.length > max_queue_bytes) {
                error = "RHX host stream queue overflow";
            } else {
                stats.buffers += 1;
                stats.bytes += data.length;
                if (data.length > stats.max_buffer_bytes) stats.max_buffer_bytes = data.length;
                queued_bytes += data.length;
                items.push_back(Item{std::move(data), at});
                if (queued_bytes > stats.max_queued_bytes) stats.max_queued_bytes = queued_bytes;
                if (items.size() > stats.max_queued_buffers) stats.max_queued_buffers = items.size();
            }
            // Wake the reader only when it waits and its wait is over: no wake-up per buffer.
            wake = wanted != 0 && (queued_bytes >= wanted || error.has_value());
        }
        if (wake) ready.notify_one();
    }

    void push_error(std::string message)
    {
        {
            std::lock_guard lock(mutex);
            if (!error) error = std::move(message);
        }
        ready.notify_one();
    }

    void push_stop()
    {
        {
            std::lock_guard lock(mutex);
            stopped = true;
        }
        ready.notify_one();
    }

    // The receive callback a device's start_read_stream takes.
    static xdaq::DataStream::receive_callback sink(std::shared_ptr<QueuedReadStream> queue)
    {
        return {[queue = std::move(queue)](xdaq::DataStream::Event &&event) {
            std::visit(
                [&](auto &&event) {
                    using T = std::decay_t<decltype(event)>;
                    namespace Events = xdaq::DataStream::Events;
                    if constexpr (std::is_same_v<T, Events::OwnedData>) {
                        queue->push(std::move(event));
                    } else if constexpr (std::is_same_v<T, Events::DataView>) {
                        OwnedData copy{
                            .buffer = {new unsigned char[event.data.size()],
                                       [](unsigned char *p) { delete[] p; }},
                            .length = event.data.size()};
                        std::memcpy(copy.data(), event.data.data(), event.data.size());
                        queue->push(std::move(copy));
                    } else if constexpr (std::is_same_v<T, Events::Error>) {
                        queue->push_error(std::move(event.error));
                    } else if constexpr (std::is_same_v<T, Events::Stop>) {
                        queue->push_stop();
                    } else {
                        static_assert(xdaq::always_false_v<T>, "non-exhaustive visitor");
                    }
                },
                std::move(event)
            );
        }};
    }

    // Consumer side (Python, with the GIL released) ----------------------------------------------

    // Wait up to `timeout` seconds until at least `min_bytes` (at least one buffer) are queued, or
    // an error or the stop arrives; then take whole buffers, in order, up to `max_bytes` (always at
    // least one), copied into one block. With `min_bytes` not reached by the timeout, whatever is
    // queued is taken. Nothing queued: no batch.
    std::optional<Batch> take(std::size_t max_bytes, std::size_t min_bytes, double timeout)
    {
        std::vector<Item> taken;
        {
            std::unique_lock lock(mutex);
            const auto want = std::max<std::size_t>(min_bytes, 1);
            if (queued_bytes < want && timeout > 0.0 && !error && !stopped) {
                wanted = want;
                ready.wait_for(lock, std::chrono::duration<double>(timeout), [&] {
                    return queued_bytes >= want || error.has_value() || stopped;
                });
                wanted = 0;
            }
            std::size_t size = 0;
            while (!items.empty() && (taken.empty() || size + items.front().data.length <= max_bytes)) {
                size += items.front().data.length;
                queued_bytes -= items.front().data.length;
                taken.push_back(std::move(items.front()));
                items.pop_front();
            }
            if (taken.empty()) return std::nullopt;
            stats.taken_bytes += size;
            stats.batches += 1;
        }
        Batch batch;
        for (const auto &item : taken) batch.size += item.data.length;
        batch.data = std::make_unique<unsigned char[]>(batch.size);
        std::size_t offset = 0;
        for (auto &item : taken) {
            std::memcpy(batch.data.get() + offset, item.data.data(), item.data.length);
            offset += item.data.length;
        }
        batch.buffers = taken.size();
        batch.received_at = taken.front().received_at;
        return batch;  // `taken` frees the device buffers here
    }

    std::size_t queued() const
    {
        std::lock_guard lock(mutex);
        return queued_bytes;
    }

    std::size_t queued_count() const
    {
        std::lock_guard lock(mutex);
        return items.size();
    }

    std::optional<std::string> error_message() const
    {
        std::lock_guard lock(mutex);
        return error;
    }

    bool is_stopped() const
    {
        std::lock_guard lock(mutex);
        return stopped;
    }

    Stats snapshot() const
    {
        std::lock_guard lock(mutex);
        return stats;
    }

    // Drop whatever is queued (on close, while the device that owns the buffers' memory is alive).
    void clear()
    {
        std::deque<Item> dropped;
        {
            std::lock_guard lock(mutex);
            dropped.swap(items);
            queued_bytes = 0;
        }
    }

    const std::size_t max_queue_bytes;
    const double arrival_gap_threshold;

private:
    mutable std::mutex mutex;
    std::condition_variable ready;
    std::deque<Item> items;
    std::size_t queued_bytes = 0;
    std::size_t wanted = 0;  // bytes a waiting reader needs; 0: nobody waits
    std::optional<std::string> error;
    bool stopped = false;
    std::optional<double> last_arrival;
    Stats stats;
};

}  // namespace pyxdaq
