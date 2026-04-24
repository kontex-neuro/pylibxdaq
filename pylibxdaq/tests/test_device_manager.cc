#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <xdaq/device.h>
#include <xdaq/device_manager.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

#ifdef WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

using nlohmann::json;
namespace fs = std::filesystem;

struct MockDataStream final : public xdaq::Device::DataStream {
    MockDataStream(
        xdaq::DataStream::receive_callback &&callback, std::shared_ptr<void> resource,
        double event_rate, uint64_t max_events
    )
        : resource(std::move(resource)), event_rate(event_rate), max_events(max_events)
    {
        fmt::println(stderr, "MockDataStream constructed {}", (void *) this);
        data_thread = std::thread{[this, callback = std::move(callback)]() mutable {
            if (this->event_rate <= 0.0) {
                // Legacy behavior for existing tests
                for (int i = 0; i < 5; ++i) {
                    if (stopped.load(std::memory_order_relaxed)) break;
                    std::vector<unsigned char> dummy_data(16, static_cast<unsigned char>(i));
                    callback(xdaq::DataStream::Events::DataView{.data = dummy_data});
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                if (!stopped.load(std::memory_order_relaxed)) {
                    callback(xdaq::DataStream::Events::Error{.error = "Simulated error"});
                }
                callback(xdaq::DataStream::Events::Stop{});
            } else {
                // Benchmark continuous behavior
                auto start_time = std::chrono::steady_clock::now();
                uint64_t events_sent = 0;

                while (!stopped.load(std::memory_order_relaxed)) {
                    if (this->max_events > 0 && events_sent >= this->max_events) {
                        break;
                    }
                    auto now = std::chrono::steady_clock::now();
                    std::chrono::duration<double> elapsed = now - start_time;
                    double expected_events = elapsed.count() * this->event_rate;

                    if (events_sent >= expected_events) {
                        std::this_thread::yield();
                        continue;
                    }

                    std::vector<unsigned char> dummy_data(
                        16, static_cast<unsigned char>(events_sent % 256)
                    );
                    callback(xdaq::DataStream::Events::DataView{.data = dummy_data});
                    events_sent++;
                }
                callback(xdaq::DataStream::Events::Stop{});
            }
        }};
    }

    ~MockDataStream() override
    {
        fmt::println(stderr, "MockDataStream destroyed {}", (void *) this);
        stop();
        wait_stop();
    }

    void stop() override { stopped.store(true, std::memory_order_relaxed); }

    void wait_stop() override
    {
        if (data_thread.joinable()) {
            data_thread.join();
        }
    }
    std::shared_ptr<void> resource;
    std::thread data_thread;
    std::atomic<bool> stopped{false};
    double event_rate;
    uint64_t max_events;
};

struct MockDevice final : public xdaq::Device {
    std::shared_ptr<void> resource;
    int id;
    double event_rate;
    uint64_t max_events;

    explicit MockDevice(int id, double event_rate, uint64_t max_events)
        : id(id), event_rate(event_rate), max_events(max_events)
    {
        fmt::println(stderr, "MockDevice({}) constructed {}", id, (void *) this);
    }

    ~MockDevice() override
    {
        fmt::println(stderr, "MockDevice({}) destroyed {}", id, (void *) this);
    }

    return_t set_register(addr_t addr, value_t value, value_t mask) noexcept override
    {
        return return_t::Success;
    }
    return_t send_registers() noexcept override { return return_t::Success; }

    value_t get_register(addr_t addr) noexcept override { return 0x12345678; }

    return_t read_registers() noexcept override { return return_t::Success; }

    return_t trigger(addr_t addr, int bit) noexcept override { return return_t::Success; }

    std::size_t write(addr_t addr, std::size_t length, const unsigned char *data) noexcept override
    {
        return length;
    }

    std::size_t read(addr_t addr, std::size_t length, unsigned char *data) noexcept override
    {
        return length;
    }

    std::expected<std::string, std::string> get_status() override
    {
        return fmt::format(R"({{"API": "mock-1.0"}})");
    }

    std::expected<std::string, std::string> get_info() override
    {
        return fmt::format(R"({{"Serial Number": "MOCK{:03d}"}})", id);
    }

    std::unique_ptr<xdaq::Device::DataStream> start_read_stream(
        addr_t addr, xdaq::DataStream::receive_callback &&callback, std::size_t chunk_size
    ) override
    {
        return std::make_unique<MockDataStream>(
            std::move(callback), resource, event_rate, max_events
        );
    }
};

static inline std::atomic<int> instance_count{0};
struct MockDeviceManager final : public xdaq::DeviceManager {
    MockDeviceManager()
    {
        instance_count++;
        fmt::println(
            stderr,
            "MockDeviceManager constructed {} (total instances: {})",
            (void *) this,
            instance_count.load()
        );
    }

    ~MockDeviceManager() override
    {
        instance_count--;
        fmt::println(
            stderr,
            "MockDeviceManager destroyed {} (total instances: {})",
            (void *) this,
            instance_count.load()
        );
    }

    std::string info() const noexcept override
    {
        return R"({
            "name": "Mock Device Manager",
            "description": "Test manager for lifetime tracking"
        })";
    }

    fs::path location() const override { return fs::path(__FILE__).parent_path(); }

    std::string list_devices() const override { return R"([{"id": 0}, {"id": 1}, {"id": 2}])"; }

    std::string get_device_options() const noexcept override
    {
        return R"({
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "default": 0
                }
            },
            "required": ["id"]
        })";
    }

    std::unique_ptr<xdaq::Device> create_device(const std::string &config) override
    {
        auto cfg = json::parse(config);
        int id = cfg.value("id", 0);
        double event_rate = cfg.value("event_rate", -1.0);  // -1.0 for backwards compatibility
        uint64_t max_events = cfg.value("max_events", 0);
        return std::make_unique<ManagedDevice>(
            std::make_unique<MockDevice>(id, event_rate, max_events),
            std::move(shared_from_this()),
            config
        );
    }
};

extern "C" EXPORT xdaq::DeviceManager *create_manager()
{
    spdlog::set_level(spdlog::level::info);
    return new MockDeviceManager();
}
