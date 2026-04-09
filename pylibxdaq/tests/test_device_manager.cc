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
    MockDataStream(xdaq::DataStream::receive_callback &&callback, std::shared_ptr<void> resource)
        : resource(std::move(resource))
    {
        fmt::println(stderr, "MockDataStream constructed {}", (void *) this);
        data_thread = std::thread{[callback = std::move(callback)]() mutable {
            // Simulate receiving some data events
            for (int i = 0; i < 5; ++i) {
                std::vector<unsigned char> dummy_data(16, static_cast<unsigned char>(i));
                callback(xdaq::DataStream::Events::DataView{.data = dummy_data});
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            // Simulate an error event
            callback(xdaq::DataStream::Events::Error{.error = "Simulated error"});
            // Simulate stream stop
            callback(xdaq::DataStream::Events::Stop{});
        }};
    }

    ~MockDataStream() override
    {
        fmt::println(stderr, "MockDataStream destroyed {}", (void *) this);
        stop();
        wait_stop();
    }

    void stop() override {}

    void wait_stop() override
    {
        if (data_thread.joinable()) {
            data_thread.join();
        }
    }
    std::shared_ptr<void> resource;
    std::thread data_thread;
};

struct MockDevice final : public xdaq::Device {
    std::shared_ptr<void> resource;
    int id;

    explicit MockDevice(int id) : id(id)
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
        return fmt::format(R"({{ }})");
    }

    std::expected<std::string, std::string> get_info() override { return fmt::format(R"({{ }})"); }

    std::unique_ptr<xdaq::Device::DataStream> start_read_stream(
        addr_t addr, xdaq::DataStream::receive_callback &&callback, std::size_t chunk_size
    ) override
    {
        return std::make_unique<MockDataStream>(std::move(callback), resource);
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
        return std::make_unique<ManagedDevice>(
            std::make_unique<MockDevice>(id), std::move(shared_from_this())
        );
    }
};

extern "C" EXPORT xdaq::DeviceManager *create_manager()
{
    spdlog::set_level(spdlog::level::debug);
    return new MockDeviceManager();
}
