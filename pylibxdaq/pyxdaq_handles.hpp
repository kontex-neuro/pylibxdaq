#pragma once

// Handle types bound in the pyxdaq_device extension but named by other extension modules in
// this package (currently pylibxdaq.xdaqnp.xdaqnp_core). Effectively a binary interface between
// the two: changing a layout below means rebuilding both.
//
// nanobind's type registry is per-process, so a class bound in one module can appear in
// another's signatures, provided:
//
//   * Both modules are built in this tree against the same nanobind, with the same ABI flags.
//   * Both link the same libxdaq_device, so `xdaq::Device` has one RTTI identity across them.
//   * The consuming module imports pylibxdaq.pyxdaq_device *before* defining any binding that
//     names a type from this header, otherwise the signature renders under its C++ name.

#include <nanobind/nanobind.h>
#include <xdaq/device.h>

#include <memory>
#include <utility>

namespace nb = nanobind;

namespace pyxdaq
{

/// An open xdaq::Device, bound as `pyxdaq_device.Device`.
///
/// The shared_ptr carries a deleter that owns the device manager it came from (see
/// `DeviceManager.create_device`), so holding one of these keeps the plugin loaded; there is no
/// separate manager member to propagate.
struct DeviceHandle {
    std::shared_ptr<xdaq::Device> device;

    explicit DeviceHandle(std::shared_ptr<xdaq::Device> d) : device(std::move(d)) {}

    void check() const
    {
        if (!device) throw nb::value_error("Device is already closed");
    }

    void close() noexcept { device.reset(); }

    bool is_closed() const noexcept { return !device; }
};

}  // namespace pyxdaq
