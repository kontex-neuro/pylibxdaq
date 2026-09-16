#include "pynp.hpp"

using namespace xdaqnp;

NB_MODULE(xdaqnp_core, m)
{
    m.doc() = "Python binding for the XDAQ Neuropixels (libxdaqnp) API";

    // Controller takes a pyxdaq_device.Device. That type is bound in the sibling extension, and
    // nanobind's type registry is per-process, so importing the module here is what makes the
    // type resolvable in the signatures defined below -- and in the generated stub. It has to
    // happen before any binding names it.
    nb::module_::import_("pylibxdaq.pyxdaq_device");

    nb::enum_<probe_opmode_t>(m, "OpMode")
        .value("RECORDING", RECORDING)
        .value("CALIBRATION", CALIBRATION)
        .value("DIGITAL_TEST", DIGITAL_TEST)
        .value("CALIBRATION_TEST", CALIBRATION_TEST);

    nb::enum_<testinputmode_t>(m, "CalMode")
        .value("PIXEL_MODE", PIXEL_MODE)
        .value("CHANNEL_MODE", CHANNEL_MODE)
        .value("NO_TEST_MODE", NO_TEST_MODE)
        .value("ADC_MODE", ADC_MODE);

    nb::enum_<channelreference_t>(m, "ChannelReference")
        .value("EXT_REF", EXT_REF)
        .value("TIP_REF", TIP_REF)
        .value("INT_REF", INT_REF)
        .value("GND_REF", GND_REF)
        .value("NONE_REF", NONE_REF);

    nb::enum_<NP2::Dock>(m, "Dock").value("FRONT", NP2::Dock::Front).value("BACK", NP2::Dock::Back);

    // Dependency order: a type has to be registered before another binding names it, otherwise
    // it appears in docstrings and in the generated stub under its C++ name.
    pynp::bind_data(m);
    pynp::bind_probes(m);
    pynp::bind_headstages(m);
    pynp::bind_controller(m);
}
