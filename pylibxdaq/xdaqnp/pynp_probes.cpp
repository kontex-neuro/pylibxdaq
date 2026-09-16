#include <fstream>

#include "pynp.hpp"

using namespace nb::literals;

namespace pynp
{

void bind_probes(nb::module_ &m)
{
    nb::class_<ProbeInfo>(m, "ProbeInfo")
        .def_ro("serial", &ProbeInfo::serial)
        .def_ro("probe_pn", &ProbeInfo::probe_pn)
        .def_ro("flex_pn", &ProbeInfo::flex_pn)
        .def_ro("version_major", &ProbeInfo::version_major)
        .def_ro("version_minor", &ProbeInfo::version_minor)
        .def("__repr__", [](const ProbeInfo &i) {
            return fmt::format("<ProbeInfo serial={} pn={}>", i.serial, i.probe_pn);
        });

    nb::class_<pynp::NP1Probe>(m, "NP1Probe")
        .def_prop_ro("info", [](pynp::NP1Probe &p) { return p.resolve()->info; })
        .def_prop_ro(
            "stream",
            [](pynp::NP1Probe &p) { return pynp::NP1Stream{p.resolve()->queue}; },
            "Data-plane handle. Holds the packet queue directly, so reading from it neither "
            "blocks nor is blocked by probe configuration."
        )
        .def(
            "set_opmode",
            [](pynp::NP1Probe &p, probe_opmode_t mode) {
                pynp::check(p.resolve()->setOPMODE(mode), "setOPMODE");
            },
            "mode"_a
        )
        .def(
            "set_calmode",
            [](pynp::NP1Probe &p, testinputmode_t mode) {
                pynp::check(p.resolve()->setCALMODE(mode), "setCALMODE");
            },
            "mode"_a
        )
        .def(
            "set_test_signal",
            [](pynp::NP1Probe &p, bool enable) {
                pynp::check(p.resolve()->setTestSignal(enable), "setTestSignal");
            },
            "enable"_a
        )
        .def(
            "set_test_config",
            [](pynp::NP1Probe &p, bool enable) {
                pynp::check(p.resolve()->setTestConfig(enable), "setTestConfig");
            },
            "enable"_a,
            "Set bit 0 of the probe's TEST_CONFIG1 register.\n\n"
            "The meaning of this bit is undocumented; its only known use is the noise BIST, which "
            "sets it before shorting the channel inputs. This is *not* set_test_signal(), which "
            "gates the headstage oscillator."
        )
        .def(
            "select_electrode",
            [](pynp::NP1Probe &p, std::uint16_t channel, std::uint8_t bank, bool exclusive) {
                pynp::check(
                    p.resolve()->selectElectrode(channel, bank, exclusive), "selectElectrode"
                );
            },
            "channel"_a,
            "bank"_a,
            "exclusive"_a,
            "`exclusive` first disconnects the other banks on this channel (IMEC DLL semantics); "
            "false shorts them together."
        )
        .def(
            "set_reference",
            [](pynp::NP1Probe &p,
               std::uint16_t channel,
               channelreference_t reference,
               std::uint8_t int_ref_electrode_bank) {
                pynp::check(
                    p.resolve()->setReference(channel, reference, int_ref_electrode_bank),
                    "setReference"
                );
            },
            "channel"_a,
            "reference"_a,
            "int_ref_electrode_bank"_a = 0
        )
        .def(
            "set_gain",
            [](pynp::NP1Probe &p, std::uint16_t channel, int ap_gain, int lfp_gain) {
                pynp::check(p.resolve()->setGain(channel, ap_gain, lfp_gain), "setGain");
            },
            "channel"_a,
            "ap_gain"_a,
            "lfp_gain"_a
        )
        .def(
            "apply_shift_register_config",
            [](pynp::NP1Probe &p, bool read_check) {
                auto probe = p.resolve();
                nb::gil_scoped_release release;
                auto result = probe->apply_shift_register_config(read_check);
                std::vector<std::string> errors;
                pynp::collect(errors, "electrode", result.electro);
                pynp::collect(errors, "base_odd", result.base_odd);
                pynp::collect(errors, "base_even", result.base_even);
                pynp::raise_if_any(errors, "apply_shift_register_config");
            },
            "read_check"_a = false
        )
        .def(
            "load_gain_calibration",
            [](pynp::NP1Probe &p, const std::string &path) {
                auto probe = p.resolve();
                std::ifstream file(path);
                if (!file) throw std::runtime_error("Cannot open " + path);
                pynp::unwrap(probe->load_gain_calibration(file));
            },
            "path"_a
        )
        .def("__repr__", [](pynp::NP1Probe &p) {
            return fmt::format("<NP1Probe serial={}>", p.resolve()->info.serial);
        });

    // NP2013NP2014 and NP2003NP2004 differ only in shank count, so their bindings are generated
    // from one lambda parameterised on the probe type.
    auto bind_np2_probe = [&m]<typename ProbeT>(const char *name, auto &&bind_specific) {
        using Handle = pynp::NP2Probe<ProbeT>;
        auto cls =
            nb::class_<Handle>(m, name)
                .def_prop_ro("info", [](Handle &p) { return p.resolve_base()->info; })
                .def_prop_ro("dock", [](Handle &p) { return p.dock; })
                .def_prop_ro(
                    "stream",
                    [](Handle &p) { return pynp::NP2Stream{p.resolve_base()->queue}; },
                    "Data-plane handle. Holds the packet queue directly, so reading from it "
                    "neither blocks nor is blocked by probe configuration."
                )
                .def(
                    "set_opmode",
                    [](Handle &p, probe_opmode_t mode) {
                        pynp::check(p.resolve_base()->setOPMODE(mode), "setOPMODE");
                    },
                    "mode"_a
                )
                .def(
                    "set_calmode",
                    [](Handle &p, testinputmode_t mode) {
                        pynp::check(p.resolve_base()->setCALMODE(mode), "setCALMODE");
                    },
                    "mode"_a
                )
                .def(
                    "set_test_config",
                    [](Handle &p, bool enable) {
                        pynp::check(p.resolve_base()->setTestConfig(enable), "setTestConfig");
                    },
                    "enable"_a,
                    "Set bit 0 of the probe's TEST_CONFIG1 register.\n\n"
                    "Both the register address and the bit's meaning are unverified for NP2 -- "
                    "the address is carried over from the NP1 map. Its only known use is the "
                    "noise BIST."
                )
                .def(
                    "set_standby",
                    [](Handle &p, std::uint16_t channel, bool standby) {
                        pynp::check(p.resolve()->setStdb(channel, standby), "setStdb");
                    },
                    "channel"_a,
                    "standby"_a
                )
                .def(
                    "load_gain_calibration",
                    [](Handle &p, const std::string &path) {
                        auto probe = p.resolve_base();
                        std::ifstream file(path);
                        if (!file) throw std::runtime_error("Cannot open " + path);
                        pynp::unwrap(probe->load_gain_calibration(file));
                    },
                    "path"_a
                )
                .def("__repr__", [name](Handle &p) {
                    return fmt::format("<{} serial={}>", name, p.resolve_base()->info.serial);
                });
        bind_specific(cls);
    };

    bind_np2_probe.template operator()<NP2::NP2013NP2014>("NP2013NP2014Probe", [](auto &cls) {
        using Handle = pynp::NP2013Probe;
        cls.def(
               "apply_shift_register_config",
               [](Handle &p, bool read_check) {
                   auto probe = p.resolve();
                   nb::gil_scoped_release release;
                   auto result = probe->apply_shift_register_config(read_check);
                   std::vector<std::string> errors;
                   pynp::collect(errors, "base_even", result.base.even);
                   pynp::collect(errors, "base_odd", result.base.odd);
                   for (std::size_t i = 0; i < result.shanks.size(); ++i)
                       pynp::collect(errors, fmt::format("shank{}", i), result.shanks[i]);
                   pynp::raise_if_any(errors, "apply_shift_register_config");
               },
               "read_check"_a = false
        )
            .def(
                "select_electrode",
                [](Handle &p,
                   std::uint16_t channel,
                   std::uint8_t shank,
                   std::uint8_t bank,
                   bool exclusive) {
                    pynp::check(
                        p.resolve()->selectElectrode(channel, shank, bank, exclusive),
                        "selectElectrode"
                    );
                },
                "channel"_a,
                "shank"_a,
                "bank"_a,
                "exclusive"_a,
                "`exclusive` first disconnects every other shank/bank on this channel (IMEC DLL "
                "semantics); false shorts them together and limits 0xFF to `shank`."
            )
            .def(
                "set_reference",
                [](Handle &p,
                   std::uint16_t channel,
                   std::uint8_t shank,
                   channelreference_t reference) {
                    pynp::check(
                        p.resolve()->setReference(channel, shank, reference), "setReference"
                    );
                },
                "channel"_a,
                "shank"_a,
                "reference"_a
            );
    });

    bind_np2_probe.template operator()<NP2::NP2003NP2004>("NP2003NP2004Probe", [](auto &cls) {
        using Handle = pynp::NP2003Probe;
        cls.def(
               "apply_shift_register_config",
               [](Handle &p, bool read_check) {
                   auto probe = p.resolve();
                   nb::gil_scoped_release release;
                   auto result = probe->apply_shift_register_config(read_check);
                   std::vector<std::string> errors;
                   pynp::collect(errors, "base_even", result.base.even);
                   pynp::collect(errors, "base_odd", result.base.odd);
                   pynp::collect(errors, "shank", result.shank);
                   pynp::raise_if_any(errors, "apply_shift_register_config");
               },
               "read_check"_a = false
        )
            .def(
                "select_electrode",
                [](Handle &p, std::uint16_t channel, std::uint8_t bank, bool exclusive) {
                    pynp::check(
                        p.resolve()->selectElectrode(channel, bank, exclusive), "selectElectrode"
                    );
                },
                "channel"_a,
                "bank"_a,
                "exclusive"_a,
                "`exclusive` first disconnects the other banks on this channel (IMEC DLL "
                "semantics); "
                "false shorts them together."
            )
            .def(
                "set_reference",
                [](Handle &p, std::uint16_t channel, channelreference_t reference) {
                    pynp::check(p.resolve()->setReference(channel, reference), "setReference");
                },
                "channel"_a,
                "reference"_a
            );
    });
}

void bind_headstages(nb::module_ &m)
{
    nb::class_<pynp::NP1Headstage>(m, "NP1Headstage")
        .def_prop_ro("part_number", [](pynp::NP1Headstage &h) { return h.resolve()->part_number; })
        .def_prop_ro(
            "probe",
            [](pynp::NP1Headstage &h) -> std::optional<pynp::NP1Probe> {
                if (!h.resolve()->probe) return std::nullopt;
                return pynp::NP1Probe{h.port, h.control, h.generation};
            },
            "The initialized probe, or None."
        )
        .def(
            "set_led",
            [](pynp::NP1Headstage &h, bool enable) { h.resolve()->setHSLed(enable); },
            "enable"_a
        )
        .def(
            "bist_eeprom",
            [](pynp::NP1Headstage &h) {
                auto headstage = h.resolve();
                nb::gil_scoped_release release;
                pynp::unwrap(headstage->bist_eeprom());
            }
        )
        .def(
            "init_probe",
            [](pynp::NP1Headstage &h) {
                {
                    auto headstage = h.resolve();
                    nb::gil_scoped_release release;
                    if (!headstage->probe) {
                        headstage->probe = pynp::unwrap(NP1::Probe::init(headstage->serializer));
                    }
                }
                return pynp::NP1Probe{h.port, h.control, h.generation};
            },
            "Initialize the probe if needed and return a handle to it."
        );

    nb::class_<pynp::NP2Headstage>(m, "NP2Headstage")
        .def_prop_ro("part_number", [](pynp::NP2Headstage &h) { return h.resolve()->part_number; })
        .def(
            "bist_eeprom",
            [](pynp::NP2Headstage &h) {
                auto headstage = h.resolve();
                nb::gil_scoped_release release;
                pynp::unwrap(headstage->bist_eeprom());
            }
        )
        .def(
            "probe",
            [](pynp::NP2Headstage &h, NP2::Dock dock) -> nb::object {
                auto headstage = h.resolve();
                auto *probes = pynp::np2_slot(*headstage, dock);
                if (probes == nullptr) return nb::none();
                return *probes | match{
                                     [&](NP2::NP2013NP2014 &) {
                                         return nb::cast(pynp::NP2013Probe{
                                             h.port, h.control, dock, h.generation
                                         });
                                     },
                                     [&](NP2::NP2003NP2004 &) {
                                         return nb::cast(pynp::NP2003Probe{
                                             h.port, h.control, dock, h.generation
                                         });
                                     }
                                 };
            },
            "dock"_a,
            "The initialized probe on `dock`, or None."
        )
        .def(
            "init_probe",
            [](pynp::NP2Headstage &h, NP2::Dock dock) -> nb::object {
                auto headstage = h.resolve();
                auto &slot = (dock == NP2::Dock::Front) ? headstage->probe_a : headstage->probe_b;
                {
                    nb::gil_scoped_release release;
                    if (!slot) slot = pynp::unwrap(NP2::init(headstage->serializer, dock));
                }
                return *slot | match{
                                   [&](NP2::NP2013NP2014 &) {
                                       return nb::cast(
                                           pynp::NP2013Probe{h.port, h.control, dock, h.generation}
                                       );
                                   },
                                   [&](NP2::NP2003NP2004 &) {
                                       return nb::cast(
                                           pynp::NP2003Probe{h.port, h.control, dock, h.generation}
                                       );
                                   }
                               };
            },
            "dock"_a,
            "Initialize the probe on `dock` if needed and return a handle to it."
        );
}

}  // namespace pynp
