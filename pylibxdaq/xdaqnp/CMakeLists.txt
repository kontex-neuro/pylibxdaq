# The Neuropixels binding, built as the compiled submodule pylibxdaq.xdaqnp.
#
# Unlike pyxdaq_device this links a second shared library, libxdaqnp, which is installed into the
# parent package directory next to libxdaq_device so there is exactly one copy of each.

find_package(libxdaqnp REQUIRED)

if(UNIX)
    file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/xdaq-runtime-$<CONFIG>.sh"
        CONTENT "export LD_LIBRARY_PATH=\"${XDAQ_ROOT_DIR}/lib/xdaq:$<TARGET_FILE_DIR:xdaq::xdaqnp>:\$LD_LIBRARY_PATH\"\nexport DYLD_LIBRARY_PATH=\"${XDAQ_ROOT_DIR}/lib/xdaq:$<TARGET_FILE_DIR:xdaq::xdaqnp>:\$DYLD_LIBRARY_PATH\"\n"
    )
endif()

if(BUILD_TESTS)
    nanobind_add_module(_test_np_native STABLE_ABI ../tests/test_np_native.cpp)
    target_include_directories(_test_np_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(_test_np_native PRIVATE xdaq::xdaqnp fmt::fmt)
    target_compile_features(_test_np_native PRIVATE cxx_std_23)
    if(WIN32)
        add_custom_command(TARGET _test_np_native POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:_test_np_native>
                $<TARGET_FILE_DIR:_test_np_native>
            COMMAND_EXPAND_LISTS
        )
    endif()
    add_test(NAME np_native
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/check_np_native.py
            $<TARGET_FILE_DIR:_test_np_native>
    )
    set_tests_properties(np_native PROPERTIES TIMEOUT 15)
endif()

# STABLE_ABI: one build works across CPython 3.12+, which matters because the module ships
# alongside prebuilt device manager plugins.
nanobind_add_module(xdaqnp_core
    STABLE_ABI
    xdaqnp_core.cc
    pynp_data.cpp
    pynp_probes.cpp
    pynp_controller.cpp
    stream_read.cpp
)

target_link_libraries(xdaqnp_core PRIVATE
    xdaq::xdaqnp
    xdaq::xdaq_device
    fmt::fmt
    nlohmann_json::nlohmann_json
)
# For pyxdaq_handles.hpp: Controller takes the Device handle bound by the pyxdaq_device
# extension, so both modules must compile against the same definition of it.
target_include_directories(xdaqnp_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(TARGET xdaqnp_core PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(xdaqnp_core PRIVATE cxx_std_23)
# Build into the package-shaped staging directory, next to the pyxdaq_device this module
# imports. See the comment on PYLIBXDAQ_STAGE_DIR in the parent CMakeLists.
set_target_properties(xdaqnp_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PYLIBXDAQ_STAGE_DIR}/xdaqnp"
)

# This module sits one directory below the package root, where libxdaqnp and libxdaq_device live.
if(APPLE)
    set(_rpath_origin "@loader_path/..")
elseif(UNIX)
    set(_rpath_origin "$ORIGIN/..")
endif()

if(DEFINED _rpath_origin)
    set_target_properties(xdaqnp_core PROPERTIES
        INSTALL_RPATH "${_rpath_origin}"
        BUILD_RPATH_USE_ORIGIN ON
    )
endif()

# On Windows there is no rpath; DLLs must live next to the .pyd so that the stub generator can
# import xdaqnp_core out of the build tree. The installed layout is handled below.
if(WIN32)
    file(GLOB _xdaq_core_dlls "${XDAQ_ROOT_DIR}/bin/*.dll")
    add_custom_command(TARGET xdaqnp_core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:xdaqnp_core>
            ${_xdaq_core_dlls}
            $<TARGET_FILE_DIR:xdaqnp_core>
        COMMAND_EXPAND_LISTS
    )
endif()

# Importing xdaqnp_core pulls in pylibxdaq.pyxdaq_device, so the staging directory's parent has
# to be on the path for the `pylibxdaq` namespace package to resolve.
nanobind_add_stub(xdaqnp_core_stub
    MODULE xdaqnp_core
    OUTPUT xdaqnp_core.pyi
    PYTHON_PATH ${PYLIBXDAQ_STAGE_DIR}/xdaqnp ${CMAKE_BINARY_DIR}
    DEPENDS xdaqnp_core pyxdaq_device
)

install(TARGETS xdaqnp_core DESTINATION xdaqnp)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/xdaqnp_core.pyi DESTINATION xdaqnp)

# libxdaqnp itself. Everywhere but Windows it goes to the package root, alongside the
# libxdaq_device that xdaq_install_core() puts there, and the rpath above finds it. Windows has no
# rpath and CPython only searches the extension's own directory, so there it sits next to the .pyd.
libxdaqnp_install_xdaqnp_runtime(
    LIBRARY_DEST .
    RUNTIME_DEST xdaqnp
)
