cmake_minimum_required(VERSION 3.25)

project(pylibxdaq VERSION 1.0 LANGUAGES CXX)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

if(MSVC)
    add_compile_definitions(NOMINMAX)
    add_compile_definitions(__PRETTY_FUNCTION__=__FUNCSIG__)
    # nanobind temporarily #undef _DEBUG before including Python.h to avoid linking
    # python312_d.lib. This causes corecrt.h to skip declaring _invalid_parameter,
    # which MSVC STL later calls as ::_invalid_parameter in debug builds.
    # Redirect those calls to abort() using the appropriate escape hatch:
    #   old STL : _STL_CALL_ABORT_INSTEAD_OF_INVALID_PARAMETER
    #   new STL : _MSVC_STL_USE_ABORT_AS_DOOM_FUNCTION
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles([=[
        #define _STL_CALL_ABORT_INSTEAD_OF_INVALID_PARAMETER
        #include <yvals.h>
        int main() { return 0; }
    ]=] _HAVE_STL_CALL_ABORT_FLAG)
    if(_HAVE_STL_CALL_ABORT_FLAG)
        add_compile_definitions(_STL_CALL_ABORT_INSTEAD_OF_INVALID_PARAMETER)
    else()
        add_compile_definitions(_MSVC_STL_USE_ABORT_AS_DOOM_FUNCTION)
    endif()
elseif(
    CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
    (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fexperimental-library>)
    # https://gitlab.kitware.com/cmake/cmake/-/issues/25297
    # CMake 3.29
    add_link_options(LINKER:-no_warn_duplicate_libraries)
endif()

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

find_package(fmt REQUIRED)
find_package(Python
    REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule
)
find_package(nanobind REQUIRED)
find_package(spdlog REQUIRED)
find_package(xdaq REQUIRED)
find_package(nlohmann_json)

option(BUILD_TESTS "Build test modules" ON)

nanobind_add_module(pyxdaq_device
    STABLE_ABI
    pyxdaq_device.cc
)

if(APPLE)
    set(_rpath_origin "@loader_path")
elseif(UNIX)
    set(_rpath_origin "$ORIGIN")
endif()

if(DEFINED _rpath_origin)
    set_target_properties(pyxdaq_device PROPERTIES
        INSTALL_RPATH "${_rpath_origin}"
        BUILD_RPATH_USE_ORIGIN ON
    )
endif()

target_link_libraries(pyxdaq_device PRIVATE
    xdaq::xdaq_device
    spdlog::spdlog
)
set_property(TARGET pyxdaq_device PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(pyxdaq_device PRIVATE cxx_std_23)

# On Windows there is no rpath; DLLs must live next to the .pyd so that the
# stub generator (and tests) can import pyxdaq_device.
if(WIN32)
    file(GLOB _xdaq_core_dlls "${XDAQ_ROOT_DIR}/bin/*.dll")
    add_custom_command(TARGET pyxdaq_device POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:pyxdaq_device>
            ${_xdaq_core_dlls}
            $<TARGET_FILE_DIR:pyxdaq_device>
        COMMAND_EXPAND_LISTS
    )
endif()

nanobind_add_stub(pyxdaq_device_stub
    MODULE pyxdaq_device
    OUTPUT pyxdaq_device.pyi
    DEPENDS pyxdaq_device
)

xdaq_install_core(. .)
xdaq_install_device_managers(./managers)
install(TARGETS pyxdaq_device DESTINATION .)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pyxdaq_device.pyi DESTINATION .)

if(BUILD_TESTS)
    add_library(mock_device_manager SHARED tests/test_device_manager.cc)
    target_link_libraries(mock_device_manager PRIVATE
        xdaq::xdaq_device
        spdlog::spdlog
        fmt::fmt
        nlohmann_json::nlohmann_json
    )
    set_property(TARGET mock_device_manager PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET mock_device_manager PROPERTY PREFIX "")
    set_property(TARGET mock_device_manager PROPERTY OUTPUT_NAME "mock_device_manager")
    target_compile_features(mock_device_manager PRIVATE cxx_std_23)
endif()
