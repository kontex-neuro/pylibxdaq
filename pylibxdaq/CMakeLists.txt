cmake_minimum_required(VERSION 3.25)

project(pylibxdaq VERSION 1.0 LANGUAGES CXX)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

if(MSVC)
    add_compile_definitions(NOMINMAX)
    add_compile_definitions(__PRETTY_FUNCTION__=__FUNCSIG__)
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
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(xdaq REQUIRED)


pybind11_add_module(pyxdaq_device pyxdaq_device.cc)
target_link_libraries(pyxdaq_device PRIVATE
    xdaq::xdaq_device
    spdlog::spdlog
)
set_property(TARGET pyxdaq_device PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(pyxdaq_device PRIVATE cxx_std_23)

install(DIRECTORY ${XDAQ_DEVICE_MANAGER_DIR} DESTINATION .)
install(DIRECTORY ${XDAQ_RESOURCES_DIR} DESTINATION .)
install(TARGETS pyxdaq_device DESTINATION .)