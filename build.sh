#!/bin/sh
set -e

# Default build type
BUILD_TYPE_INPUT=${1:-Release}
BUILD_TYPE_LOWER=$(echo "$BUILD_TYPE_INPUT" | tr '[:upper:]' '[:lower:]')

if [ "$BUILD_TYPE_LOWER" = "debug" ]; then
    PRESET="conan-debug"
    BUILD_TYPE="Debug"
    Sanitizer=OFF
elif [ "$BUILD_TYPE_LOWER" = "release" ]; then
    PRESET="conan-release"
    BUILD_TYPE="Release"
    Sanitizer=OFF
else
    echo "Error: Unrecognized build type '$BUILD_TYPE_INPUT'. Use 'Debug' or 'Release'."
    exit 1
fi
mkdir -p build
find build -mindepth 1 -maxdepth 1 -type d -name 'bdist.*' -exec rm -rf -- {} +
rm -rf build/lib
rm -rf dist
rm -rf pylibxdaq.egg-info
find pylibxdaq -maxdepth 1 -type f -name '*.so' -delete
find pylibxdaq/managers -maxdepth 1 -type f -name '*.so' -delete
rm -rf pylibxdaq/resources

cd pylibxdaq

BUILD_DIR="../build/$BUILD_TYPE"

conan install . --build=missing --profile:all default -s build_type=$BUILD_TYPE -of ..
cmake -S . -B $BUILD_DIR --preset $PRESET -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX="../pylibxdaq"
. "$BUILD_DIR/xdaq-runtime-$BUILD_TYPE.sh"
cmake --build $BUILD_DIR --preset $PRESET --target install
