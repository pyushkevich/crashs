# This CMakeLists.txt does not compile anything itself. The actual Java -> native-image build
# happens out-of-band, driven by scripts/build_native.sh, invoked via CIBW_BEFORE_ALL before
# scikit-build-core's configure/build/install steps run (see ../pyproject.toml). This file's only
# job is to stage that already-built artifact plus the LUT data into the wheel, and to give
# scikit-build-core a CMake project to drive so it can platform-tag the resulting wheel correctly.
cmake_minimum_required(VERSION 3.20)
project(crashs_native LANGUAGES NONE)

if(NOT DEFINED CBSTOOLS_NATIVE_BUILD_DIR OR CBSTOOLS_NATIVE_BUILD_DIR STREQUAL "")
    set(CBSTOOLS_NATIVE_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
endif()

# Fail the build loudly if the native library hasn't been built yet, rather than
# silently producing a wheel/install with no _native_lib at all (install(DIRECTORY)
# on a missing or non-matching source directory is a silent no-op in CMake).
file(GLOB _cbstools_native_lib_files
    "${CBSTOOLS_NATIVE_BUILD_DIR}/out/*.so"
    "${CBSTOOLS_NATIVE_BUILD_DIR}/out/*.dylib"
    "${CBSTOOLS_NATIVE_BUILD_DIR}/out/*.dll"
)
if(NOT _cbstools_native_lib_files)
    message(FATAL_ERROR
        "No compiled cbstools native library found under "
        "${CBSTOOLS_NATIVE_BUILD_DIR}/out/ - run native/scripts/build_native.sh first "
        "(or set CBSTOOLS_NATIVE_BUILD_DIR to point at its output).")
endif()

install(
    DIRECTORY "${CBSTOOLS_NATIVE_BUILD_DIR}/out/"
    DESTINATION crashs/_native_lib
    FILES_MATCHING
        PATTERN "*.so"
        PATTERN "*.dylib"
        PATTERN "*.dll"
)

install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/topology_lut/"
    DESTINATION crashs/_native_data/topology_lut
)
