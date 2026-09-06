# =============================================================================
#  h3-components/CMakeLists.txt -- build the H3 memory system components
# =============================================================================
#  WHAT THIS BUILDS
#      libh3components.a -- the H3 address router, latency hiding buffer, LLM
#      prefetch scheduler, and the memory backend that ties them together.
#
#  HOW IT IS CONSUMED
#      GPGPU-Sim's src/gpgpu-sim/CMakeLists.txt links it after the patch in
#      patches/gpgpu-sim-h3-backend.patch is applied:
#
#          add_subdirectory(${H3_COMPONENTS_DIR} h3-components)
#          target_link_libraries(gpgpusim PUBLIC h3components)
#          target_include_directories(gpgpusim PUBLIC ${H3_COMPONENTS_DIR})
#
#      scripts/apply_gpgpu_sim_patch.sh performs that wiring.
#
#  DELIBERATE PROPERTY: NO SIMULATOR DEPENDENCIES
#      Nothing here includes a GPGPU-Sim or Ramulator header. The whole library
#      compiles with a bare C++17 toolchain, which is what lets
#      scripts/run_unit_tests.sh verify H3 logic in seconds instead of after a
#      30-minute simulator build. Do not add such a dependency: use the
#      IH3MemoryDevice / IMemoryBackend / ILhbFillEngine interfaces instead.
#
#  OPTIONS
#      H3_WITH_RAMULATOR   ON  -> compile the Ramulator-backed memory device;
#                                 requires the ramulator target to exist.
#                          OFF -> analytic devices only (default).
#      H3_BUILD_TESTS      ON  -> build the standalone unit tests.
# =============================================================================
cmake_minimum_required(VERSION 3.17)

project(h3-components
        VERSION 1.0.0
        DESCRIPTION "H3 hybrid HBM+HBF memory system components"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Required: this library is linked into libgpgpusim.a, which ends up in the
# shared libcudart.so that GPGPU-Sim builds.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(H3_WITH_RAMULATOR "Use Ramulator 2.1 device models instead of the analytic ones" OFF)
option(H3_BUILD_TESTS "Build the standalone H3 unit tests" OFF)

set(H3_SOURCES
    h3_address_router.cpp
    latency_hiding_buffer.cpp
    llm_prefetch_scheduler.cpp
    h3_memory_backend.cpp)

add_library(h3components STATIC ${H3_SOURCES})

target_include_directories(h3components PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Warnings are errors here but NOT in GPGPU-Sim: this is new code with no
# legacy to accommodate, and a warning in the memory model is usually a bug.
target_compile_options(h3components PRIVATE -Wall -Wextra)

# -----------------------------------------------------------------------------
# Optional Ramulator device models
# -----------------------------------------------------------------------------
if(H3_WITH_RAMULATOR)
    if(NOT TARGET ramulator)
        set(H3_RAMULATOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ramulator2"
            CACHE PATH "Path to the ramulator2 checkout")
        if(NOT EXISTS "${H3_RAMULATOR_DIR}/CMakeLists.txt")
            message(FATAL_ERROR
                "H3_WITH_RAMULATOR=ON but no Ramulator found at ${H3_RAMULATOR_DIR}. "
                "Run 'git submodule update --init --recursive', or set "
                "-DH3_RAMULATOR_DIR=<path>.")
        endif()
        # Ramulator's Python bindings need nanobind and are not needed here.
        set(RAMULATOR_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
        add_subdirectory(${H3_RAMULATOR_DIR} ramulator2)
    endif()

    target_sources(h3components PRIVATE h3_ramulator_device.cpp)
    target_compile_definitions(h3components PUBLIC H3_WITH_RAMULATOR)
    target_link_libraries(h3components PUBLIC ramulator)

    # RUNTIME LOADER PATH.
    # Ramulator builds libramulator.so into its own SOURCE directory
    # (LIBRARY_OUTPUT_DIRECTORY = PROJECT_SOURCE_DIR), which is on no standard
    # search path. CMake gives build-tree binaries an RPATH automatically, but
    # accel-sim.out is INSTALLED to bin/release, and installing strips it --
    # so the installed binary fails at startup with
    #     error while loading shared libraries: libramulator.so
    #
    # INTERFACE_LINK_OPTIONS propagates transitively, so anything that links
    # h3components (libgpgpusim.a -> accel-sim.out) gets the RPATH without
    # patching Accel-Sim's own CMakeLists.
    get_target_property(_h3_ram_src ramulator SOURCE_DIR)
    if(_h3_ram_src)
        set_property(TARGET h3components APPEND PROPERTY
                     INTERFACE_LINK_OPTIONS "-Wl,-rpath,${_h3_ram_src}")
        message(STATUS "H3: embedding RPATH for libramulator.so -> ${_h3_ram_src}")
    endif()
    # Ramulator is C++20; this library is C++17. They interoperate across the
    # IH3MemoryDevice boundary, but the Ramulator-facing translation unit must
    # be compiled as C++20.
    set_source_files_properties(h3_ramulator_device.cpp
        PROPERTIES COMPILE_OPTIONS "-std=c++20")
    message(STATUS "H3: Ramulator device models ENABLED")
else()
    message(STATUS "H3: using analytic device models (H3_WITH_RAMULATOR=OFF)")
endif()

# -----------------------------------------------------------------------------
# Standalone unit tests
# -----------------------------------------------------------------------------
# These need no simulator and no Ramulator. scripts/run_unit_tests.sh builds
# them directly with g++ and does not require CMake at all; this target exists
# so they can also run under CTest in CI.
if(H3_BUILD_TESTS)
    enable_testing()
    foreach(test_name test_h3_router test_lhb test_llm_scheduler test_macro_hazard test_h3_backend)
        add_executable(${test_name} ../tests/${test_name}.cpp)
        target_link_libraries(${test_name} PRIVATE h3components)
        add_test(NAME ${test_name}
                 COMMAND ${test_name}
                 # Tests load configs/*.yaml by relative path.
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
    endforeach()
endif()
