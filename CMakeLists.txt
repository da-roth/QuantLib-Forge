# QuantLib-Forge CMake Configuration
# Adapter module to integrate Forge AAD engine with QuantLib

cmake_minimum_required(VERSION 3.22)
project(QuantLib-Forge VERSION 0.1.0 LANGUAGES CXX)

# C++17 is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Build Options
# ============================================================================

option(QLFORGE_BUILD_EXAMPLES "Build QuantLib-Forge examples" ON)
option(QLFORGE_BUILD_TESTS "Build QuantLib-Forge tests" ON)
option(QLFORGE_DISABLE_AAD "Disable AAD (use regular double, for benchmarking)" OFF)

# ============================================================================
# Find or configure Forge
# ============================================================================

# Forge can be provided in several ways:
# 1. As part of QL_EXTERNAL_SUBDIRECTORIES (when building with QuantLib)
# 2. Via FORGE_ROOT CMake variable
# 3. As an installed package

if(NOT TARGET forge)
    if(DEFINED FORGE_ROOT)
        message(STATUS "Using Forge from FORGE_ROOT: ${FORGE_ROOT}")
        # Add Forge as subdirectory if it exists
        if(EXISTS "${FORGE_ROOT}/CMakeLists.txt")
            add_subdirectory("${FORGE_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/forge-build")
        else()
            # Assume Forge is pre-built, add include/lib paths
            include_directories("${FORGE_ROOT}/include")
            link_directories("${FORGE_ROOT}/lib")
        endif()
    else()
        # Try to find Forge as a package
        find_package(Forge QUIET)
        if(NOT Forge_FOUND)
            message(STATUS "Forge not found. Assuming it will be provided via QL_EXTERNAL_SUBDIRECTORIES")
        endif()
    endif()
endif()

# ============================================================================
# QuantLib-Forge Interface Library
# ============================================================================

# This is an INTERFACE library that injects the adapter headers into QuantLib's build
add_library(QuantLibForge INTERFACE)

# Add include directories for the adapter headers
target_include_directories(QuantLibForge INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link to Forge if available as a target
if(TARGET forge)
    message(STATUS "QuantLib-Forge: Linking to forge target")
    target_link_libraries(QuantLibForge INTERFACE forge)
elseif(Forge_FOUND)
    message(STATUS "QuantLib-Forge: Linking to Forge package")
    target_link_libraries(QuantLibForge INTERFACE Forge::forge)
else()
    # Fallback: assume forge library is available
    message(STATUS "QuantLib-Forge: Using forge library (fallback)")
    target_link_libraries(QuantLibForge INTERFACE forge)
endif()

# Compile definitions
target_compile_definitions(QuantLibForge INTERFACE
    QL_NULL_AS_FUNCTIONS=1  # QuantLib compatibility for overloaded types
)

if(QLFORGE_DISABLE_AAD)
    target_compile_definitions(QuantLibForge INTERFACE QLFORGE_DISABLE_AAD)
    message(STATUS "QuantLib-Forge: AAD DISABLED (using regular double)")
endif()

# ============================================================================
# Examples (only when configured standalone)
# ============================================================================

if(QLFORGE_BUILD_EXAMPLES AND PROJECT_IS_TOP_LEVEL)
    message(STATUS "QuantLib-Forge: Building examples")
    add_subdirectory(examples)
endif()

# ============================================================================
# Tests
# ============================================================================

if(QLFORGE_BUILD_TESTS)
    enable_testing()
    message(STATUS "QuantLib-Forge: Building tests")
    add_subdirectory(tests)
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS QuantLibForge
    EXPORT QuantLibForgeTargets
    INCLUDES DESTINATION include
)

install(DIRECTORY ql/
    DESTINATION include/ql
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT QuantLibForgeTargets
    FILE QuantLibForgeTargets.cmake
    NAMESPACE QuantLibForge::
    DESTINATION lib/cmake/QuantLibForge
)

# Generate and install package config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/QuantLibForgeConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/QuantLibForgeConfig.cmake"
    INSTALL_DESTINATION lib/cmake/QuantLibForge
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/QuantLibForgeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/QuantLibForgeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/QuantLibForgeConfigVersion.cmake"
    DESTINATION lib/cmake/QuantLibForge
)

# ============================================================================
# Print Configuration Summary
# ============================================================================

message(STATUS "")
message(STATUS "QuantLib-Forge Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Examples: ${QLFORGE_BUILD_EXAMPLES}")
message(STATUS "  Build Tests: ${QLFORGE_BUILD_TESTS}")
message(STATUS "  AAD Enabled: ${NOT QLFORGE_DISABLE_AAD}")
message(STATUS "")
