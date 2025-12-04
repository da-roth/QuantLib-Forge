##############################################################################
#
#  quantlib-forge - Forge integration layer for QuantLib
#
#  Copyright (C) 2025 The QuantLib-Forge Authors
#
#  Derived from QuantLib-Risks-Cpp / XAD (https://github.com/auto-differentiation/XAD)
#  Original XAD code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.
#
#  SPDX-License-Identifier: AGPL-3.0-or-later
#
##############################################################################

option(QL_FORGE_DISABLE_AAD "Disable using Forge AAD for QuantLib's Real, allowing to run samples with double" OFF)
option(QL_FORGE_BUILD_TEST_SUITE "Build QuantLib-Forge test suite" ON)

# ========== Forge Integration: Use pre-built Forge package ==========
# Forge is now built and installed separately via forge/tools/packaging
message(STATUS "Looking for Forge package...")
find_package(Forge CONFIG REQUIRED)
message(STATUS "Found Forge package version ${Forge_VERSION}")
# The Forge::forge target is now available and includes:
#   - Core Forge library (types/fdouble, graph recording, JIT compiler)
#   - All necessary dependencies (asmjit, sleef, nlohmann_json)
# ====================================================================

# ========== expressions: Forge expression templates ==========
# expressions provides the expression template layer for QuantLib integration
# It links to Forge::forge for fdouble, graph recording, etc.
add_subdirectory(expressions)
# =================================================================

add_subdirectory(ql)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# TODO: Add examples
# add_subdirectory(Examples)

if(NOT QL_FORGE_DISABLE_AAD AND QL_FORGE_BUILD_TEST_SUITE)
    add_subdirectory(forge-test-suite)
endif()
