##############################################################################
#
#  CMake support for QuantLib-Forge test suite
#
#  Copyright (C) 2025 The QuantLib-Forge Authors
#
#  Derived from QuantLib-Risks-Cpp / XAD (https://github.com/auto-differentiation/XAD)
#  Original XAD code: Copyright (C) 2010-2024 Xcelerit Computing Ltd.
#
#  SPDX-License-Identifier: AGPL-3.0-or-later
#
##############################################################################

set(QLFORGE_TEST_SOURCES
    utilities_forge.cpp
    quantlibtestsuite_forge.cpp
)

set(QLFORGE_TEST_HEADERS
    utilities_forge.hpp
    toplevelfixture.hpp
)

if(QL_BUILD_TEST_SUITE)
    add_executable(QuantLib-Forge_test_suite ${QLFORGE_TEST_SOURCES} ${QLFORGE_TEST_HEADERS})
    set_target_properties(QuantLib-Forge_test_suite PROPERTIES OUTPUT_NAME "quantlib-forge-test-suite")
    if (NOT Boost_USE_STATIC_LIBS)
        target_compile_definitions(QuantLib-Forge_test_suite PRIVATE BOOST_ALL_DYN_LINK)
    endif()
    target_link_libraries(QuantLib-Forge_test_suite PRIVATE
        ql_library
        Forge::forge
        ${QL_THREAD_LIBRARIES})
    if (QL_INSTALL_TEST_SUITE)
        install(TARGETS QuantLib-Forge_test_suite RUNTIME DESTINATION ${QL_INSTALL_BINDIR})
    endif()
    add_test(NAME QuantLib-Forge_test_suite COMMAND QuantLib-Forge_test_suite --log_level=message)
endif()
