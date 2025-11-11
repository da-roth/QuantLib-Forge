# QuantLib-Forge Examples

# Determine the correct QuantLib target name
# When building as part of QuantLib (via QL_EXTERNAL_SUBDIRECTORIES),
# the target is ql_library. When building standalone, we need to find QuantLib.
# Check if we're being built as part of QuantLib by looking for QL_EXTERNAL_SUBDIRECTORIES
if(DEFINED QL_EXTERNAL_SUBDIRECTORIES)
    # Building as part of QuantLib - use ql_library target
    # Note: ql_library will be defined later in the CMake run
    set(QL_TARGET ql_library)
    message(STATUS "QuantLib-Forge Examples: Using ql_library (integrated build)")
elseif(TARGET QuantLib)
    # QuantLib already found
    set(QL_TARGET QuantLib)
    message(STATUS "QuantLib-Forge Examples: Using QuantLib target")
else()
    # Standalone build - need to find QuantLib
    find_package(QuantLib REQUIRED)
    set(QL_TARGET QuantLib::QuantLib)
    message(STATUS "QuantLib-Forge Examples: Found QuantLib package")
endif()

# European Option example
add_executable(european_option european_option.cpp)
target_link_libraries(european_option
    PRIVATE
        ${QL_TARGET}
        QuantLibForge
)

# Barrier Option example
add_executable(barrier_option barrier_option.cpp)
target_link_libraries(barrier_option
    PRIVATE
        ${QL_TARGET}
        QuantLibForge
)

# Curve Bootstrapping example
add_executable(curve_bootstrap curve_bootstrap.cpp)
target_link_libraries(curve_bootstrap
    PRIVATE
        ${QL_TARGET}
        QuantLibForge
)

# Adjoint European Equity Option example (matches XAD portfolio example)
add_executable(adjoint_european_equity_option adjoint_european_equity_option.cpp)
target_link_libraries(adjoint_european_equity_option
    PRIVATE
        ${QL_TARGET}
        QuantLibForge
)

# Set C++ standard for all examples
set_target_properties(
    european_option
    barrier_option
    curve_bootstrap
    adjoint_european_equity_option
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

# Add tests (optional)
add_test(NAME european_option_test COMMAND european_option)
add_test(NAME barrier_option_test COMMAND barrier_option)
add_test(NAME curve_bootstrap_test COMMAND curve_bootstrap)
add_test(NAME adjoint_european_equity_option_test COMMAND adjoint_european_equity_option)
