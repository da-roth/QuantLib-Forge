# QuantLib-Forge Examples

# Find QuantLib (only if not already available as a target)
if(NOT TARGET QuantLib)
    find_package(QuantLib REQUIRED)
endif()

# Determine the correct QuantLib target name
if(TARGET QuantLib)
    set(QL_TARGET QuantLib)
else()
    set(QL_TARGET QuantLib::QuantLib)
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
