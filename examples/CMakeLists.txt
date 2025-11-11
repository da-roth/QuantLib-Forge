# QuantLib-Forge Examples

# Find QuantLib
find_package(QuantLib REQUIRED)

# European Option example
add_executable(european_option european_option.cpp)
target_link_libraries(european_option
    PRIVATE
        QuantLib::QuantLib
        QuantLibForge
)

# Barrier Option example
add_executable(barrier_option barrier_option.cpp)
target_link_libraries(barrier_option
    PRIVATE
        QuantLib::QuantLib
        QuantLibForge
)

# Curve Bootstrapping example
add_executable(curve_bootstrap curve_bootstrap.cpp)
target_link_libraries(curve_bootstrap
    PRIVATE
        QuantLib::QuantLib
        QuantLibForge
)

# Set C++ standard for all examples
set_target_properties(
    european_option
    barrier_option
    curve_bootstrap
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

# Add tests (optional)
add_test(NAME european_option_test COMMAND european_option)
add_test(NAME barrier_option_test COMMAND barrier_option)
add_test(NAME curve_bootstrap_test COMMAND curve_bootstrap)
