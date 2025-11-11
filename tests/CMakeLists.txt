# QuantLib-Forge Tests

# Find QuantLib
find_package(QuantLib REQUIRED)

# Test: Vanilla Greeks
add_executable(test_vanilla_greeks test_vanilla_greeks.cpp)
target_link_libraries(test_vanilla_greeks
    PRIVATE
        QuantLib::QuantLib
        QuantLibForge
)

set_target_properties(test_vanilla_greeks PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Register with CTest
add_test(NAME test_vanilla_greeks
    COMMAND test_vanilla_greeks
)

# Set test properties
set_tests_properties(test_vanilla_greeks PROPERTIES
    TIMEOUT 60
    LABELS "AAD;Greeks;Validation"
)
