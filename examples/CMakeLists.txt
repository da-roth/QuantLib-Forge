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

# Set C++ standard
set_target_properties(european_option PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add test (optional)
add_test(NAME european_option_test
    COMMAND european_option
)
