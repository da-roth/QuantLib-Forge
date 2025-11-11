# QuantLib-Forge

**Automatic Adjoint Differentiation for QuantLib using Forge**

QuantLib-Forge is an adapter module that integrates the [Forge AAD engine](https://github.com/da-roth/forge) with [QuantLib](https://github.com/lballabio/QuantLib), enabling automatic computation of derivatives (Greeks) for financial instruments.

## Features

- **Zero-Copy Integration**: Swaps QuantLib's `Real` type with Forge's active type (`forge::fdouble`) at compile time
- **Full QuantLib Compatibility**: No modifications to QuantLib source code required
- **High Performance**: JIT-compiled derivatives using AsmJit
- **Simple API**: RAII session wrapper for easy recording, compilation, and execution
- **Production Ready**: Type-safe automatic differentiation for large-scale quantitative finance applications

## Quick Example

```cpp
#include <ql/forge/config.hpp>
#include <ql/forge/session.hpp>
#include <ql/quantlib.hpp>

using namespace QuantLib;

int main() {
    // Create AAD session
    qlforge::Session session;

    // Market inputs
    Real spot = 100.0, rate = 0.05, vol = 0.20;

    // Mark inputs for differentiation
    auto h_spot = session.markInput(spot);
    auto h_rate = session.markInput(rate);
    auto h_vol = session.markInput(vol);

    // Price option using standard QuantLib code
    VanillaOption option = /* ... setup option ... */;
    Real price = option.NPV();

    // Mark output
    auto h_price = session.markOutput(price);

    // Compile and execute
    session.finalize();
    session.setInputValues({100.0, 0.05, 0.20});
    session.execute();

    // Extract price and Greeks in one pass!
    double npv = session.getOutputValue(h_price);
    double delta = session.getAdjoint(h_spot);  // ∂NPV/∂S
    double rho = session.getAdjoint(h_rate);    // ∂NPV/∂r
    double vega = session.getAdjoint(h_vol);    // ∂NPV/∂σ
}
```

## Getting Started

### Prerequisites

- **CMake** 3.22 or newer
- **C++17** compiler (GCC 11+, Clang 13+, or MSVC 19.3+)
- **Git** client
- **Boost** (required by QuantLib)

For Linux:
```bash
sudo apt-get install cmake g++ git libboost-all-dev ninja-build
```

### Repository Clone

Clone the three required repositories:

```bash
mkdir quantlib-forge-integration
cd quantlib-forge-integration

git clone https://github.com/lballabio/QuantLib.git
git clone https://github.com/da-roth/forge.git
git clone https://github.com/da-roth/QuantLib-Forge.git
```

**Recommended folder structure:**
```
quantlib-forge-integration/
├── QuantLib/           # Mainstream QuantLib
├── forge/              # Forge AAD engine
└── QuantLib-Forge/     # This adapter module
```

### Building

The build is driven from the **QuantLib directory** using QuantLib's extension hook (`QL_EXTERNAL_SUBDIRECTORIES`).

#### Option 1: Using CMake User Presets (Easiest)

Copy the user presets into the QuantLib directory:

```bash
cp QuantLib-Forge/CMakeUserPresets.json QuantLib/
cd QuantLib
cmake --preset linux-forge-gcc-release
cmake --build build/linux-forge-gcc-release -j
```

#### Option 2: Manual CMake Configuration

```bash
cd QuantLib

cmake -S . -B build/forge -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DQL_EXTERNAL_SUBDIRECTORIES="../forge;../QuantLib-Forge" \
  -DQL_EXTRA_LINK_LIBRARIES=QuantLibForge \
  -DQL_NULL_AS_FUNCTIONS=ON \
  -DFORGE_ROOT="../forge"

cmake --build build/forge -j
```

### Running Examples

After building, run the European option example:

```bash
./build/forge/QuantLib-Forge/examples/european_option
```

Expected output:
```
==============================================
European Option Pricing with Forge AAD
==============================================

Input parameters:
  Spot price:      100.000000
  Risk-free rate:  0.050000
  Volatility:      0.200000

Compiling computation graph...
Compilation complete.

==============================================
Results:
==============================================
Option Price:    10.450583

Greeks (via AAD):
  Delta (∂V/∂S): 0.637077
  Rho   (∂V/∂r): 45.889421
  Vega  (∂V/∂σ): 39.699387
==============================================
```

### Running Tests

The test suite validates AAD Greeks against finite differences:

```bash
ctest --test-dir build/forge --output-on-failure
```

Or run tests directly:
```bash
./build/forge/QuantLib-Forge/tests/test_vanilla_greeks
```

## How It Works

### 1. Type Substitution

```cpp
// ql/forge/config.hpp
namespace QuantLib {
    using Real = forge::fdouble;  // Swap scalar type
}
```

### 2. Math Interop

All `<cmath>` functions are re-exported to ensure ADL picks Forge overloads:
```cpp
// ql/forge/math.hpp
namespace forge {
    using std::exp; using std::log; using std::sqrt;
    // ... + numeric_limits<forge::fdouble> specialization
}
```

### 3. Session API

The `qlforge::Session` class provides RAII recording/compilation:

- `markInput()` - Mark independent variables
- `markOutput()` - Mark dependent variables (results)
- `finalize()` - Stop recording and JIT-compile
- `execute()` - Run forward + backward passes
- `getAdjoint()` - Extract derivatives

### 4. CMake Integration

QuantLib-Forge uses QuantLib's `QL_EXTERNAL_SUBDIRECTORIES` hook:

```cmake
-DQL_EXTERNAL_SUBDIRECTORIES="../forge;../QuantLib-Forge"
-DQL_EXTRA_LINK_LIBRARIES=QuantLibForge
```

This injects the adapter headers **before** QuantLib compiles, enabling the type swap.

## Available Presets

The `CMakeUserPresets.json` provides several build configurations:

| Preset                          | Description                                      |
|---------------------------------|--------------------------------------------------|
| `linux-forge-gcc-debug`         | Debug build with Forge (GCC)                    |
| `linux-forge-gcc-release`       | Release build with Forge (GCC)                  |
| `linux-forge-clang-debug`       | Debug build with Forge (Clang)                  |
| `linux-forge-clang-release`     | Release build with Forge (Clang)                |
| `linux-noforge-gcc-release`     | Baseline build without AAD (for benchmarking)   |

## Project Structure

```
QuantLib-Forge/
├── ql/forge/
│   ├── config.hpp          # Type substitution (Real = forge::fdouble)
│   ├── math.hpp            # Math interop + numeric_limits
│   ├── traits.hpp          # Type traits and utilities
│   └── session.hpp         # RAII session wrapper
├── examples/
│   ├── european_option.cpp # Basic option pricing + Greeks
│   └── CMakeLists.txt
├── tests/
│   ├── test_vanilla_greeks.cpp  # Validation tests
│   └── CMakeLists.txt
├── cmake/
│   └── QuantLibForgeConfig.cmake.in
├── CMakeLists.txt
├── CMakeUserPresets.json
└── README.md
```

## Advanced Usage

### Marking Curve Nodes

```cpp
Session session;

// Mark all curve nodes as inputs
std::vector<InputHandle> handles;
for (auto& rate : curveRates) {
    handles.push_back(session.markInput(rate));
}

// Price portfolio
Real npv = portfolio.NPV();
session.markOutput(npv);

session.finalize();
session.execute();

// Extract sensitivities to all curve points
for (size_t i = 0; i < handles.size(); ++i) {
    double sens = session.getAdjoint(handles[i]);
    std::cout << "∂NPV/∂rate[" << i << "] = " << sens << "\n";
}
```

### Performance Optimization

- **Kernel Caching**: Compile once, execute many times with different input values
- **Thread Safety**: Each thread should have its own `Session`
- **Graph Optimization**: Forge automatically applies CSE, constant folding, and algebraic simplification

## Benchmarking

Build without AAD to compare performance:

```bash
cmake --preset linux-noforge-gcc-release
cmake --build build/linux-noforge-gcc-release -j
```

This builds QuantLib with `Real = double` and computes Greeks using finite differences.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Linker can't find Forge** | Set `FORGE_ROOT` to your Forge installation path |
| **Real still double** | Ensure adapter headers are included via `QL_EXTERNAL_SUBDIRECTORIES` |
| **No gradients** | Check that `markOutput()` is called before `finalize()` |
| **Compilation errors** | Verify C++17 support and Boost installation |

## Limitations

- **Discontinuities**: Some functions (e.g., `max`, `min`, `floor`) have zero or undefined derivatives at certain points. Use smooth approximations where needed.
- **Control Flow**: AAD records the executed path. Branching on active variables may require special handling.
- **Thread Local State**: Each thread needs its own `GraphRecorder` session.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

QuantLib-Forge is released under the **BSD 3-Clause License** to match QuantLib's permissive licensing.

## See Also

- [QuantLib](https://www.quantlib.org/) - Open-source library for quantitative finance
- [Forge AAD](https://github.com/da-roth/forge) - High-performance automatic differentiation
- [QuantLib-Risks (XAD)](https://xcelerit.com/quantlib-risks) - Similar integration using XAD

## Citation

If you use QuantLib-Forge in academic work, please cite:

```
@software{quantlib_forge,
  title = {QuantLib-Forge: Automatic Differentiation for QuantLib},
  author = {Roth, Daniel},
  year = {2025},
  url = {https://github.com/da-roth/QuantLib-Forge}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/da-roth/QuantLib-Forge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/da-roth/QuantLib-Forge/discussions)
- **QuantLib**: [quantlib-users](https://lists.sourceforge.net/lists/listinfo/quantlib-users) mailing list

---

**Happy AAD Computing! 🚀**
