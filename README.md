# QuantLib-Forge

Forge integration layer for QuantLib - enabling high-performance automatic differentiation (AAD) for financial derivatives pricing.

## Overview

This project provides:

1. **Integration Layer** - Connects Forge's JIT-compiled AAD with QuantLib's pricing engines
2. **ABool Patch** - Minimal patch for QuantLib to support conditional recording via `ABool`
3. **Benchmark Harness** - Compare performance: Forge AAD vs finite differences vs XAD

## Performance

XVA benchmark: Vanilla swap pricing with 100 risk factors, 3 time steps. The tradeoff — Forge has a one-time kernel creation cost, but faster evaluation:

```
XAD vs Forge-AAD
═══════════════════════════════════════════════════════════════════════════════
  Scenarios       XAD             Forge-AAD
               (0.06 ms/eval)     (22 ms kernel + 0.001 ms/eval)
───────────────────────────────────────────────────────────────────────────────
          1        1.0 ms           22.6 ms
         10       10.3 ms           23.3 ms
        100       93.4 ms           25.4 ms
      1,000        859 ms           42.5 ms
═══════════════════════════════════════════════════════════════════════════════
```

**Notes:**
- Bump-reval could be faster — currently using the same kernel that computes AAD
- AVX vectorization not yet enabled

<details>
<summary>Full benchmark results</summary>

Raw output: [baseline](docs/benchmarkResults/benchmark-baseline.txt) | [forge](docs/benchmarkResults/benchmark-forge.txt) | [xad](docs/benchmarkResults/benchmark-xad.txt)

```
Total Execution Time
═════════════════════════════════════════════════════════════════════════════════════════════════════════════
  Test Case          Scenarios      QuantLib     Forge-Bump        Forge-AAD            XAD
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1 (10 RF, 1 step)          1        2.8 ms         5.6 ms           5.4 ms         0.4 ms
                                        (1×)         (0.5×)           (0.5×)            (7×)

  2 (100 RF, 1 step)         1       25.5 ms         9.0 ms           8.6 ms         0.4 ms
                                        (1×)         (2.8×)             (3×)           (64×)

  3 (100 RF, 3 steps)        1       59.5 ms        24.0 ms          22.6 ms         1.0 ms
                                        (1×)         (2.5×)           (2.6×)           (60×)

  3 (100 RF, 3 steps)       10        585 ms        37.0 ms          23.3 ms        10.3 ms
                                        (1×)          (16×)            (25×)           (57×)

  3 (100 RF, 3 steps)      100      4,289 ms         162 ms          25.4 ms        93.4 ms
                                        (1×)          (27×)           (169×)           (46×)

  3 (100 RF, 3 steps)    1,000     43,159 ms       1,406 ms          42.5 ms         859 ms
                                        (1×)          (31×)         (1,016×)           (50×)
═════════════════════════════════════════════════════════════════════════════════════════════════════════════
```

```
Per-Scenario Timing Breakdown
═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  Test Case          Scenarios      QuantLib       Forge Kernel      Forge-Bump          Forge-AAD           XAD
                                  per Scenario       Creation      per Scenario       per Scenario    per Scenario
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  1 (10 RF, 1 step)          1      0.56 ms           1.1 ms      0.01 ms (100×)         < 0.01 ms       0.07 ms
  2 (100 RF, 1 step)         1      5.10 ms           1.7 ms      0.10 ms (100×)         < 0.01 ms       0.07 ms
  3 (100 RF, 3 steps)        1      3.97 ms           4.6 ms      0.09 ms  (90×)          0.001 ms       0.06 ms
  3 (100 RF, 3 steps)       10      3.90 ms           4.6 ms      0.09 ms  (90×)          0.001 ms       0.06 ms
  3 (100 RF, 3 steps)      100      2.86 ms           4.6 ms      0.09 ms  (90×)          0.001 ms       0.05 ms
  3 (100 RF, 3 steps)    1,000      2.88 ms           4.6 ms      0.09 ms  (90×)          0.001 ms       0.05 ms
═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
```

</details>

## Dependencies

- **[Forge](https://github.com/da-roth/forge)** - JIT compiler and AAD engine (zlib license)
- **[QuantLib](https://github.com/lballabio/QuantLib)** - Financial library (permissive license)
- **Boost** - Required by QuantLib

## Building

```bash
# Clone workspace
git clone <this-repo> quantlib-forge
cd quantlib-forge

# Build Forge package first (see forge/tools/packaging)
# Then configure and build:

cmake -B build -S . \
    -DForge_DIR=<path-to-forge-install>/lib/cmake/Forge \
    -DQuantLib_DIR=<path-to-quantlib-install>/lib/cmake/QuantLib

cmake --build build
```

## Project Structure

```
quantlib-forge/
  LICENSE              # AGPL-3.0-or-later
  NOTICE.md            # Third-party attributions
  CMakeLists.txt       # Main build configuration
  docs/                # Documentation
  patches/             # QuantLib ABool patch
  integration/         # Forge <-> QuantLib adapter code
    ql/                # QuantLib-specific headers
  harness/             # Benchmark harness (engine-agnostic)
  benchmarks/          # Benchmark executables
  examples/            # Usage examples
  cmake/               # CMake modules and presets
```

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for details.

Some code is derived from [QuantLib-Risks-Cpp](https://github.com/auto-differentiation/QuantLib-Risks-Cpp) (Xcelerit Computing Ltd., AGPL-3.0).
