# **AI KERNEL ACCURACY & PERFORMANCE BENCHMARK**

[![CI](https://github.com/justaformality/AIKernelAccuracyPerformanceBenchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/justaformality/AIKernelAccuracyPerformanceBenchmark/actions/workflows/ci.yml)

This project is a small C++ benchmark framework I built to better understand how AI numerical operations work under the hood. It focuses on implementing core AI-style kernels from scratch, validating floating point accuracy, and benchmarking runtime performance.

The project currently includes:
- Matrix multiplication
- ReLU activation
- Numerically stable softmax
- NumPy based validation
- CPU runtime benchmarking

---
# **Why I Built This**
A lot of AI frameworks rely heavily on optimized numerical kernels behind the scenes, especially operations like matrix multiplication and activation functions.

Instead of just using existing libraries, I wanted to build some of these operations myself in C++ and understand:
- how floating point computations behave
- how numerical validation works
- how softmax stability is handled
- how runtime changes with larger workloads
- how AI-style kernels are structured internally

---
# **Tech Stack**
- C++17
- Python
- NumPy
- CMake
- Command line development

---
# **Features**
- Matrix multiplication implemented from scratch in C++
- ReLU activation kernel
- Numerically stable softmax implementation
- Floating point validation against NumPy reference outputs
- Absolute and relative error checking
- Runtime benchmarking across different matrix sizes
- Modular C++ project structure using CMake

---
# **Prerequisites**
- A C++17 compiler (GCC, Clang, or MSVC)
- CMake 3.15+
- Python 3 with NumPy (`pip install numpy`), for the accuracy check

---
# **How to Build**
The `build/` folder already exists in the repo, so there's no need to create it:

```bash
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

`CMAKE_BUILD_TYPE=Release` matters. Without it, CMake builds with no optimization and the timings below aren't comparable.

---
# **Run the Benchmark**
Run it from inside `build/`. The benchmark writes its 128x128 `.csv` outputs to the current directory, and `tests/validate.py` reads them from `build/`.

```bash
./ai_benchmark
```

EXAMPLE OUTPUT:
```text
AI Numeric Kernel Benchmark
Kernel: CPU Matrix Multiplication + ReLU

Size: 64x64 | Runtime: 3.06977 ms | Output elements: 4096
Size: 128x128 | Runtime: 20.5759 ms | Output elements: 16384
Size: 256x256 | Runtime: 83.3207 ms | Output elements: 65536
Size: 512x512 | Runtime: 746.832 ms | Output elements: 262144

Saved 128x128 outputs for NumPy validation.
```

Timings depend on the CPU, compiler, and flags. Note your machine when you compare numbers.

---
# **Validate Numerical Accuracy**
From the project root:

```bash
python3 tests/validate.py
```

The script prints the max absolute and max relative error for each kernel, then `Validation PASSED.` or `Validation FAILED`. It exits with code 1 on failure, so CI fails on an accuracy regression.

**How pass/fail is decided:** every output element must satisfy `|cpp - numpy| <= atol + rtol * |numpy|` (`np.allclose`), with per-kernel tolerances defined in `TOLERANCES` in [`tests/validate.py`](tests/validate.py):

| Kernel | rtol | atol | Why |
|---|---|---|---|
| Matrix multiplication | 1e-4 | 1e-4 | Float32 has ~7 significant digits, and the C++ loop sums in a different order than NumPy |
| ReLU | 1e-4 | 1e-4 | Same inputs as matmul |
| Softmax | 1e-4 | 1e-12 | Probabilities go down to ~1e-14, so the absolute floor must be tiny |

The "max relative error" line can still look large for matmul. That comes from elements whose true value is near zero, where relative error isn't meaningful, and the `atol` term covers those.

**A note on an earlier bug:** previous versions reported a softmax max relative error of ~0.5. The kernel was correct. The CSVs were written with 8 fixed decimals (`std::fixed`), which rounds probabilities near 1e-8 by up to 50%. They're now written in scientific notation with `max_digits10` significant digits, which round-trips every float exactly.

---
# **Benchmarking**
The benchmark currently tests runtime performance for:
- 64x64 matrices
- 128x128 matrices
- 256x256 matrices
- 512x512 matrices

This helps analyze how computation time scales as matrix size increases
