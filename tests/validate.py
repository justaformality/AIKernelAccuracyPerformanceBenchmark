import sys
from pathlib import Path

import numpy as np

BUILD_DIR = Path("build")


def load_csv(filename):
    return np.loadtxt(BUILD_DIR / filename, delimiter=",", dtype=np.float32)


def max_errors(expected, actual):
    """Max absolute and max relative error (relative floored at 1e-8)."""
    abs_err = np.abs(expected - actual)
    rel_err = abs_err / np.maximum(np.abs(expected), 1e-8)
    return float(np.max(abs_err)), float(np.max(rel_err))


# Element-wise check: |actual - expected| <= atol + rtol * |expected|.
#
# rtol: float32 carries ~7 significant digits (eps ~1.2e-7). The C++ and NumPy
#   results legitimately differ because they sum in a different order (a naive
#   loop vs. NumPy's pairwise/BLAS summation), so allow ~1e-4 relative, roughly
#   1000x float32 eps.
# atol: covers elements whose true value is near zero, where relative error
#   is meaningless.
#   - Matmul/ReLU outputs are O(1-10) (128 products of values in [-1, 1]), so
#     a 1e-4 floor is far below any real signal.
#   - Softmax outputs go down to ~1e-18, so its floor must be tiny. A large
#     atol would let a wrong probability like 1e-8 vs 0 pass unnoticed.
TOLERANCES = {
    "Matrix multiplication": {"rtol": 1e-4, "atol": 1e-4},
    "ReLU": {"rtol": 1e-4, "atol": 1e-4},
    "Softmax": {"rtol": 1e-4, "atol": 1e-12},
}


def within_tolerance(name, expected, actual):
    """Return True if the C++ output `actual` matches NumPy's `expected`
    element-wise, using the per-kernel tolerances above."""
    tol = TOLERANCES[name]
    return bool(np.allclose(actual, expected, rtol=tol["rtol"], atol=tol["atol"]))


def main():
    A = load_csv("A_128.csv")
    B = load_csv("B_128.csv")
    C_cpp = load_csv("C_128_cpp.csv")
    R_cpp = load_csv("R_128_cpp.csv")
    S_cpp = load_csv("S_128_cpp.csv").flatten()

    C_numpy = A @ B
    R_numpy = np.maximum(C_numpy, 0)

    C_flat = C_numpy.flatten()
    S_numpy = np.exp(C_flat - np.max(C_flat))
    S_numpy = S_numpy / np.sum(S_numpy)

    checks = [
        ("Matrix multiplication", C_numpy, C_cpp),
        ("ReLU", R_numpy, R_cpp),
        ("Softmax", S_numpy, S_cpp),
    ]

    print("Floating-Point Accuracy Validation")
    print("----------------------------------")

    all_passed = True
    for name, expected, actual in checks:
        abs_err, rel_err = max_errors(expected, actual)
        print(f"{name} max absolute error: {abs_err:.8e}")
        print(f"{name} max relative error: {rel_err:.8e}")
        if not within_tolerance(name, expected, actual):
            print(f"  -> {name} FAILED")
            all_passed = False

    if all_passed:
        print("\nValidation PASSED.")
        return 0

    print("\nValidation FAILED: error exceeded tolerance.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
