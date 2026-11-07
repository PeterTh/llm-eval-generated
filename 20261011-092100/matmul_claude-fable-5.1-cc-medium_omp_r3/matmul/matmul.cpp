#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    double* __restrict__ m = mat.data();
    // Static schedule over rows matches the row-block distribution used in
    // matrixMultiply, so the first-touch placement of pages is NUMA friendly.
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            m[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Cache blocking parameters (in elements).
//   B tile  : KB x JB doubles = 256 KiB
//   C tile  : IB x JB doubles = 128 KiB
//   A tile  : IB x KB doubles =  64 KiB
// Together they fit in the per-core L2 of modern x86 CPUs, and a JB-wide row
// of C plus a JB-wide row of B comfortably fit in L1.
constexpr size_t IB = 64;
constexpr size_t JB = 256;
constexpr size_t KB = 128;

// Computes C[i0:i1, j0:j1] = A[i0:i1, :] * B[:, j0:j1].
//
// The accumulation into every C element proceeds strictly in increasing k
// order starting from 0.0, i.e. exactly the same sequence of arithmetic
// operations as the reference triple loop.  The result is therefore
// bit-identical to the original implementation, independent of the number
// of threads and of the blocking parameters.
static inline void multiplyTile(const double* __restrict__ A, const double* __restrict__ B,
                                double* __restrict__ C, const size_t N,
                                const size_t i0, const size_t i1,
                                const size_t j0, const size_t j1) {
    const size_t jlen = j1 - j0;

    // C tile is zero initialised (std::vector value-initialisation); writing
    // zeros here makes the kernel independent of the caller's initial contents
    // and keeps "sum = 0.0" semantics.
    for (size_t i = i0; i < i1; ++i) {
        double* __restrict__ c = C + i * N + j0;
        for (size_t j = 0; j < jlen; ++j) {
            c[j] = 0.0;
        }
    }

    // The reference implementation (sum += A[i][k] * B[k][j] for k = 0..N-1)
    // is compiled by GCC into a separately rounded multiply and add for every
    // k, except that a single fused multiply-add is used for the last k when
    // N is odd.  To reproduce the reference values bit for bit, the k loop is
    // split into a non-fused main part [0, kFused) and, for odd N, one fused
    // final term.  Implicit contraction is disabled in CMakeLists.txt
    // (-ffp-contract=off), so "c += a * b" below is never turned into an FMA.
    const size_t kFused = (N % 2 == 1) ? N - 1 : N;

    for (size_t k0 = 0; k0 < kFused; k0 += KB) {
        const size_t k1 = std::min(k0 + KB, kFused);
        for (size_t i = i0; i < i1; ++i) {
            const double* __restrict__ a = A + i * N;
            double* __restrict__ c = C + i * N + j0;
            for (size_t k = k0; k < k1; ++k) {
                const double aik = a[k];
                const double* __restrict__ b = B + k * N + j0;
                // Vectorised rank-1 update of a C row segment.
#pragma omp simd
                for (size_t j = 0; j < jlen; ++j) {
                    c[j] += aik * b[j];
                }
            }
        }
    }

    if (kFused != N) {
        const size_t k = kFused;
        const double* __restrict__ b = B + k * N + j0;
        for (size_t i = i0; i < i1; ++i) {
            const double aik = A[i * N + k];
            double* __restrict__ c = C + i * N + j0;
#pragma omp simd
            for (size_t j = 0; j < jlen; ++j) {
                c[j] = std::fma(aik, b[j], c[j]);
            }
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const double* __restrict__ a = A.data();
    const double* __restrict__ b = B.data();
    double* __restrict__ c = C.data();

    const size_t numIBlocks = (N + IB - 1) / IB;
    const size_t numJBlocks = (N + JB - 1) / JB;

    // Each (i-block, j-block) pair owns a disjoint tile of C, so tiles can be
    // computed fully independently without any synchronisation.  Collapsing
    // both block loops exposes enough parallelism for many-core machines even
    // at moderate N; the static schedule keeps the mapping of C/A rows to
    // threads stable across calls and consistent with first-touch in init.
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t ib = 0; ib < numIBlocks; ++ib) {
        for (size_t jb = 0; jb < numJBlocks; ++jb) {
            const size_t i0 = ib * IB;
            const size_t i1 = std::min(i0 + IB, N);
            const size_t j0 = jb * JB;
            const size_t j1 = std::min(j0 + JB, N);
            multiplyTile(a, b, c, N, i0, i1, j0, j1);
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }
            
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            
            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(A, B, C, N);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
