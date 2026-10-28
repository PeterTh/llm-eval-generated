#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    double* const m = mat.data();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            m[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Portable 4-wide double vector (GCC/Clang vector extension); keeps all
// accumulators in SIMD registers instead of relying on auto-vectorization.
typedef double v4d __attribute__((vector_size(32)));

// Register-blocked micro-kernel: updates a 4 x 8 block of C.
//   a:  pointer to row 0 of the A block (row stride N)
//   bp: packed B panel for these 8 columns (k-major, 8 contiguous values per k)
//   c:  pointer to element (0,0) of the C block (row stride N)
// Accumulators start from the current C values and add A*B terms for k in
// [k0, k1) in ascending order, so each C element sees the same operation
// sequence as the sequential reference.
inline void microKernel4x8(const double* __restrict a, const double* __restrict bp,
                           double* __restrict c, const size_t N, const size_t k0, const size_t k1) {
    double* const c0 = c;
    double* const c1 = c + N;
    double* const c2 = c + 2 * N;
    double* const c3 = c + 3 * N;

    v4d x00, x01, x10, x11, x20, x21, x30, x31;
    std::memcpy(&x00, c0,     sizeof(v4d));
    std::memcpy(&x01, c0 + 4, sizeof(v4d));
    std::memcpy(&x10, c1,     sizeof(v4d));
    std::memcpy(&x11, c1 + 4, sizeof(v4d));
    std::memcpy(&x20, c2,     sizeof(v4d));
    std::memcpy(&x21, c2 + 4, sizeof(v4d));
    std::memcpy(&x30, c3,     sizeof(v4d));
    std::memcpy(&x31, c3 + 4, sizeof(v4d));

    const double* const a0 = a;
    const double* const a1 = a + N;
    const double* const a2 = a + 2 * N;
    const double* const a3 = a + 3 * N;

    for (size_t k = k0; k < k1; ++k) {
        v4d b0, b1;
        std::memcpy(&b0, bp + k * 8,     sizeof(v4d));
        std::memcpy(&b1, bp + k * 8 + 4, sizeof(v4d));

        const v4d s0 = {a0[k], a0[k], a0[k], a0[k]};
        const v4d s1 = {a1[k], a1[k], a1[k], a1[k]};
        const v4d s2 = {a2[k], a2[k], a2[k], a2[k]};
        const v4d s3 = {a3[k], a3[k], a3[k], a3[k]};

        x00 += s0 * b0;  x01 += s0 * b1;
        x10 += s1 * b0;  x11 += s1 * b1;
        x20 += s2 * b0;  x21 += s2 * b1;
        x30 += s3 * b0;  x31 += s3 * b1;
    }

    std::memcpy(c0,     &x00, sizeof(v4d));
    std::memcpy(c0 + 4, &x01, sizeof(v4d));
    std::memcpy(c1,     &x10, sizeof(v4d));
    std::memcpy(c1 + 4, &x11, sizeof(v4d));
    std::memcpy(c2,     &x20, sizeof(v4d));
    std::memcpy(c2 + 4, &x21, sizeof(v4d));
    std::memcpy(c3,     &x30, sizeof(v4d));
    std::memcpy(c3 + 4, &x31, sizeof(v4d));
}

// Scalar edge update for a single row segment [j0, j1), k in [k0, k1), ascending.
inline void edgeUpdate(const double* __restrict a, const double* __restrict b,
                       double* __restrict c, const size_t N,
                       const size_t j0, const size_t j1, const size_t k0, const size_t k1) {
    for (size_t k = k0; k < k1; ++k) {
        const double ak = a[k];
        const double* __restrict brow = b + k * N;
        for (size_t j = j0; j < j1; ++j) {
            c[j] += ak * brow[j];
        }
    }
}

// Parallel, cache-blocked matrix multiply.
// Each C[i][j] is accumulated over k in strictly ascending order starting from 0.0.
// Rows of C are partitioned among threads, so no two threads write the same element.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    constexpr size_t BI = 16;   // rows of C per parallel task
    constexpr size_t BJ = 256;  // columns of C per tile
    constexpr size_t BK = 128;  // inner-dimension tile
    constexpr size_t MR = 4;    // micro-kernel rows
    constexpr size_t NR = 8;    // micro-kernel columns
    static_assert(MR == 4 && NR == 8, "microKernel4x8 requires a 4x8 register block");

    const double* const a = A.data();
    const double* const b = B.data();
    double* const c = C.data();

    // Pack the full-width NR-column panels of B into contiguous k-major storage:
    // panel p holds columns [p*NR, p*NR+NR), laid out as Bp[p*N*NR + k*NR + q].
    // This avoids strided access (and cache-set conflicts) in the micro-kernel.
    const size_t numPanels = N / NR;           // full panels only
    const size_t fullCols = numPanels * NR;    // columns covered by packed panels
    // Left uninitialized on purpose: every element is written by the parallel
    // packing loop below, which also performs first-touch in parallel.
    const std::unique_ptr<double[]> Bp(new double[numPanels * N * NR]);
    double* const bp = Bp.get();

    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < numPanels; ++p) {
        double* const dst = bp + p * N * NR;
        for (size_t k = 0; k < N; ++k) {
            const double* const src = b + k * N + p * NR;
            for (size_t q = 0; q < NR; ++q) {
                dst[k * NR + q] = src[q];
            }
        }
    }

    const size_t numRowBlocks = (N + BI - 1) / BI;

    #pragma omp parallel for schedule(static)
    for (size_t rb = 0; rb < numRowBlocks; ++rb) {
        const size_t i0 = rb * BI;
        const size_t i1 = std::min(i0 + BI, N);

        for (size_t i = i0; i < i1; ++i) {
            std::fill(c + i * N, c + i * N + N, 0.0);
        }

        for (size_t j0 = 0; j0 < N; j0 += BJ) {
            const size_t j1 = std::min(j0 + BJ, N);
            for (size_t k0 = 0; k0 < N; k0 += BK) {
                const size_t k1 = std::min(k0 + BK, N);

                size_t i = i0;
                for (; i + MR <= i1; i += MR) {
                    // Packed full-width panels inside this column tile.
                    for (size_t j = j0; j + NR <= j1; j += NR) {
                        microKernel4x8(a + i * N, bp + (j / NR) * N * NR,
                                       c + i * N + j, N, k0, k1);
                    }
                    // Column remainder (j >= fullCols), read directly from B.
                    const size_t jr = std::max(j0, fullCols);
                    if (jr < j1) {
                        for (size_t r = 0; r < MR; ++r) {
                            edgeUpdate(a + (i + r) * N, b + jr, c + (i + r) * N + jr, N,
                                       0, j1 - jr, k0, k1);
                        }
                    }
                }
                for (; i < i1; ++i) {
                    // Row remainder: scalar, full column tile.
                    edgeUpdate(a + i * N, b + j0, c + i * N + j0, N, 0, j1 - j0, k0, k1);
                }
            }
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
