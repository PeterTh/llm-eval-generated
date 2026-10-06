#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <omp.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Blocked, packed, OpenMP-parallel matrix multiplication.
//
// Every element C[i][j] is accumulated in strictly ascending k order starting
// from 0.0, exactly like the original loop as compiled by GCC (-O3
// -march=native): each term is sum = sum + (A[i][k] * B[k][j]) with a rounded
// product, except that for odd N the final term (k = N-1) is a fused
// multiply-add. Results are therefore bitwise identical to the original code.
// ---------------------------------------------------------------------------
namespace {

typedef double v4d __attribute__((vector_size(32)));

constexpr size_t MR = 6;    // micro-tile rows
constexpr size_t NR = 8;    // micro-tile columns (2 x 4 doubles)
constexpr size_t KC = 256;  // k-block: B micro-panel (KC x NR) stays in L1
constexpr size_t MC_MAX = 32 * MR;  // A block (MC x KC) stays in L2
constexpr size_t NC_MAX = 64 * NR;

struct AlignedBuffer {
    double* ptr = nullptr;
    explicit AlignedBuffer(size_t count) {
        size_t bytes = ((count * sizeof(double) + 63) / 64) * 64;
        if (bytes == 0) bytes = 64;
        ptr = static_cast<double*>(std::aligned_alloc(64, bytes));
        if (!ptr) {
            fprintf(stderr, "Allocation failed\n");
            std::exit(1);
        }
    }
    ~AlignedBuffer() { std::free(ptr); }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
};

inline v4d loadv(const double* p) {
    v4d v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
inline void storev(double* p, v4d v) { std::memcpy(p, &v, sizeof(v)); }

// acc + a * b with a separately rounded product (the asm barrier prevents the
// compiler from contracting the multiply and add into an FMA).
inline v4d mulAdd(v4d acc, v4d a, v4d b) {
    v4d p = a * b;
    asm("" : "+x"(p));
    return acc + p;
}

inline v4d fusedMulAdd(v4d acc, v4d a, v4d b) {
    v4d r;
    for (int l = 0; l < 4; ++l) r[l] = std::fma(a[l], b[l], acc[l]);
    return r;
}

// C[0:MR][0:NR] (row stride ldc) = (first ? 0 : C) + Ap * Bp over kc steps.
// Ap: kc x MR (k-major), Bp: kc x NR (k-major). If fuseLast, the last k step
// uses a fused multiply-add.
inline void microKernel(const double* __restrict Ap, const double* __restrict Bp,
                        double* __restrict C, const size_t ldc, const size_t kc,
                        const bool first, const bool fuseLast) {
    v4d c[MR][2];
    if (first) {
        for (size_t r = 0; r < MR; ++r) {
            c[r][0] = v4d{0.0, 0.0, 0.0, 0.0};
            c[r][1] = v4d{0.0, 0.0, 0.0, 0.0};
        }
    } else {
        for (size_t r = 0; r < MR; ++r) {
            c[r][0] = loadv(C + r * ldc);
            c[r][1] = loadv(C + r * ldc + 4);
        }
    }
    const size_t kEnd = fuseLast ? kc - 1 : kc;
    for (size_t k = 0; k < kEnd; ++k) {
        const v4d b0 = loadv(Bp + k * NR);
        const v4d b1 = loadv(Bp + k * NR + 4);
        const double* a = Ap + k * MR;
        #pragma GCC unroll 6
        for (size_t r = 0; r < MR; ++r) {
            const v4d av = v4d{a[r], a[r], a[r], a[r]};
            c[r][0] = mulAdd(c[r][0], av, b0);
            c[r][1] = mulAdd(c[r][1], av, b1);
        }
    }
    if (fuseLast) {
        const size_t k = kc - 1;
        const v4d b0 = loadv(Bp + k * NR);
        const v4d b1 = loadv(Bp + k * NR + 4);
        const double* a = Ap + k * MR;
        for (size_t r = 0; r < MR; ++r) {
            const v4d av = v4d{a[r], a[r], a[r], a[r]};
            c[r][0] = fusedMulAdd(c[r][0], av, b0);
            c[r][1] = fusedMulAdd(c[r][1], av, b1);
        }
    }
    for (size_t r = 0; r < MR; ++r) {
        storev(C + r * ldc, c[r][0]);
        storev(C + r * ldc + 4, c[r][1]);
    }
}

}  // namespace

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    if (N == 0) return;

    const size_t mPanels = (N + MR - 1) / MR;
    const size_t nPanels = (N + NR - 1) / NR;

    // Packed copies: Ap[panel][k][MR], Bp[panel][k][NR], zero-padded edges.
    AlignedBuffer Ap(mPanels * MR * N);
    AlignedBuffer Bp(nPanels * NR * N);

    // Limit the team size for small problems (>= ~4 MFLOP per thread) so that
    // thread wake-up and synchronization do not dominate.
    const double flops = 2.0 * static_cast<double>(N) * N * N;
    const size_t maxThreads = static_cast<size_t>(omp_get_max_threads());
    const size_t nThreads = std::max<size_t>(
        1, std::min(maxThreads, static_cast<size_t>(flops / (1 << 22))));
    // Choose tile sizes so that there are enough tiles for all threads.
    size_t mcPanels = MC_MAX / MR;
    size_t ncPanels = NC_MAX / NR;
    auto numTiles = [&]() {
        return ((mPanels + mcPanels - 1) / mcPanels) * ((nPanels + ncPanels - 1) / ncPanels);
    };
    while (numTiles() < 2 * nThreads && (mcPanels > 2 || ncPanels > 2)) {
        if (ncPanels >= mcPanels && ncPanels > 2) ncPanels /= 2;
        else if (mcPanels > 2) mcPanels /= 2;
        else ncPanels /= 2;
    }
    const size_t mTiles = (mPanels + mcPanels - 1) / mcPanels;
    const size_t nTiles = (nPanels + ncPanels - 1) / ncPanels;

    const double* __restrict a = A.data();
    const double* __restrict b = B.data();
    double* __restrict c = C.data();
    double* __restrict ap = Ap.ptr;
    double* __restrict bp = Bp.ptr;

    #pragma omp parallel num_threads(static_cast<int>(nThreads))
    {
        // Pack A into row panels of height MR.
        #pragma omp for schedule(static) nowait
        for (size_t p = 0; p < mPanels; ++p) {
            double* dst = ap + p * MR * N;
            const size_t i0 = p * MR;
            const size_t rows = std::min(MR, N - i0);
            for (size_t k = 0; k < N; ++k) {
                size_t r = 0;
                for (; r < rows; ++r) dst[k * MR + r] = a[(i0 + r) * N + k];
                for (; r < MR; ++r) dst[k * MR + r] = 0.0;
            }
        }
        // Pack B into column panels of width NR.
        #pragma omp for schedule(static)
        for (size_t p = 0; p < nPanels; ++p) {
            double* dst = bp + p * NR * N;
            const size_t j0 = p * NR;
            const size_t cols = std::min(NR, N - j0);
            for (size_t k = 0; k < N; ++k) {
                size_t cc = 0;
                for (; cc < cols; ++cc) dst[k * NR + cc] = b[k * N + j0 + cc];
                for (; cc < NR; ++cc) dst[k * NR + cc] = 0.0;
            }
        }
        // implicit barrier: packing done

        alignas(64) double edge[MR * NR];

        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (size_t mt = 0; mt < mTiles; ++mt) {
            for (size_t nt = 0; nt < nTiles; ++nt) {
                const size_t pBeg = mt * mcPanels;
                const size_t pEnd = std::min(mPanels, pBeg + mcPanels);
                const size_t qBeg = nt * ncPanels;
                const size_t qEnd = std::min(nPanels, qBeg + ncPanels);

                for (size_t k0 = 0; k0 < N; k0 += KC) {
                    const size_t kc = std::min(KC, N - k0);
                    const bool first = (k0 == 0);
                    const bool fuseLast = (N % 2 == 1) && (k0 + kc == N);
                    for (size_t q = qBeg; q < qEnd; ++q) {
                        const double* bPanel = bp + q * NR * N + k0 * NR;
                        const size_t j0 = q * NR;
                        const size_t cols = std::min(NR, N - j0);
                        for (size_t p = pBeg; p < pEnd; ++p) {
                            const double* aPanel = ap + p * MR * N + k0 * MR;
                            const size_t i0 = p * MR;
                            const size_t rows = std::min(MR, N - i0);
                            double* cTile = c + i0 * N + j0;
                            if (rows == MR && cols == NR) {
                                microKernel(aPanel, bPanel, cTile, N, kc, first, fuseLast);
                            } else {
                                if (!first) {
                                    for (size_t r = 0; r < rows; ++r)
                                        for (size_t cc = 0; cc < cols; ++cc)
                                            edge[r * NR + cc] = cTile[r * N + cc];
                                }
                                microKernel(aPanel, bPanel, edge, NR, kc, first, fuseLast);
                                for (size_t r = 0; r < rows; ++r)
                                    for (size_t cc = 0; cc < cols; ++cc)
                                        cTile[r * N + cc] = edge[r * NR + cc];
                            }
                        }
                    }
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
