#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Blocked Cholesky decomposition (OpenMP, shared-memory parallel)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// The matrix is processed in panels of blockSize columns.  For every panel the
// diagonal block is factored, the panel below it is solved with a triangular
// solve, and the trailing submatrix is updated with a symmetric rank-k update.
// The panel solve and the trailing update are the expensive parts and are
// parallelized over rows (each row is independent).  All work-shares live in a
// single persistent parallel region so that a thread team is created only once.

namespace {

constexpr size_t SMALL_BLOCK_SIZE = 64;
constexpr size_t LARGE_BLOCK_SIZE = 128;
constexpr size_t LARGE_BLOCK_THRESHOLD = 1024;
constexpr size_t UPDATE_TILE = 64;
constexpr size_t UPDATE_CHUNK_SMALL = 3;
constexpr size_t UPDATE_CHUNK_LARGE = 6;
constexpr size_t GEN_TILE = 64;
constexpr size_t GEN_KBLOCK = 256;

inline bool factorDiagonalBlock(std::vector<double>& A, const size_t n,
                                const size_t k0, const size_t k1) {
    for (size_t i = k0; i < k1; ++i) {
        const size_t ri = i * n;
        for (size_t j = k0; j <= i; ++j) {
            const size_t rj = j * n;
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (size_t k = k0; k < j; ++k) {
                sum += A[ri + k] * A[rj + k];
            }

            if (i == j) {
                const double val = A[ri + j] - sum;
                if (val <= 0.0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    return false;
                }
                A[ri + j] = sqrt(val);
            } else {
                A[ri + j] = (A[ri + j] - sum) / A[rj + j];
            }
        }
    }
    return true;
}

} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order (only the lower triangle is meaningful).
    const size_t blockSize =
        (n <= LARGE_BLOCK_THRESHOLD) ? SMALL_BLOCK_SIZE : LARGE_BLOCK_SIZE;
    const size_t updateChunk =
        (n <= LARGE_BLOCK_THRESHOLD) ? UPDATE_CHUNK_SMALL : UPDATE_CHUNK_LARGE;
    bool failed = false;

#pragma omp parallel shared(A, n, failed, blockSize, updateChunk)
    {
        for (size_t k0 = 0; k0 < n && !failed; k0 += blockSize) {
            const size_t k1 = std::min(k0 + blockSize, n);

            // Factor the small diagonal block (single thread, other threads wait).
#pragma omp single
            {
                if (!factorDiagonalBlock(A, n, k0, k1)) {
                    failed = true;
                }
            }

            if (failed) {
                break;
            }

            // Solve the panel below the diagonal block:
            //   L(i,j) = (A(i,j) - sum_{p=k0}^{j-1} L(i,p) * L(j,p)) / L(j,j)
#pragma omp for schedule(static)
            for (size_t i = k1; i < n; ++i) {
                const size_t ri = i * n;
                double* __restrict Ai = A.data() + ri;
                for (size_t j = k0; j < k1; ++j) {
                    const double* __restrict Aj = A.data() + j * n;
                    double sum = 0.0;
#pragma omp simd reduction(+ : sum)
                    for (size_t k = k0; k < j; ++k) {
                        sum += Ai[k] * Aj[k];
                    }
                    Ai[j] = (Ai[j] - sum) / Aj[j];
                }
            }

            // Update the trailing submatrix (symmetric rank-k update):
            //   A(i,j) -= sum_{p=k0}^{k1-1} L(i,p) * L(j,p),  j <= i
            // Each task owns a strip of rows, so no two threads write to the
            // same cache lines.  Within a strip the column panel is traversed
            // in tiles and reused for all rows of the strip (cache friendly).
            const size_t nChunks =
                (n - k1 + updateChunk - 1) / updateChunk;
#pragma omp for schedule(dynamic, 1)
            for (size_t ci = 0; ci < nChunks; ++ci) {
                const size_t c = k1 + ci * updateChunk;
                const size_t cend = std::min(c + updateChunk, n);

                for (size_t jb = k1; jb < cend; jb += UPDATE_TILE) {
                    const size_t j1 = std::min(jb + UPDATE_TILE, n);
                    const size_t istart = std::max(c, jb);
                    for (size_t i = istart; i < cend; ++i) {
                        const double* __restrict Ai = A.data() + i * n;
                        const size_t jend = std::min(i + 1, j1);
                        for (size_t j = jb; j < jend; ++j) {
                            const double* __restrict Aj = A.data() + j * n;
                            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
                            for (size_t k = k0; k < k1; ++k) {
                                sum += Ai[k] * Aj[k];
                            }
                            A[i * n + j] -= sum;
                        }
                    }
                }
            }
        }

        if (!failed) {
            // Zero out upper triangular part.
#pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                const size_t ri = i * n;
                for (size_t j = i + 1; j < n; ++j) {
                    A[ri + j] = 0.0;
                }
            }
        }
    }

    return !failed;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (kept sequential so the input is unchanged).
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T.  The lower triangle is produced tile by tile with
    // cache blocking; each tile stays in L1/L2 while the k dimension is walked
    // in blocks.  A is symmetric, so the upper triangle is an exact mirror.
    const size_t nTiles = (n + GEN_TILE - 1) / GEN_TILE;
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (size_t bi = 0; bi < nTiles; ++bi) {
        for (size_t bj = 0; bj < nTiles; ++bj) {
            if (bj > bi) {
                continue;
            }

            const size_t i0 = bi * GEN_TILE;
            const size_t i1 = std::min(i0 + GEN_TILE, n);
            const size_t j0 = bj * GEN_TILE;
            const size_t j1 = std::min(j0 + GEN_TILE, n);
            const size_t ni = i1 - i0;
            const size_t nj = j1 - j0;

            alignas(64) double tile[GEN_TILE][GEN_TILE];
            for (size_t ii = 0; ii < ni; ++ii) {
                for (size_t jj = 0; jj < nj; ++jj) {
                    tile[ii][jj] = 0.0;
                }
            }

            for (size_t kb = 0; kb < n; kb += GEN_KBLOCK) {
                const size_t ke = std::min(kb + GEN_KBLOCK, n);
                for (size_t i = i0; i < i1; ++i) {
                    const double* __restrict Bi = B.data() + i * n;
                    double* __restrict ti = tile[i - i0];
                    for (size_t j = j0; j < j1; ++j) {
                        const double* __restrict Bj = B.data() + j * n;
                        double sum = ti[j - j0];
#pragma omp simd reduction(+ : sum)
                        for (size_t k = kb; k < ke; ++k) {
                            sum += Bi[k] * Bj[k];
                        }
                        ti[j - j0] = sum;
                    }
                }
            }

            if (bi == bj) {
                for (size_t i = i0; i < i1; ++i) {
                    for (size_t j = j0; j < j1; ++j) {
                        A[i * n + j] = tile[i - i0][j - j0];
                    }
                }
            } else {
                for (size_t i = i0; i < i1; ++i) {
                    for (size_t j = j0; j < j1; ++j) {
                        const double val = tile[i - i0][j - j0];
                        A[i * n + j] = val;
                        A[j * n + i] = val;
                    }
                }
            }
        }
    }

    // Add diagonal dominance to ensure positive definiteness
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with the original matrix.
    // L * L^T is symmetric, so only the lower triangular tiles are computed;
    // each such tile is compared against both the lower and mirrored upper
    // entries of the (symmetric) original matrix.
    const size_t nTiles = (n + GEN_TILE - 1) / GEN_TILE;
    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for collapse(2) schedule(dynamic, 1) \
    reduction(max : maxError, relError)
    for (size_t bi = 0; bi < nTiles; ++bi) {
        for (size_t bj = 0; bj < nTiles; ++bj) {
            if (bj > bi) {
                continue;
            }

            const size_t i0 = bi * GEN_TILE;
            const size_t i1 = std::min(i0 + GEN_TILE, n);
            const size_t j0 = bj * GEN_TILE;
            const size_t j1 = std::min(j0 + GEN_TILE, n);
            const size_t ni = i1 - i0;
            const size_t nj = j1 - j0;

            // For this tile L(j,k) is zero for k >= j1.
            const size_t kend = j1;

            alignas(64) double tile[GEN_TILE][GEN_TILE];
            for (size_t ii = 0; ii < ni; ++ii) {
                for (size_t jj = 0; jj < nj; ++jj) {
                    tile[ii][jj] = 0.0;
                }
            }

            for (size_t kb = 0; kb < kend; kb += GEN_KBLOCK) {
                const size_t ke = std::min(kb + GEN_KBLOCK, kend);
                for (size_t i = i0; i < i1; ++i) {
                    const double* __restrict Li = L.data() + i * n;
                    double* __restrict ti = tile[i - i0];
                    for (size_t j = j0; j < j1; ++j) {
                        const double* __restrict Lj = L.data() + j * n;
                        double sum = ti[j - j0];
#pragma omp simd reduction(+ : sum)
                        for (size_t k = kb; k < ke; ++k) {
                            sum += Li[k] * Lj[k];
                        }
                        ti[j - j0] = sum;
                    }
                }
            }

            for (size_t i = i0; i < i1; ++i) {
                const size_t ri = i * n;
                for (size_t j = j0; j < j1; ++j) {
                    const double val = tile[i - i0][j - j0];

                    double error = fabs(val - A_orig[ri + j]);
                    maxError = std::max(maxError, error);
                    double rel = error / (fabs(A_orig[ri + j]) + 1e-10);
                    relError = std::max(relError, rel);

                    if (bi != bj) {
                        const size_t rj = j * n;
                        error = fabs(val - A_orig[rj + i]);
                        maxError = std::max(maxError, error);
                        rel = error / (fabs(A_orig[rj + i]) + 1e-10);
                        relError = std::max(relError, rel);
                    }
                }
            }
        }
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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

    // Always run with the fully parallel thread team.
    omp_set_dynamic(0);

    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);

    if (validate) {
        A_orig = A; // Save original for validation
    }

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double seconds = std::chrono::duration<double>(end - start).count();

    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (seconds > 0.0 ? seconds : 1e-9) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);

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
