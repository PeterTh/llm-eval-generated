#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Cache tile sizes for the local multiply (J x K tile of B stays resident in cache)
constexpr size_t kBlockJ = 256;
constexpr size_t kBlockK = 64;

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Fills rows [rowBegin, rowBegin + rowCount) of the NxN matrix into mat (rowCount x N).
// Values depend only on global (i, j), so every rank can generate its part independently.
void initRows(std::vector<double>& mat, const size_t N, const size_t rowBegin, const size_t rowCount) {
    mat.resize(rowCount * N);
    for (size_t i = 0; i < rowCount; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, rowBegin + i, j);
        }
    }
}

// Contiguous block of rows owned by one rank
struct RowBlock {
    size_t begin;
    size_t count;
};

RowBlock rowBlockFor(const int rank, const int size, const size_t N) {
    const size_t r = static_cast<size_t>(rank);
    const size_t p = static_cast<size_t>(size);
    const size_t base = N / p;
    const size_t rem = N % p;
    return { r * base + std::min(r, rem), base + (r < rem ? 1 : 0) };
}

// C (rows x N) = A (rows x N) * B (N x N), for the rows owned by this rank.
// For every C[i][j], k is accumulated in increasing order, exactly as in the serial
// reference, so the result is bit-identical regardless of the number of ranks.
void matrixMultiplyRows(const double* A, const double* B, double* C, const size_t rows, const size_t N) {
    std::fill(C, C + rows * N, 0.0);

    for (size_t jb = 0; jb < N; jb += kBlockJ) {
        const size_t jEnd = std::min(N, jb + kBlockJ);
        for (size_t kb = 0; kb < N; kb += kBlockK) {
            const size_t kEnd = std::min(N, kb + kBlockK);
            for (size_t i = 0; i < rows; ++i) {
                double* c = C + i * N;
                const double* a = A + i * N;
                for (size_t k = kb; k < kEnd; ++k) {
                    const double aik = a[k];
                    const double* b = B + k * N;
                    for (size_t j = jb; j < jEnd; ++j) {
                        c[j] += aik * b[j];
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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank, so all ranks exit alike)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (root) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Each rank owns a block of rows of A and C; B is needed in full everywhere
    const RowBlock block = rowBlockFor(rank, size, N);
    std::vector<double> A;
    std::vector<double> B;
    std::vector<double> C_local(block.count * N);

    // Initialize matrices
    if (root) {
        printf("Initializing matrices...\n");
    }
    initRows(A, N, block.begin, block.count);
    initRows(B, N, 0, N);

    // Gather layout for assembling the full C on the root
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        const RowBlock b = rowBlockFor(r, size, N);
        counts[r] = static_cast<int>(b.count * N);
        displs[r] = static_cast<int>(b.begin * N);
    }
    std::vector<double> C;
    if (root) {
        C.resize(N * N);
    }

    // Perform matrix multiplication
    if (root) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    matrixMultiplyRows(A.data(), B.data(), C_local.data(), block.count, N);
    MPI_Gatherv(C_local.data(), static_cast<int>(C_local.size()), MPI_DOUBLE,
                C.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int status = 0;
    if (root) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000.0));

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / max_elapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation (root regenerates the full A, which is only needed here)
        if (validate) {
            printf("Validating result...\n");
            std::vector<double> fullA;
            initRows(fullA, N, 0, N);
            const bool valid = validateResult(fullA, B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }

    // Propagate the root's result so every rank returns the same exit code
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
