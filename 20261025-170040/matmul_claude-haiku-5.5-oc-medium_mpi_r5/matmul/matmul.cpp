#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Cache-blocking tile sizes for the local multiplication kernel
constexpr size_t kBlockK = 128;
constexpr size_t kBlockJ = 512;

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Fill `rowCount` rows of an N x N matrix starting at global row `rowBegin`.
// `mat` holds rowCount * N elements in row-major order.
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t rowBegin, const size_t rowCount) {
    for (size_t li = 0; li < rowCount; ++li) {
        const size_t i = rowBegin + li;
        for (size_t j = 0; j < N; ++j) {
            mat[li * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Contiguous block of rows owned by one rank
struct RowPartition {
    size_t begin;
    size_t count;
};

RowPartition partitionRows(const size_t N, const int rank, const int size) {
    const size_t p = static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = N / p;
    const size_t rem = N % p;
    return {r * base + std::min(r, rem), base + (r < rem ? 1 : 0)};
}

// Local kernel: C (rows x N) = A (rows x N) * B (N x N).
// Loop order is i-k-j with blocking over k and j, so the inner loop is
// vectorizable over j. For every C element the products are accumulated in
// increasing k order starting from 0.0 (FP contraction is disabled in the build
// so each term is rounded exactly as in the sequential reference), so the
// results are bit-identical to the original implementation.
void matrixMultiplyRows(const double* A, const double* B, double* C, const size_t rows, const size_t N) {
    std::fill(C, C + rows * N, 0.0);

    for (size_t kk = 0; kk < N; kk += kBlockK) {
        const size_t kEnd = std::min(kk + kBlockK, N);
        for (size_t jj = 0; jj < N; jj += kBlockJ) {
            const size_t jEnd = std::min(jj + kBlockJ, N);

            for (size_t i = 0; i < rows; ++i) {
                double* c = C + i * N;
                const double* a = A + i * N;
                for (size_t k = kk; k < kEnd; ++k) {
                    const double aik = a[k];
                    const double* b = B + k * N;
                    for (size_t j = jj; j < jEnd; ++j) {
                        c[j] += aik * b[j];
                    }
                }
            }
        }
    }
}

// Distributed validation. Each sample point is computed by the rank that owns
// its row; all other ranks contribute zero, so an allreduce yields the exact
// values on every rank. The comparison logic matches the sequential version.
bool validateResult(const double* A_local, const double* B, const double* C_local,
                    const size_t N, const RowPartition part, MPI_Comm comm, const int rank) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    constexpr size_t numPoints = 5 * 5;

    double expectedVals[numPoints] = {};
    double actualVals[numPoints] = {};
    double globalExpected[numPoints];
    double globalActual[numPoints];

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            const size_t idx = pi * 5 + pj;

            if (i >= part.begin && i < part.begin + part.count) {
                const size_t li = i - part.begin;
                double expected = 0.0;
                for (size_t k = 0; k < N; ++k) {
                    expected += A_local[li * N + k] * B[k * N + j];
                }
                expectedVals[idx] = expected;
                actualVals[idx] = C_local[li * N + j];
            }
        }
    }

    MPI_Allreduce(expectedVals, globalExpected, numPoints, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(actualVals, globalActual, numPoints, MPI_DOUBLE, MPI_SUM, comm);

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            const size_t idx = pi * 5 + pj;

            const double expected = globalExpected[idx];
            const double actual = globalActual[idx];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                if (rank == 0) {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, actual, relError);
                }
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

int runBenchmark(int argc, char** argv, const int rank, const int size) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        printf("Initializing matrices...\n");
    }

    const MPI_Comm comm = MPI_COMM_WORLD;
    const RowPartition part = partitionRows(N, rank, size);

    // Each rank holds its block of rows of A and C, plus a full copy of B.
    // All inputs are generated locally from the deterministic formula, so no
    // input communication is required.
    std::vector<double> A_local(part.count * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(part.count * N);

    initMatrixRows(A_local, N, part.begin, part.count);
    initMatrixRows(B, N, 0, N);

    // Perform distributed matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();

    matrixMultiplyRows(A_local.data(), B.data(), C_local.data(), part.count, N);

    const double localSeconds = MPI_Wtime() - t0;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        const long ms = static_cast<long>(seconds * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        // Calculate GFLOPS
        const double gflops = (2.0 * N * N * N) / seconds / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather full result on rank 0 for external validation output.
    // Row-block distribution is contiguous in row-major order, so gather rows.
    if (printResults) {
        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType);
        MPI_Type_commit(&rowType);

        std::vector<double> C_full;
        std::vector<int> recvCounts;
        std::vector<int> displs;
        if (rank == 0) {
            C_full.resize(N * N);
            recvCounts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                const RowPartition p = partitionRows(N, r, size);
                recvCounts[r] = static_cast<int>(p.count);
                displs[r] = static_cast<int>(p.begin);
            }
        }

        MPI_Gatherv(C_local.data(), static_cast<int>(part.count), rowType,
                    C_full.data(), rank == 0 ? recvCounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr, rowType, 0, comm);
        MPI_Type_free(&rowType);

        if (rank == 0) {
            print_results(C_full, "MatrixC");
        }
    }

    // Validation (all ranks take part so that the verdict is consistent)
    bool valid = true;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = validateResult(A_local.data(), B.data(), C_local.data(), N, part, comm, rank);

        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int status = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return status;
}
