#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize a full matrix
void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Local matrix multiply for a sub-block of C: C_block = A * B (subset of rows and columns)
// A: full matrix N x N, B: full matrix N x N
// C_block: localRows x localCols, covering rows [rowStart:rowStart+localRows],
//          cols [colStart:colStart+localCols]
// Uses i-k-j loop ordering for cache-friendly access
void matrixMultiplyBlock(const double* A, const double* B, double* C_block,
                         const size_t localRows, const size_t localCols,
                         const size_t rowStart, const size_t colStart, const size_t N) {
    // Zero out C_block
    std::memset(C_block, 0, localRows * localCols * sizeof(double));

    for (size_t i = 0; i < localRows; ++i) {
        const size_t global_i = rowStart + i;
        const double* A_row = A + global_i * N;
        for (size_t k = 0; k < N; ++k) {
            const double a_ik = A_row[k];
            const double* B_row = B + k * N + colStart;
            double* C_row = C_block + i * localCols;
            for (size_t j = 0; j < localCols; ++j) {
                C_row[j] += a_ik * B_row[j];
            }
        }
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all processes parse identically)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Determine 2D process grid (prefer square-ish layout)
    int procs_x = 1, procs_y = size;
    for (int i = static_cast<int>(std::sqrt(static_cast<double>(size))); i >= 1; --i) {
        if (size % i == 0) {
            procs_x = i;
            procs_y = size / i;
            break;
        }
    }

    // Process coordinates
    int px = rank / procs_y;
    int py = rank % procs_y;

    // Compute block sizes with remainder handling (last row/col gets extra)
    size_t baseRows = N / procs_x;
    size_t baseCols = N / procs_y;
    size_t extraRows = N % procs_x;
    size_t extraCols = N % procs_y;

    size_t myLocalRows = baseRows + (static_cast<size_t>(px) < extraRows ? 1 : 0);
    size_t myLocalCols = baseCols + (static_cast<size_t>(py) < extraCols ? 1 : 0);

    // Global start indices for this process's C block
    size_t myRowStart = static_cast<size_t>(px) * baseRows + std::min(static_cast<size_t>(px), extraRows);
    size_t myColStart = static_cast<size_t>(py) * baseCols + std::min(static_cast<size_t>(py), extraCols);

    // Print banner (rank 0 only)
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d (grid: %d x %d)\n", size, procs_x, procs_y);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Each process initializes full A and B matrices
    // (init is O(N^2) which is negligible compared to O(N^3) multiply)
    std::vector<double> A(N * N), B(N * N);
    initMatrix(A, N);
    initMatrix(B, N);

    // C block: myLocalRows x myLocalCols
    std::vector<double> C_block(myLocalRows * myLocalCols);

    if (rank == 0) {
        printf("Initializing matrices...\n");
        printf("Computing matrix multiplication...\n");
    }

    // Synchronize and time the computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = MPI_Wtime();

    matrixMultiplyBlock(A.data(), B.data(), C_block.data(),
                        myLocalRows, myLocalCols, myRowStart, myColStart, N);

    double localTime = MPI_Wtime() - start;

    // Get max time across all processes (slowest process determines wall time)
    double globalTime;
    MPI_Reduce(&localTime, &globalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", globalTime * 1000.0);

        // Calculate GFLOPS: 2*N^3 floating point ops / time in seconds
        double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N))
                       / globalTime / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather C blocks to root for validation and/or result printing
    bool needGather = validate || printResults;
    int retCode = 0;

    if (needGather) {
        // Compute recvCounts and displs for C gather (contiguous placement)
        std::vector<int> recvCounts(size);
        std::vector<int> displs(size);

        if (rank == 0) {
            // Gather into contiguous buffer, then rearrange into row-major layout
            size_t totalElements = 0;
            for (int r = 0; r < size; ++r) {
                int rpx = r / procs_y;
                int rpy = r % procs_y;
                size_t rLocalRows = baseRows + (static_cast<size_t>(rpx) < extraRows ? 1 : 0);
                size_t rLocalCols = baseCols + (static_cast<size_t>(rpy) < extraCols ? 1 : 0);
                totalElements += rLocalRows * rLocalCols;
            }

            std::vector<double> C_gathered(totalElements);
            size_t offset = 0;
            for (int r = 0; r < size; ++r) {
                int rpx = r / procs_y;
                int rpy = r % procs_y;
                size_t rLocalRows = baseRows + (static_cast<size_t>(rpx) < extraRows ? 1 : 0);
                size_t rLocalCols = baseCols + (static_cast<size_t>(rpy) < extraCols ? 1 : 0);
                recvCounts[r] = static_cast<int>(rLocalRows * rLocalCols);
                displs[r] = static_cast<int>(offset);
                offset += rLocalRows * rLocalCols;
            }

            MPI_Gatherv(C_block.data(), static_cast<int>(myLocalRows * myLocalCols), MPI_DOUBLE,
                        C_gathered.data(), recvCounts.data(), displs.data(),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Rearrange from contiguous gather layout to row-major matrix layout
            std::vector<double> C_full(N * N);
            offset = 0;
            for (int r = 0; r < size; ++r) {
                int rpx = r / procs_y;
                int rpy = r % procs_y;
                size_t rLocalRows = baseRows + (static_cast<size_t>(rpx) < extraRows ? 1 : 0);
                size_t rLocalCols = baseCols + (static_cast<size_t>(rpy) < extraCols ? 1 : 0);

                // Global start for this block
                size_t rRowStart = static_cast<size_t>(rpx) * baseRows +
                                   std::min(static_cast<size_t>(rpx), extraRows);
                size_t rColStart = static_cast<size_t>(rpy) * baseCols +
                                   std::min(static_cast<size_t>(rpy), extraCols);

                // Copy block elements to correct positions in C_full
                for (size_t i = 0; i < rLocalRows; ++i) {
                    for (size_t j = 0; j < rLocalCols; ++j) {
                        C_full[(rRowStart + i) * N + (rColStart + j)] =
                            C_gathered[offset + i * rLocalCols + j];
                    }
                }
                offset += rLocalRows * rLocalCols;
            }

            if (printResults) {
                print_results(C_full, "MatrixC");
            }

            if (validate) {
                // Validate using locally available full A and B
                printf("Validating result...\n");
                bool valid = true;
                constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

                for (size_t pi = 0; pi < 5 && valid; ++pi) {
                    for (size_t pj = 0; pj < 5 && valid; ++pj) {
                        const size_t i = checkPoints[pi] % N;
                        const size_t j = checkPoints[pj] % N;

                        double expected = 0.0;
                        for (size_t k = 0; k < N; ++k) {
                            expected += A[i * N + k] * B[k * N + j];
                        }

                        const double actual = C_full[i * N + j];
                        const double relError = std::abs((actual - expected) / (expected + 1e-10));

                        if (relError > 1e-6) {
                            printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                                   i, j, expected, actual, relError);
                            valid = false;
                        }
                    }
                }

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    retCode = 1;
                }
            }
        } else {
            // Non-root processes participate in gather but don't receive data
            MPI_Gatherv(C_block.data(), static_cast<int>(myLocalRows * myLocalCols), MPI_DOUBLE,
                        nullptr, nullptr, nullptr,
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return retCode;
}
