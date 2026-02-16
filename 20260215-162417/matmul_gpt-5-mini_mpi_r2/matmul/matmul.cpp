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

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Local block multiply: A_block has rows = rowsA, full B (N x N), produce C_block (rowsA x N)
static void localMatrixMultiply(const double* A_block, const double* B, double* C_block, size_t rowsA, size_t N) {
    for (size_t ii = 0; ii < rowsA; ++ii) {
        const size_t i_offset = ii * N;
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A_block[i_offset + k] * B[k * N + j];
            }
            C_block[i_offset + j] = sum;
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
                if (MPI_COMM_WORLD == MPI_COMM_NULL) {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, actual, relError);
                } else {
                    // Only root prints (handled in caller)
                }
                return false;
            }
        }
    }
    
    return true;
}

void printUsage(const char* progName) {
    if (MPI::COMM_WORLD.Get_rank() == 0) {
        printf("Usage: %s [options]\n", progName);
        printf("Options:\n");
        printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
        printf("  -v           Enable validation\n");
        printf("  -r           Print results for external validation\n");
        printf("  -h           Show this help message\n");
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks parse the same args)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI world size: %d\n", world_size);
    }

    // Prepare distribution of rows
    std::vector<int> rows_per_rank(world_size, 0);
    for (int r = 0; r < world_size; ++r) {
        rows_per_rank[r] = static_cast<int>(N / world_size);
    }
    int remainder = static_cast<int>(N % world_size);
    for (int r = 0; r < remainder; ++r) rows_per_rank[r]++;

    std::vector<int> sendcounts(world_size, 0);
    std::vector<int> displs(world_size, 0);
    for (int r = 0; r < world_size; ++r) {
        sendcounts[r] = rows_per_rank[r] * static_cast<int>(N); // number of doubles
    }
    int offset = 0;
    for (int r = 0; r < world_size; ++r) {
        displs[r] = offset;
        offset += sendcounts[r];
    }

    // Root initializes full matrices A and B
    std::vector<double> A;
    std::vector<double> B;
    std::vector<double> C;
    if (rank == 0) {
        A.resize(N * N);
        B.resize(N * N);
        C.resize(N * N);
        if (N > 0) {
            initMatrix(A, N);
            initMatrix(B, N);
        }
    } else {
        // non-root allocate B to receive broadcast
        B.resize(N * N);
    }

    // Each rank allocates local A block and C block
    const int local_rows = rows_per_rank[rank];
    std::vector<double> A_local(local_rows * N);
    std::vector<double> C_local(local_rows * N);

    // Scatter rows of A from root to all ranks
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), static_cast<int>(A_local.size()), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Broadcast B to all ranks
    MPI_Bcast(B.data(), static_cast<int>(B.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Sync before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Local multiplication
    localMatrixMultiply(A_local.data(), B.data(), C_local.data(), static_cast<size_t>(local_rows), N);

    double t1 = MPI_Wtime();
    double local_time = t1 - t0;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather C_local back to root
    MPI_Gatherv(C_local.data(), static_cast<int>(C_local.size()), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Computation time and GFLOPS based on max_time across ranks
        long ms = static_cast<long>(max_time * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N)) / (max_time) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
