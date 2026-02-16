#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

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

// Multiply local block of rows: A_local is local_rows x N, B is N x N, C_local is local_rows x N
void matrixMultiplyLocal(const std::vector<double>& A_local, const std::vector<double>& B,
                         std::vector<double>& C_local, const size_t N, const size_t local_rows) {
    for (size_t i = 0; i < local_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A_local[i * N + k] * B[k * N + j];
            }
            C_local[i * N + j] = sum;
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
    
    // Parse command line arguments (done by all ranks)
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
            if (i == 1) {
                // only have root print unknown option to avoid duplicates
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    // Initialize MPI
    MPI_Init(&argc, &argv);
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine distribution of rows
    const int P = world_size;
    std::vector<int> rows(P);
    int base = static_cast<int>(N) / P;
    int rem = static_cast<int>(N) % P;
    for (int r = 0; r < P; ++r) rows[r] = base + (r < rem ? 1 : 0);

    // prepare sendcounts and displacements for Scatterv/Gatherv (counts in elements)
    std::vector<int> sendcounts(P);
    std::vector<int> displs(P);
    int offset = 0;
    for (int r = 0; r < P; ++r) {
        sendcounts[r] = rows[r] * static_cast<int>(N);
        displs[r] = offset;
        offset += sendcounts[r];
    }

    const int local_rows = rows[world_rank];

    // Root allocates full matrices
    std::vector<double> A_full;
    std::vector<double> B_full(N * N);
    std::vector<double> C_full;
    if (world_rank == 0) {
        A_full.resize(N * N);
        C_full.resize(N * N);
        if (world_size == 1) {
            // single process: keep original behavior
            printf("Initializing matrices...\n");
            initMatrix(A_full, N);
            initMatrix(B_full, N);

            printf("Computing matrix multiplication...\n");
            auto start = std::chrono::high_resolution_clock::now();
            matrixMultiplyLocal(A_full, B_full, C_full, N, N);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            printf("Computation time: %ld ms\n", duration.count());
            double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
            printf("Performance: %.3f GFLOPS\n", gflops);

            if (printResults) print_results(C_full, "MatrixC");
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(A_full, B_full, C_full, N);
                if (valid) { printf("Validation: PASSED\n"); MPI_Finalize(); return 0; }
                else { printf("Validation: FAILED\n"); MPI_Finalize(); return 1; }
            }

            MPI_Finalize();
            return 0;
        }

        // multi-process root initialization
        if (world_rank == 0) {
            printf("Initializing matrices...\n");
            initMatrix(A_full, N);
            initMatrix(B_full, N);
        }
    } else {
        // non-root ranks need buffer for B as well
        std::fill(B_full.begin(), B_full.end(), 0.0);
    }

    // Allocate local A and C blocks
    std::vector<double> A_local(local_rows * N);
    std::vector<double> C_local(local_rows * N);

    // Scatter rows of A to all ranks
    MPI_Scatterv(world_rank == 0 ? A_full.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), local_rows * static_cast<int>(N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast full B to all ranks
    MPI_Bcast(B_full.data(), static_cast<int>(N) * static_cast<int>(N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) printf("Computing matrix multiplication...\n");

    double t0 = MPI_Wtime();
    matrixMultiplyLocal(A_local, B_full, C_local, N, local_rows);
    double t1 = MPI_Wtime();
    double local_time = t1 - t0;

    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather C blocks back to root
    MPI_Gatherv(C_local.data(), local_rows * static_cast<int>(N), MPI_DOUBLE,
                world_rank == 0 ? C_full.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        long ms = static_cast<long>(max_time * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        double gflops = (2.0 * N * N * N) / (max_time) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) print_results(C_full, "MatrixC");

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A_full, B_full, C_full, N);
            if (valid) { printf("Validation: PASSED\n"); }
            else { printf("Validation: FAILED\n"); MPI_Finalize(); return 1; }
        }
    }

    MPI_Finalize();
    return 0;
}
