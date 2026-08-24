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

// Initialize a sub-rectangle of a matrix
void initMatrixBlock(std::vector<double>& mat, const size_t N, const size_t row_start, const size_t row_end,
                     const size_t col_start, const size_t col_end) {
    for (size_t i = row_start; i < row_end; ++i) {
        for (size_t j = col_start; j < col_end; ++j) {
            mat[(i - row_start) * (col_end - col_start) + (j - col_start)] = getPseudoRndValue(N, i, j);
        }
    }
}

// Compute a block of C = A * B using local blocks
// localA: rows of A owned by this rank (row_start..row_end, all columns)
// localB: columns of B owned by this rank (all rows, col_start..col_end)
// localC: output block (row_start..row_end, col_start..col_end)
void computeBlock(const std::vector<double>& localA, const std::vector<double>& localB,
                  std::vector<double>& localC, const size_t N,
                  const size_t local_rows, const size_t local_cols) {
    for (size_t i = 0; i < local_rows; ++i) {
        for (size_t j = 0; j < local_cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += localA[i * N + k] * localB[k * N + j];
            }
            localC[i * local_cols + j] = sum;
        }
    }
}

// Compute the row range for a given grid row in a 2D block distribution
static inline void getRowRange(size_t N, int grid_rows, int my_grid_row,
                               size_t& row_start, size_t& row_end) {
    size_t block_rows = N / grid_rows;
    size_t remainder_rows = N % grid_rows;
    row_start = my_grid_row * block_rows + (my_grid_row < static_cast<int>(remainder_rows) ? my_grid_row : remainder_rows);
    row_end = (my_grid_row + 1) * block_rows + (my_grid_row + 1 <= static_cast<int>(remainder_rows) ? my_grid_row + 1 : remainder_rows);
}

// Compute the column range for a given grid col in a 2D block distribution
static inline void getColRange(size_t N, int grid_cols, int my_grid_col,
                               size_t& col_start, size_t& col_end) {
    size_t block_cols = N / grid_cols;
    size_t remainder_cols = N % grid_cols;
    col_start = my_grid_col * block_cols + (my_grid_col < static_cast<int>(remainder_cols) ? my_grid_col : remainder_cols);
    col_end = (my_grid_col + 1) * block_cols + (my_grid_col + 1 <= static_cast<int>(remainder_cols) ? my_grid_col + 1 : remainder_cols);
}

// 2D block-distributed matrix multiplication
// Ranks are arranged in a 2D grid (grid_rows x grid_cols)
// Each rank (ri, rj) owns block C[ri*block_rows:(ri+1)*block_rows][rj*block_cols:(rj+1)*block_cols]
void matrixMultiplyMPI(const size_t N, std::vector<double>& globalC,
                       int rank, int num_ranks, int grid_rows, int grid_cols,
                       int my_grid_row, int my_grid_col) {
    size_t my_row_start, my_row_end, my_col_start, my_col_end;
    getRowRange(N, grid_rows, my_grid_row, my_row_start, my_row_end);
    getColRange(N, grid_cols, my_grid_col, my_col_start, my_col_end);

    size_t my_local_rows = my_row_end - my_row_start;
    size_t my_local_cols = my_col_end - my_col_start;

    // Local storage: my rows of A (need full width for dot products)
    std::vector<double> localA(my_local_rows * N);
    initMatrixBlock(localA, N, my_row_start, my_row_end, 0, N);

    // Local storage: my columns of B (need full height for dot products)
    std::vector<double> localB(N * my_local_cols);
    for (size_t k = 0; k < N; ++k) {
        for (size_t j = 0; j < my_local_cols; ++j) {
            localB[k * my_local_cols + j] = getPseudoRndValue(N, k, my_col_start + j);
        }
    }

    // Local C block
    std::vector<double> localC(my_local_rows * my_local_cols);

    // Compute my block: C[i][j] = sum_k A[i][k] * B[k][j]
    for (size_t i = 0; i < my_local_rows; ++i) {
        for (size_t j = 0; j < my_local_cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += localA[i * N + k] * localB[k * my_local_cols + j];
            }
            localC[i * my_local_cols + j] = sum;
        }
    }

    // Gather all local C blocks into a flat buffer on rank 0 using MPI_Gatherv,
    // then rearrange into the correct 2D positions in globalC.
    std::vector<int> counts(num_ranks);
    std::vector<int> displs(num_ranks);
    int total_elements = 0;
    for (int r = 0; r < num_ranks; ++r) {
        int ri = r / grid_cols;
        int rj = r % grid_cols;
        size_t r_row_start, r_row_end, r_col_start, r_col_end;
        getRowRange(N, grid_rows, ri, r_row_start, r_row_end);
        getColRange(N, grid_cols, rj, r_col_start, r_col_end);
        counts[r] = static_cast<int>((r_row_end - r_row_start) * (r_col_end - r_col_start));
        displs[r] = total_elements;
        total_elements += counts[r];
    }

    std::vector<double> gatheredC;
    if (rank == 0) {
        gatheredC.resize(total_elements);
    }

    MPI_Gatherv(localC.data(), static_cast<int>(my_local_rows * my_local_cols), MPI_DOUBLE,
                rank == 0 ? gatheredC.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // On rank 0, scatter gathered data into correct 2D positions
    if (rank == 0) {
        int offset = 0;
        for (int r = 0; r < num_ranks; ++r) {
            int ri = r / grid_cols;
            int rj = r % grid_cols;
            size_t r_row_start, r_row_end, r_col_start, r_col_end;
            getRowRange(N, grid_rows, ri, r_row_start, r_row_end);
            getColRange(N, grid_cols, rj, r_col_start, r_col_end);
            size_t r_local_rows = r_row_end - r_row_start;
            size_t r_local_cols = r_col_end - r_col_start;
            for (size_t i = 0; i < r_local_rows; ++i) {
                for (size_t j = 0; j < r_local_cols; ++j) {
                    globalC[(r_row_start + i) * N + (r_col_start + j)] = gatheredC[offset++];
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

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
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

    // Compute 2D grid dimensions (as square as possible)
    int grid_rows = 1;
    int grid_cols = num_ranks;
    // Find the best grid shape close to sqrt(num_ranks)
    int sqrt_ranks = static_cast<int>(std::sqrt(num_ranks));
    for (int r = sqrt_ranks; r >= 1; --r) {
        if (num_ranks % r == 0) {
            grid_rows = r;
            grid_cols = num_ranks / r;
            break;
        }
    }

    int my_grid_row = rank / grid_cols;
    int my_grid_col = rank % grid_cols;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d (grid: %dx%d)\n", num_ranks, grid_rows, grid_cols);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate global C only on rank 0 for gathering
    std::vector<double> globalC;
    if (rank == 0) {
        globalC.resize(N * N);
    }

    // Initialize matrices and compute
    if (rank == 0) printf("Initializing matrices...\n");
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyMPI(N, globalC, rank, num_ranks, grid_rows, grid_cols, my_grid_row, my_grid_col);

    auto end = std::chrono::high_resolution_clock::now();

    long local_duration_ms = static_cast<long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(globalC, "MatrixC");
        }

        // For validation, reconstruct A and B locally
        if (validate) {
            printf("Validating result...\n");
            std::vector<double> A(N * N);
            std::vector<double> B(N * N);
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    A[i * N + j] = getPseudoRndValue(N, i, j);
                    B[i * N + j] = getPseudoRndValue(N, i, j);
                }
            }
            bool valid = validateResult(A, B, globalC, N);

            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
