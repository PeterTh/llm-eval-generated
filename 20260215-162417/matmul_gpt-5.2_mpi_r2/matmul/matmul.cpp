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

static void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

static inline void row_range_for_rank(const size_t N, const int size, const int rank,
                                     size_t& start_row, size_t& local_rows) {
    const size_t base = N / static_cast<size_t>(size);
    const size_t rem = N % static_cast<size_t>(size);
    local_rows = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    start_row = base * static_cast<size_t>(rank) + (static_cast<size_t>(rank) < rem ? static_cast<size_t>(rank) : rem);
}

static inline int owner_of_row(const size_t N, const int size, const size_t row) {
    const size_t base = N / static_cast<size_t>(size);
    const size_t rem = N % static_cast<size_t>(size);
    if (base == 0) return static_cast<int>(row); // N < size: first N ranks own one row each
    const size_t cutoff = (base + 1) * rem;
    if (row < cutoff) return static_cast<int>(row / (base + 1));
    return static_cast<int>(rem + (row - cutoff) / base);
}

static void transposeB(const std::vector<double>& B, std::vector<double>& BT, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            BT[j * N + i] = B[i * N + j];
        }
    }
}

static void matrixMultiplyLocal(const std::vector<double>& A_local, const std::vector<double>& BT,
                               std::vector<double>& C_local, const size_t N, const size_t local_rows) {
    for (size_t i = 0; i < local_rows; ++i) {
        const double* arow = &A_local[i * N];
        double* crow = &C_local[i * N];
        for (size_t j = 0; j < N; ++j) {
            const double* brow = &BT[j * N];
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += arow[k] * brow[k];
            }
            crow[j] = sum;
        }
    }
}

static bool validate_full_C_root(const std::vector<double>& B, const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
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

static bool validate_sampled_MPI(const std::vector<double>& B, const std::vector<double>& C_local,
                                const size_t N, const size_t start_row, const size_t local_rows,
                                const int rank, const int size, MPI_Comm comm) {
    (void)local_rows;
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    // Non-root ranks send the 5 sampled columns for each sampled row they own.
    if (rank != 0) {
        for (size_t pi = 0; pi < 5; ++pi) {
            const size_t i = checkPoints[pi] % N;
            const int owner = owner_of_row(N, size, i);
            if (owner != rank) continue;

            const size_t li = i - start_row;
            double vals[5];
            for (size_t pj = 0; pj < 5; ++pj) {
                const size_t j = checkPoints[pj] % N;
                vals[pj] = C_local[li * N + j];
            }
            MPI_Send(vals, 5, MPI_DOUBLE, 0, 100 + static_cast<int>(pi), comm);
        }
        return true;
    }

    // Root receives sampled values as needed and validates.
    for (size_t pi = 0; pi < 5; ++pi) {
        const size_t i = checkPoints[pi] % N;
        const int owner = owner_of_row(N, size, i);

        double vals[5];
        if (owner == 0) {
            const size_t li = i - start_row;
            for (size_t pj = 0; pj < 5; ++pj) {
                const size_t j = checkPoints[pj] % N;
                vals[pj] = C_local[li * N + j];
            }
        } else {
            MPI_Recv(vals, 5, MPI_DOUBLE, owner, 100 + static_cast<int>(pi), comm, MPI_STATUS_IGNORE);
        }

        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
            }

            const double actual = vals[pj];
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

static void printUsage(const char* progName) {
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

    size_t N = 512;
    int validate = 0;
    int printResults = 0;
    int action = 0; // 0=run, 1=help, 2=error

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                action = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                action = 2;
            }
        }

        if (action == 1) {
            printUsage(argv[0]);
        } else if (action == 2) {
            printUsage(argv[0]);
        }
    }

    unsigned long long N64 = static_cast<unsigned long long>(N);
    MPI_Bcast(&action, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N64, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(N64);

    if (action != 0) {
        MPI_Finalize();
        return action == 1 ? 0 : 1;
    }

    size_t start_row = 0, local_rows = 0;
    row_range_for_rank(N, size, rank, start_row, local_rows);
    const size_t local_elems = local_rows * N;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate local matrices (row-block distributed A and C) and replicated B.
    std::vector<double> A_local(local_elems);
    std::vector<double> C_local(local_elems);
    std::vector<double> B(N * N);
    std::vector<double> BT(N * N);

    if (rank == 0) {
        printf("Initializing matrices...\n");
    }

    // Deterministic initialization: each rank initializes its own A rows and full B.
    for (size_t li = 0; li < local_rows; ++li) {
        const size_t i = start_row + li;
        for (size_t j = 0; j < N; ++j) {
            A_local[li * N + j] = getPseudoRndValue(N, i, j);
        }
    }
    initMatrix(B, N);
    transposeB(B, BT, N);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    matrixMultiplyLocal(A_local, BT, C_local, N, local_rows);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    double local_time = t1 - t0;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> C_full;
    if (rank == 0) {
        const long ms = static_cast<long>(max_time * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        const double secs = max_time > 0.0 ? max_time : 1e-12;
        const double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N)) / secs / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            C_full.resize(N * N);
        }
    }

    // Gather full C only when required for printing.
    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displs;
        if (rank == 0) {
            counts.resize(static_cast<size_t>(size));
            displs.resize(static_cast<size_t>(size));
            size_t disp = 0;
            for (int r = 0; r < size; ++r) {
                size_t sr = 0, lr = 0;
                row_range_for_rank(N, size, r, sr, lr);
                const size_t cnt = lr * N;
                counts[static_cast<size_t>(r)] = static_cast<int>(cnt);
                displs[static_cast<size_t>(r)] = static_cast<int>(disp);
                disp += cnt;
            }
        }

        MPI_Gatherv(C_local.data(), static_cast<int>(local_elems), MPI_DOUBLE,
                    rank == 0 ? C_full.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(C_full, "MatrixC");
        }
    }

    int valid_int = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }

        bool valid_bool = true;
        if (printResults) {
            if (rank == 0) {
                valid_bool = validate_full_C_root(B, C_full, N);
            }
        } else {
            valid_bool = validate_sampled_MPI(B, C_local, N, start_row, local_rows, rank, size, MPI_COMM_WORLD);
        }

        if (rank == 0) {
            valid_int = valid_bool ? 1 : 0;
        }
        MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validation: %s\n", valid_int ? "PASSED" : "FAILED");
        }
    }

    MPI_Finalize();
    return (validate && !valid_int) ? 1 : 0;
}
