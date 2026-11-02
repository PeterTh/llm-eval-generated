#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Consecutive row ranges keep both initialization and optional result collection local.
size_t rowCount(size_t N, int ranks, int rank) {
    return N / static_cast<size_t>(ranks) + (static_cast<size_t>(rank) < N % static_cast<size_t>(ranks));
}

size_t firstRow(size_t N, int ranks, int rank) {
    const size_t r = static_cast<size_t>(rank);
    return r * (N / static_cast<size_t>(ranks)) + std::min(r, N % static_cast<size_t>(ranks));
}

void initRows(std::vector<double>& mat, size_t N, size_t first, size_t rows) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < N; ++j)
            mat[i * N + j] = getPseudoRndValue(N, first + i, j);
}

// Work on four rows and a short column panel at a time. Each element is
// accumulated in increasing k order, as in the original dot products.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t N, size_t rows) {
    constexpr size_t columnBlock = 64;
    for (size_t i = 0; i < rows; i += 4) {
        const size_t nr = std::min(size_t{4}, rows - i);
        for (size_t jj = 0; jj < N; jj += columnBlock) {
            const size_t end = std::min(N, jj + columnBlock);
            for (size_t k = 0; k < N; ++k) {
                const double* b = B.data() + k * N;
                const double a0 = A[i * N + k];
                const double a1 = nr > 1 ? A[(i + 1) * N + k] : 0.0;
                const double a2 = nr > 2 ? A[(i + 2) * N + k] : 0.0;
                const double a3 = nr > 3 ? A[(i + 3) * N + k] : 0.0;
                double* c0 = C.data() + i * N;
                double* c1 = nr > 1 ? C.data() + (i + 1) * N : c0;
                double* c2 = nr > 2 ? C.data() + (i + 2) * N : c0;
                double* c3 = nr > 3 ? C.data() + (i + 3) * N : c0;
                if (nr == 4) {
                    for (size_t j = jj; j < end; ++j) {
                        c0[j] += a0 * b[j];
                        c1[j] += a1 * b[j];
                        c2[j] += a2 * b[j];
                        c3[j] += a3 * b[j];
                    }
                } else {
                    for (size_t j = jj; j < end; ++j) c0[j] += a0 * b[j];
                    if (nr > 1) for (size_t j = jj; j < end; ++j) c1[j] += a1 * b[j];
                    if (nr > 2) for (size_t j = jj; j < end; ++j) c2[j] += a2 * b[j];
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                    const std::vector<double>& C, size_t N, size_t first, size_t rows) {
    for (size_t pi = 0; pi < 5; ++pi) {
        const size_t i = pi % N;
        if (i < first || i - first >= rows) continue;
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t j = pj % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += A[(i - first) * N + k] * B[k * N + j];
            const double actual = C[(i - first) * N + j];
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

// MPI counts are int, so transfer large row ranges in multiple messages.
void transferResult(double* data, size_t count, int peer, bool send) {
    while (count > 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t N = 512;
    bool validate = false, printResults = false;
    int status = 0;
    bool help = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = strtoull(arg, &end, 10);
            if (errno || end == arg || *end || arg[0] == '-' || value == 0 ||
                value > std::numeric_limits<size_t>::max()) {
                if (rank == 0) printf("Invalid matrix size: %s\n", arg);
                status = 1;
                break;
            }
            N = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { help = true; break; }
        else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            status = 1;
            break;
        }
    }
    if (help || status) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return status;
    }
    if (N > std::numeric_limits<size_t>::max() / N ||
        N * N > std::vector<double>().max_size()) {
        if (rank == 0) printf("Matrix size is too large\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    const size_t rows = rowCount(N, ranks, rank);
    const size_t first = firstRow(N, ranks, rank);
    std::vector<double> A, B, C, result;
    try {
        A.resize(rows * N);
        if (rows) B.resize(N * N);
        C.resize(rows * N);
        if (rank == 0 && printResults) result.resize(N * N);
    } catch (const std::exception&) {
        fprintf(stderr, "Rank %d: unable to allocate matrices\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    initRows(A, N, first, rows);
    if (rows) initRows(B, N, 0, N);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, N, rows);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(maxElapsed * 1000.0));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / maxElapsed / 1e9);
    }

    if (printResults) {
        if (rank == 0) {
            std::copy(C.begin(), C.end(), result.begin());
            for (int r = 1; r < ranks; ++r) {
                const size_t count = rowCount(N, ranks, r) * N;
                if (count) transferResult(result.data() + firstRow(N, ranks, r) * N, count, r, false);
            }
            print_results(result, "MatrixC");
        } else if (!C.empty()) transferResult(C.data(), C.size(), 0, true);
    }

    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = validateResult(A, B, C, N, first, rows) ? 1 : 0;
        int allValid = 0;
        MPI_Allreduce(&localValid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
        status = allValid ? 0 : 1;
    }
    MPI_Finalize();
    return status;
}
