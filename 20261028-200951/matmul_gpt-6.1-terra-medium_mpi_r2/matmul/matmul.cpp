#include <mpi.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            unsigned long long value = strtoull(arg, &end, 10);
            if (*arg == '-' || *arg == '\0' || *end != '\0' || value == 0 ||
                value > std::numeric_limits<size_t>::max() ||
                value > std::numeric_limits<size_t>::max() / value) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", arg);
                MPI_Finalize();
                return 1;
            }
            N = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    const size_t base = N / static_cast<size_t>(ranks), extra = N % static_cast<size_t>(ranks);
    const size_t rows = base + (static_cast<size_t>(rank) < extra);
    const size_t first = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", N, N, validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    std::vector<double> A(rows * N), B(N * N), C(rows * N, 0.0);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < N; ++j) A[i * N + j] = getPseudoRndValue(N, first + i, j);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) B[i * N + j] = getPseudoRndValue(N, i, j);
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    // Keep k in ascending order for every output element, while traversing B and C contiguously.
    constexpr size_t block = 64;
    for (size_t ii = 0; ii < rows; ii += block)
        for (size_t kk = 0; kk < N; kk += block)
            for (size_t jj = 0; jj < N; jj += block)
                for (size_t i = ii; i < std::min(ii + block, rows); ++i)
                    for (size_t k = kk; k < std::min(kk + block, N); ++k) {
                        const double a = A[i * N + k];
                        for (size_t j = jj; j < std::min(jj + block, N); ++j)
                            C[i * N + j] += a * B[k * N + j];
                    }
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        printf("Performance: %.3f GFLOPS\n", 2.0 * N * N * N / maxElapsed / 1e9);
    }
    if (printResults) {
        std::vector<double> result;
        if (rank == 0) {
            result.resize(N * N);
            std::copy(C.begin(), C.end(), result.begin());
            for (int r = 1; r < ranks; ++r) {
                size_t count = (base + (static_cast<size_t>(r) < extra)) * N;
                size_t offset = (static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), extra)) * N;
                while (count) {
                    int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(result.data() + offset, chunk, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    offset += chunk; count -= chunk;
                }
            }
            print_results(result, "MatrixC");
        } else {
            size_t offset = 0, count = C.size();
            while (count) {
                int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
                MPI_Send(C.data() + offset, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                offset += chunk; count -= chunk;
            }
        }
    }
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        for (size_t pi = 0; pi < 5; ++pi) {
            size_t i = pi % N;
            if (i < first || i >= first + rows) continue;
            for (size_t pj = 0; pj < 5; ++pj) {
                size_t j = pj % N;
                double expected = 0;
                for (size_t k = 0; k < N; ++k) expected += A[(i - first) * N + k] * B[k * N + j];
                double actual = C[(i - first) * N + j];
                double error = std::abs((actual - expected) / (expected + 1e-10));
                if (error > 1e-6) {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i, j, expected, actual, error);
                    valid = 0;
                }
            }
        }
        int allValid;
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        valid = allValid;
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
