#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Generate the same values as the original, using global matrix indices.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

struct Block {
    size_t begin;
    size_t size;
};

static Block partition(size_t n, int parts, int index) {
    const size_t p = static_cast<size_t>(parts);
    const size_t i = static_cast<size_t>(index);
    return {i * (n / p) + std::min(i, n % p), n / p + (i < n % p)};
}

// B is packed in eight-column panels. A microkernel computes four rows at a
// time, reusing each B value while accumulating each dot product in k order.
using Vec4 = double __attribute__((vector_size(32)));

template <int Rows>
static void multiplyPanel(const double* A, const double* B, double* C,
                          size_t N, size_t cols, size_t width) {
    Vec4 sums[Rows][2] = {};
    for (size_t k = 0; k < N; ++k) {
        Vec4 b0, b1;
        std::memcpy(&b0, B + k * 8, sizeof(b0));
        std::memcpy(&b1, B + k * 8 + 4, sizeof(b1));
        for (int r = 0; r < Rows; ++r) {
            const double a = A[static_cast<size_t>(r) * N + k];
            const Vec4 av = {a, a, a, a};
            sums[r][0] += av * b0;
            sums[r][1] += av * b1;
        }
    }
    for (int r = 0; r < Rows; ++r) {
        double values[8];
        std::memcpy(values, &sums[r][0], sizeof(Vec4));
        std::memcpy(values + 4, &sums[r][1], sizeof(Vec4));
        std::memcpy(C + static_cast<size_t>(r) * cols, values, width * sizeof(double));
    }
}

static void matrixMultiply(const std::vector<double>& A, const std::vector<double>& packedB,
                           std::vector<double>& C, size_t N, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) return;
    for (size_t j = 0; j < cols; j += 8) {
        const double* b = packedB.data() + (j / 8) * N * 8;
        const size_t width = std::min(size_t{8}, cols - j);
        size_t i = 0;
        for (; i + 4 <= rows; i += 4)
            multiplyPanel<4>(A.data() + i * N, b, C.data() + i * cols + j, N, cols, width);
        switch (rows - i) {
            case 3: multiplyPanel<3>(A.data() + i * N, b, C.data() + i * cols + j, N, cols, width); break;
            case 2: multiplyPanel<2>(A.data() + i * N, b, C.data() + i * cols + j, N, cols, width); break;
            case 1: multiplyPanel<1>(A.data() + i * N, b, C.data() + i * cols + j, N, cols, width); break;
        }
    }
}

static bool validateResult(const std::vector<double>& C, size_t N, Block rows, Block cols) {
    if (N == 0) return true;
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = pi % N, j = pj % N;
            if (i < rows.begin || i >= rows.begin + rows.size ||
                j < cols.begin || j >= cols.begin + cols.size) continue;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            const double actual = C[(i - rows.begin) * cols.size + j - cols.begin];
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

// MPI counts are int, so transfer arbitrarily large blocks in chunks.
static void transfer(double* data, size_t count, int peer, bool send) {
    while (count != 0) {
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
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
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

    int dims[2] = {0, 0};
    MPI_Dims_create(ranks, 2, dims);
    const Block rows = partition(N, dims[0], rank / dims[1]);
    const Block cols = partition(N, dims[1], rank % dims[1]);
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    const bool hasWork = rows.size != 0 && cols.size != 0;
    std::vector<double> A(hasWork ? rows.size * N : 0);
    std::vector<double> B(hasWork ? ((cols.size + 7) / 8) * N * 8 : 0);
    std::vector<double> C(rows.size * cols.size);
    if (hasWork) {
        for (size_t i = 0; i < rows.size; ++i)
            for (size_t k = 0; k < N; ++k)
                A[i * N + k] = getPseudoRndValue(N, rows.begin + i, k);
        for (size_t j = 0; j < cols.size; j += 8)
            for (size_t k = 0; k < N; ++k)
                for (size_t lane = 0; lane < std::min(size_t{8}, cols.size - j); ++lane)
                    B[(j / 8) * N * 8 + k * 8 + lane] = getPseudoRndValue(N, k, cols.begin + j + lane);
    }

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, N, rows.size, cols.size);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / maxElapsed / 1e9);
    }

    if (printResults) {
        if (rank == 0) {
            std::vector<double> full(N * N);
            for (int source = 0; source < ranks; ++source) {
                const Block r = partition(N, dims[0], source / dims[1]);
                const Block c = partition(N, dims[1], source % dims[1]);
                std::vector<double> received;
                const double* block = C.data();
                if (source != 0) {
                    received.resize(r.size * c.size);
                    transfer(received.data(), received.size(), source, false);
                    block = received.data();
                }
                if (c.size != 0)
                    for (size_t i = 0; i < r.size; ++i)
                        std::memcpy(full.data() + (r.begin + i) * N + c.begin,
                                    block + i * c.size, c.size * sizeof(double));
            }
            print_results(full, "MatrixC");
        } else {
            transfer(C.data(), C.size(), 0, true);
        }
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = validateResult(C, N, rows, cols) ? 1 : 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
