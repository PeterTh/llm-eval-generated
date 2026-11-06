#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// The inputs are deterministic, so every rank can generate its own slices.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

struct Slice {
    size_t begin;
    size_t size;
};

Slice partition(size_t n, int parts, int index) {
    const size_t base = n / static_cast<size_t>(parts);
    const size_t extra = n % static_cast<size_t>(parts);
    const size_t idx = static_cast<size_t>(index);
    return {idx * base + std::min(idx, extra), base + (idx < extra)};
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Each element is accumulated in increasing k order. Blocking the columns
// keeps the output tile hot while SIMD processes independent output elements.
// The original dot product uses separate multiply/add operations for pairs
// of k values and, on FMA-capable builds, a fused operation for an odd tail.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t rows, size_t cols, size_t N) {
    constexpr size_t columnBlock = 128;
    constexpr size_t rowBlock = 4;
    for (size_t ib = 0; ib < rows; ib += rowBlock) {
        const size_t iend = std::min(ib + rowBlock, rows);
        for (size_t jb = 0; jb < cols; jb += columnBlock) {
            const size_t jend = std::min(jb + columnBlock, cols);
            for (size_t k = 0; k < (N & ~size_t(1)); ++k) {
                const double* b = B.data() + k * cols;
                for (size_t i = ib; i < iend; ++i) {
                    const double a = A[i * N + k];
                    double* c = C.data() + i * cols;
                    for (size_t j = jb; j < jend; ++j) {
                        c[j] += a * b[j];
                    }
                }
            }
            if (N % 2 != 0) {
                const size_t k = N - 1;
                const double* b = B.data() + k * cols;
                for (size_t i = ib; i < iend; ++i) {
                    const double a = A[i * N + k];
                    double* c = C.data() + i * cols;
                    for (size_t j = jb; j < jend; ++j) {
#if defined(__FMA__)
                        c[j] = std::fma(a, b[j], c[j]);
#else
                        c[j] += a * b[j];
#endif
                    }
                }
            }
        }
    }
}

// MPI's traditional count parameter is an int; transfer long rows in chunks.
void transferRow(double* data, size_t count, int peer, bool send) {
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
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    bool earlyExit = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            earlyExit = true;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            earlyExit = true;
            break;
        }
    }
    if (earlyExit) {
        MPI_Finalize();
        return exitCode;
    }

    int dims[2] = {0, 0};
    MPI_Dims_create(ranks, 2, dims);
    const Slice rows = partition(N, dims[0], rank / dims[1]);
    const Slice cols = partition(N, dims[1], rank % dims[1]);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    std::vector<double> A(rows.size * N);
    std::vector<double> B(N * cols.size);
    std::vector<double> C(rows.size * cols.size, 0.0);
    for (size_t i = 0; i < rows.size; ++i)
        for (size_t k = 0; k < N; ++k)
            A[i * N + k] = getPseudoRndValue(N, rows.begin + i, k);
    for (size_t k = 0; k < N; ++k)
        for (size_t j = 0; j < cols.size; ++j)
            B[k * cols.size + j] = getPseudoRndValue(N, k, cols.begin + j);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(A, B, C, rows.size, cols.size, N);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double gflops = (2.0 * N * N * N) / maxElapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> result;
        if (rank == 0) {
            result.resize(N * N);
            for (int source = 0; source < ranks; ++source) {
                const Slice sr = partition(N, dims[0], source / dims[1]);
                const Slice sc = partition(N, dims[1], source % dims[1]);
                for (size_t i = 0; i < sr.size; ++i) {
                    if (sc.size == 0) continue;
                    double* destination = result.data() + (sr.begin + i) * N + sc.begin;
                    if (source == 0)
                        std::copy_n(C.data() + i * cols.size, sc.size, destination);
                    else
                        transferRow(destination, sc.size, source, false);
                }
            }
            print_results(result, "MatrixC");
        } else if (cols.size != 0) {
            for (size_t i = 0; i < rows.size; ++i)
                transferRow(C.data() + i * cols.size, cols.size, 0, true);
        }
    }

    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int firstFailure = 25;
        double failureActual = 0.0;
        if (N != 0) {
            for (int p = 0; p < 25; ++p) {
                const size_t i = static_cast<size_t>(p / 5) % N;
                const size_t j = static_cast<size_t>(p % 5) % N;
                if (i < rows.begin || i >= rows.begin + rows.size ||
                    j < cols.begin || j >= cols.begin + cols.size) continue;
                double expected = 0.0;
                for (size_t k = 0; k < N; ++k)
                    expected += A[(i - rows.begin) * N + k] * B[k * cols.size + j - cols.begin];
                const double actual = C[(i - rows.begin) * cols.size + j - cols.begin];
                if (std::abs((actual - expected) / (expected + 1e-10)) > 1e-6) {
                    firstFailure = p;
                    failureActual = actual;
                    break;
                }
            }
        } else {
            firstFailure = 0;
        }
        int globalFailure;
        MPI_Allreduce(&firstFailure, &globalFailure, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (globalFailure != 25) {
            double actual = firstFailure == globalFailure ? failureActual : 0.0;
            double globalActual = 0.0;
            MPI_Reduce(&actual, &globalActual, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                if (N != 0) {
                    const size_t i = static_cast<size_t>(globalFailure / 5) % N;
                    const size_t j = static_cast<size_t>(globalFailure % 5) % N;
                    double expected = 0.0;
                    for (size_t k = 0; k < N; ++k)
                        expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                    const double relError = std::abs((globalActual - expected) / (expected + 1e-10));
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, globalActual, relError);
                }
                printf("Validation: FAILED\n");
            }
            exitCode = 1;
        } else if (rank == 0) {
            printf("Validation: PASSED\n");
        }
    }
    MPI_Finalize();
    return exitCode;
}
