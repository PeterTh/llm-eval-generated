#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <mpi.h>
#include "../common/results_output.hpp"

// Use the original generator, including its size_t arithmetic.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Balanced contiguous partitions, including empty partitions for small matrices.
size_t boundary(size_t N, int part, int parts) {
    return (N / parts) * part + std::min(static_cast<size_t>(part), N % parts);
}

constexpr size_t panelSize = 128;

// Chunk transfers so MPI's int count does not limit the matrix dimensions.
void broadcast(std::vector<double>& data, int root, MPI_Comm comm) {
    for (size_t offset = 0; offset < data.size();) {
        int count = static_cast<int>(std::min(data.size() - offset, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(data.data() + offset, count, MPI_DOUBLE, root, comm);
        offset += count;
    }
}

// Four rows reuse each B vector. Contiguous columns permit SIMD without
// reordering the k accumulation of any output element.
void multiplyPanel(const double* A, const double* B, double* C,
                   size_t rows, size_t cols, size_t depth) {
    constexpr size_t columnBlock = 128;
    for (size_t jj = 0; jj < cols; jj += columnBlock) {
        const size_t end = std::min(jj + columnBlock, cols);
        size_t i = 0;
        for (; i + 3 < rows; i += 4) {
            double* c0 = C + i * cols;
            double* c1 = c0 + cols;
            double* c2 = c1 + cols;
            double* c3 = c2 + cols;
            for (size_t k = 0; k < depth; ++k) {
                const double a0 = A[i * depth + k];
                const double a1 = A[(i + 1) * depth + k];
                const double a2 = A[(i + 2) * depth + k];
                const double a3 = A[(i + 3) * depth + k];
                const double* b = B + k * cols;
                for (size_t j = jj; j < end; ++j) {
                    c0[j] += a0 * b[j];
                    c1[j] += a1 * b[j];
                    c2[j] += a2 * b[j];
                    c3[j] += a3 * b[j];
                }
            }
        }
        for (; i < rows; ++i) {
            double* c = C + i * cols;
            for (size_t k = 0; k < depth; ++k) {
                const double a = A[i * depth + k];
                const double* b = B + k * cols;
                for (size_t j = jj; j < end; ++j) c[j] += a * b[j];
            }
        }
    }
}

bool validateResult(const std::vector<double>& C, size_t N,
                    size_t rowBegin, size_t colBegin, size_t rows, size_t cols) {
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = pi % N, j = pj % N;
            if (i < rowBegin || i >= rowBegin + rows ||
                j < colBegin || j >= colBegin + cols) continue;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k)
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            const double actual = C[(i - rowBegin) * cols + j - colBegin];
            const double error = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || error > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, error);
                return false;
            }
        }
    }
    return true;
}

// Only -r needs the full result. Reassemble tiles in the original row-major
// order, preserving the ordered statistics and hash in print_results.
void printDistributedResult(const std::vector<double>& C, size_t N, int rank,
                            const int dims[2], MPI_Comm grid) {
    std::vector<double> result;
    if (rank == 0) result.resize(N * N);
    for (int source = 0; source < dims[0] * dims[1]; ++source) {
        int coord[2];
        MPI_Cart_coords(grid, source, 2, coord);
        size_t rb = boundary(N, coord[0], dims[0]);
        size_t cb = boundary(N, coord[1], dims[1]);
        size_t rows = boundary(N, coord[0] + 1, dims[0]) - rb;
        size_t cols = boundary(N, coord[1] + 1, dims[1]) - cb;
        if (rank == 0) {
            std::vector<double> received;
            if (source != 0) received.resize(rows * cols);
            const double* tile = source == 0 ? C.data() : received.data();
            for (size_t offset = 0; source != 0 && offset < received.size();) {
                int count = static_cast<int>(std::min(received.size() - offset, static_cast<size_t>(INT_MAX)));
                MPI_Recv(received.data() + offset, count, MPI_DOUBLE, source, 0, grid, MPI_STATUS_IGNORE);
                offset += count;
            }
            if (cols != 0) {
                for (size_t i = 0; i < rows; ++i)
                    std::copy_n(tile + i * cols, cols, result.data() + (rb + i) * N + cb);
            }
        } else if (rank == source) {
            for (size_t offset = 0; offset < C.size();) {
                int count = static_cast<int>(std::min(C.size() - offset, static_cast<size_t>(INT_MAX)));
                MPI_Send(C.data() + offset, count, MPI_DOUBLE, 0, 0, grid);
                offset += count;
            }
        }
    }
    if (rank == 0) print_results(result, "MatrixC");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank, int processes) {
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* text = argv[++i];
            char* end = nullptr;
            errno = 0;
            unsigned long long value = std::strtoull(text, &end, 10);
            if (errno || text[0] == '-' || end == text || *end || value == 0 ||
                value > std::numeric_limits<size_t>::max() ||
                value > std::numeric_limits<size_t>::max() / value / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", text);
                return 1;
            }
            N = static_cast<size_t>(value);
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

    int dims[2] = {0, 0}, periods[2] = {0, 0}, coord[2];
    MPI_Dims_create(processes, 2, dims);
    MPI_Comm grid, rowComm, colComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
    MPI_Cart_coords(grid, rank, 2, coord);
    MPI_Comm_split(grid, coord[0], coord[1], &rowComm);
    MPI_Comm_split(grid, coord[1], coord[0], &colComm);
    const size_t rb = boundary(N, coord[0], dims[0]);
    const size_t cb = boundary(N, coord[1], dims[1]);
    const size_t rows = boundary(N, coord[0] + 1, dims[0]) - rb;
    const size_t cols = boundary(N, coord[1] + 1, dims[1]) - cb;
    const size_t stages = (N - 1) / panelSize + 1;
    std::vector<double> C(rows * cols, 0.0);
    std::vector<std::vector<double>> A(stages), B(stages);
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    // Deterministic initialization at panel owners avoids a root scatter and
    // replicated NxN inputs. Panel ownership cycles through each communicator.
    for (size_t s = 0; s < stages; ++s) {
        const size_t kb = s * panelSize, depth = std::min(panelSize, N - kb);
        if (s % dims[1] == static_cast<size_t>(coord[1])) {
            A[s].resize(rows * depth);
            for (size_t i = 0; i < rows; ++i)
                for (size_t k = 0; k < depth; ++k)
                    A[s][i * depth + k] = getPseudoRndValue(N, rb + i, kb + k);
        }
        if (s % dims[0] == static_cast<size_t>(coord[0])) {
            B[s].resize(depth * cols);
            for (size_t k = 0; k < depth; ++k)
                for (size_t j = 0; j < cols; ++j)
                    B[s][k * cols + j] = getPseudoRndValue(N, kb + k, cb + j);
        }
    }
    std::vector<double> aPanel(rows * std::min(panelSize, N));
    std::vector<double> bPanel(cols * std::min(panelSize, N));
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(grid);
    const double start = MPI_Wtime();
    for (size_t s = 0; s < stages; ++s) {
        const size_t depth = std::min(panelSize, N - s * panelSize);
        aPanel.resize(rows * depth);
        bPanel.resize(depth * cols);
        const int aRoot = static_cast<int>(s % dims[1]);
        const int bRoot = static_cast<int>(s % dims[0]);
        auto& a = coord[1] == aRoot ? A[s] : aPanel;
        auto& b = coord[0] == bRoot ? B[s] : bPanel;
        broadcast(a, aRoot, rowComm);
        broadcast(b, bRoot, colComm);
        if (rows && cols) multiplyPanel(a.data(), b.data(), C.data(), rows, cols, depth);
    }
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
    }
    if (printResults) printDistributedResult(C, N, rank, dims, grid);
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int localValid = validateResult(C, N, rb, cb, rows, cols) ? 1 : 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, grid);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&grid);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    int status = 1;
    try {
        status = run(argc, argv, rank, processes);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
