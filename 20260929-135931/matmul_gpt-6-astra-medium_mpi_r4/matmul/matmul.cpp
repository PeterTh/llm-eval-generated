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

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Balanced contiguous output blocks, including empty blocks when N < grid size.
size_t boundary(size_t n, int part, int parts) {
    return (n / parts) * part + std::min(n % parts, static_cast<size_t>(part));
}

// Each element accumulates in the original increasing-k order. The contiguous
// j loop vectorizes without a floating-point reduction or reassociation.
void matrixMultiply(const double* A, const double* B, double* C,
                    size_t rows, size_t cols, size_t depth) {
    constexpr size_t rowTile = 32, colTile = 128;
    for (size_t ii = 0; ii < rows; ii += rowTile) {
        for (size_t jj = 0; jj < cols; jj += colTile) {
            const size_t ie = std::min(rows, ii + rowTile);
            const size_t je = std::min(cols, jj + colTile);
            for (size_t k = 0; k < depth; ++k) {
                for (size_t i = ii; i < ie; ++i) {
                    const double a = A[i * depth + k];
                    double* c = C + i * cols;
                    const double* b = B + k * cols;
                    for (size_t j = jj; j < je; ++j) {
                        c[j] += a * b[j];
                    }
                }
            }
        }
    }
}

// Chunk transfers to avoid the int count limit of MPI-3 implementations.
void broadcast(double* data, size_t count, int root, MPI_Comm comm,
               std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(count - offset, size_t(INT_MAX)));
        MPI_Request request;
        MPI_Ibcast(data + offset, chunk, MPI_DOUBLE, root, comm, &request);
        requests.push_back(request);
        offset += chunk;
    }
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
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max() ||
                parsed > (std::numeric_limits<size_t>::max() / sizeof(double)) / parsed) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            N = static_cast<size_t>(parsed);
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

    int dims[2] = {0, 0};
    MPI_Dims_create(processes, 2, dims);
    const int row = rank / dims[1], col = rank % dims[1];
    MPI_Comm rowComm, colComm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
    const size_t rowStart = boundary(N, row, dims[0]);
    const size_t colStart = boundary(N, col, dims[1]);
    const size_t rows = boundary(N, row + 1, dims[0]) - rowStart;
    const size_t cols = boundary(N, col + 1, dims[1]) - colStart;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    // SUMMA: distribute A panels cyclically across process columns and B
    // panels across process rows. Deterministic initialization needs no scatter.
    constexpr size_t panelWidth = 128;
    const size_t stages = (N + panelWidth - 1) / panelWidth;
    std::vector<std::vector<double>> A(stages), B(stages);
    std::vector<double> C(rows * cols, 0.0);
    for (size_t s = 0; s < stages; ++s) {
        const size_t k0 = s * panelWidth;
        const size_t depth = std::min(panelWidth, N - k0);
        if (s % dims[1] == static_cast<size_t>(col)) {
            A[s].resize(rows * depth);
            for (size_t i = 0; i < rows; ++i)
                for (size_t k = 0; k < depth; ++k)
                    A[s][i * depth + k] = getPseudoRndValue(N, rowStart + i, k0 + k);
        }
        if (s % dims[0] == static_cast<size_t>(row)) {
            B[s].resize(depth * cols);
            for (size_t k = 0; k < depth; ++k)
                for (size_t j = 0; j < cols; ++j)
                    B[s][k * cols + j] = getPseudoRndValue(N, k0 + k, colStart + j);
        }
    }

    // Double buffering allows the next panel's communication to overlap work.
    std::vector<double> recvA[2], recvB[2];
    std::vector<MPI_Request> requests[2];
    double* panelA[2];
    double* panelB[2];
    for (int slot = 0; slot < 2; ++slot) {
        recvA[slot].resize(rows * std::min(N, panelWidth));
        recvB[slot].resize(cols * std::min(N, panelWidth));
    }
    auto startPanel = [&](size_t s) {
        const size_t slot = s % 2;
        const size_t depth = std::min(panelWidth, N - s * panelWidth);
        const int aRoot = static_cast<int>(s % dims[1]);
        const int bRoot = static_cast<int>(s % dims[0]);
        panelA[slot] = col == aRoot ? A[s].data() : recvA[slot].data();
        panelB[slot] = row == bRoot ? B[s].data() : recvB[slot].data();
        requests[slot].clear();
        broadcast(panelA[slot], rows * depth, aRoot, rowComm, requests[slot]);
        broadcast(panelB[slot], cols * depth, bRoot, colComm, requests[slot]);
    };

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    startPanel(0);
    for (size_t s = 0; s < stages; ++s) {
        auto& pending = requests[s % 2];
        for (auto& request : pending) MPI_Wait(&request, MPI_STATUS_IGNORE);
        if (s + 1 < stages) startPanel(s + 1);
        matrixMultiply(panelA[s % 2], panelB[s % 2], C.data(), rows, cols,
                       std::min(panelWidth, N - s * panelWidth));
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", duration * 1000.0);
        const double gflops = duration > 0 ? (2.0 * N * N * N) / duration / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Only external result reporting requires a global matrix. Assemble in
    // row-major order to preserve the original statistics, samples and hash.
    if (printResults) {
        std::vector<double> result;
        if (rank == 0) result.resize(N * N);
        for (int source = 0; source < processes; ++source) {
            const int sr = source / dims[1], sc = source % dims[1];
            const size_t r0 = boundary(N, sr, dims[0]), c0 = boundary(N, sc, dims[1]);
            const size_t nr = boundary(N, sr + 1, dims[0]) - r0;
            const size_t nc = boundary(N, sc + 1, dims[1]) - c0;
            std::vector<double> received;
            if (rank == 0 && source != 0) received.resize(nr * nc);
            for (size_t offset = 0; offset < nr * nc;) {
                const int count = static_cast<int>(std::min(nr * nc - offset, size_t(INT_MAX)));
                if (source != 0) {
                    if (rank == source)
                        MPI_Send(C.data() + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    if (rank == 0)
                        MPI_Recv(received.data() + offset, count, MPI_DOUBLE, source, 0,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                offset += count;
            }
            if (rank == 0 && nc != 0) {
                const auto& block = source == 0 ? C : received;
                for (size_t i = 0; i < nr; ++i)
                    std::copy_n(block.data() + i * nc, nc, result.data() + (r0 + i) * N + c0);
            }
        }
        if (rank == 0) print_results(result, "MatrixC");
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        for (size_t pi = 0; pi < 5; ++pi) {
            for (size_t pj = 0; pj < 5; ++pj) {
                const size_t i = pi % N, j = pj % N;
                if (i < rowStart || i >= rowStart + rows || j < colStart || j >= colStart + cols)
                    continue;
                double expected = 0.0;
                for (size_t k = 0; k < N; ++k)
                    expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                const double actual = C[(i - rowStart) * cols + j - colStart];
                const double error = std::abs((actual - expected) / (expected + 1e-10));
                if (!std::isfinite(actual) || error > 1e-6) {
                    printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                           i, j, expected, actual, error);
                    valid = 0;
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
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
