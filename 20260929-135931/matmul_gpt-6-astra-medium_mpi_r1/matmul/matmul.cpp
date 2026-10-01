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

// Balanced contiguous blocks, including empty blocks when there are more ranks
// than matrix rows or columns.
size_t blockStart(size_t n, int coordinate, int parts) {
    return (n / parts) * coordinate + std::min(n % parts, size_t(coordinate));
}

// Each output element accumulates k in its original order. Blocking across
// independent rows and columns improves locality without a floating-point reduction.
void multiplyPanel(const double* a, const double* b, double* c,
                   size_t rows, size_t cols, size_t depth) {
    constexpr size_t rowTile = 32, colTile = 256;
    for (size_t ii = 0; ii < rows; ii += rowTile) {
        for (size_t jj = 0; jj < cols; jj += colTile) {
            const size_t iend = std::min(rows, ii + rowTile);
            const size_t jend = std::min(cols, jj + colTile);
            for (size_t k = 0; k < depth; ++k) {
                for (size_t i = ii; i < iend; ++i) {
                    const double av = a[i * depth + k];
                    double* __restrict__ out = c + i * cols;
                    const double* __restrict__ in = b + k * cols;
                    for (size_t j = jj; j < jend; ++j)
                        out[j] += av * in[j];
                }
            }
        }
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

int run(int argc, char** argv, int rank, int ranks) {
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value == end || *end || *value == '-' || !parsed ||
                parsed > std::numeric_limits<size_t>::max()) {
                if (!rank) fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            N = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (N > std::numeric_limits<size_t>::max() / N / sizeof(double)) {
        if (!rank) fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }

    int dims[2] = {0, 0};
    MPI_Dims_create(ranks, 2, dims);
    const int row = rank / dims[1], col = rank % dims[1];
    MPI_Comm rowComm, colComm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);
    const size_t rowStart = blockStart(N, row, dims[0]);
    const size_t colStart = blockStart(N, col, dims[1]);
    const size_t rows = blockStart(N, row + 1, dims[0]) - rowStart;
    const size_t cols = blockStart(N, col + 1, dims[1]) - colStart;
    // Bound every panel broadcast by MPI's int count limit.
    const size_t largestBlock = std::max((N - 1) / dims[0] + 1,
                                         (N - 1) / dims[1] + 1);
    if (largestBlock > INT_MAX) {
        if (!rank) fprintf(stderr, "Matrix block exceeds MPI count capacity\n");
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
        return 1;
    }
    const size_t width = std::min(size_t(128), size_t(INT_MAX) / largestBlock);
    const size_t steps = (N - 1) / width + 1;
    std::vector<double> C(rows * cols, 0.0);
    // A panels are distributed cyclically across process columns; B panels
    // across process rows. Deterministic initialization needs no root scatter.
    std::vector<std::vector<double>> ownedA, ownedB;
    if (!rank) {
        printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\nInitializing matrices...\n", validate ? "enabled" : "disabled");
    }
    for (size_t s = 0; s < steps; ++s) {
        const size_t begin = s * width, depth = std::min(width, N - begin);
        if (s % dims[1] == size_t(col)) {
            ownedA.emplace_back(rows * depth);
            auto& a = ownedA.back();
            for (size_t i = 0; i < rows; ++i)
                for (size_t k = 0; k < depth; ++k)
                    a[i * depth + k] = getPseudoRndValue(N, rowStart + i, begin + k);
        }
        if (s % dims[0] == size_t(row)) {
            ownedB.emplace_back(depth * cols);
            auto& b = ownedB.back();
            for (size_t k = 0; k < depth; ++k)
                for (size_t j = 0; j < cols; ++j)
                    b[k * cols + j] = getPseudoRndValue(N, begin + k, colStart + j);
        }
    }
    std::vector<double> a[2], b[2];
    for (int slot = 0; slot < 2; ++slot) {
        a[slot].resize(std::max(size_t(1), rows * width));
        b[slot].resize(std::max(size_t(1), cols * width));
    }
    MPI_Request requests[2][2];
    auto startPanel = [&](size_t s) {
        const int slot = s % 2;
        const size_t depth = std::min(width, N - s * width);
        const int aRoot = s % dims[1], bRoot = s % dims[0];
        if (col == aRoot)
            std::copy(ownedA[s / dims[1]].begin(), ownedA[s / dims[1]].end(), a[slot].begin());
        if (row == bRoot)
            std::copy(ownedB[s / dims[0]].begin(), ownedB[s / dims[0]].end(), b[slot].begin());
        MPI_Ibcast(a[slot].data(), int(rows * depth), MPI_DOUBLE, aRoot, rowComm, &requests[slot][0]);
        MPI_Ibcast(b[slot].data(), int(cols * depth), MPI_DOUBLE, bRoot, colComm, &requests[slot][1]);
    };
    if (!rank) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    startPanel(0);
    for (size_t s = 0; s < steps; ++s) {
        const int slot = s % 2;
        MPI_Waitall(2, requests[slot], MPI_STATUSES_IGNORE);
        // Double buffering allows the next broadcasts to progress during GEMM.
        if (s + 1 < steps) startPanel(s + 1);
        multiplyPanel(a[slot].data(), b[slot].data(), C.data(), rows, cols,
                      std::min(width, N - s * width));
    }
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %.0f ms\n", std::floor(seconds * 1000));
        printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
    }

    // Preserve row-major statistics, samples, and hash exactly by gathering only
    // when requested. Chunk transfers to support results larger than INT_MAX.
    if (printResults) {
        std::vector<double> full;
        if (!rank) full.resize(N * N);
        for (int source = 0; source < ranks; ++source) {
            const int sr = source / dims[1], sc = source % dims[1];
            const size_t r0 = blockStart(N, sr, dims[0]), c0 = blockStart(N, sc, dims[1]);
            const size_t nr = blockStart(N, sr + 1, dims[0]) - r0;
            const size_t nc = blockStart(N, sc + 1, dims[1]) - c0;
            std::vector<double> received;
            if (!rank && source) received.resize(nr * nc);
            for (size_t offset = 0; offset < nr * nc;) {
                const int count = int(std::min(size_t(INT_MAX), nr * nc - offset));
                if (source && rank == source)
                    MPI_Send(C.data() + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                if (!rank && source)
                    MPI_Recv(received.data() + offset, count, MPI_DOUBLE, source, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += count;
            }
            if (!rank && nc) {
                const auto& block = source ? received : C;
                for (size_t i = 0; i < nr; ++i)
                    std::copy_n(block.data() + i * nc, nc, full.data() + (r0 + i) * N + c0);
            }
        }
        if (!rank) print_results(full, "MatrixC");
    }
    int valid = 1;
    if (validate) {
        if (!rank) printf("Validating result...\n");
        for (size_t i = rowStart; i < std::min(rowStart + rows, std::min(N, size_t(5))); ++i) {
            for (size_t j = colStart; j < std::min(colStart + cols, std::min(N, size_t(5))); ++j) {
                double expected = 0.0;
                for (size_t k = 0; k < N; ++k)
                    expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                const double actual = C[(i - rowStart) * cols + j - colStart];
                if (!std::isfinite(actual) || std::abs((actual - expected) / (expected + 1e-10)) > 1e-6)
                    valid = 0;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!rank) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
