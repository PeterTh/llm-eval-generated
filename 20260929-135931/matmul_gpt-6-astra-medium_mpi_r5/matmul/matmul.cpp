#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Balanced contiguous partitions, including empty blocks for very small matrices.
size_t boundary(size_t n, int coordinate, int parts) {
    return (n / parts) * coordinate + std::min(n % parts, size_t(coordinate));
}

constexpr size_t panelWidth = 128;

template <size_t tileRows>
void multiplyTile(const double* a, const double* b, double* c,
                  size_t cols, size_t depth) {
    // Keep a small output tile in SIMD registers for the entire panel.
    double sums[tileRows][8];
    for (size_t i = 0; i < tileRows; ++i)
        for (size_t j = 0; j < 8; ++j)
            sums[i][j] = c[i * cols + j];
    for (size_t k = 0; k < depth; ++k)
        for (size_t i = 0; i < tileRows; ++i) {
            const double aik = a[i * depth + k];
            for (size_t j = 0; j < 8; ++j)
                sums[i][j] += aik * b[k * cols + j];
        }
    for (size_t i = 0; i < tileRows; ++i)
        for (size_t j = 0; j < 8; ++j)
            c[i * cols + j] = sums[i][j];
}

void multiplyPanel(const double* a, const double* b, double* c,
                   size_t rows, size_t cols, size_t depth) {
    // Vectorize across columns, keeping the k accumulation in its original order.
    for (size_t ii = 0; ii < rows; ii += 32) {
        for (size_t jj = 0; jj < cols; jj += 128) {
            const size_t iend = std::min(rows, ii + 32);
            const size_t jend = std::min(cols, jj + 128);
            size_t i = ii;
            for (; i < iend; ) {
                const size_t tileRows = i + 4 <= iend ? 4 : 1;
                size_t j = jj;
                for (; j + 8 <= jend; j += 8) {
                    if (tileRows == 4)
                        multiplyTile<4>(a + i * depth, b + j, c + i * cols + j, cols, depth);
                    else
                        multiplyTile<1>(a + i * depth, b + j, c + i * cols + j, cols, depth);
                }
                for (; j < jend; ++j)
                    for (size_t r = 0; r < tileRows; ++r) {
                        double sum = c[(i + r) * cols + j];
                        for (size_t k = 0; k < depth; ++k)
                            sum += a[(i + r) * depth + k] * b[k * cols + j];
                        c[(i + r) * cols + j] = sum;
                    }
                i += tileRows;
            }
        }
    }
}

// Chunk communication to avoid the int count limit in MPI implementations.
void broadcast(double* data, size_t count, int root, MPI_Comm comm,
               std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < count;) {
        int chunk = int(std::min(count - offset, size_t(INT_MAX)));
        requests.emplace_back();
        MPI_Ibcast(data + offset, chunk, MPI_DOUBLE, root, comm, &requests.back());
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    size_t N = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            errno = 0;
            unsigned long long value = strtoull(arg, &end, 10);
            if (errno || *arg == '-' || end == arg || *end || value == 0 ||
                value > std::numeric_limits<size_t>::max() ||
                value > (std::numeric_limits<size_t>::max() / sizeof(double)) / value) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", arg);
                MPI_Finalize();
                return 1;
            }
            N = size_t(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else {
            bool help = strcmp(argv[i], "-h") == 0;
            if (rank == 0) {
                if (!help) printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return help ? 0 : 1;
        }
    }

    try {
        int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2];
        MPI_Dims_create(processes, 2, dims);
        MPI_Comm grid, rowComm, colComm;
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
        MPI_Cart_coords(grid, rank, 2, coords);
        MPI_Comm_split(grid, coords[0], coords[1], &rowComm);
        MPI_Comm_split(grid, coords[1], coords[0], &colComm);
        const size_t rowStart = boundary(N, coords[0], dims[0]);
        const size_t colStart = boundary(N, coords[1], dims[1]);
        const size_t rows = boundary(N, coords[0] + 1, dims[0]) - rowStart;
        const size_t cols = boundary(N, coords[1] + 1, dims[1]) - colStart;
        const size_t panels = (N + panelWidth - 1) / panelWidth;
        std::vector<double> C(rows * cols, 0.0);
        std::vector<std::vector<double>> A(panels), B(panels);
        if (rank == 0) {
            printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", N, N);
            printf("Validation: %s\nInitializing matrices...\n", validate ? "enabled" : "disabled");
        }
        // Each input panel has one owner per process row/column. Generate it
        // directly there using global indices; no centralized initialization.
        for (size_t p = 0; p < panels; ++p) {
            size_t begin = p * panelWidth, depth = std::min(panelWidth, N - begin);
            if (size_t(coords[1]) == p % dims[1]) {
                A[p].resize(rows * depth);
                for (size_t i = 0; i < rows; ++i)
                    for (size_t k = 0; k < depth; ++k)
                        A[p][i * depth + k] = getPseudoRndValue(N, rowStart + i, begin + k);
            }
            if (size_t(coords[0]) == p % dims[0]) {
                B[p].resize(depth * cols);
                for (size_t k = 0; k < depth; ++k)
                    for (size_t j = 0; j < cols; ++j)
                        B[p][k * cols + j] = getPseudoRndValue(N, begin + k, colStart + j);
            }
        }
        std::array<std::vector<double>, 2> recvA, recvB;
        std::array<std::vector<MPI_Request>, 2> requests;
        std::array<double*, 2> a{}, b{};
        for (int slot = 0; slot < 2; ++slot) {
            recvA[slot].resize(rows * std::min(N, panelWidth));
            recvB[slot].resize(cols * std::min(N, panelWidth));
        }
        auto post = [&](size_t p) {
            size_t slot = p % 2, depth = std::min(panelWidth, N - p * panelWidth);
            a[slot] = size_t(coords[1]) == p % dims[1] ? A[p].data() : recvA[slot].data();
            b[slot] = size_t(coords[0]) == p % dims[0] ? B[p].data() : recvB[slot].data();
            requests[slot].clear();
            broadcast(a[slot], rows * depth, int(p % dims[1]), rowComm, requests[slot]);
            broadcast(b[slot], cols * depth, int(p % dims[0]), colComm, requests[slot]);
        };
        if (rank == 0) printf("Computing matrix multiplication...\n");
        MPI_Barrier(grid);
        double start = MPI_Wtime();
        post(0);
        for (size_t p = 0; p < panels; ++p) {
            size_t slot = p % 2;
            for (auto& request : requests[slot]) MPI_Wait(&request, MPI_STATUS_IGNORE);
            // Overlap the next panel's communication with this panel's kernel.
            if (p + 1 < panels) post(p + 1);
            multiplyPanel(a[slot], b[slot], C.data(), rows, cols,
                          std::min(panelWidth, N - p * panelWidth));
        }
        double elapsed = MPI_Wtime() - start, duration;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
        if (rank == 0) {
            printf("Computation time: %.0f ms\n", std::floor(duration * 1000));
            printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / duration / 1e9);
        }
        if (printResults) {
            // Reassemble in original row-major order only for external output.
            std::vector<double> full, block;
            if (rank == 0) full.resize(N * N);
            for (int source = 0; source < processes; ++source) {
                int xy[2];
                MPI_Cart_coords(grid, source, 2, xy);
                size_t r0 = boundary(N, xy[0], dims[0]), c0 = boundary(N, xy[1], dims[1]);
                size_t nr = boundary(N, xy[0] + 1, dims[0]) - r0;
                size_t nc = boundary(N, xy[1] + 1, dims[1]) - c0;
                if (rank == 0 && source != 0) block.resize(nr * nc);
                for (size_t offset = 0; offset < nr * nc;) {
                    int count = int(std::min(nr * nc - offset, size_t(INT_MAX)));
                    if (source != 0 && rank == source)
                        MPI_Send(C.data() + offset, count, MPI_DOUBLE, 0, 0, grid);
                    if (source != 0 && rank == 0)
                        MPI_Recv(block.data() + offset, count, MPI_DOUBLE, source, 0, grid, MPI_STATUS_IGNORE);
                    offset += count;
                }
                if (rank == 0 && nc != 0) {
                    const auto& data = source == 0 ? C : block;
                    for (size_t i = 0; i < nr; ++i)
                        std::copy_n(data.data() + i * nc, nc, full.data() + (r0 + i) * N + c0);
                }
            }
            if (rank == 0) print_results(full, "MatrixC");
        }
        int valid = 1, allValid = 1;
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            for (size_t i = rowStart; i < std::min(rowStart + rows, std::min(N, size_t(5))); ++i) {
                for (size_t j = colStart; j < std::min(colStart + cols, std::min(N, size_t(5))); ++j) {
                    double expected = 0.0;
                    for (size_t k = 0; k < N; ++k)
                        expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
                    double actual = C[(i - rowStart) * cols + j - colStart];
                    double error = std::abs((actual - expected) / (expected + 1e-10));
                    if (!std::isfinite(actual) || error > 1e-6) {
                        printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                               i, j, expected, actual, error);
                        valid = 0;
                    }
                }
            }
            MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, grid);
            if (rank == 0) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
        }
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
        MPI_Comm_free(&grid);
        MPI_Finalize();
        return allValid ? 0 : 1;
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
