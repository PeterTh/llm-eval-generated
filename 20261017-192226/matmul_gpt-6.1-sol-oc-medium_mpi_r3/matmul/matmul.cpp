#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <exception>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

size_t boundary(size_t n, int coordinate, int parts) {
    return (n / parts) * coordinate + std::min(n % parts, size_t(coordinate));
}

constexpr size_t panelWidth = 128;

// SIMD across columns preserves the original increasing-k accumulation order.
void multiplyPanel(const double* A, const double* B, double* C,
                   size_t rows, size_t cols, size_t width) {
    for (size_t ii = 0; ii < rows; ii += 32) {
        for (size_t jj = 0; jj < cols; jj += 128) {
            const size_t iEnd = std::min(rows, ii + 32);
            const size_t jEnd = std::min(cols, jj + 128);
            for (size_t k = 0; k < width; ++k) {
                const double* b = B + k * cols;
                for (size_t i = ii; i < iEnd; ++i) {
                    const double a = A[i * panelWidth + k];
                    double* c = C + i * cols;
                    for (size_t j = jj; j < jEnd; ++j) c[j] += a * b[j];
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& C, size_t N,
                    size_t rowStart, size_t colStart, size_t rows, size_t cols) {
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = pi % N;
            const size_t j = pj % N;
            if (i < rowStart || i >= rowStart + rows ||
                j < colStart || j >= colStart + cols) continue;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }
            
            const double actual = C[(i - rowStart) * cols + j - colStart];
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

// Chunk transfers to support matrices larger than MPI's int count limit.
void transfer(double* data, size_t count, int peer, bool send, MPI_Comm comm) {
    for (size_t offset = 0; offset < count;) {
        int chunk = static_cast<int>(std::min(count - offset, size_t(INT_MAX)));
        if (send) MPI_Send(data + offset, chunk, MPI_DOUBLE, peer, 0, comm);
        else MPI_Recv(data + offset, chunk, MPI_DOUBLE, peer, 0, comm, MPI_STATUS_IGNORE);
        offset += chunk;
    }
}

void broadcastPanel(double* data, size_t count, int root, MPI_Comm comm,
                    std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < count;) {
        int chunk = static_cast<int>(std::min(count - offset, size_t(INT_MAX)));
        requests.push_back(MPI_REQUEST_NULL);
        MPI_Ibcast(data + offset, chunk, MPI_DOUBLE, root, comm, &requests.back());
        offset += chunk;
    }
}

void waitPanel(std::vector<MPI_Request>& requests) {
    for (auto& request : requests) MPI_Wait(&request, MPI_STATUS_IGNORE);
    requests.clear();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            errno = 0;
            unsigned long long parsed = std::strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max() / parsed / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", value);
                MPI_Finalize();
                return 1;
            }
            N = static_cast<size_t>(parsed);
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
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    try {
        // Exclude empty blocks when the process count exceeds the matrix size.
        int active = static_cast<int>(std::min(size_t(processes), N * N));
        int dims[2] = {0, 0};
        for (;;) {
            dims[0] = dims[1] = 0;
            MPI_Dims_create(active, 2, dims);
            if (size_t(dims[0]) <= N && size_t(dims[1]) <= N) break;
            --active;
        }
        MPI_Comm work;
        MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &work);
        int valid = 1;
        if (rank < active) {
            MPI_Comm grid, rowComm, colComm;
            int periods[2] = {0, 0};
            MPI_Cart_create(work, 2, dims, periods, 0, &grid);
            int coords[2];
            MPI_Cart_coords(grid, rank, 2, coords);
            MPI_Comm_split(grid, coords[0], coords[1], &rowComm);
            MPI_Comm_split(grid, coords[1], coords[0], &colComm);
            const size_t rowStart = boundary(N, coords[0], dims[0]);
            const size_t colStart = boundary(N, coords[1], dims[1]);
            const size_t rows = boundary(N, coords[0] + 1, dims[0]) - rowStart;
            const size_t cols = boundary(N, coords[1] + 1, dims[1]) - colStart;
            const size_t panels = (N + panelWidth - 1) / panelWidth;
            const size_t aCount = panels / dims[1] + (size_t(coords[1]) < panels % dims[1]);
            const size_t bCount = panels / dims[0] + (size_t(coords[0]) < panels % dims[0]);
            const size_t aStride = rows * panelWidth, bStride = panelWidth * cols;
            std::vector<double> A(aCount * aStride), B(bCount * bStride), C(rows * cols);
            // Cyclic k-panel ownership balances storage without a root scatter.
            for (size_t p = coords[1]; p < panels; p += dims[1]) {
                double* a = A.data() + (p / dims[1]) * aStride;
                for (size_t i = 0; i < rows; ++i)
                    for (size_t k = 0; k < std::min(panelWidth, N - p * panelWidth); ++k)
                        a[i * panelWidth + k] = getPseudoRndValue(N, rowStart + i, p * panelWidth + k);
            }
            for (size_t p = coords[0]; p < panels; p += dims[0]) {
                double* b = B.data() + (p / dims[0]) * bStride;
                for (size_t k = 0; k < std::min(panelWidth, N - p * panelWidth); ++k)
                    for (size_t j = 0; j < cols; ++j)
                        b[k * cols + j] = getPseudoRndValue(N, p * panelWidth + k, colStart + j);
            }
            std::vector<double> aBuffer[2], bBuffer[2];
            std::vector<MPI_Request> requests[2];
            for (int s = 0; s < 2; ++s) {
                aBuffer[s].resize(aStride);
                bBuffer[s].resize(bStride);
            }
            auto post = [&](size_t p) {
                const int s = p % 2;
                const int aRoot = p % dims[1], bRoot = p % dims[0];
                const size_t width = std::min(panelWidth, N - p * panelWidth);
                if (coords[1] == aRoot)
                    std::copy_n(A.data() + (p / dims[1]) * aStride, aStride, aBuffer[s].data());
                if (coords[0] == bRoot)
                    std::copy_n(B.data() + (p / dims[0]) * bStride, width * cols, bBuffer[s].data());
                broadcastPanel(aBuffer[s].data(), aStride, aRoot, rowComm, requests[s]);
                broadcastPanel(bBuffer[s].data(), width * cols, bRoot, colComm, requests[s]);
            };
            if (rank == 0) printf("Computing matrix multiplication...\n");
            MPI_Barrier(grid);
            const double start = MPI_Wtime();
            post(0);
            for (size_t p = 0; p < panels; ++p) {
                const int s = p % 2;
                waitPanel(requests[s]);
                if (p + 1 < panels) post(p + 1);
                multiplyPanel(aBuffer[s].data(), bBuffer[s].data(), C.data(), rows, cols,
                              std::min(panelWidth, N - p * panelWidth));
            }
            const double elapsed = MPI_Wtime() - start;
            double seconds = 0;
            MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
            if (rank == 0) {
                printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
                printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);
            }
            if (printResults) {
                if (rank == 0) {
                    std::vector<double> result(N * N), block;
                    for (int source = 0; source < active; ++source) {
                        int location[2];
                        MPI_Cart_coords(grid, source, 2, location);
                        const size_t r = boundary(N, location[0], dims[0]);
                        const size_t c = boundary(N, location[1], dims[1]);
                        const size_t nr = boundary(N, location[0] + 1, dims[0]) - r;
                        const size_t nc = boundary(N, location[1] + 1, dims[1]) - c;
                        const double* data = C.data();
                        if (source != 0) {
                            block.resize(nr * nc);
                            transfer(block.data(), block.size(), source, false, grid);
                            data = block.data();
                        }
                        for (size_t i = 0; i < nr; ++i)
                            std::copy_n(data + i * nc, nc, result.data() + (r + i) * N + c);
                    }
                    print_results(result, "MatrixC");
                } else {
                    transfer(C.data(), C.size(), 0, true, grid);
                }
            }
            if (validate) {
                if (rank == 0) printf("Validating result...\n");
                int localValid = validateResult(C, N, rowStart, colStart, rows, cols);
                MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, grid);
                if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
            MPI_Comm_free(&rowComm);
            MPI_Comm_free(&colComm);
            MPI_Comm_free(&grid);
            MPI_Comm_free(&work);
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return valid ? 0 : 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
