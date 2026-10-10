#include <algorithm>
#include <array>
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

#if defined(__AVX__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

// Generate exactly the same input values using global matrix coordinates.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
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

// Called only on rank zero; the outcome and options are broadcast to all ranks.
int parseOptions(int argc, char** argv, size_t& N, bool& validate, bool& printResults) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (value[0] == '-' || end == value || *end != '\0' || errno == ERANGE ||
                parsed > std::numeric_limits<size_t>::max()) {
                fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 2;
            }
            N = static_cast<size_t>(parsed);
            if (N != 0 && N > std::vector<double>().max_size() / N) {
                fprintf(stderr, "Matrix size is too large: %s\n", value);
                return 2;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 1;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 2;
        }
    }
    return 0;
}

// Balanced contiguous output tiles, including empty tiles when ranks exceed N.
size_t blockSize(size_t N, int coordinate, int parts) {
    return N / parts + (static_cast<size_t>(coordinate) < N % parts);
}

size_t blockBegin(size_t N, int coordinate, int parts) {
    return static_cast<size_t>(coordinate) * (N / parts) +
           std::min(static_cast<size_t>(coordinate), N % parts);
}

struct ProcessGrid {
    int rank, size;
    int dimensions[2] = {0, 0};
    int row, column;
    size_t rows, columns, rowBegin, columnBegin;
    MPI_Comm rowComm, columnComm;

    explicit ProcessGrid(size_t N) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Dims_create(size, 2, dimensions);
        row = rank / dimensions[1];
        column = rank % dimensions[1];
        rows = blockSize(N, row, dimensions[0]);
        columns = blockSize(N, column, dimensions[1]);
        rowBegin = blockBegin(N, row, dimensions[0]);
        columnBegin = blockBegin(N, column, dimensions[1]);
        // Keys make communicator ranks equal to the corresponding grid coordinate.
        MPI_Comm_split(MPI_COMM_WORLD, row, column, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, column, row, &columnComm);
    }
};

struct InputPanel {
    size_t width;
    std::vector<double> A, B;
};

std::vector<InputPanel> initMatrices(size_t N, size_t panelWidth, const ProcessGrid& grid) {
    const size_t panelCount = N / panelWidth + (N % panelWidth != 0);
    std::vector<InputPanel> panels(panelCount);
    // Block-cyclic ownership of k panels keeps both input matrices distributed.
    // Initialization needs no communication because the original generator is indexed.
    for (size_t p = 0; p < panelCount; ++p) {
        InputPanel& panel = panels[p];
        const size_t begin = p * panelWidth;
        panel.width = std::min(panelWidth, N - begin);
        if (static_cast<int>(p % grid.dimensions[1]) == grid.column) {
            panel.A.resize(grid.rows * panel.width);
            for (size_t i = 0; i < grid.rows; ++i) {
                for (size_t k = 0; k < panel.width; ++k) {
                    panel.A[i * panel.width + k] = getPseudoRndValue(N, grid.rowBegin + i, begin + k);
                }
            }
        }
        if (static_cast<int>(p % grid.dimensions[0]) == grid.row) {
            panel.B.resize(panel.width * grid.columns);
            // Pack consecutive eight-column microtiles before timing. The
            // inner kernel then streams B instead of accessing widely spaced
            // rows, which avoids cache-set conflicts at power-of-two sizes.
            for (size_t j = 0; j < grid.columns; j += 8) {
                const size_t lanes = std::min<size_t>(8, grid.columns - j);
                for (size_t k = 0; k < panel.width; ++k) {
                    for (size_t lane = 0; lane < lanes; ++lane) {
                        panel.B[j * panel.width + k * lanes + lane] =
                            getPseudoRndValue(N, begin + k, grid.columnBegin + j + lane);
                    }
                }
            }
        }
    }
    return panels;
}

// Fixed register tiles let the compiler SIMD-vectorize independent columns.
// k is always visited in ascending order; there is no floating-point reduction
// across ranks and no reassociation of the dot products.
template <size_t Rows>
inline void multiplyTile(const double* A, const double* B, double* C,
                         size_t width, size_t columns) {
#if defined(__AVX__)
    // Explicit column vectors avoid compilers vectorizing the reduction over k
    // and spilling its partial products. All eight accumulators stay in registers.
    __m256d sums[Rows][2];
    for (size_t i = 0; i < Rows; ++i) {
        sums[i][0] = _mm256_loadu_pd(C + i * columns);
        sums[i][1] = _mm256_loadu_pd(C + i * columns + 4);
    }
    for (size_t k = 0; k < width; ++k) {
        const __m256d b0 = _mm256_loadu_pd(B + k * 8);
        const __m256d b1 = _mm256_loadu_pd(B + k * 8 + 4);
        for (size_t i = 0; i < Rows; ++i) {
            const __m256d a = _mm256_broadcast_sd(A + i * width + k);
            sums[i][0] = _mm256_add_pd(sums[i][0], _mm256_mul_pd(a, b0));
            sums[i][1] = _mm256_add_pd(sums[i][1], _mm256_mul_pd(a, b1));
        }
    }
    for (size_t i = 0; i < Rows; ++i) {
        _mm256_storeu_pd(C + i * columns, sums[i][0]);
        _mm256_storeu_pd(C + i * columns + 4, sums[i][1]);
    }
#else
    double sums[Rows][8];
    for (size_t i = 0; i < Rows; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            sums[i][j] = C[i * columns + j];
        }
    }
    for (size_t k = 0; k < width; ++k) {
        const double* b = B + k * 8;
        for (size_t i = 0; i < Rows; ++i) {
            const double a = A[i * width + k];
            for (size_t j = 0; j < 8; ++j) {
                sums[i][j] += a * b[j];
            }
        }
    }
    for (size_t i = 0; i < Rows; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            C[i * columns + j] = sums[i][j];
        }
    }
#endif
}

void multiplyPanel(const double* A, const double* B, double* C,
                   size_t rows, size_t columns, size_t width,
                   std::vector<MPI_Request>* pending) {
    constexpr size_t rowBlock = 64, columnBlock = 128;
    for (size_t jj = 0; jj < columns; jj += columnBlock) {
        const size_t jEnd = std::min(jj + columnBlock, columns);
        for (size_t ii = 0; ii < rows; ii += rowBlock) {
            const size_t iEnd = std::min(ii + rowBlock, rows);
            size_t j = jj;
            for (; j + 8 <= jEnd; j += 8) {
                size_t i = ii;
                for (; i + 4 <= iEnd; i += 4) {
                    multiplyTile<4>(A + i * width, B + j * width, C + i * columns + j, width, columns);
                }
                for (; i < iEnd; ++i) {
                    multiplyTile<1>(A + i * width, B + j * width, C + i * columns + j, width, columns);
                }
            }
            for (; j < jEnd; ++j) {
                const size_t tailBegin = columns - columns % 8;
                const double* b = B + tailBegin * width + j - tailBegin;
                for (size_t i = ii; i < iEnd; ++i) {
                    double sum = C[i * columns + j];
                    for (size_t k = 0; k < width; ++k) {
                        sum += A[i * width + k] * b[k * (columns % 8)];
                    }
                    C[i * columns + j] = sum;
                }
            }
            // Drive communication on MPI implementations without a background
            // progress thread, once per cache block rather than per microtile.
            if (pending != nullptr) {
                for (MPI_Request& request : *pending) {
                    if (request != MPI_REQUEST_NULL) {
                        int complete;
                        MPI_Test(&request, &complete, MPI_STATUS_IGNORE);
                    }
                }
            }
        }
    }
}

struct BroadcastPanel {
    std::vector<double> A, B;
    std::vector<MPI_Request> requests;
};

// Chunk MPI counts rather than narrowing potentially large size_t products.
void broadcastAsync(double* data, size_t count, int root, MPI_Comm comm,
                    std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
        MPI_Request request;
        MPI_Ibcast(data + offset, chunk, MPI_DOUBLE, root, comm, &request);
        requests.push_back(request);
        offset += chunk;
    }
}

void startBroadcast(size_t p, const InputPanel& input, BroadcastPanel& work, const ProcessGrid& grid) {
    work.requests.clear();
    const int rootA = static_cast<int>(p % grid.dimensions[1]);
    const int rootB = static_cast<int>(p % grid.dimensions[0]);
    if (grid.column == rootA) {
        std::copy(input.A.begin(), input.A.end(), work.A.begin());
    }
    if (grid.row == rootB) {
        std::copy(input.B.begin(), input.B.end(), work.B.begin());
    }
    broadcastAsync(work.A.data(), grid.rows * input.width, rootA, grid.rowComm, work.requests);
    broadcastAsync(work.B.data(), input.width * grid.columns, rootB, grid.columnComm, work.requests);
}

void matrixMultiply(const std::vector<InputPanel>& inputs, std::vector<double>& C,
                    std::array<BroadcastPanel, 2>& work, const ProcessGrid& grid) {
    if (inputs.empty()) {
        return;
    }
    startBroadcast(0, inputs[0], work[0], grid);
    for (size_t p = 0; p < inputs.size(); ++p) {
        BroadcastPanel& current = work[p % 2];
        for (MPI_Request& request : current.requests) {
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        // The next broadcasts can progress while this panel is computed. Never
        // read or overwrite a buffer with an outstanding nonblocking operation.
        if (p + 1 < inputs.size()) {
            startBroadcast(p + 1, inputs[p + 1], work[(p + 1) % 2], grid);
        }
        multiplyPanel(current.A.data(), current.B.data(), C.data(),
                      grid.rows, grid.columns, inputs[p].width,
                      p + 1 < inputs.size() ? &work[(p + 1) % 2].requests : nullptr);
    }
}

bool validateElement(const std::vector<double>& C, size_t N, size_t i, size_t j,
                     const ProcessGrid& grid) {
    double expected = 0.0;
    for (size_t k = 0; k < N; ++k) {
        expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
    }
    const double actual = C[(i - grid.rowBegin) * grid.columns + j - grid.columnBegin];
    const double relError = std::abs((actual - expected) / (expected + 1e-10));
    if (!std::isfinite(actual) || relError > 1e-6) {
        fprintf(stderr, "Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                i, j, expected, actual, relError);
        return false;
    }
    return true;
}

bool validateResult(const std::vector<double>& C, size_t N, const ProcessGrid& grid) {
    int valid = 1;
    if (N != 0) {
        // Preserve the original 25 checks and also sample every nonempty tile.
        for (size_t pi = 0; pi < 5; ++pi) {
            for (size_t pj = 0; pj < 5; ++pj) {
                const size_t i = pi % N, j = pj % N;
                if (i >= grid.rowBegin && i - grid.rowBegin < grid.rows &&
                    j >= grid.columnBegin && j - grid.columnBegin < grid.columns) {
                    valid &= validateElement(C, N, i, j, grid);
                }
            }
        }
        if (grid.rows != 0 && grid.columns != 0) {
            const size_t rowSamples[] = {0, grid.rows / 2, grid.rows - 1};
            const size_t columnSamples[] = {0, grid.columns / 2, grid.columns - 1};
            for (size_t i : rowSamples) {
                for (size_t j : columnSamples) {
                    valid &= validateElement(C, N, grid.rowBegin + i, grid.columnBegin + j, grid);
                }
            }
        }
    }
    int allValid;
    MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return allValid != 0;
}

std::vector<double> gatherResult(const std::vector<double>& local, size_t N, const ProcessGrid& grid) {
    std::vector<double> result;
    if (grid.rank == 0) {
        result.resize(N * N);
        // Rank zero has the largest tile. Reuse one tile-sized receive buffer,
        // avoiding another full matrix and MPI_Gatherv's int displacements.
        std::vector<double> received(local.size());
        for (int rank = 0; rank < grid.size; ++rank) {
            const int row = rank / grid.dimensions[1], column = rank % grid.dimensions[1];
            const size_t rows = blockSize(N, row, grid.dimensions[0]);
            const size_t columns = blockSize(N, column, grid.dimensions[1]);
            const size_t rowBegin = blockBegin(N, row, grid.dimensions[0]);
            const size_t columnBegin = blockBegin(N, column, grid.dimensions[1]);
            const size_t count = rows * columns;
            if (count == 0) {
                continue;
            }
            if (rank != 0) {
                for (size_t offset = 0; offset < count;) {
                    const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(received.data() + offset, chunk, MPI_DOUBLE, rank, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    offset += chunk;
                }
            }
            const double* tile = rank == 0 ? local.data() : received.data();
            for (size_t i = 0; i < rows; ++i) {
                std::copy_n(tile + i * columns, columns,
                            result.data() + (rowBegin + i) * N + columnBegin);
            }
        }
    } else {
        for (size_t offset = 0; offset < local.size();) {
            const int chunk = static_cast<int>(std::min(local.size() - offset, static_cast<size_t>(INT_MAX)));
            MPI_Send(local.data() + offset, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            offset += chunk;
        }
    }
    return result;
}

int runBenchmark(size_t N, bool validate, bool printResults) {
    ProcessGrid grid(N);
    if (grid.rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d (%d x %d grid)\n", grid.size, grid.dimensions[0], grid.dimensions[1]);
        printf("Initializing matrices...\n");
    }
    const size_t maxDimension = static_cast<size_t>(std::max(grid.dimensions[0], grid.dimensions[1]));
    const size_t panelWidth = std::min<size_t>(128, std::max<size_t>(1, N / maxDimension + (N % maxDimension != 0)));
    const auto inputs = initMatrices(N, panelWidth, grid);
    std::vector<double> C(grid.rows * grid.columns, 0.0);
    std::array<BroadcastPanel, 2> work;
    for (BroadcastPanel& panel : work) {
        panel.A.resize(grid.rows * panelWidth);
        panel.B.resize(panelWidth * grid.columns);
        panel.requests.reserve(2 + panel.A.size() / INT_MAX + panel.B.size() / INT_MAX);
    }

    if (grid.rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    matrixMultiply(inputs, C, work, grid);
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (grid.rank == 0) {
        printf("Computation time: %.0f ms\n", std::floor(seconds * 1000.0));
        const double gflops = seconds > 0.0 ? (2.0 * N * N * N) / seconds / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        const auto result = gatherResult(C, N, grid);
        if (grid.rank == 0) {
            print_results(result, "MatrixC");
        }
    }
    bool valid = true;
    if (validate) {
        if (grid.rank == 0) {
            printf("Validating result...\n");
        }
        valid = validateResult(C, N, grid);
        if (grid.rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_Comm_free(&grid.rowComm);
    MPI_Comm_free(&grid.columnComm);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        return 1;
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    size_t N = 512;
    bool validate = false, printResults = false;
    int status = rank == 0 ? parseOptions(argc, argv, N, validate, printResults) : 0;
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0) {
        MPI_Finalize();
        return status == 1 ? 0 : 1;
    }
    unsigned long long options[3] = {static_cast<unsigned long long>(N), validate, printResults};
    MPI_Bcast(options, 3, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(options[0]);
    validate = options[1] != 0;
    printResults = options[2] != 0;
    try {
        status = runBenchmark(N, validate, printResults);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return status;
}
