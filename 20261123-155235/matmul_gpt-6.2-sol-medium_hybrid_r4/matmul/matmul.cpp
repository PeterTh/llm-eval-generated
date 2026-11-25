#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

namespace {

__host__ __device__ double getPseudoRndValue(size_t nSquared, size_t i, size_t j) {
    const double numerator = ((i + 1) * (i + j + 1) * 1299709) % nSquared;
#ifdef __CUDA_ARCH__
    return __ddiv_rn(numerator, static_cast<double>(nSquared));
#else
    return numerator / static_cast<double>(nSquared);
#endif
}

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

// Assign each GPU to one rank that can see it, including scheduler-masked devices.
std::vector<int> assignDevices(MPI_Comm node) {
    int rank, ranks;
    MPI_Comm_rank(node, &rank);
    MPI_Comm_size(node, &ranks);

    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status != cudaErrorNoDevice) {
        checkCuda(status);
    }

    using Uuid = std::array<char, 16>;
    constexpr int uuidBytes = sizeof(Uuid);
    std::vector<Uuid> localUuids(count);
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp props;
        checkCuda(cudaGetDeviceProperties(&props, device));
        std::memcpy(localUuids[device].data(), props.uuid.bytes, uuidBytes);
    }

    const int localBytes = count * uuidBytes;
    std::vector<int> sizes(ranks), offsets(ranks);
    MPI_Allgather(&localBytes, 1, MPI_INT, sizes.data(), 1, MPI_INT, node);
    std::exclusive_scan(sizes.begin(), sizes.end(), offsets.begin(), 0);
    std::vector<Uuid> allUuids((offsets.back() + sizes.back()) / uuidBytes);
    MPI_Allgatherv(localUuids.data(), localBytes, MPI_BYTE, allUuids.data(),
                   sizes.data(), offsets.data(), MPI_BYTE, node);

    std::map<Uuid, std::vector<int>> visibleOn;
    for (int r = 0; r < ranks; ++r) {
        for (int d = 0; d < sizes[r] / uuidBytes; ++d) {
            visibleOn[allUuids[offsets[r] / uuidBytes + d]].push_back(r);
        }
    }

    std::vector<decltype(visibleOn)::const_iterator> devices;
    for (auto device = visibleOn.cbegin(); device != visibleOn.cend(); ++device) {
        devices.push_back(device);
    }
    std::stable_sort(devices.begin(), devices.end(), [](auto a, auto b) {
        return a->second.size() < b->second.size();
    });

    std::vector<int> assigned(ranks);
    std::vector<int> result;
    for (auto device : devices) {
        const auto& candidates = device->second;
        const int owner = *std::min_element(candidates.begin(), candidates.end(), [&](int a, int b) {
            return assigned[a] < assigned[b];
        });
        ++assigned[owner];
        if (owner == rank) {
            result.push_back(std::find(localUuids.begin(), localUuids.end(), device->first)
                             - localUuids.begin());
        }
    }
    return result;
}

struct Tile {
    int firstRow, firstCol, rows, cols;
};

// Split the output into nearly square tiles, including when the GPU count is prime.
std::vector<Tile> partition(int n, int workers) {
    std::vector<Tile> tiles;
    tiles.reserve(workers);
    const int bands = static_cast<int>(std::ceil(std::sqrt(workers)));
    int firstRow = 0;
    for (int band = 0; band < bands; ++band) {
        const int cols = workers / bands + (band < workers % bands);
        const size_t completed = tiles.size() + cols;
        // Give bands with more column tiles proportionally more rows.
        const int boundary = (size_t(n) * completed + workers / 2) / workers;
        const int lastRow = std::clamp(boundary, firstRow + 1, n - (bands - band - 1));
        for (int col = 0; col < cols; ++col) {
            const int firstCol = size_t(n) * col / cols;
            const int lastCol = size_t(n) * (col + 1) / cols;
            tiles.push_back({firstRow, firstCol, lastRow - firstRow, lastCol - firstCol});
        }
        firstRow = lastRow;
    }
    return tiles;
}

__global__ void initMatrix(double* matrix, int rows, int cols, int firstRow,
                           int firstCol, size_t nSquared) {
    const size_t count = size_t(rows) * cols;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(blockDim.x) * gridDim.x) {
        matrix[i] = getPseudoRndValue(nSquared, firstRow + i / cols, firstCol + i % cols);
    }
}

template <int RowsPerThread, int ColsPerThread>
__global__ void matrixMultiply(const double* __restrict__ a, const double* __restrict__ b,
                               double* __restrict__ c, int n, int rows, int cols) {
    constexpr int threadsX = 16, threadsY = 8, blockK = 16;
    constexpr int blockRows = threadsY * RowsPerThread, blockCols = threadsX * ColsPerThread;
    // Transpose the A tile in shared memory with padding for bank conflicts.
    __shared__ double aTile[blockK][blockRows + 1];
    __shared__ double bTile[blockK][blockCols];
    const int x = threadIdx.x, y = threadIdx.y;
    const int lane = y * threadsX + x;
    const int firstRow = blockIdx.y * blockRows;
    const int firstCol = blockIdx.x * blockCols;
    double sum[RowsPerThread][ColsPerThread] = {};

    // Full tiles use a branch-free inner loop. The final tile preserves the
    // serial order, including GCC's scalar FMA for an odd-length tail.
    const int fullK = n / blockK * blockK;
    for (int base = 0; base < fullK; base += blockK) {
        #pragma unroll
        for (int i = lane; i < blockRows * blockK; i += threadsX * threadsY) {
            const int row = firstRow + i / blockK;
            const int k = i % blockK;
            aTile[k][i / blockK] = row < rows ? a[size_t(row) * n + base + k] : 0.0;
        }
        #pragma unroll
        for (int i = lane; i < blockK * blockCols; i += threadsX * threadsY) {
            const int col = firstCol + i % blockCols;
            bTile[i / blockCols][i % blockCols] =
                col < cols ? b[size_t(base + i / blockCols) * cols + col] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < blockK; ++k) {
            #pragma unroll
            for (int i = 0; i < RowsPerThread; ++i) {
                #pragma unroll
                for (int j = 0; j < ColsPerThread; ++j) {
                    sum[i][j] += aTile[k][y + i * threadsY] * bTile[k][x + j * threadsX];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < RowsPerThread; ++i) {
        const int row = firstRow + y + i * threadsY;
        #pragma unroll
        for (int j = 0; j < ColsPerThread; ++j) {
            const int col = firstCol + x + j * threadsX;
            if (row < rows && col < cols) {
                double value = sum[i][j];
                const int pairedK = n - n % 2;
                for (int k = fullK; k < pairedK; ++k) {
                    value += a[size_t(row) * n + k] * b[size_t(k) * cols + col];
                }
                if (pairedK < n) {
                    value = __fma_rn(a[size_t(row) * n + pairedK],
                                     b[size_t(pairedK) * cols + col], value);
                }
                c[size_t(row) * cols + col] = value;
            }
        }
    }
}

struct Worker {
    int device;
    Tile tile;
    const double* a = nullptr;
    double *aStorage = nullptr, *b = nullptr, *c = nullptr;
    std::vector<double> result;

    Worker(int device, Tile tile) : device(device), tile(tile) {}

    void prepare(int n) {
        const size_t aSize = size_t(tile.rows) * n;
        const size_t bSize = size_t(n) * tile.cols;
        checkCuda(cudaMalloc(&b, bSize * sizeof(double)));
        checkCuda(cudaMalloc(&c, size_t(tile.rows) * tile.cols * sizeof(double)));

        constexpr int threads = 256;
        const auto blocks = [=](size_t size) {
            return std::min<size_t>((size + threads - 1) / threads, 65535);
        };
        const size_t nSquared = size_t(n) * n;
        // A and B have identical values. A full-width B also contains this A slice.
        if (tile.cols == n) {
            a = b + size_t(tile.firstRow) * n;
        } else {
            checkCuda(cudaMalloc(&aStorage, aSize * sizeof(double)));
            a = aStorage;
            initMatrix<<<blocks(aSize), threads>>>(aStorage, tile.rows, n, tile.firstRow, 0, nSquared);
        }
        initMatrix<<<blocks(bSize), threads>>>(b, n, tile.cols, 0, tile.firstCol, nSquared);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
    }

    template <int RowWork, int ColWork>
    void launch(int n) {
        constexpr int blockRows = 8 * RowWork, blockCols = 16 * ColWork;
        const dim3 grid((tile.cols - 1) / blockCols + 1, (tile.rows - 1) / blockRows + 1);
        matrixMultiply<RowWork, ColWork><<<grid, dim3(16, 8)>>>(a, b, c, n, tile.rows, tile.cols);
    }

    void multiply(int n) {
        // Use smaller thread tiles to expose more blocks for small problems.
        if (size_t(tile.rows) * tile.cols < 256 * 512) {
            launch<2, 2>(n);
        } else {
            launch<4, 4>(n);
        }
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
    }

    void copyResult() {
        result.resize(size_t(tile.rows) * tile.cols);
        checkCuda(cudaMemcpy(result.data(), c, result.size() * sizeof(double), cudaMemcpyDeviceToHost));
    }

    bool validate(int n, bool hasResult) const {
        const int rows = std::min(tile.rows, 5 - tile.firstRow);
        const int cols = std::min(tile.cols, 5 - tile.firstCol);
        if (rows <= 0 || cols <= 0) {
            return true;
        }

        double samples[5][5];
        if (hasResult) {
            for (int i = 0; i < rows; ++i) {
                std::copy_n(result.data() + size_t(i) * tile.cols, cols, samples[i]);
            }
        } else {
            checkCuda(cudaMemcpy2D(samples, sizeof(samples[0]), c, size_t(tile.cols) * sizeof(double),
                                   cols * sizeof(double), rows, cudaMemcpyDeviceToHost));
        }

        const size_t nSquared = size_t(n) * n;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                const int row = tile.firstRow + i, col = tile.firstCol + j;
                double expected = 0.0;
                for (int k = 0; k < n; ++k) {
                    expected += getPseudoRndValue(nSquared, row, k) *
                                getPseudoRndValue(nSquared, k, col);
                }
                const double actual = samples[i][j];
                const double relError = std::abs((actual - expected) / (expected + 1e-10));
                if (!std::isfinite(actual) || relError > 1e-6) {
                    printf("Validation failed at (%d, %d): expected %.10f, got %.10f (error: %.10e)\n",
                           row, col, expected, actual, relError);
                    return false;
                }
            }
        }
        return true;
    }

    void release() {
        checkCuda(cudaFree(aStorage));
        checkCuda(cudaFree(b));
        checkCuda(cudaFree(c));
    }
};

template <typename Action>
void runOnDevices(std::vector<Worker>& workers, Action action, int rank) {
    const int count = workers.size();
    if (count == 0) {
        return;
    }

    int failed = 0;
    #pragma omp parallel for num_threads(count) schedule(static) reduction(| : failed)
    for (int i = 0; i < count; ++i) {
        try {
            checkCuda(cudaSetDevice(workers[i].device));
            action(workers[i], i);
        } catch (const std::exception& error) {
            fprintf(stderr, "Rank %d, CUDA device %d: %s\n", rank, workers[i].device, error.what());
            failed = 1;
        }
    }
    if (failed) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

void printDistributedResults(int n, int rank, const std::vector<int>& offsets,
                             const std::vector<Tile>& tiles, const std::vector<Worker>& workers) {
    std::vector<double> matrix;
    if (rank == 0) {
        matrix.resize(size_t(n) * n);
        for (const auto& worker : workers) {
            const auto& tile = worker.tile;
            for (int i = 0; i < tile.rows; ++i) {
                std::copy_n(worker.result.data() + size_t(i) * tile.cols, tile.cols,
                            matrix.data() + size_t(tile.firstRow + i) * n + tile.firstCol);
            }
        }
    }

    std::vector<MPI_Request> requests;
    const int firstRank = rank == 0 ? 1 : rank;
    const int lastRank = rank == 0 ? offsets.size() - 1 : rank + 1;
    for (int sender = firstRank; sender < lastRank; ++sender) {
        for (int id = offsets[sender]; id < offsets[sender + 1]; ++id) {
            const auto& tile = tiles[id];
            // MPI-3 messages use int counts. Transfer large tiles in row chunks.
            const int rowsPerMessage = INT_MAX / tile.cols;
            for (int row = 0; row < tile.rows;) {
                const int rows = std::min(rowsPerMessage, tile.rows - row);
                requests.push_back(MPI_REQUEST_NULL);
                if (rank == 0) {
                    MPI_Datatype destination;
                    MPI_Type_vector(rows, tile.cols, n, MPI_DOUBLE, &destination);
                    MPI_Type_commit(&destination);
                    MPI_Irecv(matrix.data() + size_t(tile.firstRow + row) * n + tile.firstCol,
                              1, destination, sender, 0, MPI_COMM_WORLD, &requests.back());
                    MPI_Type_free(&destination);
                } else {
                    const auto& result = workers[id - offsets[rank]].result;
                    MPI_Isend(result.data() + size_t(row) * tile.cols, rows * tile.cols,
                              MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requests.back());
                }
                row += rows;
            }
        }
    }
    for (auto& request : requests) {
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    if (rank == 0) {
        print_results(matrix, "MatrixC");
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

int benchmark(int argc, char** argv, int rank, int ranks) {
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end;
            errno = 0;
            const char* value = argv[++i];
            const long long parsed = std::strtoll(value, &end, 10);
            if (errno || end == value || *end || parsed < 0 || parsed > INT_MAX ||
                size_t(parsed) * size_t(parsed) > std::numeric_limits<size_t>::max() / sizeof(double)) {
                if (rank == 0) {
                    fprintf(stderr, "Invalid matrix size: %s\n", value);
                }
                return EXIT_FAILURE;
            }
            n = parsed;
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return EXIT_SUCCESS;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return EXIT_FAILURE;
        }
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    auto devices = assignDevices(node);
    MPI_Comm_free(&node);

    const int localCount = devices.size();
    std::vector<int> counts(ranks), offsets(ranks + 1);
    MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::partial_sum(counts.begin(), counts.end(), offsets.begin() + 1);
    if (offsets.back() == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices are available.\n");
        }
        return EXIT_FAILURE;
    }

    const int active = std::min<size_t>(offsets.back(), size_t(n) * n);
    for (auto& offset : offsets) {
        offset = std::min(offset, active);
    }
    const auto tiles = partition(n, active);
    std::vector<Worker> workers;
    for (int i = offsets[rank]; i < offsets[rank + 1]; ++i) {
        workers.emplace_back(devices[i - offsets[rank]], tiles[i]);
    }

    runOnDevices(workers, [n](Worker& w, int) { w.prepare(n); }, rank);
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runOnDevices(workers, [n](Worker& w, int) { w.multiply(n); }, rank);
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double gflops = elapsed > 0.0 ? (2.0 * n * n * n) / elapsed / 1e9 : 0.0;
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        runOnDevices(workers, [](Worker& w, int) { w.copyResult(); }, rank);
        printDistributedResults(n, rank, offsets, tiles, workers);
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        std::vector<int> validTiles(workers.size());
        runOnDevices(workers, [&](Worker& w, int i) { validTiles[i] = w.validate(n, printResults); }, rank);
        const int localValid = std::all_of(validTiles.begin(), validTiles.end(), [](int ok) { return ok; });
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    runOnDevices(workers, [](Worker& w, int) { w.release(); }, rank);
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return EXIT_FAILURE;
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int result = EXIT_FAILURE;
    try {
        result = benchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    MPI_Finalize();
    return result;
}
