#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr int checkWidth = 5;
using CheckValues = std::array<double, checkWidth * checkWidth>;
using DeviceId = std::array<unsigned char, 16>;

struct Tile {
    int row;
    int col;
    int rows;
    int cols;
};

struct GpuTile {
    int device;
    Tile tile;
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    double* a = nullptr;
    double* b = nullptr;
    double* c = nullptr;
};

[[noreturn]] void abortRun(int rank, const char* operation, const char* error) {
    fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, error);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status, int rank, const char* operation) {
    if (status != cudaSuccess) {
        abortRun(rank, operation, cudaGetErrorString(status));
    }
}

void checkBlas(cublasStatus_t status, int rank, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char error[48];
        snprintf(error, sizeof(error), "cuBLAS status %d", static_cast<int>(status));
        abortRun(rank, operation, error);
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

constexpr double getPseudoRndValue(size_t n, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (n * n)) / static_cast<double>(n * n);
}

size_t elements(int rows, int cols) {
    return static_cast<size_t>(rows) * cols;
}

// Assign each visible GPU to one rank on its node. UUIDs account for GPU
// visibility masks and rank-specific CUDA device numbering.
std::vector<int> assignDevices(int rank) {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank = 0;
    int localSize = 0;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_size(node, &localSize);

    int deviceCount = 0;
    const auto status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaErrorNoDevice) {
        checkCuda(status, rank, "cudaGetDeviceCount");
    }

    std::vector<DeviceId> ids(deviceCount);
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp properties{};
        checkCuda(cudaGetDeviceProperties(&properties, device), rank, "cudaGetDeviceProperties");
        static_assert(sizeof(properties.uuid.bytes) == sizeof(DeviceId));
        std::memcpy(ids[device].data(), properties.uuid.bytes, sizeof(DeviceId));
    }

    constexpr int idBytes = sizeof(DeviceId);
    const int visibleBytes = deviceCount * idBytes;
    std::vector<int> counts(localSize);
    std::vector<int> offsets(localSize);
    MPI_Allgather(&visibleBytes, 1, MPI_INT, counts.data(), 1, MPI_INT, node);
    std::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), 0);
    std::vector<DeviceId> allIds((offsets.back() + counts.back()) / idBytes);
    MPI_Allgatherv(ids.data(), visibleBytes, MPI_BYTE, allIds.data(), counts.data(),
                   offsets.data(), MPI_BYTE, node);
    MPI_Comm_free(&node);

    std::map<DeviceId, std::vector<int>> visibleTo;
    for (int r = 0; r < localSize; ++r) {
        for (int i = offsets[r]; i < offsets[r] + counts[r]; i += idBytes) {
            visibleTo[allIds[i / idBytes]].push_back(r);
        }
    }

    std::vector<int> load(localSize);
    std::vector<int> devices;
    for (const auto& [id, candidates] : visibleTo) {
        const int owner = *std::min_element(candidates.begin(), candidates.end(),
            [&](int a, int b) {
                return std::tie(load[a], counts[a], a) < std::tie(load[b], counts[b], b);
            });
        ++load[owner];
        if (owner == localRank) {
            devices.push_back(static_cast<int>(std::find(ids.begin(), ids.end(), id) - ids.begin()));
        }
    }
    return devices;
}

// Each device computes a rectangle of C with the full inner dimension.
Tile getTile(int index, int count, int n) {
    const int rowGroups = std::min(n, static_cast<int>(std::ceil(std::sqrt(count))));
    const int colsPerGroup = count / rowGroups;
    const int widerGroups = count % rowGroups;
    const int widerSlots = widerGroups * (colsPerGroup + 1);
    const bool wider = index < widerSlots;
    const int colGroups = colsPerGroup + wider;
    const int slot = wider ? index : index - widerSlots;
    const int rowGroup = slot / colGroups + (wider ? 0 : widerGroups);
    const int colGroup = slot % colGroups;

    const int firstSlot = index - colGroup;
    const bool weightedRows = elements(n, colsPerGroup) >= static_cast<size_t>(count);
    const int row = weightedRows ? elements(n, firstSlot) / count
                                 : elements(n, rowGroup) / rowGroups;
    const int rowEnd = weightedRows ? elements(n, firstSlot + colGroups) / count
                                    : elements(n, rowGroup + 1) / rowGroups;
    const int col = elements(n, colGroup) / colGroups;
    const int colEnd = elements(n, colGroup + 1) / colGroups;
    return {row, col, rowEnd - row, colEnd - col};
}

void launchMultiply(const GpuTile& gpu, int n, int limit, int rank) {
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;
    // cuBLAS sees the row-major buffers as transposed column-major matrices.
    checkBlas(cublasDgemm(gpu.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         std::min(gpu.tile.cols, limit), std::min(gpu.tile.rows, limit),
                         std::min(n, limit), &alpha, gpu.b, gpu.tile.cols, gpu.a, n,
                         &beta, gpu.c, gpu.tile.cols), rank, "cublasDgemm");
}

void prepareTile(GpuTile& gpu, int n, int rank) {
    const Tile tile = gpu.tile;
    const size_t aCount = elements(tile.rows, n);
    const size_t bCount = elements(n, tile.cols);
    const size_t cCount = elements(tile.rows, tile.cols);
    checkCuda(cudaSetDevice(gpu.device), rank, "cudaSetDevice");
    checkCuda(cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking), rank, "cudaStreamCreate");
    checkBlas(cublasCreate(&gpu.handle), rank, "cublasCreate");
    checkBlas(cublasSetStream(gpu.handle, gpu.stream), rank, "cublasSetStream");
    checkCuda(cudaMalloc(&gpu.a, aCount * sizeof(double)), rank, "allocate device matrix A");
    checkCuda(cudaMalloc(&gpu.b, bCount * sizeof(double)), rank, "allocate device matrix B");
    checkCuda(cudaMalloc(&gpu.c, cCount * sizeof(double)), rank, "allocate device matrix C");

    const std::unique_ptr<double[]> a(new double[aCount]);
    const std::unique_ptr<double[]> b(new double[bCount]);
    const int threads = std::min<size_t>(omp_get_max_threads(), 1 + (aCount + bCount) / 16384);
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < tile.rows; ++i) {
            for (int j = 0; j < n; ++j) {
                a[elements(i, n) + j] = getPseudoRndValue(n, tile.row + i, j);
            }
        }

        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < tile.cols; ++j) {
                b[elements(i, tile.cols) + j] = getPseudoRndValue(n, i, tile.col + j);
            }
        }
    }

    checkCuda(cudaMemcpyAsync(gpu.a, a.get(), aCount * sizeof(double), cudaMemcpyHostToDevice,
                              gpu.stream), rank, "upload matrix A");
    checkCuda(cudaMemcpyAsync(gpu.b, b.get(), bCount * sizeof(double), cudaMemcpyHostToDevice,
                              gpu.stream), rank, "upload matrix B");
    // Initialize cuBLAS's GPU kernels before timing, with bounded work at large N.
    launchMultiply(gpu, n, 1024, rank);
    checkCuda(cudaStreamSynchronize(gpu.stream), rank, "complete setup and warm-up");
}

void multiply(const std::vector<GpuTile>& tiles, int n, int rank) {
    for (const auto& gpu : tiles) {
        checkCuda(cudaSetDevice(gpu.device), rank, "cudaSetDevice");
        launchMultiply(gpu, n, n, rank);
    }

    for (const auto& gpu : tiles) {
        checkCuda(cudaSetDevice(gpu.device), rank, "cudaSetDevice");
        checkCuda(cudaStreamSynchronize(gpu.stream), rank, "complete matrix multiplication");
    }
}

void releaseTile(const GpuTile& gpu, int rank) {
    checkCuda(cudaSetDevice(gpu.device), rank, "cudaSetDevice");
    checkBlas(cublasDestroy(gpu.handle), rank, "cublasDestroy");
    checkCuda(cudaFree(gpu.a), rank, "free device matrix A");
    checkCuda(cudaFree(gpu.b), rank, "free device matrix B");
    checkCuda(cudaFree(gpu.c), rank, "free device matrix C");
    checkCuda(cudaStreamDestroy(gpu.stream), rank, "cudaStreamDestroy");
}

int outputBatchRows(int cols) {
    constexpr int stagingElements = 1 << 20;
    return std::max(1, stagingElements / cols);
}

std::vector<double> gatherResult(const std::vector<GpuTile>& tiles, int n, int rank,
                                 int rankCount, int tileCount) {
    const int localCount = static_cast<int>(tiles.size());
    std::vector<int> counts(rank == 0 ? rankCount : 0);
    MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> result;
    std::vector<MPI_Request> requests;
    if (rank == 0) {
        result.resize(elements(n, n));
        int index = localCount;
        for (int source = 1; source < rankCount; ++source) {
            for (int i = 0; i < counts[source]; ++i, ++index) {
                const Tile tile = getTile(index, tileCount, n);
                const int rowsPerMessage = outputBatchRows(tile.cols);
                for (int row = 0; row < tile.rows;) {
                    const int rows = std::min(tile.rows - row, rowsPerMessage);
                    MPI_Datatype stripe;
                    MPI_Type_create_hvector(rows, tile.cols, static_cast<MPI_Aint>(n) * sizeof(double),
                                            MPI_DOUBLE, &stripe);
                    MPI_Type_commit(&stripe);
                    requests.push_back(MPI_REQUEST_NULL);
                    MPI_Irecv(result.data() + elements(tile.row + row, n) + tile.col, 1,
                              stripe, source, 0, MPI_COMM_WORLD, &requests.back());
                    MPI_Type_free(&stripe);
                    row += rows;
                }
            }
        }
    }

    for (const auto& gpu : tiles) {
        const Tile tile = gpu.tile;
        checkCuda(cudaSetDevice(gpu.device), rank, "cudaSetDevice");
        if (rank == 0) {
            checkCuda(cudaMemcpy2D(result.data() + elements(tile.row, n) + tile.col,
                                   n * sizeof(double), gpu.c, tile.cols * sizeof(double),
                                   tile.cols * sizeof(double), tile.rows, cudaMemcpyDeviceToHost),
                      rank, "download local output tile");
            continue;
        }

        const int rowsPerMessage = outputBatchRows(tile.cols);
        const std::unique_ptr<double[]> buffer(new double[elements(std::min(tile.rows, rowsPerMessage),
                                                                   tile.cols)]);
        for (int row = 0; row < tile.rows;) {
            const int rows = std::min(tile.rows - row, rowsPerMessage);
            checkCuda(cudaMemcpy(buffer.get(), gpu.c + elements(row, tile.cols),
                                 elements(rows, tile.cols) * sizeof(double), cudaMemcpyDeviceToHost),
                      rank, "download output stripe");
            MPI_Send(buffer.get(), rows * tile.cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            row += rows;
        }
    }

    if (rank == 0) {
        for (auto& request : requests) {
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
    }
    return result;
}

CheckValues gatherChecks(const std::vector<GpuTile>& tiles, int n, int rank) {
    const int width = std::min(n, checkWidth);
    CheckValues local{};
    for (const auto& gpu : tiles) {
        const Tile tile = gpu.tile;
        const int rows = std::min(tile.rows, width - tile.row);
        const int cols = std::min(tile.cols, width - tile.col);
        if (rows <= 0 || cols <= 0) {
            continue;
        }
        checkCuda(cudaSetDevice(gpu.device), rank, "cudaSetDevice");
        checkCuda(cudaMemcpy2D(local.data() + tile.row * checkWidth + tile.col,
                               checkWidth * sizeof(double), gpu.c, tile.cols * sizeof(double),
                               cols * sizeof(double), rows, cudaMemcpyDeviceToHost),
                  rank, "download validation region");
    }
    CheckValues values{};
    MPI_Reduce(local.data(), values.data(), values.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return values;
}

bool validateResult(const CheckValues& values, int n) {
    const int width = std::min(n, checkWidth);
    CheckValues expected{};
    const int threads = std::min(omp_get_max_threads(), width * width);
    #pragma omp parallel for collapse(2) schedule(static) if (n >= 256) num_threads(std::max(1, threads))
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
            }
            expected[i * checkWidth + j] = sum;
        }
    }

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            const int index = i * checkWidth + j;
            const double relError = std::abs((values[index] - expected[index]) /
                                             (expected[index] + 1e-10));
            if (!(relError <= 1e-6)) {
                printf("Validation failed at (%d, %d): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected[index], values[index], relError);
                return false;
            }
        }
    }
    return true;
}

int run(int argc, char** argv, int rank, int rankCount) {
    std::array<int, 4> arguments{512, 0, 0, 0};
    auto& [n, validate, printResults, parseStatus] = arguments;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const char* value = argv[++i];
                char* end = nullptr;
                errno = 0;
                const long size = std::strtol(value, &end, 10);
                if (errno || end == value || *end || size < 0 || size > INT_MAX) {
                    fprintf(stderr, "Invalid matrix size: %s\n", value);
                    parseStatus = 1;
                    break;
                }
                n = static_cast<int>(size);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parseStatus = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseStatus = 1;
                break;
            }
        }
    }
    MPI_Bcast(arguments.data(), arguments.size(), MPI_INT, 0, MPI_COMM_WORLD);
    if (parseStatus != 0) {
        return parseStatus == 1 ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    const auto devices = assignDevices(rank);
    const int localCount = static_cast<int>(devices.size());
    int totalDevices = 0;
    int firstDevice = 0;
    MPI_Allreduce(&localCount, &totalDevices, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Exscan(&localCount, &firstDevice, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        firstDevice = 0;
    }
    if (totalDevices == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices are visible to the MPI job.\n");
        }
        return EXIT_FAILURE;
    }

    const int tileCount = std::min<size_t>(totalDevices, elements(n, n));
    std::vector<GpuTile> tiles;
    for (int i = 0; i < localCount && firstDevice + i < tileCount; ++i) {
        GpuTile gpu{};
        gpu.device = devices[i];
        gpu.tile = getTile(firstDevice + i, tileCount, n);
        prepareTile(gpu, n, rank);
        tiles.push_back(std::move(gpu));
    }

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    multiply(tiles, n, rank);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double gflops = n == 0 ? 0.0 : (2.0 * n * n * n) / duration / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    CheckValues values{};
    if (printResults) {
        const auto result = gatherResult(tiles, n, rank, rankCount, tileCount);
        if (rank == 0) {
            print_results(result, "MatrixC");
            const int width = validate ? std::min(n, checkWidth) : 0;
            for (int i = 0; i < width; ++i) {
                for (int j = 0; j < width; ++j) {
                    values[i * checkWidth + j] = result[elements(i, n) + j];
                }
            }
        }
    } else if (validate) {
        values = gatherChecks(tiles, n, rank);
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(values, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    for (const auto& gpu : tiles) {
        releaseTile(gpu, rank);
    }
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        fprintf(stderr, "Failed to initialize MPI.\n");
        return EXIT_FAILURE;
    }
    int rank = 0;
    int rankCount = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rankCount);
    if (provided < MPI_THREAD_FUNNELED) {
        abortRun(rank, "MPI_Init_thread", "MPI_THREAD_FUNNELED is required");
    }

    int result = EXIT_FAILURE;
    try {
        result = run(argc, argv, rank, rankCount);
    } catch (const std::exception& error) {
        abortRun(rank, "matmul", error.what());
    }
    MPI_Finalize();
    return result;
}
