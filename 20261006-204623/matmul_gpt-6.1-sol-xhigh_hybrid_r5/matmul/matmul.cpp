#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// Keep the original unsigned arithmetic, including its wraparound behavior.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void checkMPI(int status) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        throw std::runtime_error(std::string(message, length));
    }
}

void checkCUDA(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
    }
}

void checkBLAS(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS error " + std::to_string(static_cast<int>(status)));
    }
}

// Balanced, contiguous output blocks; input panels are owned cyclically along K.
int blockSize(int N, int coordinate, int parts) {
    return N / parts + (coordinate < N % parts ? 1 : 0);
}

int blockStart(int N, int coordinate, int parts) {
    return coordinate * (N / parts) + std::min(coordinate, N % parts);
}

struct ProcessGrid {
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Comm row = MPI_COMM_NULL;
    MPI_Comm column = MPI_COMM_NULL;
    int dimensions[2] = {0, 0};
    int coordinates[2] = {0, 0};
    int rank = 0;
    int size = 0;
    int rows = 0;
    int columns = 0;
    int rowStart = 0;
    int columnStart = 0;

    explicit ProcessGrid(int N) {
        checkMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
        checkMPI(MPI_Dims_create(size, 2, dimensions));
        const int periodic[2] = {0, 0};
        // No reordering: world rank zero remains the output rank.
        checkMPI(MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periodic, 0, &cart));
        checkMPI(MPI_Comm_rank(cart, &rank));
        checkMPI(MPI_Cart_coords(cart, rank, 2, coordinates));
        checkMPI(MPI_Comm_split(cart, coordinates[0], coordinates[1], &row));
        checkMPI(MPI_Comm_split(cart, coordinates[1], coordinates[0], &column));
        rows = blockSize(N, coordinates[0], dimensions[0]);
        columns = blockSize(N, coordinates[1], dimensions[1]);
        rowStart = blockStart(N, coordinates[0], dimensions[0]);
        columnStart = blockStart(N, coordinates[1], dimensions[1]);
    }

    ~ProcessGrid() {
        // The top-level error handler aborts all ranks; avoid collective cleanup
        // while peers may still be waiting for an operation that has failed.
        if (std::uncaught_exceptions() != 0) return;
        if (column != MPI_COMM_NULL) MPI_Comm_free(&column);
        if (row != MPI_COMM_NULL) MPI_Comm_free(&row);
        if (cart != MPI_COMM_NULL) MPI_Comm_free(&cart);
    }

    ProcessGrid(const ProcessGrid&) = delete;
    ProcessGrid& operator=(const ProcessGrid&) = delete;
};

void selectDevice() {
    MPI_Comm local = MPI_COMM_NULL;
    checkMPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                MPI_INFO_NULL, &local));
    int localRank = 0;
    int localSize = 0;
    checkMPI(MPI_Comm_rank(local, &localRank));
    checkMPI(MPI_Comm_size(local, &localSize));
    checkMPI(MPI_Comm_free(&local));

    int deviceCount = 0;
    checkCUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        throw std::runtime_error("a CUDA device is required on every MPI rank");
    }
    // Also handles launchers that expose just one GPU to each rank.
    checkCUDA(cudaSetDevice(localRank % deviceCount));
    checkCUDA(cudaFree(nullptr));

    omp_set_dynamic(0);
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        // Avoid large thread teams and CPU oversubscription with unbound ranks.
        const int threads = std::max(1, std::min(16, omp_get_num_procs() / localSize));
        omp_set_num_threads(threads);
    }
}

struct GPU {
    cudaStream_t copyStream = nullptr;
    cudaStream_t computeStream = nullptr;
    cublasHandle_t handle = nullptr;
    double* C = nullptr;

    explicit GPU(size_t elements) {
        checkCUDA(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
        checkCUDA(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
        checkBLAS(cublasCreate(&handle));
        checkBLAS(cublasSetStream(handle, computeStream));
        // DGEMM uses full FP64, including on GPUs with FP64 tensor cores.
        checkBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        checkCUDA(cudaMalloc(reinterpret_cast<void**>(&C),
                             std::max(size_t{1}, elements) * sizeof(double)));
    }

    ~GPU() {
        if (std::uncaught_exceptions() != 0) return;
        if (C != nullptr) cudaFree(C);
        if (handle != nullptr) cublasDestroy(handle);
        if (computeStream != nullptr) cudaStreamDestroy(computeStream);
        if (copyStream != nullptr) cudaStreamDestroy(copyStream);
    }

    GPU(const GPU&) = delete;
    GPU& operator=(const GPU&) = delete;
};

struct Panel {
    double* hostA = nullptr;
    double* hostB = nullptr;
    double* deviceA = nullptr;
    double* deviceB = nullptr;
    cudaEvent_t copied = nullptr;
    cudaEvent_t consumed = nullptr;
    MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    int width = 0;
    bool used = false;

    Panel(int rows, int columns, int capacity) {
        const size_t bytesA = std::max(size_t{1}, size_t(rows) * capacity) * sizeof(double);
        const size_t bytesB = std::max(size_t{1}, size_t(columns) * capacity) * sizeof(double);
        // Pinned staging works with ordinary MPI; CUDA-aware MPI is not required.
        checkCUDA(cudaMallocHost(reinterpret_cast<void**>(&hostA), bytesA));
        checkCUDA(cudaMallocHost(reinterpret_cast<void**>(&hostB), bytesB));
        checkCUDA(cudaMalloc(reinterpret_cast<void**>(&deviceA), bytesA));
        checkCUDA(cudaMalloc(reinterpret_cast<void**>(&deviceB), bytesB));
        checkCUDA(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
        checkCUDA(cudaEventCreateWithFlags(&consumed, cudaEventDisableTiming));
    }

    ~Panel() {
        // Pending MPI requests may still reference pinned buffers on failure.
        if (std::uncaught_exceptions() != 0) return;
        if (consumed != nullptr) cudaEventDestroy(consumed);
        if (copied != nullptr) cudaEventDestroy(copied);
        if (deviceB != nullptr) cudaFree(deviceB);
        if (deviceA != nullptr) cudaFree(deviceA);
        if (hostB != nullptr) cudaFreeHost(hostB);
        if (hostA != nullptr) cudaFreeHost(hostA);
    }

    Panel(const Panel&) = delete;
    Panel& operator=(const Panel&) = delete;
};

void preparePanel(Panel& panel, const ProcessGrid& grid, int N,
                  int panelIndex, int panelWidth) {
    if (panel.used) {
        // Host storage may be reused once DMA finishes, even while GEMM runs.
        checkCUDA(cudaEventSynchronize(panel.copied));
    }
    const int firstK = panelIndex * panelWidth;
    panel.width = std::min(panelWidth, N - firstK);
    const int ownerA = panelIndex % grid.dimensions[1];
    const int ownerB = panelIndex % grid.dimensions[0];
    const bool makeA = grid.coordinates[1] == ownerA;
    const bool makeB = grid.coordinates[0] == ownerB;
    const size_t entries = (makeA ? size_t(grid.rows) * panel.width : 0)
                        + (makeB ? size_t(grid.columns) * panel.width : 0);

    // Each input entry is generated once globally. Both panels are column-major
    // for cuBLAS, eliminating transposes and a root-owned copy of the matrices.
    #pragma omp parallel if(entries >= 65536)
    {
        if (makeA) {
            #pragma omp for schedule(static) nowait
            for (int k = 0; k < panel.width; ++k) {
                for (int i = 0; i < grid.rows; ++i) {
                    panel.hostA[size_t(k) * grid.rows + i] =
                        getPseudoRndValue(N, size_t(grid.rowStart) + i, size_t(firstK) + k);
                }
            }
        }
        if (makeB) {
            #pragma omp for schedule(static) nowait
            for (int j = 0; j < grid.columns; ++j) {
                for (int k = 0; k < panel.width; ++k) {
                    panel.hostB[size_t(j) * panel.width + k] =
                        getPseudoRndValue(N, size_t(firstK) + k, size_t(grid.columnStart) + j);
                }
            }
        }
    }

    checkMPI(MPI_Ibcast(panel.hostA, grid.rows * panel.width, MPI_DOUBLE,
                        ownerA, grid.row, &panel.requests[0]));
    checkMPI(MPI_Ibcast(panel.hostB, grid.columns * panel.width, MPI_DOUBLE,
                        ownerB, grid.column, &panel.requests[1]));
}

void matrixMultiply(const ProcessGrid& grid, GPU& gpu, Panel (&panels)[2],
                    int N, int panelWidth) {
    const int panelCount = 1 + (N - 1) / panelWidth;
    preparePanel(panels[0], grid, N, 0, panelWidth);
    for (int p = 0; p < panelCount; ++p) {
        Panel& panel = panels[p % 2];
        checkMPI(MPI_Waitall(2, panel.requests, MPI_STATUSES_IGNORE));
        if (grid.rows > 0 && grid.columns > 0) {
            if (panel.used) {
                // Device storage cannot be overwritten until its previous GEMM.
                checkCUDA(cudaStreamWaitEvent(gpu.copyStream, panel.consumed, 0));
            }
            checkCUDA(cudaMemcpyAsync(panel.deviceA, panel.hostA,
                                      size_t(grid.rows) * panel.width * sizeof(double),
                                      cudaMemcpyHostToDevice, gpu.copyStream));
            checkCUDA(cudaMemcpyAsync(panel.deviceB, panel.hostB,
                                      size_t(grid.columns) * panel.width * sizeof(double),
                                      cudaMemcpyHostToDevice, gpu.copyStream));
            checkCUDA(cudaEventRecord(panel.copied, gpu.copyStream));
            checkCUDA(cudaStreamWaitEvent(gpu.computeStream, panel.copied, 0));
            const double alpha = 1.0;
            const double beta = p == 0 ? 0.0 : 1.0;
            checkBLAS(cublasDgemm(gpu.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  grid.rows, grid.columns, panel.width,
                                  &alpha, panel.deviceA, grid.rows,
                                  panel.deviceB, panel.width, &beta, gpu.C, grid.rows));
            checkCUDA(cudaEventRecord(panel.consumed, gpu.computeStream));
            panel.used = true;
        }
        // MPI progress and OpenMP initialization overlap the queued GPU work.
        if (p + 1 < panelCount) {
            preparePanel(panels[(p + 1) % 2], grid, N, p + 1, panelWidth);
        }
    }
    checkCUDA(cudaStreamSynchronize(gpu.computeStream));
    checkCUDA(cudaStreamSynchronize(gpu.copyStream));
}

std::vector<double> gatherResult(const ProcessGrid& grid, const std::vector<double>& localC,
                                 int N) {
    std::vector<double> C;
    if (grid.rank == 0) {
        C.resize(size_t(N) * N);
        // Cache-blocked conversion of rank zero's column-major tile.
        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 0; j < grid.columns; j += 32) {
            for (int i = 0; i < grid.rows; i += 32) {
                for (int jj = j; jj < std::min(j + 32, grid.columns); ++jj) {
                    for (int ii = i; ii < std::min(i + 32, grid.rows); ++ii) {
                        C[(size_t(grid.rowStart) + ii) * N + grid.columnStart + jj] =
                            localC[size_t(jj) * grid.rows + ii];
                    }
                }
            }
        }
        // Receive directly into the original row-major ordering. No second
        // full-sized packed matrix, int-sized global displacement, or root input.
        for (int rank = 1; rank < grid.size; ++rank) {
            int coords[2];
            checkMPI(MPI_Cart_coords(grid.cart, rank, 2, coords));
            const int rows = blockSize(N, coords[0], grid.dimensions[0]);
            const int columns = blockSize(N, coords[1], grid.dimensions[1]);
            if (rows == 0 || columns == 0) continue;
            const int rowStart = blockStart(N, coords[0], grid.dimensions[0]);
            const int columnStart = blockStart(N, coords[1], grid.dimensions[1]);
            MPI_Datatype column = MPI_DATATYPE_NULL;
            MPI_Datatype tile = MPI_DATATYPE_NULL;
            checkMPI(MPI_Type_vector(rows, 1, N, MPI_DOUBLE, &column));
            checkMPI(MPI_Type_create_resized(column, 0, sizeof(double), &tile));
            checkMPI(MPI_Type_commit(&tile));
            checkMPI(MPI_Type_free(&column));
            const int chunkColumns = INT_MAX / rows;
            for (int first = 0; first < columns;) {
                const int count = std::min(chunkColumns, columns - first);
                checkMPI(MPI_Recv(C.data() + size_t(rowStart) * N + columnStart + first,
                                  count, tile, rank, 0, grid.cart, MPI_STATUS_IGNORE));
                first += count;
            }
            checkMPI(MPI_Type_free(&tile));
        }
    } else if (grid.rows > 0 && grid.columns > 0) {
        const int chunkColumns = INT_MAX / grid.rows;
        for (int first = 0; first < grid.columns;) {
            const int count = std::min(chunkColumns, grid.columns - first);
            checkMPI(MPI_Send(localC.data() + size_t(first) * grid.rows,
                              grid.rows * count, MPI_DOUBLE, 0, 0, grid.cart));
            first += count;
        }
    }
    return C;
}

bool validateResult(const ProcessGrid& grid, const GPU& gpu,
                    const std::vector<double>& localC, int N) {
    struct CheckPoint { int i; int j; double actual; };
    std::vector<CheckPoint> points;
    auto addPoint = [&](int i, int j) {
        if (i < grid.rowStart || i >= grid.rowStart + grid.rows ||
            j < grid.columnStart || j >= grid.columnStart + grid.columns) return;
        const size_t index = size_t(j - grid.columnStart) * grid.rows + i - grid.rowStart;
        double actual = 0.0;
        if (localC.empty()) {
            checkCUDA(cudaMemcpy(&actual, gpu.C + index, sizeof(double), cudaMemcpyDeviceToHost));
        } else {
            actual = localC[index];
        }
        points.push_back({i, j, actual});
    };
    // Retain the original 25 checks and cover every output block as well.
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) addPoint(i % N, j % N);
    }
    if (grid.rows > 0 && grid.columns > 0) {
        addPoint(grid.rowStart, grid.columnStart);
        addPoint(grid.rowStart, grid.columnStart + grid.columns - 1);
        addPoint(grid.rowStart + grid.rows - 1, grid.columnStart);
        addPoint(grid.rowStart + grid.rows - 1, grid.columnStart + grid.columns - 1);
        addPoint(grid.rowStart + grid.rows / 2, grid.columnStart + grid.columns / 2);
    }
    int failed = 0;
    #pragma omp parallel for reduction(|:failed) schedule(static)
    for (size_t p = 0; p < points.size(); ++p) {
        const CheckPoint& point = points[p];
        double expected = 0.0;
        for (int k = 0; k < N; ++k) {
            expected += getPseudoRndValue(N, point.i, k) * getPseudoRndValue(N, k, point.j);
        }
        const double relError = std::abs((point.actual - expected) / (expected + 1e-10));
        if (!std::isfinite(point.actual) || relError > 1e-6) {
            failed = 1;
            #pragma omp critical
            std::printf("Validation failed at (%d, %d): expected %.10f, got %.10f (error: %.10e)\n",
                        point.i, point.j, expected, point.actual, relError);
        }
    }
    int anyFailed = 0;
    checkMPI(MPI_Allreduce(&failed, &anyFailed, 1, MPI_INT, MPI_MAX, grid.cart));
    return anyFailed == 0;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank) {
    // Parse on rank zero and broadcast the configuration, including early exits.
    // state: success=0, help=1, invalid arguments=2.
    int options[4] = {512, 0, 0, 0};
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const char* value = argv[++i];
                char* end = nullptr;
                errno = 0;
                const unsigned long long parsed = std::strtoull(value, &end, 10);
                if (errno != 0 || *value == '-' || end == value || *end != '\0' ||
                    parsed == 0 || parsed > INT_MAX ||
                    parsed > std::numeric_limits<size_t>::max() / sizeof(double) / parsed) {
                    std::fprintf(stderr, "Invalid matrix size: %s\n", value);
                    options[3] = 2;
                    break;
                }
                options[0] = static_cast<int>(parsed);
            } else if (std::strcmp(argv[i], "-v") == 0) {
                options[1] = 1;
            } else if (std::strcmp(argv[i], "-r") == 0) {
                options[2] = 1;
            } else if (std::strcmp(argv[i], "-h") == 0) {
                options[3] = 1;
                break;
            } else {
                std::printf("Unknown option: %s\n", argv[i]);
                options[3] = 2;
                break;
            }
        }
        if (options[3] != 0) printUsage(argv[0]);
    }
    checkMPI(MPI_Bcast(options, 4, MPI_INT, 0, MPI_COMM_WORLD));
    if (options[3] != 0) return options[3] == 1 ? 0 : 1;

    const int N = options[0];
    const bool validate = options[1] != 0;
    const bool printResults = options[2] != 0;
    selectDevice();
    ProcessGrid grid(N);
    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %d x %d\n", N, N);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallelism: %d MPI ranks (%d x %d), %d OpenMP threads on rank 0, CUDA/cuBLAS\n",
                    grid.size, grid.dimensions[0], grid.dimensions[1], omp_get_max_threads());
        std::printf("Initializing matrices...\n");
    }
    // Bound panel messages to MPI's int counts on every row/column communicator.
    const int maxDimension = std::max(blockSize(N, 0, grid.dimensions[0]),
                                      blockSize(N, 0, grid.dimensions[1]));
    const int panelWidth = std::min({N, 1024, INT_MAX / maxDimension});
    GPU gpu(size_t(grid.rows) * grid.columns);
    Panel panels[2] = {{grid.rows, grid.columns, panelWidth},
                       {grid.rows, grid.columns, panelWidth}};
    if (rank == 0) std::printf("Computing matrix multiplication...\n");
    checkMPI(MPI_Barrier(grid.cart));
    const double start = MPI_Wtime();
    // Include streaming initialization, MPI communication, H2D copies and GEMM.
    // Memory stays O(N^2/P + N*panelWidth/sqrt(P)), without replicated inputs.
    matrixMultiply(grid, gpu, panels, N, panelWidth);
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    checkMPI(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, grid.cart));
    if (rank == 0) {
        const double n = N;
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        std::printf("Performance: %.3f GFLOPS\n", (2.0 * n * n * n) / seconds / 1e9);
    }

    std::vector<double> localC;
    if (printResults) {
        localC.resize(size_t(grid.rows) * grid.columns);
        if (!localC.empty()) {
            checkCUDA(cudaMemcpy(localC.data(), gpu.C, localC.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
        const std::vector<double> C = gatherResult(grid, localC, N);
        if (rank == 0) print_results(C, "MatrixC");
    }
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        const bool valid = validateResult(grid, gpu, localC, N);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI initialization failed\n");
        return 1;
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int result = 1;
    try {
        if (provided < MPI_THREAD_FUNNELED) {
            throw std::runtime_error("MPI_THREAD_FUNNELED support is required");
        }
        result = run(argc, argv, rank);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        // A failure on one rank must not leave its peers blocked in collectives.
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return result;
}
