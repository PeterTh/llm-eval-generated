#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

int g_rank = 0;

[[noreturn]] void fatal(const char* operation, const char* detail, int code) {
    std::fprintf(stderr, "Rank %d: %s failed (%s, status %d)\n", g_rank, operation,
                 detail, code);
    MPI_Abort(MPI_COMM_WORLD, code == 0 ? 1 : code);
    std::abort();
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fatal(operation, cudaGetErrorString(status), static_cast<int>(status));
    }
}

void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fatal(operation, "cuBLAS error", static_cast<int>(status));
    }
}

void checkCusolver(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        fatal(operation, "cuSOLVER error", static_cast<int>(status));
    }
}

void checkMpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING] = {};
        int length = 0;
        MPI_Error_string(status, message, &length);
        fatal(operation, message, status);
    }
}

size_t checkedProduct(size_t a, size_t b, const char* description) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        fatal(description, "size overflow", 1);
    }
    return a * b;
}

// glibc rand_r advances this 32-bit LCG three times for each returned value.
// Jumping to each OpenMP thread's first value preserves the baseline's exact
// seed-42 sequence while allowing the O(n^2) random fill to run in parallel.
uint32_t advanceLcg(uint32_t state, uint64_t steps) {
    uint32_t accumulatedMultiplier = 1U;
    uint32_t accumulatedIncrement = 0U;
    uint32_t multiplier = 1103515245U;
    uint32_t increment = 12345U;

    while (steps != 0) {
        if ((steps & 1U) != 0U) {
            accumulatedIncrement =
                multiplier * accumulatedIncrement + increment;
            accumulatedMultiplier = multiplier * accumulatedMultiplier;
        }
        increment = multiplier * increment + increment;
        multiplier = multiplier * multiplier;
        steps >>= 1U;
    }
    return accumulatedMultiplier * state + accumulatedIncrement;
}

void generateRandomMatrix(std::vector<double>& matrix) {
    const size_t count = matrix.size();
#pragma omp parallel
    {
        const size_t thread = static_cast<size_t>(omp_get_thread_num());
        const size_t threads = static_cast<size_t>(omp_get_num_threads());
        const size_t base = count / threads;
        const size_t remainder = count % threads;
        const size_t begin = thread * base + std::min(thread, remainder);
        const size_t end = begin + base + (thread < remainder ? 1U : 0U);
        unsigned int seed = advanceLcg(42U, 3ULL * begin);
        for (size_t i = begin; i < end; ++i) {
            matrix[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
        }
    }
}

int chooseBlockSize(size_t n, int ranks) {
    if (ranks == 1) {
        // One column-major tile lets cuSOLVER factor the complete matrix in a
        // single highly tuned POTRF call when no inter-rank distribution is
        // necessary.
        return static_cast<int>(n);
    }
    if (n <= 128) {
        return static_cast<int>(n);
    }

    // Two or more tile rows per rank expose distributed concurrency while
    // 256-512 sized BLAS-3 calls retain high GPU efficiency.
    const size_t targetTiles = static_cast<size_t>(ranks) * 2U;
    size_t block = (n + targetTiles - 1U) / targetTiles;
    block = ((block + 63U) / 64U) * 64U;
    block = std::max<size_t>(128U, std::min<size_t>(512U, block));
    return static_cast<int>(std::min(n, block));
}

class DistributedTiles {
public:
    DistributedTiles(int n, int blockSize, int rank, int ranks)
        : n_(n), blockSize_(blockSize),
          tileCount_((n + blockSize - 1) / blockSize), rank_(rank), ranks_(ranks),
          tileElements_(checkedProduct(static_cast<size_t>(blockSize),
                                       static_cast<size_t>(blockSize), "tile size")) {
        const size_t tileRows = static_cast<size_t>(tileCount_);
        const size_t triangularTiles =
            checkedProduct(tileRows, tileRows + 1U, "triangular tile count") / 2U;
        const size_t localTiles =
            triangularTiles > static_cast<size_t>(rank_)
                ? (triangularTiles - 1U - static_cast<size_t>(rank_)) /
                          static_cast<size_t>(ranks_) +
                      1U
                : 0U;
        elements_ = checkedProduct(localTiles, tileElements_, "tile storage");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                             checkedProduct(std::max<size_t>(elements_, 1U),
                                            sizeof(double), "tile bytes")),
                  "cudaMalloc(distributed matrix)");
    }

    DistributedTiles(const DistributedTiles&) = delete;
    DistributedTiles& operator=(const DistributedTiles&) = delete;

    ~DistributedTiles() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    bool ownsTile(int row, int column) const {
        return owner(row, column) == rank_;
    }
    int owner(int row, int column) const {
        return static_cast<int>(ordinal(row, column) % static_cast<size_t>(ranks_));
    }
    size_t ordinal(int row, int column) const {
        return static_cast<size_t>(row) * static_cast<size_t>(row + 1) / 2U +
               static_cast<size_t>(column);
    }
    int tileCount() const { return tileCount_; }
    int blockSize() const { return blockSize_; }
    size_t tileElements() const { return tileElements_; }
    size_t elements() const { return elements_; }
    double* data() { return data_; }

    int extent(int tile) const {
        return std::min(blockSize_, n_ - tile * blockSize_);
    }

    double* tile(int row, int column) {
        if (column < 0 || column > row || !ownsTile(row, column)) {
            fatal("distributed tile access", "tile is not local", 1);
        }
        const size_t localTile = ordinal(row, column) / static_cast<size_t>(ranks_);
        return data_ + localTile * tileElements_;
    }

private:
    int n_;
    int blockSize_;
    int tileCount_;
    int rank_;
    int ranks_;
    size_t tileElements_;
    size_t elements_ = 0;
    double* data_ = nullptr;
};

struct GpuContext {
    static constexpr int updateStreamCount = 4;

    cudaStream_t mainStream = nullptr;
    cublasHandle_t mainBlas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    cudaStream_t updateStreams[updateStreamCount] = {};
    cublasHandle_t updateBlas[updateStreamCount] = {};

    GpuContext() {
        checkCuda(cudaStreamCreateWithFlags(&mainStream, cudaStreamNonBlocking),
                  "cudaStreamCreate(main)");
        checkCublas(cublasCreate(&mainBlas), "cublasCreate(main)");
        checkCublas(cublasSetStream(mainBlas, mainStream), "cublasSetStream(main)");
        checkCusolver(cusolverDnCreate(&solver), "cusolverDnCreate");
        checkCusolver(cusolverDnSetStream(solver, mainStream),
                      "cusolverDnSetStream");
        for (int i = 0; i < updateStreamCount; ++i) {
            checkCuda(cudaStreamCreateWithFlags(&updateStreams[i],
                                                cudaStreamNonBlocking),
                      "cudaStreamCreate(update)");
            checkCublas(cublasCreate(&updateBlas[i]), "cublasCreate(update)");
            checkCublas(cublasSetStream(updateBlas[i], updateStreams[i]),
                        "cublasSetStream(update)");
        }
    }

    GpuContext(const GpuContext&) = delete;
    GpuContext& operator=(const GpuContext&) = delete;

    ~GpuContext() {
        for (int i = 0; i < updateStreamCount; ++i) {
            if (updateBlas[i] != nullptr) {
                cublasDestroy(updateBlas[i]);
            }
            if (updateStreams[i] != nullptr) {
                cudaStreamDestroy(updateStreams[i]);
            }
        }
        if (solver != nullptr) {
            cusolverDnDestroy(solver);
        }
        if (mainBlas != nullptr) {
            cublasDestroy(mainBlas);
        }
        if (mainStream != nullptr) {
            cudaStreamDestroy(mainStream);
        }
    }
};

struct FactorWorkspace {
    double* diagonal = nullptr;
    double* panel = nullptr;
    double* solverWorkspace = nullptr;
    int* deviceInfo = nullptr;
    double* hostDiagonal = nullptr;
    double* hostLocalPanel = nullptr;
    double* hostAllPanels = nullptr;
    int solverWorkspaceElements = 0;
    bool hasCommunicationBuffers = false;

    FactorWorkspace(DistributedTiles& matrix, int ranks, GpuContext& gpu) {
        const size_t tileElements = matrix.tileElements();
        const size_t tileBytes =
            checkedProduct(tileElements, sizeof(double), "tile bytes");
        const int tiles = matrix.tileCount();

        hasCommunicationBuffers = ranks > 1;
        if (hasCommunicationBuffers) {
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&diagonal), tileBytes),
                      "cudaMalloc(diagonal)");
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&panel),
                                 checkedProduct(static_cast<size_t>(tiles), tileBytes,
                                                "panel bytes")),
                      "cudaMalloc(panel)");
        } else {
            diagonal = matrix.tile(0, 0);
        }
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceInfo), sizeof(int)),
                  "cudaMalloc(info)");
        if (hasCommunicationBuffers) {
            checkCuda(cudaMallocHost(reinterpret_cast<void**>(&hostDiagonal), tileBytes),
                      "cudaMallocHost(diagonal)");

            // A panel's tile ordinals are not necessarily consecutive modulo
            // the rank count, so reserve the safe O(number-of-tile-rows) bound.
            const size_t maxLocalTiles = std::max<size_t>(1U, tiles);
            checkCuda(cudaMallocHost(reinterpret_cast<void**>(&hostLocalPanel),
                                     checkedProduct(maxLocalTiles, tileBytes,
                                                    "local panel bytes")),
                      "cudaMallocHost(local panel)");
            checkCuda(cudaMallocHost(
                          reinterpret_cast<void**>(&hostAllPanels),
                          checkedProduct(static_cast<size_t>(std::max(tiles, 1)),
                                         tileBytes, "global panel bytes")),
                      "cudaMallocHost(global panel)");
        }

        checkCusolver(cusolverDnDpotrf_bufferSize(
                          gpu.solver, CUBLAS_FILL_MODE_LOWER, matrix.blockSize(),
                          diagonal, matrix.blockSize(), &solverWorkspaceElements),
                      "cusolverDnDpotrf_bufferSize");
        checkCuda(cudaMalloc(
                      reinterpret_cast<void**>(&solverWorkspace),
                      checkedProduct(static_cast<size_t>(solverWorkspaceElements),
                                     sizeof(double), "POTRF workspace")),
                  "cudaMalloc(POTRF workspace)");

        // Force lazy cuSOLVER module loading before the benchmark timer.  Use
        // a separate scalar so the single-tile input matrix remains intact.
        const double warmupValue = 1.0;
        double* warmupMatrix = nullptr;
        constexpr size_t warmupElements = GpuContext::updateStreamCount + 2U;
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&warmupMatrix),
                             warmupElements * sizeof(double)),
                  "cudaMalloc(cuSOLVER warmup)");
        std::vector<double> warmupValues(warmupElements, warmupValue);
        checkCuda(cudaMemcpyAsync(warmupMatrix, warmupValues.data(),
                                  warmupElements * sizeof(double),
                                  cudaMemcpyHostToDevice, gpu.mainStream),
                  "cudaMemcpyAsync(cuSOLVER warmup)");
        checkCusolver(cusolverDnDpotrf(
                          gpu.solver, CUBLAS_FILL_MODE_LOWER, 1, warmupMatrix, 1,
                          solverWorkspace,
                          solverWorkspaceElements, deviceInfo),
                      "cusolverDnDpotrf(warmup)");
        checkCuda(cudaStreamSynchronize(gpu.mainStream),
                  "cudaStreamSynchronize(cuSOLVER warmup)");

        // Each cuBLAS handle lazily initializes on its first operation.  Warm
        // both the panel and update kernels so that initialization is not
        // mistaken for factorization time on small matrices.
        const double one = 1.0;
        checkCublas(cublasDtrsm(gpu.mainBlas, CUBLAS_SIDE_RIGHT,
                                CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                CUBLAS_DIAG_NON_UNIT, 1, 1, &one, warmupMatrix,
                                1, warmupMatrix + 1, 1),
                    "cublasDtrsm(warmup)");
        for (int stream = 0; stream < GpuContext::updateStreamCount; ++stream) {
            double* output = warmupMatrix + 2 + stream;
            checkCublas(cublasDgemm(gpu.updateBlas[stream], CUBLAS_OP_N,
                                    CUBLAS_OP_T, 1, 1, 1, &one, warmupMatrix, 1,
                                    warmupMatrix, 1, &one, output, 1),
                        "cublasDgemm(warmup)");
            checkCublas(cublasDsyrk(gpu.updateBlas[stream], CUBLAS_FILL_MODE_LOWER,
                                    CUBLAS_OP_N, 1, 1, &one, warmupMatrix, 1,
                                    &one, output, 1),
                        "cublasDsyrk(warmup)");
        }
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(cuBLAS warmup)");
        checkCuda(cudaFree(warmupMatrix), "cudaFree(cuSOLVER warmup)");
    }

    FactorWorkspace(const FactorWorkspace&) = delete;
    FactorWorkspace& operator=(const FactorWorkspace&) = delete;

    ~FactorWorkspace() {
        cudaFree(solverWorkspace);
        cudaFree(deviceInfo);
        if (hasCommunicationBuffers) {
            cudaFree(panel);
            cudaFree(diagonal);
            cudaFreeHost(hostAllPanels);
            cudaFreeHost(hostLocalPanel);
            cudaFreeHost(hostDiagonal);
        }
    }
};

void generatePositiveDefiniteMatrix(DistributedTiles& matrix, int n,
                                    std::vector<double>& randomMatrix,
                                    GpuContext& gpu) {
    const size_t matrixElements = checkedProduct(static_cast<size_t>(n),
                                                 static_cast<size_t>(n),
                                                 "random matrix");
    randomMatrix.resize(matrixElements);
    // Counter-jumped seeds make this deterministic on every rank regardless
    // of its OpenMP thread count, avoiding an O(n^2) inter-node broadcast.
    generateRandomMatrix(randomMatrix);

    double* deviceRandom = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceRandom),
                         checkedProduct(matrixElements, sizeof(double),
                                        "random matrix bytes")),
              "cudaMalloc(random matrix)");
    checkCuda(cudaMemcpyAsync(deviceRandom, randomMatrix.data(),
                              matrixElements * sizeof(double),
                              cudaMemcpyHostToDevice, gpu.mainStream),
              "cudaMemcpyAsync(random matrix)");

    std::vector<double> ones(static_cast<size_t>(matrix.blockSize()), 1.0);
    double* deviceOnes = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceOnes),
                         ones.size() * sizeof(double)), "cudaMalloc(ones)");
    checkCuda(cudaMemcpyAsync(deviceOnes, ones.data(), ones.size() * sizeof(double),
                              cudaMemcpyHostToDevice, gpu.mainStream),
              "cudaMemcpyAsync(ones)");

    const double one = 1.0;
    const double zero = 0.0;
    const double diagonalShift = static_cast<double>(n);
    const int block = matrix.blockSize();
    for (int tileRow = 0; tileRow < matrix.tileCount(); ++tileRow) {
        const int rows = matrix.extent(tileRow);
        const size_t rowStart = static_cast<size_t>(tileRow) * block;
        for (int tileColumn = 0; tileColumn <= tileRow; ++tileColumn) {
            if (!matrix.ownsTile(tileRow, tileColumn)) {
                continue;
            }
            const int columns = matrix.extent(tileColumn);
            const size_t columnStart = static_cast<size_t>(tileColumn) * block;
            // The row-major B allocation is B^T when viewed as column-major.
            if (tileColumn == tileRow) {
                checkCublas(cublasDsyrk(
                                gpu.mainBlas, CUBLAS_FILL_MODE_LOWER,
                                CUBLAS_OP_T, rows, n, &one,
                                deviceRandom + rowStart * n, n, &zero,
                                matrix.tile(tileRow, tileRow), block),
                            "cublasDsyrk(matrix generation)");
                checkCublas(cublasDaxpy(
                                gpu.mainBlas, rows, &diagonalShift, deviceOnes, 1,
                                matrix.tile(tileRow, tileRow), block + 1),
                            "cublasDaxpy(diagonal shift)");
            } else {
                checkCublas(
                    cublasDgemm(gpu.mainBlas, CUBLAS_OP_T, CUBLAS_OP_N, rows,
                                columns, n, &one, deviceRandom + rowStart * n, n,
                                deviceRandom + columnStart * n, n, &zero,
                                matrix.tile(tileRow, tileColumn), block),
                    "cublasDgemm(matrix generation)");
            }
        }
    }
    checkCuda(cudaStreamSynchronize(gpu.mainStream),
              "cudaStreamSynchronize(matrix generation)");
    checkCuda(cudaFree(deviceOnes), "cudaFree(ones)");
    checkCuda(cudaFree(deviceRandom), "cudaFree(random matrix)");
    randomMatrix.clear();
    randomMatrix.shrink_to_fit();
}

std::vector<double> collectMatrix(DistributedTiles& matrix, int n, int rank,
                                  int ranks, bool symmetric) {
    if (matrix.elements() > static_cast<size_t>(INT_MAX)) {
        fatal("MPI_Gatherv", "local tile buffer exceeds MPI count range", 1);
    }

    std::vector<double> local(matrix.elements());
    if (!local.empty()) {
        checkCuda(cudaMemcpy(local.data(), matrix.data(),
                             local.size() * sizeof(double),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(collect matrix)");
    }

    const int localCount = static_cast<int>(local.size());
    std::vector<int> counts(rank == 0 ? static_cast<size_t>(ranks) : 0U);
    checkMpi(MPI_Gather(&localCount, 1, MPI_INT,
                        rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0,
                        MPI_COMM_WORLD),
             "MPI_Gather(tile counts)");

    std::vector<int> displacements;
    std::vector<double> packed;
    if (rank == 0) {
        displacements.resize(static_cast<size_t>(ranks), 0);
        int total = 0;
        for (int process = 0; process < ranks; ++process) {
            displacements[static_cast<size_t>(process)] = total;
            if (counts[static_cast<size_t>(process)] > INT_MAX - total) {
                fatal("MPI_Gatherv", "global tile buffer exceeds MPI count range", 1);
            }
            total += counts[static_cast<size_t>(process)];
        }
        packed.resize(static_cast<size_t>(total));
    }

    checkMpi(MPI_Gatherv(local.data(), localCount, MPI_DOUBLE,
                         rank == 0 ? packed.data() : nullptr,
                         rank == 0 ? counts.data() : nullptr,
                         rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE,
                         0, MPI_COMM_WORLD),
             "MPI_Gatherv(tiles)");

    if (rank != 0) {
        return {};
    }

    const size_t fullElements = checkedProduct(static_cast<size_t>(n),
                                               static_cast<size_t>(n),
                                               "collected matrix");
    std::vector<double> full(fullElements, 0.0);
    const int tiles = matrix.tileCount();
    const int block = matrix.blockSize();
    const size_t tileElements = matrix.tileElements();
#pragma omp parallel for schedule(dynamic)
    for (int tileRow = 0; tileRow < tiles; ++tileRow) {
        const int rows = matrix.extent(tileRow);
        for (int tileColumn = 0; tileColumn <= tileRow; ++tileColumn) {
            const int columns = matrix.extent(tileColumn);
            const size_t ordinal = matrix.ordinal(tileRow, tileColumn);
            const int process = matrix.owner(tileRow, tileColumn);
            const size_t source =
                static_cast<size_t>(displacements[static_cast<size_t>(process)]) +
                (ordinal / static_cast<size_t>(ranks)) * tileElements;
            const double* tile = packed.data() + source;
            for (int column = 0; column < columns; ++column) {
                const size_t globalColumn =
                    static_cast<size_t>(tileColumn) * block + column;
                for (int row = 0; row < rows; ++row) {
                    const size_t globalRow =
                        static_cast<size_t>(tileRow) * block + row;
                    if (globalColumn > globalRow) {
                        continue;
                    }
                    const double value = tile[static_cast<size_t>(column) * block + row];
                    full[globalRow * static_cast<size_t>(n) + globalColumn] = value;
                    if (symmetric && globalRow != globalColumn) {
                        full[globalColumn * static_cast<size_t>(n) + globalRow] = value;
                    }
                }
            }
        }
    }
    return full;
}

bool choleskyDecomposition(DistributedTiles& matrix, int ranks, GpuContext& gpu,
                           FactorWorkspace& scratch) {
    const int block = matrix.blockSize();
    const int tiles = matrix.tileCount();
    const size_t tileElements = matrix.tileElements();
    const size_t tileBytes = checkedProduct(tileElements, sizeof(double), "tile bytes");

    double* const diagonal = scratch.diagonal;
    double* const panel = scratch.panel;
    double* const workspace = scratch.solverWorkspace;
    int* const deviceInfo = scratch.deviceInfo;
    double* const hostDiagonal = scratch.hostDiagonal;
    double* const hostLocalPanel = scratch.hostLocalPanel;
    double* const hostAllPanels = scratch.hostAllPanels;

    std::vector<int> receiveCounts(static_cast<size_t>(ranks));
    std::vector<int> displacements(static_cast<size_t>(ranks));
    std::vector<int> panelOffsets(static_cast<size_t>(tiles));
    const double one = 1.0;
    const double minusOne = -1.0;

    bool success = true;
    for (int step = 0; step < tiles; ++step) {
        // All updates from the preceding step must be visible before factoring
        // the next diagonal tile.
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(step)");
        const int owner = matrix.owner(step, step);
        const int diagonalSize = matrix.extent(step);
        int factorInfo = 0;
        if (g_rank == owner) {
            checkCusolver(cusolverDnDpotrf(
                              gpu.solver, CUBLAS_FILL_MODE_LOWER, diagonalSize,
                              matrix.tile(step, step), block, workspace,
                              scratch.solverWorkspaceElements, deviceInfo),
                          "cusolverDnDpotrf");
            checkCuda(cudaMemcpyAsync(&factorInfo, deviceInfo, sizeof(int),
                                      cudaMemcpyDeviceToHost, gpu.mainStream),
                      "cudaMemcpyAsync(POTRF info)");
            checkCuda(cudaStreamSynchronize(gpu.mainStream),
                      "cudaStreamSynchronize(POTRF)");
        }
        checkMpi(MPI_Bcast(&factorInfo, 1, MPI_INT, owner, MPI_COMM_WORLD),
                 "MPI_Bcast(POTRF info)");
        if (factorInfo != 0) {
            if (g_rank == 0) {
                if (factorInfo > 0) {
                    std::printf("Error: Matrix is not positive definite at diagonal element %d\n",
                                step * block + factorInfo - 1);
                } else {
                    std::printf("Error: cuSOLVER reported invalid argument %d\n",
                                -factorInfo);
                }
            }
            success = false;
            break;
        }

        const double* diagonalForSolve = diagonal;
        if (ranks == 1) {
            diagonalForSolve = matrix.tile(step, step);
        } else if (g_rank == owner) {
            checkCuda(cudaMemcpyAsync(hostDiagonal, matrix.tile(step, step), tileBytes,
                                      cudaMemcpyDeviceToHost, gpu.mainStream),
                      "cudaMemcpyAsync(diagonal D2H)");
            checkCuda(cudaStreamSynchronize(gpu.mainStream),
                      "cudaStreamSynchronize(diagonal D2H)");
        }
        if (ranks > 1) {
            checkMpi(MPI_Bcast(hostDiagonal, static_cast<int>(tileElements), MPI_DOUBLE,
                               owner, MPI_COMM_WORLD),
                     "MPI_Bcast(diagonal tile)");
            checkCuda(cudaMemcpyAsync(diagonal, hostDiagonal, tileBytes,
                                      cudaMemcpyHostToDevice, gpu.mainStream),
                      "cudaMemcpyAsync(diagonal H2D)");
        }

        size_t localPanelTiles = 0;
        for (int tileRow = step + 1; tileRow < tiles; ++tileRow) {
            if (!matrix.ownsTile(tileRow, step)) {
                continue;
            }
            const int rows = matrix.extent(tileRow);
            checkCublas(cublasDtrsm(
                            gpu.mainBlas, CUBLAS_SIDE_RIGHT,
                            CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                            CUBLAS_DIAG_NON_UNIT, rows, diagonalSize, &one,
                            diagonalForSolve, block, matrix.tile(tileRow, step),
                            block),
                        "cublasDtrsm(panel)");
            if (ranks > 1) {
                checkCuda(cudaMemcpyAsync(
                              hostLocalPanel + localPanelTiles * tileElements,
                              matrix.tile(tileRow, step), tileBytes,
                              cudaMemcpyDeviceToHost, gpu.mainStream),
                          "cudaMemcpyAsync(panel D2H)");
            }
            ++localPanelTiles;
        }
        checkCuda(cudaStreamSynchronize(gpu.mainStream),
                  "cudaStreamSynchronize(panel)");

        int totalPanelElements = 0;
        if (ranks > 1) {
            std::fill(receiveCounts.begin(), receiveCounts.end(), 0);
            for (int tileRow = step + 1; tileRow < tiles; ++tileRow) {
                const int process = matrix.owner(tileRow, step);
                if (tileElements > static_cast<size_t>(INT_MAX) ||
                    receiveCounts[static_cast<size_t>(process)] >
                        INT_MAX - static_cast<int>(tileElements)) {
                    fatal("MPI_Allgatherv", "panel exceeds MPI count range", 1);
                }
                receiveCounts[static_cast<size_t>(process)] +=
                    static_cast<int>(tileElements);
            }
            for (int process = 0; process < ranks; ++process) {
                displacements[static_cast<size_t>(process)] = totalPanelElements;
                if (receiveCounts[static_cast<size_t>(process)] >
                    INT_MAX - totalPanelElements) {
                    fatal("MPI_Allgatherv", "panel exceeds MPI count range", 1);
                }
                totalPanelElements += receiveCounts[static_cast<size_t>(process)];
            }

            std::vector<int> nextOffset(static_cast<size_t>(ranks));
            for (int process = 0; process < ranks; ++process) {
                nextOffset[static_cast<size_t>(process)] =
                    displacements[static_cast<size_t>(process)] /
                    static_cast<int>(tileElements);
            }
            for (int tileRow = step + 1; tileRow < tiles; ++tileRow) {
                const int process = matrix.owner(tileRow, step);
                panelOffsets[static_cast<size_t>(tileRow)] =
                    nextOffset[static_cast<size_t>(process)]++;
            }

            checkMpi(MPI_Allgatherv(
                         hostLocalPanel,
                         static_cast<int>(localPanelTiles * tileElements),
                         MPI_DOUBLE, hostAllPanels, receiveCounts.data(),
                         displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD),
                     "MPI_Allgatherv(panel)");
            if (totalPanelElements != 0) {
                checkCuda(cudaMemcpyAsync(panel, hostAllPanels,
                                          static_cast<size_t>(totalPanelElements) *
                                              sizeof(double),
                                          cudaMemcpyHostToDevice, gpu.mainStream),
                          "cudaMemcpyAsync(panel H2D)");
                checkCuda(cudaStreamSynchronize(gpu.mainStream),
                          "cudaStreamSynchronize(panel H2D)");
            }
        }

        size_t operation = 0;
        for (int tileRow = step + 1; tileRow < tiles; ++tileRow) {
            const int rows = matrix.extent(tileRow);
            const double* left = ranks == 1
                                     ? matrix.tile(tileRow, step)
                                     : panel + static_cast<size_t>(panelOffsets[
                                                   static_cast<size_t>(tileRow)]) *
                                                   tileElements;
            for (int tileColumn = step + 1; tileColumn <= tileRow; ++tileColumn) {
                if (!matrix.ownsTile(tileRow, tileColumn)) {
                    continue;
                }
                const int stream =
                    static_cast<int>(operation++ % GpuContext::updateStreamCount);
                const int columns = matrix.extent(tileColumn);
                const double* right = ranks == 1
                                          ? matrix.tile(tileColumn, step)
                                          : panel + static_cast<size_t>(panelOffsets[
                                                        static_cast<size_t>(tileColumn)]) *
                                                        tileElements;
                if (tileRow == tileColumn) {
                    checkCublas(cublasDsyrk(
                                    gpu.updateBlas[stream], CUBLAS_FILL_MODE_LOWER,
                                    CUBLAS_OP_N, rows, diagonalSize, &minusOne, left,
                                    block, &one, matrix.tile(tileRow, tileColumn),
                                    block),
                                "cublasDsyrk(trailing update)");
                } else {
                    checkCublas(cublasDgemm(
                                    gpu.updateBlas[stream], CUBLAS_OP_N,
                                    CUBLAS_OP_T, rows, columns, diagonalSize,
                                    &minusOne, left, block, right, block, &one,
                                    matrix.tile(tileRow, tileColumn), block),
                                "cublasDgemm(trailing update)");
                }
            }
        }
    }

    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(factorization)");
    return success;
}

bool validateCholesky(const std::vector<double>& lower,
                      const std::vector<double>& original, size_t n) {
    double maxError = 0.0;
    double relativeError = 0.0;

#pragma omp parallel for collapse(2) schedule(static) reduction(max : maxError, relativeError)
    for (size_t row = 0; row < n; ++row) {
        for (size_t column = 0; column < n; ++column) {
            double sum = 0.0;
            const size_t terms = std::min(row, column) + 1U;
#pragma omp simd reduction(+ : sum)
            for (size_t k = 0; k < terms; ++k) {
                sum += lower[row * n + k] * lower[column * n + k];
            }
            const size_t index = row * n + column;
            const double error = std::fabs(sum - original[index]);
            maxError = std::max(maxError, error);
            relativeError =
                std::max(relativeError,
                         error / (std::fabs(original[index]) + 1.0e-10));
        }
    }

    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relativeError);
    if (relativeError > 1.0e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseSize(const char* text, size_t& value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > static_cast<unsigned long long>(INT_MAX)) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = 0;
    checkMpi(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED,
                             &providedThreadLevel),
             "MPI_Init_thread");
    int ranks = 1;
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank), "MPI_Comm_rank");
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size");
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        fatal("MPI_Init_thread", "MPI_THREAD_FUNNELED is unavailable", 1);
    }

    size_t matrixSize = 512;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], matrixSize)) {
                if (g_rank == 0) {
                    std::printf("Invalid matrix size: %s\n", argv[i]);
                }
                parseStatus = 1;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (g_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (g_rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            parseStatus = 1;
        }
    }
    if (parseStatus != 0) {
        if (g_rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    checkedProduct(matrixSize, matrixSize, "matrix dimensions");
    const int n = static_cast<int>(matrixSize);
    const int blockSize = chooseBlockSize(matrixSize, ranks);

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank,
                                 MPI_INFO_NULL, &localCommunicator),
             "MPI_Comm_split_type");
    int localRank = 0;
    int localRanks = 1;
    checkMpi(MPI_Comm_rank(localCommunicator, &localRank), "MPI_Comm_rank(local)");
    checkMpi(MPI_Comm_size(localCommunicator, &localRanks), "MPI_Comm_size(local)");
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        fatal("CUDA initialization", "no CUDA-capable GPU found", 1);
    }
    const int device = localRank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice");
    checkCuda(cudaFree(nullptr), "CUDA context initialization");

    // Respect an explicit OpenMP policy; otherwise divide each node's CPU
    // hardware threads evenly among its MPI ranks to avoid oversubscription.
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_dynamic(0);
        const int threadsPerRank = std::max(1, omp_get_num_procs() / localRanks);
        // The host phases are mostly bandwidth-bound; beyond this point extra
        // threads generally add launch/NUMA overhead.  Explicit user settings
        // remain untouched for machines whose topology benefits from more.
        omp_set_num_threads(std::min(32, threadsPerRank));
    }

    const int localGpuSharing = localRanks > deviceCount ? 1 : 0;
    int anyGpuSharing = 0;
    checkMpi(MPI_Reduce(&localGpuSharing, &anyGpuSharing, 1, MPI_INT, MPI_MAX, 0,
                        MPI_COMM_WORLD),
             "MPI_Reduce(GPU sharing)");

    int minThreads = 0;
    int maxThreads = 0;
    const int localThreads = omp_get_max_threads();
    checkMpi(MPI_Reduce(&localThreads, &minThreads, 1, MPI_INT, MPI_MIN, 0,
                        MPI_COMM_WORLD),
             "MPI_Reduce(min threads)");
    checkMpi(MPI_Reduce(&localThreads, &maxThreads, 1, MPI_INT, MPI_MAX, 0,
                        MPI_COMM_WORLD),
             "MPI_Reduce(max threads)");

    if (g_rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", matrixSize, matrixSize);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid configuration: %d MPI rank%s, %d", ranks,
                    ranks == 1 ? "" : "s", minThreads);
        if (minThreads != maxThreads) {
            std::printf("-%d", maxThreads);
        }
        std::printf(" OpenMP thread%s/rank, CUDA GPUs, tile size %d\n",
                    maxThreads == 1 ? "" : "s", blockSize);
        if (anyGpuSharing != 0) {
            std::printf("Note: node-local MPI ranks share the available GPUs when necessary\n");
        }
        std::printf("Generating positive definite matrix...\n");
    }

    int returnCode = 0;
    {
        GpuContext gpu;
        DistributedTiles matrix(n, blockSize, g_rank, ranks);
        std::vector<double> randomMatrix;
        generatePositiveDefiniteMatrix(matrix, n, randomMatrix, gpu);

        std::vector<double> original;
        if (validate) {
            original = collectMatrix(matrix, n, g_rank, ranks, true);
        }

        // Allocate and initialize all factorization scratch storage outside the
        // timing interval, just as the baseline excludes matrix allocation.
        FactorWorkspace scratch(matrix, ranks, gpu);

        if (g_rank == 0) {
            std::printf("Computing Cholesky decomposition...\n");
        }
        checkMpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(start)");
        const double start = MPI_Wtime();
        const bool success = choleskyDecomposition(matrix, ranks, gpu, scratch);
        checkMpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(end)");
        const double localElapsed = MPI_Wtime() - start;
        double elapsed = 0.0;
        checkMpi(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                            MPI_COMM_WORLD),
                 "MPI_Reduce(elapsed)");

        if (!success) {
            if (g_rank == 0) {
                std::printf("Cholesky decomposition failed\n");
            }
            returnCode = 1;
        } else {
            if (g_rank == 0) {
                const long double dimension = static_cast<long double>(matrixSize);
                const long double operations = dimension * dimension * dimension / 3.0L;
                const double gflops = static_cast<double>(operations / elapsed / 1.0e9L);
                std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
                std::printf("Performance: %.3f GFLOPS\n", gflops);
            }

            std::vector<double> lower;
            if (printResults || validate) {
                lower = collectMatrix(matrix, n, g_rank, ranks, false);
            }
            if (g_rank == 0 && printResults) {
                print_results(lower, "CholeskyL");
            }
            if (g_rank == 0 && validate) {
                std::printf("Validating result...\n");
                if (validateCholesky(lower, original, matrixSize)) {
                    std::printf("Validation: PASSED\n");
                } else {
                    std::printf("Validation: FAILED\n");
                    returnCode = 1;
                }
            }
        }
    }

    checkMpi(MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD),
             "MPI_Bcast(return code)");
    MPI_Comm_free(&localCommunicator);
    MPI_Finalize();
    return returnCode;
}
