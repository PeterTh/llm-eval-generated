#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr int BLOCK_SIZE = 64;

void cudaCheck(cudaError_t error, const char* call, int rank) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "Rank %d: CUDA error in %s: %s\n", rank, call,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 2);
}

#define CUDA_CHECK(call) cudaCheck((call), #call, rank)

__device__ __forceinline__ size_t globalRow(size_t localRow, int rank,
                                             int ranks, int block) {
    const size_t localBlock = localRow / static_cast<size_t>(block);
    return (localBlock * static_cast<size_t>(ranks) + rank) * block
           + localRow % block;
}

// The diagonal tile is deliberately factored by one thread: it is small,
// latency-bound, and lies on the critical path.  All O(n^3) work is parallel.
__global__ void factorDiagonal(double* a, double* panel, size_t n,
                               size_t localBase, size_t k, int bs,
                               int* failed) {
    if (threadIdx.x || blockIdx.x) return;
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j <= i; ++j) {
            double value = a[(localBase + i) * n + k + j];
            for (int q = 0; q < j; ++q)
                value -= a[(localBase + i) * n + k + q]
                       * a[(localBase + j) * n + k + q];
            if (i == j) {
                if (!(value > 0.0)) {
                    *failed = 1;
                    return;
                }
                value = sqrt(value);
            } else {
                value /= a[(localBase + j) * n + k + j];
            }
            a[(localBase + i) * n + k + j] = value;
            // Transposed panel layout makes trailing-update reads coalesced.
            panel[static_cast<size_t>(j) * n + k + i] = value;
        }
    }
}

__global__ void solvePanel(double* a, double* panel, size_t n,
                           size_t localRows, size_t k, int bs, int rank,
                           int ranks, int block) {
    const size_t lr = blockIdx.x * blockDim.x + threadIdx.x;
    if (lr >= localRows) return;
    const size_t row = globalRow(lr, rank, ranks, block);
    if (row >= n || row < k + bs) return;

    for (int j = 0; j < bs; ++j) {
        double value = a[lr * n + k + j];
        for (int q = 0; q < j; ++q)
            value -= panel[static_cast<size_t>(q) * n + row]
                   * panel[static_cast<size_t>(q) * n + k + j];
        value /= panel[static_cast<size_t>(j) * n + k + j];
        a[lr * n + k + j] = value;
        panel[static_cast<size_t>(j) * n + row] = value;
    }
}

__global__ void clearRemoteDiagonal(double* panel, size_t k, int bs,
                                    size_t n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= bs * bs) return;
    const int i = index / bs;
    const int j = index % bs;
    if (j <= i) panel[static_cast<size_t>(j) * n + k + i] = 0.0;
}

__global__ void updateTrailing(double* a, const double* panel, size_t n,
                               size_t localRows, size_t first, int bs,
                               int rank, int ranks, int block) {
    const size_t col = first + blockIdx.x * blockDim.x + threadIdx.x;
    const size_t lr = blockIdx.y * blockDim.y + threadIdx.y;
    if (lr >= localRows || col >= n) return;
    const size_t row = globalRow(lr, rank, ranks, block);
    if (row >= n || row < first || col > row) return;

    double value = a[lr * n + col];
#pragma unroll 4
    for (int q = 0; q < bs; ++q)
        value -= panel[static_cast<size_t>(q) * n + row]
               * panel[static_cast<size_t>(q) * n + col];
    a[lr * n + col] = value;
}

__global__ void zeroUpper(double* a, size_t n, size_t localRows, int rank,
                          int ranks, int block) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t lr = blockIdx.y * blockDim.y + threadIdx.y;
    if (lr >= localRows || col >= n) return;
    const size_t row = globalRow(lr, rank, ranks, block);
    if (row >= n || col > row) a[lr * n + col] = 0.0;
}

size_t localBlockCount(size_t globalBlocks, int rank, int ranks) {
    if (static_cast<size_t>(rank) >= globalBlocks) return 0;
    return 1 + (globalBlocks - 1 - rank) / ranks;
}

size_t rowFromLocal(size_t lr, int rank, int ranks, int block) {
    return ((lr / block) * static_cast<size_t>(ranks) + rank) * block
           + lr % block;
}

void generateLocalMatrix(std::vector<double>& local, size_t localRows,
                         const std::vector<double>& b, size_t n, int rank,
                         int ranks, int block) {
#pragma omp parallel for schedule(static)
    for (long long lrSigned = 0; lrSigned < static_cast<long long>(localRows);
         ++lrSigned) {
        const size_t lr = static_cast<size_t>(lrSigned);
        const size_t row = rowFromLocal(lr, rank, ranks, block);
        if (row >= n) continue;
        for (size_t col = 0; col < n; ++col) {
            double sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (size_t q = 0; q < n; ++q)
                sum += b[row * n + q] * b[col * n + q];
            local[lr * n + col] = sum + (row == col ? n : 0.0);
        }
    }
}

bool validateCholesky(const std::vector<double>& l,
                      const std::vector<double>& original, size_t n) {
    double maxError = 0.0;
    double relError = 0.0;
#pragma omp parallel for reduction(max:maxError,relError) schedule(static)
    for (long long index = 0; index < static_cast<long long>(n * n); ++index) {
        const size_t i = static_cast<size_t>(index) / n;
        const size_t j = static_cast<size_t>(index) % n;
        double sum = 0.0;
#pragma omp simd reduction(+:sum)
        for (size_t q = 0; q <= std::min(i, j); ++q)
            sum += l[i * n + q] * l[j * n + q];
        const double error = std::fabs(sum - original[index]);
        maxError = std::max(maxError, error);
        relError = std::max(relError,
                            error / (std::fabs(original[index]) + 1e-10));
    }
    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
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

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI lacks required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    size_t n = 512;
    bool validate = false, printResults = false, help = false;
    bool argumentsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            argumentsValid = end && *end == '\0' && value > 0
                             && value <= std::numeric_limits<size_t>::max();
            if (argumentsValid) n = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else argumentsValid = false;
    }
    if (help || !argumentsValid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return help ? 0 : 1;
    }
    if (n > std::numeric_limits<size_t>::max() / n ||
        n * n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "Matrix is too large for MPI counts\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    const int block = std::min<int>(BLOCK_SIZE, static_cast<int>(n));
    const size_t globalBlocks = (n + block - 1) / block;
    const size_t localBlocks = localBlockCount(globalBlocks, rank, ranks);
    const size_t localRows = localBlocks * block;
    const size_t localElements = localRows * n;
    int countFits = localElements <=
                    static_cast<size_t>(std::numeric_limits<int>::max());
    MPI_Allreduce(MPI_IN_PLACE, &countFits, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if (!countFits) {
        if (rank == 0)
            std::fprintf(stderr, "Per-rank matrix exceeds MPI count limit\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP thread(s), CUDA block size %d\n",
                    ranks, omp_get_max_threads(), block);
        std::printf("Generating positive definite matrix...\n");
    }

    std::vector<double> b(n * n);
    unsigned int seed = 42;
    for (double& value : b)
        value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    std::vector<double> local(localElements, 0.0);
    generateLocalMatrix(local, localRows, b, n, rank, ranks, block);

    std::vector<double> original;
    if (validate && rank == 0) {
        original.resize(n * n);
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(n * n); ++i) {
            const size_t row = static_cast<size_t>(i) / n;
            const size_t col = static_cast<size_t>(i) % n;
            double sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (size_t q = 0; q < n; ++q)
                sum += b[row * n + q] * b[col * n + q];
            original[i] = sum + (row == col ? n : 0.0);
        }
    }
    b.clear();
    b.shrink_to_fit();

    double *deviceA = nullptr, *devicePanel = nullptr;
    int* deviceFailed = nullptr;
    double* hostPanel = nullptr;
    // MPI jobs may contain more ranks than matrix block rows.  Such ranks
    // still participate in collectives without issuing zero-sized launches.
    CUDA_CHECK(cudaMalloc(&deviceA,
                          std::max<size_t>(1, localElements) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&devicePanel, n * block * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&deviceFailed, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&hostPanel, n * block * sizeof(double)));
    if (localElements != 0)
        CUDA_CHECK(cudaMemcpy(deviceA, local.data(),
                              localElements * sizeof(double),
                              cudaMemcpyHostToDevice));

    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    bool success = true;

    for (size_t k = 0, blockIndex = 0; k < n; k += block, ++blockIndex) {
        const int bs = static_cast<int>(std::min<size_t>(block, n - k));
        const int owner = static_cast<int>(blockIndex % ranks);
        CUDA_CHECK(cudaMemset(devicePanel, 0, n * block * sizeof(double)));

        if (rank == owner) {
            const size_t localBlock = blockIndex / ranks;
            const size_t localBase = localBlock * block;
            CUDA_CHECK(cudaMemset(deviceFailed, 0, sizeof(int)));
            factorDiagonal<<<1, 1>>>(deviceA, devicePanel, n, localBase, k,
                                     bs, deviceFailed);
            CUDA_CHECK(cudaGetLastError());
            int failed = 0;
            CUDA_CHECK(cudaMemcpy(&failed, deviceFailed, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            success = failed == 0;
            CUDA_CHECK(cudaMemcpy2D(hostPanel, bs * sizeof(double),
                                    devicePanel + k, n * sizeof(double),
                                    bs * sizeof(double), bs,
                                    cudaMemcpyDeviceToHost));
        }
        int ok = success ? 1 : 0;
        MPI_Bcast(&ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!ok) { success = false; break; }
        MPI_Bcast(hostPanel, bs * bs, MPI_DOUBLE, owner,
                  MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy2D(devicePanel + k, n * sizeof(double),
                                hostPanel, bs * sizeof(double),
                                bs * sizeof(double), bs,
                                cudaMemcpyHostToDevice));

        if (localRows != 0) {
            solvePanel<<<static_cast<unsigned>((localRows + 255) / 256), 256>>>(
                deviceA, devicePanel, n, localRows, k, bs, rank, ranks, block);
            CUDA_CHECK(cudaGetLastError());
        }
        if (rank != owner) {
            clearRemoteDiagonal<<<(bs * bs + 255) / 256, 256>>>(
                devicePanel, k, bs, n);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMemcpy(hostPanel, devicePanel,
                              n * block * sizeof(double),
                              cudaMemcpyDeviceToHost));
        MPI_Allreduce(MPI_IN_PLACE, hostPanel, static_cast<int>(n * block),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(devicePanel, hostPanel,
                              n * block * sizeof(double),
                              cudaMemcpyHostToDevice));

        const size_t first = k + bs;
        if (first < n && localRows != 0) {
            const dim3 threads(16, 16);
            const dim3 grid(static_cast<unsigned>((n - first + 15) / 16),
                            static_cast<unsigned>((localRows + 15) / 16));
            updateTrailing<<<grid, threads>>>(deviceA, devicePanel, n,
                localRows, first, bs, rank, ranks, block);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        cudaFreeHost(hostPanel); cudaFree(deviceFailed);
        cudaFree(devicePanel); cudaFree(deviceA);
        MPI_Finalize();
        return 1;
    }

    if (localRows != 0) {
        const dim3 finalThreads(16, 16);
        const dim3 finalGrid(static_cast<unsigned>((n + 15) / 16),
                             static_cast<unsigned>((localRows + 15) / 16));
        zeroUpper<<<finalGrid, finalThreads>>>(deviceA, n, localRows, rank,
                                               ranks, block);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(local.data(), deviceA,
                              localElements * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    const int localCount = static_cast<int>(localElements);
    std::vector<int> counts(ranks), displacements(ranks);
    MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
               MPI_COMM_WORLD);
    std::vector<double> packed;
    if (rank == 0) {
        int total = 0;
        for (int r = 0; r < ranks; ++r) {
            displacements[r] = total;
            total += counts[r];
        }
        packed.resize(total);
    }
    MPI_Gatherv(local.data(), localCount, MPI_DOUBLE, packed.data(),
                counts.data(), displacements.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    std::vector<double> result;
    if (rank == 0) {
        result.assign(n * n, 0.0);
#pragma omp parallel for schedule(static)
        for (int r = 0; r < ranks; ++r) {
            const size_t rows = static_cast<size_t>(counts[r]) / n;
            const double* source = packed.data() + displacements[r];
            for (size_t lr = 0; lr < rows; ++lr) {
                const size_t row = rowFromLocal(lr, r, ranks, block);
                if (row < n)
                    std::memcpy(result.data() + row * n, source + lr * n,
                                n * sizeof(double));
            }
        }
        const long long milliseconds = static_cast<long long>(elapsed * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        std::printf("Performance: %.3f GFLOPS\n",
                    elapsed > 0.0 ? operations / elapsed / 1e9 : 0.0);
        if (printResults) print_results(result, "CholeskyL");
    }

    bool valid = true;
    if (validate && rank == 0) {
        std::printf("Validating result...\n");
        valid = validateCholesky(result, original, n);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    cudaFreeHost(hostPanel); cudaFree(deviceFailed);
    cudaFree(devicePanel); cudaFree(deviceA);
    MPI_Finalize();
    return valid ? 0 : 1;
}
