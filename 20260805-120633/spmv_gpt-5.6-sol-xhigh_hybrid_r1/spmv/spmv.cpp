#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
using offset_t = uint64_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int CUDA_BLOCK_SIZE = 256;

struct Options {
    index_t numRows = 1024;
    offset_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
    bool help = false;
};

struct RowPartition {
    offset_t begin = 0;
    offset_t count = 0;
};

struct GpuPartition {
    int device = 0;
    int multiprocessors = 1;
    offset_t rowOffset = 0;
    offset_t rows = 0;
    offset_t nnzOffset = 0;
    offset_t nnz = 0;
    double* dVal = nullptr;
    index_t* dCols = nullptr;
    offset_t* dRowDelimiters = nullptr;
    double* dVec = nullptr;
    double* dOut = nullptr;
    cudaStream_t stream = nullptr;
};

[[noreturn]] void abortMpi(const int rank, const char* message, const int errorCode = 1) {
    if (rank >= 0) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    } else {
        std::fprintf(stderr, "%s\n", message);
    }
    MPI_Abort(MPI_COMM_WORLD, errorCode);
    std::abort();
}

void checkMpi(const int status, const int rank, const char* operation) {
    if (status == MPI_SUCCESS) {
        return;
    }
    char error[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(status, error, &length);
    std::string message = std::string(operation) + " failed: " + std::string(error, length);
    abortMpi(rank, message.c_str(), status);
}

void checkCuda(const cudaError_t status, const int rank, const char* operation) {
    if (status == cudaSuccess) {
        return;
    }
    std::string message = std::string(operation) + " failed: " + cudaGetErrorString(status);
    abortMpi(rank, message.c_str(), static_cast<int>(status));
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    std::printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseUnsigned(const char* text, const offset_t maximum, offset_t& value) {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > maximum) {
        return false;
    }
    value = static_cast<offset_t>(parsed);
    return true;
}

bool parseOptions(const int argc, char** argv, Options& options, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            offset_t parsed = 0;
            if (!parseUnsigned(argv[++i], std::numeric_limits<index_t>::max(), parsed) || parsed == 0) {
                error = "-n must be an integer in [1, 4294967295]";
                return false;
            }
            options.numRows = static_cast<index_t>(parsed);
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            if (!parseUnsigned(argv[++i], std::numeric_limits<offset_t>::max(), options.sparsity) ||
                options.sparsity == 0) {
                error = "-s must be a positive integer";
                return false;
            }
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            offset_t parsed = 0;
            if (!parseUnsigned(argv[++i], std::numeric_limits<index_t>::max(), parsed) || parsed == 0) {
                error = "-i must be an integer in [1, 4294967295]";
                return false;
            }
            options.iterations = static_cast<index_t>(parsed);
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            errno = 0;
            char* end = nullptr;
            options.maxVal = std::strtod(argv[++i], &end);
            if (errno != 0 || end == argv[i] || *end != '\0' || !std::isfinite(options.maxVal)) {
                error = "-m must be a finite floating-point value";
                return false;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            options.help = true;
        } else {
            error = std::string("Unknown or incomplete option: ") + argv[i];
            return false;
        }
    }
    return true;
}

RowPartition partitionRows(const offset_t rows, const int part, const int parts) {
    const offset_t quotient = rows / static_cast<offset_t>(parts);
    const offset_t remainder = rows % static_cast<offset_t>(parts);
    RowPartition result;
    result.count = quotient + (static_cast<offset_t>(part) < remainder ? 1 : 0);
    result.begin = static_cast<offset_t>(part) * quotient +
                   std::min(static_cast<offset_t>(part), remainder);
    return result;
}

// Every row receives either floor(nnz/rows) or ceil(nnz/rows) entries. This
// prefix formula lets every MPI rank build its rows without root-side storage.
offset_t nnzPrefix(const offset_t row, const offset_t totalNnz, const offset_t rows) {
    const offset_t quotient = totalNnz / rows;
    const offset_t remainder = totalNnz % rows;
    return row * quotient + std::min(row, remainder);
}

uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

double randomUnit(const uint64_t counter, const uint64_t seed) {
    return static_cast<double>(splitmix64(counter ^ seed) >> 11U) * 0x1.0p-53;
}

offset_t coprimeStep(const offset_t row, const offset_t dimension) {
    if (dimension <= 1) {
        return 1;
    }
    offset_t step = 1 + splitmix64(row ^ 0x243f6a8885a308d3ULL) % (dimension - 1);
    while (std::gcd(step, dimension) != 1) {
        ++step;
        if (step == dimension) {
            step = 1;
        }
    }
    return step;
}

void initializeLocalData(const Options& options, const RowPartition& rankRows,
                         const offset_t totalNnz, std::vector<double>& values,
                         std::vector<index_t>& columns, std::vector<offset_t>& rowDelimiters,
                         std::vector<double>& vector) {
    const offset_t rankNnzBegin = nnzPrefix(rankRows.begin, totalNnz, options.numRows);

    // One persistent OpenMP team initializes all independent host arrays.
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (offset_t i = 0; i < options.numRows; ++i) {
            vector[static_cast<size_t>(i)] = options.maxVal * randomUnit(i, 0xa4093822299f31d0ULL);
        }

#pragma omp for schedule(static)
        for (offset_t localRow = 0; localRow <= rankRows.count; ++localRow) {
            rowDelimiters[static_cast<size_t>(localRow)] =
                nnzPrefix(rankRows.begin + localRow, totalNnz, options.numRows) - rankNnzBegin;
        }

#pragma omp for schedule(static)
        for (offset_t i = 0; i < values.size(); ++i) {
            values[static_cast<size_t>(i)] =
                options.maxVal * randomUnit(rankNnzBegin + i, 0x082efa98ec4e6c89ULL);
        }

        // An affine permutation gives each row unique, pseudo-random columns
        // without a serial dense-matrix scan or a per-row temporary set.
#pragma omp for schedule(static)
        for (offset_t localRow = 0; localRow < rankRows.count; ++localRow) {
            const offset_t globalRow = rankRows.begin + localRow;
            const offset_t first = rowDelimiters[static_cast<size_t>(localRow)];
            const offset_t last = rowDelimiters[static_cast<size_t>(localRow + 1)];
            const offset_t start = splitmix64(globalRow ^ 0x452821e638d01377ULL) % options.numRows;
            const offset_t step = coprimeStep(globalRow, options.numRows);
            for (offset_t j = 0; j < last - first; ++j) {
                columns[static_cast<size_t>(first + j)] =
                    static_cast<index_t>((start + j * step) % options.numRows);
            }
        }
    }
}

void spmvCpuLocal(const double* values, const index_t* columns, const offset_t* rowDelimiters,
                  const double* vector, const offset_t rows, double* output) {
#pragma omp parallel for schedule(static)
    for (offset_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (offset_t j = rowDelimiters[row]; j < rowDelimiters[row + 1]; ++j) {
            sum += values[j] * vector[columns[j]];
        }
        output[row] = sum;
    }
}

bool verifyLocalResults(const double* reference, const double* result, const offset_t size,
                        const offset_t globalRowBegin, const int rank) {
    offset_t firstFailure = size;
#pragma omp parallel for schedule(static) reduction(min : firstFailure)
    for (offset_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        const double error = std::abs(ref) < 1e-10 ? std::abs(res) : std::abs((res - ref) / ref);
        if (error > MAX_RELATIVE_ERROR) {
            firstFailure = std::min(firstFailure, i);
        }
    }
    if (firstFailure == size) {
        return true;
    }
    std::fprintf(stderr,
                 "Rank %d validation failed at global row %" PRIu64
                 ": reference %.10e, got %.10e\n",
                 rank, globalRowBegin + firstFailure, reference[firstFailure], result[firstFailure]);
    return false;
}

__global__ void spmvScalarKernel(const double* __restrict__ values,
                                 const index_t* __restrict__ columns,
                                 const offset_t* __restrict__ rowDelimiters,
                                 const double* __restrict__ vector, const offset_t rows,
                                 double* __restrict__ output) {
    const offset_t thread = static_cast<offset_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const offset_t stride = static_cast<offset_t>(gridDim.x) * blockDim.x;
    for (offset_t row = thread; row < rows; row += stride) {
        double sum = 0.0;
        for (offset_t j = rowDelimiters[row]; j < rowDelimiters[row + 1]; ++j) {
            sum += values[j] * vector[columns[j]];
        }
        output[row] = sum;
    }
}

__global__ void spmvWarpKernel(const double* __restrict__ values,
                               const index_t* __restrict__ columns,
                               const offset_t* __restrict__ rowDelimiters,
                               const double* __restrict__ vector, const offset_t rows,
                               double* __restrict__ output) {
    const int lane = threadIdx.x & (warpSize - 1);
    const offset_t warp =
        (static_cast<offset_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warpSize;
    const offset_t warpStride =
        (static_cast<offset_t>(gridDim.x) * blockDim.x) / warpSize;
    for (offset_t row = warp; row < rows; row += warpStride) {
        double sum = 0.0;
        for (offset_t j = rowDelimiters[row] + lane; j < rowDelimiters[row + 1]; j += warpSize) {
            sum += values[j] * vector[columns[j]];
        }
#pragma unroll
        for (int delta = warpSize / 2; delta > 0; delta /= 2) {
            sum += __shfl_down_sync(0xffffffffU, sum, delta);
        }
        if (lane == 0) {
            output[row] = sum;
        }
    }
}

__global__ void spmvBlockKernel(const double* __restrict__ values,
                                const index_t* __restrict__ columns,
                                const offset_t* __restrict__ rowDelimiters,
                                const double* __restrict__ vector, const offset_t rows,
                                double* __restrict__ output) {
    __shared__ double partial[CUDA_BLOCK_SIZE];
    for (offset_t row = blockIdx.x; row < rows; row += gridDim.x) {
        double sum = 0.0;
        for (offset_t j = rowDelimiters[row] + threadIdx.x; j < rowDelimiters[row + 1];
             j += blockDim.x) {
            sum += values[j] * vector[columns[j]];
        }
        partial[threadIdx.x] = sum;
        __syncthreads();
#pragma unroll
        for (int delta = CUDA_BLOCK_SIZE / 2; delta > 0; delta /= 2) {
            if (threadIdx.x < delta) {
                partial[threadIdx.x] += partial[threadIdx.x + delta];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            output[row] = partial[0];
        }
        __syncthreads();
    }
}

cudaError_t launchSpmv(const GpuPartition& gpu) {
    if (gpu.rows == 0) {
        return cudaSuccess;
    }
    const offset_t averageNnz = gpu.nnz / gpu.rows;
    if (averageNnz < 16) {
        const offset_t needed = (gpu.rows + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        const int blocks = static_cast<int>(std::max<offset_t>(
            1, std::min<offset_t>(needed, static_cast<offset_t>(gpu.multiprocessors) * 8)));
        spmvScalarKernel<<<blocks, CUDA_BLOCK_SIZE, 0, gpu.stream>>>(
            gpu.dVal, gpu.dCols, gpu.dRowDelimiters, gpu.dVec, gpu.rows, gpu.dOut);
    } else if (averageNnz <= 512) {
        constexpr int warpsPerBlock = CUDA_BLOCK_SIZE / 32;
        const offset_t needed = (gpu.rows + warpsPerBlock - 1) / warpsPerBlock;
        const int blocks = static_cast<int>(std::max<offset_t>(
            1, std::min<offset_t>(needed, static_cast<offset_t>(gpu.multiprocessors) * 8)));
        spmvWarpKernel<<<blocks, CUDA_BLOCK_SIZE, 0, gpu.stream>>>(
            gpu.dVal, gpu.dCols, gpu.dRowDelimiters, gpu.dVec, gpu.rows, gpu.dOut);
    } else {
        const int blocks = static_cast<int>(std::max<offset_t>(
            1, std::min<offset_t>(gpu.rows, static_cast<offset_t>(gpu.multiprocessors) * 8)));
        spmvBlockKernel<<<blocks, CUDA_BLOCK_SIZE, 0, gpu.stream>>>(
            gpu.dVal, gpu.dCols, gpu.dRowDelimiters, gpu.dVec, gpu.rows, gpu.dOut);
    }
    return cudaGetLastError();
}

cudaError_t initializeGpu(GpuPartition& gpu, const std::vector<double>& values,
                          const std::vector<index_t>& columns,
                          const std::vector<offset_t>& rowDelimiters,
                          const std::vector<double>& vector) {
    cudaError_t status = cudaSetDevice(gpu.device);
    if (status != cudaSuccess) return status;

    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, gpu.device);
    if (status != cudaSuccess) return status;
    gpu.multiprocessors = properties.multiProcessorCount;

    status = cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) return status;

    if (gpu.nnz != 0) {
        status = cudaMalloc(reinterpret_cast<void**>(&gpu.dVal), static_cast<size_t>(gpu.nnz) * sizeof(double));
        if (status != cudaSuccess) return status;
        status = cudaMalloc(reinterpret_cast<void**>(&gpu.dCols), static_cast<size_t>(gpu.nnz) * sizeof(index_t));
        if (status != cudaSuccess) return status;
    }
    status = cudaMalloc(reinterpret_cast<void**>(&gpu.dRowDelimiters),
                        static_cast<size_t>(gpu.rows + 1) * sizeof(offset_t));
    if (status != cudaSuccess) return status;
    status = cudaMalloc(reinterpret_cast<void**>(&gpu.dVec), vector.size() * sizeof(double));
    if (status != cudaSuccess) return status;
    if (gpu.rows != 0) {
        status = cudaMalloc(reinterpret_cast<void**>(&gpu.dOut), static_cast<size_t>(gpu.rows) * sizeof(double));
        if (status != cudaSuccess) return status;
    }

    std::vector<offset_t> adjustedRows(static_cast<size_t>(gpu.rows + 1));
    const offset_t firstNnz = rowDelimiters[static_cast<size_t>(gpu.rowOffset)];
    for (offset_t i = 0; i <= gpu.rows; ++i) {
        adjustedRows[static_cast<size_t>(i)] =
            rowDelimiters[static_cast<size_t>(gpu.rowOffset + i)] - firstNnz;
    }

    if (gpu.nnz != 0) {
        status = cudaMemcpyAsync(gpu.dVal, values.data() + gpu.nnzOffset,
                                 static_cast<size_t>(gpu.nnz) * sizeof(double),
                                 cudaMemcpyHostToDevice, gpu.stream);
        if (status != cudaSuccess) return status;
        status = cudaMemcpyAsync(gpu.dCols, columns.data() + gpu.nnzOffset,
                                 static_cast<size_t>(gpu.nnz) * sizeof(index_t),
                                 cudaMemcpyHostToDevice, gpu.stream);
        if (status != cudaSuccess) return status;
    }
    status = cudaMemcpyAsync(gpu.dRowDelimiters, adjustedRows.data(),
                             adjustedRows.size() * sizeof(offset_t), cudaMemcpyHostToDevice,
                             gpu.stream);
    if (status != cudaSuccess) return status;
    status = cudaMemcpyAsync(gpu.dVec, vector.data(), vector.size() * sizeof(double),
                             cudaMemcpyHostToDevice, gpu.stream);
    if (status != cudaSuccess) return status;
    return cudaStreamSynchronize(gpu.stream);
}

void destroyGpu(GpuPartition& gpu) {
    if (cudaSetDevice(gpu.device) != cudaSuccess) {
        return;
    }
    if (gpu.dOut != nullptr) cudaFree(gpu.dOut);
    if (gpu.dVec != nullptr) cudaFree(gpu.dVec);
    if (gpu.dRowDelimiters != nullptr) cudaFree(gpu.dRowDelimiters);
    if (gpu.dCols != nullptr) cudaFree(gpu.dCols);
    if (gpu.dVal != nullptr) cudaFree(gpu.dVal);
    if (gpu.stream != nullptr) cudaStreamDestroy(gpu.stream);
}

cudaError_t runGpuIterations(GpuPartition& gpu, const index_t iterations) {
    cudaError_t status = cudaSetDevice(gpu.device);
    if (status != cudaSuccess) return status;
    for (index_t iteration = 0; iteration < iterations; ++iteration) {
        status = launchSpmv(gpu);
        if (status != cudaSuccess) return status;
    }
    return cudaStreamSynchronize(gpu.stream);
}

cudaError_t copyGpuOutput(GpuPartition& gpu, double* output) {
    cudaError_t status = cudaSetDevice(gpu.device);
    if (status != cudaSuccess) return status;
    if (gpu.rows != 0) {
        status = cudaMemcpyAsync(output + gpu.rowOffset, gpu.dOut,
                                 static_cast<size_t>(gpu.rows) * sizeof(double),
                                 cudaMemcpyDeviceToHost, gpu.stream);
        if (status != cudaSuccess) return status;
    }
    return cudaStreamSynchronize(gpu.stream);
}

// Gather in bounded chunks so -r remains correct even when global row offsets
// exceed the signed-int displacement limit of MPI_Gatherv.
void gatherOutput(const std::vector<double>& localOutput, const RowPartition& rankRows,
                  const offset_t globalRows, const int rank, const int worldSize,
                  std::vector<double>& globalOutput) {
    constexpr offset_t gatherChunk = offset_t{1} << 30U;
    std::vector<int> counts(static_cast<size_t>(worldSize));
    std::vector<int> displacements(static_cast<size_t>(worldSize));

    for (offset_t chunkBegin = 0; chunkBegin < globalRows; chunkBegin += gatherChunk) {
        const offset_t chunkEnd = std::min(globalRows, chunkBegin + gatherChunk);
        for (int source = 0; source < worldSize; ++source) {
            const RowPartition sourceRows = partitionRows(globalRows, source, worldSize);
            const offset_t begin = std::max(chunkBegin, sourceRows.begin);
            const offset_t end = std::min(chunkEnd, sourceRows.begin + sourceRows.count);
            counts[static_cast<size_t>(source)] = static_cast<int>(end > begin ? end - begin : 0);
            displacements[static_cast<size_t>(source)] =
                static_cast<int>(end > begin ? begin - chunkBegin : 0);
        }

        const offset_t localBegin = std::max(chunkBegin, rankRows.begin);
        const offset_t localEnd = std::min(chunkEnd, rankRows.begin + rankRows.count);
        const int sendCount = static_cast<int>(localEnd > localBegin ? localEnd - localBegin : 0);
        const double* sendBuffer = sendCount == 0
                                       ? localOutput.data()
                                       : localOutput.data() + (localBegin - rankRows.begin);
        double* receiveBuffer = rank == 0 ? globalOutput.data() + chunkBegin : nullptr;
        checkMpi(MPI_Gatherv(sendBuffer, sendCount, MPI_DOUBLE, receiveBuffer, counts.data(),
                             displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD),
                 rank, "MPI_Gatherv");
    }
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel) != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI initialization failed\n");
        return 1;
    }

    int rank = 0;
    int worldSize = 1;
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), rank, "MPI_Comm_rank");
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &worldSize), rank, "MPI_Comm_size");
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortMpi(rank, "MPI implementation does not provide MPI_THREAD_FUNNELED");
    }

    Options options;
    std::string parseError;
    const bool parsed = parseOptions(argc, argv, options, parseError);
    int allParsed = 0;
    const int locallyParsed = parsed ? 1 : 0;
    checkMpi(MPI_Allreduce(&locallyParsed, &allParsed, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD),
             rank, "MPI_Allreduce(options)");
    if (!allParsed) {
        if (rank == 0) {
            if (!parseError.empty()) std::fprintf(stderr, "%s\n", parseError.c_str());
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    if (options.help) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }

    const offset_t dimension = options.numRows;
    const offset_t denseItems = dimension * dimension;
    const offset_t totalNnz = denseItems / options.sparsity;
    const RowPartition rankRows = partitionRows(dimension, rank, worldSize);
    const offset_t localNnzBegin = nnzPrefix(rankRows.begin, totalNnz, dimension);
    const offset_t localNnzEnd = nnzPrefix(rankRows.begin + rankRows.count, totalNnz, dimension);
    const offset_t localNnz = localNnzEnd - localNnzBegin;

    if (localNnz > std::numeric_limits<size_t>::max() / sizeof(double) ||
        rankRows.count + 1 > std::numeric_limits<size_t>::max() / sizeof(offset_t)) {
        abortMpi(rank, "Local problem is too large for this platform's address space");
    }

    // Split by shared-memory node and assign every node-local CUDA device once
    // when there are no more ranks than devices.
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                 &localCommunicator),
             rank, "MPI_Comm_split_type");
    int localRank = 0;
    int localSize = 1;
    checkMpi(MPI_Comm_rank(localCommunicator, &localRank), rank, "MPI_Comm_rank(local)");
    checkMpi(MPI_Comm_size(localCommunicator, &localSize), rank, "MPI_Comm_size(local)");

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), rank, "cudaGetDeviceCount");
    if (deviceCount == 0) {
        abortMpi(rank, "No CUDA device is visible; this benchmark requires CUDA unconditionally");
    }
    std::vector<int> assignedDevices;
    if (localSize <= deviceCount) {
        for (int device = localRank; device < deviceCount; device += localSize) {
            assignedDevices.push_back(device);
        }
    } else {
        assignedDevices.push_back(localRank % deviceCount);
    }
    checkMpi(MPI_Comm_free(&localCommunicator), rank, "MPI_Comm_free(local)");

    omp_set_dynamic(0);
    const int hostThreads = omp_get_max_threads();
    if (rank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        std::printf("Matrix size: %u x %u\n", options.numRows, options.numRows);
        std::printf("Sparsity: 1 out of %" PRIu64 " entries is non-zero\n", options.sparsity);
        std::printf("Non-zero elements: %" PRIu64 " (%.2f%% sparse)\n", totalNnz,
                    100.0 * (1.0 - static_cast<double>(totalNnz) /
                                       static_cast<double>(denseItems)));
        std::printf("Iterations: %u\n", options.iterations);
        std::printf("Max value: %.2f\n", options.maxVal);
        std::printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP thread(s)/rank\n",
                    worldSize, hostThreads);
        std::printf("Rank 0 CUDA devices: %zu (node-local devices are divided among ranks)\n",
                    assignedDevices.size());
        std::printf("Initializing distributed data structures...\n");
    }

    std::vector<double> hostValues(static_cast<size_t>(localNnz));
    std::vector<index_t> hostColumns(static_cast<size_t>(localNnz));
    std::vector<offset_t> hostRowDelimiters(static_cast<size_t>(rankRows.count + 1));
    std::vector<double> hostVector(static_cast<size_t>(dimension));
    std::vector<double> hostOutput(static_cast<size_t>(rankRows.count));
    initializeLocalData(options, rankRows, totalNnz, hostValues, hostColumns,
                        hostRowDelimiters, hostVector);

    std::vector<double> hostReference;
    if (options.validate) {
        if (rank == 0) std::printf("Computing OpenMP reference solution...\n");
        hostReference.resize(static_cast<size_t>(rankRows.count));
        spmvCpuLocal(hostValues.data(), hostColumns.data(), hostRowDelimiters.data(),
                     hostVector.data(), rankRows.count, hostReference.data());
    }

    std::vector<GpuPartition> gpus(assignedDevices.size());
    for (size_t i = 0; i < gpus.size(); ++i) {
        const RowPartition gpuRows = partitionRows(rankRows.count, static_cast<int>(i),
                                                   static_cast<int>(gpus.size()));
        GpuPartition& gpu = gpus[i];
        gpu.device = assignedDevices[i];
        gpu.rowOffset = gpuRows.begin;
        gpu.rows = gpuRows.count;
        gpu.nnzOffset = hostRowDelimiters[static_cast<size_t>(gpu.rowOffset)];
        gpu.nnz = hostRowDelimiters[static_cast<size_t>(gpu.rowOffset + gpu.rows)] - gpu.nnzOffset;
    }

    std::vector<cudaError_t> gpuStatus(gpus.size(), cudaSuccess);
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(gpus.size()))
    for (size_t i = 0; i < gpus.size(); ++i) {
        gpuStatus[i] = initializeGpu(gpus[i], hostValues, hostColumns, hostRowDelimiters, hostVector);
    }
    for (size_t i = 0; i < gpus.size(); ++i) {
        checkCuda(gpuStatus[i], rank, "CUDA data initialization");
    }

    // Warm up all devices before the timed region to exclude context creation
    // and first-launch setup from steady-state SpMV throughput.
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(gpus.size()))
    for (size_t i = 0; i < gpus.size(); ++i) {
        gpuStatus[i] = runGpuIterations(gpus[i], 1);
    }
    for (size_t i = 0; i < gpus.size(); ++i) {
        checkCuda(gpuStatus[i], rank, "CUDA warm-up");
    }

    if (rank == 0) std::printf("Computing SpMV on CUDA devices...\n");
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), rank, "MPI_Barrier(start)");
    const double start = MPI_Wtime();

#pragma omp parallel for schedule(static) num_threads(static_cast<int>(gpus.size()))
    for (size_t i = 0; i < gpus.size(); ++i) {
        gpuStatus[i] = runGpuIterations(gpus[i], options.iterations);
    }

    const double localDuration = MPI_Wtime() - start;
    for (size_t i = 0; i < gpus.size(); ++i) {
        checkCuda(gpuStatus[i], rank, "CUDA SpMV");
    }
    double duration = 0.0;
    checkMpi(MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
             rank, "MPI_Reduce(time)");

#pragma omp parallel for schedule(static) num_threads(static_cast<int>(gpus.size()))
    for (size_t i = 0; i < gpus.size(); ++i) {
        gpuStatus[i] = copyGpuOutput(gpus[i], hostOutput.data());
    }
    for (size_t i = 0; i < gpus.size(); ++i) {
        checkCuda(gpuStatus[i], rank, "CUDA output copy");
    }

    if (rank == 0) {
        const double milliseconds = duration * 1.0e3;
        const double averageMilliseconds = milliseconds / options.iterations;
        const double gflops = duration > 0.0
                                  ? 2.0 * static_cast<double>(totalNnz) * options.iterations /
                                        duration / 1.0e9
                                  : 0.0;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Average time per iteration: %.6f ms\n", averageMilliseconds);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> globalOutput;
    if (options.printResults) {
        if (rank == 0) globalOutput.resize(static_cast<size_t>(dimension));
        gatherOutput(hostOutput, rankRows, dimension, rank, worldSize, globalOutput);
        if (rank == 0) print_results(globalOutput, "OutputVector");
    }

    int exitCode = 0;
    if (options.validate) {
        const int localValid = verifyLocalResults(hostReference.data(), hostOutput.data(),
                                                  rankRows.count, rankRows.begin, rank)
                                   ? 1
                                   : 0;
        int globallyValid = 0;
        checkMpi(MPI_Allreduce(&localValid, &globallyValid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD),
                 rank, "MPI_Allreduce(validation)");
        if (rank == 0) {
            std::printf("Validating result...\n");
            std::printf("Validation: %s\n", globallyValid ? "PASSED" : "FAILED");
        }
        exitCode = globallyValid ? 0 : 1;
    }

#pragma omp parallel for schedule(static) num_threads(static_cast<int>(gpus.size()))
    for (size_t i = 0; i < gpus.size(); ++i) {
        destroyGpu(gpus[i]);
    }
    checkMpi(MPI_Finalize(), rank, "MPI_Finalize");
    return exitCode;
}
