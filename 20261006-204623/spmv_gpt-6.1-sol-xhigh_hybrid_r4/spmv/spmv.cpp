#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values
//
// Arguments:
//   A: pointer to the array to initialize
//   n: number of elements in the array
//   maxVal: specifies range of random values [0, maxVal]
//
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: initRandomMatrix
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix. The function encodes these positions in compressed sparse
//   row (CSR) format.
//
// Arguments:
//   cols:          array for column indexes of elements (size should be = n)
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   n:             number of nonzero elements
//   dim:           number of rows/columns in the matrix
//
// ****************************************************************************
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;
    const uint64_t entries = static_cast<uint64_t>(dim) * dim;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            uint64_t numEntriesLeft = entries - (static_cast<uint64_t>(i) * dim + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    // Convention: put the number of non-zeroes at the end of the row delimiters array
    rowDelimiters[dim] = n;
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   vec: dense vector of size dim to be used for multiplication
//   dim: number of rows/columns in the matrix
//   out: output - result from the spmv calculation
//
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
    // Rows are independent; preserve the original accumulation order per row.
    const int threads = static_cast<int>(std::min<int64_t>(omp_get_max_threads(),
        std::max<int64_t>(1, (static_cast<int64_t>(dim) + 255) / 256)));
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t i = 0; i < static_cast<int64_t>(dim); ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Function: verifyResults
//
// Purpose:
//   Verifies correctness of results by comparing to reference solution
//
// Arguments:
//   reference: array holding the reference result vector
//   result: array holding the result vector to verify
//   size: number of elements per vector
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (!std::isfinite(ref) || !std::isfinite(res)) {
            printf("Validation failed at index %u: non-finite result\n", i);
            return false;
        }
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i, ref, res, relError);
                return false;
            }
        }
    }
    return true;
}

void mpiCheck(int status) {
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(std::string(message, length));
}

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// MPI-3 collective counts are signed ints. Chunk unusually large buffers,
// without limiting the CSR representation to INT_MAX nonzeros.
template <typename T>
void broadcastBuffer(T* buffer, uint64_t count, MPI_Datatype type) {
    for (uint64_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min<uint64_t>(count - offset, INT_MAX));
        mpiCheck(MPI_Bcast(buffer + offset, chunk, type, 0, MPI_COMM_WORLD));
        offset += chunk;
    }
}

template <typename T>
void distributeBuffer(const std::vector<T>& global, std::vector<T>& local,
                      const std::vector<uint64_t>& counts,
                      const std::vector<uint64_t>& offsets, MPI_Datatype type,
                      int rank) {
    bool small = true;
    for (size_t i = 0; i < counts.size(); ++i)
        small = small && counts[i] <= INT_MAX && offsets[i] <= INT_MAX;
    if (small) {
        std::vector<int> sizes(counts.begin(), counts.end());
        std::vector<int> starts(offsets.begin(), offsets.end());
        mpiCheck(MPI_Scatterv(global.data(), sizes.data(), starts.data(), type,
                             local.data(), sizes[rank], type, 0, MPI_COMM_WORLD));
        return;
    }
    if (rank == 0) {
        std::copy_n(global.data() + offsets[0], counts[0], local.data());
        for (size_t peer = 1; peer < counts.size(); ++peer) {
            for (uint64_t offset = 0; offset < counts[peer];) {
                const int chunk = static_cast<int>(std::min<uint64_t>(counts[peer] - offset, INT_MAX));
                mpiCheck(MPI_Send(global.data() + offsets[peer] + offset, chunk,
                                  type, static_cast<int>(peer), 0, MPI_COMM_WORLD));
                offset += chunk;
            }
        }
    } else {
        for (uint64_t offset = 0; offset < counts[rank];) {
            const int chunk = static_cast<int>(std::min<uint64_t>(counts[rank] - offset, INT_MAX));
            mpiCheck(MPI_Recv(local.data() + offset, chunk, type, 0, 0,
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            offset += chunk;
        }
    }
}

void gatherOutput(const std::vector<double>& local, std::vector<double>& global,
                  const std::vector<uint64_t>& counts,
                  const std::vector<uint64_t>& offsets, int rank) {
    bool small = true;
    for (size_t i = 0; i < counts.size(); ++i)
        small = small && counts[i] <= INT_MAX && offsets[i] <= INT_MAX;
    if (small) {
        std::vector<int> sizes(counts.begin(), counts.end());
        std::vector<int> starts(offsets.begin(), offsets.end());
        mpiCheck(MPI_Gatherv(local.data(), sizes[rank], MPI_DOUBLE, global.data(),
                            sizes.data(), starts.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
        return;
    }
    if (rank == 0) {
        std::copy(local.begin(), local.end(), global.begin());
        for (size_t peer = 1; peer < counts.size(); ++peer) {
            for (uint64_t offset = 0; offset < counts[peer];) {
                const int chunk = static_cast<int>(std::min<uint64_t>(counts[peer] - offset, INT_MAX));
                mpiCheck(MPI_Recv(global.data() + offsets[peer] + offset, chunk,
                                  MPI_DOUBLE, static_cast<int>(peer), 0,
                                  MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                offset += chunk;
            }
        }
    } else {
        for (uint64_t offset = 0; offset < counts[rank];) {
            const int chunk = static_cast<int>(std::min<uint64_t>(counts[rank] - offset, INT_MAX));
            mpiCheck(MPI_Send(local.data() + offset, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
            offset += chunk;
        }
    }
}

// Balance nonzeros plus one unit per row (including empty rows). Whole rows
// remain together, so neither GPU atomics nor distributed reductions are needed.
std::vector<index_t> partitionRows(const std::vector<index_t>& rowOffsets,
                                  const std::vector<int>& weights) {
    std::vector<index_t> boundaries(weights.size() + 1, 0);
    const uint64_t rows = rowOffsets.size() - 1;
    const uint64_t work = rows + rowOffsets.back();
    uint64_t totalWeight = 0, prefixWeight = 0;
    for (int weight : weights) totalWeight += weight;
    for (size_t part = 1; part < weights.size(); ++part) {
        prefixWeight += weights[part - 1];
        const uint64_t target = work * prefixWeight / totalWeight;
        uint64_t low = boundaries[part - 1], high = rows;
        while (low < high) {
            const uint64_t mid = low + (high - low) / 2;
            if (static_cast<uint64_t>(rowOffsets[mid]) + mid < target) low = mid + 1;
            else high = mid;
        }
        boundaries[part] = static_cast<index_t>(low);
    }
    boundaries.back() = static_cast<index_t>(rows);
    return boundaries;
}

// Several cooperating lanes per row coalesce CSR reads. Subwarp widths adapt
// to short rows; each group has its own synchronization mask for partial warps.
template <int Lanes>
__global__ void csrKernel(const double* __restrict__ values,
                          const index_t* __restrict__ columns,
                          const index_t* __restrict__ rowOffsets,
                          const double* __restrict__ vector, index_t rows,
                          double* __restrict__ output) {
    const unsigned lane = threadIdx.x & (Lanes - 1);
    const unsigned mask = (0xffffffffu >> (32 - Lanes))
                        << ((threadIdx.x & 31) & ~(Lanes - 1));
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x / Lanes;
    for (uint64_t row = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / Lanes;
         row < rows; row += stride) {
        double sum = 0.0;
        for (uint64_t item = static_cast<uint64_t>(rowOffsets[row]) + lane;
             item < rowOffsets[row + 1]; item += Lanes)
            sum += values[item] * __ldg(vector + columns[item]);
        for (int delta = Lanes / 2; delta > 0; delta /= 2)
            sum += __shfl_down_sync(mask, sum, delta, Lanes);
        if (lane == 0) output[row] = sum;
    }
}

// A whole block per long row exposes more parallelism when a rank owns too
// few rows to occupy the device with one warp per row.
__global__ void csrBlockKernel(const double* __restrict__ values,
                               const index_t* __restrict__ columns,
                               const index_t* __restrict__ rowOffsets,
                               const double* __restrict__ vector, index_t rows,
                               double* __restrict__ output) {
    __shared__ double partial[4];
    const unsigned lane = threadIdx.x & 31;
    for (uint64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        double sum = 0.0;
        for (uint64_t item = static_cast<uint64_t>(rowOffsets[row]) + threadIdx.x;
             item < rowOffsets[row + 1]; item += 128)
            sum += values[item] * __ldg(vector + columns[item]);
        for (int delta = 16; delta > 0; delta /= 2)
            sum += __shfl_down_sync(0xffffffffu, sum, delta);
        if (lane == 0) partial[threadIdx.x / 32] = sum;
        __syncthreads();
        if (threadIdx.x < 32) {
            sum = lane < 4 ? partial[lane] : 0.0;
            for (int delta = 16; delta > 0; delta /= 2)
                sum += __shfl_down_sync(0xffffffffu, sum, delta);
            if (lane == 0) output[row] = sum;
        }
        __syncthreads();
    }
}

struct DeviceShard {
    int device = 0;
    index_t firstRow = 0, rows = 0, firstItem = 0, items = 0;
    int lanes = 32, blocks = 1;
    bool blockPerRow = false;
    double *values = nullptr, *vector = nullptr, *output = nullptr;
    index_t *columns = nullptr, *rowOffsets = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    index_t graphIterations = 0;
};

void launchSpmv(const DeviceShard& shard) {
    if (shard.rows == 0) return;
    if (shard.blockPerRow) {
        csrBlockKernel<<<shard.blocks, 128, 0, shard.stream>>>(
            shard.values, shard.columns, shard.rowOffsets, shard.vector, shard.rows, shard.output);
    } else {
        #define LAUNCH_CSR(width) csrKernel<width><<<shard.blocks, 128, 0, shard.stream>>>( \
            shard.values, shard.columns, shard.rowOffsets, shard.vector, shard.rows, shard.output)
        switch (shard.lanes) {
            case 1: LAUNCH_CSR(1); break;
            case 2: LAUNCH_CSR(2); break;
            case 4: LAUNCH_CSR(4); break;
            case 8: LAUNCH_CSR(8); break;
            case 16: LAUNCH_CSR(16); break;
            default: LAUNCH_CSR(32); break;
        }
        #undef LAUNCH_CSR
    }
    cudaCheck(cudaGetLastError());
}

void prepareDevice(DeviceShard& shard, const std::vector<double>& values,
                   const std::vector<index_t>& columns,
                   const std::vector<index_t>& rowOffsets,
                   const std::vector<double>& vector, index_t iterations) {
    cudaCheck(cudaSetDevice(shard.device));
    cudaCheck(cudaStreamCreateWithFlags(&shard.stream, cudaStreamNonBlocking));
    if (shard.rows == 0) return;
    cudaDeviceProp properties;
    cudaCheck(cudaGetDeviceProperties(&properties, shard.device));
    const double average = static_cast<double>(shard.items) / shard.rows;
    shard.lanes = average <= 2 ? 1 : average <= 4 ? 2 : average <= 8 ? 4
                : average <= 16 ? 8 : average <= 32 ? 16 : 32;
    shard.blockPerRow = average >= 128 && shard.rows < static_cast<unsigned>(properties.multiProcessorCount * 16);
    const uint64_t requestedBlocks = shard.blockPerRow ? shard.rows
        : (static_cast<uint64_t>(shard.rows) * shard.lanes + 127) / 128;
    shard.blocks = static_cast<int>(std::min<uint64_t>(requestedBlocks, properties.multiProcessorCount * 32));

    if (shard.items != 0) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&shard.values), static_cast<size_t>(shard.items) * sizeof(double)));
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&shard.columns), static_cast<size_t>(shard.items) * sizeof(index_t)));
        cudaCheck(cudaMemcpyAsync(shard.values, values.data() + shard.firstItem,
                                 static_cast<size_t>(shard.items) * sizeof(double), cudaMemcpyHostToDevice, shard.stream));
        cudaCheck(cudaMemcpyAsync(shard.columns, columns.data() + shard.firstItem,
                                 static_cast<size_t>(shard.items) * sizeof(index_t), cudaMemcpyHostToDevice, shard.stream));
    }
    std::vector<index_t> offsets(static_cast<size_t>(shard.rows) + 1);
    for (size_t row = 0; row < offsets.size(); ++row)
        offsets[row] = rowOffsets[static_cast<size_t>(shard.firstRow) + row] - shard.firstItem;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&shard.rowOffsets), offsets.size() * sizeof(index_t)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&shard.vector), vector.size() * sizeof(double)));
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&shard.output), static_cast<size_t>(shard.rows) * sizeof(double)));
    cudaCheck(cudaMemcpyAsync(shard.rowOffsets, offsets.data(), offsets.size() * sizeof(index_t), cudaMemcpyHostToDevice, shard.stream));
    cudaCheck(cudaMemcpyAsync(shard.vector, vector.data(), vector.size() * sizeof(double), cudaMemcpyHostToDevice, shard.stream));
    launchSpmv(shard);
    cudaCheck(cudaStreamSynchronize(shard.stream));

    // Replay bounded batches of kernels to amortize launch overhead without
    // a graph whose size grows with the user's iteration count.
    if (iterations > 1) {
        shard.graphIterations = std::min<index_t>(iterations, 64);
        cudaCheck(cudaStreamBeginCapture(shard.stream, cudaStreamCaptureModeThreadLocal));
        for (index_t i = 0; i < shard.graphIterations; ++i) launchSpmv(shard);
        cudaCheck(cudaStreamEndCapture(shard.stream, &shard.graph));
        cudaCheck(cudaGraphInstantiate(&shard.executable, shard.graph, nullptr, nullptr, 0));
        cudaCheck(cudaGraphLaunch(shard.executable, shard.stream));
        cudaCheck(cudaStreamSynchronize(shard.stream));
    }
}

void computeDevice(const DeviceShard& shard, index_t iterations) {
    cudaCheck(cudaSetDevice(shard.device));
    if (shard.rows == 0) return;
    uint64_t remaining = iterations;
    if (shard.executable) {
        while (remaining >= shard.graphIterations) {
            cudaCheck(cudaGraphLaunch(shard.executable, shard.stream));
            remaining -= shard.graphIterations;
        }
    }
    while (remaining != 0) {
        launchSpmv(shard);
        --remaining;
    }
}

void releaseDevice(DeviceShard& shard) {
    cudaCheck(cudaSetDevice(shard.device));
    if (shard.executable) cudaCheck(cudaGraphExecDestroy(shard.executable));
    if (shard.graph) cudaCheck(cudaGraphDestroy(shard.graph));
    cudaCheck(cudaFree(shard.values));
    cudaCheck(cudaFree(shard.columns));
    cudaCheck(cudaFree(shard.rowOffsets));
    cudaCheck(cudaFree(shard.vector));
    cudaCheck(cudaFree(shard.output));
    cudaCheck(cudaStreamDestroy(shard.stream));
}

// Exceptions must leave the OpenMP region before MPI_Abort is called by the
// main thread, as required by MPI_THREAD_FUNNELED.
template <typename Function>
void forEachDevice(std::vector<DeviceShard>& shards, Function function, bool synchronize = false) {
    std::vector<std::string> errors(shards.size());
    const int threads = std::min<int>(shards.size(), omp_get_max_threads());
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static, 1)
        for (int i = 0; i < static_cast<int>(shards.size()); ++i) {
            try {
                function(shards[i]);
            } catch (const std::exception& error) {
                errors[i] = error.what();
            }
        }
        // Enqueue every device before waiting, even if OMP_NUM_THREADS is
        // smaller than the device count. This preserves GPU concurrency.
        if (synchronize) {
            #pragma omp for schedule(static, 1)
            for (int i = 0; i < static_cast<int>(shards.size()); ++i) {
                if (!errors[i].empty()) continue;
                try {
                    cudaCheck(cudaSetDevice(shards[i].device));
                    cudaCheck(cudaStreamSynchronize(shards[i].stream));
                } catch (const std::exception& error) {
                    errors[i] = error.what();
                }
            }
        }
    }
    for (size_t i = 0; i < errors.size(); ++i)
        if (!errors[i].empty()) throw std::runtime_error("CUDA device " + std::to_string(shards[i].device) + ": " + errors[i]);
}

index_t parseIndex(const char* text) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno || *text == '-' || end == text || *end != '\0' || value > std::numeric_limits<index_t>::max())
        throw std::runtime_error(std::string("Invalid integer: ") + text);
    return static_cast<index_t>(value);
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = parseIndex(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = parseIndex(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = parseIndex(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* text = argv[++i];
            maxVal = std::strtod(text, &end);
            if (end == text || *end != '\0' || !std::isfinite(maxVal))
                throw std::runtime_error("Maximum value must be finite");
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    // Calculate number of non-zero elements
    if (sparsity == 0) throw std::runtime_error("Sparsity must be positive");
    const uint64_t entries = static_cast<uint64_t>(numRows) * numRows;
    if (entries / sparsity > std::numeric_limits<index_t>::max())
        throw std::runtime_error("Matrix exceeds the 32-bit CSR nonzero index range");
    const index_t nItems = static_cast<index_t>(entries / sparsity);

    // Respect rank-local CUDA_VISIBLE_DEVICES masks. With a shared visibility
    // list, divide the node's GPUs among its ranks; with one GPU per rank, use
    // that visible GPU. Oversubscribed ranks share a device rather than falling
    // back to a CPU implementation.
    MPI_Comm node;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int nodeRank = 0, nodeRanks = 0, deviceCount = 0;
    mpiCheck(MPI_Comm_rank(node, &nodeRank));
    mpiCheck(MPI_Comm_size(node, &nodeRanks));
    mpiCheck(MPI_Comm_free(&node));
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) throw std::runtime_error("Every MPI rank requires a visible CUDA device");
    std::vector<int> devices;
    for (int device = nodeRank; device < deviceCount; device += nodeRanks) devices.push_back(device);
    if (devices.empty()) devices.push_back(nodeRank % deviceCount);
    const int localDevices = static_cast<int>(devices.size());
    std::vector<int> rankWeights(ranks);
    mpiCheck(MPI_Allgather(&localDevices, 1, MPI_INT, rankWeights.data(), 1, MPI_INT, MPI_COMM_WORLD));
    int totalDevices = 0;
    for (int weight : rankWeights) totalDevices += weight;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               entries ? 100.0 * (1.0 - static_cast<double>(nItems) / entries) : 100.0);
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; CUDA device assignments: %d; OpenMP threads per rank: up to %d\n",
               ranks, totalDevices, omp_get_max_threads());
        printf("Initializing data structures...\n");
    }

    std::vector<double> globalValues;
    std::vector<index_t> globalColumns, globalRows;
    std::vector<double> h_vec(numRows);
    std::vector<uint64_t> rowCounts(ranks), rowStarts(ranks), itemCounts(ranks), itemStarts(ranks);
    if (rank == 0) {
        globalValues.resize(nItems);
        globalColumns.resize(nItems);
        globalRows.resize(static_cast<size_t>(numRows) + 1);
        // MPI/CUDA initialization must not change the original rand() stream.
        // Default rand() initialization is equivalent to srand(1).
        srand(1);
        fill(h_vec.data(), numRows, maxVal);
        fill(globalValues.data(), nItems, maxVal);
        initRandomMatrix(globalColumns.data(), globalRows.data(), nItems, numRows);
        const auto boundaries = partitionRows(globalRows, rankWeights);
        for (int peer = 0; peer < ranks; ++peer) {
            rowStarts[peer] = boundaries[peer];
            rowCounts[peer] = static_cast<uint64_t>(boundaries[peer + 1]) - boundaries[peer];
            itemStarts[peer] = globalRows[boundaries[peer]];
            itemCounts[peer] = static_cast<uint64_t>(globalRows[boundaries[peer + 1]]) - itemStarts[peer];
        }
    }
    broadcastBuffer(rowCounts.data(), ranks, MPI_UINT64_T);
    broadcastBuffer(rowStarts.data(), ranks, MPI_UINT64_T);
    broadcastBuffer(itemCounts.data(), ranks, MPI_UINT64_T);
    broadcastBuffer(itemStarts.data(), ranks, MPI_UINT64_T);
    broadcastBuffer(h_vec.data(), numRows, MPI_DOUBLE);

    const index_t localRows = static_cast<index_t>(rowCounts[rank]);
    std::vector<double> h_val(itemCounts[rank]);
    std::vector<index_t> h_cols(itemCounts[rank]);
    std::vector<index_t> h_rowDelimiters(static_cast<size_t>(localRows) + 1);
    std::vector<double> h_out(localRows, 0.0);
    distributeBuffer(globalValues, h_val, itemCounts, itemStarts, MPI_DOUBLE, rank);
    distributeBuffer(globalColumns, h_cols, itemCounts, itemStarts, MPI_UINT32_T, rank);
    // Scatter row beginnings without overlapping reads of adjacent boundaries.
    distributeBuffer(globalRows, h_rowDelimiters, rowCounts, rowStarts, MPI_UINT32_T, rank);
    h_rowDelimiters[localRows] = static_cast<index_t>(itemStarts[rank] + itemCounts[rank]);
    const int preprocessingThreads = static_cast<int>(std::min<int64_t>(omp_get_max_threads(),
        std::max<int64_t>(1, (static_cast<int64_t>(localRows) + 4096) / 4096)));
    #pragma omp parallel for schedule(static) num_threads(preprocessingThreads)
    for (int64_t row = 0; row <= static_cast<int64_t>(localRows); ++row)
        h_rowDelimiters[row] -= static_cast<index_t>(itemStarts[rank]);
    // Release the root's full matrix before allocating device partitions.
    std::vector<double>().swap(globalValues);
    std::vector<index_t>().swap(globalColumns);
    std::vector<index_t>().swap(globalRows);

    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        h_reference.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), localRows, h_reference.data());
    }

    const auto deviceBoundaries = partitionRows(h_rowDelimiters, std::vector<int>(devices.size(), 1));
    std::vector<DeviceShard> shards(devices.size());
    for (size_t i = 0; i < shards.size(); ++i) {
        shards[i].device = devices[i];
        shards[i].firstRow = deviceBoundaries[i];
        shards[i].rows = deviceBoundaries[i + 1] - deviceBoundaries[i];
        shards[i].firstItem = h_rowDelimiters[deviceBoundaries[i]];
        shards[i].items = h_rowDelimiters[deviceBoundaries[i + 1]] - shards[i].firstItem;
    }
    forEachDevice(shards, [&](DeviceShard& shard) {
        prepareDevice(shard, h_val, h_cols, h_rowDelimiters, h_vec, iterations);
    });

    if (rank == 0) printf("Computing SpMV...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    forEachDevice(shards, [&](DeviceShard& shard) { computeDevice(shard, iterations); }, true);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    // Report completion of the slowest rank, including launch/synchronization
    // overhead, with no millisecond rounding or barriers between iterations.
    mpiCheck(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) {
        const double gflops = seconds > 0 ? 2.0 * nItems * iterations / seconds / 1e9 : 0.0;
        const double avgTime = iterations ? seconds * 1000.0 / iterations : 0.0;
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    forEachDevice(shards, [&](DeviceShard& shard) {
        cudaCheck(cudaSetDevice(shard.device));
        if (iterations && shard.rows && (validate || printResults)) {
            cudaCheck(cudaMemcpyAsync(h_out.data() + shard.firstRow, shard.output,
                                     static_cast<size_t>(shard.rows) * sizeof(double),
                                     cudaMemcpyDeviceToHost, shard.stream));
            cudaCheck(cudaStreamSynchronize(shard.stream));
        }
        releaseDevice(shard);
    });

    if (printResults) {
        std::vector<double> globalOutput;
        if (rank == 0) globalOutput.resize(numRows);
        gatherOutput(h_out, globalOutput, rowCounts, rowStarts, rank);
        if (rank == 0) print_results(globalOutput, "OutputVector");
    }

    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = verifyResults(h_reference.data(), h_out.data(), localRows) ? 1 : 0;
        int valid = 0;
        mpiCheck(MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    const int status = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (status != MPI_SUCCESS) {
        fprintf(stderr, "MPI initialization failed\n");
        return 1;
    }
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
        if (provided < MPI_THREAD_FUNNELED)
            throw std::runtime_error("MPI_THREAD_FUNNELED support is required");
        const int result = runBenchmark(argc, argv, rank, ranks);
        mpiCheck(MPI_Finalize());
        return result;
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
