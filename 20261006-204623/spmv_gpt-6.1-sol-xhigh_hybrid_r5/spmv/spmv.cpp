#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <numeric>
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
    // Preserve the original rand() streams. Positions are generated only on
    // rank zero, after the vector and values have consumed the default stream.
    srand(8675309);
    const uint64_t cells = static_cast<uint64_t>(dim) * dim;
    if (n == 0) {
        #pragma omp parallel for schedule(static) if(dim > 4096)
        for (int64_t row = 0; row <= dim; ++row) {
            rowDelimiters[row] = 0;
        }
        return;
    }
    if (n == cells) {
        // Every position is selected; the discarded random draws cannot
        // affect any later computation, so populate dense CSR in parallel.
        #pragma omp parallel for schedule(static) if(dim > 128)
        for (int64_t row = 0; row < dim; ++row) {
            const uint64_t offset = static_cast<uint64_t>(row) * dim;
            rowDelimiters[row] = static_cast<index_t>(offset);
            for (index_t col = 0; col < dim; ++col) {
                cols[offset + col] = col;
            }
        }
        rowDelimiters[dim] = n;
        return;
    }

    const double prob = static_cast<double>(n) / static_cast<double>(cells);
    index_t nnzAssigned = 0;
    bool fillRemaining = false;
    for (index_t row = 0; row < dim; ++row) {
        rowDelimiters[row] = nnzAssigned;
        for (index_t col = 0; col < dim && nnzAssigned < n; ++col) {
            const uint64_t entriesLeft = cells - (static_cast<uint64_t>(row) * dim + col);
            if (entriesLeft <= n - nnzAssigned) {
                fillRemaining = true;
            }
            const double randVal = static_cast<double>(rand()) / RAND_MAX;
            if (randVal <= prob || fillRemaining) {
                cols[nnzAssigned++] = col;
            }
        }
        if (nnzAssigned == n) {
            #pragma omp parallel for schedule(static) if(dim - row > 4096)
            for (int64_t tail = static_cast<int64_t>(row) + 1; tail <= dim; ++tail) {
                rowDelimiters[tail] = n;
            }
            return;
        }
    }
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
    #pragma omp parallel for schedule(dynamic, 32) if(dim > 1024)
    for (int64_t i = 0; i < dim; ++i) {
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
//   globalOffset: first row owned by this MPI rank
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t globalOffset) {
    int64_t firstBad = size;
    #pragma omp parallel for schedule(static) reduction(min:firstBad) if(size > 4096)
    for (int64_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        const double error = std::abs(ref) < 1e-10
            ? std::abs(res) : std::abs((res - ref) / ref);
        if (!std::isfinite(res) || error > MAX_RELATIVE_ERROR) {
            firstBad = std::min(firstBad, i);
        }
    }
    if (firstBad != size) {
        printf("Validation failed at index %u: reference %.10e, got %.10e\n",
               globalOffset + static_cast<index_t>(firstBad), reference[firstBad], result[firstBad]);
        return false;
    }
    return true;
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

namespace {

[[noreturn]] void abortMpi(const std::string& message) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s\n", rank, message.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void mpiCheck(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        abortMpi(std::string(operation) + ": " + std::string(message, length));
    }
}

// CUDA failures in OpenMP workers are caught there and reported by the main
// thread. Thus MPI calls always obey MPI_THREAD_FUNNELED.
void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
T* deviceAllocate(size_t count) {
    T* pointer = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&pointer), std::max(size_t{1}, count) * sizeof(T)),
              "cudaMalloc");
    return pointer;
}

template <typename T>
void upload(T* device, const T* host, size_t count, cudaStream_t stream) {
    if (count != 0) {
        cudaCheck(cudaMemcpyAsync(device, host, count * sizeof(T), cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync to device");
    }
}

// Subwarps coalesce CSR reads; short rows use fewer lanes to avoid wasting a
// full warp. Each row has exactly one owner, so no atomics are necessary.
template <unsigned Lanes>
__global__ void spmvKernel(const double* __restrict__ values,
                           const index_t* __restrict__ columns,
                           const index_t* __restrict__ rows,
                           const double* __restrict__ vector,
                           double* __restrict__ output,
                           index_t numRows, index_t nonzeroBase) {
    const unsigned lane = threadIdx.x & (Lanes - 1);
    const unsigned group = (threadIdx.x & 31u) & ~(Lanes - 1);
    const unsigned mask = (0xffffffffu >> (32 - Lanes)) << group;
    const uint64_t first = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / Lanes;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x / Lanes;
    for (uint64_t row = first; row < numRows; row += stride) {
        double sum = 0.0;
        const uint64_t begin = rows[row] - nonzeroBase;
        const uint64_t end = rows[row + 1] - nonzeroBase;
        for (uint64_t item = begin + lane; item < end; item += Lanes) {
            sum += values[item] * __ldg(vector + columns[item]);
        }
        #pragma unroll
        for (unsigned offset = Lanes / 2; offset != 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset, Lanes);
        }
        if (lane == 0) {
            output[row] = sum;
        }
    }
}

// Long rows benefit from a whole block: more outstanding memory requests
// and a shorter dependency chain than a single warp can provide.
template <unsigned Threads>
__global__ void spmvBlockKernel(const double* __restrict__ values,
                                const index_t* __restrict__ columns,
                                const index_t* __restrict__ rows,
                                const double* __restrict__ vector,
                                double* __restrict__ output,
                                index_t numRows, index_t nonzeroBase) {
    __shared__ double partials[Threads / 32];
    const unsigned lane = threadIdx.x & 31u;
    for (uint64_t row = blockIdx.x; row < numRows; row += gridDim.x) {
        const uint64_t begin = rows[row] - nonzeroBase;
        const uint64_t end = rows[row + 1] - nonzeroBase;
        double sum = 0.0;
        for (uint64_t item = begin + threadIdx.x; item < end; item += Threads) {
            sum += values[item] * __ldg(vector + columns[item]);
        }
        #pragma unroll
        for (unsigned offset = 16; offset != 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) partials[threadIdx.x / 32] = sum;
        __syncthreads();
        if (threadIdx.x < 32) {
            sum = lane < Threads / 32 ? partials[lane] : 0.0;
            #pragma unroll
            for (unsigned offset = Threads / 64; offset != 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            }
            if (lane == 0) output[row] = sum;
        }
        __syncthreads();
    }
}

struct GpuWorker {
    int device = -1;
    index_t firstRow = 0;
    index_t numRows = 0;
    index_t nonzeroBase = 0;
    unsigned lanes = 1;
    unsigned blocks = 1;
    bool blockRows = false;
    index_t batchSize = 0;
    double* values = nullptr;
    index_t* columns = nullptr;
    index_t* rows = nullptr;
    double* vector = nullptr;
    double* output = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t graph = nullptr;

    GpuWorker() = default;
    GpuWorker(const GpuWorker&) = delete;
    GpuWorker& operator=(const GpuWorker&) = delete;

    ~GpuWorker() {
        if (device < 0) return;
        cudaSetDevice(device);
        if (graph) cudaGraphExecDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
        if (output) cudaFree(output);
        if (vector) cudaFree(vector);
        if (rows) cudaFree(rows);
        if (columns) cudaFree(columns);
        if (values) cudaFree(values);
    }

    void launch() {
        if (blockRows) {
            spmvBlockKernel<128><<<blocks, 128, 0, stream>>>(values, columns, rows, vector, output, numRows, nonzeroBase);
            cudaCheck(cudaGetLastError(), "SpMV block kernel launch");
            return;
        }
        switch (lanes) {
            case 1: spmvKernel<1><<<blocks, 256, 0, stream>>>(values, columns, rows, vector, output, numRows, nonzeroBase); break;
            case 2: spmvKernel<2><<<blocks, 256, 0, stream>>>(values, columns, rows, vector, output, numRows, nonzeroBase); break;
            case 4: spmvKernel<4><<<blocks, 256, 0, stream>>>(values, columns, rows, vector, output, numRows, nonzeroBase); break;
            case 8: spmvKernel<8><<<blocks, 256, 0, stream>>>(values, columns, rows, vector, output, numRows, nonzeroBase); break;
            case 16: spmvKernel<16><<<blocks, 256, 0, stream>>>(values, columns, rows, vector, output, numRows, nonzeroBase); break;
            default: spmvKernel<32><<<blocks, 256, 0, stream>>>(values, columns, rows, vector, output, numRows, nonzeroBase); break;
        }
        cudaCheck(cudaGetLastError(), "SpMV kernel launch");
    }

    void initialize(const double* hostValues, const index_t* hostColumns,
                    const index_t* hostRows, const double* hostVector,
                    index_t dimension, index_t iterations) {
        cudaCheck(cudaSetDevice(device), "cudaSetDevice");
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        nonzeroBase = hostRows[firstRow];
        const index_t nonzeros = hostRows[firstRow + numRows] - nonzeroBase;
        values = deviceAllocate<double>(nonzeros);
        columns = deviceAllocate<index_t>(nonzeros);
        rows = deviceAllocate<index_t>(static_cast<size_t>(numRows) + 1);
        vector = deviceAllocate<double>(dimension);
        output = deviceAllocate<double>(numRows);
        if (nonzeros != 0) {
            upload(values, hostValues + nonzeroBase, nonzeros, stream);
            upload(columns, hostColumns + nonzeroBase, nonzeros, stream);
        }
        upload(rows, hostRows + firstRow, static_cast<size_t>(numRows) + 1, stream);
        upload(vector, hostVector, dimension, stream);
        cudaCheck(cudaMemsetAsync(output, 0, std::max(size_t{1}, static_cast<size_t>(numRows)) * sizeof(double), stream),
                  "cudaMemsetAsync");
        cudaCheck(cudaStreamSynchronize(stream), "initial uploads");
        if (numRows == 0 || iterations == 0) return;

        const double average = static_cast<double>(nonzeros) / numRows;
        blockRows = average >= 512;
        lanes = average <= 2 ? 1 : average <= 4 ? 2 : average <= 8 ? 4
              : average <= 16 ? 8 : average <= 32 ? 16 : 32;
        cudaDeviceProp properties;
        cudaCheck(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        blocks = static_cast<unsigned>(std::min<uint64_t>(
            blockRows ? numRows : (static_cast<uint64_t>(numRows) * lanes + 255) / 256,
            static_cast<uint64_t>(properties.multiProcessorCount) * 16));

        // Instantiate a bounded graph once, then reuse it without changing
        // the matrix or input vector. A tail handles non-multiple iteration counts.
        batchSize = std::min<index_t>(iterations, 64);
        cudaGraph_t captured = nullptr;
        cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
        for (index_t i = 0; i < batchSize; ++i) launch();
        cudaCheck(cudaStreamEndCapture(stream, &captured), "cudaStreamEndCapture");
        const cudaError_t status = cudaGraphInstantiate(&graph, captured, nullptr, nullptr, 0);
        cudaGraphDestroy(captured);
        cudaCheck(status, "cudaGraphInstantiate");
        // Warm up graph upload, JIT compilation and the device before timing.
        cudaCheck(cudaGraphLaunch(graph, stream), "warmup cudaGraphLaunch");
        cudaCheck(cudaStreamSynchronize(stream), "warmup synchronization");
    }

    void submit(index_t iterations) {
        if (!graph) return;
        cudaCheck(cudaSetDevice(device), "cudaSetDevice");
        for (index_t i = 0; i < iterations / batchSize; ++i) {
            cudaCheck(cudaGraphLaunch(graph, stream), "cudaGraphLaunch");
        }
        for (index_t i = 0; i < iterations % batchSize; ++i) launch();
    }

    void synchronize() {
        cudaCheck(cudaSetDevice(device), "cudaSetDevice");
        cudaCheck(cudaStreamSynchronize(stream), "SpMV synchronization");
    }

    void download(double* hostOutput) {
        if (numRows == 0) return;
        cudaCheck(cudaSetDevice(device), "cudaSetDevice");
        cudaCheck(cudaMemcpyAsync(hostOutput + firstRow, output, static_cast<size_t>(numRows) * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync to host");
        cudaCheck(cudaStreamSynchronize(stream), "result download");
    }
};

void checkWorkers(const std::vector<std::string>& errors) {
    for (const auto& error : errors) {
        if (!error.empty()) abortMpi(error);
    }
}

// Balance nonzeros plus a small per-row cost (important for empty rows).
// Whole rows preserve CSR ownership and avoid partial-result communication.
index_t balancedCut(const index_t* delimiters, index_t numRows, uint64_t part,
                    uint64_t parts, index_t previous) {
    const uint64_t total = delimiters[numRows] + uint64_t{4} * numRows;
    const uint64_t target = (total / parts) * part + ((total % parts) * part) / parts;
    index_t low = previous;
    index_t high = numRows;
    while (low < high) {
        const index_t middle = low + (high - low) / 2;
        const uint64_t cost = delimiters[middle] + uint64_t{4} * middle;
        if (cost < target) low = middle + 1;
        else high = middle;
    }
    if (low > previous) {
        const uint64_t before = delimiters[low - 1] + uint64_t{4} * (low - 1);
        const uint64_t after = delimiters[low] + uint64_t{4} * low;
        if (after - target > target - before) --low;
    }
    return low;
}

// Fast collective path for usual sizes. The chunked path also supports CSR
// arrays beyond MPI's signed-int count/displacement limit without MPI-4.
template <typename T>
void scatterArray(const T* source, T* destination, const std::vector<index_t>& cuts,
                  MPI_Datatype datatype, int rank, int tag) {
    const int ranks = static_cast<int>(cuts.size()) - 1;
    const uint64_t localCount = static_cast<uint64_t>(cuts[rank + 1]) - cuts[rank];
    if (cuts.back() <= static_cast<index_t>(INT_MAX)) {
        std::vector<int> counts(ranks), displacements(ranks);
        for (int r = 0; r < ranks; ++r) {
            counts[r] = static_cast<int>(cuts[r + 1] - cuts[r]);
            displacements[r] = static_cast<int>(cuts[r]);
        }
        mpiCheck(MPI_Scatterv(source, counts.data(), displacements.data(), datatype,
                             destination, static_cast<int>(localCount), datatype, 0, MPI_COMM_WORLD),
                 "MPI_Scatterv");
        return;
    }
    constexpr uint64_t chunk = uint64_t{1} << 26;
    if (rank == 0) {
        if (localCount != 0) std::copy_n(source, static_cast<size_t>(localCount), destination);
        std::vector<MPI_Request> requests;
        for (int r = 1; r < ranks; ++r) {
            for (uint64_t offset = cuts[r]; offset < cuts[r + 1]; offset += chunk) {
                const int count = static_cast<int>(std::min<uint64_t>(chunk, cuts[r + 1] - offset));
                requests.push_back(MPI_REQUEST_NULL);
                mpiCheck(MPI_Isend(source + offset, count, datatype, r, tag, MPI_COMM_WORLD, &requests.back()),
                         "MPI_Isend matrix chunk");
            }
        }
        if (!requests.empty()) {
            mpiCheck(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE),
                     "MPI_Waitall matrix chunks");
        }
    } else {
        for (uint64_t offset = 0; offset < localCount; offset += chunk) {
            const int count = static_cast<int>(std::min(chunk, localCount - offset));
            mpiCheck(MPI_Recv(destination + offset, count, datatype, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                     "MPI_Recv matrix chunk");
        }
    }
}

index_t parseIndex(const char* text, const char* option, uint64_t maximum = UINT32_MAX) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (*text == '-' || end == text || *end != '\0' || errno == ERANGE || value > maximum) {
        throw std::runtime_error(std::string("Invalid value for ") + option + ": " + text);
    }
    return static_cast<index_t>(value);
}

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = parseIndex(argv[++i], "-n", INT_MAX);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = parseIndex(argv[++i], "-s");
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = parseIndex(argv[++i], "-i");
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* text = argv[++i];
            maxVal = std::strtod(text, &end);
            if (end == text || *end != '\0' || errno == ERANGE || !std::isfinite(maxVal)) {
                throw std::runtime_error("Invalid value for -m");
            }
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
    if (sparsity == 0) throw std::runtime_error("Sparsity must be positive");
    const uint64_t cells = static_cast<uint64_t>(numRows) * numRows;
    if (cells / sparsity > UINT32_MAX) {
        throw std::runtime_error("Matrix exceeds the 32-bit CSR nonzero index range");
    }
    const index_t nItems = static_cast<index_t>(cells / sparsity);

    MPI_Comm node;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node),
             "MPI_Comm_split_type");
    int nodeRank = 0, nodeRanks = 0;
    mpiCheck(MPI_Comm_rank(node, &nodeRank), "MPI_Comm_rank node");
    mpiCheck(MPI_Comm_size(node, &nodeRanks), "MPI_Comm_size node");
    mpiCheck(MPI_Comm_free(&node), "MPI_Comm_free node");
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) throw std::runtime_error("The MPI/OpenMP/CUDA benchmark requires a CUDA device on every rank");
    std::vector<int> devices;
    // Use all visible GPUs, even for a single MPI rank. With one rank per
    // accelerator this naturally becomes one OpenMP-managed GPU per rank.
    if (deviceCount >= nodeRanks) {
        for (int device = nodeRank; device < deviceCount; device += nodeRanks) devices.push_back(device);
    } else {
        // Also handles launchers giving each rank its own CUDA_VISIBLE_DEVICES.
        devices.push_back(nodeRank % deviceCount);
    }
    const int gpuCount = static_cast<int>(devices.size());
    std::vector<int> rankGpuCounts(ranks);
    mpiCheck(MPI_Allgather(&gpuCount, 1, MPI_INT, rankGpuCounts.data(), 1, MPI_INT, MPI_COMM_WORLD),
             "MPI_Allgather GPU counts");
    const uint64_t totalGpus = std::accumulate(rankGpuCounts.begin(), rankGpuCounts.end(), uint64_t{0});
    const int gpuThreads = std::min(gpuCount, omp_get_max_threads());

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               cells == 0 ? 100.0 : 100.0 * (1.0 - static_cast<double>(nItems) / cells));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, CUDA workers: %llu, OpenMP GPU threads on rank 0: %d\n",
               ranks, static_cast<unsigned long long>(totalGpus), gpuThreads);
        printf("Initializing data structures...\n");
    }

    std::vector<double> globalValues;
    std::vector<index_t> globalColumns, globalRows;
    std::vector<double> vector(numRows);
    std::vector<index_t> rowCuts(static_cast<size_t>(ranks) + 1);
    std::vector<index_t> nonzeroCuts(static_cast<size_t>(ranks) + 1);
    if (rank == 0) {
        globalValues.resize(nItems);
        globalColumns.resize(nItems);
        globalRows.resize(static_cast<size_t>(numRows) + 1);
        fill(vector.data(), numRows, maxVal);
        fill(globalValues.data(), nItems, maxVal);
        initRandomMatrix(globalColumns.data(), globalRows.data(), nItems, numRows);
        uint64_t prefixGpus = 0;
        for (int r = 1; r < ranks; ++r) {
            prefixGpus += rankGpuCounts[r - 1];
            rowCuts[r] = balancedCut(globalRows.data(), numRows, prefixGpus, totalGpus, rowCuts[r - 1]);
            nonzeroCuts[r] = globalRows[rowCuts[r]];
        }
        rowCuts[ranks] = numRows;
        nonzeroCuts[ranks] = nItems;
    }
    mpiCheck(MPI_Bcast(vector.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD), "MPI_Bcast vector");
    mpiCheck(MPI_Bcast(rowCuts.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD), "MPI_Bcast row cuts");
    mpiCheck(MPI_Bcast(nonzeroCuts.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD), "MPI_Bcast nonzero cuts");
    const index_t localRows = rowCuts[rank + 1] - rowCuts[rank];
    const index_t localNonzeros = nonzeroCuts[rank + 1] - nonzeroCuts[rank];
    std::vector<double> values(localNonzeros);
    std::vector<index_t> columns(localNonzeros), delimiters(static_cast<size_t>(localRows) + 1);
    scatterArray(globalValues.data(), values.data(), nonzeroCuts, MPI_DOUBLE, rank, 0);
    scatterArray(globalColumns.data(), columns.data(), nonzeroCuts, MPI_UINT32_T, rank, 1);
    scatterArray(globalRows.data(), delimiters.data(), rowCuts, MPI_UINT32_T, rank, 2);
    delimiters[localRows] = nonzeroCuts[rank + 1];
    #pragma omp parallel for schedule(static) if(localRows > 4096)
    for (int64_t row = 0; row <= localRows; ++row) delimiters[row] -= nonzeroCuts[rank];
    // Release root's full CSR before GPU allocation and distributed validation.
    std::vector<double>().swap(globalValues);
    std::vector<index_t>().swap(globalColumns);
    std::vector<index_t>().swap(globalRows);

    std::vector<double> reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        reference.resize(localRows);
        spmvCpu(values.data(), columns.data(), delimiters.data(), vector.data(), localRows, reference.data());
    }
    std::vector<index_t> gpuRows(static_cast<size_t>(gpuCount) + 1);
    for (int gpu = 1; gpu < gpuCount; ++gpu) {
        gpuRows[gpu] = balancedCut(delimiters.data(), localRows, gpu, gpuCount, gpuRows[gpu - 1]);
    }
    gpuRows[gpuCount] = localRows;
    std::vector<GpuWorker> workers(gpuCount);
    std::vector<std::string> errors(gpuCount);
    double start = 0.0, elapsed = 0.0;

    // Keep the OpenMP team alive across setup and timed execution. All CUDA
    // streams are submitted before waiting, including with OMP_NUM_THREADS=1.
    #pragma omp parallel num_threads(gpuThreads)
    {
        #pragma omp for schedule(static)
        for (int gpu = 0; gpu < gpuCount; ++gpu) {
            try {
                workers[gpu].device = devices[gpu];
                workers[gpu].firstRow = gpuRows[gpu];
                workers[gpu].numRows = gpuRows[gpu + 1] - gpuRows[gpu];
                workers[gpu].initialize(values.data(), columns.data(), delimiters.data(), vector.data(), numRows, iterations);
            } catch (const std::exception& error) {
                errors[gpu] = error.what();
            }
        }
        #pragma omp master
        {
            checkWorkers(errors);
            if (rank == 0) printf("Computing SpMV...\n");
            mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier before timing");
            start = MPI_Wtime();
        }
        #pragma omp barrier
        #pragma omp for schedule(static)
        for (int gpu = 0; gpu < gpuCount; ++gpu) {
            try {
                workers[gpu].submit(iterations);
            } catch (const std::exception& error) {
                errors[gpu] = error.what();
            }
        }
        #pragma omp for schedule(static)
        for (int gpu = 0; gpu < gpuCount; ++gpu) {
            try {
                workers[gpu].synchronize();
            } catch (const std::exception& error) {
                errors[gpu] = error.what();
            }
        }
        #pragma omp master
        {
            elapsed = MPI_Wtime() - start;
            checkWorkers(errors);
        }
    }
    double seconds = 0.0;
    mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), "MPI_Reduce timing");
    if (rank == 0) {
        const double gflops = seconds > 0 ? (2.0 * nItems * iterations) / seconds / 1e9 : 0.0;
        const double average = iterations != 0 ? seconds * 1000.0 / iterations : 0.0;
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Average time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n", average, gflops);
    }

    std::vector<double> localOutput;
    if (validate || printResults) {
        localOutput.resize(localRows);
        #pragma omp parallel for schedule(static) num_threads(gpuThreads)
        for (int gpu = 0; gpu < gpuCount; ++gpu) {
            try {
                workers[gpu].download(localOutput.data());
            } catch (const std::exception& error) {
                errors[gpu] = error.what();
            }
        }
        checkWorkers(errors);
    }
    if (printResults) {
        std::vector<double> output;
        if (rank == 0) output.resize(numRows);
        std::vector<int> counts(ranks), displacements(ranks);
        for (int r = 0; r < ranks; ++r) {
            counts[r] = static_cast<int>(rowCuts[r + 1] - rowCuts[r]);
            displacements[r] = static_cast<int>(rowCuts[r]);
        }
        mpiCheck(MPI_Gatherv(localOutput.data(), static_cast<int>(localRows), MPI_DOUBLE,
                            output.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD),
                 "MPI_Gatherv output");
        if (rank == 0) print_results(output, "OutputVector");
    }
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int valid = verifyResults(reference.data(), localOutput.data(), localRows, rowCuts[rank]) ? 1 : 0;
        int allValid = 0;
        mpiCheck(MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD), "MPI_Allreduce validation");
        if (rank == 0) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
        return allValid ? 0 : 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init_thread failed\n");
        return 1;
    }
    if (provided < MPI_THREAD_FUNNELED) abortMpi("MPI_THREAD_FUNNELED support is required");
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN), "MPI_Comm_set_errhandler");
    int rank = 0, ranks = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size");
    int result = 0;
    try {
        result = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        abortMpi(error.what());
    }
    if (MPI_Finalize() != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Finalize failed\n");
        return 1;
    }
    return result;
}
