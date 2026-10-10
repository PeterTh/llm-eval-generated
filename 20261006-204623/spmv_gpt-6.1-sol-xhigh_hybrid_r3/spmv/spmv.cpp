#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
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

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t numEntriesLeft = uint64_t(dim) * dim - (uint64_t(i) * dim + j);
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
    // Independent rows retain the original order of summation within each row.
    const int threads = std::min(omp_get_max_threads(),
                                std::max(1, static_cast<int>(dim / 128)));
#pragma omp parallel for schedule(static) num_threads(threads)
    for (index_t i = 0; i < dim; ++i) {
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
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t offset = 0) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (!std::isfinite(ref) || !std::isfinite(res)) {
            printf("Validation failed at index %u: non-finite result\n", i + offset);
            return false;
        }
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i + offset, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i + offset, ref, res, relError);
                return false;
            }
        }
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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void checkMpi(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        throw std::runtime_error(std::string(operation) + ": " + std::string(message, length));
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)
#define MPI_CHECK(operation) checkMpi((operation), #operation)

// Identify physical GPUs within a node, including per-rank CUDA_VISIBLE_DEVICES
// masks. Give each GPU to exactly one rank; OpenMP drives all GPUs owned by it.
std::vector<int> assignDevices() {
    MPI_Comm node;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node));
    int nodeRank = 0, nodeSize = 0, visible = 0;
    MPI_CHECK(MPI_Comm_rank(node, &nodeRank));
    MPI_CHECK(MPI_Comm_size(node, &nodeSize));
    const cudaError_t status = cudaGetDeviceCount(&visible);
    if (status != cudaErrorNoDevice) CUDA_CHECK(status);

    std::vector<int> counts(nodeSize), displacements(nodeSize);
    const int bytes = visible * 16;
    MPI_CHECK(MPI_Allgather(&bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, node));
    int total = 0;
    for (int r = 0; r < nodeSize; ++r) {
        displacements[r] = total;
        if (counts[r] > std::numeric_limits<int>::max() - total)
            throw std::runtime_error("Too many GPU identifiers for MPI");
        total += counts[r];
    }
    std::vector<unsigned char> mine(bytes), all(total);
    for (int d = 0; d < visible; ++d) {
        cudaDeviceProp properties;
        CUDA_CHECK(cudaGetDeviceProperties(&properties, d));
        std::memcpy(mine.data() + d * 16, properties.uuid.bytes, 16);
    }
    MPI_CHECK(MPI_Allgatherv(mine.data(), bytes, MPI_BYTE, all.data(), counts.data(),
                             displacements.data(), MPI_BYTE, node));
    std::vector<std::array<unsigned char, 16>> seen;
    std::vector<int> owned(nodeSize, 0), devices;
    for (int p = 0; p < total; p += 16) {
        std::array<unsigned char, 16> uuid;
        std::memcpy(uuid.data(), all.data() + p, 16);
        if (std::find(seen.begin(), seen.end(), uuid) != seen.end()) continue;
        seen.push_back(uuid);
        int owner = -1;
        for (int r = 0; r < nodeSize; ++r) {
            for (int q = displacements[r]; q < displacements[r] + counts[r]; q += 16) {
                if (std::memcmp(all.data() + q, uuid.data(), 16) == 0 &&
                    (owner < 0 || owned[r] < owned[owner])) owner = r;
            }
        }
        ++owned[owner];
        if (owner == nodeRank) {
            for (int d = 0; d < visible; ++d) {
                if (std::memcmp(mine.data() + d * 16, uuid.data(), 16) == 0) {
                    devices.push_back(d);
                    break;
                }
            }
        }
    }
    MPI_CHECK(MPI_Comm_free(&node));
    return devices;
}

// Nonzero work plus a small per-row cost also balances empty/very sparse rows.
// The weights are the numbers of GPUs owned by the MPI ranks.
std::vector<index_t> partitionRows(const index_t* rows, index_t count,
                                 const std::vector<int>& weights) {
    uint64_t totalWeight = 0;
    for (int weight : weights) totalWeight += weight;
    std::vector<index_t> boundaries(weights.size() + 1, 0);
    const uint64_t totalWork = uint64_t(rows[count]) + uint64_t(count) * 8;
    uint64_t weight = 0;
    for (size_t p = 0; p < weights.size(); ++p) {
        weight += weights[p];
        const uint64_t target = (totalWork / totalWeight) * weight +
                                ((totalWork % totalWeight) * weight) / totalWeight;
        index_t lo = boundaries[p], hi = count;
        while (lo < hi) {
            const index_t mid = lo + (hi - lo) / 2;
            if (uint64_t(rows[mid]) + uint64_t(mid) * 8 < target) lo = mid + 1;
            else hi = mid;
        }
        boundaries[p + 1] = lo;
    }
    boundaries.back() = count;
    return boundaries;
}

template <class T>
void broadcast(T* data, uint64_t count, MPI_Datatype type) {
    for (uint64_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min<uint64_t>(count - offset, INT_MAX));
        MPI_CHECK(MPI_Bcast(data + offset, chunk, type, 0, MPI_COMM_WORLD));
        offset += chunk;
    }
}

// Use scalable collectives normally, and chunked transfers when 32-bit MPI
// counts/displacements cannot represent a legal 32-bit CSR matrix.
template <class T>
void transferSlices(T* whole, T* local, const std::vector<index_t>& offsets,
                    MPI_Datatype type, int rank, bool gather = false) {
    const int ranks = static_cast<int>(offsets.size() - 1);
    std::vector<int> counts(ranks), displacements(ranks);
    bool collective = true;
    for (int r = 0; r < ranks; ++r) {
        const uint64_t count = uint64_t(offsets[r + 1]) - offsets[r];
        if (count > INT_MAX || offsets[r] > INT_MAX) collective = false;
        counts[r] = static_cast<int>(count);
        displacements[r] = static_cast<int>(offsets[r]);
    }
    if (collective) {
        if (gather) {
            MPI_CHECK(MPI_Gatherv(local, counts[rank], type, whole, counts.data(),
                                  displacements.data(), type, 0, MPI_COMM_WORLD));
        } else {
            MPI_CHECK(MPI_Scatterv(whole, counts.data(), displacements.data(), type,
                                   local, counts[rank], type, 0, MPI_COMM_WORLD));
        }
        return;
    }
    std::vector<MPI_Request> requests;
    if (rank == 0) {
        if (offsets[1] != 0) {
            if (gather) std::copy_n(local, offsets[1], whole);
            else std::copy_n(whole, offsets[1], local);
        }
        for (int r = 1; r < ranks; ++r) {
            for (uint64_t offset = offsets[r]; offset < offsets[r + 1];) {
                const int chunk = static_cast<int>(std::min<uint64_t>(offsets[r + 1] - offset, INT_MAX));
                MPI_Request request;
                if (gather) {
                    MPI_CHECK(MPI_Irecv(whole + offset, chunk, type, r, 0, MPI_COMM_WORLD, &request));
                } else {
                    MPI_CHECK(MPI_Isend(whole + offset, chunk, type, r, 0, MPI_COMM_WORLD, &request));
                }
                requests.push_back(request);
                offset += chunk;
            }
        }
        MPI_CHECK(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
    } else {
        const uint64_t count = uint64_t(offsets[rank + 1]) - offsets[rank];
        for (uint64_t offset = 0; offset < count;) {
            const int chunk = static_cast<int>(std::min<uint64_t>(count - offset, INT_MAX));
            if (gather) MPI_CHECK(MPI_Send(local + offset, chunk, type, 0, 0, MPI_COMM_WORLD));
            else MPI_CHECK(MPI_Recv(local + offset, chunk, type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            offset += chunk;
        }
    }
}

// Subwarp CSR kernels coalesce matrix reads. Several rows share a warp for
// short rows; a whole block reduces long rows to expose more parallelism.
template <int Width>
__global__ void spmvVector(const double* __restrict__ values,
                           const index_t* __restrict__ columns,
                           const index_t* __restrict__ rows,
                           const double* __restrict__ vector, index_t count,
                           index_t base, double* __restrict__ output) {
    const uint64_t thread = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t row = thread / Width;
    const int lane = threadIdx.x % Width;
    double sum = 0.0;
    if (row < count) {
        const uint64_t start = rows[row] - base, end = rows[row + 1] - base;
        for (uint64_t j = start + lane; j < end; j += Width)
            sum += values[j] * __ldg(vector + columns[j]);
    }
    // Every lane participates, including padding lanes in the final warp.
    for (int step = Width / 2; step > 0; step /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, step, Width);
    if (row < count && lane == 0) output[row] = sum;
}

__global__ void spmvBlock(const double* __restrict__ values,
                          const index_t* __restrict__ columns,
                          const index_t* __restrict__ rows,
                          const double* __restrict__ vector, index_t count,
                          index_t base, double* __restrict__ output) {
    __shared__ double partial[4];
    for (uint64_t row = blockIdx.x; row < count; row += gridDim.x) {
        double sum = 0.0;
        const uint64_t start = rows[row] - base, end = rows[row + 1] - base;
        for (uint64_t j = start + threadIdx.x; j < end; j += 128)
            sum += values[j] * __ldg(vector + columns[j]);
        for (int step = 16; step > 0; step /= 2)
            sum += __shfl_down_sync(0xffffffffu, sum, step);
        if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum;
        __syncthreads();
        if (threadIdx.x < 32) {
            sum = threadIdx.x < 4 ? partial[threadIdx.x] : 0.0;
            for (int step = 16; step > 0; step /= 2)
                sum += __shfl_down_sync(0xffffffffu, sum, step);
            if (threadIdx.x == 0) output[row] = sum;
        }
        __syncthreads();
    }
}

struct DevicePartition {
    int device = -1, width = 1;
    index_t first = 0, count = 0, base = 0;
    double* values = nullptr;
    double* vector = nullptr;
    double* output = nullptr;
    index_t* columns = nullptr;
    index_t* rows = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    index_t batch = 1;

    DevicePartition() = default;
    DevicePartition(const DevicePartition&) = delete;
    DevicePartition& operator=(const DevicePartition&) = delete;
    ~DevicePartition() {
        if (device < 0) return;
        cudaSetDevice(device);
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
        if (values) cudaFree(values);
        if (vector) cudaFree(vector);
        if (output) cudaFree(output);
        if (columns) cudaFree(columns);
        if (rows) cudaFree(rows);
    }

    void launch() const {
        if (count == 0) return;
        if (width == 128) {
            const unsigned blocks = static_cast<unsigned>(std::min<uint64_t>(count, 65535));
            spmvBlock<<<blocks, 128, 0, stream>>>(values, columns, rows, vector, count, base, output);
        } else {
            const unsigned blocks = static_cast<unsigned>((uint64_t(count) * width + 255) / 256);
            switch (width) {
                case 1: spmvVector<1><<<blocks, 256, 0, stream>>>(values, columns, rows, vector, count, base, output); break;
                case 2: spmvVector<2><<<blocks, 256, 0, stream>>>(values, columns, rows, vector, count, base, output); break;
                case 4: spmvVector<4><<<blocks, 256, 0, stream>>>(values, columns, rows, vector, count, base, output); break;
                case 8: spmvVector<8><<<blocks, 256, 0, stream>>>(values, columns, rows, vector, count, base, output); break;
                case 16: spmvVector<16><<<blocks, 256, 0, stream>>>(values, columns, rows, vector, count, base, output); break;
                default: spmvVector<32><<<blocks, 256, 0, stream>>>(values, columns, rows, vector, count, base, output); break;
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }

    template <class T>
    void upload(T*& target, const T* source, uint64_t elements) {
        if (elements == 0) return;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&target), elements * sizeof(T)));
        CUDA_CHECK(cudaMemcpyAsync(target, source, elements * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    void prepare(int id, index_t begin, index_t end, const std::vector<double>& hostValues,
                 const std::vector<index_t>& hostColumns, const std::vector<index_t>& hostRows,
                 const std::vector<double>& hostVector, index_t iterations) {
        device = id;
        first = begin;
        count = end - begin;
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        if (count == 0) return;
        base = hostRows[begin];
        const uint64_t nonzeros = uint64_t(hostRows[end]) - base;
        // Avoid pointer arithmetic on a null data() for zero-nonzero matrices.
        if (nonzeros) {
            upload(values, hostValues.data() + base, nonzeros);
            upload(columns, hostColumns.data() + base, nonzeros);
        }
        upload(rows, hostRows.data() + begin, uint64_t(count) + 1);
        upload(vector, hostVector.data(), hostVector.size());
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), size_t(count) * sizeof(double)));
        CUDA_CHECK(cudaMemsetAsync(output, 0, size_t(count) * sizeof(double), stream));
        cudaDeviceProp properties;
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        const double average = static_cast<double>(nonzeros) / count;
        width = average <= 2 ? 1 : average <= 4 ? 2 : average <= 8 ? 4 :
                average <= 16 ? 8 : average <= 64 ? 16 : 32;
        if (average > 256 || (average >= 32 && uint64_t(count) < uint64_t(properties.multiProcessorCount) * 8))
            width = 128;
        if (iterations) launch(); // Warm up contexts, code and caches outside timing.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (iterations == 0) return;
        batch = std::min<index_t>(iterations, 32);
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (index_t i = 0; i < batch; ++i) launch();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void compute(index_t iterations) const {
        CUDA_CHECK(cudaSetDevice(device));
        if (count == 0 || iterations == 0) return;
        for (uint64_t i = 0; i < iterations / batch; ++i)
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
        for (index_t i = 0; i < iterations % batch; ++i) launch();
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void download(double* host) const {
        if (count == 0) return;
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaMemcpyAsync(host + first, output, size_t(count) * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

// CUDA failures are returned to the main thread. Only it calls MPI under
// MPI_THREAD_FUNNELED, even on error paths.
template <class Function>
void forDevices(size_t count, Function function) {
    std::vector<std::exception_ptr> errors(count);
    const int threads = std::max(1, std::min<int>(static_cast<int>(count), omp_get_max_threads()));
#pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t d = 0; d < count; ++d) {
        try { function(d); }
        catch (...) { errors[d] = std::current_exception(); }
    }
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
}

bool parseIndex(const char* text, index_t& result) {
    if (*text == '-' || *text == '\0') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno || *end || value > std::numeric_limits<index_t>::max()) return false;
    result = static_cast<index_t>(value);
    return true;
}

int run(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // All ranks parse the same arguments; only rank zero prints diagnostics.
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-s") == 0 ||
             strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            index_t* destination = strcmp(argv[i], "-n") == 0 ? &numRows :
                                   strcmp(argv[i], "-s") == 0 ? &sparsity : &iterations;
            if (!parseIndex(argv[++i], *destination)) {
                if (rank == 0) fprintf(stderr, "Invalid integer: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            maxVal = std::strtod(argv[++i], &end);
            if (errno || end == argv[i] || *end || !std::isfinite(maxVal)) {
                if (rank == 0) fprintf(stderr, "Invalid maximum value: %s\n", argv[i]);
                return 1;
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
                printf("Unknown or incomplete option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (sparsity == 0) {
        if (rank == 0) fprintf(stderr, "Sparsity must be greater than zero\n");
        return 1;
    }
    const uint64_t entries = uint64_t(numRows) * numRows;
    const uint64_t nonzeros = entries / sparsity;
    if (nonzeros > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Matrix exceeds the 32-bit CSR nonzero limit\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nonzeros);
    const std::vector<int> devices = assignDevices();
    const int deviceCount = static_cast<int>(devices.size());
    std::vector<int> weights(ranks);
    MPI_CHECK(MPI_Allgather(&deviceCount, 1, MPI_INT, weights.data(), 1, MPI_INT, MPI_COMM_WORLD));
    int totalDevices = 0;
    for (int weight : weights) totalDevices += weight;
    if (totalDevices == 0) throw std::runtime_error("The hybrid benchmark requires CUDA GPUs");

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, entries ? 100.0 * (1.0 - static_cast<double>(nItems) / entries) : 100.0);
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, CUDA GPUs: %d\n", ranks, totalDevices);
        printf("Initializing data structures...\n");
    }

    std::vector<double> globalValues, h_vec(numRows);
    std::vector<index_t> globalColumns, globalRows;
    std::vector<index_t> rowOffsets(size_t(ranks) + 1), nonzeroOffsets(size_t(ranks) + 1);
    if (rank == 0) {
        globalValues.resize(nItems);
        globalColumns.resize(nItems);
        globalRows.resize(size_t(numRows) + 1);
        // Keep the original libc rand() stream, seed, and initialization order.
        // Explicitly restore rand()'s default seed after MPI/CUDA initialization.
        srand(1);
        fill(h_vec.data(), numRows, maxVal);
        fill(globalValues.data(), nItems, maxVal);
        initRandomMatrix(globalColumns.data(), globalRows.data(), nItems, numRows);
        rowOffsets = partitionRows(globalRows.data(), numRows, weights);
        for (int r = 0; r <= ranks; ++r) nonzeroOffsets[r] = globalRows[rowOffsets[r]];
    }
    broadcast(rowOffsets.data(), rowOffsets.size(), MPI_UINT32_T);
    broadcast(nonzeroOffsets.data(), nonzeroOffsets.size(), MPI_UINT32_T);
    broadcast(h_vec.data(), h_vec.size(), MPI_DOUBLE);

    const index_t localRows = rowOffsets[rank + 1] - rowOffsets[rank];
    const index_t localNonzeros = nonzeroOffsets[rank + 1] - nonzeroOffsets[rank];
    std::vector<double> h_val(localNonzeros), h_out(localRows, 0.0);
    std::vector<index_t> h_cols(localNonzeros), h_rowDelimiters(size_t(localRows) + 1);
    transferSlices(globalValues.data(), h_val.data(), nonzeroOffsets, MPI_DOUBLE, rank);
    transferSlices(globalColumns.data(), h_cols.data(), nonzeroOffsets, MPI_UINT32_T, rank);
    transferSlices(globalRows.data(), h_rowDelimiters.data(), rowOffsets, MPI_UINT32_T, rank);
    h_rowDelimiters[localRows] = nonzeroOffsets[rank + 1];
    const int hostThreads = std::min(omp_get_max_threads(),
                                   std::max(1, static_cast<int>(localRows / 4096)));
#pragma omp parallel for schedule(static) num_threads(hostThreads)
    for (uint64_t i = 0; i <= localRows; ++i) h_rowDelimiters[i] -= nonzeroOffsets[rank];
    std::vector<double>().swap(globalValues);
    std::vector<index_t>().swap(globalColumns);
    std::vector<index_t>().swap(globalRows);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        h_reference.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), localRows, h_reference.data());
    }

    std::vector<index_t> gpuOffsets;
    if (deviceCount) gpuOffsets = partitionRows(h_rowDelimiters.data(), localRows,
                                              std::vector<int>(deviceCount, 1));
    std::vector<DevicePartition> partitions(deviceCount);
    forDevices(partitions.size(), [&](size_t d) {
        partitions[d].prepare(devices[d], gpuOffsets[d], gpuOffsets[d + 1],
                              h_val, h_cols, h_rowDelimiters, h_vec, iterations);
    });
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    // The input vector is unchanged across iterations. There is no reason to
    // communicate, transfer the matrix, or synchronize ranks in this loop.
    forDevices(partitions.size(), [&](size_t d) { partitions[d].compute(iterations); });
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    forDevices(partitions.size(), [&](size_t d) { partitions[d].download(h_out.data()); });

    if (rank == 0) {
        const double milliseconds = seconds * 1000.0;
        const double gflops = seconds > 0 ? (2.0 * nItems * iterations) / seconds / 1e9 : 0.0;
        const double avgTime = iterations ? milliseconds / iterations : 0.0;
        printf("Computation time: %.3f ms\n", milliseconds);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Assemble the vector only when requested; validation is distributed.
    if (printResults) {
        std::vector<double> result;
        if (rank == 0) result.resize(numRows);
        transferSlices(result.data(), h_out.data(), rowOffsets, MPI_DOUBLE, rank, true);
        if (rank == 0) print_results(result, "OutputVector");
    }

    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int valid = verifyResults(h_reference.data(), h_out.data(), localRows, rowOffsets[rank]) ? 1 : 0;
        int allValid = 0;
        MPI_CHECK(MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        if (rank == 0) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
        return allValid ? 0 : 1;
    }

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    int rank = 0, ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI must support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    omp_set_dynamic(0);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int result = 1;
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return result;
}
