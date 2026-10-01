#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
            uint64_t numEntriesLeft = uint64_t(dim) * dim - (uint64_t(i) * dim + j);
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
    #pragma omp parallel for schedule(dynamic, 64)
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
bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (!std::isfinite(res)) {
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


// MPI calls are confined to the main thread (MPI_THREAD_FUNNELED).
void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(operation) cudaCheck((operation), #operation)

// Chunking avoids the signed int count limit of MPI's ordinary collectives.
template<class T>
void transfer(T* data, size_t count, MPI_Datatype type, int peer, int tag, bool send) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        if (send) MPI_Send(data, chunk, type, peer, tag, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, type, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

// A power-of-two group of lanes handles each row. Small groups avoid wasting
// lanes on short rows; warp groups coalesce accesses for the usual CSR case.
template<int Width>
__global__ void spmvKernel(const double* __restrict__ values,
                           const index_t* __restrict__ columns,
                           const index_t* __restrict__ offsets,
                           const double* __restrict__ vector,
                           double* __restrict__ output, index_t rows) {
    const unsigned lane = threadIdx.x % Width;
    const size_t first = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / Width;
    const size_t stride = size_t(gridDim.x) * blockDim.x / Width;
    for (size_t row = first; row < rows; row += stride) {
        double sum = 0.0;
        for (size_t j = size_t(offsets[row]) + lane; j < offsets[row + 1]; j += Width)
            sum += values[j] * vector[columns[j]];
        // Name exactly this row's lanes, including in a partial final warp.
        const unsigned mask = (0xffffffffu >> (32 - Width))
                              << ((threadIdx.x % 32) / Width * Width);
        for (int delta = Width / 2; delta; delta /= 2)
            sum += __shfl_down_sync(mask, sum, delta, Width);
        if (lane == 0) output[row] = sum;
    }
}

// Long rows benefit from a whole block, especially when there are few rows.
__global__ void spmvLongKernel(const double* __restrict__ values,
                              const index_t* __restrict__ columns,
                              const index_t* __restrict__ offsets,
                              const double* __restrict__ vector,
                              double* __restrict__ output, index_t rows) {
    __shared__ double partial[8];
    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        double sum = 0.0;
        for (size_t j = size_t(offsets[row]) + threadIdx.x; j < offsets[row + 1]; j += 256)
            sum += values[j] * vector[columns[j]];
        for (int delta = 16; delta; delta /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, delta);
        if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum;
        __syncthreads();
        if (threadIdx.x < 32) {
            sum = threadIdx.x < 8 ? partial[threadIdx.x] : 0.0;
            for (int delta = 16; delta; delta /= 2)
                sum += __shfl_down_sync(0xffffffff, sum, delta);
            if (threadIdx.x == 0) output[row] = sum;
        }
        __syncthreads();
    }
}

struct DeviceData {
    double *values = nullptr, *vector = nullptr, *output = nullptr;
    index_t *columns = nullptr, *offsets = nullptr;
    template<class T> static void allocate(T*& ptr, size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), std::max(count, size_t(1)) * sizeof(T)));
    }
    ~DeviceData() {
        cudaFree(values); cudaFree(vector); cudaFree(output);
        cudaFree(columns); cudaFree(offsets);
    }
};

void launch(const DeviceData& d, index_t rows, int width, cudaStream_t stream) {
    if (!rows) return;
    unsigned blocks = static_cast<unsigned>(std::min(size_t(65535),
                          (size_t(rows) * width + 255) / 256));
    switch (width) {
        case 1: spmvKernel<1><<<blocks, 256, 0, stream>>>(d.values, d.columns, d.offsets, d.vector, d.output, rows); break;
        case 2: spmvKernel<2><<<blocks, 256, 0, stream>>>(d.values, d.columns, d.offsets, d.vector, d.output, rows); break;
        case 4: spmvKernel<4><<<blocks, 256, 0, stream>>>(d.values, d.columns, d.offsets, d.vector, d.output, rows); break;
        case 8: spmvKernel<8><<<blocks, 256, 0, stream>>>(d.values, d.columns, d.offsets, d.vector, d.output, rows); break;
        case 16: spmvKernel<16><<<blocks, 256, 0, stream>>>(d.values, d.columns, d.offsets, d.vector, d.output, rows); break;
        case 32: spmvKernel<32><<<blocks, 256, 0, stream>>>(d.values, d.columns, d.offsets, d.vector, d.output, rows); break;
        default: spmvLongKernel<<<blocks, 256, 0, stream>>>(d.values, d.columns, d.offsets, d.vector, d.output, rows); break;
    }
}

int run(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    auto parseIndex = [](const char* text) -> index_t {
        char* end = nullptr;
        unsigned long long value = strtoull(text, &end, 10);
        if (text[0] == '-' || end == text || *end || value == 0 ||
            value > std::numeric_limits<index_t>::max())
            throw std::runtime_error("Expected a positive 32-bit integer");
        return static_cast<index_t>(value);
    };
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numRows = parseIndex(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) sparsity = parseIndex(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = parseIndex(argv[++i]);
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            char* end = nullptr;
            maxVal = strtod(argv[++i], &end);
            if (end == argv[i] || *end || !std::isfinite(maxVal))
                throw std::runtime_error("Expected a finite maximum value");
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    const uint64_t entries = uint64_t(numRows) * numRows;
    if (entries / sparsity > std::numeric_limits<index_t>::max())
        throw std::runtime_error("Nonzero count exceeds the 32-bit CSR index capacity");
    const index_t nItems = static_cast<index_t>(entries / sparsity);

    // Respect scheduler GPU visibility, otherwise assign by node-local rank.
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("The hybrid benchmark requires a CUDA GPU on every rank");
    CUDA_CHECK(cudaSetDevice(localRank % devices));

    std::vector<double> globalValues, vector(numRows), output;
    std::vector<index_t> globalColumns, globalOffsets, boundaries(size_t(ranks) + 1);
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / entries));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal,
               validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; OpenMP threads per rank: %d; CUDA enabled\n", ranks, omp_get_max_threads());
        printf("Initializing data structures...\n");
        globalValues.resize(nItems); globalColumns.resize(nItems);
        globalOffsets.resize(size_t(numRows) + 1);
        // Keep the original libc random stream and initialization order exactly.
        srand(1); // The original program starts with libc's default seed.
        fill(vector.data(), numRows, maxVal);
        fill(globalValues.data(), nItems, maxVal);
        initRandomMatrix(globalColumns.data(), globalOffsets.data(), nItems, numRows);
        // Weight each row by nnz + 1 so empty rows are also distributed.
        boundaries[0] = 0; boundaries[ranks] = numRows;
        for (int p = 1; p < ranks; ++p) {
            uint64_t target = (uint64_t(nItems) + numRows) * p / ranks;
            size_t lo = 0, hi = numRows;
            while (lo < hi) {
                size_t mid = lo + (hi - lo) / 2;
                if (uint64_t(globalOffsets[mid]) + mid < target) lo = mid + 1;
                else hi = mid;
            }
            boundaries[p] = static_cast<index_t>(lo);
        }
    }
    MPI_Bcast(boundaries.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    for (size_t pos = 0; pos < numRows;) {
        int count = static_cast<int>(std::min(size_t(numRows) - pos, size_t(INT_MAX)));
        MPI_Bcast(vector.data() + pos, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        pos += count;
    }
    const index_t localRows = boundaries[rank + 1] - boundaries[rank];
    std::vector<index_t> offsets(size_t(localRows) + 1), columns;
    std::vector<double> values, localOutput(localRows), reference;
    if (rank == 0) {
        for (int p = 1; p < ranks; ++p) {
            index_t begin = boundaries[p], end = boundaries[p + 1];
            transfer(globalOffsets.data() + begin, size_t(end) - begin + 1, MPI_UINT32_T, p, 0, true);
            size_t count = size_t(globalOffsets[end]) - globalOffsets[begin];
            if (count) {
                transfer(globalColumns.data() + globalOffsets[begin], count, MPI_UINT32_T, p, 1, true);
                transfer(globalValues.data() + globalOffsets[begin], count, MPI_DOUBLE, p, 2, true);
            }
        }
        std::copy_n(globalOffsets.data(), size_t(localRows) + 1, offsets.data());
        globalValues.resize(offsets.back()); globalColumns.resize(offsets.back());
        values.swap(globalValues); columns.swap(globalColumns);
        std::vector<index_t>().swap(globalOffsets);
    } else {
        transfer(offsets.data(), offsets.size(), MPI_UINT32_T, 0, 0, false);
        size_t count = size_t(offsets.back()) - offsets.front();
        values.resize(count); columns.resize(count);
        transfer(columns.data(), count, MPI_UINT32_T, 0, 1, false);
        transfer(values.data(), count, MPI_DOUBLE, 0, 2, false);
    }
    const index_t base = offsets.front();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < offsets.size(); ++i) offsets[i] -= base;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        reference.resize(localRows);
        spmvCpu(values.data(), columns.data(), offsets.data(), vector.data(), localRows, reference.data());
    }

    DeviceData d;
    DeviceData::allocate(d.values, values.size()); DeviceData::allocate(d.columns, columns.size());
    DeviceData::allocate(d.offsets, offsets.size()); DeviceData::allocate(d.vector, vector.size());
    DeviceData::allocate(d.output, localRows);
    if (!values.empty()) {
        CUDA_CHECK(cudaMemcpy(d.values, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.columns, columns.data(), columns.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d.offsets, offsets.data(), offsets.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vector, vector.data(), vector.size() * sizeof(double), cudaMemcpyHostToDevice));
    double average = localRows ? double(values.size()) / localRows : 0.0;
    int width = 1;
    while (width < 32 && width < average) width *= 2;
    if (average >= 512) width = 256;
    // Data is now resident; release host CSR storage before timing.
    std::vector<double>().swap(values); std::vector<double>().swap(vector);
    std::vector<index_t>().swap(columns); std::vector<index_t>().swap(offsets);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    launch(d, localRows, width, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // Reuse a bounded graph to amortize launch overhead without O(iterations)
    // graph storage. Each node still performs one complete multiplication.
    const index_t batch = std::min(iterations, index_t(32));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (localRows) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (index_t i = 0; i < batch; ++i) launch(d, localRows, width, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (localRows) {
        for (index_t i = 0; i < iterations / batch; ++i)
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
        for (index_t i = 0; i < iterations % batch; ++i) launch(d, localRows, width, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (localRows && (validate || printResults))
        CUDA_CHECK(cudaMemcpy(localOutput.data(), d.output, size_t(localRows) * sizeof(double), cudaMemcpyDeviceToHost));
    if (executable) CUDA_CHECK(cudaGraphExecDestroy(executable));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Average time per iteration: %.3f ms\n", seconds * 1000.0 / iterations);
        printf("Performance: %.3f GFLOPS\n", 2.0 * nItems * iterations / seconds / 1e9);
    }
    if (printResults) {
        if (rank == 0) {
            output.resize(numRows);
            std::copy(localOutput.begin(), localOutput.end(), output.begin());
            for (int p = 1; p < ranks; ++p)
                transfer(output.data() + boundaries[p], size_t(boundaries[p + 1]) - boundaries[p], MPI_DOUBLE, p, 3, false);
            print_results(output, "OutputVector");
        } else transfer(localOutput.data(), localRows, MPI_DOUBLE, 0, 3, true);
    }
    int valid = 1;
    if (validate) {
        int localValid = verifyResults(reference.data(), localOutput.data(), localRows) ? 1 : 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validating result...\nValidation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank = 0, ranks = 1, result = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
