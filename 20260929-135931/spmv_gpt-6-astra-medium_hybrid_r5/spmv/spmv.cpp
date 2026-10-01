#include <algorithm>
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
        if (!std::isfinite(res)) return false;
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


// A rank owns contiguous whole rows. MPI is only called by the main thread;
// OpenMP handles host row work and CUDA handles all timed multiplications.
static void cudaCheck(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, expression, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t n) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), std::max(size_t(1), n) * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Subwarps reduce short rows without wasting a full warp on each one. Long
// rows use a whole warp; grid-striding keeps the launch size bounded.
template<int Width>
__global__ void spmvKernel(const double* __restrict__ values,
                           const index_t* __restrict__ columns,
                           const index_t* __restrict__ offsets,
                           const double* __restrict__ vector,
                           double* __restrict__ output, index_t rows) {
    const unsigned lane = threadIdx.x % Width;
    const size_t first = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / Width;
    const size_t stride = size_t(gridDim.x) * blockDim.x / Width;
    // All lanes execute shuffles, including subwarps beyond the last row.
    for (size_t base = first - (threadIdx.x % 32) / Width;
         base < rows; base += stride) {
        const size_t row = base + (threadIdx.x % 32) / Width;
        double sum = 0.0;
        if (row < rows) {
            const size_t end = offsets[row + 1];
            for (size_t j = size_t(offsets[row]) + lane; j < end; j += Width)
                sum += values[j] * vector[columns[j]];
        }
        for (int delta = Width / 2; delta; delta /= 2)
            sum += __shfl_down_sync(0xffffffffu, sum, delta, Width);
        if (lane == 0 && row < rows) output[row] = sum;
    }
}

template<int Width>
void launchSpmv(const double* values, const index_t* columns, const index_t* offsets,
                const double* vector, double* output, index_t rows,
                int maxBlocks, cudaStream_t stream) {
    if (!rows) return;
    const int blocks = static_cast<int>(std::min<size_t>(maxBlocks,
        (size_t(rows) * Width + 255) / 256));
    spmvKernel<Width><<<blocks, 256, 0, stream>>>(values, columns, offsets, vector, output, rows);
}

// Chunking avoids the signed-int count limit of MPI-3, including for a CSR
// matrix whose nonzero count approaches the uint32_t index limit.
template<class T>
void transfer(T* data, size_t count, MPI_Datatype type, int peer, bool send) {
    constexpr size_t chunkSize = 1 << 26;
    for (size_t offset = 0; offset < count; offset += chunkSize) {
        const int chunk = static_cast<int>(std::min(chunkSize, count - offset));
        if (send) MPI_Send(data + offset, chunk, type, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data + offset, chunk, type, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

static index_t parseCount(const char* value) {
    char* end = nullptr;
    const unsigned long long n = strtoull(value, &end, 10);
    if (!*value || *value == '-' || *end || n > std::numeric_limits<index_t>::max())
        throw std::runtime_error("Invalid integer argument");
    return static_cast<index_t>(n);
}

int run(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numRows = parseCount(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) sparsity = parseCount(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = parseCount(argv[++i]);
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            char* end = nullptr;
            maxVal = strtod(argv[++i], &end);
            if (!*argv[i] || *end || !std::isfinite(maxVal))
                throw std::runtime_error("Invalid maximum value");
        }
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    if (!numRows || !sparsity || !iterations)
        throw std::runtime_error("Matrix size, sparsity and iterations must be positive");
    const uint64_t entries = uint64_t(numRows) * numRows;
    if (entries / sparsity > std::numeric_limits<index_t>::max())
        throw std::runtime_error("Nonzero count exceeds the CSR uint32_t index range");
    const index_t nItems = static_cast<index_t>(entries / sparsity);

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank = 0;
    MPI_Comm_rank(node, &localRank);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("Every MPI rank requires a CUDA device");
    // Also works with a scheduler exposing a single assigned GPU to each rank.
    const int device = localRank % devices;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    MPI_Comm_free(&node);

    if (!rank) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / entries));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal,
               validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads per rank: %d, CUDA device: %s\n",
               ranks, omp_get_max_threads(), properties.name);
        printf("Initializing data structures...\n");
    }

    std::vector<double> globalValues, vector(numRows);
    std::vector<index_t> globalColumns, globalOffsets, boundaries(size_t(ranks) + 1);
    if (!rank) {
        globalValues.resize(nItems);
        globalColumns.resize(nItems);
        globalOffsets.resize(size_t(numRows) + 1);
        // Keep the exact original rand() sequence; parallel rand() would change
        // the problem and introduce a contended global RNG lock.
        srand(1);
        fill(vector.data(), numRows, maxVal);
        fill(globalValues.data(), nItems, maxVal);
        initRandomMatrix(globalColumns.data(), globalOffsets.data(), nItems, numRows);
        boundaries[0] = 0;
        boundaries[ranks] = numRows;
        for (int p = 1; p < ranks; ++p) {
            if (nItems) {
                const index_t target = static_cast<index_t>(uint64_t(nItems) * p / ranks);
                boundaries[p] = static_cast<index_t>(std::lower_bound(
                    globalOffsets.begin(), globalOffsets.end(), target) - globalOffsets.begin());
            } else boundaries[p] = static_cast<index_t>(uint64_t(numRows) * p / ranks);
        }
    }
    MPI_Bcast(boundaries.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    for (size_t offset = 0; offset < numRows; offset += size_t(1) << 26)
        MPI_Bcast(vector.data() + offset, static_cast<int>(std::min<size_t>(
            size_t(1) << 26, size_t(numRows) - offset)), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const index_t rows = boundaries[rank + 1] - boundaries[rank];
    std::vector<index_t> offsets(size_t(rows) + 1), columns;
    std::vector<double> values;
    if (!rank) {
        // Distribute only each rank's rows and nonzeros, rather than replicating
        // the matrix. All messages are outside the steady-state timing.
        for (int p = 1; p < ranks; ++p) {
            const index_t start = globalOffsets[boundaries[p]];
            const index_t count = globalOffsets[boundaries[p + 1]] - start;
            transfer(globalOffsets.data() + boundaries[p],
                     size_t(boundaries[p + 1]) - boundaries[p] + 1, MPI_UINT32_T, p, true);
            if (count) {
                transfer(globalColumns.data() + start, count, MPI_UINT32_T, p, true);
                transfer(globalValues.data() + start, count, MPI_DOUBLE, p, true);
            }
        }
        std::copy_n(globalOffsets.begin(), size_t(rows) + 1, offsets.begin());
        columns.assign(globalColumns.begin(), globalColumns.begin() + offsets.back());
        values.assign(globalValues.begin(), globalValues.begin() + offsets.back());
        std::vector<index_t>().swap(globalColumns);
        std::vector<index_t>().swap(globalOffsets);
        std::vector<double>().swap(globalValues);
    } else {
        transfer(offsets.data(), offsets.size(), MPI_UINT32_T, 0, false);
        const index_t count = offsets.back() - offsets.front();
        columns.resize(count);
        values.resize(count);
        transfer(columns.data(), count, MPI_UINT32_T, 0, false);
        transfer(values.data(), count, MPI_DOUBLE, 0, false);
    }
    const index_t base = offsets.front();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < offsets.size(); ++i) offsets[i] -= base;

    std::vector<double> output(rows), reference;
    if (validate) {
        if (!rank) printf("Computing reference solution...\n");
        reference.resize(rows);
        spmvCpu(values.data(), columns.data(), offsets.data(), vector.data(), rows, reference.data());
    }

    DeviceBuffer<double> dValues(values.size()), dVector(numRows), dOutput(rows);
    DeviceBuffer<index_t> dColumns(columns.size()), dOffsets(offsets.size());
    if (!values.empty()) {
        CUDA_CHECK(cudaMemcpy(dValues.data, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dColumns.data, columns.data(), columns.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(dOffsets.data, offsets.data(), offsets.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVector.data, vector.data(), vector.size() * sizeof(double), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const size_t average = rows ? values.size() / rows : 0;
    auto launch = [&]() {
        const int blocks = properties.multiProcessorCount * 32;
        if (average <= 2) launchSpmv<1>(dValues.data, dColumns.data, dOffsets.data, dVector.data, dOutput.data, rows, blocks, stream);
        else if (average <= 8) launchSpmv<4>(dValues.data, dColumns.data, dOffsets.data, dVector.data, dOutput.data, rows, blocks, stream);
        else if (average <= 16) launchSpmv<8>(dValues.data, dColumns.data, dOffsets.data, dVector.data, dOutput.data, rows, blocks, stream);
        else if (average <= 32) launchSpmv<16>(dValues.data, dColumns.data, dOffsets.data, dVector.data, dOutput.data, rows, blocks, stream);
        else launchSpmv<32>(dValues.data, dColumns.data, dOffsets.data, dVector.data, dOutput.data, rows, blocks, stream);
    };
    launch(); // Warm up context, caches, and kernel before timing.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Reuse a bounded CUDA graph to amortize host launch overhead for the
    // unchanged matrix/vector. Each graph node still performs one full SpMV.
    const index_t batch = std::min<index_t>(iterations, 64);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (rows) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (index_t i = 0; i < batch; ++i) launch();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    if (!rank) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (rows) {
        for (index_t i = 0; i < iterations / batch; ++i)
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
        for (index_t i = 0; i < iterations % batch; ++i) launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rows) CUDA_CHECK(cudaMemcpy(output.data(), dOutput.data, size_t(rows) * sizeof(double), cudaMemcpyDeviceToHost));
    if (executable) CUDA_CHECK(cudaGraphExecDestroy(executable));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (!rank) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Average time per iteration: %.3f ms\n", seconds * 1000.0 / iterations);
        printf("Performance: %.3f GFLOPS\n", 2.0 * nItems * iterations / seconds / 1e9);
    }
    if (printResults) {
        if (!rank) {
            std::vector<double> fullOutput(numRows);
            std::copy(output.begin(), output.end(), fullOutput.begin());
            for (int p = 1; p < ranks; ++p)
                transfer(fullOutput.data() + boundaries[p],
                         boundaries[p + 1] - boundaries[p], MPI_DOUBLE, p, false);
            print_results(fullOutput, "OutputVector");
        } else transfer(output.data(), rows, MPI_DOUBLE, 0, true);
    }
    int valid = !validate || verifyResults(reference.data(), output.data(), rows);
    int allValid = 0;
    MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (validate && !rank) printf("Validating result...\nValidation: %s\n", allValid ? "PASSED" : "FAILED");
    return allValid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        if (provided < MPI_THREAD_FUNNELED)
            throw std::runtime_error("MPI_THREAD_FUNNELED is required");
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
