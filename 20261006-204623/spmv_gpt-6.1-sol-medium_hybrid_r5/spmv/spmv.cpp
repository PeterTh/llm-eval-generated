#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <exception>
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
    #pragma omp parallel for schedule(static)
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


namespace {
int worldRank = 0;

void mpiCheck(int status) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        fprintf(stderr, "Rank %d: MPI error: %.*s\n", worldRank, length, message);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA error: %s\n", worldRank, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Chunk transfers to support CSR arrays larger than MPI's int count limit.
template<class T>
void transfer(T* data, uint64_t count, MPI_Datatype type, int peer, bool sending) {
    constexpr uint64_t chunk = 1u << 26;
    for (uint64_t offset = 0; offset < count; offset += chunk) {
        int n = static_cast<int>(std::min(chunk, count - offset));
        if (sending)
            mpiCheck(MPI_Send(data + offset, n, type, peer, 0, MPI_COMM_WORLD));
        else
            mpiCheck(MPI_Recv(data + offset, n, type, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
}

// A power-of-two group cooperates on each row, with contiguous CSR loads.
// Grid-striding also supports very large row counts without oversized grids.
template<int Width>
__global__ void spmvKernel(const double* __restrict__ values,
                           const index_t* __restrict__ columns,
                           const index_t* __restrict__ rows,
                           const double* __restrict__ vector,
                           index_t rowCount, double* __restrict__ output) {
    const unsigned lane = threadIdx.x % Width;
    uint64_t row = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / Width;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x / Width;
    for (; row < rowCount; row += stride) {
        double sum = 0.0;
        for (uint64_t j = uint64_t(rows[row]) + lane; j < rows[row + 1]; j += Width)
            sum += values[j] * __ldg(vector + columns[j]);
        // All participating row groups are complete, even in a partial warp.
        // Restrict shuffles to active lanes and to this row's group.
        for (unsigned offset = Width / 2; offset; offset /= 2)
            sum += __shfl_down_sync(__activemask(), sum, offset, Width);
        if (lane == 0) output[row] = sum;
    }
}

struct DeviceData {
    double *values = nullptr, *vector = nullptr, *output = nullptr;
    index_t *columns = nullptr, *rows = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~DeviceData() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        cudaFree(values); cudaFree(vector); cudaFree(output);
        cudaFree(columns); cudaFree(rows);
        if (stream) cudaStreamDestroy(stream);
    }
};

template<class T>
void upload(T*& device, const std::vector<T>& host, cudaStream_t stream) {
    // Allocate at least one element for empty partitions.
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device), std::max(size_t(1), host.size()) * sizeof(T)));
    if (!host.empty())
        cudaCheck(cudaMemcpyAsync(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
}

void launch(const DeviceData& d, index_t rowCount, int width) {
    if (!rowCount) return;
    unsigned blocks = static_cast<unsigned>(std::min(uint64_t(65535),
                          (uint64_t(rowCount) * width + 255) / 256));
    switch (width) {
        case 2: spmvKernel<2><<<blocks, 256, 0, d.stream>>>(d.values,d.columns,d.rows,d.vector,rowCount,d.output); break;
        case 4: spmvKernel<4><<<blocks, 256, 0, d.stream>>>(d.values,d.columns,d.rows,d.vector,rowCount,d.output); break;
        case 8: spmvKernel<8><<<blocks, 256, 0, d.stream>>>(d.values,d.columns,d.rows,d.vector,rowCount,d.output); break;
        case 16: spmvKernel<16><<<blocks, 256, 0, d.stream>>>(d.values,d.columns,d.rows,d.vector,rowCount,d.output); break;
        default: spmvKernel<32><<<blocks, 256, 0, d.stream>>>(d.values,d.columns,d.rows,d.vector,rowCount,d.output); break;
    }
}

index_t parseIndex(const char* text) {
    char* end = nullptr;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || text[0] == '-' || end == text || *end || value > UINT32_MAX)
        throw std::invalid_argument("invalid integer argument");
    return static_cast<index_t>(value);
}

int run(int argc, char** argv, int ranks) {
    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numRows = parseIndex(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) sparsity = parseIndex(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = parseIndex(argv[++i]);
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            char* end = nullptr;
            maxVal = strtod(argv[++i], &end);
            if (end == argv[i] || *end || !std::isfinite(maxVal))
                throw std::invalid_argument("invalid maximum value");
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!worldRank) printUsage(argv[0]);
            return 0;
        } else {
            if (!worldRank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    if (!sparsity) throw std::invalid_argument("sparsity must be positive");
    const uint64_t entries = uint64_t(numRows) * numRows;
    const uint64_t nonzeros = entries / sparsity;
    if (nonzeros > UINT32_MAX || numRows == UINT32_MAX)
        throw std::invalid_argument("matrix exceeds 32-bit CSR index capacity");
    const index_t nItems = static_cast<index_t>(nonzeros);

    MPI_Comm node;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &node));
    int localRank = 0, devices = 0;
    mpiCheck(MPI_Comm_rank(node, &localRank));
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("CUDA GPU required on every MPI rank");
    // Respect CUDA_VISIBLE_DEVICES; normally use one MPI rank per visible GPU.
    cudaCheck(cudaSetDevice(localRank % devices));
    mpiCheck(MPI_Comm_free(&node));

    if (!worldRank) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               entries ? 100.0 * (1.0 - double(nItems) / entries) : 100.0);
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; OpenMP threads per rank: %d\n", ranks, omp_get_max_threads());
        printf("Initializing data structures...\n");
    }

    std::vector<double> vector(numRows), globalValues;
    std::vector<index_t> globalColumns, globalRows;
    std::vector<index_t> boundaries(size_t(ranks) + 1);
    if (!worldRank) {
        globalValues.resize(nItems); globalColumns.resize(nItems);
        globalRows.resize(size_t(numRows) + 1);
        // libc rand() is stateful. Keep the original draw order exactly, and
        // initialize only once rather than regenerating the matrix on every rank.
        fill(vector.data(), numRows, maxVal);
        fill(globalValues.data(), nItems, maxVal);
        initRandomMatrix(globalColumns.data(), globalRows.data(), nItems, numRows);
        boundaries[0] = 0; boundaries[ranks] = numRows;
        // Include per-row overhead so zero-nonzero matrices are balanced too.
        const uint64_t work = nonzeros + numRows;
        #pragma omp parallel for schedule(static)
        for (int p = 1; p < ranks; ++p) {
            const uint64_t target = (work / ranks) * p + (work % ranks) * p / ranks;
            index_t lo = 0, hi = numRows;
            while (lo < hi) {
                index_t mid = lo + (hi - lo) / 2;
                if (uint64_t(globalRows[mid]) + mid < target) lo = mid + 1;
                else hi = mid;
            }
            boundaries[p] = lo;
        }
    }
    mpiCheck(MPI_Bcast(boundaries.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
    constexpr uint64_t broadcastChunk = 1u << 26;
    for (uint64_t offset = 0; offset < numRows; offset += broadcastChunk)
        mpiCheck(MPI_Bcast(vector.data() + offset, static_cast<int>(std::min(broadcastChunk, uint64_t(numRows) - offset)),
                           MPI_DOUBLE, 0, MPI_COMM_WORLD));

    const index_t localRows = boundaries[worldRank + 1] - boundaries[worldRank];
    index_t localItems = 0;
    std::vector<index_t> rows(size_t(localRows) + 1), columns;
    std::vector<double> values;
    if (!worldRank) {
        // Transfer only each rank's CSR slice, with no replicated full matrix.
        for (int p = 1; p < ranks; ++p) {
            index_t first = globalRows[boundaries[p]];
            index_t count = globalRows[boundaries[p + 1]] - first;
            mpiCheck(MPI_Send(&count, 1, MPI_UINT32_T, p, 0, MPI_COMM_WORLD));
            transfer(globalRows.data() + boundaries[p], uint64_t(boundaries[p + 1]) - boundaries[p] + 1, MPI_UINT32_T, p, true);
            transfer(globalColumns.data() + first, count, MPI_UINT32_T, p, true);
            transfer(globalValues.data() + first, count, MPI_DOUBLE, p, true);
        }
        localItems = globalRows[localRows];
        std::copy_n(globalRows.data(), size_t(localRows) + 1, rows.data());
        values.assign(globalValues.begin(), globalValues.begin() + localItems);
        columns.assign(globalColumns.begin(), globalColumns.begin() + localItems);
        std::vector<double>().swap(globalValues);
        std::vector<index_t>().swap(globalColumns);
        std::vector<index_t>().swap(globalRows);
    } else {
        mpiCheck(MPI_Recv(&localItems, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        columns.resize(localItems); values.resize(localItems);
        transfer(rows.data(), uint64_t(localRows) + 1, MPI_UINT32_T, 0, false);
        transfer(columns.data(), localItems, MPI_UINT32_T, 0, false);
        transfer(values.data(), localItems, MPI_DOUBLE, 0, false);
    }
    const index_t base = rows[0];
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i <= localRows; ++i) rows[i] -= base;

    std::vector<double> reference, output(localRows, 0.0);
    if (validate) {
        if (!worldRank) printf("Computing reference solution...\n");
        reference.resize(localRows);
        spmvCpu(values.data(), columns.data(), rows.data(), vector.data(), localRows, reference.data());
    }
    DeviceData d;
    cudaCheck(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking));
    upload(d.values, values, d.stream); upload(d.columns, columns, d.stream);
    upload(d.rows, rows, d.stream); upload(d.vector, vector, d.stream);
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d.output), std::max(size_t(1), output.size()) * sizeof(double)));
    cudaCheck(cudaStreamSynchronize(d.stream));
    // Host inputs are no longer needed, even when validation is enabled.
    std::vector<double>().swap(values); std::vector<double>().swap(vector);
    std::vector<index_t>().swap(columns); std::vector<index_t>().swap(rows);
    const double average = localRows ? double(localItems) / localRows : 0.0;
    int width = average <= 2 ? 2 : average <= 4 ? 4 : average <= 8 ? 8 : average <= 16 ? 16 : 32;
    const index_t batch = std::min(index_t(32), iterations);
    if (iterations && localRows) {
        launch(d, localRows, width); // warm caches and exclude first-launch overhead
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaStreamSynchronize(d.stream));
        cudaCheck(cudaStreamBeginCapture(d.stream, cudaStreamCaptureModeThreadLocal));
        for (index_t i = 0; i < batch; ++i) launch(d, localRows, width);
        cudaCheck(cudaStreamEndCapture(d.stream, &d.graph));
        cudaCheck(cudaGraphInstantiate(&d.executable, d.graph, nullptr, nullptr, 0));
        cudaCheck(cudaGraphUpload(d.executable, d.stream));
        cudaCheck(cudaStreamSynchronize(d.stream));
    }
    if (!worldRank) printf("Computing SpMV...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    if (iterations && localRows) {
        for (index_t i = 0; i < iterations / batch; ++i)
            cudaCheck(cudaGraphLaunch(d.executable, d.stream));
        for (index_t i = 0; i < iterations % batch; ++i) launch(d, localRows, width);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaStreamSynchronize(d.stream));
    }
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!worldRank) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Average time per iteration: %.3f ms\n", iterations ? seconds * 1000.0 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", seconds > 0 ? 2.0 * nItems * iterations / seconds / 1e9 : 0.0);
    }
    if (iterations && localRows && (printResults || validate)) {
        cudaCheck(cudaMemcpyAsync(output.data(), d.output, output.size() * sizeof(double), cudaMemcpyDeviceToHost, d.stream));
        cudaCheck(cudaStreamSynchronize(d.stream));
    }
    if (printResults) {
        std::vector<double> globalOutput;
        if (!worldRank) {
            globalOutput.resize(numRows);
            std::copy(output.begin(), output.end(), globalOutput.begin());
            for (int p = 1; p < ranks; ++p)
                transfer(globalOutput.data() + boundaries[p], boundaries[p + 1] - boundaries[p], MPI_DOUBLE, p, false);
            print_results(globalOutput, "OutputVector");
        } else transfer(output.data(), localRows, MPI_DOUBLE, 0, true);
    }
    int valid = 1;
    if (validate) {
        int localValid = verifyResults(reference.data(), output.data(), localRows) ? 1 : 0;
        mpiCheck(MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        if (!worldRank) printf("Validating result...\nValidation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid ? 0 : 1;
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    int ranks = 1;
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int result = 1;
    try {
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED required");
        result = run(argc, argv, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", worldRank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpiCheck(MPI_Finalize());
    return result;
}
