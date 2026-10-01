#include <algorithm>
#include <cerrno>
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
    #pragma omp parallel for schedule(static)
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


// MPI is called only by the main thread (MPI_THREAD_FUNNELED).
void mpiCheck(int status) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        fprintf(stderr, "MPI error: %.*s\n", length, message);
        MPI_Abort(MPI_COMM_WORLD, status);
    }
}

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: CUDA error: %s\n", rank, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Chunk transfers to support CSR arrays exceeding MPI's signed-int count limit.
template<class T>
void transfer(T* data, size_t count, MPI_Datatype type, int peer, bool send) {
    while (count) {
        const int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        if (send)
            mpiCheck(MPI_Send(data, chunk, type, peer, 0, MPI_COMM_WORLD));
        else
            mpiCheck(MPI_Recv(data, chunk, type, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        data += chunk;
        count -= chunk;
    }
}

template<class T>
struct DeviceArray {
    T* data = nullptr;
    explicit DeviceArray(size_t count) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), std::max(count, size_t(1)) * sizeof(T)));
    }
    ~DeviceArray() { cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    void upload(const std::vector<T>& source) {
        if (!source.empty())
            cudaCheck(cudaMemcpy(data, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};

// Short rows use subwarps; long rows use a whole warp. Each group owns a row,
// so there are no atomics, and adjacent lanes load contiguous CSR entries.
template<int Width>
__global__ void spmvKernel(const double* __restrict__ values,
                           const index_t* __restrict__ columns,
                           const index_t* __restrict__ offsets,
                           const double* __restrict__ vector,
                           double* __restrict__ output, size_t rows) {
    const unsigned lane = threadIdx.x % Width;
    const size_t first = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / Width;
    const size_t stride = size_t(gridDim.x) * blockDim.x / Width;
    for (size_t row = first; row < rows; row += stride) {
        double sum = 0.0;
        const size_t end = offsets[row + 1];
        for (size_t j = size_t(offsets[row]) + lane; j < end; j += Width)
            sum += values[j] * vector[columns[j]];
        const unsigned mask = (0xffffffffu >> (32 - Width)) << ((threadIdx.x % 32) / Width * Width);
        for (int delta = Width / 2; delta; delta /= 2)
            sum += __shfl_down_sync(mask, sum, delta, Width);
        if (lane == 0) output[row] = sum;
    }
}

// Dense/very long rows expose more parallelism with a full block per row.
__global__ void spmvLongKernel(const double* __restrict__ values,
                               const index_t* __restrict__ columns,
                               const index_t* __restrict__ offsets,
                               const double* __restrict__ vector,
                               double* __restrict__ output, size_t rows) {
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

index_t parseIndex(const char* text) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno || text[0] == '-' || end == text || *end || value > UINT32_MAX)
        throw std::runtime_error("Invalid nonnegative integer argument");
    return static_cast<index_t>(value);
}

int run(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numRows = parseIndex(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) sparsity = parseIndex(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = parseIndex(argv[++i]);
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            char* end = nullptr;
            maxVal = strtod(argv[++i], &end);
            if (end == argv[i] || *end || !std::isfinite(maxVal))
                throw std::runtime_error("Invalid maximum value");
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) { printf("Unknown or incomplete option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    if (!numRows || !sparsity) throw std::runtime_error("Matrix size and sparsity must be positive");
    const uint64_t entries = uint64_t(numRows) * numRows;
    if (entries / sparsity > UINT32_MAX)
        throw std::runtime_error("Nonzero count exceeds the 32-bit CSR index capacity");
    const index_t nItems = static_cast<index_t>(entries / sparsity);

    // Honor scheduler-provided CUDA_VISIBLE_DEVICES; otherwise map local MPI
    // ranks round-robin over visible GPUs. Launch one rank per GPU for throughput.
    MPI_Comm localComm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm));
    int localRank = 0, devices = 0;
    mpiCheck(MPI_Comm_rank(localComm, &localRank));
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("A CUDA GPU is required on every MPI rank");
    cudaCheck(cudaSetDevice(localRank % devices));
    mpiCheck(MPI_Comm_free(&localComm));
    cudaDeviceProp properties;
    cudaCheck(cudaGetDeviceProperties(&properties, localRank % devices));

    std::vector<double> allValues, vector(numRows);
    std::vector<index_t> allColumns, allOffsets;
    std::vector<index_t> boundaries(size_t(ranks) + 1);
    if (!rank) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\n", numRows, numRows, sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, 100.0 * (1.0 - double(nItems) / entries));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; OpenMP threads per rank: %d\n", ranks, omp_get_max_threads());
        printf("Initializing data structures...\n");
        allValues.resize(nItems);
        allColumns.resize(nItems);
        allOffsets.resize(size_t(numRows) + 1);
        // Preserve the original libc RNG stream and ordering, including the
        // seed reset inside initRandomMatrix. Parallel rand() changes the input.
        fill(vector.data(), numRows, maxVal);
        fill(allValues.data(), nItems, maxVal);
        initRandomMatrix(allColumns.data(), allOffsets.data(), nItems, numRows);
        // Balance both nonzero work and row overhead, including empty matrices.
        const uint64_t totalWork = uint64_t(nItems) + numRows;
        for (int p = 1; p < ranks; ++p) {
            const uint64_t target = totalWork * uint64_t(p) / ranks;
            size_t low = boundaries[p - 1], high = numRows;
            while (low < high) {
                const size_t mid = low + (high - low) / 2;
                if (uint64_t(allOffsets[mid]) + mid < target) low = mid + 1;
                else high = mid;
            }
            boundaries[p] = static_cast<index_t>(low);
        }
        boundaries[ranks] = numRows;
    }
    mpiCheck(MPI_Bcast(boundaries.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
    for (size_t offset = 0; offset < numRows;) {
        const int count = static_cast<int>(std::min(size_t(numRows) - offset, size_t(INT_MAX)));
        mpiCheck(MPI_Bcast(vector.data() + offset, count, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        offset += count;
    }
    const size_t rows = size_t(boundaries[rank + 1]) - boundaries[rank];
    std::vector<index_t> offsets(rows + 1), columns;
    std::vector<double> values, output(rows, 0.0);
    if (!rank) {
        for (int p = 1; p < ranks; ++p) {
            const size_t first = boundaries[p], count = size_t(boundaries[p + 1]) - first;
            const size_t start = allOffsets[first], nonzeros = size_t(allOffsets[first + count]) - start;
            transfer(allOffsets.data() + first, count + 1, MPI_UINT32_T, p, true);
            if (nonzeros) {
                transfer(allColumns.data() + start, nonzeros, MPI_UINT32_T, p, true);
                transfer(allValues.data() + start, nonzeros, MPI_DOUBLE, p, true);
            }
        }
        offsets.assign(allOffsets.begin(), allOffsets.begin() + rows + 1);
        columns.assign(allColumns.begin(), allColumns.begin() + offsets.back());
        values.assign(allValues.begin(), allValues.begin() + offsets.back());
        std::vector<index_t>().swap(allColumns);
        std::vector<index_t>().swap(allOffsets);
        std::vector<double>().swap(allValues);
    } else {
        transfer(offsets.data(), rows + 1, MPI_UINT32_T, 0, false);
        const size_t nonzeros = size_t(offsets.back()) - offsets.front();
        columns.resize(nonzeros);
        values.resize(nonzeros);
        transfer(columns.data(), nonzeros, MPI_UINT32_T, 0, false);
        transfer(values.data(), nonzeros, MPI_DOUBLE, 0, false);
    }
    const index_t base = offsets.front();
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(offsets.size()); ++i) offsets[i] -= base;

    DeviceArray<double> dValues(values.size()), dVector(vector.size()), dOutput(rows);
    DeviceArray<index_t> dColumns(columns.size()), dOffsets(offsets.size());
    dValues.upload(values);
    dVector.upload(vector);
    dColumns.upload(columns);
    dOffsets.upload(offsets);
    std::vector<double> reference;
    if (validate) {
        if (!rank) printf("Computing reference solution...\n");
        reference.resize(rows);
        spmvCpu(values.data(), columns.data(), offsets.data(), vector.data(), static_cast<index_t>(rows), reference.data());
    }
    // Inputs stay resident throughout all iterations; x is unchanged in this
    // benchmark, so neither halo exchange nor result gathering is needed here.
    const size_t average = rows ? values.size() / rows : 0;
    const int width = average <= 4 ? 4 : average <= 8 ? 8 : average <= 16 ? 16 : 32;
    const int blocks = static_cast<int>(std::min((rows * width + 255) / 256,
                                                size_t(properties.multiProcessorCount) * 32));
    auto launch = [&]() {
        if (!rows) return;
        if (average >= 512) {
            const int longBlocks = static_cast<int>(std::min(rows, size_t(properties.multiProcessorCount) * 32));
            spmvLongKernel<<<longBlocks, 256>>>(dValues.data, dColumns.data, dOffsets.data, dVector.data, dOutput.data, rows);
        } else if (width == 4)
            spmvKernel<4><<<blocks, 256>>>(dValues.data, dColumns.data, dOffsets.data, dVector.data, dOutput.data, rows);
        else if (width == 8)
            spmvKernel<8><<<blocks, 256>>>(dValues.data, dColumns.data, dOffsets.data, dVector.data, dOutput.data, rows);
        else if (width == 16)
            spmvKernel<16><<<blocks, 256>>>(dValues.data, dColumns.data, dOffsets.data, dVector.data, dOutput.data, rows);
        else
            spmvKernel<32><<<blocks, 256>>>(dValues.data, dColumns.data, dOffsets.data, dVector.data, dOutput.data, rows);
    };
    if (iterations) launch(); // Warm up context, kernel, and caches outside timing.
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    if (!rank) printf("Computing SpMV...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) launch();
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (iterations && rows)
        cudaCheck(cudaMemcpy(output.data(), dOutput.data, rows * sizeof(double), cudaMemcpyDeviceToHost));
    if (!rank) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Average time per iteration: %.3f ms\n", iterations ? seconds * 1000.0 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", seconds > 0 ? 2.0 * nItems * iterations / seconds / 1e9 : 0.0);
    }
    if (printResults) {
        if (!rank) {
            std::vector<double> globalOutput(numRows);
            std::copy(output.begin(), output.end(), globalOutput.begin());
            for (int p = 1; p < ranks; ++p)
                transfer(globalOutput.data() + boundaries[p], size_t(boundaries[p + 1]) - boundaries[p], MPI_DOUBLE, p, false);
            print_results(globalOutput, "OutputVector");
        } else transfer(output.data(), rows, MPI_DOUBLE, 0, true);
    }
    int valid = 1;
    if (validate) {
        int localValid = verifyResults(reference.data(), output.data(), static_cast<index_t>(rows)) ? 1 : 0;
        mpiCheck(MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        if (!rank) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank = 0, ranks = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int status = 1;
    try {
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED is required");
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpiCheck(MPI_Finalize());
    return status;
}
