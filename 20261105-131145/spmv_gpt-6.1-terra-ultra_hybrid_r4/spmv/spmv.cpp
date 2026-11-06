#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <mpi.h>
#include <cuda_runtime.h>
#include <chrono>
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


// A warp cooperates on each row, coalescing matrix accesses. For very short
// rows, one thread per row avoids wasting most lanes on the reduction.
__global__ void spmvWarp(const double* val, const index_t* cols, const index_t* rows,
                         const double* vec, index_t nrows, double* out) {
    const unsigned lane = threadIdx.x & 31;
    const uint64_t first = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x / 32;
    for (uint64_t row = first; row < nrows; row += stride) {
        double sum = 0.0;
        for (uint64_t j = uint64_t(rows[row]) + lane; j < rows[row + 1]; j += 32)
            sum += val[j] * vec[cols[j]];
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) out[row] = sum;
    }
}

__global__ void spmvScalar(const double* val, const index_t* cols, const index_t* rows,
                           const double* vec, index_t nrows, double* out) {
    const uint64_t first = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t row = first; row < nrows; row += stride) {
        double sum = 0.0;
        for (index_t j = rows[row]; j < rows[row + 1]; ++j)
            sum = __dadd_rn(sum, __dmul_rn(val[j], vec[cols[j]]));
        out[row] = sum;
    }
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Chunk transfers so the CSR and vectors are not limited by MPI's int count.
template <typename T>
void transfer(T* data, uint64_t count, MPI_Datatype type, int peer, int tag, bool send) {
    while (count) {
        int chunk = static_cast<int>(std::min<uint64_t>(count, INT_MAX));
        if (send) MPI_Send(data, chunk, type, peer, tag, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, type, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

template <typename T>
void broadcast(T* data, uint64_t count, MPI_Datatype type) {
    while (count) {
        int chunk = static_cast<int>(std::min<uint64_t>(count, INT_MAX));
        MPI_Bcast(data, chunk, type, 0, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

static index_t rowStart(index_t rows, int rank, int ranks) {
    return static_cast<index_t>((uint64_t(rows) * rank) / ranks);
}

template <typename T>
T* deviceAlloc(size_t count) {
    T* ptr = nullptr;
    if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), "cudaMalloc");
    return ptr;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED support.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numRows = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) sparsity = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) maxVal = atof(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    const uint64_t matrixSize = uint64_t(numRows) * numRows;
    if (!sparsity || matrixSize / sparsity > std::numeric_limits<index_t>::max()) {
        if (!rank) fprintf(stderr, "Invalid sparsity or too many non-zero elements for 32-bit CSR indices.\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(matrixSize / sparsity);
    if (!rank) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               matrixSize ? 100.0 * (1.0 - double(nItems) / double(matrixSize)) : 0.0);
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal,
               validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, deviceCount;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");
    MPI_Comm_free(&localComm);

    std::vector<double> vec(numRows), values, reference, output;
    std::vector<index_t> cols, rows;
    if (!rank) {
        values.resize(nItems);
        cols.resize(nItems);
        rows.resize(size_t(numRows) + 1);
        // The C library's initial rand() state is equivalent to srand(1).
        srand(1);
        fill(vec.data(), numRows, maxVal);
        fill(values.data(), nItems, maxVal);
        initRandomMatrix(cols.data(), rows.data(), nItems, numRows);
        if (validate) {
            printf("Computing reference solution...\n");
            reference.resize(numRows);
            spmvCpu(values.data(), cols.data(), rows.data(), vec.data(), numRows, reference.data());
        }
    }
    broadcast(vec.data(), numRows, MPI_DOUBLE);
    const index_t firstRow = rowStart(numRows, rank, ranks);
    const index_t endRow = rowStart(numRows, rank + 1, ranks);
    const index_t localRows = endRow - firstRow;
    index_t bounds[2] = {};
    if (!rank) {
        for (int dest = 1; dest < ranks; ++dest) {
            index_t begin = rowStart(numRows, dest, ranks), end = rowStart(numRows, dest + 1, ranks);
            index_t b[2] = {rows[begin], rows[end]};
            MPI_Send(b, 2, MPI_UINT32_T, dest, 0, MPI_COMM_WORLD);
            transfer(rows.data() + begin, uint64_t(end - begin) + 1, MPI_UINT32_T, dest, 1, true);
            if (b[1] != b[0]) {
                transfer(cols.data() + b[0], b[1] - b[0], MPI_UINT32_T, dest, 2, true);
                transfer(values.data() + b[0], b[1] - b[0], MPI_DOUBLE, dest, 3, true);
            }
        }
        bounds[0] = rows[firstRow]; bounds[1] = rows[endRow];
    } else MPI_Recv(bounds, 2, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    const index_t localNnz = bounds[1] - bounds[0];
    std::vector<index_t> localOffsets(size_t(localRows) + 1), localCols(localNnz);
    std::vector<double> localValues(localNnz), localOutput(localRows);
    if (!rank) {
        std::copy_n(rows.data() + firstRow, size_t(localRows) + 1, localOffsets.data());
        if (localNnz) {
            std::copy_n(cols.data() + bounds[0], localNnz, localCols.data());
            std::copy_n(values.data() + bounds[0], localNnz, localValues.data());
        }
    } else {
        transfer(localOffsets.data(), uint64_t(localRows) + 1, MPI_UINT32_T, 0, 1, false);
        transfer(localCols.data(), localNnz, MPI_UINT32_T, 0, 2, false);
        transfer(localValues.data(), localNnz, MPI_DOUBLE, 0, 3, false);
    }
    #pragma omp parallel for schedule(static) if(localRows > 4096)
    for (uint64_t i = 0; i <= localRows; ++i) localOffsets[i] -= bounds[0];
    std::vector<double>().swap(values);
    std::vector<index_t>().swap(cols);
    std::vector<index_t>().swap(rows);

    double *dValues = deviceAlloc<double>(localNnz), *dVec = deviceAlloc<double>(numRows),
           *dOut = deviceAlloc<double>(localRows);
    index_t *dCols = deviceAlloc<index_t>(localNnz), *dRows = deviceAlloc<index_t>(size_t(localRows) + 1);
    if (numRows) cudaCheck(cudaMemcpy(dVec, vec.data(), size_t(numRows) * sizeof(double), cudaMemcpyHostToDevice), "copy vector");
    if (localNnz) {
        cudaCheck(cudaMemcpy(dValues, localValues.data(), size_t(localNnz) * sizeof(double), cudaMemcpyHostToDevice), "copy values");
        cudaCheck(cudaMemcpy(dCols, localCols.data(), size_t(localNnz) * sizeof(index_t), cudaMemcpyHostToDevice), "copy columns");
    }
    cudaCheck(cudaMemcpy(dRows, localOffsets.data(), (size_t(localRows) + 1) * sizeof(index_t), cudaMemcpyHostToDevice), "copy rows");
    if (localRows) cudaCheck(cudaMemset(dOut, 0, size_t(localRows) * sizeof(double)), "clear output");
    constexpr unsigned threads = 256;
    // External result hashes require the original per-row summation order.
    const bool scalar = printResults || uint64_t(localNnz) < uint64_t(localRows) * 8;
    const uint64_t work = scalar ? localRows : uint64_t(localRows) * 32;
    const unsigned blocks = static_cast<unsigned>(std::min<uint64_t>((work + threads - 1) / threads, 65535));
    if (localRows && iterations) {
        if (scalar) spmvScalar<<<blocks, threads>>>(dValues, dCols, dRows, dVec, localRows, dOut);
        else spmvWarp<<<blocks, threads>>>(dValues, dCols, dRows, dVec, localRows, dOut);
        cudaCheck(cudaGetLastError(), "launch warm-up SpMV");
    }
    cudaCheck(cudaDeviceSynchronize(), "synchronize setup");
    if (!rank) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (localRows) {
        for (index_t iter = 0; iter < iterations; ++iter) {
            if (scalar) spmvScalar<<<blocks, threads>>>(dValues, dCols, dRows, dVec, localRows, dOut);
            else spmvWarp<<<blocks, threads>>>(dValues, dCols, dRows, dVec, localRows, dOut);
        }
        cudaCheck(cudaGetLastError(), "launch SpMV");
        cudaCheck(cudaDeviceSynchronize(), "execute SpMV");
    }
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (validate || printResults) {
        if (localRows) cudaCheck(cudaMemcpy(localOutput.data(), dOut, size_t(localRows) * sizeof(double), cudaMemcpyDeviceToHost), "copy output");
        if (!rank) {
            output.resize(numRows);
            if (localRows) std::copy(localOutput.begin(), localOutput.end(), output.begin() + firstRow);
            for (int src = 1; src < ranks; ++src) {
                index_t begin = rowStart(numRows, src, ranks), end = rowStart(numRows, src + 1, ranks);
                if (end != begin) transfer(output.data() + begin, end - begin, MPI_DOUBLE, src, 4, false);
            }
        } else transfer(localOutput.data(), localRows, MPI_DOUBLE, 0, 4, true);
    }
    int status = 0;
    if (!rank) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        printf("Average time per iteration: %.3f ms\n", iterations ? maxElapsed * 1000.0 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", maxElapsed > 0 ? (2.0 * nItems * iterations) / maxElapsed / 1e9 : 0.0);
        if (printResults) print_results(output, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            bool valid = verifyResults(reference.data(), output.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    cudaFree(dValues); cudaFree(dCols); cudaFree(dRows); cudaFree(dVec); cudaFree(dOut);
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
