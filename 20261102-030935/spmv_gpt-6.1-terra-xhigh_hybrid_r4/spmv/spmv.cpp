#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <mpi.h>
#include <cuda_runtime.h>

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
    #pragma omp parallel for schedule(static) if(dim > 4096)
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
bool verifyResults(const double* reference, const double* result, const index_t size, const index_t firstRow) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", firstRow + i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       firstRow + i, ref, res, relError);
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


// One warp cooperates on each row; short rows use one thread per row.
__global__ void spmv_warp(const double* val, const index_t* cols,
                          const index_t* rows, const double* vec,
                          index_t nrows, double* out) {
    const unsigned lane = threadIdx.x & 31;
    const uint64_t first = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x / 32;
    for (uint64_t row = first; row < nrows; row += stride) {
        double sum = 0.0;
        for (uint64_t j = uint64_t(rows[row]) + lane; j < rows[row + 1]; j += 32)
            sum += val[j] * vec[cols[j]];
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) out[row] = sum;
    }
}

__global__ void spmv_scalar(const double* val, const index_t* cols,
                            const index_t* rows, const double* vec,
                            index_t nrows, double* out) {
    for (uint64_t row = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         row < nrows; row += uint64_t(gridDim.x) * blockDim.x) {
        double sum = 0.0;
        for (index_t j = rows[row]; j < rows[row + 1]; ++j)
            sum += val[j] * vec[cols[j]];
        out[row] = sum;
    }
}

static void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cuda_check((call), #call)

// MPI counts are int, so transfer large arrays in bounded chunks.
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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (numRows == 0 || sparsity == 0 || uint64_t(numRows) * numRows / sparsity > UINT32_MAX) {
        if (rank == 0) fprintf(stderr, "Matrix dimension and sparsity must be positive and the number of nonzeros must fit in 32 bits.\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(uint64_t(numRows) * numRows / sparsity);
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - double(nItems) / (double(numRows) * numRows)));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }
    // Preserve the original random-number sequence, then distribute its CSR rows.
    std::vector<double> vec(numRows), allValues;
    std::vector<index_t> allCols, globalRows(size_t(numRows) + 1);
    if (rank == 0) {
        allValues.resize(nItems);
        allCols.resize(nItems);
        fill(vec.data(), numRows, maxVal);
        fill(allValues.data(), nItems, maxVal);
        initRandomMatrix(allCols.data(), globalRows.data(), nItems, numRows);
    }
    // Chunk the broadcasts too, to avoid overflow of MPI's int count.
    for (uint64_t pos = 0; pos < numRows; ) {
        int count = static_cast<int>(std::min<uint64_t>(uint64_t(numRows) - pos, INT_MAX));
        MPI_Bcast(vec.data() + pos, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        pos += count;
    }
    for (uint64_t pos = 0; pos < uint64_t(numRows) + 1; ) {
        int count = static_cast<int>(std::min<uint64_t>(uint64_t(numRows) + 1 - pos, INT_MAX));
        MPI_Bcast(globalRows.data() + pos, count, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        pos += count;
    }
    // Balance both nonzeros and per-row work. The generator can leave a
    // concentration of entries at the end of the matrix.
    std::vector<index_t> boundaries(size_t(ranks) + 1);
    const uint64_t totalWork = uint64_t(nItems) + 4 * uint64_t(numRows);
    for (int r = 1; r < ranks; ++r) {
        const uint64_t target = (totalWork / ranks) * r + (totalWork % ranks) * r / ranks;
        index_t lo = 0, hi = numRows;
        while (lo < hi) {
            const index_t mid = lo + (hi - lo) / 2;
            if (uint64_t(globalRows[mid]) + 4 * uint64_t(mid) < target) lo = mid + 1;
            else hi = mid;
        }
        boundaries[r] = lo;
    }
    boundaries[ranks] = numRows;
    auto rowStart = [&boundaries](int r) -> index_t { return boundaries[r]; };
    const index_t first = rowStart(rank), last = rowStart(rank + 1), localRows = last - first;
    const index_t offset = globalRows[first], localNnz = globalRows[last] - offset;
    std::vector<index_t> rows(size_t(localRows) + 1), cols(localNnz);
    std::vector<double> values(localNnz), out(localRows, 0.0);
    #pragma omp parallel for schedule(static) if(localRows > 4096)
    for (uint64_t i = 0; i <= localRows; ++i) rows[i] = globalRows[first + i] - offset;
    if (rank == 0) {
        if (localNnz) {
            std::copy_n(allValues.data(), localNnz, values.data());
            std::copy_n(allCols.data(), localNnz, cols.data());
        }
        for (int r = 1; r < ranks; ++r) {
            index_t begin = globalRows[rowStart(r)], end = globalRows[rowStart(r + 1)];
            if (end != begin) {
                transfer(allValues.data() + begin, uint64_t(end) - begin, MPI_DOUBLE, r, 1, true);
                transfer(allCols.data() + begin, uint64_t(end) - begin, MPI_UINT32_T, r, 2, true);
            }
        }
        std::vector<double>().swap(allValues);
        std::vector<index_t>().swap(allCols);
    } else {
        transfer(values.data(), localNnz, MPI_DOUBLE, 0, 1, false);
        transfer(cols.data(), localNnz, MPI_UINT32_T, 0, 2, false);
    }
    std::vector<index_t>().swap(globalRows);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, deviceCount;
    MPI_Comm_rank(localComm, &localRank);
    CUDA(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);
    double *dValues = nullptr, *dVec = nullptr, *dOut = nullptr;
    index_t *dCols = nullptr, *dRows = nullptr;
    if (localNnz) {
        CUDA(cudaMalloc(&dValues, size_t(localNnz) * sizeof(double)));
        CUDA(cudaMalloc(&dCols, size_t(localNnz) * sizeof(index_t)));
        CUDA(cudaMemcpy(dValues, values.data(), size_t(localNnz) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(dCols, cols.data(), size_t(localNnz) * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA(cudaMalloc(&dVec, size_t(numRows) * sizeof(double)));
    CUDA(cudaMalloc(&dRows, (size_t(localRows) + 1) * sizeof(index_t)));
    CUDA(cudaMemcpy(dVec, vec.data(), size_t(numRows) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dRows, rows.data(), (size_t(localRows) + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    if (localRows) CUDA(cudaMalloc(&dOut, size_t(localRows) * sizeof(double)));
    std::vector<double> reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        reference.resize(localRows);
        spmvCpu(values.data(), cols.data(), rows.data(), vec.data(), localRows, reference.data());
    }
    const bool scalar = localRows == 0 || uint64_t(localNnz) <= uint64_t(localRows) * 4;
    const unsigned blocks = localRows ? static_cast<unsigned>(std::min<uint64_t>(65535,
        (uint64_t(localRows) + (scalar ? 255 : 7)) / (scalar ? 256 : 8))) : 0;
    auto launch = [&]() {
        if (blocks) {
            if (scalar) spmv_scalar<<<blocks, 256>>>(dValues, dCols, dRows, dVec, localRows, dOut);
            else spmv_warp<<<blocks, 256>>>(dValues, dCols, dRows, dVec, localRows, dOut);
        }
    };
    if (iterations && localRows) { launch(); CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize()); }
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) launch();
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (iterations && localRows) CUDA(cudaMemcpy(out.data(), dOut, size_t(localRows) * sizeof(double), cudaMemcpyDeviceToHost));
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        printf("Average time per iteration: %.3f ms\n", iterations ? maxElapsed * 1000.0 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", maxElapsed > 0 ? 2.0 * nItems * iterations / maxElapsed / 1e9 : 0.0);
    }
    if (printResults) {
        std::vector<double> result;
        if (rank == 0) {
            result.resize(numRows);
            std::copy(out.begin(), out.end(), result.begin());
            for (int r = 1; r < ranks; ++r)
                transfer(result.data() + rowStart(r), rowStart(r + 1) - rowStart(r), MPI_DOUBLE, r, 3, false);
            print_results(result, "OutputVector");
        } else transfer(out.data(), localRows, MPI_DOUBLE, 0, 3, true);
    }
    int valid = 1, allValid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = verifyResults(reference.data(), out.data(), localRows, first) ? 1 : 0;
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
    }
    if (dValues) CUDA(cudaFree(dValues));
    if (dCols) CUDA(cudaFree(dCols));
    if (dOut) CUDA(cudaFree(dOut));
    CUDA(cudaFree(dRows));
    CUDA(cudaFree(dVec));
    MPI_Finalize();
    return allValid ? 0 : 1;
}
