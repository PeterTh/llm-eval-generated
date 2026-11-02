#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include <climits>
#include <algorithm>
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

// A group of lanes works on each row. Short rows use smaller groups so that
// a warp can process several rows, while long rows use a full warp.
template <int LANES>
__global__ void spmvKernel(const double* val, const index_t* cols,
                           const index_t* offsets, const double* vec,
                           index_t rows, double* out) {
    const uint64_t thread = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t row = thread / LANES;
    const unsigned lane = threadIdx.x % LANES;
    double sum = 0.0;
    if (row < rows) {
        for (uint64_t j = uint64_t(offsets[row]) + lane; j < offsets[row + 1]; j += LANES)
            sum += val[j] * vec[cols[j]];
    }
    for (int delta = LANES / 2; delta; delta /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, delta, LANES);
    if (row < rows && lane == 0) out[row] = sum;
}

static void cudaCheck(cudaError_t error, const char* operation, int rank) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call, rank)

// MPI's traditional count parameter is an int. Chunk transfers so the CSR
// representation can use its full 32-bit index range.
template <typename T>
static void sendArray(const T* data, uint64_t count, MPI_Datatype type, int dest, int tag) {
    while (count) {
        int n = static_cast<int>(std::min<uint64_t>(count, INT_MAX));
        MPI_Send(data, n, type, dest, tag, MPI_COMM_WORLD);
        data += n;
        count -= n;
    }
}
template <typename T>
static void receiveArray(T* data, uint64_t count, MPI_Datatype type, int source, int tag) {
    while (count) {
        int n = static_cast<int>(std::min<uint64_t>(count, INT_MAX));
        MPI_Recv(data, n, type, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += n;
        count -= n;
    }
}
static index_t rowStart(index_t rows, int rank, int ranks) {
    return static_cast<index_t>(uint64_t(rows) * rank / ranks);
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
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            numRows = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc)
            sparsity = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
            iterations = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)
            maxVal = atof(argv[++i]);
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
    const uint64_t matrixSize = uint64_t(numRows) * numRows;
    if (!numRows || !sparsity || matrixSize / sparsity > UINT32_MAX) {
        if (rank == 0) fprintf(stderr, "Matrix dimensions and sparsity must be positive and the nonzero count must fit in 32 bits.\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(matrixSize / sparsity);
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, 100.0 * (1.0 - double(nItems) / matrixSize));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    // Match the original random number sequence exactly, then distribute CSR rows.
    std::vector<double> vec(numRows), allVal, reference;
    std::vector<index_t> allCols, allOffsets;
    if (rank == 0) {
        allVal.resize(nItems);
        allCols.resize(nItems);
        allOffsets.resize(size_t(numRows) + 1);
        // The original process starts with the C library default seed.
        srand(1);
        fill(vec.data(), numRows, maxVal);
        fill(allVal.data(), nItems, maxVal);
        initRandomMatrix(allCols.data(), allOffsets.data(), nItems, numRows);
        if (validate) {
            printf("Computing reference solution...\n");
            reference.resize(numRows);
            spmvCpu(allVal.data(), allCols.data(), allOffsets.data(), vec.data(), numRows, reference.data());
        }
    }
    for (uint64_t offset = 0; offset < numRows;) {
        int count = static_cast<int>(std::min<uint64_t>(uint64_t(numRows) - offset, INT_MAX));
        MPI_Bcast(vec.data() + offset, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        offset += count;
    }
    const index_t first = rowStart(numRows, rank, ranks);
    const index_t last = rowStart(numRows, rank + 1, ranks);
    const index_t localRows = last - first;
    std::vector<index_t> offsets(size_t(localRows) + 1), cols;
    std::vector<double> val;
    if (rank == 0) {
        for (int r = 1; r < ranks; ++r) {
            index_t begin = rowStart(numRows, r, ranks), end = rowStart(numRows, r + 1, ranks);
            index_t nzBegin = allOffsets[begin], nzEnd = allOffsets[end];
            sendArray(allOffsets.data() + begin, uint64_t(end) - begin + 1, MPI_UINT32_T, r, 10);
            if (nzEnd > nzBegin) {
                sendArray(allCols.data() + nzBegin, uint64_t(nzEnd) - nzBegin, MPI_UINT32_T, r, 11);
                sendArray(allVal.data() + nzBegin, uint64_t(nzEnd) - nzBegin, MPI_DOUBLE, r, 12);
            }
        }
        std::copy_n(allOffsets.data() + first, size_t(localRows) + 1, offsets.data());
        if (offsets.back() > offsets.front()) {
            cols.assign(allCols.begin() + offsets.front(), allCols.begin() + offsets.back());
            val.assign(allVal.begin() + offsets.front(), allVal.begin() + offsets.back());
        }
        std::vector<index_t>().swap(allCols);
        std::vector<index_t>().swap(allOffsets);
        std::vector<double>().swap(allVal);
    } else {
        receiveArray(offsets.data(), uint64_t(localRows) + 1, MPI_UINT32_T, 0, 10);
        const uint64_t localNnz = uint64_t(offsets.back()) - offsets.front();
        cols.resize(localNnz);
        val.resize(localNnz);
        receiveArray(cols.data(), localNnz, MPI_UINT32_T, 0, 11);
        receiveArray(val.data(), localNnz, MPI_DOUBLE, 0, 12);
    }
    const index_t base = offsets.front();
    #pragma omp parallel for schedule(static) if(localRows > 4096)
    for (uint64_t i = 0; i <= localRows; ++i) offsets[i] -= base;

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, deviceCount;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    double *dVal, *dVec, *dOut;
    index_t *dCols, *dOffsets;
    CUDA_CHECK(cudaMalloc(&dVal, std::max<size_t>(val.size(), 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCols, std::max<size_t>(cols.size(), 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dOffsets, offsets.size() * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dVec, vec.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dOut, std::max<size_t>(localRows, 1) * sizeof(double)));
    if (!val.empty()) {
        CUDA_CHECK(cudaMemcpy(dVal, val.data(), val.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dCols, cols.data(), cols.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(dOffsets, offsets.data(), offsets.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVec, vec.data(), vec.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dOut, 0, size_t(localRows) * sizeof(double)));

    const double avgNnz = localRows ? double(val.size()) / localRows : 0;
    const int lanes = avgNnz <= 2 ? 2 : avgNnz <= 4 ? 4 : avgNnz <= 8 ? 8 : avgNnz <= 16 ? 16 : 32;
    constexpr unsigned blockSize = 256;
    const unsigned blocks = static_cast<unsigned>((uint64_t(localRows) * lanes + blockSize - 1) / blockSize);
    auto launch = [&]() {
        switch (lanes) {
            case 2: spmvKernel<2><<<blocks, blockSize>>>(dVal, dCols, dOffsets, dVec, localRows, dOut); break;
            case 4: spmvKernel<4><<<blocks, blockSize>>>(dVal, dCols, dOffsets, dVec, localRows, dOut); break;
            case 8: spmvKernel<8><<<blocks, blockSize>>>(dVal, dCols, dOffsets, dVec, localRows, dOut); break;
            case 16: spmvKernel<16><<<blocks, blockSize>>>(dVal, dCols, dOffsets, dVec, localRows, dOut); break;
            default: spmvKernel<32><<<blocks, blockSize>>>(dVal, dCols, dOffsets, dVec, localRows, dOut);
        }
    };
    if (rank == 0) printf("Computing SpMV...\n");
    // Load the kernel and warm the device before measuring steady-state SpMV.
    if (iterations && localRows) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations && localRows; ++iter) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        printf("Average time per iteration: %.3f ms\n", iterations ? maxElapsed * 1000.0 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", maxElapsed > 0 ? 2.0 * nItems * iterations / maxElapsed / 1e9 : 0.0);
    }

    std::vector<double> out(localRows), allOut;
    if (localRows) CUDA_CHECK(cudaMemcpy(out.data(), dOut, size_t(localRows) * sizeof(double), cudaMemcpyDeviceToHost));
    if (printResults || validate) {
        if (rank == 0) {
            allOut.resize(numRows);
            std::copy(out.begin(), out.end(), allOut.begin() + first);
            for (int r = 1; r < ranks; ++r) {
                index_t begin = rowStart(numRows, r, ranks), end = rowStart(numRows, r + 1, ranks);
                receiveArray(allOut.data() + begin, uint64_t(end) - begin, MPI_DOUBLE, r, 20);
            }
        } else sendArray(out.data(), localRows, MPI_DOUBLE, 0, 20);
    }
    int valid = 1;
    if (rank == 0) {
        if (printResults) print_results(allOut, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            valid = verifyResults(reference.data(), allOut.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(dVal));
    CUDA_CHECK(cudaFree(dCols));
    CUDA_CHECK(cudaFree(dOffsets));
    CUDA_CHECK(cudaFree(dVec));
    CUDA_CHECK(cudaFree(dOut));
    MPI_Finalize();
    return valid ? 0 : 1;
}
