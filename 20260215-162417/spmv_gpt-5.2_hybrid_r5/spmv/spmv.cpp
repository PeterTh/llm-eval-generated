#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

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
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
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

static inline void cudaCheck(cudaError_t e, const char* call, const char* file, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s failed: %s\n", file, line, call, cudaGetErrorString(e));
        fflush(stderr);
        std::abort();
    }
}
#define CUDA_CHECK(x) cudaCheck((x), #x, __FILE__, __LINE__)

static __inline__ __device__ double warpReduceSum(double v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

static __global__ void spmvCsrVectorKernel(const index_t numRows, const index_t* __restrict__ rowPtr,
                                          const index_t* __restrict__ colIdx, const double* __restrict__ val,
                                          const double* __restrict__ x, double* __restrict__ y) {
    constexpr int WARP = 32;
    const int lane = threadIdx.x & (WARP - 1);
    const int warpId = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;
    const index_t row = static_cast<index_t>(blockIdx.x) * static_cast<index_t>(warpsPerBlock) +
                        static_cast<index_t>(warpId);

    if (row >= numRows) return;

    const index_t start = rowPtr[row];
    const index_t end = rowPtr[row + 1];
    double sum = 0.0;
    for (index_t jj = start + static_cast<index_t>(lane); jj < end; jj += static_cast<index_t>(WARP)) {
        const index_t c = colIdx[jj];
        sum += val[jj] * x[c];
    }
    sum = warpReduceSum(sum);
    if (lane == 0) y[row] = sum;
}

static void selectCudaDeviceForRank(int worldRank) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        std::abort();
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    const int dev = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(dev));
}

static inline int checkedCountToInt(size_t n, const char* what) {
    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "%s exceeds MPI int count limit\n", what);
        std::abort();
    }
    return static_cast<int>(n);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            maxVal = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (worldRank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d\n", worldSize);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    selectCudaDeviceForRank(worldRank);

    // Root initializes full problem to preserve original RNG semantics
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out;  // only rank 0 allocates full output

    if (worldRank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_out.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast vector and row delimiters to all ranks
    MPI_Bcast(h_vec.data(), checkedCountToInt(numRows, "vec"), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), checkedCountToInt(static_cast<size_t>(numRows) + 1, "rowDelimiters"),
              MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Row-wise partition (contiguous)
    const index_t rowStart = static_cast<index_t>((static_cast<uint64_t>(numRows) * worldRank) / worldSize);
    const index_t rowEnd = static_cast<index_t>((static_cast<uint64_t>(numRows) * (worldRank + 1)) / worldSize);
    const index_t localRows = rowEnd - rowStart;

    const index_t nnzStart = h_rowDelimiters[rowStart];
    const index_t nnzEnd = h_rowDelimiters[rowEnd];
    const index_t localNnz = nnzEnd - nnzStart;

    // Root computes scatter metadata
    std::vector<int> sendcountsNnz, displsNnz;
    std::vector<int> recvcountsRows, displsRows;
    if (worldRank == 0) {
        sendcountsNnz.resize(worldSize);
        displsNnz.resize(worldSize);
        recvcountsRows.resize(worldSize);
        displsRows.resize(worldSize);
        for (int r = 0; r < worldSize; ++r) {
            const index_t rs = static_cast<index_t>((static_cast<uint64_t>(numRows) * r) / worldSize);
            const index_t re = static_cast<index_t>((static_cast<uint64_t>(numRows) * (r + 1)) / worldSize);
            const index_t ns = h_rowDelimiters[rs];
            const index_t ne = h_rowDelimiters[re];
            const index_t ln = ne - ns;
            sendcountsNnz[r] = checkedCountToInt(ln, "nnz sendcount");
            displsNnz[r] = checkedCountToInt(ns, "nnz displacement");
            recvcountsRows[r] = checkedCountToInt(re - rs, "row recvcount");
            displsRows[r] = checkedCountToInt(rs, "row displacement");
        }
    }

    // Scatter CSR values/cols for local nnz range
    std::vector<double> h_valLocal(localNnz);
    std::vector<index_t> h_colsLocal(localNnz);

    MPI_Scatterv(worldRank == 0 ? h_val.data() : nullptr,
                 worldRank == 0 ? sendcountsNnz.data() : nullptr,
                 worldRank == 0 ? displsNnz.data() : nullptr, MPI_DOUBLE,
                 h_valLocal.data(), checkedCountToInt(localNnz, "localNnz"), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(worldRank == 0 ? h_cols.data() : nullptr,
                 worldRank == 0 ? sendcountsNnz.data() : nullptr,
                 worldRank == 0 ? displsNnz.data() : nullptr, MPI_UINT32_T,
                 h_colsLocal.data(), checkedCountToInt(localNnz, "localNnz"), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Build local row pointers (rebased to 0)
    std::vector<index_t> h_rowPtrLocal(localRows + 1);
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= localRows; ++i) {
        h_rowPtrLocal[i] = h_rowDelimiters[rowStart + i] - nnzStart;
    }

    // Reference solution (rank 0 only)
    std::vector<double> h_reference;
    if (validate && worldRank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows, h_reference.data());
    }

    // Device allocations
    index_t* d_rowPtr = nullptr;
    index_t* d_cols = nullptr;
    double* d_val = nullptr;
    double* d_x = nullptr;
    double* d_y = nullptr;

    const auto allocCount = [](size_t n) { return n ? n : size_t{1}; };

    CUDA_CHECK(cudaMalloc(&d_rowPtr, allocCount(static_cast<size_t>(localRows) + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_cols, allocCount(static_cast<size_t>(localNnz)) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_val, allocCount(static_cast<size_t>(localNnz)) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, allocCount(static_cast<size_t>(numRows)) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, allocCount(static_cast<size_t>(localRows)) * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_rowPtr, h_rowPtrLocal.data(), (static_cast<size_t>(localRows) + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    if (localNnz) {
        CUDA_CHECK(cudaMemcpy(d_cols, h_colsLocal.data(), static_cast<size_t>(localNnz) * sizeof(index_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_val, h_valLocal.data(), static_cast<size_t>(localNnz) * sizeof(double),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_x, h_vec.data(), static_cast<size_t>(numRows) * sizeof(double), cudaMemcpyHostToDevice));

    if (worldRank == 0) printf("Computing SpMV...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    if (localRows) {
        constexpr int threads = 256;
        const int warpsPerBlock = threads / 32;
        const int blocks = static_cast<int>((static_cast<size_t>(localRows) + warpsPerBlock - 1) / warpsPerBlock);
        for (index_t iter = 0; iter < iterations; ++iter) {
            spmvCsrVectorKernel<<<blocks, threads>>>(localRows, d_rowPtr, d_cols, d_val, d_x, d_y);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        // still participate in timing/barriers
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double localSeconds = t1 - t0;
    double maxSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy local result back
    std::vector<double> h_outLocal(localRows);
    if (localRows) {
        CUDA_CHECK(cudaMemcpy(h_outLocal.data(), d_y, static_cast<size_t>(localRows) * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    // Gather output to rank 0 for printing/validation
    if (worldRank == 0) {
        MPI_Gatherv(h_outLocal.data(), checkedCountToInt(localRows, "localRows"), MPI_DOUBLE,
                    h_out.data(), recvcountsRows.data(), displsRows.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(h_outLocal.data(), checkedCountToInt(localRows, "localRows"), MPI_DOUBLE,
                    nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (worldRank == 0) {
        printf("Computation time: %.3f ms\n", maxSeconds * 1e3);
        const double avgTime = (maxSeconds * 1e3) / static_cast<double>(iterations);
        const double gflops = (maxSeconds > 0.0)
                                  ? (2.0 * static_cast<double>(nItems) * static_cast<double>(iterations)) /
                                        (maxSeconds * 1e9)
                                  : 0.0;
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }

            CUDA_CHECK(cudaFree(d_rowPtr));
            CUDA_CHECK(cudaFree(d_cols));
            CUDA_CHECK(cudaFree(d_val));
            CUDA_CHECK(cudaFree(d_x));
            CUDA_CHECK(cudaFree(d_y));

            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    MPI_Finalize();
    return 0;
}
