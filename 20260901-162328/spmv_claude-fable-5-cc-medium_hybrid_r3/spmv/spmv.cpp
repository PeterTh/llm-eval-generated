#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),    \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

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
//   (OpenMP-parallel reference implementation used for validation)
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
// CUDA kernels: CSR SpMV
//
// spmvKernelVector: one warp per row (good for rows with many non-zeros)
// spmvKernelScalar: one thread per row (good for very sparse rows)
// ****************************************************************************
__global__ void spmvKernelVector(const double* __restrict__ val,
                                 const index_t* __restrict__ cols,
                                 const index_t* __restrict__ rowDelimiters,
                                 const double* __restrict__ vec, const index_t nRows,
                                 double* __restrict__ out) {
    const index_t warpId = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warpId >= nRows) return;

    const index_t rowStart = rowDelimiters[warpId];
    const index_t rowEnd = rowDelimiters[warpId + 1];

    double sum = 0.0;
    for (index_t j = rowStart + lane; j < rowEnd; j += 32) {
        sum += val[j] * __ldg(&vec[cols[j]]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        out[warpId] = sum;
    }
}

__global__ void spmvKernelScalar(const double* __restrict__ val,
                                 const index_t* __restrict__ cols,
                                 const index_t* __restrict__ rowDelimiters,
                                 const double* __restrict__ vec, const index_t nRows,
                                 double* __restrict__ out) {
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nRows) return;

    double sum = 0.0;
    const index_t rowEnd = rowDelimiters[i + 1];
    for (index_t j = rowDelimiters[i]; j < rowEnd; ++j) {
        sum += val[j] * __ldg(&vec[cols[j]]);
    }
    out[i] = sum;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    const bool isRoot = (rank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank sees the same argv)
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
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (isRoot) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("MPI ranks: %d, OpenMP threads per rank: %d\n", nProcs, omp_get_max_threads());
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Select a GPU for this rank
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    // Full data structures (populated on rank 0 only)
    std::vector<double> h_val;                          // Non-zero values
    std::vector<index_t> h_cols;                        // Column indices
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters (all ranks)
    std::vector<double> h_vec(numRows);                 // Dense vector (all ranks)
    std::vector<double> h_out;                          // Output vector (rank 0)

    if (isRoot) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_out.resize(numRows);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast the dense vector and the row delimiters so every rank can
    // determine its own partition sizes.
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0,
              MPI_COMM_WORLD);

    // Contiguous row-block decomposition across ranks
    std::vector<index_t> rowStart(nProcs + 1);
    for (int r = 0; r <= nProcs; ++r) {
        const index_t base = numRows / static_cast<index_t>(nProcs);
        const index_t rem = numRows % static_cast<index_t>(nProcs);
        const index_t rr = static_cast<index_t>(r);
        rowStart[r] = rr * base + (rr < rem ? rr : rem);
    }
    const index_t myRowStart = rowStart[rank];
    const index_t myRows = rowStart[rank + 1] - myRowStart;
    const index_t myNnzStart = h_rowDelimiters[myRowStart];
    const index_t myNnz = h_rowDelimiters[rowStart[rank + 1]] - myNnzStart;

    // Scatter the non-zero values and column indices for the local row block
    std::vector<int> nnzCounts(nProcs), nnzDispls(nProcs), rowCounts(nProcs), rowDispls(nProcs);
    for (int r = 0; r < nProcs; ++r) {
        nnzCounts[r] = static_cast<int>(h_rowDelimiters[rowStart[r + 1]] -
                                        h_rowDelimiters[rowStart[r]]);
        nnzDispls[r] = static_cast<int>(h_rowDelimiters[rowStart[r]]);
        rowCounts[r] = static_cast<int>(rowStart[r + 1] - rowStart[r]);
        rowDispls[r] = static_cast<int>(rowStart[r]);
    }

    std::vector<double> h_localVal(myNnz);
    std::vector<index_t> h_localCols(myNnz);
    MPI_Scatterv(isRoot ? h_val.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 h_localVal.data(), static_cast<int>(myNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(isRoot ? h_cols.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                 MPI_UINT32_T, h_localCols.data(), static_cast<int>(myNnz), MPI_UINT32_T, 0,
                 MPI_COMM_WORLD);

    // Rebase local row delimiters to start at zero (OpenMP-parallel)
    std::vector<index_t> h_localRowDelimiters(myRows + 1);
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= myRows; ++i) {
        h_localRowDelimiters[i] = h_rowDelimiters[myRowStart + i] - myNnzStart;
    }

    // For validation, compute reference solution on the host with OpenMP
    std::vector<double> h_reference;
    if (validate && isRoot) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Upload the local partition to this rank's GPU (outside the timed region,
    // matching the original code where setup precedes the timed loop)
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val, myNnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, myNnz * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (myRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, (myRows > 0 ? myRows : 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_val, h_localVal.data(), myNnz * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_localCols.data(), myNnz * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, h_localRowDelimiters.data(),
                          (myRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Pick kernel based on average non-zeros per row
    const double avgNnzPerRow =
        myRows > 0 ? static_cast<double>(myNnz) / static_cast<double>(myRows) : 0.0;
    const bool useVectorKernel = avgNnzPerRow >= 8.0;
    constexpr int BLOCK_SIZE = 256;
    const index_t vectorBlocks =
        myRows > 0 ? (myRows * 32 + BLOCK_SIZE - 1) / BLOCK_SIZE : 0;
    const index_t scalarBlocks = myRows > 0 ? (myRows + BLOCK_SIZE - 1) / BLOCK_SIZE : 0;

    // Perform SpMV computation
    if (isRoot) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (myRows > 0) {
            if (useVectorKernel) {
                spmvKernelVector<<<vectorBlocks, BLOCK_SIZE>>>(
                    d_val, d_cols, d_rowDelimiters, d_vec, myRows, d_out);
            } else {
                spmvKernelScalar<<<scalarBlocks, BLOCK_SIZE>>>(
                    d_val, d_cols, d_rowDelimiters, d_vec, myRows, d_out);
            }
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy the local result back and gather the full output on rank 0
    std::vector<double> h_localOut(myRows);
    CUDA_CHECK(cudaMemcpy(h_localOut.data(), d_out, myRows * sizeof(double),
                          cudaMemcpyDeviceToHost));
    MPI_Gatherv(h_localOut.data(), static_cast<int>(myRows), MPI_DOUBLE,
                isRoot ? h_out.data() : nullptr, rowCounts.data(), rowDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    int exitCode = 0;
    if (isRoot) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    // All ranks agree on the exit code
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
