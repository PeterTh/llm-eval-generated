#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[Rank %d] CUDA error at %s:%d: %s\n", \
                mpi_rank, __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// ****************************************************************************
// CUDA Kernel: spmv_cuda_kernel
//
// Sparse matrix-vector multiplication using CSR format.
// One thread per row with __ldg read-only cache hints for optimal
// memory throughput on NVIDIA GPUs.
// ****************************************************************************
__global__ void spmv_cuda_kernel(const double* __restrict__ val,
                                  const index_t* __restrict__ cols,
                                  const index_t* __restrict__ rowDelimiters,
                                  const double* __restrict__ vec,
                                  const index_t numRows,
                                  double* __restrict__ out) {
    const index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows) return;

    double sum = 0.0;
    const index_t start = rowDelimiters[row];
    const index_t end = rowDelimiters[row + 1];
    for (index_t j = start; j < end; ++j) {
        sum += __ldg(&val[j]) * __ldg(&vec[cols[j]]);
    }
    out[row] = sum;
}

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
//   Computes sparse matrix-vector multiplication using CSR format (OpenMP)
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
    bool allValid = true;
    #pragma omp parallel for reduction(&&:allValid) schedule(static)
    for (index_t i = 0; i < size; ++i) {
        if (!allValid) continue;
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                #pragma omp critical
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                allValid = false;
            }
        } else {
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                #pragma omp critical
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i, ref, res, relError);
                allValid = false;
            }
        }
    }
    return allValid;
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
    // ========================================================================
    // MPI Initialization
    // ========================================================================
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // ========================================================================
    // CUDA Device Setup: assign one GPU per MPI rank (round-robin)
    // ========================================================================
    int numGPUs = 0;
    cudaGetDeviceCount(&numGPUs);
    if (numGPUs == 0) {
        fprintf(stderr, "[Rank %d] No CUDA devices found\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int gpuId = mpi_rank % numGPUs;
    CUDA_CHECK(cudaSetDevice(gpuId));

    // Warm up the GPU context
    double* d_dummy;
    CUDA_CHECK(cudaMalloc(&d_dummy, 256));
    CUDA_CHECK(cudaFree(d_dummy));

    // ========================================================================
    // Parse command line arguments (identically on all ranks)
    // ========================================================================
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

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
            printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    // ========================================================================
    // Compute row distribution across MPI ranks
    // ========================================================================
    // Each rank gets a contiguous block of rows
    std::vector<index_t> rowStart(mpi_size + 1);
    rowStart[0] = 0;
    for (int r = 0; r < mpi_size; ++r) {
        index_t base = numRows / mpi_size;
        index_t remainder = numRows % mpi_size;
        rowStart[r + 1] = rowStart[r] + base + (static_cast<index_t>(r) < remainder ? 1 : 0);
    }

    const index_t myStartRow = rowStart[mpi_rank];
    const index_t myEndRow = rowStart[mpi_rank + 1];
    const index_t myNumRows = myEndRow - myStartRow;

    if (mpi_rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Parallelization: MPI (%d ranks) + OpenMP + CUDA (%d GPUs)\n", mpi_size, numGPUs);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ========================================================================
    // Data Generation (Rank 0 only, to preserve reproducibility)
    // ========================================================================
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);

    if (mpi_rank == 0) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);

        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // ========================================================================
    // Distribute CSR data from Rank 0 to all ranks
    // ========================================================================

    // First, broadcast the nnz count per rank so each rank can allocate
    std::vector<index_t> nnzPerRank(mpi_size, 0);
    if (mpi_rank == 0) {
        for (int r = 0; r < mpi_size; ++r) {
            nnzPerRank[r] = h_rowDelimiters[rowStart[r + 1]] - h_rowDelimiters[rowStart[r]];
        }
    }
    MPI_Bcast(nnzPerRank.data(), mpi_size, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t myNnz = nnzPerRank[mpi_rank];
    const index_t myNnzStart = (mpi_rank == 0) ? 0 :
        [&]() -> index_t {
            index_t s = 0;
            for (int r = 0; r < mpi_rank; ++r) s += nnzPerRank[r];
            return s;
        }();

    // Local CSR data for this rank's rows
    std::vector<double> my_val(myNnz);
    std::vector<index_t> my_cols(myNnz);
    std::vector<index_t> my_rowDelimiters(myNumRows + 1);

    if (mpi_rank == 0) {
        // Extract rank 0's local data
        const index_t globalNnzStart = h_rowDelimiters[myStartRow];
        for (index_t i = 0; i < myNnz; ++i) {
            my_val[i] = h_val[globalNnzStart + i];
            my_cols[i] = h_cols[globalNnzStart + i];
        }
        for (index_t i = 0; i <= myNumRows; ++i) {
            my_rowDelimiters[i] = h_rowDelimiters[myStartRow + i] - globalNnzStart;
        }

        // Send data to other ranks
        for (int r = 1; r < mpi_size; ++r) {
            const index_t rGlobalNnzStart = h_rowDelimiters[rowStart[r]];
            const index_t rNnz = nnzPerRank[r];
            const index_t rNumRows = rowStart[r + 1] - rowStart[r];

            MPI_Send(&h_val[rGlobalNnzStart], rNnz, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
            MPI_Send(&h_cols[rGlobalNnzStart], rNnz, MPI_UINT32_T, r, 1, MPI_COMM_WORLD);

            // Send adjusted row delimiters (0-based for local portion)
            std::vector<index_t> rRowDelims(rNumRows + 1);
            for (index_t i = 0; i <= rNumRows; ++i) {
                rRowDelims[i] = h_rowDelimiters[rowStart[r] + i] - rGlobalNnzStart;
            }
            MPI_Send(rRowDelims.data(), static_cast<int>(rNumRows + 1), MPI_UINT32_T, r, 2, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(my_val.data(), myNnz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(my_cols.data(), myNnz, MPI_UINT32_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(my_rowDelimiters.data(), static_cast<int>(myNumRows + 1), MPI_UINT32_T, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Broadcast the dense vector to all ranks (needed since any row can reference any column)
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Free global data on rank 0
    if (mpi_rank == 0) {
        h_val.clear(); h_val.shrink_to_fit();
        h_cols.clear(); h_cols.shrink_to_fit();
        h_rowDelimiters.clear(); h_rowDelimiters.shrink_to_fit();
    }

    // ========================================================================
    // Compute reference solution (OpenMP CPU) for validation
    // ========================================================================
    std::vector<double> my_reference;
    if (validate) {
        if (mpi_rank == 0) printf("Computing reference solution...\n");
        my_reference.resize(myNumRows);
        spmvCpu(my_val.data(), my_cols.data(), my_rowDelimiters.data(),
                h_vec.data(), myNumRows, my_reference.data());
    }

    // ========================================================================
    // Allocate GPU memory and transfer data
    // ========================================================================
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_val, myNnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, myNnz * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (myNumRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, myNumRows * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_val, my_val.data(), myNnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, my_cols.data(), myNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, my_rowDelimiters.data(),
                          (myNumRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    // Free host CSR data (no longer needed after GPU upload)
    my_val.clear(); my_val.shrink_to_fit();
    my_cols.clear(); my_cols.shrink_to_fit();
    my_rowDelimiters.clear(); my_rowDelimiters.shrink_to_fit();
    // Keep h_vec for potential re-use, free after reference computation
    // Actually we already computed reference above, so free it
    h_vec.clear(); h_vec.shrink_to_fit();

    // ========================================================================
    // CUDA Kernel Launch Configuration
    // ========================================================================
    const int blockSize = 256;
    const int numBlocks = (static_cast<int>(myNumRows) + blockSize - 1) / blockSize;

    // ========================================================================
    // Timed SpMV Computation (GPU with CUDA events for accuracy)
    // ========================================================================
    if (mpi_rank == 0) printf("Computing SpMV...\n");

    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);

    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    CUDA_CHECK(cudaEventRecord(startEvent));

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmv_cuda_kernel<<<numBlocks, blockSize>>>(
            d_val, d_cols, d_rowDelimiters, d_vec, myNumRows, d_out);
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float localMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&localMs, startEvent, stopEvent));

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    // Get the maximum time across all ranks (bottleneck determines total time)
    double maxMs = 0.0;
    double localMsD = static_cast<double>(localMs);
    MPI_Reduce(&localMsD, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ========================================================================
    // Copy results back from GPU
    // ========================================================================
    std::vector<double> my_out(myNumRows);
    CUDA_CHECK(cudaMemcpy(my_out.data(), d_out, myNumRows * sizeof(double), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    // ========================================================================
    // Validation (each rank validates its own portion before gathering)
    // ========================================================================
    int globalValid = 1;
    if (validate) {
        bool localValid = verifyResults(my_reference.data(), my_out.data(), myNumRows);
        int localValidInt = localValid ? 1 : 0;
        MPI_Reduce(&localValidInt, &globalValid, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
    }

    // ========================================================================
    // Gather results to Rank 0
    // ========================================================================
    std::vector<double> h_out;
    std::vector<int> recvCounts(mpi_size);
    std::vector<int> recvDispls(mpi_size);
    if (mpi_rank == 0) {
        h_out.resize(numRows);
        recvDispls[0] = 0;
        for (int r = 0; r < mpi_size; ++r) {
            recvCounts[r] = static_cast<int>(rowStart[r + 1] - rowStart[r]);
            if (r > 0) recvDispls[r] = recvDispls[r - 1] + recvCounts[r - 1];
        }
    }
    int sendCount = static_cast<int>(myNumRows);

    MPI_Gatherv(my_out.data(), sendCount, MPI_DOUBLE,
                h_out.data(), recvCounts.data(), recvDispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Free local output and reference
    my_out.clear(); my_out.shrink_to_fit();
    my_reference.clear(); my_reference.shrink_to_fit();

    // ========================================================================
    // Output Results (Rank 0 only)
    // ========================================================================
    if (mpi_rank == 0) {
        const double totalMs = maxMs;
        const double totalSec = totalMs / 1000.0;

        printf("Computation time: %ld ms\n", static_cast<long>(totalMs));

        // Calculate performance metrics
        const double gflops = (totalSec > 0.0) ?
            (2.0 * nItems * iterations) / totalSec / 1e9 : 0.0;
        const double avgTime = totalMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        // Report validation result
        if (validate) {
            if (mpi_rank == 0) printf("Validating result...\n");
            if (globalValid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    // Cleanup
    MPI_Finalize();
    return 0;
}
