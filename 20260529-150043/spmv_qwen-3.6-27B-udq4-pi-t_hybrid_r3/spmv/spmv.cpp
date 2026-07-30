#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// CUDA error checking
// ****************************************************************************
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err), err);           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

// ****************************************************************************
// MPI error checking
// ****************************************************************************
#define MPI_CHECK(call)                                                          \
    do {                                                                         \
        int err = call;                                                          \
        if (err != MPI_SUCCESS) {                                                \
            char errbuf[MPI_MAX_ERROR_STRING];                                   \
            int len;                                                             \
            MPI_Error_string(err, errbuf, &len);                                 \
            fprintf(stderr, "MPI error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    errbuf);                                                     \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
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
    const uint64_t totalEntries = static_cast<uint64_t>(dim) * dim;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            uint64_t numEntriesLeft = totalEntries - (static_cast<uint64_t>(i) * dim + j);
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
// CUDA Kernel: spmvCudaKernel
//
// Purpose:
//   Computes sparse matrix-vector multiplication on GPU using CSR format.
//   Each thread block handles one row, using shared memory for parallel
//   reduction within the block.
//
// Arguments:
//   val:           array holding the non-zero values for the matrix
//   cols:          array of column indices for each element
//   rowDelimiters: local row delimiters (size: localDim+1)
//   vec:           dense vector of size dim to be used for multiplication
//   localDim:      number of rows in this rank's partition
//   out:           output - result from the spmv calculation (local portion)
//
// ****************************************************************************
__global__ void spmvCudaKernel(const double* val, const index_t* cols,
                                const index_t* rowDelimiters,
                                const double* vec, const index_t localDim,
                                double* out) {
    // Each block handles one row
    const index_t row = blockIdx.x;
    if (row >= localDim) return;

    // rowDelimiters uses local indices (0 to localNnz)
    const index_t rowStart = rowDelimiters[row];
    const index_t rowEnd = rowDelimiters[row + 1];
    const index_t rowLen = rowEnd - rowStart;

    // Shared memory for parallel reduction within the block
    __shared__ double sdata[256];

    // Each thread processes elements with grid-stride loop
    double localSum = 0.0;
    for (index_t j = threadIdx.x; j < rowLen; j += blockDim.x) {
        const index_t colIdx = cols[rowStart + j];
        localSum += val[rowStart + j] * vec[colIdx];
    }

    // Store partial sum in shared memory
    sdata[threadIdx.x] = localSum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (index_t s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
    }

    // Thread 0 writes the final result for this row
    if (threadIdx.x == 0) {
        out[row] = sdata[0];
    }
}

// ****************************************************************************
// Function: spmvCuda
//
// Purpose:
//   Launches CUDA SpMV kernel with optimal configuration
//
// ****************************************************************************
void spmvCuda(const double* d_val, const index_t* d_cols,
              const index_t* d_rowDelimiters, const double* d_vec,
              const index_t localDim, double* d_out) {
    // 256 threads per block for optimal occupancy on modern GPUs
    const int blockSize = 256;
    const int numBlocks = localDim;

    spmvCudaKernel<<<numBlocks, blockSize>>>(
        d_val, d_cols, d_rowDelimiters, d_vec, localDim, d_out);

    CUDA_CHECK(cudaGetLastError());
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format
//   (OpenMP parallelized for reference/validation)
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
#ifndef __CUDACC__
#pragma omp parallel for schedule(static)
#endif
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
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ****************************************************************************
// Distributed CSR structure for MPI partitioning
// ****************************************************************************
struct DistributedCSR {
    std::vector<double> val;           // Local non-zero values
    std::vector<index_t> cols;         // Local column indices
    std::vector<index_t> rowDelimiters; // Local row delimiters (size: localDim+1)
    index_t localDim;                  // Number of rows on this rank
    index_t globalOffset;              // Offset into original rowDelimiters
    index_t localNnz;                  // Number of non-zeros on this rank
};

// ****************************************************************************
// Function: distributeCSR
//
// Purpose:
//   Distribute CSR matrix rows across MPI ranks using static block distribution
//
// ****************************************************************************
DistributedCSR distributeCSR(const double* h_val, const index_t* h_cols,
                              const index_t* h_rowDelimiters,
                              const index_t numRows,
                              const int rank, const int numRanks) {
    // Compute row distribution: static block distribution
    const index_t rowsPerRank = numRows / numRanks;
    const index_t remainder = numRows % numRanks;

    // Determine start and end row for this rank
    index_t localStartRow = 0;
    for (int r = 0; r < rank; ++r) {
        localStartRow += rowsPerRank + (r < static_cast<int>(remainder) ? 1 : 0);
    }
    const index_t localEndRow = localStartRow + rowsPerRank +
                                (rank < static_cast<int>(remainder) ? 1 : 0);
    const index_t localDim = localEndRow - localStartRow;

    // Count local non-zeros
    index_t localNnz = 0;
    for (index_t i = localStartRow; i < localEndRow; ++i) {
        localNnz += h_rowDelimiters[i + 1] - h_rowDelimiters[i];
    }

    // Allocate local arrays
    DistributedCSR local;
    local.localDim = localDim;
    local.globalOffset = localStartRow;
    local.localNnz = localNnz;

    local.val.resize(localNnz);
    local.cols.resize(localNnz);
    local.rowDelimiters.resize(localDim + 1);

    // Copy local data using OpenMP for parallelization
    index_t localIdx = 0;
    for (index_t i = 0; i < localDim; ++i) {
        const index_t globalRow = localStartRow + i;
        local.rowDelimiters[i] = localIdx;
        for (index_t j = h_rowDelimiters[globalRow]; j < h_rowDelimiters[globalRow + 1]; ++j) {
            local.val[localIdx] = h_val[j];
            local.cols[localIdx] = h_cols[j];
            localIdx++;
        }
    }
    local.rowDelimiters[localDim] = localNnz;

    return local;
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank, numRanks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

    // Set CUDA device for this rank (one GPU per MPI rank, cycle if more ranks than GPUs)
    int numGpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numGpus));
    int cudaDevice = rank % std::max(numGpus, 1);
    CUDA_CHECK(cudaSetDevice(cudaDevice));

    // Query GPU properties
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, cudaDevice);
    if (err != cudaSuccess) {
        prop.name[0] = '\0';
        prop.major = 0;
        prop.minor = 0;
    }

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse, but only rank 0 prints)
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_CHECK(MPI_Finalize());
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_CHECK(MPI_Finalize());
            return 1;
        }
    }

    // Calculate number of non-zero elements (use 64-bit arithmetic to avoid overflow)
    const index_t nItems = static_cast<index_t>(
        (static_cast<uint64_t>(numRows) * numRows) / sparsity);

    // Print configuration (rank 0 only)
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Parallelization: MPI (%d ranks) + OpenMP + CUDA\n", numRanks);
        if (numGpus > 0) {
            printf("GPU: %s (CC %d.%d, %d GPUs available)\n",
                   prop.name, prop.major, prop.minor, numGpus);
        } else {
            printf("GPU: none available\n");
        }
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) /
               (static_cast<double>(numRows) * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("OpenMP threads available: %d\n", omp_get_max_threads());
    }

    // Allocate and initialize data structures on rank 0
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec;

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_vec.resize(numRows);

        printf("Initializing data structures...\n");

        // Use OpenMP to parallelize initialization
#ifndef __CUDACC__
#pragma omp parallel for schedule(static)
#endif
        for (index_t i = 0; i < numRows; ++i) {
            h_vec[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
        }

#ifndef __CUDACC__
#pragma omp parallel for schedule(static)
#endif
        for (index_t i = 0; i < nItems; ++i) {
            h_val[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
        }

        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // For validation, compute reference solution on rank 0
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Broadcast vector to all ranks (every rank needs the full vector for SpMV)
    std::vector<double> local_vec(numRows);
    if (rank == 0) {
        local_vec = h_vec;  // Copy initialized vector to local_vec on rank 0
    }
    MPI_CHECK(MPI_Bcast(local_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD));

    // Distribute CSR matrix across MPI ranks
    // Rank 0 computes distribution for all ranks and sends to others
    DistributedCSR localCSR;

    if (rank == 0) {
        localCSR = distributeCSR(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                                 numRows, 0, numRanks);

        // Send distributed data to other ranks
        for (int r = 1; r < numRanks; ++r) {
            DistributedCSR remoteCSR = distributeCSR(
                h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                numRows, r, numRanks);

            MPI_CHECK(MPI_Send(&remoteCSR.localNnz, 1, MPI_UINT32_T, r, 0, MPI_COMM_WORLD));
            MPI_CHECK(MPI_Send(&remoteCSR.localDim, 1, MPI_UINT32_T, r, 1, MPI_COMM_WORLD));
            MPI_CHECK(MPI_Send(&remoteCSR.globalOffset, 1, MPI_UINT32_T, r, 2, MPI_COMM_WORLD));
            if (remoteCSR.localNnz > 0) {
                MPI_CHECK(MPI_Send(remoteCSR.val.data(), remoteCSR.localNnz, MPI_DOUBLE, r, 3,
                                   MPI_COMM_WORLD));
                MPI_CHECK(MPI_Send(remoteCSR.cols.data(), remoteCSR.localNnz, MPI_UNSIGNED, r, 4,
                                   MPI_COMM_WORLD));
            }
            MPI_CHECK(MPI_Send(remoteCSR.rowDelimiters.data(), remoteCSR.localDim + 1, MPI_UNSIGNED,
                               r, 5, MPI_COMM_WORLD));
        }
    } else {
        // Receive metadata
        MPI_CHECK(MPI_Recv(&localCSR.localNnz, 1, MPI_UINT32_T, 0, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CHECK(MPI_Recv(&localCSR.localDim, 1, MPI_UINT32_T, 0, 1,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CHECK(MPI_Recv(&localCSR.globalOffset, 1, MPI_UINT32_T, 0, 2,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));

        localCSR.val.resize(localCSR.localNnz);
        localCSR.cols.resize(localCSR.localNnz);
        localCSR.rowDelimiters.resize(localCSR.localDim + 1);

        // Receive data arrays
        if (localCSR.localNnz > 0) {
            MPI_CHECK(MPI_Recv(localCSR.val.data(), localCSR.localNnz, MPI_DOUBLE, 0, 3,
                               MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            MPI_CHECK(MPI_Recv(localCSR.cols.data(), localCSR.localNnz, MPI_UNSIGNED, 0, 4,
                               MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        }
        MPI_CHECK(MPI_Recv(localCSR.rowDelimiters.data(), localCSR.localDim + 1, MPI_UNSIGNED, 0, 5,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }

    // Barrier to ensure all ranks have received their data
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Allocate GPU memory
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    if (localCSR.localNnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, localCSR.localNnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, localCSR.localNnz * sizeof(index_t)));
    }
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (localCSR.localDim + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, localCSR.localDim * sizeof(double)));

    // Transfer data to GPU
    if (localCSR.localNnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_val, localCSR.val.data(),
                              localCSR.localNnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, localCSR.cols.data(),
                              localCSR.localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, localCSR.rowDelimiters.data(),
                          (localCSR.localDim + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, local_vec.data(),
                          numRows * sizeof(double), cudaMemcpyHostToDevice));

    // Allocate host output for this rank
    std::vector<double> local_out(localCSR.localDim);

    // Warmup iteration to ensure kernels are loaded and compiled
    if (localCSR.localDim > 0) {
        spmvCuda(d_val, d_cols, d_rowDelimiters, d_vec,
                 localCSR.localDim, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    // Use CUDA events for precise GPU timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    CUDA_CHECK(cudaEventRecord(startEvent));

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localCSR.localDim > 0) {
            spmvCuda(d_val, d_cols, d_rowDelimiters, d_vec,
                     localCSR.localDim, d_out);
        }
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    // Copy results back from GPU
    if (localCSR.localDim > 0) {
        CUDA_CHECK(cudaMemcpy(local_out.data(), d_out,
                              localCSR.localDim * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Gather results to rank 0
    std::vector<double> h_out(numRows);
    std::vector<int> recvcounts(numRanks);
    std::vector<int> displs(numRanks);

    for (int r = 0; r < numRanks; ++r) {
        index_t rowsPerRank_local = numRows / numRanks;
        index_t remainder_local = numRows % numRanks;
        recvcounts[r] = rowsPerRank_local + (r < static_cast<int>(remainder_local) ? 1 : 0);
        displs[r] = 0;
        for (int rr = 0; rr < r; ++rr) {
            displs[r] += rowsPerRank_local + (rr < static_cast<int>(remainder_local) ? 1 : 0);
        }
    }

    MPI_CHECK(MPI_Gatherv(local_out.data(), localCSR.localDim, MPI_DOUBLE,
                          h_out.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                          0, MPI_COMM_WORLD));

    // Calculate performance metrics (rank 0)
    if (rank == 0) {
        const double gflops = (2.0 * nItems * iterations) / (milliseconds / 1000.0) / 1e9;
        const double avgTime = milliseconds / static_cast<double>(iterations);

        printf("Computation time: %.2f ms\n", milliseconds);
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
            }
        }
    }

    // Cleanup GPU memory
    if (d_val) CUDA_CHECK(cudaFree(d_val));
    if (d_cols) CUDA_CHECK(cudaFree(d_cols));
    if (d_rowDelimiters) CUDA_CHECK(cudaFree(d_rowDelimiters));
    if (d_vec) CUDA_CHECK(cudaFree(d_vec));
    if (d_out) CUDA_CHECK(cudaFree(d_out));

    // Finalize MPI
    MPI_CHECK(MPI_Finalize());

    return 0;
}
