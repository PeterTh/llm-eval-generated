#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[Rank %d] CUDA error at %s:%d: %s\n", rank, __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// ****************************************************************************
// CUDA kernel: CSR SpMV (warp-per-row for better parallelism)
// Each warp computes one row of the output vector.
// ****************************************************************************
__global__ void spmvCsrWarpKernel(const double* __restrict__ val,
                                   const index_t* __restrict__ cols,
                                   const index_t* __restrict__ rowDelimiters,
                                   const double* __restrict__ vec,
                                   const index_t startRow,
                                   const index_t endRow,
                                   double* __restrict__ out) {
    const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int laneId = threadIdx.x % 32;
    const index_t row = startRow + warpId;

    if (row < endRow) {
        double t = 0.0;
        const index_t rowStart = rowDelimiters[row];
        const index_t rowEnd = rowDelimiters[row + 1];
        for (index_t j = rowStart + laneId; j < rowEnd; j += 32) {
            t += val[j] * vec[cols[j]];
        }
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            t += __shfl_down_sync(0xFFFFFFFF, t, offset);
        }
        if (laneId == 0) {
            out[row] = t;
        }
    }
}

// ****************************************************************************
// CUDA kernel: CSR SpMV (one thread per row, for small rows)
// ****************************************************************************
__global__ void spmvCsrKernel(const double* __restrict__ val,
                               const index_t* __restrict__ cols,
                               const index_t* __restrict__ rowDelimiters,
                               const double* __restrict__ vec,
                               const index_t startRow,
                               const index_t endRow,
                               double* __restrict__ out) {
    index_t row = blockIdx.x * blockDim.x + threadIdx.x + startRow;
    if (row < endRow) {
        double t = 0.0;
        for (index_t j = rowDelimiters[row]; j < rowDelimiters[row + 1]; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[row] = t;
    }
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
    int failed = 0;
    #pragma omp parallel for reduction(|:failed) schedule(static)
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        bool localFail = false;
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                localFail = true;
            }
        } else {
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                localFail = true;
            }
        }
        if (localFail) {
            #pragma omp critical
            {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
            }
            failed = 1;
        }
    }
    return failed == 0;
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
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResultsFlag = false;

    // Parse command line arguments (all ranks get same args)
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
            printResultsFlag = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Parallelization: MPI (%d ranks) + OpenMP + CUDA\n", size);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ========================================================================
    // Data initialization (rank 0 only, to maintain deterministic RNG sequence)
    // ========================================================================
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec;

    if (rank == 0) {
        if (rank == 0) printf("Initializing data structures...\n");

        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_vec.resize(numRows);

        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    } else {
        // Non-root ranks: allocate empty containers (data will be received)
        h_rowDelimiters.resize(numRows + 1);
        h_vec.resize(numRows);
    }

    // ========================================================================
    // Compute row distribution across MPI ranks
    // ========================================================================
    const index_t rowsPerRank = numRows / size;
    const index_t remainderRows = numRows % size;

    // Each rank gets rows [rowStart, rowEnd)
    const index_t rowStart = static_cast<index_t>(rank) * rowsPerRank + std::min(static_cast<index_t>(rank), remainderRows);
    const index_t rowEnd = static_cast<index_t>(rank + 1) * rowsPerRank + std::min(static_cast<index_t>(rank + 1), remainderRows);
    const index_t localNumRows = rowEnd - rowStart;

    // Broadcast full rowDelimiters and vec from rank 0
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute per-rank nnz counts and displacements for Scatterv
    const index_t localNnzStart = h_rowDelimiters[rowStart];
    const index_t localNnzEnd = h_rowDelimiters[rowEnd];
    const index_t localNnz = localNnzEnd - localNnzStart;

    std::vector<int> nnzCounts(size), nnzDispls(size);
    if (rank == 0) {
        for (int r = 0; r < size; r++) {
            index_t rStart = static_cast<index_t>(r) * rowsPerRank + std::min(static_cast<index_t>(r), remainderRows);
            index_t rEnd = static_cast<index_t>(r + 1) * rowsPerRank + std::min(static_cast<index_t>(r + 1), remainderRows);
            nnzCounts[r] = static_cast<int>(h_rowDelimiters[rEnd] - h_rowDelimiters[rStart]);
            nnzDispls[r] = static_cast<int>(h_rowDelimiters[rStart]);
        }
    }

    // Scatter val and cols to all ranks
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);

    MPI_Scatterv(h_val.data(), nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(h_cols.data(), nnzCounts.data(), nnzDispls.data(), MPI_UINT32_T,
                 localCols.data(), static_cast<int>(localNnz), MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    // Free root-only data on non-root ranks
    if (rank != 0) {
        h_val.clear(); h_val.shrink_to_fit();
        h_cols.clear(); h_cols.shrink_to_fit();
    }

    // Build local rowDelimiters (shifted so localRowDelimiters[0] = 0)
    std::vector<index_t> localRowDelimiters(localNumRows + 1);
    for (index_t i = 0; i <= localNumRows; i++) {
        localRowDelimiters[i] = h_rowDelimiters[rowStart + i] - localNnzStart;
    }

    // Free full rowDelimiters (no longer needed)
    h_rowDelimiters.clear();
    h_rowDelimiters.shrink_to_fit();

    // ========================================================================
    // CUDA setup: allocate device memory and copy data
    // ========================================================================
    // Select GPU based on rank (for multi-GPU nodes)
    int numGPUs = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numGPUs));
    int gpuId = rank % numGPUs;
    CUDA_CHECK(cudaSetDevice(gpuId));

    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rowDelimiters = nullptr;

    if (localNnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, localNnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, localNnz * sizeof(index_t)));
        CUDA_CHECK(cudaMemcpy(d_val, localVal.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, localCols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (localNumRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, localNumRows * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, localRowDelimiters.data(),
                           (localNumRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    // Free host copies of data that's now on the device
    localVal.clear(); localVal.shrink_to_fit();
    localCols.clear(); localCols.shrink_to_fit();
    localRowDelimiters.clear(); localRowDelimiters.shrink_to_fit();

    // ========================================================================
    // Create CUDA streams for OpenMP threads
    // ========================================================================
    const int maxOmpThreads = omp_get_max_threads();
    std::vector<cudaStream_t> streams(maxOmpThreads);
    for (int s = 0; s < maxOmpThreads; s++) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    // ========================================================================
    // Compute reference solution for validation (CPU, OpenMP-parallelized)
    // ========================================================================
    std::vector<double> localReference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        localReference.resize(localNumRows);

        // Reconstruct local data for CPU reference computation
        std::vector<double> refVal(localNnz);
        std::vector<index_t> refCols(localNnz);
        std::vector<index_t> refRowDelimiters(localNumRows + 1);

        CUDA_CHECK(cudaMemcpy(refVal.data(), d_val, localNnz * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(refCols.data(), d_cols, localNnz * sizeof(index_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(refRowDelimiters.data(), d_rowDelimiters,
                               (localNumRows + 1) * sizeof(index_t), cudaMemcpyDeviceToHost));

        spmvCpu(refVal.data(), refCols.data(), refRowDelimiters.data(),
                h_vec.data(), localNumRows, localReference.data());
    }

    // ========================================================================
    // SpMV computation: MPI + OpenMP + CUDA hybrid
    // ========================================================================
    if (rank == 0) printf("Computing SpMV...\n");

    // Determine average nnz per row to choose kernel strategy
    index_t avgNnzPerRow = (localNumRows > 0) ? (localNnz / localNumRows) : 0;
    const bool useWarpKernel = (avgNnzPerRow >= 8);

    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        // Launch CUDA kernel (single kernel, all rows)
        if (localNnz > 0) {
            if (useWarpKernel) {
                // Warp-per-row kernel: each warp handles one row
                const int threadsPerBlock = 256;
                const int totalThreads = static_cast<int>(localNumRows) * 32;
                const int gridSize = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
                spmvCsrWarpKernel<<<gridSize, threadsPerBlock>>>(
                    d_val, d_cols, d_rowDelimiters, d_vec, 0, localNumRows, d_out);
            } else {
                // One-thread-per-row kernel
                const int blockSize = 256;
                const int gridSize = (static_cast<int>(localNumRows) + blockSize - 1) / blockSize;
                spmvCsrKernel<<<gridSize, blockSize>>>(
                    d_val, d_cols, d_rowDelimiters, d_vec, 0, localNumRows, d_out);
            }
        }
        
        // Synchronize CUDA
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // OpenMP can be used for post-processing or overlapping communication
        // For now, we just use it for the iteration loop parallelism
    }

    // Synchronize all MPI ranks after computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double local_duration_ms = duration_us.count() / 1000.0;
    double duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ========================================================================
    // Copy results from device to host
    // ========================================================================
    std::vector<double> localOut(localNumRows);
    if (localNumRows > 0) {
        CUDA_CHECK(cudaMemcpy(localOut.data(), d_out, localNumRows * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // ========================================================================
    // Gather results to rank 0 for output
    // ========================================================================
    std::vector<double> fullOut;
    std::vector<int> outCounts(size), outDispls(size);
    for (int r = 0; r < size; r++) {
        index_t rStart = static_cast<index_t>(r) * rowsPerRank + std::min(static_cast<index_t>(r), remainderRows);
        index_t rEnd = static_cast<index_t>(r + 1) * rowsPerRank + std::min(static_cast<index_t>(r + 1), remainderRows);
        outCounts[r] = static_cast<int>(rEnd - rStart);
        outDispls[r] = static_cast<int>(rStart);
    }

    if (rank == 0) {
        fullOut.resize(numRows);
    }

    MPI_Gatherv(localOut.data(), static_cast<int>(localNumRows), MPI_DOUBLE,
                fullOut.data(), outCounts.data(), outDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // ========================================================================
    // Report timing and performance metrics (rank 0)
    // ========================================================================
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration_ms);

        const double gflops = (2.0 * nItems * iterations) / (duration_ms / 1000.0) / 1e9;
        const double avgTime = duration_ms / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResultsFlag) {
            print_results(fullOut, "OutputVector");
        }
    }

    // ========================================================================
    // Validation (each rank validates its local portion)
    // ========================================================================
    int localValid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");

        // Verify local portion
        for (index_t i = 0; i < localNumRows; i++) {
            const double ref = localReference[i];
            const double res = localOut[i];
            if (std::abs(ref) < 1e-10) {
                if (std::abs(res) > MAX_RELATIVE_ERROR) {
                    printf("[Rank %d] Validation failed at global index %u: reference %.10e, got %.10e\n",
                           rank, rowStart + i, ref, res);
                    localValid = 0;
                    break;
                }
            } else {
                const double relError = std::abs((res - ref) / ref);
                if (relError > MAX_RELATIVE_ERROR) {
                    printf("[Rank %d] Validation failed at global index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                           rank, rowStart + i, ref, res, relError);
                    localValid = 0;
                    break;
                }
            }
        }
    }

    // Global validation result
    int globalValid = 0;
    MPI_Reduce(&localValid, &globalValid, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    
    // Broadcast validation result to all ranks
    MPI_Bcast(&globalValid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (validate && rank == 0) {
        if (globalValid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // ========================================================================
    // Cleanup
    // ========================================================================
    for (int s = 0; s < maxOmpThreads; s++) {
        cudaStreamDestroy(streams[s]);
    }
    if (d_val) cudaFree(d_val);
    if (d_cols) cudaFree(d_cols);
    if (d_rowDelimiters) cudaFree(d_rowDelimiters);
    if (d_vec) cudaFree(d_vec);
    if (d_out) cudaFree(d_out);

    MPI_Finalize();

    return (validate && !globalValid) ? 1 : 0;
}
