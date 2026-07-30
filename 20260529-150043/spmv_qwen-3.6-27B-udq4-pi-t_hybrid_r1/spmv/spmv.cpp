#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <omp.h>

#include <mpi.h>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// CUDA error checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err), #call);          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

// MPI error checking macro
#define MPI_CHECK(call)                                                          \
    do {                                                                         \
        int err = call;                                                          \
        if (err != MPI_SUCCESS) {                                                \
            char errbuf[MPI_MAX_ERROR_STRING];                                   \
            int len;                                                             \
            MPI_Error_string(err, errbuf, &len);                                 \
            fprintf(stderr, "MPI error at %s:%d: %s\n", __FILE__, __LINE__, errbuf); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values (OpenMP parallelized)
//
// Arguments:
//   A: pointer to the array to initialize
//   n: number of elements in the array
//   maxVal: specifies range of random values [0, maxVal]
//
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
#pragma omp parallel for schedule(static)
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
// CUDA Kernel: spmvCuda
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format on GPU
//   Each thread block handles one row using a parallel reduction
//
// Arguments:
//   val:           array holding the non-zero values for the matrix
//   cols:          array of column indices for each element
//   rowDelimiters: array of size dim+1 holding indices to rows
//   vec:           dense vector of size dim to be used for multiplication
//   dim:           number of rows/columns in the matrix
//   out:           output - result from the spmv calculation
//
// ****************************************************************************
__global__ void spmvCudaKernel(const double* val, const index_t* cols,
                                const index_t* rowDelimiters, const double* vec,
                                const index_t dim, double* out) {
    index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dim) {
        double sum = 0.0;
        index_t start = rowDelimiters[row];
        index_t end = rowDelimiters[row + 1];
        for (index_t j = start; j < end; ++j) {
            sum += val[j] * vec[cols[j]];
        }
        out[row] = sum;
    }
}

// ****************************************************************************
// Function: spmvCuda
//
// Purpose:
//   Host-side SpMV using CUDA GPU acceleration
//
// Arguments:
//   d_val:           device array holding the non-zero values
//   d_cols:          device array of column indices
//   d_rowDelimiters: device array of row delimiters
//   d_vec:           device dense vector
//   dim:             number of rows/columns in the matrix
//   d_out:           device output vector
//
// ****************************************************************************
void spmvCuda(const double* d_val, const index_t* d_cols, const index_t* d_rowDelimiters,
              const double* d_vec, const index_t dim, double* d_out) {
    const int blockSize = 256;
    const int numBlocks = (dim + blockSize - 1) / blockSize;

    spmvCudaKernel<<<numBlocks, blockSize>>>(d_val, d_cols, d_rowDelimiters, d_vec, dim, d_out);
    CUDA_CHECK(cudaGetLastError());
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format (OpenMP parallelized)
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
    bool valid = true;
#pragma omp parallel for schedule(static) reduction(||:valid)
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                valid = false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i, ref, res, relError);
                valid = false;
            }
        }
    }
    return valid;
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

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank, numRanks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

    // Get number of available GPUs
    int numGpus;
    CUDA_CHECK(cudaGetDeviceCount(&numGpus));

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse)
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

    // Select GPU for this rank (round-robin across available GPUs)
    int gpuId = rank % numGpus;
    CUDA_CHECK(cudaSetDevice(gpuId));

    // Print configuration from rank 0
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Hybrid MPI/OpenMP/CUDA parallelization\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
        printf("CUDA GPUs available: %d\n", numGpus);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        // Print device info
        cudaDeviceProp prop;
        for (int g = 0; g < numGpus; ++g) {
            cudaGetDeviceProperties(&prop, g);
            printf("  GPU %d: %s (SM %d, %zu MB)\n", g, prop.name, prop.multiProcessorCount,
                   prop.totalGlobalMem / (1024 * 1024));
        }
    }

    // =========================================================================
    // Phase 1: Rank 0 generates the full matrix, then distributes to all ranks
    // =========================================================================

    // Rank 0 builds the full CSR matrix
    std::vector<double> h_val(nItems);
    std::vector<index_t> h_cols(nItems);
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);

    // Initialize data on rank 0
    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast rowDelimiters to all ranks (needed for partitioning)
    MPI_CHECK(MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));

    // Broadcast the full input vector to all ranks (needed for SpMV)
    MPI_CHECK(MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD));

    // =========================================================================
    // Phase 2: Partition matrix data across MPI ranks
    // =========================================================================

    // Each rank gets a contiguous block of rows
    const index_t rowsPerRank = numRows / numRanks;
    const index_t remainder = numRows % numRanks;
    const index_t localRowStart = rank * rowsPerRank + std::min(static_cast<index_t>(rank), remainder);
    const index_t localRowEnd = (rank + 1) * rowsPerRank + std::min(static_cast<index_t>(rank + 1), remainder);
    const index_t localNumRows = localRowEnd - localRowStart;

    // Determine local nnz range
    const index_t localNnzStart = h_rowDelimiters[localRowStart];
    const index_t localNnzEnd = h_rowDelimiters[localRowEnd];
    const index_t localNnz = localNnzEnd - localNnzStart;

    // Receive local val and cols from rank 0
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);

    // Build local rowDelimiters (offset to start at 0)
    std::vector<index_t> localRowDelimiters(localNumRows + 1);
    for (index_t i = 0; i <= localNumRows; ++i) {
        localRowDelimiters[i] = h_rowDelimiters[localRowStart + i] - localNnzStart;
    }

    if (rank == 0) {
        // Rank 0 sends data to other ranks
        for (int r = 1; r < numRanks; ++r) {
            const index_t rStart = r * rowsPerRank + std::min(static_cast<index_t>(r), remainder);
            const index_t rEnd = (r + 1) * rowsPerRank + std::min(static_cast<index_t>(r + 1), remainder);
            const index_t rNnzStart = h_rowDelimiters[rStart];
            const index_t rNnzEnd = h_rowDelimiters[rEnd];
            const index_t rNnz = rNnzEnd - rNnzStart;
            MPI_CHECK(MPI_Send(h_val.data() + rNnzStart, rNnz, MPI_DOUBLE, r, 0, MPI_COMM_WORLD));
            MPI_CHECK(MPI_Send(h_cols.data() + rNnzStart, rNnz, MPI_UINT32_T, r, 1, MPI_COMM_WORLD));
        }
        // Rank 0 uses its own local data
        localVal.assign(h_val.begin() + localNnzStart, h_val.begin() + localNnzEnd);
        localCols.assign(h_cols.begin() + localNnzStart, h_cols.begin() + localNnzEnd);
    } else {
        MPI_CHECK(MPI_Recv(localVal.data(), localNnz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CHECK(MPI_Recv(localCols.data(), localNnz, MPI_UINT32_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }

    // Local output vector
    std::vector<double> localOut(localNumRows);

    // =========================================================================
    // Phase 3: Validate on CPU (if requested) - reference solution on rank 0
    // =========================================================================
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution (CPU, OpenMP)...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // =========================================================================
    // Phase 4: GPU SpMV computation with CUDA
    // =========================================================================

    // Allocate device memory for local data
    double* d_val;
    index_t* d_cols;
    index_t* d_rowDelimiters;
    double* d_vec;
    double* d_out;

    CUDA_CHECK(cudaMalloc(&d_val, localNnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, localNnz * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (localNumRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, localNumRows * sizeof(double)));

    // Copy local matrix data to GPU
    CUDA_CHECK(cudaMemcpy(d_val, localVal.data(), localNnz * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, localCols.data(), localNnz * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, localRowDelimiters.data(),
                          (localNumRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Use CUDA streams for potential overlap
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warm-up iteration
    if (rank == 0) {
        printf("Computing SpMV (CUDA GPU)...\n");
    }

    // Synchronize before timing
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCuda(d_val, d_cols, d_rowDelimiters, d_vec, localNumRows, d_out);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Copy result back from GPU
    CUDA_CHECK(cudaMemcpy(localOut.data(), d_out, localNumRows * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // Clean up GPU resources
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    // =========================================================================
    // Phase 5: Gather results and report performance
    // =========================================================================

    // Gather all local timing data to rank 0
    double localDurationUs = static_cast<double>(duration.count());
    double globalDurationUs;
    MPI_CHECK(MPI_Reduce(&localDurationUs, &globalDurationUs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    // Gather local output vectors to rank 0
    std::vector<double> globalOut;
    if (rank == 0) {
        globalOut.resize(numRows);
    }

    // Compute displacement and count arrays for MPI_Gatherv
    std::vector<int> recvCounts(numRanks);
    std::vector<int> displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        const index_t rStart = r * rowsPerRank + std::min(static_cast<index_t>(r), remainder);
        const index_t rEnd = (r + 1) * rowsPerRank + std::min(static_cast<index_t>(r + 1), remainder);
        recvCounts[r] = rEnd - rStart;
        displs[r] = rStart;
    }

    MPI_CHECK(MPI_Gatherv(localOut.data(), localNumRows, MPI_DOUBLE,
                          globalOut.data(), recvCounts.data(), displs.data(),
                          MPI_DOUBLE, 0, MPI_COMM_WORLD));

    // Print results from rank 0
    if (rank == 0) {
        const double globalDurationMs = globalDurationUs / 1000.0;
        printf("Computation time: %.3f ms (max across ranks)\n", globalDurationMs);

        const double gflops = (2.0 * nItems * iterations) / (globalDurationMs / 1000.0) / 1e9;
        const double avgTime = globalDurationMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(globalOut, "OutputVector");
        }

        // Validation against CPU reference
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), globalOut.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
