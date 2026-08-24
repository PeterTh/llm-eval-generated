#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

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
// CUDA kernel for SpMV
// ****************************************************************************
__global__ void spmvKernel(const double* val, const index_t* cols, 
                           const index_t* rowDelimiters, const double* vec,
                           const index_t dim, double* out) {
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const index_t col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
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
    #pragma omp parallel for schedule(dynamic, 32)
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
    bool printResults = false;

    // Parse command line arguments
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

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI processes: %d\n", size);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate and initialize data structures on rank 0
    std::vector<double> h_val(nItems);
    std::vector<index_t> h_cols(nItems);
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out_global(numRows);

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast vector to all ranks
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_cols.data(), nItems, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_val.data(), nItems, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Partition rows among MPI ranks
    index_t rowsPerRank = (numRows + size - 1) / size;
    index_t startRow = rank * rowsPerRank;
    index_t endRow = std::min(startRow + rowsPerRank, numRows);
    index_t localRows = endRow - startRow;

    // Compute local non-zero range
    index_t localStartNZ = h_rowDelimiters[startRow];
    index_t localEndNZ = h_rowDelimiters[endRow];
    index_t localNZ = localEndNZ - localStartNZ;

    // Allocate local data
    std::vector<double> h_out_local(localRows);
    std::vector<index_t> h_localRowDelimiters(localRows + 1);
    
    // Adjust row delimiters for local partition
    for (index_t i = 0; i <= localRows; ++i) {
        h_localRowDelimiters[i] = h_rowDelimiters[startRow + i] - localStartNZ;
    }

    // For validation, compute reference solution on rank 0
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Check CUDA availability
    int deviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    bool useCuda = (cudaStatus == cudaSuccess && deviceCount > 0);

    // Select GPU based on rank
    if (useCuda) {
        int deviceId = rank % deviceCount;
        CUDA_CHECK(cudaSetDevice(deviceId));
        if (rank == 0) {
            printf("Using CUDA with %d GPU(s)\n", deviceCount);
        }
    } else if (rank == 0) {
        printf("CUDA not available, using CPU with OpenMP\n");
    }

    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rowDelimiters = nullptr;

    if (useCuda) {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_val, localNZ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, localNZ * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (localRows + 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_out, localRows * sizeof(double)));

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_val, h_val.data() + localStartNZ, localNZ * sizeof(double), 
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data() + localStartNZ, localNZ * sizeof(index_t), 
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rowDelimiters, h_localRowDelimiters.data(), 
                              (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), 
                              cudaMemcpyHostToDevice));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (useCuda) {
            const int blockSize = 256;
            const int numBlocks = (localRows + blockSize - 1) / blockSize;
            spmvKernel<<<numBlocks, blockSize>>>(d_val, d_cols, d_rowDelimiters,
                                                  d_vec, localRows, d_out);
            CUDA_CHECK(cudaGetLastError());
        } else {
            spmvCpu(h_val.data() + localStartNZ, h_cols.data() + localStartNZ, 
                    h_localRowDelimiters.data(), h_vec.data(), localRows, 
                    h_out_local.data());
        }
    }

    if (useCuda) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long localDurationMs = static_cast<long long>(duration.count());
    long long globalDurationMs = 0;
    MPI_Reduce(&localDurationMs, &globalDurationMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy results back from device if using CUDA
    if (useCuda) {
        CUDA_CHECK(cudaMemcpy(h_out_local.data(), d_out, localRows * sizeof(double), 
                              cudaMemcpyDeviceToHost));
    }

    // Gather results to rank 0
    std::vector<int> recvCounts(size);
    std::vector<int> displs(size);
    int localCount = static_cast<int>(localRows);
    
    MPI_Gather(&localCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i - 1] + recvCounts[i - 1];
        }
    }
    
    MPI_Gatherv(h_out_local.data(), localCount, MPI_DOUBLE,
                h_out_global.data(), recvCounts.data(), displs.data(), 
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", globalDurationMs);
        
        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (globalDurationMs / 1000.0) / 1e9;
        const double avgTime = globalDurationMs / static_cast<double>(iterations);
        
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(h_out_global, "OutputVector");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out_global.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Cleanup
    if (useCuda) {
        cudaFree(d_val);
        cudaFree(d_cols);
        cudaFree(d_rowDelimiters);
        cudaFree(d_vec);
        cudaFree(d_out);
    }

    MPI_Finalize();
    return 0;
}
