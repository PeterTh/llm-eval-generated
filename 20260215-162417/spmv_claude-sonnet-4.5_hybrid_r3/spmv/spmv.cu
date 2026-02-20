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

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// CUDA kernel for SpMV
__global__ void spmvKernel(const double* val, const index_t* cols, 
                           const index_t* rowDelimiters, const double* vec,
                           index_t numRows, double* out) {
    const index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < numRows) {
        double sum = 0.0;
        const index_t start = rowDelimiters[row];
        const index_t end = rowDelimiters[row + 1];
        
        for (index_t j = start; j < end; ++j) {
            sum += val[j] * vec[cols[j]];
        }
        
        out[row] = sum;
    }
}

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
    // Generate deterministically - same sequence regardless of parallelism
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
//   (OpenMP parallelized for speed)
//
// Arguments:
//   reference: array holding the reference result vector
//   result: array holding the result vector to verify
//   size: number of elements per vector
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    bool valid = true;
    
    #pragma omp parallel for reduction(&& : valid)
    for (index_t i = 0; i < size; ++i) {
        if (!valid) continue;
        
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                #pragma omp critical
                {
                    printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                    valid = false;
                }
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                #pragma omp critical
                {
                    printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                           i, ref, res, relError);
                    valid = false;
                }
            }
        }
    }
    return valid;
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

    // Parse command line arguments (rank 0)
    if (rank == 0) {
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
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Broadcast parameters to all ranks
    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Parallelization: Hybrid MPI+OpenMP+CUDA\n");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Distribute rows across MPI ranks
    std::vector<int> rowsPerRank(size);
    std::vector<int> rowOffsets(size);
    for (int i = 0; i < size; ++i) {
        rowsPerRank[i] = numRows / size + (i < static_cast<int>(numRows % size) ? 1 : 0);
        rowOffsets[i] = (i == 0) ? 0 : rowOffsets[i - 1] + rowsPerRank[i - 1];
    }
    
    const index_t localRows = rowsPerRank[rank];
    const index_t rowOffset = rowOffsets[rank];

    // Full data structures (rank 0 generates everything)
    std::vector<double> h_val_full;
    std::vector<index_t> h_cols_full;
    std::vector<index_t> h_rowDelimiters_full;
    std::vector<double> h_vec(numRows);  // All ranks need full vector
    std::vector<double> h_out_full;
    std::vector<double> h_reference;
    
    if (rank == 0) {
        printf("Initializing data structures...\n");
        h_val_full.resize(nItems);
        h_cols_full.resize(nItems);
        h_rowDelimiters_full.resize(numRows + 1);
        h_out_full.resize(numRows);
        
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val_full.data(), nItems, maxVal);
        initRandomMatrix(h_cols_full.data(), h_rowDelimiters_full.data(), nItems, numRows);
    }
    
    // Broadcast input vector to all ranks
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Broadcast rowDelimiters to determine local matrix sizes
    if (rank != 0) {
        h_rowDelimiters_full.resize(numRows + 1);
    }
    MPI_Bcast(h_rowDelimiters_full.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    
    // Calculate local matrix size
    const index_t localStart = h_rowDelimiters_full[rowOffset];
    const index_t localEnd = h_rowDelimiters_full[rowOffset + localRows];
    const index_t localNItems = localEnd - localStart;
    
    // Allocate local data structures
    std::vector<double> h_val_local(localNItems);
    std::vector<index_t> h_cols_local(localNItems);
    std::vector<index_t> h_rowDelimiters_local(localRows + 1);
    std::vector<double> h_out_local(localRows);
    
    // Scatter matrix data to ranks
    if (rank == 0) {
        // Copy rank 0's data
        std::copy(h_val_full.begin() + localStart, h_val_full.begin() + localEnd, h_val_local.begin());
        std::copy(h_cols_full.begin() + localStart, h_cols_full.begin() + localEnd, h_cols_local.begin());
        
        // Adjust row delimiters for rank 0
        for (index_t i = 0; i <= localRows; ++i) {
            h_rowDelimiters_local[i] = h_rowDelimiters_full[rowOffset + i] - localStart;
        }
        
        // Send to other ranks
        for (int r = 1; r < size; ++r) {
            const index_t rRowOffset = rowOffsets[r];
            const index_t rLocalRows = rowsPerRank[r];
            const index_t rLocalStart = h_rowDelimiters_full[rRowOffset];
            const index_t rLocalEnd = h_rowDelimiters_full[rRowOffset + rLocalRows];
            const index_t rLocalNItems = rLocalEnd - rLocalStart;
            
            MPI_Send(&h_val_full[rLocalStart], rLocalNItems, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
            MPI_Send(&h_cols_full[rLocalStart], rLocalNItems, MPI_UINT32_T, r, 1, MPI_COMM_WORLD);
            
            std::vector<index_t> tempRowDelim(rLocalRows + 1);
            for (index_t i = 0; i <= rLocalRows; ++i) {
                tempRowDelim[i] = h_rowDelimiters_full[rRowOffset + i] - rLocalStart;
            }
            MPI_Send(tempRowDelim.data(), rLocalRows + 1, MPI_UINT32_T, r, 2, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(h_val_local.data(), localNItems, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(h_cols_local.data(), localNItems, MPI_UINT32_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(h_rowDelimiters_local.data(), localRows + 1, MPI_UINT32_T, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Set CUDA device (one GPU per MPI rank)
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));
    
    // Allocate device memory
    double *d_val, *d_vec, *d_out;
    index_t *d_cols, *d_rowDelimiters;
    
    CUDA_CHECK(cudaMalloc(&d_val, localNItems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, localNItems * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, localRows * sizeof(double)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_val, h_val_local.data(), localNItems * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_cols_local.data(), localNItems * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, h_rowDelimiters_local.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));
    
    // For validation, compute reference solution on rank 0
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val_full.data(), h_cols_full.data(), h_rowDelimiters_full.data(), 
                h_vec.data(), numRows, h_reference.data());
    }
    
    // Warmup
    const int threadsPerBlock = 256;
    const int numBlocks = (localRows + threadsPerBlock - 1) / threadsPerBlock;
    spmvKernel<<<numBlocks, threadsPerBlock>>>(d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Barrier before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvKernel<<<numBlocks, threadsPerBlock>>>(d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out_local.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Gather results to rank 0
    std::vector<int> recvCounts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; ++i) {
        recvCounts[i] = rowsPerRank[i];
        displs[i] = rowOffsets[i];
    }
    
    if (rank == 0) {
        MPI_Gatherv(h_out_local.data(), localRows, MPI_DOUBLE,
                    h_out_full.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(h_out_local.data(), localRows, MPI_DOUBLE,
                    nullptr, nullptr, nullptr, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);
        
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(h_out_full, "OutputVector");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out_full.data(), numRows);

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
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));
    
    MPI_Finalize();
    return 0;
}
