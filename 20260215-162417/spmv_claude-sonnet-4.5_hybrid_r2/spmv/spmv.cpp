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

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)

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
// CUDA kernel: spmvKernel
//
// Purpose:
//   GPU kernel for sparse matrix-vector multiplication using CSR format
//
// ****************************************************************************
__global__ void spmvKernel(const double* val, const index_t* cols, 
                           const index_t* rowDelimiters, const double* vec,
                           const index_t localRows, const index_t rowOffset, double* out) {
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < localRows) {
        double t = 0.0;
        const index_t rowStart = rowDelimiters[i];
        const index_t rowEnd = rowDelimiters[i + 1];
        for (index_t j = rowStart; j < rowEnd; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format with OpenMP
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
    
    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    
    // Set GPU device based on local rank
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    int deviceId = mpiRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(deviceId));
    
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only rank 0 prints)
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
            if (mpiRank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (mpiRank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d\n", mpiSize);
        printf("GPU devices: %d\n", deviceCount);
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
    std::vector<double> h_out(numRows);

    if (mpiRank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast full vector to all ranks
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Determine local row range for each rank
    const index_t rowsPerRank = (numRows + mpiSize - 1) / mpiSize;
    const index_t localRowStart = mpiRank * rowsPerRank;
    const index_t localRowEnd = std::min(localRowStart + rowsPerRank, numRows);
    const index_t localRows = localRowEnd - localRowStart;

    // Broadcast row delimiters to all ranks to determine local nnz
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Calculate local non-zero count
    const index_t localNnzStart = h_rowDelimiters[localRowStart];
    const index_t localNnzEnd = h_rowDelimiters[localRowEnd];
    const index_t localNnz = localNnzEnd - localNnzStart;

    // Allocate local data structures
    std::vector<double> local_val(localNnz);
    std::vector<index_t> local_cols(localNnz);
    std::vector<index_t> local_rowDelimiters(localRows + 1);
    std::vector<double> local_out(localRows);

    // Scatter matrix data
    if (mpiRank == 0) {
        // Copy rank 0's portion
        if (localNnz > 0) {
            std::copy(h_val.begin() + localNnzStart, h_val.begin() + localNnzEnd, local_val.begin());
            std::copy(h_cols.begin() + localNnzStart, h_cols.begin() + localNnzEnd, local_cols.begin());
        }
        for (index_t i = 0; i <= localRows; ++i) {
            local_rowDelimiters[i] = h_rowDelimiters[localRowStart + i] - localNnzStart;
        }

        // Send to other ranks
        for (int r = 1; r < mpiSize; ++r) {
            const index_t rRowStart = r * rowsPerRank;
            const index_t rRowEnd = std::min(rRowStart + rowsPerRank, numRows);
            const index_t rLocalRows = rRowEnd - rRowStart;
            const index_t rNnzStart = h_rowDelimiters[rRowStart];
            const index_t rNnzEnd = h_rowDelimiters[rRowEnd];
            const index_t rLocalNnz = rNnzEnd - rNnzStart;

            if (rLocalNnz > 0) {
                MPI_Send(&h_val[rNnzStart], rLocalNnz, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
                MPI_Send(&h_cols[rNnzStart], rLocalNnz, MPI_UNSIGNED, r, 1, MPI_COMM_WORLD);
            }
            
            std::vector<index_t> rRowDel(rLocalRows + 1);
            for (index_t i = 0; i <= rLocalRows; ++i) {
                rRowDel[i] = h_rowDelimiters[rRowStart + i] - rNnzStart;
            }
            MPI_Send(rRowDel.data(), rLocalRows + 1, MPI_UNSIGNED, r, 2, MPI_COMM_WORLD);
        }
    } else {
        // Receive data
        if (localNnz > 0) {
            MPI_Recv(local_val.data(), localNnz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(local_cols.data(), localNnz, MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Recv(local_rowDelimiters.data(), localRows + 1, MPI_UNSIGNED, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // For validation, compute reference solution on rank 0
    std::vector<double> h_reference;
    if (validate && mpiRank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Allocate device memory
    double *d_val, *d_vec, *d_out;
    index_t *d_cols, *d_rowDelimiters;

    CUDA_CHECK(cudaMalloc(&d_val, localNnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, localNnz * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, localRows * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_val, local_val.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, local_cols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, local_rowDelimiters.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    // Warmup
    const int threadsPerBlock = 256;
    const int numBlocks = (localRows + threadsPerBlock - 1) / threadsPerBlock;
    spmvKernel<<<numBlocks, threadsPerBlock>>>(d_val, d_cols, d_rowDelimiters, d_vec, localRows, localRowStart, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Barrier before timing
    MPI_Barrier(MPI_COMM_WORLD);

    if (mpiRank == 0) {
        printf("Computing SpMV...\n");
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvKernel<<<numBlocks, threadsPerBlock>>>(d_val, d_cols, d_rowDelimiters, d_vec, localRows, localRowStart, d_out);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long long localDurationMs = duration.count();
    long long globalDurationMs = 0;
    MPI_Reduce(&localDurationMs, &globalDurationMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(local_out.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost));

    // Gather results at rank 0
    std::vector<int> recvCounts(mpiSize);
    std::vector<int> displs(mpiSize);
    
    for (int r = 0; r < mpiSize; ++r) {
        const index_t rRowStart = r * rowsPerRank;
        const index_t rRowEnd = std::min(rRowStart + rowsPerRank, numRows);
        recvCounts[r] = rRowEnd - rRowStart;
        displs[r] = rRowStart;
    }

    MPI_Gatherv(local_out.data(), localRows, MPI_DOUBLE,
                h_out.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (mpiRank == 0) {
        printf("Computation time: %lld ms\n", globalDurationMs);
        
        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (globalDurationMs / 1000.0) / 1e9;
        const double avgTime = globalDurationMs / static_cast<double>(iterations);
        
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
                CUDA_CHECK(cudaFree(d_val));
                CUDA_CHECK(cudaFree(d_cols));
                CUDA_CHECK(cudaFree(d_rowDelimiters));
                CUDA_CHECK(cudaFree(d_vec));
                CUDA_CHECK(cudaFree(d_out));
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
