#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <omp.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)
#endif

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
    #pragma omp parallel for
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

#ifdef USE_CUDA
// ****************************************************************************
// CUDA Kernel: spmvKernel
//
// Purpose:
//   CUDA kernel for sparse matrix-vector multiplication using CSR format
//
// ****************************************************************************
__global__ void spmvKernel(const double* __restrict__ val, 
                           const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters, 
                           const double* __restrict__ vec,
                           const index_t dim, 
                           double* __restrict__ out) {
    const index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < dim) {
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
// Function: spmvGpu
//
// Purpose:
//   Wrapper function for GPU SpMV computation
//
// ****************************************************************************
void spmvGpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out,
             double* d_val, index_t* d_cols, index_t* d_rowDelimiters,
             double* d_vec, double* d_out, const index_t nItems) {
    
    const int blockSize = 256;
    const int gridSize = (dim + blockSize - 1) / blockSize;
    
    spmvKernel<<<gridSize, blockSize>>>(d_val, d_cols, d_rowDelimiters, d_vec, dim, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
#endif

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
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
#ifdef USE_CUDA
    // Set GPU device based on local rank
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices > 0) {
        int device = mpi_rank % num_devices;
        CUDA_CHECK(cudaSetDevice(device));
    } else {
        if (mpi_rank == 0) {
            fprintf(stderr, "No CUDA devices available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#endif
    
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
            if (mpi_rank == 0) {
                printUsage(argv[0]);
            }
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

    // Distribute rows across MPI ranks
    const index_t rowsPerRank = (numRows + mpi_size - 1) / mpi_size;
    const index_t localStartRow = mpi_rank * rowsPerRank;
    const index_t localEndRow = std::min(localStartRow + rowsPerRank, numRows);
    const index_t localNumRows = (localStartRow < numRows) ? (localEndRow - localStartRow) : 0;

    if (mpi_rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Parallelization: MPI + OpenMP");
#ifdef USE_CUDA
        printf(" + CUDA\n");
#else
        printf("\n");
#endif
        printf("MPI ranks: %d\n", mpi_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
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

    if (mpi_rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast data to all ranks
    MPI_Bcast(h_val.data(), nItems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_cols.data(), nItems, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Extract local portion of matrix
    std::vector<double> local_val;
    std::vector<index_t> local_cols;
    std::vector<index_t> local_rowDelimiters(localNumRows + 1);
    
    if (localNumRows > 0) {
        const index_t localStart = h_rowDelimiters[localStartRow];
        const index_t localEnd = h_rowDelimiters[localEndRow];
        const index_t localNItems = localEnd - localStart;
        
        local_val.resize(localNItems);
        local_cols.resize(localNItems);
        
        std::copy(h_val.begin() + localStart, h_val.begin() + localEnd, local_val.begin());
        std::copy(h_cols.begin() + localStart, h_cols.begin() + localEnd, local_cols.begin());
        
        for (index_t i = 0; i <= localNumRows; ++i) {
            local_rowDelimiters[i] = h_rowDelimiters[localStartRow + i] - localStart;
        }
    }

#ifdef USE_CUDA
    // Allocate GPU memory
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    
    if (localNumRows > 0) {
        const index_t localNItems = local_val.size();
        CUDA_CHECK(cudaMalloc(&d_val, localNItems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, localNItems * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (localNumRows + 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_out, localNumRows * sizeof(double)));
        
        // Copy data to GPU
        CUDA_CHECK(cudaMemcpy(d_val, local_val.data(), localNItems * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, local_cols.data(), localNItems * sizeof(index_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rowDelimiters, local_rowDelimiters.data(), (localNumRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));
    }
#endif

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && mpi_rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

#ifdef USE_CUDA
    // Warm-up run
    if (localNumRows > 0) {
        spmvGpu(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                h_vec.data(), localNumRows, nullptr,
                d_val, d_cols, d_rowDelimiters, d_vec, d_out, local_val.size());
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);

    // Perform SpMV computation
    if (mpi_rank == 0) {
        printf("Computing SpMV...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
#ifdef USE_CUDA
        if (localNumRows > 0) {
            spmvGpu(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                    h_vec.data(), localNumRows, nullptr,
                    d_val, d_cols, d_rowDelimiters, d_vec, d_out, local_val.size());
        }
#else
        // CPU-only path with OpenMP
        if (localNumRows > 0) {
            spmvCpu(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                    h_vec.data(), localNumRows, h_out.data() + localStartRow);
        }
#endif
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = duration.count();
    long long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy results back
    std::vector<double> local_out(localNumRows);
#ifdef USE_CUDA
    if (localNumRows > 0) {
        CUDA_CHECK(cudaMemcpy(local_out.data(), d_out, localNumRows * sizeof(double), cudaMemcpyDeviceToHost));
    }
#else
    if (localNumRows > 0) {
        std::copy(h_out.data() + localStartRow, h_out.data() + localEndRow, local_out.begin());
    }
#endif

    // Gather results to rank 0
    std::vector<int> recvcounts(mpi_size);
    std::vector<int> displs(mpi_size);
    
    int local_count = static_cast<int>(localNumRows);
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < mpi_size; ++i) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }
    
    MPI_Gatherv(local_out.data(), local_count, MPI_DOUBLE,
                h_out.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("Computation time: %lld ms\n", global_duration_ms);
        
        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (global_duration_ms / 1000.0) / 1e9;
        const double avgTime = global_duration_ms / static_cast<double>(iterations);
        
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
                MPI_Finalize();
                return 1;
            }
        }
    }

#ifdef USE_CUDA
    // Clean up GPU memory
    if (localNumRows > 0) {
        CUDA_CHECK(cudaFree(d_val));
        CUDA_CHECK(cudaFree(d_cols));
        CUDA_CHECK(cudaFree(d_rowDelimiters));
        CUDA_CHECK(cudaFree(d_vec));
        CUDA_CHECK(cudaFree(d_out));
    }
#endif

    MPI_Finalize();
    return 0;
}
