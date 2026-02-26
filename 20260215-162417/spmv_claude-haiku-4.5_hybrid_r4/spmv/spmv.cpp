#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

#ifdef USE_CUDA
// CUDA kernel for sparse matrix-vector multiplication
__global__ void spmvCudaKernel(const double* val, const index_t* cols, 
                                const index_t* rowDelimiters, const double* vec,
                                const index_t dim, double* out, 
                                index_t rowStart, index_t rowEnd) {
    index_t i = rowStart + (index_t)blockIdx.x * (index_t)blockDim.x + (index_t)threadIdx.x;
    if (i < rowEnd) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}
#endif

// ****************************************************************************
// Function: spmvCpuParallel
//
// Purpose:
//   Hybrid parallel SpMV computation using MPI + OpenMP + CUDA
//   - MPI: distribute rows across processes
//   - OpenMP: parallelize row loop on each process
//   - CUDA: accelerate computation on GPU (if available)
//
// Arguments:
//   val: array holding the non-zero values
//   cols: array of column indices
//   rowDelimiters: array of row delimiters
//   vec: dense vector
//   localRowStart: starting row for this MPI rank
//   localRowEnd: ending row for this MPI rank
//   out: output vector
//
// ****************************************************************************
void spmvCpuParallel(const double* val, const index_t* cols, 
                     const index_t* rowDelimiters, const double* vec, 
                     index_t localRowStart, index_t localRowEnd, double* out) {
    #pragma omp parallel for default(none) shared(val, cols, rowDelimiters, vec, out, localRowStart, localRowEnd)
    for (index_t i = localRowStart; i < localRowEnd; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

#ifdef USE_CUDA
// ****************************************************************************
// Function: spmvGpu
//
// Purpose:
//   GPU-accelerated SpMV computation
//
// ****************************************************************************
void spmvGpu(const double* h_val, const index_t* h_cols,
             const index_t* h_rowDelimiters, const double* h_vec,
             const index_t dim, double* h_out,
             index_t localRowStart, index_t localRowEnd) {
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    
    const index_t nItems = h_rowDelimiters[dim];
    
    // Allocate GPU memory
    cudaMalloc(&d_val, nItems * sizeof(double));
    cudaMalloc(&d_cols, nItems * sizeof(index_t));
    cudaMalloc(&d_rowDelimiters, (dim + 1) * sizeof(index_t));
    cudaMalloc(&d_vec, dim * sizeof(double));
    cudaMalloc(&d_out, dim * sizeof(double));
    
    // Copy data to GPU
    cudaMemcpy(d_val, h_val, nItems * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, h_cols, nItems * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowDelimiters, h_rowDelimiters, (dim + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec, dim * sizeof(double), cudaMemcpyHostToDevice);
    
    // Compute grid and block sizes
    index_t rowCount = localRowEnd - localRowStart;
    int blockSize = 256;
    int gridSize = (rowCount + blockSize - 1) / blockSize;
    
    // Launch kernel
    spmvCudaKernel<<<gridSize, blockSize>>>(d_val, d_cols, d_rowDelimiters, d_vec,
                                             dim, d_out, localRowStart, localRowEnd);
    
    // Copy results back
    cudaMemcpy(h_out, d_out, dim * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_val);
    cudaFree(d_cols);
    cudaFree(d_rowDelimiters);
    cudaFree(d_vec);
    cudaFree(d_out);
}
#endif

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

    // Parse command line arguments (only on rank 0)
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
    
    // Broadcast parameters
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
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        #ifdef USE_CUDA
        printf("CUDA: enabled\n");
        #else
        printf("CUDA: disabled\n");
        #endif
    }

    // Allocate and initialize data structures
    std::vector<double> h_val(nItems);
    std::vector<index_t> h_cols(nItems);
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out(numRows, 0.0);

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast matrix data to all processes
    MPI_Bcast(h_val.data(), nItems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_cols.data(), nItems, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute row distribution for this MPI rank
    index_t rowsPerProcess = numRows / size;
    index_t extraRows = numRows % size;
    index_t localRowStart = rank * rowsPerProcess + std::min(static_cast<index_t>(rank), extraRows);
    index_t localRowEnd = localRowStart + rowsPerProcess + (rank < static_cast<int>(extraRows) ? 1 : 0);

    // For validation, compute reference solution (only on rank 0)
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        #ifdef USE_CUDA
        spmvGpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_out.data(), localRowStart, localRowEnd);
        #else
        spmvCpuParallel(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                        h_vec.data(), localRowStart, localRowEnd, h_out.data());
        #endif
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather all local output vectors to rank 0
    std::vector<double> h_out_gathered(numRows, 0.0);
    std::vector<int> recvCounts(size);
    std::vector<int> displs(size);
    
    for (int i = 0; i < size; ++i) {
        index_t iRowsPerProcess = numRows / size;
        index_t iExtraRows = numRows % size;
        recvCounts[i] = iRowsPerProcess + (i < static_cast<int>(iExtraRows) ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i-1] + recvCounts[i-1];
    }
    
    MPI_Gatherv(h_out.data() + localRowStart, localRowEnd - localRowStart, MPI_DOUBLE,
                h_out_gathered.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        h_out = h_out_gathered;
        
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
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
