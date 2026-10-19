#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

__global__ void spmvKernel(const double* val, const index_t* cols,
                           const index_t* rows, const double* vec,
                           index_t count, double* out) {
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        double sum = 0.0;
        for (index_t j = rows[i]; j < rows[i + 1]; ++j)
            sum += val[j] * vec[cols[j]];
        out[i] = sum;
    }
}

template <typename T>
static T* deviceAlloc(size_t count) {
    T* ptr = nullptr;
    if (count && cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)) != cudaSuccess) {
        fprintf(stderr, "CUDA allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return ptr;
}

static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

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
    int failed = 0;
    #pragma omp parallel for reduction(|:failed) schedule(static)
    for (int i = 0; i < static_cast<int>(size); ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                failed = 1;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                failed = 1;
            }
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
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cudaCheck(cudaSetDevice(rank % deviceCount));
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (sparsity == 0) { if (rank == 0) fprintf(stderr, "Sparsity must be positive\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (rank == 0) printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    if (rank == 0) printf("Matrix size: %u x %u\n", numRows, numRows);
    if (rank == 0) printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    if (rank == 0) printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
    if (rank == 0) printf("Iterations: %u\n", iterations);
    if (rank == 0) printf("Max value: %.2f\n", maxVal);
    if (rank == 0) printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate and initialize data structures
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out(numRows);                 // Output vector

    if (rank == 0) printf("Initializing data structures...\n");
    fill(h_vec.data(), numRows, maxVal);
    fill(h_val.data(), nItems, maxVal);
    initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    const index_t rowBegin = static_cast<index_t>((static_cast<uint64_t>(numRows) * rank) / world);
    const index_t rowEnd = static_cast<index_t>((static_cast<uint64_t>(numRows) * (rank + 1)) / world);
    const index_t nnzBegin = h_rowDelimiters[rowBegin];
    const index_t nnzEnd = h_rowDelimiters[rowEnd];
    const index_t localRows = rowEnd - rowBegin;
    const index_t localNnz = nnzEnd - nnzBegin;
    std::vector<index_t> localRowsPtr(localRows + 1);
    for (index_t i = 0; i <= localRows; ++i) localRowsPtr[i] = h_rowDelimiters[rowBegin + i] - nnzBegin;
    double *d_val = deviceAlloc<double>(localNnz), *d_vec = deviceAlloc<double>(numRows), *d_out = deviceAlloc<double>(localRows);
    index_t *d_cols = deviceAlloc<index_t>(localNnz), *d_rows = deviceAlloc<index_t>(localRows + 1);
    if (localNnz) {
        cudaCheck(cudaMemcpy(d_val, h_val.data() + nnzBegin, localNnz * sizeof(double), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_cols, h_cols.data() + nnzBegin, localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    cudaCheck(cudaMemcpy(d_rows, localRowsPtr.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> localOut(localRows);
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows) spmvKernel<<<(localRows + 255) / 256, 256>>>(d_val, d_cols, d_rows, d_vec, localRows, d_out);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());
    if (localRows) cudaCheck(cudaMemcpy(localOut.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost));
    const int localCount = static_cast<int>(localRows);
    std::vector<int> recvCounts(world), displs(world);
    for (int r = 0; r < world; ++r) { recvCounts[r] = static_cast<int>((static_cast<uint64_t>(numRows) * (r + 1)) / world - (static_cast<uint64_t>(numRows) * r) / world); displs[r] = static_cast<int>((static_cast<uint64_t>(numRows) * r) / world); }
    MPI_Allgatherv(localOut.data(), localCount, MPI_DOUBLE, h_out.data(), recvCounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count(), maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration<double>(maxElapsed));

    if (rank == 0) printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    const double seconds = maxElapsed;
    const double gflops = seconds > 0 ? (2.0 * nItems * iterations) / seconds / 1e9 : 0.0;
    const double avgTime = seconds * 1000.0 / iterations;
    
    if (rank == 0) printf("Average time per iteration: %.3f ms\n", avgTime);
    if (rank == 0) printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(h_out, "OutputVector");
    }

    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
            cudaFree(d_val); cudaFree(d_cols); cudaFree(d_rows); cudaFree(d_vec); cudaFree(d_out); MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            cudaFree(d_val); cudaFree(d_cols); cudaFree(d_rows); cudaFree(d_vec); cudaFree(d_out); MPI_Finalize(); return 1;
        }
    }

    cudaFree(d_val); cudaFree(d_cols); cudaFree(d_rows); cudaFree(d_vec); cudaFree(d_out);
    MPI_Finalize();
    return 0;
}
