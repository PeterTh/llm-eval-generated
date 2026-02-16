#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>
#ifdef __CUDACC__
#include <cuda_runtime.h>
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
    #pragma omp parallel for
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

#ifdef __CUDACC__
extern "C" __global__ void spmvCudaKernel(const double* val, const index_t* cols, const index_t* rowDelimiters, const double* vec, index_t dim, double* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[i] = t;
    }
}

void spmvCuda(const double* h_val, const index_t* h_cols, const index_t* h_rowDelimiters, const double* h_vec, index_t dim, double* h_out, index_t nnz) {
    double *d_val, *d_vec, *d_out;
    index_t *d_cols, *d_rowDelimiters;
    cudaMalloc(&d_val, nnz * sizeof(double));
    cudaMalloc(&d_cols, nnz * sizeof(index_t));
    cudaMalloc(&d_rowDelimiters, (dim + 1) * sizeof(index_t));
    cudaMalloc(&d_vec, dim * sizeof(double));
    cudaMalloc(&d_out, dim * sizeof(double));
    cudaMemcpy(d_val, h_val, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, h_cols, nnz * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowDelimiters, h_rowDelimiters, (dim + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec, dim * sizeof(double), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (dim + blockSize - 1) / blockSize;
    spmvCudaKernel<<<numBlocks, blockSize>>>(d_val, d_cols, d_rowDelimiters, d_vec, dim, d_out);
    cudaMemcpy(h_out, d_out, dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_val); cudaFree(d_cols); cudaFree(d_rowDelimiters); cudaFree(d_vec); cudaFree(d_out);
}
#else
void spmvCuda(const double*, const index_t*, const index_t*, const double*, index_t, double*, index_t) {
    // Dummy for non-CUDA builds
}
#endif

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
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
    bool use_gpu = true; // Always use CUDA for local SpMV

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
            MPI_Finalize();
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;
    index_t local_numRows = numRows / world_size;
    index_t local_row_start = world_rank * local_numRows;
    index_t local_row_end = (world_rank == world_size - 1) ? numRows : (world_rank + 1) * local_numRows;
    local_numRows = local_row_end - local_row_start;

    // Calculate local nnz (approximate)
    index_t local_nItems = (nItems * local_numRows) / numRows;

    if (world_rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", world_size);
    }

    // Allocate and initialize data structures
    std::vector<double> h_val(local_nItems);                  // Local non-zero values
    std::vector<index_t> h_cols(local_nItems);                // Local column indices
    std::vector<index_t> h_rowDelimiters(local_numRows + 1);  // Local row delimiters
    std::vector<double> h_vec(numRows);                       // Full vector (broadcasted)
    std::vector<double> h_out(local_numRows);                 // Local output vector

    // Only rank 0 initializes the full matrix/vector
    std::vector<double> full_val, full_vec;
    std::vector<index_t> full_cols, full_rowDelimiters;
    if (world_rank == 0) {
        full_val.resize(nItems);
        full_cols.resize(nItems);
        full_rowDelimiters.resize(numRows + 1);
        full_vec.resize(numRows);
        fill(full_vec.data(), numRows, maxVal);
        fill(full_val.data(), nItems, maxVal);
        initRandomMatrix(full_cols.data(), full_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast vector to all ranks
    if (world_rank == 0) memcpy(h_vec.data(), full_vec.data(), numRows * sizeof(double));
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter CSR matrix to all ranks (manual scatter for CSR)
    if (world_rank == 0) {
        for (int r = 0; r < world_size; ++r) {
            index_t r_start = r * (numRows / world_size);
            index_t r_end = (r == world_size - 1) ? numRows : (r + 1) * (numRows / world_size);
            index_t r_rows = r_end - r_start;
            index_t r_nnz = (nItems * r_rows) / numRows;
            if (r == 0) {
                memcpy(h_val.data(), full_val.data(), r_nnz * sizeof(double));
                memcpy(h_cols.data(), full_cols.data(), r_nnz * sizeof(index_t));
                memcpy(h_rowDelimiters.data(), full_rowDelimiters.data() + r_start, (r_rows + 1) * sizeof(index_t));
                // Adjust rowDelimiters to local base
                index_t base = h_rowDelimiters[0];
                for (index_t i = 0; i < h_rowDelimiters.size(); ++i) h_rowDelimiters[i] -= base;
            } else {
                MPI_Send(full_val.data() + ((nItems * r_start) / numRows), r_nnz, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
                MPI_Send(full_cols.data() + ((nItems * r_start) / numRows), r_nnz, MPI_UNSIGNED, r, 1, MPI_COMM_WORLD);
                MPI_Send(full_rowDelimiters.data() + r_start, r_rows + 1, MPI_UNSIGNED, r, 2, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(h_val.data(), local_nItems, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(h_cols.data(), local_nItems, MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(h_rowDelimiters.data(), local_numRows + 1, MPI_UNSIGNED, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Adjust rowDelimiters to local base
        index_t base = h_rowDelimiters[0];
        for (index_t i = 0; i < h_rowDelimiters.size(); ++i) h_rowDelimiters[i] -= base;
    }

    // For validation, compute reference solution (on rank 0)
    std::vector<double> h_reference;
    if (validate && world_rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(full_val.data(), full_cols.data(), full_rowDelimiters.data(), 
                full_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation
    if (world_rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (use_gpu) {
            spmvCuda(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), local_numRows, h_out.data(), local_nItems);
        } else {
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), local_numRows, h_out.data());
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather results to rank 0
    std::vector<double> full_out;
    if (world_rank == 0) full_out.resize(numRows);
    std::vector<int> recvcounts(world_size), displs(world_size);
    for (int r = 0; r < world_size; ++r) {
        index_t r_rows = (r == world_size - 1) ? (numRows - r * (numRows / world_size)) : (numRows / world_size);
        recvcounts[r] = r_rows;
        displs[r] = r * (numRows / world_size);
    }
    MPI_Gatherv(h_out.data(), local_numRows, MPI_DOUBLE, full_out.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
        // Print results for external validation
        if (printResults) {
            print_results(full_out, "OutputVector");
        }
        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), full_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    MPI_Finalize();
    return 0;
}
