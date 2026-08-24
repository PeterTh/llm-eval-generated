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
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// CUDA kernel for SpMV
__global__ void spmvKernel(const double* val, const index_t* cols, const index_t* rowDelimiters,
                           const double* vec, const index_t numRows, double* out) {
    index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
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
    // Use fixed seed for reproducibility across ranks if needed, but here we parallelize
    // We can't easily parallelize the exact sequence of rand(), so we'll use a local approach
    // that approximates the distribution or just have rank 0 generate and scatter.
    // Given the constraints, rank 0 generating is the safest for exact reproducibility of the specific matrix
    // unless we change the generation algorithm.
    // The prompt says "maintaining correctness and equivalent semantics".
    // "Equivalent semantics" -> same matrix? Or just a random sparse matrix?
    // "Correctness" -> y = Ax.
    // I will stick to Rank 0 generates, then scatters.
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
    bool passed = true;
    #pragma omp parallel for reduction(&:passed)
    for (index_t i = 0; i < size; ++i) {
        if (!passed) continue;
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                #pragma omp critical
                {
                    printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                    passed = false;
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
                    passed = false;
                }
            }
        }
    }
    return passed;
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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments on rank 0
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

    // Broadcast arguments
    MPI_Bcast(&numRows, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int flags[2] = {validate ? 1 : 0, printResults ? 1 : 0};
    MPI_Bcast(flags, 2, MPI_INT, 0, MPI_COMM_WORLD);
    validate = flags[0];
    printResults = flags[1];

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
        printf("MPI Ranks: %d\n", size);
        printf("Initializing data structures...\n");
    }

    // Host data structures
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out(numRows);
    std::vector<double> h_reference;

    // Only rank 0 allocates full matrix
    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);

        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    // Broadcast input vector
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calculate local rows
    index_t rowsPerRank = numRows / static_cast<index_t>(size);
    index_t remainder = numRows % static_cast<index_t>(size);
    index_t localRows = rowsPerRank + (static_cast<index_t>(rank) < remainder ? 1 : 0);
    index_t startRow = static_cast<index_t>(rank) * rowsPerRank + (static_cast<index_t>(rank) < remainder ? static_cast<index_t>(rank) : remainder);

    // Distribute rowDelimiters
    // We need the full rowDelimiters array on all ranks to easily determine NNZ counts for scatter
    // Or we can scatter rowDelimiters first.
    // Let's broadcast it for simplicity as analyzed before.
    if (rank != 0) h_rowDelimiters.resize(numRows + 1);
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Determine local NNZ
    index_t local_nnz_start = h_rowDelimiters[startRow];
    index_t local_nnz_end = h_rowDelimiters[startRow + localRows];
    index_t local_nnz_count = local_nnz_end - local_nnz_start;

    // Allocate local arrays
    std::vector<double> local_val(local_nnz_count);
    std::vector<index_t> local_cols(local_nnz_count);

    // Prepare Scatterv counts
    std::vector<int> nnz_counts(size);
    std::vector<int> nnz_displs(size);
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            index_t r_rows = rowsPerRank + (static_cast<index_t>(i) < remainder ? 1 : 0);
            index_t r_start = static_cast<index_t>(i) * rowsPerRank + (static_cast<index_t>(i) < remainder ? static_cast<index_t>(i) : remainder);
            nnz_counts[i] = h_rowDelimiters[r_start + r_rows] - h_rowDelimiters[r_start];
            nnz_displs[i] = h_rowDelimiters[r_start];
        }
    }

    // Scatter val and cols
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, nnz_counts.data(), nnz_displs.data(), MPI_DOUBLE,
                 local_val.data(), local_nnz_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, nnz_counts.data(), nnz_displs.data(), MPI_UNSIGNED,
                 local_cols.data(), local_nnz_count, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Adjust row delimiters for device
    std::vector<index_t> device_rowDelimiters(localRows + 1);
    for (index_t i = 0; i <= localRows; ++i) {
        device_rowDelimiters[i] = h_rowDelimiters[startRow + i] - local_nnz_start;
    }

    // CUDA
    double *d_val, *d_vec, *d_out;
    index_t *d_cols, *d_rowDelimiters;

    CUDA_CHECK(cudaMalloc(&d_val, local_nnz_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, local_nnz_count * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, localRows * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_val, local_val.data(), local_nnz_count * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, local_cols.data(), local_nnz_count * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, device_rowDelimiters.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    if (rank == 0) printf("Computing SpMV...\n");

    // Warmup
    int blockSize = 256;
    int numBlocks = (localRows + blockSize - 1) / blockSize;
    if (localRows > 0)
        spmvKernel<<<numBlocks, blockSize>>>(d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows > 0)
            spmvKernel<<<numBlocks, blockSize>>>(d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = duration.count();
    long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Get results back
    std::vector<double> local_out(localRows);
    CUDA_CHECK(cudaMemcpy(local_out.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost));

    // Gather results
    std::vector<int> recvcounts(size);
    std::vector<int> rdispls(size);
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = rowsPerRank + (static_cast<index_t>(i) < remainder ? 1 : 0);
            rdispls[i] = static_cast<index_t>(i) * rowsPerRank + (static_cast<index_t>(i) < remainder ? static_cast<index_t>(i) : remainder);
        }
    }

    MPI_Gatherv(local_out.data(), localRows, MPI_DOUBLE,
                h_out.data(), recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_duration_ms);
        
        const double gflops = (2.0 * nItems * iterations) / (global_duration_ms / 1000.0) / 1e9;
        const double avgTime = global_duration_ms / static_cast<double>(iterations);
        
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) printf("Validation: PASSED\n");
            else printf("Validation: FAILED\n");
        }
    }

    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    MPI_Finalize();
    return 0;
}
