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
#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    }

// ****************************************************************************
// CUDA Kernel: spmvKernel
// ****************************************************************************
__global__ void spmvKernel(const double* val, const index_t* cols, const index_t* rowDelimiters,
                           const double* vec, const index_t num_rows, double* out) {
    index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        double t = 0.0;
        index_t start = rowDelimiters[row];
        index_t end = rowDelimiters[row + 1];
        for (index_t j = start; j < end; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[row] = t;
    }
}

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
    #pragma omp parallel for
    for (index_t i = 0; i < n; ++i) {
        // Simple thread-safe random logic or just fill deterministically per index
        // Since original used rand(), we use a simple hash to be parallel-safe and deterministic
        // A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
        unsigned int seed = i;
        A[i] = maxVal * (static_cast<double>(rand_r(&seed)) / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: initRandomMatrixDistributed
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix. Generates only the local portion for this MPI rank.
//
// Arguments:
//   cols:          output vector for column indexes of local elements
//   val:           output vector for values of local elements
//   rowDelimiters: output vector for row delimiters (size local_rows + 1)
//   n:             total number of nonzero elements globally
//   dim:           global dimension of the matrix
//   rank:          MPI rank
//   size:          MPI size
//   maxVal:        max value for elements
//
// Returns:
//   Number of local non-zero elements
// ****************************************************************************
index_t initRandomMatrixDistributed(std::vector<index_t>& cols, std::vector<double>& val, 
                                   std::vector<index_t>& rowDelimiters,
                                   const index_t n, const index_t dim, 
                                   int rank, int size, double maxVal) {
    
    index_t rows_per_rank = dim / size;
    index_t start_row = rank * rows_per_rank;
    index_t end_row = (rank == size - 1) ? dim : (rank + 1) * rows_per_rank;
    index_t local_rows = end_row - start_row;

    rowDelimiters.resize(local_rows + 1);

    index_t nnzAssigned = 0;
    index_t local_nnz = 0;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator - must be same on all ranks!
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    
    // Iterate through ALL rows globally to maintain RNG state consistency
    // But only store data for local rows
    for (index_t i = 0; i < dim; ++i) {
        bool is_local = (i >= start_row && i < end_row);
        if (is_local) {
            rowDelimiters[i - start_row] = local_nnz;
        }

        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                if (is_local) {
                    cols.push_back(j);
                    // Use consistent random generation for values based on global index
                    unsigned int seed = nnzAssigned; 
                    val.push_back(maxVal * (static_cast<double>(rand_r(&seed)) / (static_cast<double>(RAND_MAX) + 1.0)));
                    local_nnz++;
                }
                nnzAssigned++;
            }
        }
    }
    
    if (rank == size - 1) {
         // Last rank might process up to dim, so rowDelimiters needs correct size
         // Already handled by loop condition and resizing
    }

    // Convention: put the number of non-zeroes at the end of the row delimiters array
    rowDelimiters[local_rows] = local_nnz;
    
    return local_nnz;
}

// ****************************************************************************
// Function: spmvCpu (Validation only)
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t num_rows, double* out) {
    #pragma omp parallel for
    for (index_t i = 0; i < num_rows; ++i) {
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
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                return false;
            }
        } else {
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

    // Determine local rows
    index_t rows_per_rank = numRows / size;
    index_t start_row = rank * rows_per_rank;
    index_t end_row = (rank == size - 1) ? numRows : (rank + 1) * rows_per_rank;
    index_t local_rows = end_row - start_row;

    // Calculate global number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("MPI Ranks: %d\n", size);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate and initialize data structures
    std::vector<double> h_val;                  // Local Non-zero values
    std::vector<index_t> h_cols;                // Local Column indices
    std::vector<index_t> h_rowDelimiters;       // Local Row delimiters
    std::vector<double> h_vec(numRows);         // Global Dense vector (needed for multiply)
    std::vector<double> h_out(local_rows);      // Local Output vector

    if (rank == 0) printf("Initializing data structures...\n");
    
    // Initialize dense vector - Replicated on all ranks for simplicity
    // In a real large-scale app, we might distribute this too
    fill(h_vec.data(), numRows, maxVal);

    // Initialize distributed matrix
    index_t local_nnz = initRandomMatrixDistributed(h_cols, h_val, h_rowDelimiters, nItems, numRows, rank, size, maxVal);

    // CUDA Setup
    double *d_val, *d_vec, *d_out;
    index_t *d_cols, *d_rowDelimiters;

    CHECK_CUDA(cudaMalloc(&d_val, local_nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_cols, local_nnz * sizeof(index_t)));
    CHECK_CUDA(cudaMalloc(&d_rowDelimiters, (local_rows + 1) * sizeof(index_t)));
    CHECK_CUDA(cudaMalloc(&d_vec, numRows * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out, local_rows * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_val, h_val.data(), local_nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, h_cols.data(), local_nnz * sizeof(index_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(), (local_rows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));

    // For validation, compute reference solution on CPU (Rank 0 gathers everything or computes locally)
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution (serial on rank 0)...\n");
        // We need full matrix for reference check. This is expensive but necessary for full validation.
        // For simplicity, re-generate full matrix on rank 0 just for validation logic
        std::vector<double> ref_val;
        std::vector<index_t> ref_cols;
        std::vector<index_t> ref_rowDelimiters(numRows + 1);
        
        // Re-run init logic for full matrix
        index_t ref_nnzAssigned = 0;
        srand(8675309);
        double prob = static_cast<double>(nItems) / (static_cast<double>(numRows) * static_cast<double>(numRows));
        bool fillRemaining = false;
        
        // Allocate estimated size to avoid reallocs
        ref_val.reserve(nItems);
        ref_cols.reserve(nItems);

        for (index_t i = 0; i < numRows; ++i) {
            ref_rowDelimiters[i] = ref_nnzAssigned;
            for (index_t j = 0; j < numRows; ++j) {
                index_t numEntriesLeft = (numRows * numRows) - ((i * numRows) + j);
                index_t needToAssign = nItems - ref_nnzAssigned;
                if (numEntriesLeft <= needToAssign) fillRemaining = true;
                double randVal = static_cast<double>(rand()) / RAND_MAX;
                if ((ref_nnzAssigned < nItems && randVal <= prob) || fillRemaining) {
                    ref_cols.push_back(j);
                    unsigned int seed = ref_nnzAssigned;
                    ref_val.push_back(maxVal * (static_cast<double>(rand_r(&seed)) / (static_cast<double>(RAND_MAX) + 1.0)));
                    ref_nnzAssigned++;
                }
            }
        }
        ref_rowDelimiters[numRows] = nItems;
        
        h_reference.resize(numRows);
        spmvCpu(ref_val.data(), ref_cols.data(), ref_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        // Copy vector to device (in real app, only copy updated parts or keep on device)
        CHECK_CUDA(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (local_rows + threadsPerBlock - 1) / threadsPerBlock;
        spmvKernel<<<blocksPerGrid, threadsPerBlock>>>(d_val, d_cols, d_rowDelimiters, d_vec, local_rows, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Copy result back (needed for next iteration if it were iterative solver, or for validation)
        CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, local_rows * sizeof(double), cudaMemcpyDeviceToHost));

        // If this were a real solver like CG, we would need MPI_Allgather here to update h_vec
        // But benchmark is just SpMV.
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);
        
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Gather results for validation
    std::vector<double> global_out;
    if (validate || printResults) {
        if (rank == 0) global_out.resize(numRows);
        
        // Gather local results from all ranks
        // Since local_rows can vary slightly for last rank, use Gatherv
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        int local_count = local_rows;
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + recvcounts[i-1];
        }
        
        MPI_Gatherv(h_out.data(), local_rows, MPI_DOUBLE, 
                   global_out.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                   0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(global_out, "OutputVector");
    }

    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), global_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_val));
    CHECK_CUDA(cudaFree(d_cols));
    CHECK_CUDA(cudaFree(d_rowDelimiters));
    CHECK_CUDA(cudaFree(d_vec));
    CHECK_CUDA(cudaFree(d_out));

    MPI_Finalize();
    return 0;
}
