#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values (OpenMP-parallelized)
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
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format
//   (OpenMP-parallelized for reference computation)
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
// CUDA kernel: spmvCuda
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format on GPU.
//   Each thread block handles one row. Threads within a block cooperatively
//   compute the dot product for that row.
//
// Arguments:
//   val:           non-zero values (device)
//   cols:          column indices (device)
//   rowDelimiters: row pointers (device) - local indices [0, local_nnz+1]
//   vec:           input dense vector (device) - full vector
//   local_nnz:     number of non-zeros owned by this rank
//   local_rows:    number of rows owned by this rank
//   out:           output vector (device) - only local rows
//   global_row_start: first global row index owned by this rank
//
// ****************************************************************************
__global__ void spmvCudaKernel(const double* val, const index_t* cols,
                                const index_t* rowDelimiters,
                                const double* vec,
                                const index_t local_nnz,
                                const index_t local_rows,
                                double* out) {
    // Each thread handles one row
    index_t row = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= local_rows) return;

    double sum = 0.0;
    index_t start = rowDelimiters[row];
    index_t end = rowDelimiters[row + 1];

    for (index_t j = start; j < end; ++j) {
        sum += val[j] * vec[cols[j]];
    }

    out[row] = sum;
}

// ****************************************************************************
// Function: spmvCuda
//
// Purpose:
//   Host-side wrapper for CUDA SpMV kernel
//
// Arguments:
//   d_val, d_cols, d_rowDelimiters: device pointers to CSR data (already allocated)
//   d_vec: device pointer to full input vector (already allocated)
//   d_out: device pointer to output vector (already allocated)
//   local_nnz: number of non-zeros for this rank
//   local_rows: number of rows for this rank
//
// ****************************************************************************
void spmvCuda(const double* d_val, const index_t* d_cols,
              const index_t* d_rowDelimiters,
              const double* d_vec,
              double* d_out,
              const index_t local_nnz,
              const index_t local_rows) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (local_rows + threadsPerBlock - 1) / threadsPerBlock;

    spmvCudaKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_val, d_cols, d_rowDelimiters, d_vec, local_nnz, local_rows, d_out);
}

// ****************************************************************************
// Function: verifyResults
//
// Purpose:
//   Verifies correctness of results by comparing to reference solution
//   (OpenMP-parallelized)
//
// Arguments:
//   reference: array holding the reference result vector
//   result: array holding the result vector to verify
//   size: number of elements per vector
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    bool failed = false;

#pragma omp parallel for reduction(||:failed) schedule(static)
    for (index_t i = 0; i < size; ++i) {
        if (failed) continue;
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
#pragma omp critical
                {
                    if (!failed) {
                        printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                               i, ref, res);
                    }
                }
                failed = true;
            }
        } else {
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
#pragma omp critical
                {
                    if (!failed) {
                        printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                               i, ref, res, relError);
                    }
                }
                failed = true;
            }
        }
    }
    return !failed;
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

    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (on all ranks, but only print from rank 0)
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
            if (worldRank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    // Print benchmark info (rank 0 only)
    if (worldRank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Parallelization: Hybrid MPI + CUDA + OpenMP\n");
        printf("MPI ranks: %d | OpenMP threads per rank: %d\n",
               worldSize, omp_get_max_threads());
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // =========================================================================
    // Row distribution: each MPI rank gets a contiguous chunk of rows
    // =========================================================================
    const index_t rowsPerRank = numRows / worldSize;
    const index_t remainder = numRows % worldSize;
    const index_t localRowStart = worldRank * rowsPerRank + (worldRank < remainder ? worldRank : remainder);
    const index_t localRowEnd = (worldRank + 1) * rowsPerRank + (worldRank + 1 <= remainder ? worldRank + 1 : remainder);
    const index_t localRows = localRowEnd - localRowStart;

    // =========================================================================
    // Allocate and initialize data structures (on rank 0, then broadcast)
    // =========================================================================
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out(numRows);                 // Output vector

    if (worldRank == 0) {
        if (worldRank == 0) printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast full data to all ranks
    MPI_Bcast(h_val.data(), nItems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_cols.data(), nItems, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // =========================================================================
    // Compute local CSR data for this rank's row chunk
    // =========================================================================
    // Local row delimiters: remap from global indices
    // Global row_delimiters[row] gives the start index of row 'row' in the global CSR arrays
    // For local rows [localRowStart, localRowEnd), we need:
    //   local_rowDelimiters[0] = h_rowDelimiters[localRowStart]
    //   local_rowDelimiters[i] = h_rowDelimiters[localRowStart + i]
    // The local nnz range is [h_rowDelimiters[localRowStart], h_rowDelimiters[localRowEnd])
    const index_t globalNnzStart = h_rowDelimiters[localRowStart];
    const index_t globalNnzEnd = h_rowDelimiters[localRowEnd];
    const index_t localNnz = globalNnzEnd - globalNnzStart;

    // Extract local CSR data
    std::vector<double> local_val(localNnz);
    std::vector<index_t> local_cols(localNnz);
    std::vector<index_t> local_rowDelimiters(localRows + 1);

#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < localNnz; ++i) {
        local_val[i] = h_val[globalNnzStart + i];
        local_cols[i] = h_cols[globalNnzStart + i];
    }

    // Build local row delimiters (remapped to start from 0)
    for (index_t i = 0; i <= localRows; ++i) {
        local_rowDelimiters[i] = h_rowDelimiters[localRowStart + i] - globalNnzStart;
    }

    // =========================================================================
    // Reference solution (rank 0 only, for validation)
    // =========================================================================
    std::vector<double> h_reference;
    if (validate && worldRank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // =========================================================================
    // CUDA setup: allocate device memory
    // =========================================================================
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    cudaSetDevice(worldRank % 4); // Use round-robin device assignment for up to 4 GPUs

    if (localNnz > 0) {
        cudaMalloc(&d_val, localNnz * sizeof(double));
        cudaMalloc(&d_cols, localNnz * sizeof(index_t));
        cudaMalloc(&d_rowDelimiters, (localRows + 1) * sizeof(index_t));
        cudaMemcpy(d_val, local_val.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cols, local_cols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rowDelimiters, local_rowDelimiters.data(),
                   (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    }

    // Full vector on device (needed for indirect access)
    cudaMalloc(&d_vec, numRows * sizeof(double));
    cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice);

    // Local output on device
    cudaMalloc(&d_out, localRows * sizeof(double));

    // =========================================================================
    // Perform SpMV computation
    // =========================================================================
    if (worldRank == 0) printf("Computing SpMV...\n");

    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows > 0 && localNnz > 0) {
            spmvCuda(d_val, d_cols, d_rowDelimiters, d_vec, d_out, localNnz, localRows);
        }
    }

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // =========================================================================
    // Gather output from all ranks
    // =========================================================================
    // Each rank writes its local output to its portion of h_out
    if (localRows > 0) {
        cudaMemcpy(h_out.data() + localRowStart, d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // Gather all outputs to rank 0 using MPI_Gatherv
    std::vector<int> recvCounts(worldSize);
    std::vector<int> displacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const index_t rStart = r * rowsPerRank + (r < static_cast<int>(remainder) ? r : remainder);
        const index_t rEnd = (r + 1) * rowsPerRank + (r + 1 <= static_cast<int>(remainder) ? r + 1 : remainder);
        recvCounts[r] = rEnd - rStart;
        displacements[r] = rStart;
    }
    MPI_Gatherv(h_out.data() + localRowStart, localRows, MPI_DOUBLE,
                h_out.data(), recvCounts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // =========================================================================
    // Print timing (rank 0 only)
    // =========================================================================
    if (worldRank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // =========================================================================
    // Print results for external validation
    // =========================================================================
    if (printResults && worldRank == 0) {
        print_results(h_out, "OutputVector");
    }

    // =========================================================================
    // Validation
    // =========================================================================
    if (validate) {
        if (worldRank == 0) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // =========================================================================
    // Cleanup CUDA
    // =========================================================================
    if (d_val) cudaFree(d_val);
    if (d_cols) cudaFree(d_cols);
    if (d_rowDelimiters) cudaFree(d_rowDelimiters);
    if (d_vec) cudaFree(d_vec);
    if (d_out) cudaFree(d_out);

    MPI_Finalize();
    return 0;
}
