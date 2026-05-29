#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

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
#pragma omp parallel
    {
        unsigned int state = 8675309 + static_cast<unsigned int>(omp_get_thread_num()) * 1013;
#pragma omp for schedule(static)
        for (index_t i = 0; i < n; ++i) {
            state = state * 1103515245u + 12345u;
            A[i] = maxVal * (static_cast<double>(state >> 16 & 0x7FFF) /
                             (static_cast<double>(0x7FFF) + 1.0));
        }
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
// CUDA Kernel: spmvCuda
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format on GPU
//
// Arguments:
//   val:           non-zero values (global device memory)
//   cols:          column indices (global device memory)
//   rowDelimiters: row pointer array (global device memory), locally indexed
//   vec:           dense input vector (global device memory), full size
//   localStart:    first local row index (global row numbering)
//   localEnd:      last local row index (exclusive)
//   dim:           total number of rows/columns in the matrix
//   out:           output vector (global device memory), local rows only
//
// ****************************************************************************
__global__ void spmvCudaKernel(const double* __restrict__ val,
                                const index_t* __restrict__ cols,
                                const index_t* __restrict__ rowDelimiters,
                                const double* __restrict__ vec,
                                const index_t localRows,
                                double* __restrict__ out) {
    index_t row = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row < localRows) {
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
// Function: spmvCuda
//
// Purpose:
//   Wrapper to launch the CUDA SpMV kernel and manage data transfers
//
// Arguments:
//   d_val:           device pointer to non-zero values
//   d_cols:          device pointer to column indices
//   d_rowDelimiters: device pointer to local row delimiters
//   d_vec:           device pointer to full input vector
//   d_out:           device pointer to local output
//   localStart:      first local row (global numbering)
//   localEnd:        last local row (exclusive)
//
// ****************************************************************************
void spmvCuda(const double* d_val, const index_t* d_cols,
              const index_t* d_rowDelimiters, const double* d_vec,
              double* d_out, index_t localRows) {
    const int blockSize = 256;
    int numBlocks = (localRows + blockSize - 1) / blockSize;
    // Limit blocks to reasonable number for occupancy
    if (numBlocks > 65535) numBlocks = 65535;

    spmvCudaKernel<<<numBlocks, blockSize>>>(d_val, d_cols, d_rowDelimiters,
                                              d_vec, localRows, d_out);
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format (OpenMP)
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

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse)
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

    // Set CUDA device for each rank (round-robin across available GPUs)
    int numGpus = 0;
    cudaGetDeviceCount(&numGpus);
    if (numGpus > 0) {
        int device = rank % numGpus;
        cudaSetDevice(device);
    }

    // Print benchmark info (rank 0 only)
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Hybrid: MPI (%d ranks) + OpenMP + CUDA (%d GPU%s)\n",
               numRanks, numGpus, numGpus != 1 ? "s" : "");
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
    std::vector<double> h_out(numRows, 0.0);

    // Initialize on rank 0
    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast all data to all ranks
    MPI_Bcast(h_val.data(), nItems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_cols.data(), nItems, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local row distribution: contiguous chunks per rank
    index_t rowsPerRank = numRows / numRanks;
    index_t remainder = numRows % numRanks;
    index_t localStart = static_cast<index_t>(rank) * rowsPerRank +
                         static_cast<index_t>(rank < static_cast<int>(remainder) ? rank : remainder);
    index_t localEnd = localStart + rowsPerRank +
                       static_cast<index_t>(rank < static_cast<int>(remainder) ? 1 : 0);
    index_t localRows = localEnd - localStart;

    // Determine local nnz count
    index_t localNnz = 0;
    for (index_t i = localStart; i < localEnd; ++i) {
        localNnz += h_rowDelimiters[i + 1] - h_rowDelimiters[i];
    }

    // Allocate local CSR data
    std::vector<double> local_val(localNnz);
    std::vector<index_t> local_cols(localNnz);
    std::vector<index_t> local_rowDelimiters(localRows + 1);

    // Fill local CSR data (rowDelimiters relative to local start, val and cols copied)
    {
        index_t localIdx = 0;
        local_rowDelimiters[0] = 0;
        for (index_t i = 0; i < localRows; ++i) {
            index_t globalRow = localStart + i;
            index_t start = h_rowDelimiters[globalRow];
            index_t end = h_rowDelimiters[globalRow + 1];
            for (index_t j = start; j < end; ++j) {
                local_val[localIdx] = h_val[j];
                local_cols[localIdx] = h_cols[j];
                localIdx++;
            }
            local_rowDelimiters[i + 1] = localIdx;
        }
    }

    // Allocate GPU memory
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rowDelimiters = nullptr;

    bool useCuda = (numGpus > 0);
    if (useCuda) {
        cudaMalloc(&d_val, localNnz * sizeof(double));
        cudaMalloc(&d_cols, localNnz * sizeof(index_t));
        cudaMalloc(&d_rowDelimiters, (localRows + 1) * sizeof(index_t));
        cudaMalloc(&d_vec, numRows * sizeof(double));
        cudaMalloc(&d_out, localRows * sizeof(double));

        // Copy data to GPU
        cudaMemcpy(d_val, local_val.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cols, local_cols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rowDelimiters, local_rowDelimiters.data(),
                   (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice);
    }

    // Synchronize all ranks before benchmark
    MPI_Barrier(MPI_COMM_WORLD);

    // For validation, compute reference solution on rank 0
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
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (useCuda) {
            spmvCuda(d_val, d_cols, d_rowDelimiters, d_vec,
                     d_out, localRows);
            cudaDeviceSynchronize();
        } else {
            // Fallback to CPU with OpenMP if no GPU available
            std::vector<double> local_out_cpu(localRows);
            for (index_t i = 0; i < localRows; ++i) {
                double t = 0.0;
                for (index_t j = h_rowDelimiters[localStart + i];
                     j < h_rowDelimiters[localStart + i + 1]; ++j) {
                    t += h_val[j] * h_vec[h_cols[j]];
                }
                local_out_cpu[i] = t;
            }
            // Copy result back
            for (index_t i = 0; i < localRows; ++i) {
                h_out[localStart + i] = local_out_cpu[i];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy results back from GPU
    std::vector<double> local_out(localRows);
    if (useCuda) {
        cudaMemcpy(local_out.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // Gather local results to rank 0
    {
        // Build displacements for MPI_Gatherv
        std::vector<int> recvCounts(numRanks);
        std::vector<int> displs(numRanks);
        int count = 0;
        for (int r = 0; r < numRanks; ++r) {
            int rRows = rowsPerRank + (r < static_cast<int>(remainder) ? 1 : 0);
            recvCounts[r] = rRows;
            displs[r] = count;
            count += rRows;
        }
        MPI_Gatherv(local_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    h_out.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // Synchronize all ranks after computation
    MPI_Barrier(MPI_COMM_WORLD);

    // Print timing and performance (rank 0 only)
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
        print_results(h_out, "OutputVector");
    }

    // Validation (rank 0 only)
    if (validate && rank == 0) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // Free GPU memory
    if (useCuda) {
        cudaFree(d_val);
        cudaFree(d_cols);
        cudaFree(d_rowDelimiters);
        cudaFree(d_vec);
        cudaFree(d_out);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
