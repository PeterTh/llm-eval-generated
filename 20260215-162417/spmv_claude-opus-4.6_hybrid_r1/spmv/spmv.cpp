#include <algorithm>
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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

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
//   Computes sparse matrix-vector multiplication using CSR format (OpenMP)
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
// CUDA SpMV kernel: CSR-vector approach (one warp per row)
// Each warp cooperatively processes one row for load-balanced execution
// ****************************************************************************
__global__ void spmv_kernel(const double* __restrict__ val,
                            const index_t* __restrict__ cols,
                            const index_t* __restrict__ rowDelimiters,
                            const double* __restrict__ vec,
                            const index_t numRows,
                            double* __restrict__ out) {
    index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    index_t row = tid / 32;
    index_t lane = tid & 31;

    if (row < numRows) {
        double t = 0.0;
        index_t rowStart = rowDelimiters[row];
        index_t rowEnd = rowDelimiters[row + 1];
        for (index_t j = rowStart + lane; j < rowEnd; j += 32) {
            t += __ldg(&val[j]) * __ldg(&vec[__ldg(&cols[j])]);
        }
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            t += __shfl_down_sync(0xffffffff, t, offset);
        }
        if (lane == 0) {
            out[row] = t;
        }
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
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Assign GPU to rank (round-robin across available devices)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks)
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

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices: %d\n",
               nprocs, omp_get_max_threads(), deviceCount);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Rank 0 allocates and initializes full data
    std::vector<double> h_val(rank == 0 ? nItems : 0);
    std::vector<index_t> h_cols(rank == 0 ? nItems : 0);
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast dense vector and row delimiters to all ranks
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Distribute rows across MPI ranks
    index_t rowsPerRank = numRows / nprocs;
    index_t rowRemainder = numRows % nprocs;
    index_t localStartRow = rank * rowsPerRank + std::min(static_cast<index_t>(rank), rowRemainder);
    index_t localNumRows = rowsPerRank + (static_cast<index_t>(rank) < rowRemainder ? 1 : 0);
    index_t localEndRow = localStartRow + localNumRows;
    index_t localNnz = h_rowDelimiters[localEndRow] - h_rowDelimiters[localStartRow];

    // Build scatter counts/displacements for nnz data and row results
    std::vector<int> nnzCounts(nprocs), nnzDispls(nprocs);
    std::vector<int> rowCounts(nprocs), rowDispls(nprocs);
    for (int r = 0; r < nprocs; r++) {
        index_t rStart = r * rowsPerRank + std::min(static_cast<index_t>(r), rowRemainder);
        index_t rEnd = rStart + rowsPerRank + (static_cast<index_t>(r) < rowRemainder ? 1 : 0);
        nnzDispls[r] = static_cast<int>(h_rowDelimiters[rStart]);
        nnzCounts[r] = static_cast<int>(h_rowDelimiters[rEnd] - h_rowDelimiters[rStart]);
        rowDispls[r] = static_cast<int>(rStart);
        rowCounts[r] = static_cast<int>(rEnd - rStart);
    }

    // Scatter CSR values and column indices to each rank's local portion
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 nnzCounts.data(), nnzDispls.data(), MPI_UNSIGNED,
                 localCols.data(), static_cast<int>(localNnz), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // Build zero-based local row delimiters
    std::vector<index_t> localRowDelimiters(localNumRows + 1);
    index_t nnzOffset = h_rowDelimiters[localStartRow];
    #pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= localNumRows; i++) {
        localRowDelimiters[i] = h_rowDelimiters[localStartRow + i] - nnzOffset;
    }

    // Compute CPU reference solution for validation before freeing rank 0's data
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Allocate device memory
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rowDelimiters = nullptr;

    if (localNnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, localNnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, localNnz * sizeof(index_t)));
    }
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (localNumRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    if (localNumRows > 0) {
        CUDA_CHECK(cudaMalloc(&d_out, localNumRows * sizeof(double)));
    }

    // Transfer data to device
    if (localNnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_val, localVal.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, localCols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, localRowDelimiters.data(),
               (localNumRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    // Kernel launch configuration: one warp (32 threads) per row
    const int threadsPerBlock = 256;
    const int blocksNeeded = localNumRows > 0
        ? static_cast<int>((static_cast<size_t>(localNumRows) * 32 + threadsPerBlock - 1) / threadsPerBlock)
        : 0;

    if (rank == 0) printf("Computing SpMV...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (blocksNeeded > 0) {
            spmv_kernel<<<blocksNeeded, threadsPerBlock>>>(
                d_val, d_cols, d_rowDelimiters, d_vec, localNumRows, d_out);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long localDurationMs = static_cast<long long>(duration.count());
    long long maxDurationMs = 0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG_LONG_INT, MPI_MAX,
               0, MPI_COMM_WORLD);

    // Copy results back from device
    std::vector<double> localOut(localNumRows);
    if (localNumRows > 0) {
        CUDA_CHECK(cudaMemcpy(localOut.data(), d_out, localNumRows * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Gather results to rank 0
    std::vector<double> h_out(rank == 0 ? numRows : 0);
    MPI_Gatherv(localOut.data(), static_cast<int>(localNumRows), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr,
                rowCounts.data(), rowDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int exitCode = 0;

    if (rank == 0) {
        printf("Computation time: %lld ms\n", maxDurationMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (maxDurationMs / 1000.0) / 1e9;
        const double avgTime = maxDurationMs / static_cast<double>(iterations);

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
                exitCode = 1;
            }
        }
    }

    // Broadcast exit code so all ranks agree
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Cleanup device memory
    if (d_val) cudaFree(d_val);
    if (d_cols) cudaFree(d_cols);
    if (d_rowDelimiters) cudaFree(d_rowDelimiters);
    if (d_vec) cudaFree(d_vec);
    if (d_out) cudaFree(d_out);

    MPI_Finalize();
    return exitCode;
}
