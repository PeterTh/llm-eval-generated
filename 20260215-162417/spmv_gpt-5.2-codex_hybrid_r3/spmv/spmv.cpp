#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int THREADS_PER_BLOCK = 256;

inline void checkCuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", message, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void spmvKernel(const index_t numRows, const double* __restrict__ val,
                           const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec,
                           double* out) {
    const index_t row = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row < numRows) {
        double t = 0.0;
        for (index_t j = rowDelimiters[row]; j < rowDelimiters[row + 1]; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[row] = t;
    }
}

void computeRowPartition(const index_t numRows, const int size,
                         std::vector<index_t>& rowStarts,
                         std::vector<index_t>& rowCounts) {
    rowStarts.resize(size);
    rowCounts.resize(size);

    const index_t base = numRows / static_cast<index_t>(size);
    const int remainder = static_cast<int>(numRows % static_cast<index_t>(size));
    index_t offset = 0;

    for (int rank = 0; rank < size; ++rank) {
        const index_t rows = base + (rank < remainder ? 1u : 0u);
        rowStarts[rank] = offset;
        rowCounts[rank] = rows;
        offset += rows;
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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    int shouldExit = 0;
    int exitCode = 0;
    if (rank == 0) {
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
                shouldExit = 1;
                exitCode = 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                shouldExit = 1;
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&shouldExit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (shouldExit) {
        MPI_Finalize();
        return exitCode;
    }

    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int validateInt = validate ? 1 : 0;
    int printResultsInt = printResults ? 1 : 0;
    MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResultsInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = validateInt != 0;
    printResults = printResultsInt != 0;

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
    }

    std::vector<index_t> rowStarts;
    std::vector<index_t> rowCounts;
    computeRowPartition(numRows, size, rowStarts, rowCounts);

    const index_t localRows = rowCounts[rank];

    std::vector<double> h_vec(numRows);
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_reference;

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0 && validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    std::vector<int> rowCountsWithDelimiter(size);
    std::vector<int> rowDisplacements(size);
    for (int r = 0; r < size; ++r) {
        rowCountsWithDelimiter[r] = static_cast<int>(rowCounts[r] + 1);
        rowDisplacements[r] = static_cast<int>(rowStarts[r]);
    }

    std::vector<index_t> local_rowDelimiters(localRows + 1);
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 rowCountsWithDelimiter.data(), rowDisplacements.data(), MPI_UINT32_T,
                 local_rowDelimiters.data(), static_cast<int>(localRows + 1), MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    const index_t baseOffset = local_rowDelimiters[0];
    const index_t local_nnz = local_rowDelimiters[localRows] - baseOffset;
    for (index_t i = 0; i <= localRows; ++i) {
        local_rowDelimiters[i] -= baseOffset;
    }

    std::vector<int> nnzCounts(size, 0);
    std::vector<int> nnzDisplacements(size, 0);
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            const index_t startNnz = h_rowDelimiters[rowStarts[r]];
            const index_t endNnz = h_rowDelimiters[rowStarts[r] + rowCounts[r]];
            nnzCounts[r] = static_cast<int>(endNnz - startNnz);
            nnzDisplacements[r] = static_cast<int>(startNnz);
        }
    }
    MPI_Bcast(nnzCounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzDisplacements.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> local_val(local_nnz);
    std::vector<index_t> local_cols(local_nnz);

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 nnzCounts.data(),
                 nnzDisplacements.data(),
                 MPI_DOUBLE,
                 local_val.data(), static_cast<int>(local_nnz), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 nnzCounts.data(),
                 nnzDisplacements.data(),
                 MPI_UINT32_T,
                 local_cols.data(), static_cast<int>(local_nnz), MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    int deviceCount = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices available.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(rank % deviceCount), "cudaSetDevice");

    const bool hasWork = localRows > 0;
    const index_t valAlloc = local_nnz > 0 ? local_nnz : 1;

    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    if (hasWork) {
        checkCuda(cudaMalloc(&d_vec, numRows * sizeof(double)), "cudaMalloc d_vec");
        checkCuda(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice),
                  "cudaMemcpy d_vec");

        checkCuda(cudaMalloc(&d_val, valAlloc * sizeof(double)), "cudaMalloc d_val");
        checkCuda(cudaMalloc(&d_cols, valAlloc * sizeof(index_t)), "cudaMalloc d_cols");
        if (local_nnz > 0) {
            checkCuda(cudaMemcpy(d_val, local_val.data(), local_nnz * sizeof(double), cudaMemcpyHostToDevice),
                      "cudaMemcpy d_val");
            checkCuda(cudaMemcpy(d_cols, local_cols.data(), local_nnz * sizeof(index_t), cudaMemcpyHostToDevice),
                      "cudaMemcpy d_cols");
        }

        checkCuda(cudaMalloc(&d_rowDelimiters, (localRows + 1) * sizeof(index_t)),
                  "cudaMalloc d_rowDelimiters");
        checkCuda(cudaMemcpy(d_rowDelimiters, local_rowDelimiters.data(),
                             (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice),
                  "cudaMemcpy d_rowDelimiters");

        checkCuda(cudaMalloc(&d_out, localRows * sizeof(double)), "cudaMalloc d_out");
    }

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    if (hasWork) {
        const dim3 block(THREADS_PER_BLOCK);
        const dim3 grid((localRows + block.x - 1) / block.x);
        for (index_t iter = 0; iter < iterations; ++iter) {
            spmvKernel<<<grid, block>>>(localRows, d_val, d_cols, d_rowDelimiters, d_vec, d_out);
        }
        checkCuda(cudaGetLastError(), "spmvKernel launch");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    const double localTime = end - start;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> local_out(localRows);
    if (hasWork && localRows > 0) {
        checkCuda(cudaMemcpy(local_out.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost),
                  "cudaMemcpy d_out");
    }

    std::vector<int> rowCountsRaw(size);
    std::vector<int> rowDisplacementsRaw(size);
    for (int r = 0; r < size; ++r) {
        rowCountsRaw[r] = static_cast<int>(rowCounts[r]);
        rowDisplacementsRaw[r] = static_cast<int>(rowStarts[r]);
    }

    std::vector<double> h_out;
    if (rank == 0) {
        h_out.resize(numRows);
    }

    MPI_Gatherv(local_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr,
                rowCountsRaw.data(), rowDisplacementsRaw.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double totalTimeMs = maxTime * 1000.0;
        const double avgTime = iterations > 0 ? totalTimeMs / static_cast<double>(iterations) : 0.0;
        const double gflops = maxTime > 0.0
                                  ? (2.0 * static_cast<double>(nItems) * static_cast<double>(iterations)) /
                                        (maxTime * 1e9)
                                  : 0.0;

        printf("Computation time: %.3f ms\n", totalTimeMs);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    if (hasWork) {
        cudaFree(d_out);
        cudaFree(d_rowDelimiters);
        cudaFree(d_cols);
        cudaFree(d_val);
        cudaFree(d_vec);
    }

    MPI_Finalize();
    return exitCode;
}
