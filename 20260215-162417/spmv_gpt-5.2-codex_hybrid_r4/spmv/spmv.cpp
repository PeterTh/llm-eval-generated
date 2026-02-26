#include <chrono>
#include <cmath>
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

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        const cudaError_t err = call;                                                             \
        if (err != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                         \
        }                                                                                         \
    } while (0)

// ****************************************************************************
// CUDA kernel: CSR SpMV
// ****************************************************************************
__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters, const double* __restrict__ vec,
                           double* __restrict__ out, const index_t numRows) {
    const index_t row = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row < numRows) {
        double t = 0.0;
        for (index_t j = rowDelimiters[row]; j < rowDelimiters[row + 1]; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[row] = t;
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
    const int64_t dim64 = static_cast<int64_t>(dim);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < dim64; ++i) {
        const index_t row = static_cast<index_t>(i);
        double t = 0.0;
        for (index_t j = rowDelimiters[row]; j < rowDelimiters[row + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[row] = t;
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
    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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

    if (worldRank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate and initialize data structures
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_val;                          // Non-zero values
    std::vector<index_t> h_cols;                        // Column indices

    if (worldRank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const index_t baseRows = numRows / static_cast<index_t>(worldSize);
    const index_t remainder = numRows % static_cast<index_t>(worldSize);
    const index_t localRows = baseRows + (worldRank < static_cast<int>(remainder) ? 1U : 0U);
    const index_t startRow = static_cast<index_t>(worldRank) * baseRows +
                             (worldRank < static_cast<int>(remainder) ? static_cast<index_t>(worldRank) : remainder);
    const index_t localRowStart = h_rowDelimiters[startRow];
    const index_t localRowEnd = h_rowDelimiters[startRow + localRows];
    const index_t localNnz = localRowEnd - localRowStart;

    std::vector<index_t> localRowDelimiters(localRows + 1);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(localRows + 1); ++i) {
        localRowDelimiters[static_cast<index_t>(i)] =
            h_rowDelimiters[startRow + static_cast<index_t>(i)] - localRowStart;
    }

    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    std::vector<int> nnzCounts;
    std::vector<int> nnzDispls;
    if (worldRank == 0) {
        nnzCounts.resize(worldSize);
        nnzDispls.resize(worldSize);
        for (int r = 0; r < worldSize; ++r) {
            const index_t rBase = numRows / static_cast<index_t>(worldSize);
            const index_t rRem = numRows % static_cast<index_t>(worldSize);
            const index_t rRows = rBase + (r < static_cast<int>(rRem) ? 1U : 0U);
            const index_t rStartRow = static_cast<index_t>(r) * rBase +
                                      (r < static_cast<int>(rRem) ? static_cast<index_t>(r) : rRem);
            const index_t rStartNnz = h_rowDelimiters[rStartRow];
            const index_t rEndNnz = h_rowDelimiters[rStartRow + rRows];
            nnzCounts[r] = static_cast<int>(rEndNnz - rStartNnz);
            nnzDispls[r] = static_cast<int>(rStartNnz);
        }
    }

    const int* nnzCountsPtr = (worldRank == 0) ? nnzCounts.data() : nullptr;
    const int* nnzDisplsPtr = (worldRank == 0) ? nnzDispls.data() : nullptr;
    double* localValPtr = localVal.empty() ? nullptr : localVal.data();
    index_t* localColsPtr = localCols.empty() ? nullptr : localCols.data();

    MPI_Scatterv((worldRank == 0) ? h_val.data() : nullptr, nnzCountsPtr, nnzDisplsPtr, MPI_DOUBLE,
                 localValPtr, static_cast<int>(localNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv((worldRank == 0) ? h_cols.data() : nullptr, nnzCountsPtr, nnzDisplsPtr, MPI_UINT32_T,
                 localColsPtr, static_cast<int>(localNnz), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && worldRank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (worldRank == 0) {
            fprintf(stderr, "No CUDA devices found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = worldRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    if (localNnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, localNnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, localNnz * sizeof(index_t)));
        CUDA_CHECK(cudaMemcpy(d_val, localVal.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, localCols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    }

    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(&d_out, localRows * sizeof(double)));
    }
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, localRowDelimiters.data(),
                          (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));
    if (localRows > 0) {
        CUDA_CHECK(cudaMemset(d_out, 0, localRows * sizeof(double)));
    }

    if (worldRank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const int blockSize = 256;
    const int gridSize = (localRows + blockSize - 1) / blockSize;
    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows > 0) {
            spmvKernel<<<gridSize, blockSize>>>(d_val, d_cols, d_rowDelimiters, d_vec, d_out, localRows);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double localMs = static_cast<double>(duration.count());
    double maxMs = 0.0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        printf("Computation time: %.0f ms\n", maxMs);
        const double gflops = (2.0 * nItems * iterations) / (maxMs / 1000.0) / 1e9;
        const double avgTime = maxMs / static_cast<double>(iterations);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    bool valid = true;
    if (printResults || validate) {
        std::vector<double> h_out_local(localRows);
        if (localRows > 0) {
            CUDA_CHECK(cudaMemcpy(h_out_local.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost));
        }

        std::vector<double> h_out;
        std::vector<int> rowCounts;
        std::vector<int> rowDispls;
        if (worldRank == 0) {
            h_out.resize(numRows);
            rowCounts.resize(worldSize);
            rowDispls.resize(worldSize);
            for (int r = 0; r < worldSize; ++r) {
                const index_t rBase = numRows / static_cast<index_t>(worldSize);
                const index_t rRem = numRows % static_cast<index_t>(worldSize);
                const index_t rRows = rBase + (r < static_cast<int>(rRem) ? 1U : 0U);
                const index_t rStartRow = static_cast<index_t>(r) * rBase +
                                          (r < static_cast<int>(rRem) ? static_cast<index_t>(r) : rRem);
                rowCounts[r] = static_cast<int>(rRows);
                rowDispls[r] = static_cast<int>(rStartRow);
            }
        }

        MPI_Gatherv(h_out_local.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    (worldRank == 0) ? h_out.data() : nullptr,
                    (worldRank == 0) ? rowCounts.data() : nullptr,
                    (worldRank == 0) ? rowDispls.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (worldRank == 0 && printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate && worldRank == 0) {
            printf("Validating result...\n");
            valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    if (d_val) {
        CUDA_CHECK(cudaFree(d_val));
    }
    if (d_cols) {
        CUDA_CHECK(cudaFree(d_cols));
    }
    if (d_rowDelimiters) {
        CUDA_CHECK(cudaFree(d_rowDelimiters));
    }
    if (d_vec) {
        CUDA_CHECK(cudaFree(d_vec));
    }
    if (d_out) {
        CUDA_CHECK(cudaFree(d_out));
    }

    int validationStatus = 0;
    if (validate) {
        if (worldRank == 0) {
            validationStatus = valid ? 0 : 1;
        }
        MPI_Bcast(&validationStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return validationStatus;
}
