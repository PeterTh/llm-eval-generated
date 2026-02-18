#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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
#pragma omp parallel
    {
        unsigned int seed = 8675309u + 1103515245u * static_cast<unsigned int>(omp_get_thread_num());
#pragma omp for schedule(static)
        for (index_t i = 0; i < n; ++i) {
            A[i] = maxVal * (rand_r(&seed) / (static_cast<double>(RAND_MAX) + 1.0));
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

__global__ void spmvKernel(const double* val, const index_t* cols, const index_t* rowDelimiters,
                           const double* vec, double* out, index_t numRows) {
    const index_t row = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row < numRows) {
        double sum = 0.0;
        for (index_t j = rowDelimiters[row]; j < rowDelimiters[row + 1]; ++j) {
            sum += val[j] * vec[cols[j]];
        }
        out[row] = sum;
    }
}

static inline void checkCuda(cudaError_t result, const char* context) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", context, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static inline void computeRowRange(index_t numRows, int rank, int size,
                                   index_t& rowStart, index_t& localRows) {
    const index_t baseRows = numRows / static_cast<index_t>(size);
    const index_t extra = numRows % static_cast<index_t>(size);
    localRows = baseRows + (rank < static_cast<int>(extra) ? 1 : 0);
    rowStart = baseRows * static_cast<index_t>(rank) + static_cast<index_t>(rank < static_cast<int>(extra) ? rank : extra);
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
    bool showHelp = false;
    bool parseOk = true;

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
                showHelp = true;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parseOk = false;
                break;
            }
        }

        if (showHelp) {
            printUsage(argv[0]);
        } else if (!parseOk) {
            printUsage(argv[0]);
        }
    }

    int parseOkInt = parseOk ? 1 : 0;
    int showHelpInt = showHelp ? 1 : 0;
    MPI_Bcast(&parseOkInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&showHelpInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!parseOkInt) {
        MPI_Finalize();
        return 1;
    }
    if (showHelpInt) {
        MPI_Finalize();
        return 0;
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
        printf("MPI ranks: %d\n", size);
    }

    index_t rowStart = 0;
    index_t localRows = 0;
    computeRowRange(numRows, rank, size, rowStart, localRows);
    const index_t rowEnd = rowStart + localRows;

    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);

    std::vector<int> rowCounts;
    std::vector<int> rowDispls;
    std::vector<int> rowCountsDelim;
    std::vector<int> rowDisplsDelim;
    std::vector<int> nnzCounts;
    std::vector<int> nnzDispls;

    if (rank == 0) {
        // Allocate and initialize data structures
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        rowCounts.resize(size);
        rowDispls.resize(size);
        rowCountsDelim.resize(size);
        rowDisplsDelim.resize(size);
        nnzCounts.resize(size);
        nnzDispls.resize(size);

        for (int r = 0; r < size; ++r) {
            index_t start = 0;
            index_t rows = 0;
            computeRowRange(numRows, r, size, start, rows);
            rowCounts[r] = static_cast<int>(rows);
            rowDispls[r] = static_cast<int>(start);
            rowCountsDelim[r] = static_cast<int>(rows + 1);
            rowDisplsDelim[r] = static_cast<int>(start);
            index_t end = start + rows;
            index_t base = h_rowDelimiters[start];
            index_t nnz = h_rowDelimiters[end] - base;
            nnzCounts[r] = static_cast<int>(nnz);
            nnzDispls[r] = static_cast<int>(base);
        }
    }

    int localNnz = 0;
    MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_INT,
                &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> localVal(static_cast<size_t>(localNnz));
    std::vector<index_t> localCols(static_cast<size_t>(localNnz));
    std::vector<index_t> localRowDelimiters(static_cast<size_t>(localRows + 1));

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr,
                 MPI_DOUBLE,
                 localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr,
                 MPI_UINT32_T,
                 localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 rank == 0 ? rowCountsDelim.data() : nullptr,
                 rank == 0 ? rowDisplsDelim.data() : nullptr,
                 MPI_UINT32_T,
                 localRowDelimiters.data(), static_cast<int>(localRows + 1), MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    const index_t baseOffset = localRowDelimiters.empty() ? 0 : localRowDelimiters[0];
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < localRows + 1; ++i) {
        localRowDelimiters[i] -= baseOffset;
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> localOut(static_cast<size_t>(localRows));

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices available.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = rank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice");

    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    if (localNnz > 0) {
        checkCuda(cudaMalloc(&d_val, static_cast<size_t>(localNnz) * sizeof(double)), "cudaMalloc val");
        checkCuda(cudaMalloc(&d_cols, static_cast<size_t>(localNnz) * sizeof(index_t)), "cudaMalloc cols");
        checkCuda(cudaMemcpy(d_val, localVal.data(), static_cast<size_t>(localNnz) * sizeof(double), cudaMemcpyHostToDevice),
                  "cudaMemcpy val");
        checkCuda(cudaMemcpy(d_cols, localCols.data(), static_cast<size_t>(localNnz) * sizeof(index_t), cudaMemcpyHostToDevice),
                  "cudaMemcpy cols");
    }

    if (localRows > 0) {
        checkCuda(cudaMalloc(&d_rowDelimiters, static_cast<size_t>(localRows + 1) * sizeof(index_t)), "cudaMalloc rowDelimiters");
        checkCuda(cudaMalloc(&d_out, static_cast<size_t>(localRows) * sizeof(double)), "cudaMalloc out");
        checkCuda(cudaMemcpy(d_rowDelimiters, localRowDelimiters.data(),
                             static_cast<size_t>(localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice),
                  "cudaMemcpy rowDelimiters");
    }

    if (numRows > 0) {
        checkCuda(cudaMalloc(&d_vec, static_cast<size_t>(numRows) * sizeof(double)), "cudaMalloc vec");
        checkCuda(cudaMemcpy(d_vec, h_vec.data(), static_cast<size_t>(numRows) * sizeof(double), cudaMemcpyHostToDevice),
                  "cudaMemcpy vec");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    if (localRows > 0) {
        const int threadsPerBlock = 256;
        const int blocks = static_cast<int>((localRows + threadsPerBlock - 1) / threadsPerBlock);
        for (index_t iter = 0; iter < iterations; ++iter) {
            spmvKernel<<<blocks, threadsPerBlock>>>(d_val, d_cols, d_rowDelimiters, d_vec, d_out, localRows);
        }
        checkCuda(cudaGetLastError(), "spmvKernel launch");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    const double endTime = MPI_Wtime();
    const double localTime = endTime - startTime;

    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (localRows > 0) {
        checkCuda(cudaMemcpy(localOut.data(), d_out, static_cast<size_t>(localRows) * sizeof(double), cudaMemcpyDeviceToHost),
                  "cudaMemcpy out");
    }

    std::vector<double> h_out;
    if (rank == 0) {
        h_out.resize(numRows);
    }

    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr,
                rank == 0 ? rowCounts.data() : nullptr,
                rank == 0 ? rowDispls.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double totalMs = maxTime * 1000.0;
        printf("Computation time: %.3f ms\n", totalMs);

        const double gflops = (2.0 * static_cast<double>(nItems) * iterations) / maxTime / 1e9;
        const double avgTime = totalMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }
    }

    int exitCode = 0;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        std::vector<double> h_reference(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
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

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (d_val) {
        cudaFree(d_val);
    }
    if (d_cols) {
        cudaFree(d_cols);
    }
    if (d_rowDelimiters) {
        cudaFree(d_rowDelimiters);
    }
    if (d_vec) {
        cudaFree(d_vec);
    }
    if (d_out) {
        cudaFree(d_out);
    }

    MPI_Finalize();
    return exitCode;
}
