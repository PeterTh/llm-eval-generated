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

// ---------------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(ans) \
    do { \
        cudaError_t _e = (ans); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(_e)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA kernel: CSR SpMV (one thread per row)
// ---------------------------------------------------------------------------
__global__ void spmvKernel(const double* __restrict__ val,
                           const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec,
                           index_t numRows,
                           double* __restrict__ out) {
    index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        double sum = 0.0;
        for (index_t j = rowDelimiters[row]; j < rowDelimiters[row + 1]; ++j) {
            sum += val[j] * vec[cols[j]];
        }
        out[row] = sum;
    }
}

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values (serial, rank 0 only)
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
//   Computes sparse matrix-vector multiplication using CSR format (serial CPU)
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
//   (parallelized with OpenMP)
//
// Arguments:
//   reference: array holding the reference result vector
//   result: array holding the result vector to verify
//   size: number of elements per vector
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    int valid = 1;
    #pragma omp parallel for reduction(&& : valid)
    for (index_t i = 0; i < size; ++i) {
        if (!valid) continue;
        const double ref = reference[i];
        const double res = result[i];
        int ok;
        if (std::abs(ref) < 1e-10) {
            ok = (std::abs(res) <= MAX_RELATIVE_ERROR);
            if (!ok) {
                #pragma omp critical
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
            }
        } else {
            const double relError = std::abs((res - ref) / ref);
            ok = (relError <= MAX_RELATIVE_ERROR);
            if (!ok) {
                #pragma omp critical
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i, ref, res, relError);
            }
        }
        valid = valid && ok;
    }
    return valid != 0;
}

// ****************************************************************************
// Function: printUsage
//
// Purpose:
//   Print command-line usage information
//
// ****************************************************************************
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

// ****************************************************************************
// Function: distributeRows
//
// Purpose:
//   Compute the local row range for each MPI rank
//
// Arguments:
//   numRows:      total number of rows
//   mpiSize:      number of MPI processes
//   mpiRank:      process rank
//   localRows:    (out) number of rows assigned to this rank
//   rowStart:     (out) global row index where this rank's block starts
//   rowCounts:    (out, optional) array of size mpiSize with per-rank row counts
//   rowDispls:    (out, optional) array of size mpiSize with per-rank row displacements
//
// ****************************************************************************
void distributeRows(index_t numRows, int mpiSize, int mpiRank,
                    index_t& localRows, index_t& rowStart,
                    int* rowCounts, int* rowDispls) {
    index_t rowsPerRank = numRows / mpiSize;
    index_t remainder = numRows % mpiSize;

    // Compute for each rank
    for (int r = 0; r < mpiSize; ++r) {
        index_t rLocal, rStart;
        if (r < static_cast<int>(remainder)) {
            rLocal = rowsPerRank + 1;
            rStart = r * rLocal;
        } else {
            rLocal = rowsPerRank;
            rStart = remainder * (rowsPerRank + 1) + (r - remainder) * rowsPerRank;
        }
        if (r == mpiRank) {
            localRows = rLocal;
            rowStart = rStart;
        }
        if (rowCounts) rowCounts[r] = static_cast<int>(rLocal);
        if (rowDispls) rowDispls[r] = static_cast<int>(rStart);
    }
}

// ****************************************************************************
// Function: main
//
// Purpose:
//   Hybrid MPI + OpenMP + CUDA parallel SpMV benchmark
//
// ****************************************************************************
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // Select GPU: round-robin across MPI ranks
    int nDevices = 0;
    cudaError_t devErr = cudaGetDeviceCount(&nDevices);
    if (devErr != cudaSuccess || nDevices == 0) {
        fprintf(stderr, "Rank %d: No CUDA-capable devices found\n", mpiRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(mpiRank % nDevices));

    // -----------------------------------------------------------------------
    // Parse command-line arguments on rank 0, then broadcast
    // -----------------------------------------------------------------------
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    int validate = 0;
    int printResults = 0;

    if (mpiRank == 0) {
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
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
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

    // Broadcast parameters
    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const index_t nItems = (numRows * numRows) / sparsity;

    // -----------------------------------------------------------------------
    // Row distribution
    // -----------------------------------------------------------------------
    std::vector<int> rowCounts(mpiSize), rowDispls(mpiSize);
    std::vector<int> nnzCounts(mpiSize), nnzDispls(mpiSize);
    index_t localRows = 0, rowStart = 0;
    distributeRows(numRows, mpiSize, mpiRank, localRows, rowStart,
                   rowCounts.data(), rowDispls.data());

    // -----------------------------------------------------------------------
    // Rank 0: generate data, compute reference, prepare scatter
    // -----------------------------------------------------------------------
    std::vector<double> h_val, h_vec, h_reference;
    std::vector<index_t> h_cols, h_rowDelimiters;

    if (mpiRank == 0) {
        // Print header
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", mpiSize);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        {
            int dev;
            cudaGetDevice(&dev);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, dev);
            printf("GPU: %s\n", prop.name);
        }

        // Generate data
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_vec.resize(numRows);

        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        // Reference (serial CPU on rank 0)
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }

        // Scatter parameters for val/cols
        for (int r = 0; r < mpiSize; ++r) {
            index_t rStart = static_cast<index_t>(rowDispls[r]);
            index_t rEnd   = rStart + static_cast<index_t>(rowCounts[r]);
            index_t rNnz   = h_rowDelimiters[rEnd] - h_rowDelimiters[rStart];
            nnzCounts[r] = static_cast<int>(rNnz);
            nnzDispls[r] = static_cast<int>(h_rowDelimiters[rStart]);
        }
    }

    // Broadcast row delimiters and vector to all ranks (small O(dim) data)
    if (mpiRank != 0) {
        h_rowDelimiters.resize(numRows + 1);
        h_vec.resize(numRows);
    }
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(),          static_cast<int>(numRows),        MPI_DOUBLE,    0, MPI_COMM_WORLD);

    // Scatter nnz counts/displs
    MPI_Bcast(nnzCounts.data(), mpiSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzDispls.data(), mpiSize, MPI_INT, 0, MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // Scatter the large matrix data (val and cols)
    // -----------------------------------------------------------------------
    index_t localNnz = h_rowDelimiters[rowStart + localRows] - h_rowDelimiters[rowStart];

    std::vector<double>  localVal(static_cast<size_t>(localNnz));
    std::vector<index_t> localCols(static_cast<size_t>(localNnz));

    MPI_Scatterv(mpiRank == 0 ? h_val.data() : nullptr,
                 nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 localVal.data(),  static_cast<int>(localNnz), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(mpiRank == 0 ? h_cols.data() : nullptr,
                 nnzCounts.data(), nnzDispls.data(), MPI_UINT32_T,
                 localCols.data(), static_cast<int>(localNnz), MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // Build local row delimiters (shifted so first local row starts at 0)
    // -----------------------------------------------------------------------
    std::vector<index_t> localRowDelimiters(localRows + 1);
    {
        const index_t base = h_rowDelimiters[rowStart];
        #pragma omp parallel for
        for (index_t i = 0; i <= localRows; ++i) {
            localRowDelimiters[i] = h_rowDelimiters[rowStart + i] - base;
        }
    }

    // -----------------------------------------------------------------------
    // Allocate GPU memory and transfer data
    // -----------------------------------------------------------------------
    double*   d_val           = nullptr;
    double*   d_vec           = nullptr;
    double*   d_out           = nullptr;
    index_t*  d_cols          = nullptr;
    index_t*  d_rowDelimiters = nullptr;

    if (localNnz > 0 && localRows > 0) {
        CUDA_CHECK(cudaMalloc(&d_val,           static_cast<size_t>(localNnz) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols,          static_cast<size_t>(localNnz) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&d_rowDelimiters, static_cast<size_t>(localRows + 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&d_vec,           static_cast<size_t>(numRows) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_out,           static_cast<size_t>(localRows) * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_val,           localVal.data(),           static_cast<size_t>(localNnz)      * sizeof(double),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols,          localCols.data(),          static_cast<size_t>(localNnz)      * sizeof(index_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rowDelimiters, localRowDelimiters.data(), static_cast<size_t>(localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vec,           h_vec.data(),              static_cast<size_t>(numRows)       * sizeof(double),  cudaMemcpyHostToDevice));
    }

    // -----------------------------------------------------------------------
    // Timed SpMV computation (GPU)
    // -----------------------------------------------------------------------
    if (mpiRank == 0) {
        printf("Computing SpMV on GPU...\n");
    }

    // Synchronize all ranks before starting timed region
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows > 0) {
            constexpr int threadsPerBlock = 256;
            int blocksPerGrid = (static_cast<int>(localRows) + threadsPerBlock - 1) / threadsPerBlock;
            spmvKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long localDurationMs = static_cast<long long>(duration.count());
    long long globalDurationMs = 0;
    MPI_Reduce(&localDurationMs, &globalDurationMs, 1, MPI_LONG_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // Copy results back from GPU
    // -----------------------------------------------------------------------
    std::vector<double> localOut(localRows);
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(localOut.data(), d_out,
                              static_cast<size_t>(localRows) * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    // -----------------------------------------------------------------------
    // Gather results to rank 0 for output / validation
    // -----------------------------------------------------------------------
    std::vector<double> globalOut;
    if (mpiRank == 0) {
        globalOut.resize(numRows);
    }
    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                mpiRank == 0 ? globalOut.data() : nullptr,
                rowCounts.data(), rowDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // Report performance (rank 0 only)
    // -----------------------------------------------------------------------
    if (mpiRank == 0) {
        printf("Computation time: %lld ms\n", globalDurationMs);

        const double elapsedSec = static_cast<double>(globalDurationMs) / 1000.0;
        const double avgTime    = static_cast<double>(globalDurationMs)
                                  / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        if (elapsedSec > 0.0) {
            const double gflops = (2.0 * static_cast<double>(nItems) * iterations)
                                  / elapsedSec / 1e9;
            printf("Performance: %.3f GFLOPS\n", gflops);
        } else {
            printf("Performance: N/A (computation too fast)\n");
        }
    }

    // -----------------------------------------------------------------------
    // External validation output (rank 0 only)
    // -----------------------------------------------------------------------
    if (printResults && mpiRank == 0) {
        print_results(globalOut, "OutputVector");
    }

    // -----------------------------------------------------------------------
    // Validation (rank 0 only)
    // -----------------------------------------------------------------------
    if (validate && mpiRank == 0) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), globalOut.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    if (d_val)           CUDA_CHECK(cudaFree(d_val));
    if (d_cols)          CUDA_CHECK(cudaFree(d_cols));
    if (d_rowDelimiters) CUDA_CHECK(cudaFree(d_rowDelimiters));
    if (d_vec)           CUDA_CHECK(cudaFree(d_vec));
    if (d_out)           CUDA_CHECK(cudaFree(d_out));

    MPI_Finalize();
    return 0;
}
