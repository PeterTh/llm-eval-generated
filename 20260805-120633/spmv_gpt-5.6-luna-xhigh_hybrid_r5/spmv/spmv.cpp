#include <climits>
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
constexpr unsigned CUDA_BLOCK_SIZE = 256;

// A single CUDA thread owns one row. This preserves the CSR accumulation
// order of the reference implementation while giving the GPU a regular,
// independent work item for every output row.
__global__ void spmvKernel(const double* __restrict__ val,
                           const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec,
                           const index_t localRows,
                           double* __restrict__ out) {
    const index_t localRow = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (localRow >= localRows) {
        return;
    }

    double sum = 0.0;
    const index_t begin = rowDelimiters[localRow];
    const index_t end = rowDelimiters[localRow + 1];
    for (index_t j = begin; j < end; ++j) {
        sum += val[j] * vec[cols[j]];
    }
    out[localRow] = sum;
}

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    fprintf(stderr, "CUDA failure at %s:%d while evaluating %s: %s\n", file, line,
            expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(expression) \
    do { \
        const cudaError_t cudaError = (expression); \
        if (cudaError != cudaSuccess) { \
            cudaFailure(cudaError, #expression, __FILE__, __LINE__); \
        } \
    } while (false)

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
    for (int64_t i = 0; i < static_cast<int64_t>(dim); ++i) {
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
    int64_t firstFailure = -1;
    double firstReference = 0.0;
    double firstResult = 0.0;
    double firstRelativeError = 0.0;

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(size); ++i) {
        const double ref = reference[i];
        const double res = result[i];
        bool valid = true;
        double relativeError = 0.0;
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                valid = false;
            }
        } else {
            // Check relative error
            relativeError = std::abs((res - ref) / ref);
            if (relativeError > MAX_RELATIVE_ERROR) {
                valid = false;
            }
        }

        if (!valid) {
#pragma omp critical
            {
                if (firstFailure < 0) {
                    firstFailure = i;
                    firstReference = ref;
                    firstResult = res;
                    firstRelativeError = relativeError;
                }
            }
        }
    }

    if (firstFailure >= 0) {
        const index_t index = static_cast<index_t>(firstFailure);
        if (std::abs(firstReference) < 1e-10) {
            printf("Validation failed at index %u: reference %.10e, got %.10e\n", index,
                   firstReference, firstResult);
        } else {
            printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                   index, firstReference, firstResult, firstRelativeError);
        }
        return false;
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
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int mpiRank = 0;
    int mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (mpiRank == 0) {
            fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

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
            if (mpiRank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (numRows == 0 || sparsity == 0) {
        if (mpiRank == 0) {
            fprintf(stderr, "Matrix size and sparsity must be greater than zero\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Use the MPI shared-memory communicator to map ranks to local GPUs. This
    // works both on one node and across accelerator clusters with multiple
    // nodes, without requiring a scheduler-specific environment variable.
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (mpiRank == 0) {
            fprintf(stderr, "No CUDA devices are visible to the MPI job\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int deviceIndex = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(deviceIndex));

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceIndex));

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (mpiRank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d\n", mpiSize);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("CUDA device: %s\n", deviceProperties.name);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate and initialize data structures
    std::vector<double> h_val;                          // Non-zero values (rank 0)
    std::vector<index_t> h_cols;                        // Column indices (rank 0)
    std::vector<index_t> h_rowDelimiters(numRows + 1); // Global row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out;                          // Output vector (rank 0)

    if (mpiRank == 0) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_out.resize(numRows);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0,
              MPI_COMM_WORLD);

    // Partition rows contiguously so every rank owns a self-contained CSR
    // segment. All ranks compute the partition after the row-pointer broadcast,
    // avoiding an additional metadata exchange.
    std::vector<int> rowCounts(mpiSize);
    std::vector<int> rowDisplacements(mpiSize);
    std::vector<int> nnzCounts(mpiSize);
    std::vector<int> nnzDisplacements(mpiSize);
    for (int rank = 0; rank < mpiSize; ++rank) {
        const index_t begin = static_cast<index_t>(
            (static_cast<uint64_t>(numRows) * static_cast<uint64_t>(rank)) /
            static_cast<uint64_t>(mpiSize));
        const index_t end = static_cast<index_t>(
            (static_cast<uint64_t>(numRows) * static_cast<uint64_t>(rank + 1)) /
            static_cast<uint64_t>(mpiSize));
        const index_t nnzBegin = h_rowDelimiters[begin];
        const index_t nnzEnd = h_rowDelimiters[end];

        if (end - begin > static_cast<index_t>(INT_MAX) ||
            nnzEnd > static_cast<index_t>(INT_MAX) ||
            nnzBegin > static_cast<index_t>(INT_MAX)) {
            if (mpiRank == 0) {
                fprintf(stderr, "MPI distribution exceeds 32-bit MPI count limits\n");
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        rowCounts[rank] = static_cast<int>(end - begin);
        rowDisplacements[rank] = static_cast<int>(begin);
        nnzCounts[rank] = static_cast<int>(nnzEnd - nnzBegin);
        nnzDisplacements[rank] = static_cast<int>(nnzBegin);
    }

    const index_t rowBegin = static_cast<index_t>(
        (static_cast<uint64_t>(numRows) * static_cast<uint64_t>(mpiRank)) /
        static_cast<uint64_t>(mpiSize));
    const index_t rowEnd = static_cast<index_t>(
        (static_cast<uint64_t>(numRows) * static_cast<uint64_t>(mpiRank + 1)) /
        static_cast<uint64_t>(mpiSize));
    const index_t localRows = rowEnd - rowBegin;
    const index_t localNnz = h_rowDelimiters[rowEnd] - h_rowDelimiters[rowBegin];

    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    std::vector<index_t> localRowDelimiters(localRows + 1);
    std::vector<double> localOut(localRows);

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i <= static_cast<int64_t>(localRows); ++i) {
        localRowDelimiters[i] = h_rowDelimiters[rowBegin + static_cast<index_t>(i)] -
                                h_rowDelimiters[rowBegin];
    }

    MPI_Scatterv(mpiRank == 0 ? h_val.data() : nullptr, nnzCounts.data(),
                 nnzDisplacements.data(), MPI_DOUBLE,
                 localVal.empty() ? nullptr : localVal.data(), nnzCounts[mpiRank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(mpiRank == 0 ? h_cols.data() : nullptr, nnzCounts.data(),
                 nnzDisplacements.data(), MPI_UINT32_T,
                 localCols.empty() ? nullptr : localCols.data(), nnzCounts[mpiRank],
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Compute the reference only on rank 0, exactly from the original global
    // arrays. The OpenMP loop is row-independent and retains the per-row CSR
    // accumulation order.
    std::vector<double> h_reference;
    if (validate && mpiRank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows,
                h_reference.data());
    }

    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    cudaStream_t stream = nullptr;

    if (localNnz > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_val),
                              static_cast<size_t>(localNnz) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cols),
                              static_cast<size_t>(localNnz) * sizeof(index_t)));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rowDelimiters),
                          static_cast<size_t>(localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vec),
                          static_cast<size_t>(numRows) * sizeof(double)));
    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out),
                              static_cast<size_t>(localRows) * sizeof(double)));
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if (localNnz > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_val, localVal.data(),
                                   static_cast<size_t>(localNnz) * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_cols, localCols.data(),
                                   static_cast<size_t>(localNnz) * sizeof(index_t),
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(d_rowDelimiters, localRowDelimiters.data(),
                               static_cast<size_t>(localRows + 1) * sizeof(index_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vec, h_vec.data(), static_cast<size_t>(numRows) * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (mpiRank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent, stream));

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows > 0) {
            const unsigned gridSize =
                (static_cast<unsigned>(localRows) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            spmvKernel<<<gridSize, CUDA_BLOCK_SIZE, 0, stream>>>(
                d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float localKernelMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&localKernelMilliseconds, startEvent, stopEvent));
    float kernelMilliseconds = 0.0F;
    MPI_Reduce(&localKernelMilliseconds, &kernelMilliseconds, 1, MPI_FLOAT, MPI_MAX, 0,
               MPI_COMM_WORLD);

    // The vector is invariant across iterations, so only the final output is
    // copied and gathered. This removes redundant PCIe and interconnect traffic
    // while preserving the externally visible result of the original loop.
    if (iterations > 0 && localRows > 0) {
        CUDA_CHECK(cudaMemcpyAsync(localOut.data(), d_out,
                                   static_cast<size_t>(localRows) * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    MPI_Gatherv(localOut.empty() ? nullptr : localOut.data(), rowCounts[mpiRank], MPI_DOUBLE,
                mpiRank == 0 ? h_out.data() : nullptr, rowCounts.data(), rowDisplacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (mpiRank == 0) {
        const double durationMilliseconds = static_cast<double>(kernelMilliseconds);
        printf("Computation time: %.3f ms\n", durationMilliseconds);

        // Calculate performance metrics
        const double seconds = durationMilliseconds / 1000.0;
        const double gflops = seconds > 0.0
                                  ? (2.0 * nItems * iterations) / seconds / 1e9
                                  : 0.0;
        const double avgTime = iterations > 0
                                   ? durationMilliseconds / static_cast<double>(iterations)
                                   : 0.0;

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

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    if (d_val != nullptr) {
        CUDA_CHECK(cudaFree(d_val));
    }
    if (d_cols != nullptr) {
        CUDA_CHECK(cudaFree(d_cols));
    }
    if (d_out != nullptr) {
        CUDA_CHECK(cudaFree(d_out));
    }
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return exitCode;
}
