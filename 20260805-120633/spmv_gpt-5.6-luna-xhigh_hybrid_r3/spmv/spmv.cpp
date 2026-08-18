#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
            const uint64_t numEntriesLeft = static_cast<uint64_t>(dim) * dim -
                                             (static_cast<uint64_t>(i) * dim + j);
            const uint64_t needToAssign = static_cast<uint64_t>(n) - nnzAssigned;
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

// One warp processes one row.  This keeps long rows balanced while retaining
// coalesced accesses to the CSR arrays and avoids atomics because each row has
// exactly one owner.
constexpr int CUDA_THREADS_PER_BLOCK = 256;
constexpr int CUDA_WARPS_PER_BLOCK = CUDA_THREADS_PER_BLOCK / 32;

__global__ void spmvCsrKernel(const double* __restrict__ val,
                              const index_t* __restrict__ cols,
                              const index_t* __restrict__ rowDelimiters,
                              const double* __restrict__ vec,
                              const index_t rows,
                              double* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const index_t warp = static_cast<index_t>(blockIdx.x) * CUDA_WARPS_PER_BLOCK +
                         static_cast<index_t>(threadIdx.x >> 5);
    if (warp >= rows) {
        return;
    }

    double sum = 0.0;
    const index_t begin = rowDelimiters[warp];
    const index_t end = rowDelimiters[warp + 1];
    for (uint64_t j = static_cast<uint64_t>(begin) + lane; j < end; j += 32) {
        const index_t entry = static_cast<index_t>(j);
        sum += val[entry] * vec[cols[entry]];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        out[warp] = sum;
    }
}

void cudaCheck(const cudaError_t status, const char* expression, const int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "MPI rank %d: CUDA error in %s: %s\n", rank,
                     expression, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call, mpiRank)

int localMpiRank(MPI_Comm communicator, const int globalRank) {
    MPI_Comm sharedCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, globalRank,
                        MPI_INFO_NULL, &sharedCommunicator);
    int localRank = 0;
    MPI_Comm_rank(sharedCommunicator, &localRank);
    MPI_Comm_free(&sharedCommunicator);
    return localRank;
}

void abortWithMessage(const int mpiRank, const char* message) {
    if (mpiRank == 0) {
        std::fprintf(stderr, "%s\n", message);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
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
    int mpiProvided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiProvided);

    int mpiRank = 0;
    int mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    if (mpiProvided < MPI_THREAD_FUNNELED) {
        abortWithMessage(mpiRank, "MPI implementation does not provide MPI_THREAD_FUNNELED");
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

    if (numRows == 0 || sparsity == 0 || iterations == 0) {
        abortWithMessage(mpiRank,
                         "The matrix size, sparsity, and iteration count must be greater than zero");
    }
    if (numRows > static_cast<index_t>(std::numeric_limits<int>::max())) {
        abortWithMessage(mpiRank, "The matrix dimension exceeds the MPI count range");
    }

    // Calculate number of non-zero elements
    const uint64_t totalEntries = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nItemsWide = totalEntries / sparsity;
    if (nItemsWide > std::numeric_limits<index_t>::max()) {
        abortWithMessage(mpiRank, "The CSR index range exceeds uint32_t");
    }
    const index_t nItems = static_cast<index_t>(nItemsWide);

    if (mpiRank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / totalEntries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("MPI ranks: %d\n", mpiSize);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("CUDA: enabled\n");
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // The matrix is generated once on rank zero with the original random
    // sequence.  Rows are then partitioned between ranks, so no rank stores
    // or processes matrix rows belonging to another rank.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out;

    if (mpiRank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        h_out.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const index_t rowBegin = static_cast<index_t>(
        (static_cast<uint64_t>(numRows) * static_cast<uint64_t>(mpiRank)) /
        static_cast<uint64_t>(mpiSize));
    const index_t rowEnd = static_cast<index_t>(
        (static_cast<uint64_t>(numRows) * static_cast<uint64_t>(mpiRank + 1)) /
        static_cast<uint64_t>(mpiSize));
    const index_t localRows = rowEnd - rowBegin;

    std::vector<int> rowCounts(mpiSize);
    std::vector<int> rowDisplacements(mpiSize);
    std::vector<int> rowDelimiterCounts(mpiSize);
    std::vector<int> rowDelimiterDisplacements(mpiSize);
    std::vector<int> nnzCounts(mpiSize);
    std::vector<int> nnzDisplacements(mpiSize);
    if (mpiRank == 0) {
        for (int rank = 0; rank < mpiSize; ++rank) {
            const index_t begin = static_cast<index_t>(
                (static_cast<uint64_t>(numRows) * static_cast<uint64_t>(rank)) /
                static_cast<uint64_t>(mpiSize));
            const index_t end = static_cast<index_t>(
                (static_cast<uint64_t>(numRows) * static_cast<uint64_t>(rank + 1)) /
                static_cast<uint64_t>(mpiSize));
            const uint64_t rankNnz = static_cast<uint64_t>(h_rowDelimiters[end]) -
                                     static_cast<uint64_t>(h_rowDelimiters[begin]);
            if (end - begin >= static_cast<index_t>(std::numeric_limits<int>::max()) ||
                rankNnz > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
                begin > static_cast<index_t>(std::numeric_limits<int>::max()) ||
                h_rowDelimiters[begin] > static_cast<index_t>(std::numeric_limits<int>::max())) {
                abortWithMessage(mpiRank, "MPI counts exceed the supported 32-bit count range");
            }
            rowCounts[rank] = static_cast<int>(end - begin);
            rowDisplacements[rank] = static_cast<int>(begin);
            rowDelimiterCounts[rank] = rowCounts[rank] + 1;
            rowDelimiterDisplacements[rank] = rowDisplacements[rank];
            nnzCounts[rank] = static_cast<int>(rankNnz);
            nnzDisplacements[rank] = static_cast<int>(h_rowDelimiters[begin]);
        }
    }

    int localNnz = 0;
    MPI_Scatter(mpiRank == 0 ? nnzCounts.data() : nullptr, 1, MPI_INT,
                &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> localVal(static_cast<size_t>(localNnz));
    std::vector<index_t> localCols(static_cast<size_t>(localNnz));
    std::vector<index_t> localRowDelimiters(static_cast<size_t>(localRows) + 1);
    std::vector<double> localOut(localRows);

    MPI_Scatterv(mpiRank == 0 ? h_val.data() : nullptr,
                 mpiRank == 0 ? nnzCounts.data() : nullptr,
                 mpiRank == 0 ? nnzDisplacements.data() : nullptr,
                 MPI_DOUBLE, localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(mpiRank == 0 ? h_cols.data() : nullptr,
                 mpiRank == 0 ? nnzCounts.data() : nullptr,
                 mpiRank == 0 ? nnzDisplacements.data() : nullptr,
                 MPI_UINT32_T, localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(mpiRank == 0 ? h_rowDelimiters.data() : nullptr,
                 mpiRank == 0 ? rowDelimiterCounts.data() : nullptr,
                 mpiRank == 0 ? rowDelimiterDisplacements.data() : nullptr,
                 MPI_UINT32_T, localRowDelimiters.data(), static_cast<int>(localRows) + 1,
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // The reference keeps the original serial row order, while its rows are
    // independent and therefore use OpenMP safely.  Compute it before
    // releasing rank zero's global CSR replica.
    std::vector<double> h_reference;
    if (validate && mpiRank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }
    if (mpiRank == 0) {
        std::vector<double>().swap(h_val);
        std::vector<index_t>().swap(h_cols);
        std::vector<index_t>().swap(h_rowDelimiters);
    }

    // Normalize row offsets to the local CSR segment.  This host-side phase
    // deliberately uses OpenMP on every rank, including ranks with no rows.
    const index_t localNnzOffset = localRowDelimiters[0];
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= localRows; ++i) {
        localRowDelimiters[i] -= localNnzOffset;
    }

    int deviceCount = 0;
    const cudaError_t deviceQuery = cudaGetDeviceCount(&deviceCount);
    if (deviceQuery != cudaSuccess || deviceCount == 0) {
        std::fprintf(stderr, "MPI rank %d: no CUDA device is available (%s)\n", mpiRank,
                     cudaGetErrorString(deviceQuery));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int localRank = localMpiRank(MPI_COMM_WORLD, mpiRank);
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (localNnz > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_val),
                              static_cast<size_t>(localNnz) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cols),
                              static_cast<size_t>(localNnz) * sizeof(index_t)));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rowDelimiters),
                          (static_cast<size_t>(localRows) + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vec),
                          static_cast<size_t>(numRows) * sizeof(double)));
    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out),
                              static_cast<size_t>(localRows) * sizeof(double)));
    }
    if (localNnz > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_val, localVal.data(),
                                   static_cast<size_t>(localNnz) * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_cols, localCols.data(),
                                   static_cast<size_t>(localNnz) * sizeof(index_t),
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(d_rowDelimiters, localRowDelimiters.data(),
                               (static_cast<size_t>(localRows) + 1) * sizeof(index_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vec, h_vec.data(),
                               static_cast<size_t>(numRows) * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Perform SpMV computation
    if (mpiRank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        const unsigned int blocks = static_cast<unsigned int>(
            (static_cast<uint64_t>(localRows) + CUDA_WARPS_PER_BLOCK - 1) /
            CUDA_WARPS_PER_BLOCK);
        if (localRows > 0) {
            spmvCsrKernel<<<blocks, CUDA_THREADS_PER_BLOCK, 0, stream>>>(
                d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpyAsync(localOut.data(), d_out,
                                   static_cast<size_t>(localRows) * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                mpiRank == 0 ? h_out.data() : nullptr,
                mpiRank == 0 ? rowCounts.data() : nullptr,
                mpiRank == 0 ? rowDisplacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (localNnz > 0) {
        CUDA_CHECK(cudaFree(d_val));
        CUDA_CHECK(cudaFree(d_cols));
    }
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    if (localRows > 0) {
        CUDA_CHECK(cudaFree(d_out));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    if (mpiRank != 0) {
        MPI_Finalize();
        return 0;
    }

    const double durationMs = elapsedSeconds * 1000.0;

    printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate performance metrics
    const double gflops = elapsedSeconds > 0.0
                              ? (2.0 * nItems * iterations) / elapsedSeconds / 1e9
                              : 0.0;
    const double avgTime = durationMs / static_cast<double>(iterations);
    
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
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
