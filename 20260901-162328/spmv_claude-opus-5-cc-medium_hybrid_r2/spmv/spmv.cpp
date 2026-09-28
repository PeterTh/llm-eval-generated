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

// Threads per block for the SpMV kernel (multiple of the warp size)
constexpr int SPMV_BLOCK_SIZE = 128;
constexpr int WARP_SIZE = 32;

#define CUDA_CHECK(call)                                                                            \
    do {                                                                                            \
        const cudaError_t err_ = (call);                                                            \
        if (err_ != cudaSuccess) {                                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__,       \
                    __LINE__, cudaGetErrorString(err_));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                           \
        }                                                                                           \
    } while (0)

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
// Function: fillSlice
//
// Purpose:
//   Distributed-memory variant of fill(). Draws the exact same sequence of
//   random numbers as fill() would for an array of n elements, but only keeps
//   the values in the half-open range [first, last) which are stored at the
//   beginning of A. This lets every rank materialize just its own slice of the
//   (potentially huge) value array without any communication.
//
// ****************************************************************************
void fillSlice(double* A, const index_t n, const double maxVal, const index_t first,
               const index_t last) {
    for (index_t i = 0; i < n; ++i) {
        const double v = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
        if (i >= first && i < last) {
            A[i - first] = v;
        }
    }
}

// ****************************************************************************
// Function: initRandomMatrixSlice
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix, encoded in compressed sparse row (CSR) format. This is the
//   distributed-memory variant of the original initRandomMatrix(): it replays
//   the identical (inherently sequential) random sequence on every rank, but
//   only stores the column indices belonging to the rows owned by the calling
//   rank. The row delimiters are small (dim+1 entries) and are kept in full on
//   every rank, which is what allows the ranks to locate their slice of the
//   value/column arrays.
//
// Arguments:
//   localCols:     output vector receiving the column indexes of the elements
//                  of the locally owned rows [firstRow, lastRow)
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   n:             number of nonzero elements
//   dim:           number of rows/columns in the matrix
//   firstRow:      first row owned by this rank
//   lastRow:       one past the last row owned by this rank
//
// ****************************************************************************
void initRandomMatrixSlice(std::vector<index_t>& localCols, index_t* rowDelimiters, const index_t n,
                           const index_t dim, const index_t firstRow, const index_t lastRow) {
    index_t nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        const bool ownedRow = (i >= firstRow && i < lastRow);
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                if (ownedRow) {
                    localCols.push_back(j);
                }
                nnzAssigned++;
            }
        }
    }
    // Convention: put the number of non-zeroes at the end of the row delimiters array
    rowDelimiters[dim] = n;
}

// ****************************************************************************
// Kernel: spmvCsrVector
//
// Purpose:
//   Computes the local part of the sparse matrix-vector multiplication in CSR
//   format on the GPU. A group of THREADS_PER_ROW consecutive threads (a
//   sub-warp) cooperatively processes one row, which yields coalesced accesses
//   to the value/column arrays. The per-row partial sums are reduced with warp
//   shuffles. The group size is picked at launch time to match the average
//   number of nonzeros per row so that few lanes idle on sparse matrices.
//
// ****************************************************************************
template <int THREADS_PER_ROW>
__global__ void spmvCsrVector(const double* __restrict__ val, const index_t* __restrict__ cols,
                              const index_t* __restrict__ rowDelimiters,
                              const double* __restrict__ vec, const index_t numLocalRows,
                              double* __restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & (THREADS_PER_ROW - 1);
    const index_t groupsPerGrid = (gridDim.x * blockDim.x) / THREADS_PER_ROW;
    const index_t groupId = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_ROW;

    for (index_t row = groupId; row < numLocalRows; row += groupsPerGrid) {
        const index_t rowStart = rowDelimiters[row];
        const index_t rowEnd = rowDelimiters[row + 1];

        double t = 0.0;
        for (index_t j = rowStart + lane; j < rowEnd; j += THREADS_PER_ROW) {
            t += val[j] * __ldg(&vec[cols[j]]);
        }

        // Reduction of the partial sums within the thread group
#pragma unroll
        for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
            t += __shfl_down_sync(0xffffffffu, t, offset, THREADS_PER_ROW);
        }

        if (lane == 0) {
            out[row] = t;
        }
    }
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
//   indexOffset: global index of the first element (for reporting)
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t indexOffset) {
    // Find the lowest-indexed mismatch so the reported failure is deterministic
    // and identical to the sequential version.
    index_t firstBad = size;

#pragma omp parallel for schedule(static) reduction(min : firstBad)
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        bool bad = false;
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            bad = std::abs(res) > MAX_RELATIVE_ERROR;
        } else {
            // Check relative error
            bad = std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR;
        }
        if (bad && i < firstBad) {
            firstBad = i;
        }
    }

    if (firstBad == size) {
        return true;
    }

    const double ref = reference[firstBad];
    const double res = result[firstBad];
    if (std::abs(ref) < 1e-10) {
        printf("Validation failed at index %u: reference %.10e, got %.10e\n", firstBad + indexOffset,
               ref, res);
    } else {
        printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
               firstBad + indexOffset, ref, res, std::abs((res - ref) / ref));
    }
    return false;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int numRanks = 1;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const bool isRoot = (rank == 0);

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
            if (isRoot) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Bind this rank to one of the GPUs available on its node
    {
        MPI_Comm localComm = MPI_COMM_NULL;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
        int localRank = 0;
        MPI_Comm_rank(localComm, &localRank);
        MPI_Comm_free(&localComm);

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
        CUDA_CHECK(cudaFree(nullptr));  // force context creation outside of the timed region
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (isRoot) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads per rank: %d\n", numRanks, omp_get_max_threads());
    }

    // Row-block decomposition: rank r owns rows [firstRow, lastRow). The matrix
    // is generated with a uniform nonzero probability, so an even split of the
    // rows is also an even split of the nonzeros.
    const index_t rowsPerRank = numRows / static_cast<index_t>(numRanks);
    const index_t rowRemainder = numRows % static_cast<index_t>(numRanks);
    const index_t firstRow = static_cast<index_t>(rank) * rowsPerRank +
                             std::min(static_cast<index_t>(rank), rowRemainder);
    const index_t localRows =
        rowsPerRank + (static_cast<index_t>(rank) < rowRemainder ? 1u : 0u);
    const index_t lastRow = firstRow + localRows;

    // Allocate and initialize data structures
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters (global, small)
    std::vector<index_t> h_cols;                        // Column indices (local rows only)
    std::vector<double> h_val;                          // Non-zero values (local rows only)
    std::vector<double> h_vec(numRows);                 // Dense vector (replicated)
    std::vector<double> h_out(isRoot ? numRows : localRows);  // Output vector

    if (isRoot) {
        printf("Initializing data structures...\n");
    }

    // Determine the sparsity structure first: every rank replays the same random
    // sequence but only keeps the columns of its own rows.
    h_cols.reserve(static_cast<size_t>(localRows) * (numRows / sparsity + 1));
    initRandomMatrixSlice(h_cols, h_rowDelimiters.data(), nItems, numRows, firstRow, lastRow);

    const index_t valBegin = h_rowDelimiters[firstRow];
    const index_t valEnd = h_rowDelimiters[lastRow];
    const index_t localNnz = valEnd - valBegin;

    // Now replay the value/vector random stream (which uses the default seed,
    // i.e. the same sequence as srand(1)) and keep only the local value slice.
    srand(1);
    fill(h_vec.data(), numRows, maxVal);
    h_val.resize(localNnz);
    fillSlice(h_val.data(), nItems, maxVal, valBegin, valEnd);

    // Rebase the row delimiters of the local rows onto the local value array
    std::vector<index_t> h_localRowDelimiters(localRows + 1);
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= localRows; ++i) {
        h_localRowDelimiters[i] = h_rowDelimiters[firstRow + i] - valBegin;
    }

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (isRoot) {
            printf("Computing reference solution...\n");
        }
        h_reference.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_localRowDelimiters.data(), h_vec.data(), localRows,
                h_reference.data());
    }

    // Transfer the local problem to the GPU
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_val, std::max<size_t>(localNnz, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, std::max<size_t>(localNnz, 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, std::max<size_t>(numRows, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max<size_t>(localRows, 1) * sizeof(double)));

    if (localNnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), localNnz * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), localNnz * sizeof(index_t),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, h_localRowDelimiters.data(),
                          (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    // Pick the number of threads cooperating on a row from the average row
    // length (rounded up to a power of two, capped at the warp size), and size
    // the grid so that the whole device is filled.
    const index_t avgNnzPerRow = localRows > 0 ? (localNnz + localRows - 1) / localRows : 0;
    int threadsPerRow = 1;
    while (threadsPerRow < WARP_SIZE && static_cast<index_t>(threadsPerRow) * 2 <= avgNnzPerRow) {
        threadsPerRow *= 2;
    }

    int numBlocks = 1;
    if (localRows > 0) {
        const int groupsPerBlock = SPMV_BLOCK_SIZE / threadsPerRow;
        const long long neededBlocks =
            (static_cast<long long>(localRows) + groupsPerBlock - 1) / groupsPerBlock;
        int device = 0;
        int numSMs = 1;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
        const long long maxBlocks = static_cast<long long>(numSMs) * 32;
        numBlocks = static_cast<int>(std::max(1LL, std::min(neededBlocks, maxBlocks)));
    }

    // Launches the kernel instantiation matching the chosen group size
    const auto launchSpmv = [&]() {
        switch (threadsPerRow) {
            case 1:
                spmvCsrVector<1><<<numBlocks, SPMV_BLOCK_SIZE>>>(d_val, d_cols, d_rowDelimiters,
                                                                 d_vec, localRows, d_out);
                break;
            case 2:
                spmvCsrVector<2><<<numBlocks, SPMV_BLOCK_SIZE>>>(d_val, d_cols, d_rowDelimiters,
                                                                 d_vec, localRows, d_out);
                break;
            case 4:
                spmvCsrVector<4><<<numBlocks, SPMV_BLOCK_SIZE>>>(d_val, d_cols, d_rowDelimiters,
                                                                 d_vec, localRows, d_out);
                break;
            case 8:
                spmvCsrVector<8><<<numBlocks, SPMV_BLOCK_SIZE>>>(d_val, d_cols, d_rowDelimiters,
                                                                 d_vec, localRows, d_out);
                break;
            case 16:
                spmvCsrVector<16><<<numBlocks, SPMV_BLOCK_SIZE>>>(d_val, d_cols, d_rowDelimiters,
                                                                  d_vec, localRows, d_out);
                break;
            default:
                spmvCsrVector<32><<<numBlocks, SPMV_BLOCK_SIZE>>>(d_val, d_cols, d_rowDelimiters,
                                                                  d_vec, localRows, d_out);
                break;
        }
    };

    // Perform SpMV computation
    if (isRoot) {
        printf("Computing SpMV...\n");
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows > 0) {
            launchSpmv();
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // The slowest rank determines the wall-clock time of the parallel run.
    // Microsecond resolution is used internally because a single distributed
    // GPU iteration can be far below one millisecond.
    long long localUs = static_cast<long long>(duration.count());
    long long elapsedUs = localUs;
    MPI_Allreduce(&localUs, &elapsedUs, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    // Fetch the local results back
    double* localOut = isRoot ? h_out.data() + firstRow : h_out.data();
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(localOut, d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost));
    }

    if (isRoot) {
        const double elapsedMs = elapsedUs / 1000.0;
        printf("Computation time: %lld ms\n", elapsedUs / 1000);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (elapsedMs / 1000.0) / 1e9;
        const double avgTime = elapsedMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation (needs the full output vector)
    if (printResults) {
        std::vector<int> counts(numRanks);
        std::vector<int> displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            const index_t rf = static_cast<index_t>(r) * rowsPerRank +
                               std::min(static_cast<index_t>(r), rowRemainder);
            const index_t rc = rowsPerRank + (static_cast<index_t>(r) < rowRemainder ? 1u : 0u);
            displs[r] = static_cast<int>(rf);
            counts[r] = static_cast<int>(rc);
        }
        MPI_Gatherv(isRoot ? MPI_IN_PLACE : h_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    h_out.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (isRoot) {
            print_results(h_out, "OutputVector");
        }
    }

    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    // Validation: every rank checks its own rows against its own reference
    if (validate) {
        if (isRoot) {
            printf("Validating result...\n");
        }
        const int localValid =
            verifyResults(h_reference.data(), localOut, localRows, firstRow) ? 1 : 0;
        int valid = localValid;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (isRoot) {
            printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
