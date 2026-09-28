#include <algorithm>
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

// Largest element count transferred in a single MPI call (keeps counts within int range)
constexpr size_t MPI_CHUNK = 1u << 28;

// ****************************************************************************
// Error checking helpers
// ****************************************************************************
#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        const cudaError_t err_ = (call);                                                     \
        if (err_ != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,  \
                    __LINE__);                                                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                    \
        }                                                                                    \
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
//   Computes sparse matrix-vector multiplication using CSR format.
//   Rows are independent, so the loop is distributed over OpenMP threads.
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
// Kernel: spmvKernel
//
// Purpose:
//   CSR SpMV on the GPU. A group of THREADS_PER_ROW consecutive threads
//   cooperates on one row; the group size is chosen at launch time from the
//   average number of non-zeros per row so that memory accesses stay
//   coalesced without wasting lanes on short rows.
//
//   rowPtr is rebased to the local non-zero slice, i.e. rowPtr[0] == 0.
// ****************************************************************************
template <int THREADS_PER_ROW>
__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowPtr, const double* __restrict__ vec,
                           const index_t numLocalRows, double* __restrict__ out) {
    const unsigned int laneInWarp = threadIdx.x & 31u;
    const int lane = static_cast<int>(laneInWarp) % THREADS_PER_ROW;

    // All threads of a group work on the same row, so the group mask is uniform
    const unsigned int groupMask =
        (THREADS_PER_ROW == 32)
            ? 0xffffffffu
            : (((1u << THREADS_PER_ROW) - 1u) << (laneInWarp - static_cast<unsigned>(lane)));

    const index_t groupsPerBlock = blockDim.x / THREADS_PER_ROW;
    const index_t group = blockIdx.x * groupsPerBlock + threadIdx.x / THREADS_PER_ROW;
    const index_t stride = groupsPerBlock * gridDim.x;

    for (index_t row = group; row < numLocalRows; row += stride) {
        const index_t rowBegin = rowPtr[row];
        const index_t rowEnd = rowPtr[row + 1];

        double t = 0.0;
        for (index_t j = rowBegin + lane; j < rowEnd; j += THREADS_PER_ROW) {
            t += val[j] * __ldg(&vec[cols[j]]);
        }

#pragma unroll
        for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
            t += __shfl_down_sync(groupMask, t, offset, THREADS_PER_ROW);
        }

        if (lane == 0) {
            out[row] = t;
        }
    }
}

// ****************************************************************************
// Function: launchSpmv
//
// Purpose:
//   Dispatches the templated SpMV kernel with a group size matching the
//   average row length.
// ****************************************************************************
void launchSpmv(const double* d_val, const index_t* d_cols, const index_t* d_rowPtr,
                const double* d_vec, const index_t numLocalRows, double* d_out,
                const double avgNnzPerRow, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    constexpr index_t MAX_BLOCKS = 65535 * 8;

    if (numLocalRows == 0) {
        return;
    }

    const auto launch = [&](auto threadsPerRow) {
        constexpr int T = decltype(threadsPerRow)::value;
        const index_t groupsPerBlock = BLOCK / T;
        const index_t blocks = std::min<index_t>(
            MAX_BLOCKS, (numLocalRows + groupsPerBlock - 1) / groupsPerBlock);
        spmvKernel<T><<<blocks, BLOCK, 0, stream>>>(d_val, d_cols, d_rowPtr, d_vec, numLocalRows,
                                                    d_out);
    };

    if (avgNnzPerRow <= 4.0) {
        launch(std::integral_constant<int, 2>{});
    } else if (avgNnzPerRow <= 8.0) {
        launch(std::integral_constant<int, 4>{});
    } else if (avgNnzPerRow <= 16.0) {
        launch(std::integral_constant<int, 8>{});
    } else if (avgNnzPerRow <= 32.0) {
        launch(std::integral_constant<int, 16>{});
    } else {
        launch(std::integral_constant<int, 32>{});
    }
}

// ****************************************************************************
// Function: verifyResults
//
// Purpose:
//   Verifies correctness of results by comparing to reference solution.
//   The scan runs in parallel; the smallest mismatching index is reported so
//   that the diagnostic output matches the sequential scan.
//
// Arguments:
//   reference: array holding the reference result vector
//   result: array holding the result vector to verify
//   size: number of elements per vector
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
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
        printf("Validation failed at index %u: reference %.10e, got %.10e\n", firstBad, ref, res);
    } else {
        const double relError = std::abs((res - ref) / ref);
        printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
               firstBad, ref, res, relError);
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

// ****************************************************************************
// Function: partitionByNnz
//
// Purpose:
//   Splits the row range [rowBegin, rowEnd) into `parts` contiguous blocks
//   holding a comparable number of non-zeros each, which balances the SpMV
//   work between MPI ranks and between GPUs.
//
// Returns:
//   Vector of parts+1 row boundaries.
// ****************************************************************************
std::vector<index_t> partitionByNnz(const index_t* rowDelimiters, const index_t rowBegin,
                                    const index_t rowEnd, const int parts) {
    std::vector<index_t> bounds(parts + 1);
    bounds[0] = rowBegin;
    bounds[parts] = rowEnd;

    const index_t nnzBegin = rowDelimiters[rowBegin];
    const index_t nnzTotal = rowDelimiters[rowEnd] - nnzBegin;

    for (int p = 1; p < parts; ++p) {
        const index_t target =
            nnzBegin + static_cast<index_t>((static_cast<uint64_t>(nnzTotal) * p) / parts);
        const index_t* it =
            std::lower_bound(rowDelimiters + bounds[p - 1], rowDelimiters + rowEnd, target);
        index_t row = static_cast<index_t>(it - rowDelimiters);

        // Keep boundaries monotonic and leave one row per remaining part where possible
        const index_t remaining = static_cast<index_t>(parts - p);
        const index_t maxRow = (rowEnd - rowBegin >= remaining) ? rowEnd - remaining : rowBegin;
        bounds[p] = std::max(bounds[p - 1], std::min(row, maxRow));
    }
    return bounds;
}

// Sends/receives element counts that may exceed the int range of MPI counts.
// Sends are posted non-blocking so that transfers to all ranks can overlap.
template <typename T>
void sendLarge(const T* buf, const size_t count, const int dest, const int tag, MPI_Datatype type,
               std::vector<MPI_Request>& requests) {
    for (size_t off = 0; off < count; off += MPI_CHUNK) {
        const int n = static_cast<int>(std::min<size_t>(MPI_CHUNK, count - off));
        requests.emplace_back();
        MPI_Isend(buf + off, n, type, dest, tag, MPI_COMM_WORLD, &requests.back());
    }
}

template <typename T>
void recvLarge(T* buf, const size_t count, const int src, const int tag, MPI_Datatype type) {
    for (size_t off = 0; off < count; off += MPI_CHUNK) {
        const int n = static_cast<int>(std::min<size_t>(MPI_CHUNK, count - off));
        MPI_Recv(buf + off, n, type, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// Per-GPU state of one MPI rank
struct DeviceContext {
    int device = 0;
    index_t rowBegin = 0;  // rank-local first row
    index_t rowEnd = 0;    // rank-local last row (exclusive)
    index_t numRows = 0;
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowPtr = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    cudaStream_t stream = nullptr;
};

int main(int argc, char** argv) {
    int mpiThreadLevel = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiThreadLevel);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    const bool isRoot = (rank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (isRoot) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ------------------------------------------------------------------
    // Data generation
    //
    // The random matrix/vector must match the sequential generator bit for
    // bit, so the root generates the problem and distributes the slices.
    // ------------------------------------------------------------------
    std::vector<double> h_val;                          // Non-zero values (root: full matrix)
    std::vector<index_t> h_cols;                        // Column indices (root: full matrix)
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters (all ranks)
    std::vector<double> h_vec(numRows);                 // Dense vector (all ranks)
    std::vector<double> h_out;                          // Output vector (root: full)

    if (isRoot) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_out.resize(numRows);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && isRoot) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows,
                h_reference.data());
    }

    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Row blocks per rank, balanced by non-zero count
    const std::vector<index_t> rankBounds =
        partitionByNnz(h_rowDelimiters.data(), 0, numRows, numRanks);
    const index_t myRowBegin = rankBounds[rank];
    const index_t myRowEnd = rankBounds[rank + 1];
    const index_t myNumRows = myRowEnd - myRowBegin;
    const index_t myNnzBegin = h_rowDelimiters[myRowBegin];
    const index_t myNnz = h_rowDelimiters[myRowEnd] - myNnzBegin;

    // Distribute the matrix slices
    std::vector<double> localValStorage;
    std::vector<index_t> localColsStorage;
    const double* localVal = nullptr;
    const index_t* localCols = nullptr;

    if (isRoot) {
        localVal = h_val.data() + myNnzBegin;
        localCols = h_cols.data() + myNnzBegin;
        std::vector<MPI_Request> requests;
        for (int r = 1; r < numRanks; ++r) {
            const size_t begin = h_rowDelimiters[rankBounds[r]];
            const size_t count = h_rowDelimiters[rankBounds[r + 1]] - begin;
            sendLarge(h_val.data() + begin, count, r, 100, MPI_DOUBLE, requests);
            sendLarge(h_cols.data() + begin, count, r, 101, MPI_UINT32_T, requests);
        }
        if (!requests.empty()) {
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        }
    } else {
        localValStorage.resize(myNnz);
        localColsStorage.resize(myNnz);
        recvLarge(localValStorage.data(), myNnz, 0, 100, MPI_DOUBLE);
        recvLarge(localColsStorage.data(), myNnz, 0, 101, MPI_UINT32_T);
        localVal = localValStorage.data();
        localCols = localColsStorage.data();
    }

    // ------------------------------------------------------------------
    // GPU setup: split the rank's rows over the GPUs it owns
    //
    // Devices are shared out among the ranks of a node, so a single rank
    // drives every local GPU while one rank per GPU gets exactly one each.
    // ------------------------------------------------------------------
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (isRoot) {
            fprintf(stderr, "No CUDA devices available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    int nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);

    const int devicesPerRank = std::max(1, deviceCount / nodeSize);
    const int firstDevice =
        (nodeSize >= deviceCount) ? (nodeRank % deviceCount) : (nodeRank * devicesPerRank);
    const int myDeviceCount =
        (nodeSize >= deviceCount)
            ? 1
            : std::min(devicesPerRank, deviceCount - firstDevice);

    // Never create more device partitions than there are rows to compute
    const int numParts = std::max(
        1, std::min<int>(myDeviceCount, static_cast<int>(std::min<index_t>(myNumRows, 1u << 20))));
    const std::vector<index_t> devBounds =
        partitionByNnz(h_rowDelimiters.data(), myRowBegin, myRowEnd, numParts);

    std::vector<DeviceContext> ctx(numParts);
    const double avgNnzPerRow =
        (numRows > 0) ? static_cast<double>(nItems) / static_cast<double>(numRows) : 0.0;

    // Pinned staging buffer for the rank's part of the result vector
    double* h_localOut = nullptr;
    if (myNumRows > 0) {
        CUDA_CHECK(cudaHostAlloc(&h_localOut, myNumRows * sizeof(double), cudaHostAllocDefault));
    }

    // One OpenMP thread per GPU owns that GPU for the whole run
#pragma omp parallel num_threads(numParts)
    {
        const int p = omp_get_thread_num();
        DeviceContext& c = ctx[p];
        c.device = firstDevice + (p % std::max(1, myDeviceCount));
        c.rowBegin = devBounds[p];
        c.rowEnd = devBounds[p + 1];
        c.numRows = c.rowEnd - c.rowBegin;

        CUDA_CHECK(cudaSetDevice(c.device));
        CUDA_CHECK(cudaStreamCreate(&c.stream));

        const index_t nnzBegin = h_rowDelimiters[c.rowBegin];
        const index_t nnz = h_rowDelimiters[c.rowEnd] - nnzBegin;

        // Row pointers rebased to this device's non-zero slice
        std::vector<index_t> rowPtr(c.numRows + 1);
        for (index_t i = 0; i <= c.numRows; ++i) {
            rowPtr[i] = h_rowDelimiters[c.rowBegin + i] - nnzBegin;
        }

        CUDA_CHECK(cudaMalloc(&c.d_val, std::max<size_t>(1, nnz) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.d_cols, std::max<size_t>(1, nnz) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&c.d_rowPtr, (c.numRows + 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&c.d_vec, std::max<size_t>(1, numRows) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.d_out, std::max<size_t>(1, c.numRows) * sizeof(double)));

        const index_t localOffset = c.rowBegin - myRowBegin;
        CUDA_CHECK(cudaMemcpyAsync(c.d_val, localVal + (nnzBegin - myNnzBegin),
                                   nnz * sizeof(double), cudaMemcpyHostToDevice, c.stream));
        CUDA_CHECK(cudaMemcpyAsync(c.d_cols, localCols + (nnzBegin - myNnzBegin),
                                   nnz * sizeof(index_t), cudaMemcpyHostToDevice, c.stream));
        CUDA_CHECK(cudaMemcpyAsync(c.d_rowPtr, rowPtr.data(), (c.numRows + 1) * sizeof(index_t),
                                   cudaMemcpyHostToDevice, c.stream));
        CUDA_CHECK(cudaMemcpyAsync(c.d_vec, h_vec.data(), numRows * sizeof(double),
                                   cudaMemcpyHostToDevice, c.stream));
        CUDA_CHECK(cudaStreamSynchronize(c.stream));

        // Warm up the device (context, clocks, caches) before timing
        launchSpmv(c.d_val, c.d_cols, c.d_rowPtr, c.d_vec, c.numRows, c.d_out, avgNnzPerRow,
                   c.stream);
        CUDA_CHECK(cudaStreamSynchronize(c.stream));

        // ------------------------------------------------------------------
        // SpMV computation
        // ------------------------------------------------------------------
#pragma omp master
        {
            if (isRoot) {
                printf("Computing SpMV...\n");
            }
        }
#pragma omp barrier
#pragma omp master
        { MPI_Barrier(MPI_COMM_WORLD); }
#pragma omp barrier

        const auto start = std::chrono::high_resolution_clock::now();

        for (index_t iter = 0; iter < iterations; ++iter) {
            launchSpmv(c.d_val, c.d_cols, c.d_rowPtr, c.d_vec, c.numRows, c.d_out, avgNnzPerRow,
                       c.stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(c.stream));
        CUDA_CHECK(cudaGetLastError());

#pragma omp barrier
        const auto end = std::chrono::high_resolution_clock::now();

#pragma omp master
        {
            const long long localUs =
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            long long maxUs = localUs;
            MPI_Allreduce(&localUs, &maxUs, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

            const double seconds = static_cast<double>(maxUs) / 1e6;
            if (isRoot) {
                printf("Computation time: %ld ms\n", static_cast<long>(maxUs / 1000));

                // Calculate performance metrics
                const double gflops = (2.0 * nItems * iterations) / seconds / 1e9;
                const double avgTime = (seconds * 1000.0) / static_cast<double>(iterations);

                printf("Average time per iteration: %.3f ms\n", avgTime);
                printf("Performance: %.3f GFLOPS\n", gflops);
            }
        }

        // Copy this device's rows into the rank's staging buffer
        if (c.numRows > 0) {
            CUDA_CHECK(cudaMemcpyAsync(h_localOut + localOffset, c.d_out,
                                       c.numRows * sizeof(double), cudaMemcpyDeviceToHost,
                                       c.stream));
            CUDA_CHECK(cudaStreamSynchronize(c.stream));
        }

        CUDA_CHECK(cudaFree(c.d_val));
        CUDA_CHECK(cudaFree(c.d_cols));
        CUDA_CHECK(cudaFree(c.d_rowPtr));
        CUDA_CHECK(cudaFree(c.d_vec));
        CUDA_CHECK(cudaFree(c.d_out));
        CUDA_CHECK(cudaStreamDestroy(c.stream));
    }

    // Collect the full result vector on the root
    {
        std::vector<int> counts(numRanks);
        std::vector<int> displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            counts[r] = static_cast<int>(rankBounds[r + 1] - rankBounds[r]);
            displs[r] = static_cast<int>(rankBounds[r]);
        }
        MPI_Gatherv(h_localOut, static_cast<int>(myNumRows), MPI_DOUBLE,
                    isRoot ? h_out.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (isRoot) {
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

    if (h_localOut != nullptr) {
        CUDA_CHECK(cudaFreeHost(h_localOut));
    }
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return exitCode;
}
