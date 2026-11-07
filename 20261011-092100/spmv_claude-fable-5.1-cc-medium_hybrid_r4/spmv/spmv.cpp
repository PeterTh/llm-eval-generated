// Sparse Matrix-Vector Multiplication (SpMV) benchmark
//
// Hybrid parallel implementation:
//   * MPI    : the matrix rows are partitioned (nnz-balanced) across ranks,
//              each rank owns one GPU (selected by node-local rank).
//   * CUDA   : the local CSR block is multiplied on the GPU with an adaptive
//              "threads-per-row" CSR kernel (scalar/vector CSR hybrid).
//   * OpenMP : all host-side parallel phases (CSR block extraction, reference
//              solution, verification) run multi-threaded inside every rank.
//
// Matrix generation uses the exact same sequential rand() stream as the
// original code (on rank 0) so the generated problem is identical.

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

// ---------------------------------------------------------------------------
// Error handling helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err__),      \
                    __FILE__, __LINE__, cudaGetErrorString(err__));                       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

#define MPI_CHECK(call)                                                                   \
    do {                                                                                  \
        const int err__ = (call);                                                         \
        if (err__ != MPI_SUCCESS) {                                                       \
            fprintf(stderr, "MPI error %d at %s:%d\n", err__, __FILE__, __LINE__);        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// ---------------------------------------------------------------------------
// MPI helpers that transfer arbitrarily large buffers (MPI counts are int)
// ---------------------------------------------------------------------------
constexpr size_t MPI_CHUNK = size_t(1) << 28;

template <typename T>
MPI_Datatype mpiType();
template <>
MPI_Datatype mpiType<double>() { return MPI_DOUBLE; }
template <>
MPI_Datatype mpiType<uint32_t>() { return MPI_UINT32_T; }

template <typename T>
void mpiSendLarge(const T* buf, size_t count, int dest, int tag) {
    size_t off = 0;
    do {
        const size_t n = std::min(MPI_CHUNK, count - off);
        MPI_CHECK(MPI_Send(buf + off, static_cast<int>(n), mpiType<T>(), dest, tag, MPI_COMM_WORLD));
        off += n;
    } while (off < count);
}

template <typename T>
void mpiRecvLarge(T* buf, size_t count, int src, int tag) {
    size_t off = 0;
    do {
        const size_t n = std::min(MPI_CHUNK, count - off);
        MPI_CHECK(MPI_Recv(buf + off, static_cast<int>(n), mpiType<T>(), src, tag, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE));
        off += n;
    } while (off < count);
}

template <typename T>
void mpiBcastLarge(T* buf, size_t count, int root) {
    size_t off = 0;
    do {
        const size_t n = std::min(MPI_CHUNK, count - off);
        MPI_CHECK(MPI_Bcast(buf + off, static_cast<int>(n), mpiType<T>(), root, MPI_COMM_WORLD));
        off += n;
    } while (off < count);
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
//   (OpenMP-parallel over rows; used for the reference solution)
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
// Function: spmvKernel
//
// Purpose:
//   CUDA CSR SpMV kernel. TPR (threads per row) consecutive threads cooperate
//   on one row; partial sums are reduced with warp shuffles. TPR is chosen at
//   runtime from the average number of non-zeros per row, which gives
//   coalesced access to val/cols for long rows and good thread utilisation
//   for short rows.
//
// ****************************************************************************
constexpr int SPMV_BLOCK = 256;

template <int TPR>
__global__ void __launch_bounds__(SPMV_BLOCK)
spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
           const index_t* __restrict__ rowDelimiters, const double* __restrict__ vec,
           const index_t numRows, double* __restrict__ out) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t row = tid / TPR;
    const int lane = tid % TPR;

    double t = 0.0;
    if (row < numRows) {
        const index_t start = rowDelimiters[row];
        const index_t end = rowDelimiters[row + 1];
        for (index_t j = start + lane; j < end; j += TPR) {
            t += val[j] * __ldg(&vec[cols[j]]);
        }
    }

    // Reduce within the TPR-wide group (all threads of a warp participate).
    for (int off = TPR / 2; off > 0; off >>= 1) {
        t += __shfl_down_sync(0xffffffffu, t, off, TPR);
    }

    if (lane == 0 && row < numRows) {
        out[row] = t;
    }
}

template <int TPR>
void launchSpmv(const double* val, const index_t* cols, const index_t* rowDelimiters,
                const double* vec, const index_t numRows, double* out, cudaStream_t stream) {
    const size_t threads = static_cast<size_t>(numRows) * TPR;
    const unsigned blocks = static_cast<unsigned>((threads + SPMV_BLOCK - 1) / SPMV_BLOCK);
    if (blocks == 0) return;
    spmvKernel<TPR><<<blocks, SPMV_BLOCK, 0, stream>>>(val, cols, rowDelimiters, vec, numRows, out);
}

using SpmvLauncher = void (*)(const double*, const index_t*, const index_t*, const double*,
                              const index_t, double*, cudaStream_t);

SpmvLauncher selectLauncher(const size_t nnz, const index_t rows) {
    const double avg = rows > 0 ? static_cast<double>(nnz) / rows : 0.0;
    if (avg <= 2.0) return launchSpmv<2>;
    if (avg <= 4.0) return launchSpmv<4>;
    if (avg <= 8.0) return launchSpmv<8>;
    if (avg <= 16.0) return launchSpmv<16>;
    return launchSpmv<32>;
}

// ****************************************************************************
// Function: verifyResults
//
// Purpose:
//   Verifies correctness of results by comparing to reference solution
//   (OpenMP-parallel; reports the first failing index, like the original)
//
// Arguments:
//   reference: array holding the reference result vector
//   result: array holding the result vector to verify
//   size: number of elements per vector
//   globalOffset: global index of element 0 (for the diagnostic message)
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t globalOffset) {
    index_t firstBad = size;
#pragma omp parallel for schedule(static) reduction(min : firstBad)
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        bool bad;
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            bad = std::abs(res) > MAX_RELATIVE_ERROR;
        } else {
            // Check relative error
            bad = std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR;
        }
        if (bad && i < firstBad) firstBad = i;
    }

    if (firstBad == size) return true;

    const index_t i = firstBad;
    const double ref = reference[i];
    const double res = result[i];
    if (std::abs(ref) < 1e-10) {
        printf("Validation failed at index %u: reference %.10e, got %.10e\n", i + globalOffset, ref, res);
    } else {
        const double relError = std::abs((res - ref) / ref);
        printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
               i + globalOffset, ref, res, relError);
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
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank = 0, nRanks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
    const bool isRoot = (rank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks see the same argv)
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
            if (isRoot) printUsage(argv[0]);
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

    // ---- GPU selection: one GPU per rank, chosen by node-local rank --------
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_free(&nodeComm));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));  // establish context early

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
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPUs/node: %d\n",
               nRanks, omp_get_max_threads(), deviceCount);
    }

    // ---- Problem generation on rank 0 (identical rand() stream to original) -
    std::vector<double> h_vec(numRows);              // Dense vector (replicated on all ranks)
    std::vector<index_t> rowStarts(nRanks + 1, 0);   // Global row partition

    std::vector<double> g_val;                       // Full CSR (rank 0 only)
    std::vector<index_t> g_cols;
    std::vector<index_t> g_rowDelimiters;

    if (isRoot) {
        printf("Initializing data structures...\n");
        g_val.resize(nItems);
        g_cols.resize(nItems);
        g_rowDelimiters.resize(numRows + 1);
        fill(h_vec.data(), numRows, maxVal);
        fill(g_val.data(), nItems, maxVal);
        initRandomMatrix(g_cols.data(), g_rowDelimiters.data(), nItems, numRows);

        // nnz-balanced contiguous row partition
        rowStarts[0] = 0;
        for (int r = 1; r < nRanks; ++r) {
            const index_t target = static_cast<index_t>(
                (static_cast<uint64_t>(nItems) * static_cast<uint64_t>(r)) / nRanks);
            index_t lo = rowStarts[r - 1], hi = numRows;
            // first row whose start >= target
            const index_t* p = std::lower_bound(g_rowDelimiters.data() + lo,
                                                g_rowDelimiters.data() + hi, target);
            rowStarts[r] = static_cast<index_t>(p - g_rowDelimiters.data());
        }
        rowStarts[nRanks] = numRows;
    }

    mpiBcastLarge(rowStarts.data(), rowStarts.size(), 0);
    mpiBcastLarge(h_vec.data(), numRows, 0);

    const index_t rowBegin = rowStarts[rank];
    const index_t localRows = rowStarts[rank + 1] - rowBegin;

    // ---- Distribute local CSR blocks ----------------------------------------
    std::vector<index_t> h_rowDelimiters(localRows + 1);  // Local row delimiters (0-based)
    std::vector<double> h_val;                            // Local non-zero values
    std::vector<index_t> h_cols;                          // Local column indices

    if (isRoot) {
        for (int r = 1; r < nRanks; ++r) {
            const index_t rb = rowStarts[r], re = rowStarts[r + 1];
            const index_t nb = g_rowDelimiters[rb], ne = g_rowDelimiters[re];
            mpiSendLarge(g_rowDelimiters.data() + rb, static_cast<size_t>(re - rb) + 1, r, 0);
            mpiSendLarge(g_val.data() + nb, static_cast<size_t>(ne - nb), r, 1);
            mpiSendLarge(g_cols.data() + nb, static_cast<size_t>(ne - nb), r, 2);
        }
        const index_t nb = g_rowDelimiters[rowBegin];
        const index_t ne = g_rowDelimiters[rowBegin + localRows];
        const size_t localNnz = ne - nb;
        h_val.resize(localNnz);
        h_cols.resize(localNnz);
#pragma omp parallel
        {
#pragma omp for schedule(static) nowait
            for (index_t i = 0; i <= localRows; ++i) {
                h_rowDelimiters[i] = g_rowDelimiters[rowBegin + i] - nb;
            }
#pragma omp for schedule(static)
            for (size_t k = 0; k < localNnz; ++k) {
                h_val[k] = g_val[nb + k];
                h_cols[k] = g_cols[nb + k];
            }
        }
        // Release the global copies
        std::vector<double>().swap(g_val);
        std::vector<index_t>().swap(g_cols);
        std::vector<index_t>().swap(g_rowDelimiters);
    } else {
        mpiRecvLarge(h_rowDelimiters.data(), static_cast<size_t>(localRows) + 1, 0, 0);
        const index_t nb = h_rowDelimiters[0];
        const size_t localNnz = h_rowDelimiters[localRows] - nb;
        h_val.resize(localNnz);
        h_cols.resize(localNnz);
        mpiRecvLarge(h_val.data(), localNnz, 0, 1);
        mpiRecvLarge(h_cols.data(), localNnz, 0, 2);
#pragma omp parallel for schedule(static)
        for (index_t i = 0; i <= localRows; ++i) {
            h_rowDelimiters[i] -= nb;
        }
    }
    const size_t localNnz = h_val.size();

    // ---- For validation, compute reference solution (CPU, OpenMP) ----------
    std::vector<double> h_reference;
    if (validate) {
        if (isRoot) printf("Computing reference solution...\n");
        h_reference.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    // ---- Upload to GPU ------------------------------------------------------
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val, std::max<size_t>(localNnz, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, std::max<size_t>(localNnz, 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (static_cast<size_t>(localRows) + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, std::max<size_t>(numRows, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max<size_t>(localRows, 1) * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(),
                          (static_cast<size_t>(localRows) + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, std::max<size_t>(localRows, 1) * sizeof(double)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const SpmvLauncher launcher = selectLauncher(localNnz, localRows);

    // Warm-up launch (outside the timed region) so that module loading and
    // first-launch overhead do not distort the measurement.
    launcher(d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out, stream);
    CUDA_CHECK(cudaGetLastError());
    // Restore the zero-initialised output so that the result after the timed
    // loop matches the original semantics (zeros when iterations == 0).
    CUDA_CHECK(cudaMemsetAsync(d_out, 0, std::max<size_t>(localRows, 1) * sizeof(double), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ---- Perform SpMV computation -------------------------------------------
    if (isRoot) printf("Computing SpMV...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        launcher(d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // ---- Fetch local result -------------------------------------------------
    std::vector<double> h_out(localRows);  // Local output block
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost));

    if (isRoot) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // ---- Print results for external validation (gathered on rank 0) --------
    if (printResults) {
        if (isRoot) {
            std::vector<double> fullOut(numRows);
            std::copy(h_out.begin(), h_out.end(), fullOut.begin());
            for (int r = 1; r < nRanks; ++r) {
                mpiRecvLarge(fullOut.data() + rowStarts[r], rowStarts[r + 1] - rowStarts[r], r, 3);
            }
            print_results(fullOut, "OutputVector");
        } else {
            mpiSendLarge(h_out.data(), localRows, 0, 3);
        }
    }

    // ---- Validation (each rank checks its own block) ------------------------
    int exitCode = 0;
    if (validate) {
        if (isRoot) printf("Validating result...\n");
        const bool localValid = verifyResults(h_reference.data(), h_out.data(), localRows, rowBegin);
        int localFlag = localValid ? 1 : 0;
        int globalFlag = 0;
        MPI_CHECK(MPI_Allreduce(&localFlag, &globalFlag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));

        if (isRoot) {
            if (globalFlag) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        exitCode = globalFlag ? 0 : 1;
    }

    // ---- Cleanup ------------------------------------------------------------
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    MPI_Finalize();
    return exitCode;
}
