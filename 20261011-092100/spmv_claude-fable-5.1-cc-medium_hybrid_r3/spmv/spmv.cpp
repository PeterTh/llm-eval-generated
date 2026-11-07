// Hybrid MPI + OpenMP + CUDA sparse matrix-vector multiplication benchmark.
//
// Parallelization strategy
//   * MPI    : the CSR matrix is partitioned 1-D by rows across ranks. Row
//              boundaries are chosen so that every rank owns (almost) the same
//              number of non-zeros. Each rank drives one GPU (selected by the
//              node-local rank). Rank 0 generates the problem exactly as the
//              serial code did, scatters the matrix slices, broadcasts the
//              dense vector and gathers the result.
//   * CUDA   : every rank computes its row block with a CSR kernel. The number
//              of threads cooperating on a row (1..32) is chosen at run time
//              from the average row length; partial sums are reduced with warp
//              shuffles.
//   * OpenMP : host side work that is not bound to the sequential random
//              number stream (reference solution, verification, local row
//              pointer construction) runs multithreaded.

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

#define MPI_CHECK(call)                                                                  \
    do {                                                                                 \
        int err__ = (call);                                                              \
        if (err__ != MPI_SUCCESS) {                                                      \
            fprintf(stderr, "MPI error %d in %s at %s:%d\n", err__, #call, __FILE__,     \
                    __LINE__);                                                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
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
    // rand() carries sequential state; kept serial to reproduce the original data.
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
//   Computes sparse matrix-vector multiplication using CSR format on the host
//   (OpenMP parallel). Used for the reference solution.
//
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
#pragma omp parallel for schedule(dynamic, 256)
    for (int64_t i = 0; i < static_cast<int64_t>(dim); ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// CUDA kernels
//
// spmvCsrKernel<TPR>: TPR (threads per row, power of two <= 32) consecutive
// lanes of a warp cooperate on one row. Each lane strides over the row's
// non-zeros with coalesced loads of val/cols, the partial sums are combined
// with warp shuffles and lane 0 writes the result.
// ****************************************************************************
template <int TPR>
__global__ void __launch_bounds__(256) spmvCsrKernel(const double* __restrict__ val,
                                                     const index_t* __restrict__ cols,
                                                     const index_t* __restrict__ rowPtr,
                                                     const double* __restrict__ vec,
                                                     const index_t numRows,
                                                     double* __restrict__ out) {
    const index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t row = tid / TPR;
    const int lane = static_cast<int>(threadIdx.x) & (TPR - 1);

    if (row >= numRows) return;

    const index_t rowStart = rowPtr[row];
    const index_t rowEnd = rowPtr[row + 1];

    double t = 0.0;
    if constexpr (TPR == 32) {
        // Align the first access of the warp to a 32-element boundary so that
        // the bulk of the row is read with fully coalesced 128/256 byte lines.
        index_t j = rowStart - (rowStart & 31u) + lane;
        if (j >= rowStart && j < rowEnd) t += val[j] * __ldg(&vec[cols[j]]);
        for (j += 32; j < rowEnd; j += 32) t += val[j] * __ldg(&vec[cols[j]]);
    } else {
        for (index_t j = rowStart + lane; j < rowEnd; j += TPR) t += val[j] * __ldg(&vec[cols[j]]);
    }

    // Reduce across the TPR lanes that share this row.
#pragma unroll
    for (int offset = TPR / 2; offset > 0; offset >>= 1) {
        t += __shfl_down_sync(0xffffffffu, t, offset, TPR);
    }
    if (lane == 0) out[row] = t;
}

template <int TPR>
static void launchSpmv(const double* val, const index_t* cols, const index_t* rowPtr,
                       const double* vec, index_t numRows, double* out, cudaStream_t stream) {
    constexpr int blockSize = 256;
    const unsigned long long threads = static_cast<unsigned long long>(numRows) * TPR;
    const unsigned int blocks = static_cast<unsigned int>((threads + blockSize - 1) / blockSize);
    if (blocks == 0) return;
    spmvCsrKernel<TPR><<<blocks, blockSize, 0, stream>>>(val, cols, rowPtr, vec, numRows, out);
}

// Select threads-per-row from the average row length and launch.
static void spmvGpu(int tpr, const double* val, const index_t* cols, const index_t* rowPtr,
                    const double* vec, index_t numRows, double* out, cudaStream_t stream) {
    switch (tpr) {
        case 1: launchSpmv<1>(val, cols, rowPtr, vec, numRows, out, stream); break;
        case 2: launchSpmv<2>(val, cols, rowPtr, vec, numRows, out, stream); break;
        case 4: launchSpmv<4>(val, cols, rowPtr, vec, numRows, out, stream); break;
        case 8: launchSpmv<8>(val, cols, rowPtr, vec, numRows, out, stream); break;
        case 16: launchSpmv<16>(val, cols, rowPtr, vec, numRows, out, stream); break;
        default: launchSpmv<32>(val, cols, rowPtr, vec, numRows, out, stream); break;
    }
}

static int chooseThreadsPerRow(index_t localNnz, index_t localRows) {
    if (localRows == 0) return 32;
    const double avg = static_cast<double>(localNnz) / static_cast<double>(localRows);
    if (avg <= 2.0) return 1;
    if (avg <= 4.0) return 2;
    if (avg <= 8.0) return 4;
    if (avg <= 16.0) return 8;
    if (avg <= 32.0) return 16;
    return 32;
}

// ****************************************************************************
// Function: verifyResults
//
// Purpose:
//   Verifies correctness of results by comparing to reference solution.
//   OpenMP parallel; reports the lowest failing index like the serial code.
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    int64_t firstBad = static_cast<int64_t>(size);
#pragma omp parallel for reduction(min : firstBad) schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(size); ++i) {
        const double ref = reference[i];
        const double res = result[i];
        bool bad;
        if (std::abs(ref) < 1e-10) {
            bad = std::abs(res) > MAX_RELATIVE_ERROR;
        } else {
            bad = std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR;
        }
        if (bad && i < firstBad) firstBad = i;
    }
    if (firstBad < static_cast<int64_t>(size)) {
        const index_t i = static_cast<index_t>(firstBad);
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
        } else {
            printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                   i, ref, res, std::abs((res - ref) / ref));
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

// Bind this rank to a GPU based on its node-local rank.
static void selectDevice(int& localRank, int& localSize) {
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm));
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_size(nodeComm, &localSize));
    MPI_CHECK(MPI_Comm_free(&nodeComm));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr));  // establish the context now, outside the timed region
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank = 0, nprocs = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
    const bool isRoot = (rank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
    int parseStatus = -1;  // -1: continue, otherwise exit code
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
            parseStatus = 0;
            break;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
            break;
        }
    }
    if (parseStatus >= 0) {
        MPI_CHECK(MPI_Finalize());
        return parseStatus;
    }

    int localRank = 0, localSize = 1;
    selectDevice(localRank, localSize);

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
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPUs on node 0: %d\n",
               nprocs, omp_get_max_threads(), [] { int c = 0; cudaGetDeviceCount(&c); return c; }());
    }

    // ------------------------------------------------------------------------
    // Problem generation on rank 0 (bit-identical to the serial program).
    // ------------------------------------------------------------------------
    std::vector<double> h_val;           // Non-zero values (root only: full)
    std::vector<index_t> h_cols;         // Column indices (root only: full)
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters (all ranks)
    std::vector<double> h_vec(numRows);  // Dense vector (all ranks)
    std::vector<double> h_out;           // Output vector (root only: full)

    if (isRoot) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_out.resize(numRows);
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // For validation, compute reference solution (OpenMP, root only)
    std::vector<double> h_reference;
    if (validate && isRoot) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // ------------------------------------------------------------------------
    // Distribute: broadcast row pointer + vector, nnz-balanced row partition,
    // scatter the matrix slices.
    // ------------------------------------------------------------------------
    MPI_CHECK(MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD));

    // Row boundaries: rank r owns rows [rowStart[r], rowStart[r+1]) chosen so
    // that the nnz prefix at the boundary is >= r * nItems / nprocs.
    std::vector<index_t> rowStart(nprocs + 1);
    rowStart[0] = 0;
    rowStart[nprocs] = numRows;
    for (int r = 1; r < nprocs; ++r) {
        const index_t target = static_cast<index_t>((static_cast<uint64_t>(nItems) * r) / nprocs);
        const auto it = std::lower_bound(h_rowDelimiters.begin(), h_rowDelimiters.end(), target);
        index_t row = static_cast<index_t>(it - h_rowDelimiters.begin());
        row = std::min(row, numRows);
        row = std::max(row, rowStart[r - 1]);
        rowStart[r] = row;
    }

    std::vector<int> rowCounts(nprocs), rowDispls(nprocs), nnzCounts(nprocs), nnzDispls(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        rowCounts[r] = static_cast<int>(rowStart[r + 1] - rowStart[r]);
        rowDispls[r] = static_cast<int>(rowStart[r]);
        nnzDispls[r] = static_cast<int>(h_rowDelimiters[rowStart[r]]);
        nnzCounts[r] = static_cast<int>(h_rowDelimiters[rowStart[r + 1]] - h_rowDelimiters[rowStart[r]]);
    }

    const index_t myRowStart = rowStart[rank];
    const index_t myRows = static_cast<index_t>(rowCounts[rank]);
    const index_t myNnzStart = h_rowDelimiters[myRowStart];
    const index_t myNnz = static_cast<index_t>(nnzCounts[rank]);

    // Local CSR slice (host, pinned for fast H2D/D2H transfers)
    double* l_val = nullptr;
    index_t* l_cols = nullptr;
    index_t* l_rowPtr = nullptr;
    double* l_out = nullptr;
    CUDA_CHECK(cudaMallocHost(&l_val, std::max<size_t>(myNnz, 1) * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&l_cols, std::max<size_t>(myNnz, 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMallocHost(&l_rowPtr, (static_cast<size_t>(myRows) + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMallocHost(&l_out, std::max<size_t>(myRows, 1) * sizeof(double)));

    MPI_CHECK(MPI_Scatterv(isRoot ? h_val.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                           l_val, static_cast<int>(myNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatterv(isRoot ? h_cols.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_UINT32_T,
                           l_cols, static_cast<int>(myNnz), MPI_UINT32_T, 0, MPI_COMM_WORLD));

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i <= static_cast<int64_t>(myRows); ++i) {
        l_rowPtr[i] = h_rowDelimiters[myRowStart + i] - myNnzStart;
    }

    // Root no longer needs the full matrix once it has been scattered.
    if (isRoot) {
        std::vector<double>().swap(h_val);
        std::vector<index_t>().swap(h_cols);
    }

    // ------------------------------------------------------------------------
    // Device setup
    // ------------------------------------------------------------------------
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowPtr = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val, std::max<size_t>(myNnz, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, std::max<size_t>(myNnz, 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowPtr, (static_cast<size_t>(myRows) + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, std::max<size_t>(numRows, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max<size_t>(myRows, 1) * sizeof(double)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(d_val, l_val, myNnz * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_cols, l_cols, myNnz * sizeof(index_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_rowPtr, l_rowPtr, (myRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int tpr = chooseThreadsPerRow(myNnz, myRows);

    // Warm-up launch (not timed) so that module loading does not skew timing.
    spmvGpu(tpr, d_val, d_cols, d_rowPtr, d_vec, myRows, d_out, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ------------------------------------------------------------------------
    // Perform SpMV computation
    // ------------------------------------------------------------------------
    if (isRoot) printf("Computing SpMV...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvGpu(tpr, d_val, d_cols, d_rowPtr, d_vec, myRows, d_out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long local_ms = duration.count();
    long global_ms = 0;
    MPI_CHECK(MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD));

    // Collect the distributed result on rank 0
    CUDA_CHECK(cudaMemcpyAsync(l_out, d_out, myRows * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Gatherv(l_out, static_cast<int>(myRows), MPI_DOUBLE,
                          isRoot ? h_out.data() : nullptr, rowCounts.data(), rowDispls.data(), MPI_DOUBLE,
                          0, MPI_COMM_WORLD));

    int exitCode = 0;
    if (isRoot) {
        printf("Computation time: %ld ms\n", global_ms);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (global_ms / 1000.0) / 1e9;
        const double avgTime = global_ms / static_cast<double>(iterations);

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
    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(l_val));
    CUDA_CHECK(cudaFreeHost(l_cols));
    CUDA_CHECK(cudaFreeHost(l_rowPtr));
    CUDA_CHECK(cudaFreeHost(l_out));

    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
