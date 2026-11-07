// Hybrid MPI + OpenMP + CUDA sparse matrix-vector multiplication benchmark.
//
// Parallelization strategy
//   * MPI     : the matrix rows are partitioned across ranks so that every rank
//               owns (approximately) the same number of non-zeros. Every rank
//               generates the matrix deterministically (identical to the serial
//               code) and keeps only its row slice on the device(s).
//   * OpenMP  : inside a rank, one host thread per local GPU drives the device
//               (multi-GPU per rank); OpenMP is also used for the CPU reference
//               solution used for validation.
//   * CUDA    : CSR SpMV kernel with an adaptive number of threads per row
//               (1..32, chosen from the average row length) and warp-shuffle
//               reductions.
//
// This source is compiled as CUDA (see CMakeLists.txt).

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

#define MPI_CHECK(call)                                                                    \
    do {                                                                                   \
        int err__ = (call);                                                                \
        if (err__ != MPI_SUCCESS) {                                                        \
            fprintf(stderr, "MPI error %d in %s at %s:%d\n", err__, #call, __FILE__,       \
                    __LINE__);                                                             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
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
//   Computes sparse matrix-vector multiplication using CSR format on the host
//   (OpenMP-parallel over rows). Used for the reference solution.
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
// CUDA kernel: CSR SpMV with T threads cooperating on each row.
//
// T is a power of two in [1, 32]. Threads of a row are consecutive lanes of a
// warp, so the partial sums are reduced with warp shuffles. All lanes of a warp
// take part in the shuffle (out-of-range rows simply iterate over an empty
// range) so the full mask can be used safely.
// ****************************************************************************
template <int T>
__global__ void __launch_bounds__(256)
spmvCsrKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
              const index_t* __restrict__ rowDelimiters, const double* __restrict__ vec,
              const index_t nRows, double* __restrict__ out) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned row = tid / T;
    const unsigned lane = threadIdx.x % T;

    index_t start = 0, end = 0;
    if (row < nRows) {
        start = rowDelimiters[row];
        end = rowDelimiters[row + 1];
    }

    double t = 0.0;
    index_t j = start + lane;
    // Main loop, unrolled by 4 to expose memory-level parallelism
    for (; j + 3 * T < end; j += 4 * T) {
        t += val[j] * __ldg(&vec[cols[j]]);
        t += val[j + T] * __ldg(&vec[cols[j + T]]);
        t += val[j + 2 * T] * __ldg(&vec[cols[j + 2 * T]]);
        t += val[j + 3 * T] * __ldg(&vec[cols[j + 3 * T]]);
    }
    for (; j < end; j += T) {
        t += val[j] * __ldg(&vec[cols[j]]);
    }

    if constexpr (T > 1) {
#pragma unroll
        for (int offset = T / 2; offset > 0; offset >>= 1) {
            t += __shfl_down_sync(0xffffffffu, t, offset, T);
        }
    }

    if (lane == 0 && row < nRows) {
        out[row] = t;
    }
}

// Per-GPU state of a rank
struct GpuSlice {
    int device = -1;
    index_t rowBegin = 0;   // first global row handled by this GPU
    index_t nRows = 0;      // number of rows
    index_t nnz = 0;        // number of non-zeros
    int threadsPerRow = 1;
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    cudaStream_t stream = nullptr;
};

// Choose the number of threads per row from the average row length
static int chooseThreadsPerRow(const index_t nRows, const index_t nnz) {
    if (nRows == 0) return 1;
    const double avg = static_cast<double>(nnz) / static_cast<double>(nRows);
    if (avg <= 2.0) return 1;
    if (avg <= 4.0) return 2;
    if (avg <= 8.0) return 4;
    if (avg <= 16.0) return 8;
    if (avg <= 32.0) return 16;
    return 32;
}

// Launch the SpMV kernel for one GPU slice on its stream
static void launchSpmv(const GpuSlice& g) {
    if (g.nRows == 0) return;
    constexpr int blockSize = 256;
    const unsigned long long totalThreads =
        static_cast<unsigned long long>(g.nRows) * static_cast<unsigned long long>(g.threadsPerRow);
    const unsigned grid = static_cast<unsigned>((totalThreads + blockSize - 1) / blockSize);
    switch (g.threadsPerRow) {
        case 1:
            spmvCsrKernel<1><<<grid, blockSize, 0, g.stream>>>(g.d_val, g.d_cols, g.d_rowDelimiters,
                                                              g.d_vec, g.nRows, g.d_out);
            break;
        case 2:
            spmvCsrKernel<2><<<grid, blockSize, 0, g.stream>>>(g.d_val, g.d_cols, g.d_rowDelimiters,
                                                              g.d_vec, g.nRows, g.d_out);
            break;
        case 4:
            spmvCsrKernel<4><<<grid, blockSize, 0, g.stream>>>(g.d_val, g.d_cols, g.d_rowDelimiters,
                                                              g.d_vec, g.nRows, g.d_out);
            break;
        case 8:
            spmvCsrKernel<8><<<grid, blockSize, 0, g.stream>>>(g.d_val, g.d_cols, g.d_rowDelimiters,
                                                              g.d_vec, g.nRows, g.d_out);
            break;
        case 16:
            spmvCsrKernel<16><<<grid, blockSize, 0, g.stream>>>(g.d_val, g.d_cols, g.d_rowDelimiters,
                                                               g.d_vec, g.nRows, g.d_out);
            break;
        default:
            spmvCsrKernel<32><<<grid, blockSize, 0, g.stream>>>(g.d_val, g.d_cols, g.d_rowDelimiters,
                                                               g.d_vec, g.nRows, g.d_out);
            break;
    }
}

// ****************************************************************************
// Function: partitionRows
//
// Purpose:
//   Split the row range [rowBegin, rowEnd) into `parts` contiguous pieces with
//   (approximately) equal numbers of non-zeros. Returns parts+1 boundaries.
//
// ****************************************************************************
static std::vector<index_t> partitionRows(const index_t* rowDelimiters, const index_t rowBegin,
                                          const index_t rowEnd, const int parts) {
    std::vector<index_t> bounds(parts + 1);
    bounds[0] = rowBegin;
    bounds[parts] = rowEnd;
    const uint64_t nnzBegin = rowDelimiters[rowBegin];
    const uint64_t nnzTotal = rowDelimiters[rowEnd] - nnzBegin;
    for (int p = 1; p < parts; ++p) {
        const uint64_t target = nnzBegin + (nnzTotal * static_cast<uint64_t>(p)) / parts;
        const index_t* it = std::lower_bound(rowDelimiters + rowBegin, rowDelimiters + rowEnd,
                                             static_cast<index_t>(target));
        index_t b = static_cast<index_t>(it - rowDelimiters);
        b = std::max(b, bounds[p - 1]);
        b = std::min(b, rowEnd);
        bounds[p] = b;
    }
    return bounds;
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
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int worldRank = 0, worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    const bool isRoot = (worldRank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identically on every rank)
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

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    // ---------------------------------------------------------------------
    // GPU assignment: ranks sharing a node take disjoint subsets of the
    // node's GPUs (device d belongs to local rank d % localSize).
    // ---------------------------------------------------------------------
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL,
                                  &nodeComm));
    int localRank = 0, localSize = 1;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_size(nodeComm, &localSize));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (isRoot) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> myDevices;
    for (int d = localRank; d < deviceCount; d += localSize) myDevices.push_back(d);
    if (myDevices.empty()) myDevices.push_back(localRank % deviceCount);  // oversubscribed GPUs
    const int nGpus = static_cast<int>(myDevices.size());

    int totalGpus = 0;
    MPI_CHECK(MPI_Allreduce(&nGpus, &totalGpus, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));

    if (isRoot) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs: %d, OpenMP threads: %d\n", worldSize, totalGpus,
               omp_get_max_threads());
    }

    // Allocate and initialize data structures. Generation is deterministic
    // (fixed seeds, sequential rand()), so every rank builds the identical
    // matrix without any communication.
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out;                          // Output vector (root only)

    if (isRoot) printf("Initializing data structures...\n");
    fill(h_vec.data(), numRows, maxVal);
    fill(h_val.data(), nItems, maxVal);
    initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

    // For validation, compute reference solution (root only, OpenMP-parallel)
    std::vector<double> h_reference;
    if (validate && isRoot) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows,
                h_reference.data());
    }

    // ---------------------------------------------------------------------
    // Work distribution: balanced-nnz row partition across ranks, then across
    // the GPUs of each rank.
    // ---------------------------------------------------------------------
    const std::vector<index_t> rankBounds = partitionRows(h_rowDelimiters.data(), 0, numRows, worldSize);
    const index_t myRowBegin = rankBounds[worldRank];
    const index_t myRowEnd = rankBounds[worldRank + 1];
    const index_t myRows = myRowEnd - myRowBegin;

    const std::vector<index_t> gpuBounds =
        partitionRows(h_rowDelimiters.data(), myRowBegin, myRowEnd, nGpus);

    std::vector<GpuSlice> gpus(nGpus);
    for (int g = 0; g < nGpus; ++g) {
        gpus[g].device = myDevices[g];
        gpus[g].rowBegin = gpuBounds[g];
        gpus[g].nRows = gpuBounds[g + 1] - gpuBounds[g];
        gpus[g].nnz = h_rowDelimiters[gpuBounds[g + 1]] - h_rowDelimiters[gpuBounds[g]];
        gpus[g].threadsPerRow = chooseThreadsPerRow(gpus[g].nRows, gpus[g].nnz);
    }

    // Upload the local slices: one OpenMP thread per GPU
#pragma omp parallel for num_threads(nGpus) schedule(static)
    for (int g = 0; g < nGpus; ++g) {
        GpuSlice& s = gpus[g];
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));

        // Local row delimiters, rebased to the slice's first non-zero
        std::vector<index_t> localRowDelimiters(s.nRows + 1);
        const index_t nnzOffset = h_rowDelimiters[s.rowBegin];
        for (index_t i = 0; i <= s.nRows; ++i) {
            localRowDelimiters[i] = h_rowDelimiters[s.rowBegin + i] - nnzOffset;
        }

        CUDA_CHECK(cudaMalloc(&s.d_val, std::max<size_t>(s.nnz, 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.d_cols, std::max<size_t>(s.nnz, 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&s.d_rowDelimiters, (s.nRows + 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&s.d_vec, std::max<size_t>(numRows, 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.d_out, std::max<size_t>(s.nRows, 1) * sizeof(double)));

        CUDA_CHECK(cudaMemcpyAsync(s.d_val, h_val.data() + nnzOffset, s.nnz * sizeof(double),
                                   cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_cols, h_cols.data() + nnzOffset, s.nnz * sizeof(index_t),
                                   cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_rowDelimiters, localRowDelimiters.data(),
                                   (s.nRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice,
                                   s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_vec, h_vec.data(), numRows * sizeof(double),
                                   cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaStreamSynchronize(s.stream));

        // Warm-up launch (module load, cache fill) outside the timed region
        launchSpmv(s);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
    }

    // Release the full host matrix on non-root ranks (root keeps it for the
    // reference / result handling).
    if (!isRoot) {
        std::vector<double>().swap(h_val);
        std::vector<index_t>().swap(h_cols);
    }

    // ---------------------------------------------------------------------
    // Perform SpMV computation
    // ---------------------------------------------------------------------
    if (isRoot) printf("Computing SpMV...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(nGpus)
    {
        const int g = omp_get_thread_num();
        GpuSlice& s = gpus[g];
        CUDA_CHECK(cudaSetDevice(s.device));
        for (index_t iter = 0; iter < iterations; ++iter) {
            launchSpmv(s);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (isRoot) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // ---------------------------------------------------------------------
    // Collect the result: device -> host (per GPU), then gather on root
    // ---------------------------------------------------------------------
    std::vector<double> h_localOut(std::max<index_t>(myRows, 1));
#pragma omp parallel for num_threads(nGpus) schedule(static)
    for (int g = 0; g < nGpus; ++g) {
        GpuSlice& s = gpus[g];
        CUDA_CHECK(cudaSetDevice(s.device));
        if (s.nRows > 0) {
            CUDA_CHECK(cudaMemcpyAsync(h_localOut.data() + (s.rowBegin - myRowBegin), s.d_out,
                                       s.nRows * sizeof(double), cudaMemcpyDeviceToHost, s.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
    }

    std::vector<int> recvCounts, displs;
    if (isRoot) {
        h_out.resize(numRows);
        recvCounts.resize(worldSize);
        displs.resize(worldSize);
        for (int r = 0; r < worldSize; ++r) {
            recvCounts[r] = static_cast<int>(rankBounds[r + 1] - rankBounds[r]);
            displs[r] = static_cast<int>(rankBounds[r]);
        }
    }
    MPI_CHECK(MPI_Gatherv(h_localOut.data(), static_cast<int>(myRows), MPI_DOUBLE,
                          isRoot ? h_out.data() : nullptr, isRoot ? recvCounts.data() : nullptr,
                          isRoot ? displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD));

    // Print results for external validation
    if (isRoot && printResults) {
        print_results(h_out, "OutputVector");
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (isRoot) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }

    // Cleanup
    for (int g = 0; g < nGpus; ++g) {
        GpuSlice& s = gpus[g];
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaFree(s.d_val));
        CUDA_CHECK(cudaFree(s.d_cols));
        CUDA_CHECK(cudaFree(s.d_rowDelimiters));
        CUDA_CHECK(cudaFree(s.d_vec));
        CUDA_CHECK(cudaFree(s.d_out));
        CUDA_CHECK(cudaStreamDestroy(s.stream));
    }
    MPI_CHECK(MPI_Comm_free(&nodeComm));
    MPI_Finalize();
    return exitCode;
}
