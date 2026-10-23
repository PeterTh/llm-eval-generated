#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// Local rows above which the CPU (OpenMP) threads take a slice of the work
// concurrently with the GPU.  Small problems are left entirely to the GPU to
// avoid OpenMP region overhead dominating the kernel time.
constexpr index_t HYBRID_MIN_LOCAL_ROWS = 8192;
constexpr index_t HYBRID_CPU_FRACTION_DIVISOR = 16;

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err__ = (call);                                                                \
        if (err__ != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err__), __FILE__,       \
                    __LINE__);                                                                     \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                          \
        }                                                                                          \
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
// Function: spmvCpuRange
//
// Purpose:
//   Computes a range of rows of the sparse matrix-vector product on the CPU.
//   The rows are independent so they are distributed across OpenMP threads.
//
// ****************************************************************************
void spmvCpuRange(const double* val, const index_t* cols, const index_t* rowDelimiters,
                  const double* vec, index_t rowBegin, index_t rowEnd, double* out) {
    const index_t workRows = rowEnd > rowBegin ? rowEnd - rowBegin : 0;

    // Bound the number of helper threads: extremely large teams add far more
    // synchronisation overhead than useful work for the small slices that are
    // processed on the CPU alongside the GPU.
    int threads = omp_get_max_threads();
    if (threads > 64) {
        threads = 64;
    }
    if (workRows > 0) {
        const index_t targetThreads = (workRows + 63) / 64;
        if (static_cast<index_t>(threads) > targetThreads) {
            threads = static_cast<int>(targetThreads);
        }
    }
    if (threads < 1) {
        threads = 1;
    }

#pragma omp parallel for schedule(static) num_threads(threads)
    for (index_t i = rowBegin; i < rowEnd; ++i) {
        double t = 0.0;
        const index_t rowStart = rowDelimiters[i];
        const index_t rowEndIdx = rowDelimiters[i + 1];
        for (index_t j = rowStart; j < rowEndIdx; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// CUDA kernels
//
// Row-per-thread is used for short rows, while a warp-per-row kernel with a
// shuffle reduction is used when the rows are long enough to benefit from it.
// Both operate directly on the local CSR slice.
// ****************************************************************************
__global__ void spmvCsrThreadPerRow(const double* __restrict__ val,
                                    const index_t* __restrict__ cols,
                                    const index_t* __restrict__ rowDelimiters,
                                    const double* __restrict__ vec, const index_t numRows,
                                    double* __restrict__ out) {
    const index_t row = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < numRows) {
        double t = 0.0;
        const index_t rowStart = rowDelimiters[row];
        const index_t rowEnd = rowDelimiters[row + 1];
        for (index_t j = rowStart; j < rowEnd; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[row] = t;
    }
}

__global__ void spmvCsrWarpPerRow(const double* __restrict__ val,
                                  const index_t* __restrict__ cols,
                                  const index_t* __restrict__ rowDelimiters,
                                  const double* __restrict__ vec, const index_t numRows,
                                  double* __restrict__ out) {
    constexpr int warpSize = 32;
    const index_t globalThread = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const index_t row = globalThread / warpSize;
    const int lane = threadIdx.x & (warpSize - 1);

    if (row < numRows) {
        double t = 0.0;
        const index_t rowStart = rowDelimiters[row];
        const index_t rowEnd = rowDelimiters[row + 1];
        for (index_t j = rowStart + lane; j < rowEnd; j += warpSize) {
            t += val[j] * vec[cols[j]];
        }
#pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            t += __shfl_down_sync(0xffffffffu, t, offset);
        }
        if (lane == 0) {
            out[row] = t;
        }
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
//   offset: global index of the first element (for diagnostics)
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t offset = 0) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", offset + i,
                       ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: "
                       "%.10e)\n",
                       offset + i, ref, res, relError);
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
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Determine the rank within the current node so that each rank can be
    // assigned a distinct local GPU when possible.
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    if (nodeComm != MPI_COMM_NULL) {
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate and initialize data structures.  The full matrix is generated
    // only on rank 0 and is then distributed row-wise with MPI_Scatterv.
    std::vector<double> h_vec(numRows);                 // Dense vector (replicated)
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Full row delimiters
    std::vector<double> fullVal;                        // Rank 0: all non-zero values
    std::vector<index_t> fullCols;                      // Rank 0: all column indices

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fullVal.resize(nItems);
        fullCols.resize(nItems);
        fill(fullVal.data(), nItems, maxVal);
        initRandomMatrix(fullCols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0,
              MPI_COMM_WORLD);

    // Row-wise decomposition of the matrix over MPI ranks.
    const index_t rowStart = static_cast<index_t>((static_cast<uint64_t>(numRows) * rank) /
                                                  static_cast<uint64_t>(numRanks));
    const index_t rowEnd = static_cast<index_t>((static_cast<uint64_t>(numRows) * (rank + 1)) /
                                                static_cast<uint64_t>(numRanks));
    const index_t localRows = rowEnd - rowStart;
    const index_t localNnzOffset = h_rowDelimiters[rowStart];
    const index_t localNnz = h_rowDelimiters[rowEnd] - localNnzOffset;

    std::vector<int> nnzCounts(numRanks);
    std::vector<int> nnzDispls(numRanks);
    std::vector<int> rowCounts(numRanks);
    std::vector<int> rowDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        const index_t rs = static_cast<index_t>((static_cast<uint64_t>(numRows) * r) /
                                                static_cast<uint64_t>(numRanks));
        const index_t re = static_cast<index_t>((static_cast<uint64_t>(numRows) * (r + 1)) /
                                                static_cast<uint64_t>(numRanks));
        nnzCounts[r] = static_cast<int>(h_rowDelimiters[re] - h_rowDelimiters[rs]);
        nnzDispls[r] = static_cast<int>(h_rowDelimiters[rs]);
        rowCounts[r] = static_cast<int>(re - rs);
        rowDispls[r] = static_cast<int>(rs);
    }

    // Local CSR slice and local row pointers (0-based within the slice).
    std::vector<double> l_val(localNnz);
    std::vector<index_t> l_cols(localNnz);
    std::vector<index_t> l_rowDelimiters(localRows + 1);
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= localRows; ++i) {
        l_rowDelimiters[i] = h_rowDelimiters[rowStart + i] - localNnzOffset;
    }

    MPI_Scatterv(rank == 0 ? fullVal.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                 MPI_DOUBLE, l_val.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? fullCols.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                 MPI_UINT32_T, l_cols.data(), static_cast<int>(localNnz), MPI_UINT32_T, 0,
                 MPI_COMM_WORLD);

    // The full matrix is no longer needed.
    std::vector<double>().swap(fullVal);
    std::vector<index_t>().swap(fullCols);
    std::vector<index_t>().swap(h_rowDelimiters);

    // Select a GPU for this rank (one GPU per local rank where possible).
    int deviceCount = 0;
    const cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    const bool useGpu = (deviceStatus == cudaSuccess) && (deviceCount > 0);
    if (useGpu) {
        const int device = localRank % deviceCount;
        CUDA_CHECK(cudaSetDevice(device));
    } else {
        if (rank == 0) {
            printf("No CUDA device available; using OpenMP CPU execution.\n");
        }
    }

    // Split the local rows between the GPU and the CPU (OpenMP) threads.  For
    // large problems the CPU processes a small disjoint slice concurrently
    // with the asynchronous GPU kernel.
    index_t gpuRows = localRows;
    index_t cpuRows = 0;
    if (!useGpu) {
        gpuRows = 0;
        cpuRows = localRows;
    } else if (localRows >= HYBRID_MIN_LOCAL_ROWS) {
        cpuRows = localRows / HYBRID_CPU_FRACTION_DIVISOR;
        gpuRows = localRows - cpuRows;
    }

    std::vector<double> l_out(localRows);

    // Reference solution (validation only), computed before timing as in the
    // original implementation.
    std::vector<double> l_reference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        l_reference.resize(localRows);
        spmvCpuRange(l_val.data(), l_cols.data(), l_rowDelimiters.data(), h_vec.data(), 0,
                     localRows, l_reference.data());
    }

    // GPU buffers.
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    index_t gpuNnz = 0;

    if (useGpu && gpuRows > 0) {
        gpuNnz = l_rowDelimiters[gpuRows];

        if (gpuNnz > 0) {
            CUDA_CHECK(cudaMalloc(&d_val, static_cast<size_t>(gpuNnz) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_cols, static_cast<size_t>(gpuNnz) * sizeof(index_t)));
            CUDA_CHECK(cudaMemcpy(d_val, l_val.data(), static_cast<size_t>(gpuNnz) * sizeof(double),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_cols, l_cols.data(),
                                  static_cast<size_t>(gpuNnz) * sizeof(index_t),
                                  cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMalloc(&d_rowDelimiters,
                              static_cast<size_t>(gpuRows + 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMemcpy(d_rowDelimiters, l_rowDelimiters.data(),
                              static_cast<size_t>(gpuRows + 1) * sizeof(index_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(gpuRows) * sizeof(double)));

        if (numRows > 0) {
            CUDA_CHECK(cudaMalloc(&d_vec, static_cast<size_t>(numRows) * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), static_cast<size_t>(numRows) * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }
    }

    const bool useWarpKernel = (gpuRows > 0) && ((gpuNnz / gpuRows) >= 32);

    // Warm up the device (module load, context creation) before timing so the
    // first-launch overhead is not attributed to the benchmarked iterations.
    if (useGpu && gpuRows > 0) {
        constexpr int blockSize = 256;
        if (useWarpKernel) {
            constexpr int warpsPerBlock = blockSize / 32;
            const int blocks = static_cast<int>((gpuRows + warpsPerBlock - 1) / warpsPerBlock);
            spmvCsrWarpPerRow<<<blocks, blockSize>>>(d_val, d_cols, d_rowDelimiters, d_vec, gpuRows,
                                                     d_out);
        } else {
            const int blocks = static_cast<int>((gpuRows + blockSize - 1) / blockSize);
            spmvCsrThreadPerRow<<<blocks, blockSize>>>(d_val, d_cols, d_rowDelimiters, d_vec,
                                                       gpuRows, d_out);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(0));
    }

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    const auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (useGpu && gpuRows > 0) {
            constexpr int blockSize = 256;
            if (useWarpKernel) {
                constexpr int warpsPerBlock = blockSize / 32;
                const int blocks = static_cast<int>((gpuRows + warpsPerBlock - 1) / warpsPerBlock);
                spmvCsrWarpPerRow<<<blocks, blockSize>>>(d_val, d_cols, d_rowDelimiters, d_vec,
                                                         gpuRows, d_out);
            } else {
                const int blocks = static_cast<int>((gpuRows + blockSize - 1) / blockSize);
                spmvCsrThreadPerRow<<<blocks, blockSize>>>(d_val, d_cols, d_rowDelimiters, d_vec,
                                                           gpuRows, d_out);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // CPU threads process the remaining rows while the GPU kernel runs.
        if (cpuRows > 0) {
            spmvCpuRange(l_val.data(), l_cols.data(), l_rowDelimiters.data(), h_vec.data(), gpuRows,
                         localRows, l_out.data());
        }
    }

    if (useGpu && gpuRows > 0) {
        CUDA_CHECK(cudaStreamSynchronize(0));
        CUDA_CHECK(cudaMemcpy(l_out.data(), d_out, static_cast<size_t>(gpuRows) * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    const auto end = std::chrono::high_resolution_clock::now();

    double localSeconds = std::chrono::duration<double>(end - start).count();
    double globalSeconds = localSeconds;
    MPI_Allreduce(&localSeconds, &globalSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (globalSeconds <= 0.0) {
        globalSeconds = 1e-9;
    }

    const double durationMs = globalSeconds * 1000.0;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(durationMs));

        const double gflops = (2.0 * nItems * iterations) / globalSeconds / 1e9;
        const double avgTime = iterations > 0 ? durationMs / static_cast<double>(iterations) : 0.0;

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather the distributed output vector onto rank 0 for printing.
    std::vector<double> h_out;
    if (printResults) {
        if (rank == 0) {
            h_out.resize(numRows);
        }
        MPI_Gatherv(l_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    rank == 0 ? h_out.data() : nullptr, rowCounts.data(), rowDispls.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(h_out, "OutputVector");
    }

    // Validation
    int localValid = 1;
    if (validate) {
        localValid = verifyResults(l_reference.data(), l_out.data(), localRows, rowStart) ? 1 : 0;
    }

    int globalValid = 1;
    if (validate) {
        MPI_Allreduce(&localValid, &globalValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            if (globalValid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    if (d_val != nullptr) {
        CUDA_CHECK(cudaFree(d_val));
    }
    if (d_cols != nullptr) {
        CUDA_CHECK(cudaFree(d_cols));
    }
    if (d_rowDelimiters != nullptr) {
        CUDA_CHECK(cudaFree(d_rowDelimiters));
    }
    if (d_vec != nullptr) {
        CUDA_CHECK(cudaFree(d_vec));
    }
    if (d_out != nullptr) {
        CUDA_CHECK(cudaFree(d_out));
    }

    MPI_Finalize();

    if (validate && !globalValid) {
        return 1;
    }
    return 0;
}
