#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
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

#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        cudaError_t err_ = (call);                                                 \
        if (err_ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err_));                                     \
            MPI_Abort(MPI_COMM_WORLD, 1);                                          \
        }                                                                          \
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
//   Computes sparse matrix-vector multiplication using CSR format
//   (used to build the reference solution for validation; rows are
//   independent so this is safely parallelized across CPU cores)
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
// Kernel: spmvKernelWarp
//
// Purpose:
//   CUDA CSR SpMV kernel using one warp per matrix row, so that the
//   nonzeros of a row are read in a coalesced fashion across lanes and
//   irregular row lengths are load balanced within the warp.
//
// ****************************************************************************
__global__ void spmvKernelWarp(const double* __restrict__ val, const index_t* __restrict__ cols,
                                const index_t* __restrict__ rowDelimiters,
                                const double* __restrict__ vec, index_t numRows,
                                double* __restrict__ out) {
    const int warpsPerBlock = blockDim.x / warpSize;
    const int warpIdInBlock = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;
    const index_t row = static_cast<index_t>(blockIdx.x) * warpsPerBlock + warpIdInBlock;
    if (row >= numRows) return;

    const index_t rowStart = rowDelimiters[row];
    const index_t rowEnd = rowDelimiters[row + 1];

    double sum = 0.0;
    for (index_t j = rowStart + lane; j < rowEnd; j += warpSize) {
        sum += val[j] * vec[cols[j]];
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
    }

    if (lane == 0) {
        out[row] = sum;
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

// Splits [0, total) into numParts contiguous, near-equal chunks and returns
// the [start, end) range belonging to partIdx.
static void computeRange(index_t total, int numParts, int partIdx, index_t& start, index_t& end) {
    const index_t base = total / static_cast<index_t>(numParts);
    const index_t rem = total % static_cast<index_t>(numParts);
    const index_t idx = static_cast<index_t>(partIdx);
    start = idx * base + std::min(idx, rem);
    end = start + base + (idx < rem ? 1 : 0);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    struct Params {
        index_t numRows = 1024;
        index_t sparsity = 10;
        index_t iterations = 10;
        double maxVal = 1.0;
        int validate = 0;
        int printResults = 0;
        int showHelp = 0;
        int badArg = 0;
    } params;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                params.numRows = static_cast<index_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                params.sparsity = static_cast<index_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                params.iterations = static_cast<index_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
                params.maxVal = atof(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                params.validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                params.printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                params.showHelp = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                params.badArg = 1;
            }
        }
    }

    MPI_Bcast(&params, sizeof(Params), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (params.showHelp) {
        MPI_Finalize();
        return 0;
    }
    if (params.badArg) {
        MPI_Finalize();
        return 1;
    }

    const index_t numRows = params.numRows;
    const index_t sparsity = params.sparsity;
    const index_t iterations = params.iterations;
    const double maxVal = params.maxVal;
    const bool validate = params.validate != 0;
    const bool printResults = params.printResults != 0;

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    // Discover local CUDA devices for this rank
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices found; this benchmark requires an accelerator.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d, CUDA devices per rank: %d, OpenMP available threads: %d\n",
               numRanks, numDevices, omp_get_max_threads());
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Full matrix/vector data is generated only on rank 0, exactly as in the
    // original serial algorithm, so the problem instance is independent of
    // the number of MPI ranks used.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);
    std::vector<double> h_reference;

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows,
                    h_reference.data());
        }
    }

    // Broadcast the small, dense metadata every rank needs: the row
    // delimiters (to compute per-rank row ranges/offsets) and the dense
    // vector (any row's nonzeros may reference any column).
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Partition the matrix rows across MPI ranks in contiguous blocks and
    // scatter the corresponding CSR value/column slices.
    index_t rowStart = 0, rowEnd = 0;
    computeRange(numRows, numRanks, rank, rowStart, rowEnd);
    const index_t localRows = rowEnd - rowStart;

    std::vector<int> sendCounts(numRanks), sendDispls(numRanks);
    std::vector<int> rowCounts(numRanks), rowDispls(numRanks);
    if (rank == 0) {
        for (int r = 0; r < numRanks; ++r) {
            index_t rs = 0, re = 0;
            computeRange(numRows, numRanks, r, rs, re);
            rowCounts[r] = static_cast<int>(re - rs);
            rowDispls[r] = static_cast<int>(rs);
            sendCounts[r] = static_cast<int>(h_rowDelimiters[re] - h_rowDelimiters[rs]);
            sendDispls[r] = static_cast<int>(h_rowDelimiters[rs]);
        }
    }

    const index_t nnzStart = h_rowDelimiters[rowStart];
    const index_t nnzEnd = h_rowDelimiters[rowEnd];
    const index_t localNnz = nnzEnd - nnzStart;

    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, sendCounts.data(), sendDispls.data(),
                 MPI_DOUBLE, localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, sendCounts.data(), sendDispls.data(),
                 MPI_UINT32_T, localCols.data(), static_cast<int>(localNnz), MPI_UINT32_T, 0,
                 MPI_COMM_WORLD);

    // Local row delimiters, re-based to index into this rank's local
    // val/cols arrays.
    std::vector<index_t> localRowDelim(localRows + 1);
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= localRows; ++i) {
        localRowDelim[i] = h_rowDelimiters[rowStart + i] - nnzStart;
    }

    std::vector<double> localOut(localRows);

    // Further split this rank's row range across its local GPUs; each
    // OpenMP thread owns one GPU and drives it independently.
    const int numLocalGpus =
        static_cast<int>(std::min<index_t>(static_cast<index_t>(numDevices), std::max<index_t>(localRows, 1)));

    struct DeviceChunk {
        int device = 0;
        index_t subStart = 0;
        index_t subRows = 0;
        double* d_val = nullptr;
        index_t* d_cols = nullptr;
        index_t* d_rowDelim = nullptr;
        double* d_vec = nullptr;
        double* d_out = nullptr;
    };
    std::vector<DeviceChunk> chunks(numLocalGpus);

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    // --- Setup phase (untimed): allocate device buffers and copy the
    // per-chunk CSR data and the dense vector to each GPU. ---
#pragma omp parallel num_threads(numLocalGpus)
    {
        const int li = omp_get_thread_num();
        DeviceChunk& c = chunks[li];
        c.device = li % numDevices;
        CUDA_CHECK(cudaSetDevice(c.device));

        index_t subStart = 0, subEnd = 0;
        computeRange(localRows, numLocalGpus, li, subStart, subEnd);
        c.subStart = subStart;
        c.subRows = subEnd - subStart;

        const index_t nnzSubStart = localRowDelim[subStart];
        const index_t nnzSubEnd = localRowDelim[subEnd];
        const index_t subNnz = nnzSubEnd - nnzSubStart;

        std::vector<index_t> subRowDelim(c.subRows + 1);
        for (index_t i = 0; i <= c.subRows; ++i) {
            subRowDelim[i] = localRowDelim[subStart + i] - nnzSubStart;
        }

        CUDA_CHECK(cudaMalloc(&c.d_val, std::max<index_t>(subNnz, 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.d_cols, std::max<index_t>(subNnz, 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&c.d_rowDelim, (c.subRows + 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&c.d_vec, numRows * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.d_out, std::max<index_t>(c.subRows, 1) * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(c.d_val, localVal.data() + nnzSubStart, subNnz * sizeof(double),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(c.d_cols, localCols.data() + nnzSubStart, subNnz * sizeof(index_t),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(c.d_rowDelim, subRowDelim.data(), (c.subRows + 1) * sizeof(index_t),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(c.d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // --- Timed phase: each OpenMP thread launches the SpMV kernel on its
    // own GPU for `iterations` repetitions. ---
#pragma omp parallel num_threads(numLocalGpus)
    {
        const int li = omp_get_thread_num();
        DeviceChunk& c = chunks[li];
        CUDA_CHECK(cudaSetDevice(c.device));

        constexpr int kThreadsPerBlock = 256;
        constexpr int kRowsPerBlock = kThreadsPerBlock / 32;
        const int numBlocks = static_cast<int>((c.subRows + kRowsPerBlock - 1) / kRowsPerBlock);

        if (c.subRows > 0) {
            for (index_t iter = 0; iter < iterations; ++iter) {
                spmvKernelWarp<<<numBlocks, kThreadsPerBlock>>>(c.d_val, c.d_cols, c.d_rowDelim,
                                                                 c.d_vec, c.subRows, c.d_out);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double localElapsedMs =
        std::chrono::duration<double, std::milli>(end - start).count();
    double maxElapsedMs = 0.0;
    MPI_Allreduce(&localElapsedMs, &maxElapsedMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // --- Copy results back and release device resources (untimed). ---
#pragma omp parallel num_threads(numLocalGpus)
    {
        const int li = omp_get_thread_num();
        DeviceChunk& c = chunks[li];
        CUDA_CHECK(cudaSetDevice(c.device));
        if (c.subRows > 0) {
            CUDA_CHECK(cudaMemcpy(localOut.data() + c.subStart, c.d_out,
                                   c.subRows * sizeof(double), cudaMemcpyDeviceToHost));
        }
        cudaFree(c.d_val);
        cudaFree(c.d_cols);
        cudaFree(c.d_rowDelim);
        cudaFree(c.d_vec);
        cudaFree(c.d_out);
    }

    // Gather the per-rank row results into the full output vector on rank 0.
    std::vector<double> h_out;
    if (rank == 0) {
        h_out.resize(numRows);
    }
    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE, h_out.data(),
                rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int returnCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsedMs));

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (maxElapsedMs / 1000.0) / 1e9;
        const double avgTime = maxElapsedMs / static_cast<double>(iterations);

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
                returnCode = 0;
            } else {
                printf("Validation: FAILED\n");
                returnCode = 1;
            }
        }
    }

    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return returnCode;
}
