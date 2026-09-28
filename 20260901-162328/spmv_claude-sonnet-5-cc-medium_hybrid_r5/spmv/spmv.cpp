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

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(err__));                                     \
            MPI_Abort(MPI_COMM_WORLD, 1);                                           \
        }                                                                           \
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
// Note: This generation is kept strictly sequential (single rank) because
// the random values assigned to each entry depend on the position in the
// global rand() stream; parallelizing it would change the generated matrix.
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
//   Parallelized across rows with OpenMP; each row is an independent
//   reduction so this does not change the per-row result.
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
#pragma omp parallel for schedule(dynamic, 256)
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
//   CUDA kernel computing CSR SpMV with one warp assigned per matrix row.
//   Non-zeros of a row are strided across the 32 lanes of the warp and the
//   partial sums are combined with a warp shuffle reduction. This balances
//   work across threads regardless of per-row non-zero variance.
//
// ****************************************************************************
__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                            const index_t* __restrict__ rowDelimiters,
                            const double* __restrict__ vec, index_t numRows,
                            double* __restrict__ out) {
    const index_t warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int lane = threadIdx.x % warpSize;
    if (warpId >= numRows) return;

    const index_t rowStart = rowDelimiters[warpId];
    const index_t rowEnd = rowDelimiters[warpId + 1];

    double sum = 0.0;
    for (index_t j = rowStart + lane; j < rowEnd; j += warpSize) {
        sum += val[j] * vec[cols[j]];
    }

#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    if (lane == 0) {
        out[warpId] = sum;
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

// Splits [0, numRows) into `size` contiguous row ranges that are
// approximately balanced by non-zero count (not just row count), based on
// the prefix sums already present in rowDelimiters.
static void computeRowPartition(const std::vector<index_t>& rowDelimiters, index_t numRows,
                                 index_t nItems, int size, std::vector<index_t>& rowSplits) {
    rowSplits.assign(size + 1, 0);
    rowSplits[size] = numRows;
    for (int k = 1; k < size; ++k) {
        const index_t target = static_cast<index_t>((static_cast<uint64_t>(nItems) * k) / size);
        auto it = std::lower_bound(rowDelimiters.begin(), rowDelimiters.begin() + numRows + 1, target);
        index_t row = static_cast<index_t>(it - rowDelimiters.begin());
        row = std::min(row, numRows);
        row = std::max(row, rowSplits[k - 1]);
        rowSplits[k] = row;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldSize = 1;
    int worldRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    const bool isRoot = (worldRank == 0);

    // Determine the node-local rank so that multiple ranks on the same node
    // spread across the node's available GPUs.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount <= 0) {
        if (isRoot) {
            fprintf(stderr, "No CUDA-capable device found. This build requires a GPU per rank.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank receives the same argv from
    // the launcher, so parsing independently keeps all ranks consistent
    // without needing a broadcast).
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

    if (isRoot) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Parallelization: MPI (%d ranks) + OpenMP (%d threads/rank) + CUDA (%d device(s)/node)\n",
               worldSize, omp_get_max_threads(), deviceCount);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Full data structures are only materialized on rank 0: generation is
    // an inherently sequential random process (see initRandomMatrix), and
    // this way we avoid every rank paying the full memory cost.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out;
    std::vector<double> h_reference;

    if (isRoot) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_out.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    // Broadcast the dense vector: every rank needs it in full since a
    // locally-owned row's non-zeros can reference any global column.
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Partition rows across ranks, balanced by non-zero count.
    std::vector<index_t> rowSplits(worldSize + 1);
    std::vector<int> valCounts(worldSize), valDispls(worldSize);
    std::vector<int> rowCounts(worldSize), rowDispls(worldSize);

    if (isRoot) {
        computeRowPartition(h_rowDelimiters, numRows, nItems, worldSize, rowSplits);
        for (int r = 0; r < worldSize; ++r) {
            const index_t rs = rowSplits[r];
            const index_t re = rowSplits[r + 1];
            valCounts[r] = static_cast<int>(h_rowDelimiters[re] - h_rowDelimiters[rs]);
            valDispls[r] = static_cast<int>(h_rowDelimiters[rs]);
            rowCounts[r] = static_cast<int>(re - rs + 1);
            // Displacement into h_rowDelimiters itself (not a packed
            // buffer): ranks' row ranges are contiguous but their
            // (count+1)-sized delimiter windows intentionally overlap by
            // one element at each boundary, which Scatterv supports since
            // source reads need not be disjoint.
            rowDispls[r] = static_cast<int>(rs);
        }
    }

    MPI_Bcast(rowSplits.data(), worldSize + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(valCounts.data(), worldSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(valDispls.data(), worldSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rowCounts.data(), worldSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rowDispls.data(), worldSize, MPI_INT, 0, MPI_COMM_WORLD);

    const index_t localNumRows = rowSplits[worldRank + 1] - rowSplits[worldRank];
    const int localNnz = valCounts[worldRank];

    std::vector<double> local_val(localNnz);
    std::vector<index_t> local_cols(localNnz);
    std::vector<index_t> local_rowDelim(localNumRows + 1);

    MPI_Scatterv(isRoot ? h_val.data() : nullptr, valCounts.data(), valDispls.data(), MPI_DOUBLE,
                 local_val.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(isRoot ? h_cols.data() : nullptr, valCounts.data(), valDispls.data(), MPI_UINT32_T,
                 local_cols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(isRoot ? h_rowDelimiters.data() : nullptr, rowCounts.data(), rowDispls.data(),
                 MPI_UINT32_T, local_rowDelim.data(), static_cast<int>(localNumRows + 1), MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    // Re-base the local row delimiters to be local (0-based) offsets into
    // this rank's local_val/local_cols arrays.
    const index_t localOffset = local_rowDelim[0];
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= localNumRows; ++i) {
        local_rowDelim[i] -= localOffset;
    }

    std::vector<double> local_out(localNumRows);

    // Device buffers.
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelim = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_val, sizeof(double) * std::max<size_t>(localNnz, 1)));
    CUDA_CHECK(cudaMalloc(&d_cols, sizeof(index_t) * std::max<size_t>(localNnz, 1)));
    CUDA_CHECK(cudaMalloc(&d_rowDelim, sizeof(index_t) * (localNumRows + 1)));
    CUDA_CHECK(cudaMalloc(&d_vec, sizeof(double) * numRows));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * std::max<size_t>(localNumRows, 1)));

    CUDA_CHECK(cudaMemcpy(d_val, local_val.data(), sizeof(double) * localNnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, local_cols.data(), sizeof(index_t) * localNnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelim, local_rowDelim.data(), sizeof(index_t) * (localNumRows + 1),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), sizeof(double) * numRows, cudaMemcpyHostToDevice));

    if (isRoot) {
        printf("Computing SpMV...\n");
    }

    const int threadsPerBlock = 256;
    const int warpsPerBlock = threadsPerBlock / 32;
    const int numBlocks = static_cast<int>((localNumRows + warpsPerBlock - 1) / std::max(warpsPerBlock, 1));

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localNumRows > 0) {
            spmvKernel<<<numBlocks, threadsPerBlock>>>(d_val, d_cols, d_rowDelim, d_vec, localNumRows, d_out);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto localEnd = std::chrono::high_resolution_clock::now();
    double localElapsedMs = std::chrono::duration<double, std::milli>(localEnd - start).count();

    double elapsedMs = 0.0;
    MPI_Allreduce(&localElapsedMs, &elapsedMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const long duration_ms = static_cast<long>(elapsedMs);

    if (localNumRows > 0) {
        CUDA_CHECK(cudaMemcpy(local_out.data(), d_out, sizeof(double) * localNumRows, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelim));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    // Gather the per-rank results back into the full output vector on rank 0.
    std::vector<int> outCounts(worldSize), outDispls(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        outCounts[r] = static_cast<int>(rowSplits[r + 1] - rowSplits[r]);
        outDispls[r] = static_cast<int>(rowSplits[r]);
    }
    MPI_Gatherv(local_out.data(), static_cast<int>(localNumRows), MPI_DOUBLE,
                isRoot ? h_out.data() : nullptr, outCounts.data(), outDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (isRoot) {
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration_ms / 1000.0) / 1e9;
        const double avgTime = duration_ms / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }
    }

    int exitCode = 0;
    if (validate) {
        if (isRoot) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
