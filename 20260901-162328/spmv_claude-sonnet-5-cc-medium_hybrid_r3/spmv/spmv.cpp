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

#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        cudaError_t err_ = (call);                                                 \
        if (err_ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err_));                                      \
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
//   Computes sparse matrix-vector multiplication using CSR format. The
//   outer loop over rows is independent across rows, so it is safe to
//   parallelize with OpenMP without changing the per-row summation order
//   (and therefore without changing the numerical result).
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
// Function: spmvKernel
//
// Purpose:
//   CUDA kernel computing sparse matrix-vector multiplication for a
//   contiguous range of local rows in CSR format. One thread handles one
//   row, accumulating in the same order as the reference CPU implementation.
//
// ****************************************************************************
__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                            const index_t* __restrict__ rowDelimiters,
                            const double* __restrict__ vec, index_t rowOffset,
                            index_t numRowsInChunk, double* __restrict__ out) {
    index_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= numRowsInChunk) return;
    index_t row = rowOffset + r;
    double t = 0.0;
    for (index_t j = rowDelimiters[row]; j < rowDelimiters[row + 1]; ++j) {
        t += val[j] * vec[cols[j]];
    }
    out[row] = t;
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

// ****************************************************************************
// Function: computePartition
//
// Purpose:
//   Splits `numRows` rows as evenly as possible across `worldSize` MPI
//   ranks, producing per-rank row counts and displacements.
//
// ****************************************************************************
void computePartition(index_t numRows, int worldSize, std::vector<index_t>& counts,
                       std::vector<index_t>& displs) {
    counts.resize(worldSize);
    displs.resize(worldSize);
    const index_t base = numRows / static_cast<index_t>(worldSize);
    const index_t rem = numRows % static_cast<index_t>(worldSize);
    index_t offset = 0;
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = base + (static_cast<index_t>(r) < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const bool isRoot = (worldRank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical across all ranks: mpirun
    // forwards the same argv to every process)
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

    // Select a GPU for this rank: ranks are round-robined across the
    // GPUs visible on their node so the benchmark scales across an
    // arbitrary number of MPI ranks and accelerators per node.
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    const int device = worldRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    if (isRoot) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d, GPUs visible per node: %d\n", worldSize, deviceCount);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate and initialize data structures. Generation uses a
    // sequential RNG (rand()/srand()) so it is performed once on the
    // root rank to preserve the exact matrix produced by the original
    // single-threaded implementation, then distributed to all ranks.
    std::vector<double> h_val;                  // Non-zero values
    std::vector<index_t> h_cols;                // Column indices
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out;                          // Output vector (root only)

    if (isRoot) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_out.resize(numRows);
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast the dense vector and row delimiters to every rank: both
    // are needed in full by all ranks (the vector for the dot products,
    // the delimiters to know each rank's slice of the CSR arrays).
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // For validation, compute reference solution (OpenMP-parallel, root only)
    std::vector<double> h_reference;
    if (validate && isRoot) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Partition rows across MPI ranks as evenly as possible
    std::vector<index_t> rowCounts, rowDispls;
    computePartition(numRows, worldSize, rowCounts, rowDispls);

    // Derive the corresponding non-zero counts/displacements per rank
    // from the (already broadcast) row delimiters.
    std::vector<int> nnzCounts(worldSize), nnzDispls(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const index_t rowStart = rowDispls[r];
        const index_t rowEnd = rowStart + rowCounts[r];
        nnzDispls[r] = static_cast<int>(h_rowDelimiters[rowStart]);
        nnzCounts[r] = static_cast<int>(h_rowDelimiters[rowEnd] - h_rowDelimiters[rowStart]);
    }
    std::vector<int> rowCountsInt(worldSize), rowDisplsInt(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        rowCountsInt[r] = static_cast<int>(rowCounts[r]);
        rowDisplsInt[r] = static_cast<int>(rowDispls[r]);
    }

    const index_t localRows = rowCounts[worldRank];
    const index_t localRowStart = rowDispls[worldRank];
    const index_t localNnz = static_cast<index_t>(nnzCounts[worldRank]);

    // Scatter this rank's slice of the val/cols CSR arrays
    std::vector<double> h_localVal(localNnz);
    std::vector<index_t> h_localCols(localNnz);
    MPI_Scatterv(isRoot ? h_val.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 h_localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(isRoot ? h_cols.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_UNSIGNED,
                 h_localCols.data(), static_cast<int>(localNnz), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Build this rank's local row-delimiters array (offsets relative to
    // its own local val/cols arrays). Independent across rows, so
    // parallelized with OpenMP.
    std::vector<index_t> h_localRowDelim(localRows + 1);
    const index_t localNnzBase = h_rowDelimiters[localRowStart];
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= localRows; ++i) {
        h_localRowDelim[i] = h_rowDelimiters[localRowStart + i] - localNnzBase;
    }

    std::vector<double> h_localOut(localRows);

    // Device buffers for this rank's slice of the problem
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelim = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    if (localNnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, localNnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, localNnz * sizeof(index_t)));
    }
    CUDA_CHECK(cudaMalloc(&d_rowDelim, (localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(&d_out, localRows * sizeof(double)));
    }

    // The dense vector is needed in full by every chunk; copy it once
    // synchronously before the concurrent per-chunk transfers below.
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    // Split this rank's local rows into chunks, each driven by its own
    // OpenMP host thread and its own CUDA stream. This overlaps the
    // host-to-device transfers of independent chunks and allows their
    // kernels to execute concurrently on the GPU.
    const int maxStreams = 8;
    const int numChunks = static_cast<int>(
        std::min<index_t>(static_cast<index_t>(maxStreams), std::max<index_t>(localRows, 1)));

    std::vector<index_t> chunkRowStart(numChunks), chunkRowCount(numChunks);
    {
        const index_t base = localRows / static_cast<index_t>(numChunks);
        const index_t rem = localRows % static_cast<index_t>(numChunks);
        index_t off = 0;
        for (int c = 0; c < numChunks; ++c) {
            chunkRowCount[c] = base + (static_cast<index_t>(c) < rem ? 1 : 0);
            chunkRowStart[c] = off;
            off += chunkRowCount[c];
        }
    }

    std::vector<cudaStream_t> streams(numChunks);
    for (int c = 0; c < numChunks; ++c) {
        CUDA_CHECK(cudaStreamCreate(&streams[c]));
    }

#pragma omp parallel for num_threads(numChunks) schedule(static, 1)
    for (int c = 0; c < numChunks; ++c) {
        // Each OpenMP thread has its own CUDA "current device" state, so it
        // must be set explicitly here before any CUDA calls in this thread.
        CUDA_CHECK(cudaSetDevice(device));
        const index_t rStart = chunkRowStart[c];
        const index_t rCount = chunkRowCount[c];
        if (rCount == 0) continue;
        const index_t nnzStart = h_localRowDelim[rStart];
        const index_t nnzEnd = h_localRowDelim[rStart + rCount];
        const index_t nnzCount = nnzEnd - nnzStart;

        CUDA_CHECK(cudaMemcpyAsync(d_rowDelim + rStart, h_localRowDelim.data() + rStart,
                                    (rCount + 1) * sizeof(index_t), cudaMemcpyHostToDevice,
                                    streams[c]));
        if (nnzCount > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_val + nnzStart, h_localVal.data() + nnzStart,
                                        nnzCount * sizeof(double), cudaMemcpyHostToDevice,
                                        streams[c]));
            CUDA_CHECK(cudaMemcpyAsync(d_cols + nnzStart, h_localCols.data() + nnzStart,
                                        nnzCount * sizeof(index_t), cudaMemcpyHostToDevice,
                                        streams[c]));
        }
    }
    for (int c = 0; c < numChunks; ++c) {
        CUDA_CHECK(cudaStreamSynchronize(streams[c]));
    }

    // Perform SpMV computation
    if (isRoot) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int blockSize = 256;
    for (index_t iter = 0; iter < iterations; ++iter) {
#pragma omp parallel for num_threads(numChunks) schedule(static, 1)
        for (int c = 0; c < numChunks; ++c) {
            CUDA_CHECK(cudaSetDevice(device));
            const index_t rStart = chunkRowStart[c];
            const index_t rCount = chunkRowCount[c];
            if (rCount == 0) continue;
            const int gridSize = static_cast<int>((rCount + blockSize - 1) / blockSize);
            spmvKernel<<<gridSize, blockSize, 0, streams[c]>>>(d_val, d_cols, d_rowDelim, d_vec,
                                                                rStart, rCount, d_out);
        }
    }
    for (int c = 0; c < numChunks; ++c) {
        CUDA_CHECK(cudaStreamSynchronize(streams[c]));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // The overall completion time is set by the slowest rank
    long long localMs = localDuration.count();
    long long maxMs = 0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy this rank's local result back to host, one chunk per stream
    if (localRows > 0) {
#pragma omp parallel for num_threads(numChunks) schedule(static, 1)
        for (int c = 0; c < numChunks; ++c) {
            CUDA_CHECK(cudaSetDevice(device));
            const index_t rStart = chunkRowStart[c];
            const index_t rCount = chunkRowCount[c];
            if (rCount == 0) continue;
            CUDA_CHECK(cudaMemcpyAsync(h_localOut.data() + rStart, d_out + rStart,
                                        rCount * sizeof(double), cudaMemcpyDeviceToHost,
                                        streams[c]));
        }
        for (int c = 0; c < numChunks; ++c) {
            CUDA_CHECK(cudaStreamSynchronize(streams[c]));
        }
    }

    // Gather all ranks' results back to the root rank's output vector
    MPI_Gatherv(h_localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                isRoot ? h_out.data() : nullptr, rowCountsInt.data(), rowDisplsInt.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int c = 0; c < numChunks; ++c) {
        CUDA_CHECK(cudaStreamDestroy(streams[c]));
    }
    if (d_val) CUDA_CHECK(cudaFree(d_val));
    if (d_cols) CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelim));
    CUDA_CHECK(cudaFree(d_vec));
    if (d_out) CUDA_CHECK(cudaFree(d_out));

    int result = 0;
    if (isRoot) {
        printf("Computation time: %lld ms\n", maxMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (maxMs / 1000.0) / 1e9;
        const double avgTime = maxMs / static_cast<double>(iterations);

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
                result = 0;
            } else {
                printf("Validation: FAILED\n");
                result = 1;
            }
        }
    }

    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
