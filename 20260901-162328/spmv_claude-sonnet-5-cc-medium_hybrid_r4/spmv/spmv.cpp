#include <algorithm>
#include <chrono>
#include <cmath>
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

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _err = (call);                                               \
        if (_err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_err));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                         \
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
//   (used on rank 0 only, to build the reference solution for validation)
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
//   CUDA kernel computing SpMV (CSR) for a contiguous block of local rows.
//   One thread handles one row; a grid-stride loop lets the kernel scale to
//   any number of local rows regardless of launch configuration.
//
// ****************************************************************************
__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                            const index_t* __restrict__ rowDelimiters, const double* __restrict__ vec,
                            const index_t localRows, double* __restrict__ out) {
    for (index_t row = blockIdx.x * blockDim.x + threadIdx.x; row < localRows;
         row += blockDim.x * gridDim.x) {
        double t = 0.0;
        const index_t rowStart = rowDelimiters[row];
        const index_t rowEnd = rowDelimiters[row + 1];
        for (index_t j = rowStart; j < rowEnd; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[row] = t;
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
    bool valid = true;
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        bool ok;
        double relError = 0.0;
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            ok = std::abs(res) <= MAX_RELATIVE_ERROR;
        } else {
            // Check relative error
            relError = std::abs((res - ref) / ref);
            ok = relError <= MAX_RELATIVE_ERROR;
        }
        if (!ok) {
#pragma omp critical
            {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i, ref, res, relError);
            }
#pragma omp atomic write
            valid = false;
        }
    }
    return valid;
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
    MPI_Init(&argc, &argv);

    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const bool isRoot = (worldRank == 0);

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

    // Bind this rank to a GPU: round-robin across the devices visible on its node
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int myDevice = worldRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(myDevice));

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (isRoot) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d, GPUs visible per rank: %d\n", worldSize, deviceCount);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Full-size buffers: generated in full only on rank 0, then distributed.
    std::vector<double> h_val;                 // Non-zero values (root only)
    std::vector<index_t> h_cols;               // Column indices (root only)
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters (replicated on all ranks)
    std::vector<double> h_vec(numRows);        // Dense vector (replicated on all ranks)
    std::vector<double> h_out;                 // Full output vector (root only)

    if (isRoot) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Replicate the dense vector and the row map to every rank; both are
    // small (O(dim)) compared to the O(nnz) matrix data, so replication is
    // cheap and lets every rank compute its own row partition locally.
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Partition rows contiguously and as evenly as possible across ranks.
    std::vector<int> rowStarts(worldSize), rowCounts(worldSize);
    {
        const index_t base = numRows / static_cast<index_t>(worldSize);
        const index_t rem = numRows % static_cast<index_t>(worldSize);
        index_t offset = 0;
        for (int r = 0; r < worldSize; ++r) {
            const index_t cnt = base + (static_cast<index_t>(r) < rem ? 1 : 0);
            rowStarts[r] = static_cast<int>(offset);
            rowCounts[r] = static_cast<int>(cnt);
            offset += cnt;
        }
    }
    const index_t myRowStart = static_cast<index_t>(rowStarts[worldRank]);
    const index_t myRowCount = static_cast<index_t>(rowCounts[worldRank]);

    // Scatter the matching slices of val/cols to each rank, based on the
    // (already replicated) row delimiters.
    std::vector<int> nnzCounts(worldSize), nnzDispls(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const index_t rowLo = static_cast<index_t>(rowStarts[r]);
        const index_t rowHi = rowLo + static_cast<index_t>(rowCounts[r]);
        nnzDispls[r] = static_cast<int>(h_rowDelimiters[rowLo]);
        nnzCounts[r] = static_cast<int>(h_rowDelimiters[rowHi] - h_rowDelimiters[rowLo]);
    }
    const index_t myNnz = static_cast<index_t>(nnzCounts[worldRank]);

    std::vector<double> h_valLocal(myNnz);
    std::vector<index_t> h_colsLocal(myNnz);

    MPI_Scatterv(isRoot ? h_val.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 h_valLocal.data(), static_cast<int>(myNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(isRoot ? h_cols.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_UINT32_T,
                 h_colsLocal.data(), static_cast<int>(myNnz), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Rebase the row delimiters to be local to this rank's nnz slice.
    std::vector<index_t> h_rowDelimLocal(myRowCount + 1);
    const index_t myNnzOffset = h_rowDelimiters[myRowStart];
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= myRowCount; ++i) {
        h_rowDelimLocal[i] = h_rowDelimiters[myRowStart + i] - myNnzOffset;
    }

    std::vector<double> h_outLocal(myRowCount);

    // Device buffers for this rank's local partition.
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelim = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_val, myNnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, myNnz * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelim, (myRowCount + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, myRowCount * sizeof(double)));

    // Use independent CUDA streams driven by separate OpenMP threads so the
    // four host->device transfers can be issued concurrently and overlap on
    // the GPU's copy engines.
    constexpr int kNumStreams = 4;
    cudaStream_t streams[kNumStreams];
    for (auto& s : streams) {
        CUDA_CHECK(cudaStreamCreate(&s));
    }

#pragma omp parallel num_threads(kNumStreams)
    {
        const int tid = omp_get_thread_num();
        switch (tid) {
            case 0:
                CUDA_CHECK(cudaMemcpyAsync(d_val, h_valLocal.data(), myNnz * sizeof(double),
                                            cudaMemcpyHostToDevice, streams[0]));
                break;
            case 1:
                CUDA_CHECK(cudaMemcpyAsync(d_cols, h_colsLocal.data(), myNnz * sizeof(index_t),
                                            cudaMemcpyHostToDevice, streams[1]));
                break;
            case 2:
                CUDA_CHECK(cudaMemcpyAsync(d_rowDelim, h_rowDelimLocal.data(),
                                            (myRowCount + 1) * sizeof(index_t),
                                            cudaMemcpyHostToDevice, streams[2]));
                break;
            case 3:
                CUDA_CHECK(cudaMemcpyAsync(d_vec, h_vec.data(), numRows * sizeof(double),
                                            cudaMemcpyHostToDevice, streams[3]));
                break;
            default:
                break;
        }
    }
    for (auto& s : streams) {
        CUDA_CHECK(cudaStreamSynchronize(s));
    }

    // Perform SpMV computation on the GPU, timed across all ranks.
    if (isRoot) {
        printf("Computing SpMV...\n");
    }

    const int threadsPerBlock = 256;
    int maxBlocks = 0;
    {
        int minGridSize = 0;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlocks, spmvKernel, 0, 0));
    }
    const int blocks = std::max(
        1, std::min(maxBlocks, static_cast<int>((myRowCount + threadsPerBlock - 1) / threadsPerBlock)));

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvKernel<<<blocks, threadsPerBlock, 0, streams[0]>>>(d_val, d_cols, d_rowDelim, d_vec,
                                                                myRowCount, d_out);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    long long duration = 0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaMemcpy(h_outLocal.data(), d_out, myRowCount * sizeof(double), cudaMemcpyDeviceToHost));

    for (auto& s : streams) {
        CUDA_CHECK(cudaStreamDestroy(s));
    }
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelim));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    if (isRoot) {
        h_out.resize(numRows);
    }
    MPI_Gatherv(h_outLocal.data(), static_cast<int>(myRowCount), MPI_DOUBLE, isRoot ? h_out.data() : nullptr,
                rowCounts.data(), rowStarts.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;

    if (isRoot) {
        printf("Computation time: %lld ms\n", duration);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration / 1000.0) / 1e9;
        const double avgTime = duration / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        // Validation
        if (validate) {
            printf("Computing reference solution...\n");
            std::vector<double> h_reference(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows,
                    h_reference.data());

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
    }

    // Every rank must return the same exit status.
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exitCode;
}
