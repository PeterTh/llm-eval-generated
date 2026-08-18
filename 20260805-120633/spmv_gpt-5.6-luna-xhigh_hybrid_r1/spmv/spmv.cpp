#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr unsigned CUDA_WARPS_PER_BLOCK = 8;

// ****************************************************************************
// CUDA error handling
// ****************************************************************************
void checkCuda(const cudaError_t status, const char* operation, const int rank) {
    if (status != cudaSuccess) {
        fprintf(stderr, "MPI rank %d: CUDA failure in %s: %s\n", rank, operation,
                cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

// One warp owns one CSR row.  This gives coalesced accesses to val/cols and
// avoids the load imbalance of assigning one whole block to each row.
__global__ void spmvCudaKernel(const double* __restrict__ val,
                               const index_t* __restrict__ cols,
                               const index_t* __restrict__ rowDelimiters,
                               const double* __restrict__ vec,
                               const index_t rows,
                               double* __restrict__ out) {
    const unsigned lane = threadIdx.x & (warpSize - 1);
    const unsigned warpInBlock = threadIdx.x / warpSize;
    const index_t row = static_cast<index_t>(blockIdx.x * CUDA_WARPS_PER_BLOCK + warpInBlock);

    if (row >= rows) {
        return;
    }

    const index_t begin = rowDelimiters[row];
    const index_t end = rowDelimiters[row + 1];
    double sum = 0.0;
    for (uint64_t j = static_cast<uint64_t>(begin) + lane; j < end; j += warpSize) {
        sum += val[j] * vec[cols[j]];
    }

    for (unsigned offset = warpSize / 2; offset != 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        out[row] = sum;
    }
}

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values.  This remains on rank 0 and
//   sequential so the benchmark retains the original deterministic rand()
//   stream; the initialized data is then distributed with MPI.
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
// ****************************************************************************
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;
    const uint64_t totalEntries = static_cast<uint64_t>(dim) * dim;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / static_cast<double>(totalEntries);

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t visited = static_cast<uint64_t>(i) * dim + j;
            const uint64_t numEntriesLeft = totalEntries - visited;
            const uint64_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            const double randVal = static_cast<double>(rand()) / RAND_MAX;
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
//   Computes sparse matrix-vector multiplication using CSR format.  This is
//   the OpenMP host-side path used for the validation reference and for
//   checking the same CSR semantics as the CUDA path.
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(dim); ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const index_t col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// OpenMP is also used on every rank for host-side staging, including ranks
// which own no rows when more MPI ranks than matrix rows are requested.
void adjustRowOffsets(std::vector<index_t>& offsets, const index_t base) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(offsets.size()); ++i) {
        offsets[static_cast<size_t>(i)] -= base;
    }
}

void zeroParallel(std::vector<double>& values) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
        values[static_cast<size_t>(i)] = 0.0;
    }
}

// ****************************************************************************
// Function: verifyResults
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
    int mpiThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    if (mpiThreadLevel < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI rank %d: MPI implementation does not provide MPI_THREAD_FUNNELED.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // MPI calls are made only by the OpenMP master thread.
    omp_set_dynamic(0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = static_cast<index_t>(strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = static_cast<index_t>(strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = static_cast<index_t>(strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            maxVal = strtod(argv[++i], nullptr);
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

    const uint64_t totalEntries = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nItems64 = sparsity == 0 ? 0 : totalEntries / sparsity;
    const bool sizeIsSupported =
        sparsity != 0 && nItems64 <= std::numeric_limits<index_t>::max() &&
        numRows <= static_cast<index_t>(std::numeric_limits<int>::max()) &&
        nItems64 <= static_cast<uint64_t>(std::numeric_limits<int>::max());
    if (!sizeIsSupported) {
        if (rank == 0) {
            fprintf(stderr, "Unsupported dimensions: require nonzero sparsity and MPI-count-safe sizes.\n");
        }
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nItems64);

    // Construct a shared-memory communicator so each rank selects a GPU by
    // local rank, rather than accidentally mapping all nodes to one device.
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) {
        fprintf(stderr, "MPI rank %d: no CUDA accelerator is available.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int device = localRank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice", rank);

    cudaDeviceProp deviceProperties{};
    checkCuda(cudaGetDeviceProperties(&deviceProperties, device), "cudaGetDeviceProperties", rank);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / totalEntries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", worldSize);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("CUDA device: %s\n", deviceProperties.name);
    }

    // Rank 0 owns the original global input, preserving the exact random
    // stream.  Each rank receives only its contiguous row block below.
    std::vector<double> h_vec(numRows);
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // The dense vector is needed by every GPU rank.  MPI_UINT32_T keeps the
    // CSR index representation identical on host and device.
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> rowCounts;
    std::vector<int> rowDelimiterCounts;
    std::vector<int> rowDisplacements;
    std::vector<int> nnzCounts;
    std::vector<int> nnzDisplacements;
    if (rank == 0) {
        rowCounts.resize(worldSize);
        rowDelimiterCounts.resize(worldSize);
        rowDisplacements.resize(worldSize);
        nnzCounts.resize(worldSize);
        nnzDisplacements.resize(worldSize);
        for (int r = 0; r < worldSize; ++r) {
            const index_t begin = static_cast<index_t>((static_cast<uint64_t>(numRows) * r) / worldSize);
            const index_t end = static_cast<index_t>((static_cast<uint64_t>(numRows) * (r + 1)) / worldSize);
            rowCounts[r] = static_cast<int>(end - begin);
            rowDelimiterCounts[r] = rowCounts[r] + 1;
            rowDisplacements[r] = static_cast<int>(begin);
            nnzCounts[r] = static_cast<int>(h_rowDelimiters[end] - h_rowDelimiters[begin]);
            nnzDisplacements[r] = static_cast<int>(h_rowDelimiters[begin]);
        }
    }

    const index_t rowBegin = static_cast<index_t>((static_cast<uint64_t>(numRows) * rank) / worldSize);
    const index_t rowEnd = static_cast<index_t>((static_cast<uint64_t>(numRows) * (rank + 1)) / worldSize);
    const index_t localRows = rowEnd - rowBegin;

    std::vector<index_t> localRowDelimiters(static_cast<size_t>(localRows) + 1);
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 rank == 0 ? rowDelimiterCounts.data() : nullptr,
                 rank == 0 ? rowDisplacements.data() : nullptr, MPI_UINT32_T,
                 localRowDelimiters.data(), static_cast<int>(localRows) + 1,
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);

    int localNnzCount = 0;
    MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_INT,
                &localNnzCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const index_t localNnz = static_cast<index_t>(localNnzCount);

    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDisplacements.data() : nullptr, MPI_DOUBLE,
                 localVal.data(), localNnzCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDisplacements.data() : nullptr, MPI_UINT32_T,
                 localCols.data(), localNnzCount, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t nnzBase = localRowDelimiters.front();
    adjustRowOffsets(localRowDelimiters, nnzBase);

    std::vector<double> localOut(localRows);
    zeroParallel(localOut);
    std::vector<double> h_out;
    if (rank == 0) {
        h_out.resize(numRows);
        zeroParallel(h_out);
    }

    // Validation is intentionally computed from the global CSR on rank 0,
    // while the timed operation below uses only the distributed local CSR.
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution with OpenMP...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    cudaStream_t stream = nullptr;

    if (localNnz != 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_val),
                             static_cast<size_t>(localNnz) * sizeof(double)),
                  "cudaMalloc(d_val)", rank);
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_cols),
                             static_cast<size_t>(localNnz) * sizeof(index_t)),
                  "cudaMalloc(d_cols)", rank);
    }
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_rowDelimiters),
                         (static_cast<size_t>(localRows) + 1) * sizeof(index_t)),
              "cudaMalloc(d_rowDelimiters)", rank);
    if (numRows != 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_vec),
                             static_cast<size_t>(numRows) * sizeof(double)),
                  "cudaMalloc(d_vec)", rank);
    }
    if (localRows != 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_out),
                             static_cast<size_t>(localRows) * sizeof(double)),
                  "cudaMalloc(d_out)", rank);
    }
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags", rank);

    if (localNnz != 0) {
        checkCuda(cudaMemcpyAsync(d_val, localVal.data(),
                                  static_cast<size_t>(localNnz) * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(d_val)", rank);
        checkCuda(cudaMemcpyAsync(d_cols, localCols.data(),
                                  static_cast<size_t>(localNnz) * sizeof(index_t),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(d_cols)", rank);
    }
    checkCuda(cudaMemcpyAsync(d_rowDelimiters, localRowDelimiters.data(),
                              (static_cast<size_t>(localRows) + 1) * sizeof(index_t),
                              cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync(d_rowDelimiters)", rank);
    if (numRows != 0) {
        checkCuda(cudaMemcpyAsync(d_vec, h_vec.data(),
                                  static_cast<size_t>(numRows) * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(d_vec)", rank);
    }
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(input)", rank);

    const dim3 block(CUDA_WARPS_PER_BLOCK * 32);
    const dim3 grid((static_cast<unsigned>(localRows) + CUDA_WARPS_PER_BLOCK - 1) /
                    CUDA_WARPS_PER_BLOCK);

    // Warm up once so module loading and the first launch do not distort the
    // measured steady-state iterations.
    if (iterations != 0 && localRows != 0) {
        spmvCudaKernel<<<grid, block, 0, stream>>>(
            d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out);
        checkCuda(cudaGetLastError(), "spmvCudaKernel(warmup)", rank);
    }
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup)", rank);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows != 0) {
            spmvCudaKernel<<<grid, block, 0, stream>>>(
                d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out);
            checkCuda(cudaGetLastError(), "spmvCudaKernel", rank);
        }
    }
    if (iterations != 0 && localRows != 0) {
        checkCuda(cudaMemcpyAsync(localOut.data(), d_out,
                                  static_cast<size_t>(localRows) * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream),
                  "cudaMemcpyAsync(d_out)", rank);
    }
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(compute)", rank);
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();

    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr,
                rank == 0 ? rowCounts.data() : nullptr,
                rank == 0 ? rowDisplacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double elapsedMilliseconds = elapsedSeconds * 1000.0;
        const double avgTime = iterations == 0 ? 0.0 : elapsedMilliseconds / iterations;
        const double gflops = (elapsedSeconds > 0.0)
                                  ? (2.0 * static_cast<double>(nItems) * iterations) /
                                        elapsedSeconds / 1e9
                                  : 0.0;
        printf("Computation time: %.3f ms\n", elapsedMilliseconds);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }
    }

    int validationStatus = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            validationStatus = 1;
        }
    }
    MPI_Bcast(&validationStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);

    checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy", rank);
    if (d_out != nullptr) {
        checkCuda(cudaFree(d_out), "cudaFree(d_out)", rank);
    }
    if (d_vec != nullptr) {
        checkCuda(cudaFree(d_vec), "cudaFree(d_vec)", rank);
    }
    checkCuda(cudaFree(d_rowDelimiters), "cudaFree(d_rowDelimiters)", rank);
    if (d_cols != nullptr) {
        checkCuda(cudaFree(d_cols), "cudaFree(d_cols)", rank);
    }
    if (d_val != nullptr) {
        checkCuda(cudaFree(d_val), "cudaFree(d_val)", rank);
    }
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return validationStatus;
}
