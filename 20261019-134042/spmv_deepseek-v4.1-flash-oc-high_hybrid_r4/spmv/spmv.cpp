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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t _err = (call);                                                \
        if (_err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(_err), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
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
// Function: initRandomMatrixLocal
//
// Purpose:
//   Reproduces the exact random sequence of the original serial
//   initRandomMatrix, but only materializes the column indices and row
//   offsets belonging to the local row range [rowStart, rowEnd). The global
//   non-zero index at which the local range starts is returned so that the
//   corresponding slice of the value array can be generated identically.
//
// ****************************************************************************
void initRandomMatrixLocal(std::vector<index_t>& cols,
                           std::vector<index_t>& rowOffsets,
                           index_t& globalNnzStart,
                           const index_t n, const index_t dim,
                           const index_t rowStart, const index_t rowEnd) {
    const index_t localRows = rowEnd - rowStart;
    rowOffsets.assign(localRows + 1, 0);
    globalNnzStart = n;

    if (localRows == 0) {
        return;
    }

    cols.clear();
    cols.reserve(static_cast<size_t>(n) * localRows / dim + 16);

    index_t nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator (identical to the reference implementation)
    srand(8675309);

    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        if (i == rowStart) {
            globalNnzStart = nnzAssigned;
            rowOffsets[0] = 0;
        }
        if (i >= rowStart && i < rowEnd) {
            rowOffsets[i - rowStart] = nnzAssigned - globalNnzStart;
        }
        for (index_t j = 0; j < dim; ++j) {
            const index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            const index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            const double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                if (i >= rowStart && i < rowEnd) {
                    cols.push_back(j);
                }
                ++nnzAssigned;
            }
        }
    }
    rowOffsets[localRows] = static_cast<index_t>(cols.size());
}

// ****************************************************************************
// Function: fillValuesLocal
//
// Purpose:
//   Generates the local slice [globalStart, globalStart + count) of the
//   non-zero values using the exact same random sequence as the serial
//   implementation (vector values are consumed first, then the skipped
//   prefix of the value sequence).
//
// ****************************************************************************
void fillValuesLocal(std::vector<double>& val, const index_t globalStart,
                     const index_t count, const index_t dim, const double maxVal) {
    val.clear();
    if (count == 0) {
        return;
    }
    srand(1);
    for (index_t i = 0; i < dim; ++i) {
        (void)rand();
    }
    for (index_t i = 0; i < globalStart; ++i) {
        (void)rand();
    }
    val.resize(count);
    for (index_t i = 0; i < count; ++i) {
        val[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: spmvCpuRange
//
// Purpose:
//   Computes sparse matrix-vector multiplication for the local rows in
//   [rowBegin, rowEnd) using OpenMP across the available CPU threads.
//
// ****************************************************************************
void spmvCpuRange(const double* val, const index_t* cols, const index_t* rowOffsets,
                  const double* vec, const index_t rowBegin, const index_t rowEnd,
                  double* out) {
#pragma omp parallel for schedule(static)
    for (index_t i = rowBegin; i < rowEnd; ++i) {
        double t = 0.0;
        for (index_t j = rowOffsets[i]; j < rowOffsets[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Function: spmvCpuRows
//
// Purpose:
//   Computes sparse matrix-vector multiplication for an explicit list of local
//   rows using OpenMP. This is used for the rows that are better handled by the
//   CPU (outlier rows with an unusually large number of non-zeros, or a small
//   tail of rows used to amortize kernel launch overhead).
//
// ****************************************************************************
void spmvCpuRows(const double* val, const index_t* cols, const index_t* rowOffsets,
                 const double* vec, const index_t* rows, const index_t count,
                 double* out) {
#pragma omp parallel for schedule(static)
    for (index_t k = 0; k < count; ++k) {
        const index_t i = rows[k];
        double t = 0.0;
        for (index_t j = rowOffsets[i]; j < rowOffsets[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Function: spmvCsrKernel
//
// Purpose:
//   CUDA implementation of the CSR sparse matrix-vector product. One thread
//   processes one row from the provided row list. Floating point contraction
//   is disabled at compile time so the result is bit-identical to the scalar
//   reference.
//
// ****************************************************************************
__global__ void spmvCsrKernel(const double* __restrict__ val,
                              const index_t* __restrict__ cols,
                              const index_t* __restrict__ rowOffsets,
                              const index_t* __restrict__ rowMap,
                              const double* __restrict__ vec,
                              const index_t count,
                              double* __restrict__ out) {
    const index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const index_t row = rowMap[idx];
        double t = 0.0;
        const index_t start = rowOffsets[row];
        const index_t end = rowOffsets[row + 1];
        for (index_t j = start; j < end; ++j) {
            t += val[j] * __ldg(&vec[cols[j]]);
        }
        out[row] = t;
    }
}

// ****************************************************************************
// Function: verifyResultsLocal
//
// Purpose:
//   Verifies correctness of the local result slice against the reference
//   using OpenMP. rowOffset is the global index of the first local row and is
//   only used for diagnostics.
//
// ****************************************************************************
bool verifyResultsLocal(const double* reference, const double* result,
                        const index_t size, const index_t rowOffset) {
    int valid = 1;
#pragma omp parallel for schedule(static) reduction(&& : valid)
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        bool ok;
        if (std::abs(ref) < 1e-10) {
            ok = std::abs(res) <= MAX_RELATIVE_ERROR;
        } else {
            ok = std::abs((res - ref) / ref) <= MAX_RELATIVE_ERROR;
        }
        if (!ok) {
            valid = 0;
        }
    }

    if (!valid) {
        for (index_t i = 0; i < size; ++i) {
            const double ref = reference[i];
            const double res = result[i];
            if (std::abs(ref) < 1e-10) {
                if (std::abs(res) > MAX_RELATIVE_ERROR) {
                    printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                           rowOffset + i, ref, res);
                    break;
                }
            } else {
                const double relError = std::abs((res - ref) / ref);
                if (relError > MAX_RELATIVE_ERROR) {
                    printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                           rowOffset + i, ref, res, relError);
                    break;
                }
            }
        }
    }
    return valid != 0;
}

// ****************************************************************************
// Function: chooseCpuRows
//
// Purpose:
//   Decides how many contiguous tail rows should be handled by the OpenMP CPU
//   threads. Small problems are dominated by kernel launch overhead and
//   benefit from more CPU participation, whereas large problems are memory
//   bound and are best left almost entirely to the GPU. Outlier rows with an
//   unusually large number of non-zeros are additionally offloaded to the CPU
//   by the caller.
//
// ****************************************************************************
index_t chooseCpuRows(const index_t localRows, const size_t nnz) {
    if (localRows <= 1) {
        return 0;
    }
    double fraction;
    if (nnz < 50000) {
        fraction = 0.75;
    } else if (nnz < 500000) {
        fraction = 0.25;
    } else if (nnz < 5000000) {
        fraction = 0.05;
    } else {
        fraction = 0.0;
    }
    index_t cpuRows = static_cast<index_t>(static_cast<double>(localRows) * fraction);
    if (cpuRows >= localRows) {
        cpuRows = localRows - 1;
    }
    return cpuRows;
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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    // Determine how many MPI ranks share this node and assign one accelerator
    // (round-robin) to each local rank. Also size the OpenMP thread pool so
    // that co-located ranks do not oversubscribe the host.
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);

    int deviceCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA device available for hybrid execution\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    const int cores = omp_get_num_procs();
    const int maxHostThreads = std::max(1, std::min(cores / localSize, 128));

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    // Distribute rows across MPI ranks (contiguous, balanced partition)
    const index_t base = numRows / static_cast<index_t>(size);
    const index_t rem = numRows % static_cast<index_t>(size);
    const index_t rankIdx = static_cast<index_t>(rank);
    const index_t rowStart = rankIdx * base + std::min<index_t>(rankIdx, rem);
    const index_t localRows = base + ((rankIdx < rem) ? 1 : 0);
    const index_t rowEnd = rowStart + localRows;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    // Build the local slice of the CSR matrix (columns and row offsets) and
    // the matching slice of the non-zero values.
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowOffsets;
    index_t globalNnzStart = 0;
    initRandomMatrixLocal(h_cols, h_rowOffsets, globalNnzStart, nItems, numRows, rowStart, rowEnd);

    const size_t nnz = h_cols.size();

    // Scale the OpenMP team to the local problem size. Small problems do not
    // benefit from a large team and pay unnecessary fork/join overhead.
    int hostThreads = maxHostThreads;
    if (nnz < 100000) {
        hostThreads = std::min(hostThreads, 16);
    } else if (nnz < 1000000) {
        hostThreads = std::min(hostThreads, 32);
    } else if (nnz < 10000000) {
        hostThreads = std::min(hostThreads, 64);
    }
    omp_set_num_threads(hostThreads);

    std::vector<double> h_val;
    fillValuesLocal(h_val, globalNnzStart, static_cast<index_t>(nnz), numRows, maxVal);

    // Dense input vector (identical on every rank; it is never modified)
    std::vector<double> h_vec(numRows);
    srand(1);
    fill(h_vec.data(), numRows, maxVal);

    std::vector<double> h_out(localRows, 0.0);

    // Partition the local rows between the accelerator and the OpenMP host
    // threads. Very long rows (which occur in the deterministic dense tail of
    // the reference matrix generator) are the critical path of a one-thread-
    // per-row kernel, so they are offloaded to the CPU. For small problems a
    // contiguous tail is also given to the CPU to amortize launch overhead.
    const index_t tailRows = chooseCpuRows(localRows, nnz);
    const index_t tailBegin = localRows - tailRows;
    const double meanRowLength = (localRows > 0)
                                     ? static_cast<double>(nnz) / static_cast<double>(localRows)
                                     : 0.0;
    const double outlierThreshold = 1.5 * meanRowLength;

    std::vector<index_t> gpuRowList;
    std::vector<index_t> cpuRowList;
    gpuRowList.reserve(localRows);
    for (index_t r = 0; r < localRows; ++r) {
        const double len = static_cast<double>(h_rowOffsets[r + 1] - h_rowOffsets[r]);
        if (r >= tailBegin || len > outlierThreshold) {
            cpuRowList.push_back(r);
        } else {
            gpuRowList.push_back(r);
        }
    }
    // Ensure the hybrid CPU path is exercised whenever a rank has work to do.
    // Moving a single row (the longest one) to the CPU also slightly shortens
    // the accelerator's critical path.
    if (cpuRowList.empty() && gpuRowList.size() > 1) {
        size_t longestPos = 0;
        index_t longestLen = h_rowOffsets[gpuRowList[0] + 1] - h_rowOffsets[gpuRowList[0]];
        for (size_t k = 1; k < gpuRowList.size(); ++k) {
            const index_t r = gpuRowList[k];
            const index_t len = h_rowOffsets[r + 1] - h_rowOffsets[r];
            if (len > longestLen) {
                longestLen = len;
                longestPos = k;
            }
        }
        cpuRowList.push_back(gpuRowList[longestPos]);
        gpuRowList[longestPos] = gpuRowList.back();
        gpuRowList.pop_back();
    }

    const index_t gpuRows = static_cast<index_t>(gpuRowList.size());
    const index_t cpuRows = static_cast<index_t>(cpuRowList.size());

    // Allocate device buffers
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowOffsets = nullptr;
    index_t* d_rowMap = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_val, std::max<size_t>(nnz, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, std::max<size_t>(nnz, 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowOffsets, std::max<size_t>(localRows + 1, 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowMap, std::max<size_t>(gpuRows, 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, std::max<size_t>(numRows, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max<size_t>(localRows, 1) * sizeof(double)));

    if (nnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), nnz * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(d_rowOffsets, h_rowOffsets.data(),
                              (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    } else {
        const index_t zero = 0;
        CUDA_CHECK(cudaMemcpy(d_rowOffsets, &zero, sizeof(index_t), cudaMemcpyHostToDevice));
    }
    if (gpuRows > 0) {
        CUDA_CHECK(cudaMemcpy(d_rowMap, gpuRowList.data(),
                              gpuRows * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    if (numRows > 0) {
        CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));
    }

    // For validation, compute reference solution locally
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        h_reference.resize(localRows);
        spmvCpuRange(h_val.data(), h_cols.data(), h_rowOffsets.data(),
                     h_vec.data(), 0, localRows, h_reference.data());
    }

    constexpr int threadsPerBlock = 128;
    const int gpuBlocks = (gpuRows > 0)
                              ? static_cast<int>((gpuRows + threadsPerBlock - 1) / threadsPerBlock)
                              : 0;

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    // Warm up the accelerator so that clock/power ramp-up is not attributed
    // to the measured computation.
    for (int warm = 0; warm < 5; ++warm) {
        if (gpuRows > 0) {
            spmvCsrKernel<<<gpuBlocks, threadsPerBlock>>>(d_val, d_cols, d_rowOffsets, d_rowMap,
                                                          d_vec, gpuRows, d_out);
        }
        if (cpuRows > 0) {
            spmvCpuRows(h_val.data(), h_cols.data(), h_rowOffsets.data(), h_vec.data(),
                        cpuRowList.data(), cpuRows, h_out.data());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform SpMV computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (gpuRows > 0) {
            spmvCsrKernel<<<gpuBlocks, threadsPerBlock>>>(d_val, d_cols, d_rowOffsets, d_rowMap,
                                                          d_vec, gpuRows, d_out);
        }
        if (cpuRows > 0) {
            spmvCpuRows(h_val.data(), h_cols.data(), h_rowOffsets.data(), h_vec.data(),
                        cpuRowList.data(), cpuRows, h_out.data());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // Bring GPU results back to the host. The device buffer is indexed by the
    // full local row number; only the GPU rows are merged into h_out because
    // the CPU rows were written directly by the OpenMP path.
    if (gpuRows > 0) {
        std::vector<double> h_gpuOut(localRows);
        CUDA_CHECK(cudaMemcpy(h_gpuOut.data(), d_out,
                              localRows * sizeof(double), cudaMemcpyDeviceToHost));
        for (index_t k = 0; k < gpuRows; ++k) {
            const index_t r = gpuRowList[k];
            h_out[r] = h_gpuOut[r];
        }
    }

    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double maxSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long long ms = static_cast<long long>(maxSeconds * 1000.0);
        printf("Computation time: %lld ms\n", ms);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / maxSeconds / 1e9;
        const double avgTime = maxSeconds * 1000.0 / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation (gather full vector on rank 0)
    if (printResults) {
        std::vector<int> recvCounts(size, 0);
        std::vector<int> displs(size, 0);
        const int localRowsInt = static_cast<int>(localRows);
        MPI_Allgather(&localRowsInt, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        int acc = 0;
        for (int r = 0; r < size; ++r) {
            displs[r] = acc;
            acc += recvCounts[r];
        }

        std::vector<double> h_outGlobal;
        if (rank == 0) {
            h_outGlobal.resize(numRows);
        }
        MPI_Gatherv(localRows > 0 ? h_out.data() : nullptr, localRowsInt, MPI_DOUBLE,
                    rank == 0 ? h_outGlobal.data() : nullptr,
                    recvCounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(h_outGlobal, "OutputVector");
        }
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const int localValid = verifyResultsLocal(h_reference.data(), h_out.data(), localRows, rowStart) ? 1 : 0;
        int globalValid = 0;
        MPI_Reduce(&localValid, &globalValid, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (globalValid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowOffsets));
    CUDA_CHECK(cudaFree(d_rowMap));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();

    return exitCode;
}
