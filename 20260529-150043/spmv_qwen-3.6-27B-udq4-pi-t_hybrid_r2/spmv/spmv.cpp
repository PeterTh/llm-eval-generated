#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusparse.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// CUDA error checking
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            printf("CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__,         \
                   cudaGetErrorString(err), err);                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

// MPI error checking
#define MPI_CHECK(call)                                                          \
    do {                                                                         \
        int err = call;                                                          \
        if (err != MPI_SUCCESS) {                                                \
            char errbuf[MPI_MAX_ERROR_STRING];                                   \
            int len;                                                             \
            MPI_Error_string(err, errbuf, &len);                                 \
            printf("MPI error at %s:%d: %s\n", __FILE__, __LINE__, errbuf);     \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values (OpenMP parallelized)
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (static_cast<double>(rand()) /
                         (static_cast<double>(RAND_MAX) + 1.0));
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

// GPU context for SpMV (initialized once per rank)
struct GpuSpMvContext {
    cusparseHandle_t handle = nullptr;
    double *d_val = nullptr;
    int *d_cols = nullptr;
    int *d_rowPtr = nullptr;
    double *d_vec = nullptr;
    double *d_out = nullptr;
    void *d_buffer = nullptr;
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr;
    cusparseDnVecDescr_t vecY = nullptr;
    bool initialized = false;
    index_t localDim = 0;
    index_t nnz = 0;
    index_t globalDim = 0;

    ~GpuSpMvContext() {
        if (matA) cusparseDestroySpMat(matA);
        if (vecX) cusparseDestroyDnVec(vecX);
        if (vecY) cusparseDestroyDnVec(vecY);
        if (handle) cusparseDestroy(handle);
        if (d_val) cudaFree(d_val);
        if (d_cols) cudaFree(d_cols);
        if (d_rowPtr) cudaFree(d_rowPtr);
        if (d_vec) cudaFree(d_vec);
        if (d_out) cudaFree(d_out);
        if (d_buffer) cudaFree(d_buffer);
    }

    void init(const double* h_val, const index_t* h_cols,
              const index_t* h_rowDelimiters,
              const double* h_vec, index_t lDim,
              index_t n, index_t gDim) {
        localDim = lDim;
        nnz = n;
        globalDim = gDim;

        cusparseCreate(&handle);

        CUDA_CHECK(cudaMalloc(&d_val, nnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_rowPtr, (localDim + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_vec, globalDim * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_out, localDim * sizeof(double)));

        // Convert and copy index arrays
        std::vector<int> h_cols_int(nnz);
        std::vector<int> h_rowPtr_int(localDim + 1);
        for (index_t i = 0; i < nnz; ++i) h_cols_int[i] = static_cast<int>(h_cols[i]);
        for (index_t i = 0; i <= localDim; ++i) h_rowPtr_int[i] = static_cast<int>(h_rowDelimiters[i]);

        CUDA_CHECK(cudaMemcpy(d_val, h_val, nnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, h_cols_int.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rowPtr, h_rowPtr_int.data(), (localDim + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vec, h_vec, globalDim * sizeof(double), cudaMemcpyHostToDevice));

        // Create sparse matrix descriptor
        cusparseCreateCsr(&matA, localDim, globalDim, nnz,
                          d_rowPtr, d_cols, d_val,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        // Create dense vector descriptors
        cusparseCreateDnVec(&vecX, globalDim, d_vec, CUDA_R_64F);
        cusparseCreateDnVec(&vecY, localDim, d_out, CUDA_R_64F);

        // Query and allocate buffer
        const double alpha = 1.0, beta = 0.0;
        size_t bufferSize = 0;
        cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY,
                                CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1,
                                &bufferSize);
        CUDA_CHECK(cudaMalloc(&d_buffer, bufferSize));

        initialized = true;
    }

    void compute(double* h_out) {
        if (!initialized || localDim == 0) return;

        const double alpha = 1.0, beta = 0.0;
        CUDA_CHECK(cudaMemset(d_out, 0, localDim * sizeof(double)));
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, vecX, &beta, vecY,
                     CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, d_buffer);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out, d_out, localDim * sizeof(double), cudaMemcpyDeviceToHost));
    }
};

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format
//   (OpenMP parallelized for reference computation)
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
// Function: verifyResults
//
// Purpose:
//   Verifies correctness of results by comparing to reference solution
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    bool failed = false;
#pragma omp parallel for schedule(static) reduction(||:failed)
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                failed = true;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                failed = true;
            }
        }
    }
    // Find first failure for reporting
    if (failed) {
        for (index_t i = 0; i < size; ++i) {
            const double ref = reference[i];
            const double res = result[i];
            bool thisFailed = false;
            if (std::abs(ref) < 1e-10) {
                if (std::abs(res) > MAX_RELATIVE_ERROR) thisFailed = true;
            } else {
                if (std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR) thisFailed = true;
            }
            if (thisFailed) {
                printf("First mismatch at index %u: ref=%.10e, res=%.10e\n",
                       i, ref, res);
                break;
            }
        }
    }
    return !failed;
}

// ****************************************************************************
// Partition CSR data for MPI distribution
//
// Each MPI rank gets a contiguous range of rows. We compute the local
// CSR arrays (val, cols, rowDelimiters) for each rank.
// ****************************************************************************
struct PartitionInfo {
    index_t localDim;       // Number of rows for this rank
    index_t localNnz;       // Number of non-zeros for this rank
    index_t globalRowStart; // First global row for this rank
    index_t nnzOffset;      // Offset into global nnz arrays
};

PartitionInfo computePartition(const index_t* globalRowDelimiters,
                                const index_t globalDim,
                                const int rank, const int numRanks) {
    // Distribute rows as evenly as possible
    index_t rowsPerRank = globalDim / numRanks;
    index_t extraRows = globalDim % numRanks;

    index_t rowStart = 0;
    for (int r = 0; r < rank; ++r) {
        rowStart += rowsPerRank + (r < extraRows ? 1 : 0);
    }
    index_t rowEnd = rowStart + rowsPerRank + (rank < extraRows ? 1 : 0);

    index_t nnzOffset = globalRowDelimiters[rowStart];
    index_t localNnz = globalRowDelimiters[rowEnd] - nnzOffset;

    PartitionInfo info;
    info.localDim = rowEnd - rowStart;
    info.localNnz = localNnz;
    info.globalRowStart = rowStart;
    info.nnzOffset = nnzOffset;
    return info;
}

void printUsage(const char* progName) {
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
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
    // Initialize MPI
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank, numRanks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

    // Check GPU availability for this rank
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            printf("Error: No CUDA devices available!\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Assign GPU to rank (round-robin for multi-GPU nodes)
    int gpuId = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(gpuId));

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks must parse)
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
            MPI_CHECK(MPI_Finalize());
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_CHECK(MPI_Finalize());
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    // Print configuration (rank 0 only)
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Parallel backend: Hybrid MPI + OpenMP + CUDA\n");
        printf("MPI ranks: %d\n", numRanks);
        printf("CUDA devices: %d\n", deviceCount);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // =========================================================================
    // Data initialization on rank 0
    // =========================================================================
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec;

    // Rank 0 generates the full matrix and vector
    if (rank == 0) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_vec.resize(numRows);

        // Use OpenMP for parallel initialization
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // =========================================================================
    // Compute partition info for each rank
    // =========================================================================
    // Rank 0 broadcasts rowDelimiters to compute partition info everywhere
    std::vector<index_t> localRowDelimiters(numRows + 1);
    if (rank == 0) {
        // Copy from h_rowDelimiters before broadcasting
        std::copy(h_rowDelimiters.begin(), h_rowDelimiters.end(), localRowDelimiters.begin());
    }
    MPI_CHECK(MPI_Bcast(localRowDelimiters.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));

    PartitionInfo partition = computePartition(localRowDelimiters.data(),
                                                numRows, rank, numRanks);

    const index_t localDim = partition.localDim;
    const index_t localNnz = partition.localNnz;

    // =========================================================================
    // Allocate local buffers
    // =========================================================================
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    std::vector<index_t> localRowDelims(localDim + 1);
    std::vector<double> localVec(numRows);   // Full vector needed for SpMV
    std::vector<double> localOut(localDim);

    // =========================================================================
    // Distribute data from rank 0
    // =========================================================================
    // Build sendcounts and displs for Scatterv (needed on all ranks)
    std::vector<int> sendcounts(numRanks);
    std::vector<int> displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        PartitionInfo p = computePartition(localRowDelimiters.data(),
                                            numRows, r, numRanks);
        sendcounts[r] = p.localNnz;
        displs[r] = p.nnzOffset;
    }

    // Broadcast full vector to all ranks (each rank needs full vector for SpMV)
    if (rank == 0) {
        std::copy(h_vec.begin(), h_vec.end(), localVec.begin());
    }
    MPI_CHECK(MPI_Bcast(localVec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD));

    // Scatter val and cols using displacements based on partition
    if (rank == 0) {
        MPI_CHECK(MPI_Scatterv(h_val.data(), sendcounts.data(), displs.data(),
                                MPI_DOUBLE, localVal.data(), localNnz,
                                MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Scatterv(h_cols.data(), sendcounts.data(), displs.data(),
                                MPI_UINT32_T, localCols.data(), localNnz,
                                MPI_UINT32_T, 0, MPI_COMM_WORLD));
    } else {
        MPI_CHECK(MPI_Scatterv(nullptr, sendcounts.data(), displs.data(),
                                MPI_DOUBLE, localVal.data(), localNnz,
                                MPI_DOUBLE, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Scatterv(nullptr, sendcounts.data(), displs.data(),
                                MPI_UINT32_T, localCols.data(), localNnz,
                                MPI_UINT32_T, 0, MPI_COMM_WORLD));
    }

    // Build local rowDelimiters for this rank
    for (index_t i = 0; i <= localDim; ++i) {
        localRowDelims[i] = localRowDelimiters[partition.globalRowStart + i]
                          - partition.nnzOffset;
    }

    // =========================================================================
    // Validate on CPU (rank 0 only, using OpenMP)
    // =========================================================================
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution (CPU + OpenMP)...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // =========================================================================
    // SpMV computation on GPU (each rank on its assigned GPU)
    // =========================================================================
    if (rank == 0) {
        printf("Computing SpMV (GPU + CUDA)...\n");
    }

    // Initialize GPU context with pre-allocated memory
    GpuSpMvContext gpuCtx;
    if (localDim > 0) {
        gpuCtx.init(localVal.data(), localCols.data(), localRowDelims.data(),
                    localVec.data(), localDim, localNnz, numRows);
    }

    // Warmup iteration (not timed)
    if (localDim > 0) {
        gpuCtx.compute(localOut.data());
    }

    // Timed iterations
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localDim > 0) {
            gpuCtx.compute(localOut.data());
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // =========================================================================
    // Gather results back to rank 0
    // =========================================================================
    std::vector<double> h_out;
    if (rank == 0) {
        h_out.resize(numRows);
    }

    std::vector<int> recvcounts(numRanks);
    std::vector<int> recvdispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        PartitionInfo p = computePartition(localRowDelimiters.data(),
                                            numRows, r, numRanks);
        recvcounts[r] = p.localDim;
        recvdispls[r] = p.globalRowStart;
    }

    MPI_CHECK(MPI_Gatherv(localOut.data(), localDim, MPI_DOUBLE,
                           h_out.data(), recvcounts.data(), recvdispls.data(),
                           MPI_DOUBLE, 0, MPI_COMM_WORLD));

    // =========================================================================
    // Performance reporting
    // =========================================================================
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

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
            }
        }
    }

    // Cleanup
    MPI_CHECK(MPI_Finalize());
    return 0;
}
