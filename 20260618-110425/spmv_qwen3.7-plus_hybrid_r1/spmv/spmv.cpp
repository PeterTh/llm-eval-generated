#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_err)); \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); \
        } \
    } while (0)

// ****************************************************************************
// CUDA Kernel: spmvCsrKernel
//   Sparse matrix-vector multiplication in CSR format.
//   One thread per row — each thread computes the dot product of its row
//   with the dense vector.  Uses __ldg for read-only cache loads of vec.
// ****************************************************************************
__global__ void spmvCsrKernel(const double* __restrict__ val,
                               const index_t* __restrict__ cols,
                               const index_t* __restrict__ rowDelimiters,
                               const double* __restrict__ vec,
                               const index_t dim,
                               double* __restrict__ out) {
    const index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dim) return;

    double sum = 0.0;
    const index_t rowStart = rowDelimiters[row];
    const index_t rowEnd   = rowDelimiters[row + 1];
    for (index_t j = rowStart; j < rowEnd; ++j) {
        sum += val[j] * __ldg(&vec[cols[j]]);
    }
    out[row] = sum;
}

// ****************************************************************************
// Function: fill
//   Initialize array with random values (sequential for reproducibility)
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: initRandomMatrix
//   Assigns random positions in CSR format (sequential for reproducibility)
// ****************************************************************************
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));
    srand(8675309);
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
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    rowDelimiters[dim] = n;
}

// ****************************************************************************
// Function: spmvCpu  (OpenMP parallel — used for reference solution)
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
    #pragma omp parallel for schedule(dynamic, 64)
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Function: verifyResults  (OpenMP parallel)
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    int failCount = 0;
    #pragma omp parallel for reduction(+:failCount)
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        bool failed = false;
        if (std::abs(ref) < 1e-10) {
            failed = (std::abs(res) > MAX_RELATIVE_ERROR);
        } else {
            failed = (std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR);
        }
        if (failed) ++failCount;
    }
    if (failCount > 0) {
        printf("Validation failed for %d out of %u elements\n", failCount, size);
    }
    return failCount == 0;
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
// Main — Hybrid MPI + OpenMP + CUDA SpMV
//
// MPI:   Row-wise domain decomposition across processes
// CUDA:  SpMV kernel on GPU (one thread per row, CSR format)
// OpenMP: CPU-side parallelism (reference SpMV, verification, data adjustment)
// ****************************************************************************
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- Parse command line arguments (all ranks) ----
    index_t numRows   = 1024;
    index_t sparsity  = 10;
    index_t iterations = 10;
    double  maxVal    = 1.0;
    bool    validate  = false;
    bool    printResults = false;

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
            printUsage(argv[0]);
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

    const index_t nItems = (numRows * numRows) / sparsity;

    // ---- Banner (rank 0) ----
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Parallelization: MPI (%d ranks) + OpenMP + CUDA\n", size);
        int numThreads = 0;
        #pragma omp parallel
        {
            #pragma omp master
            numThreads = omp_get_num_threads();
        }
        printf("OpenMP threads: %d\n", numThreads);

        int deviceCount = 0;
        cudaError_t cerr = cudaGetDeviceCount(&deviceCount);
        if (cerr == cudaSuccess && deviceCount > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            printf("GPU: %s\n", prop.name);
        }

        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ===== DATA GENERATION (Rank 0 only) =====
    std::vector<double>  h_val, h_vec;
    std::vector<index_t> h_cols, h_rowDelimiters;

    if (rank == 0) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_vec.resize(numRows);

        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // ===== COMPUTE ROW DISTRIBUTION =====
    const index_t rowsPerRank = numRows / size;
    const index_t rem         = numRows % size;
    const index_t localNumRows = rowsPerRank + (static_cast<index_t>(rank) < rem ? 1 : 0);
    const index_t startRow     = static_cast<index_t>(rank) * rowsPerRank
                                 + std::min(static_cast<index_t>(rank), rem);

    // Rank 0 builds send-counts / displacements for Scatterv
    std::vector<int> nnzCounts, nnzDispls, rowCounts, rowDispls;
    if (rank == 0) {
        nnzCounts.resize(size);
        nnzDispls.resize(size);
        rowCounts.resize(size);
        rowDispls.resize(size);
        for (int r = 0; r < size; r++) {
            const index_t rStart = static_cast<index_t>(r) * rowsPerRank
                                   + std::min(static_cast<index_t>(r), rem);
            const index_t rEnd   = rStart + rowsPerRank
                                   + (static_cast<index_t>(r) < rem ? 1 : 0);
            nnzCounts[r] = static_cast<int>(h_rowDelimiters[rEnd] - h_rowDelimiters[rStart]);
            nnzDispls[r] = static_cast<int>(h_rowDelimiters[rStart]);
            rowCounts[r] = static_cast<int>(rEnd - rStart + 1);
            rowDispls[r] = static_cast<int>(rStart);
        }
    }

    // ===== SCATTER ROW DELIMITERS =====
    std::vector<index_t> local_rowDelimiters(localNumRows + 1);
    MPI_Scatterv(
        (rank == 0) ? h_rowDelimiters.data() : nullptr,
        (rank == 0) ? rowCounts.data()  : nullptr,
        (rank == 0) ? rowDispls.data()  : nullptr,
        MPI_UINT32_T,
        local_rowDelimiters.data(),
        static_cast<int>(localNumRows + 1),
        MPI_UINT32_T,
        0, MPI_COMM_WORLD);

    // Adjust to local indexing (OpenMP parallel)
    const index_t nnzOffset = local_rowDelimiters[0];
    #pragma omp parallel for
    for (int i = 0; i <= static_cast<int>(localNumRows); i++) {
        local_rowDelimiters[i] -= nnzOffset;
    }
    const index_t localNnz = local_rowDelimiters[localNumRows];

    // ===== SCATTER VAL AND COLS =====
    std::vector<double>  local_val(localNnz);
    std::vector<index_t> local_cols(localNnz);

    MPI_Scatterv(
        (rank == 0) ? h_val.data() : nullptr,
        (rank == 0) ? nnzCounts.data() : nullptr,
        (rank == 0) ? nnzDispls.data() : nullptr,
        MPI_DOUBLE,
        local_val.data(), static_cast<int>(localNnz), MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    MPI_Scatterv(
        (rank == 0) ? h_cols.data() : nullptr,
        (rank == 0) ? nnzCounts.data() : nullptr,
        (rank == 0) ? nnzDispls.data() : nullptr,
        MPI_UINT32_T,
        local_cols.data(), static_cast<int>(localNnz), MPI_UINT32_T,
        0, MPI_COMM_WORLD);

    // ===== BROADCAST DENSE VECTOR =====
    std::vector<double> local_vec(numRows);
    if (rank == 0) {
        local_vec = h_vec;  // Copy filled vector before broadcast
    }
    MPI_Bcast(local_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Free rank-0 full arrays (no longer needed)
    if (rank == 0) {
        h_val.clear(); h_val.shrink_to_fit();
        h_cols.clear(); h_cols.shrink_to_fit();
        h_rowDelimiters.clear(); h_rowDelimiters.shrink_to_fit();
    }

    // ===== SETUP GPU =====
    CUDA_CHECK(cudaSetDevice(0));

    double  *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rowDelim = nullptr;

    if (localNnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_val,  localNnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, localNnz * sizeof(index_t)));
        CUDA_CHECK(cudaMemcpy(d_val,  local_val.data(),  localNnz * sizeof(double),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, local_cols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&d_rowDelim, (localNumRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec,      numRows * sizeof(double)));
    if (localNumRows > 0) {
        CUDA_CHECK(cudaMalloc(&d_out, localNumRows * sizeof(double)));
    }

    CUDA_CHECK(cudaMemcpy(d_rowDelim, local_rowDelimiters.data(),
                           (localNumRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, local_vec.data(),
                           numRows * sizeof(double), cudaMemcpyHostToDevice));

    // ===== REFERENCE SOLUTION (OpenMP CPU, for validation) =====
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        h_reference.resize(localNumRows);
        spmvCpu(local_val.data(), local_cols.data(), local_rowDelimiters.data(),
                local_vec.data(), localNumRows, h_reference.data());
    }

    // ===== KERNEL LAUNCH CONFIG =====
    const int blockSize = 256;
    const int gridSize  = (localNumRows > 0)
                          ? (static_cast<int>(localNumRows) + blockSize - 1) / blockSize
                          : 0;

    // Warm-up launch
    if (gridSize > 0) {
        spmvCsrKernel<<<gridSize, blockSize>>>(
            d_val, d_cols, d_rowDelim, d_vec,
            static_cast<index_t>(localNumRows), d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ===== TIMED SpMV (CUDA events) =====
    if (rank == 0) printf("Computing SpMV...\n");

    MPI_Barrier(MPI_COMM_WORLD);

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);
    cudaEventRecord(evStart);

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (gridSize > 0) {
            spmvCsrKernel<<<gridSize, blockSize>>>(
                d_val, d_cols, d_rowDelim, d_vec,
                static_cast<index_t>(localNumRows), d_out);
        }
    }

    cudaEventRecord(evStop);
    cudaEventSynchronize(evStop);

    float localGpuMs = 0.0f;
    cudaEventElapsedTime(&localGpuMs, evStart, evStop);

    // Max GPU time across all ranks
    double maxGpuMs = 0.0;
    double localMs  = static_cast<double>(localGpuMs);
    MPI_Allreduce(&localMs, &maxGpuMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // ===== COPY RESULTS BACK =====
    std::vector<double> local_out(localNumRows);
    if (localNumRows > 0) {
        CUDA_CHECK(cudaMemcpy(local_out.data(), d_out,
                               localNumRows * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // ===== GATHER RESULTS TO RANK 0 =====
    std::vector<double>  h_out;
    std::vector<int>     outCounts, outDispls;
    if (rank == 0) {
        h_out.resize(numRows);
        outCounts.resize(size);
        outDispls.resize(size);
        for (int r = 0; r < size; r++) {
            const index_t rStart   = static_cast<index_t>(r) * rowsPerRank
                                     + std::min(static_cast<index_t>(r), rem);
            const index_t rNumRows = rowsPerRank
                                     + (static_cast<index_t>(r) < rem ? 1 : 0);
            outCounts[r] = static_cast<int>(rNumRows);
            outDispls[r] = static_cast<int>(rStart);
        }
    }

    MPI_Gatherv(
        local_out.data(), static_cast<int>(localNumRows), MPI_DOUBLE,
        (rank == 0) ? h_out.data()   : nullptr,
        (rank == 0) ? outCounts.data() : nullptr,
        (rank == 0) ? outDispls.data() : nullptr,
        MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // ===== PRINT RESULTS (Rank 0) =====
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", maxGpuMs);

        const double timeSec = maxGpuMs / 1000.0;
        const double gflops  = (2.0 * nItems * iterations) / timeSec / 1e9;
        const double avgTime = maxGpuMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }
    }

    // ===== VALIDATION =====
    bool localValid = true;
    if (validate) {
        localValid = verifyResults(h_reference.data(), local_out.data(), localNumRows);
    }

    int localValidInt  = localValid ? 1 : 0;
    int globalValidInt = 1;
    MPI_Allreduce(&localValidInt, &globalValidInt, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if (rank == 0 && validate) {
        if (globalValidInt) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // ===== CLEANUP =====
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);

    if (d_val)       CUDA_CHECK(cudaFree(d_val));
    if (d_cols)      CUDA_CHECK(cudaFree(d_cols));
    if (d_rowDelim)  CUDA_CHECK(cudaFree(d_rowDelim));
    if (d_vec)       CUDA_CHECK(cudaFree(d_vec));
    if (d_out)       CUDA_CHECK(cudaFree(d_out));

    MPI_Finalize();
    return (validate && !globalValidInt) ? 1 : 0;
}
