#include <mpi.h>
#include <cuda_runtime.h>
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

// Error checking macros
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define MPI_CHECK(call) do { \
    int mpi_err = (call); \
    if (mpi_err != MPI_SUCCESS) { \
        char mpi_err_str[MPI_MAX_ERROR_STRING]; \
        int mpi_err_len; \
        MPI_Error_string(mpi_err, mpi_err_str, &mpi_err_len); \
        fprintf(stderr, "MPI error at %s:%d: %s\n", __FILE__, __LINE__, mpi_err_str); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// ****************************************************************************
// CUDA Kernel: Warp-per-row CSR SpMV
// Each warp of 32 threads cooperates to compute one row's dot product.
// ****************************************************************************
__global__ void spmvCsrWarpKernel(const double* __restrict__ val,
                                   const index_t* __restrict__ cols,
                                   const index_t* __restrict__ rowDelimiters,
                                   const double* __restrict__ vec,
                                   const index_t numRows,
                                   double* __restrict__ out) {
    const index_t globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t warpId = globalTid / 32;
    const index_t laneId = globalTid % 32;

    if (warpId >= numRows) return;

    double sum = 0.0;
    const index_t rowStart = rowDelimiters[warpId];
    const index_t rowEnd = rowDelimiters[warpId + 1];

    for (index_t j = rowStart + laneId; j < rowEnd; j += 32) {
        sum += val[j] * vec[cols[j]];
    }

    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (laneId == 0) {
        out[warpId] = sum;
    }
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
// Function: spmvCpu (OpenMP-parallel reference implementation)
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
    #pragma omp parallel for schedule(dynamic, 64)
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
// Function: verifyResults (OpenMP-parallel)
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    bool allValid = true;
    #pragma omp parallel
    {
        bool localValid = true;
        #pragma omp for schedule(static)
        for (index_t i = 0; i < size; ++i) {
            if (!localValid) continue;
            const double ref = reference[i];
            const double res = result[i];
            if (std::abs(ref) < 1e-10) {
                if (std::abs(res) > MAX_RELATIVE_ERROR) {
                    #pragma omp critical
                    {
                        printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                        allValid = false;
                    }
                    localValid = false;
                }
            } else {
                const double relError = std::abs((res - ref) / ref);
                if (relError > MAX_RELATIVE_ERROR) {
                    #pragma omp critical
                    {
                        printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                               i, ref, res, relError);
                        allValid = false;
                    }
                    localValid = false;
                }
            }
        }
    }
    return allValid;
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
    // MPI initialization
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank, numRanks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks)
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
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Hybrid parallelization: MPI (%d ranks) + OpenMP + CUDA\n", numRanks);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute local row range for this rank
    const index_t startRow = (numRows * static_cast<index_t>(rank)) / static_cast<index_t>(numRanks);
    const index_t endRow = (numRows * static_cast<index_t>(rank + 1)) / static_cast<index_t>(numRanks);
    const index_t localNumRows = endRow - startRow;

    // Rank 0 initializes all data
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec;

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_vec.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // For validation, compute reference solution on rank 0 using OpenMP-parallel CPU SpMV
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Broadcast the full rowDelimiters so all ranks can compute their local nnz
    MPI_CHECK(MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));

    // Each rank computes its local nnz count
    const index_t localStartNnz = h_rowDelimiters[startRow];
    const index_t localEndNnz = h_rowDelimiters[endRow];
    const index_t localNnz = localEndNnz - localStartNnz;

    // Prepare sendcounts and displacements on rank 0 for Scatterv
    std::vector<int> sendCountsVal(numRanks);
    std::vector<int> displsVal(numRanks);
    std::vector<int> sendCountsCols(numRanks);
    std::vector<int> displsCols(numRanks);

    if (rank == 0) {
        for (int r = 0; r < numRanks; ++r) {
            index_t sr = (numRows * static_cast<index_t>(r)) / static_cast<index_t>(numRanks);
            index_t er = (numRows * static_cast<index_t>(r + 1)) / static_cast<index_t>(numRanks);
            sendCountsVal[r] = static_cast<int>(h_rowDelimiters[er] - h_rowDelimiters[sr]);
            displsVal[r] = static_cast<int>(h_rowDelimiters[sr]);
            sendCountsCols[r] = static_cast<int>(h_rowDelimiters[er] - h_rowDelimiters[sr]);
            displsCols[r] = static_cast<int>(h_rowDelimiters[sr]);
        }
    }

    // Scatter val and cols to all ranks
    std::vector<double> h_localVal(localNnz);
    std::vector<index_t> h_localCols(localNnz);

    MPI_CHECK(MPI_Scatterv(h_val.data(), sendCountsVal.data(), displsVal.data(), MPI_DOUBLE,
                           h_localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE,
                           0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatterv(h_cols.data(), sendCountsCols.data(), displsCols.data(), MPI_UINT32_T,
                           h_localCols.data(), static_cast<int>(localNnz), MPI_UINT32_T,
                           0, MPI_COMM_WORLD));

    // Broadcast the full dense vector to all ranks
    if (rank != 0) {
        h_vec.resize(numRows);
    }
    MPI_CHECK(MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD));

    // Extract and renumber local row delimiters
    std::vector<index_t> h_localRowDelims(localNumRows + 1);
    for (index_t i = 0; i <= localNumRows; ++i) {
        h_localRowDelims[i] = h_rowDelimiters[startRow + i] - localStartNnz;
    }

    // Free global data on rank 0 (no longer needed)
    if (rank == 0) {
        h_val.clear(); h_val.shrink_to_fit();
        h_cols.clear(); h_cols.shrink_to_fit();
    }
    h_rowDelimiters.clear(); h_rowDelimiters.shrink_to_fit();

    // ---- CUDA setup: allocate GPU memory and copy data ----
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rowDelims = nullptr;

    if (localNnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, localNnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, localNnz * sizeof(index_t)));
        CUDA_CHECK(cudaMemcpy(d_val, h_localVal.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, h_localCols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rowDelims, (localNumRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_out, localNumRows * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelims, h_localRowDelims.data(), (localNumRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));

    // Free host local data after transfer to GPU
    h_localVal.clear(); h_localVal.shrink_to_fit();
    h_localCols.clear(); h_localCols.shrink_to_fit();

    // Kernel launch configuration: warp-per-row (32 threads per row)
    const int blockSize = 256; // 8 warps per block
    const int numWarpsPerBlock = blockSize / 32;
    const int gridSize = (localNumRows > 0) ? (static_cast<int>(localNumRows) + numWarpsPerBlock - 1) / numWarpsPerBlock : 0;

    // Perform SpMV computation on GPU
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    // Synchronize all ranks before timing
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto startTime = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (gridSize > 0) {
            spmvCsrWarpKernel<<<gridSize, blockSize>>>(d_val, d_cols, d_rowDelims, d_vec, localNumRows, d_out);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto endTime = std::chrono::high_resolution_clock::now();

    // Compute local duration and find maximum across all ranks
    double localDurationMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    double maxDurationMs;
    MPI_CHECK(MPI_Allreduce(&localDurationMs, &maxDurationMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    // Copy result back to host
    std::vector<double> h_localOut(localNumRows);
    if (localNumRows > 0) {
        CUDA_CHECK(cudaMemcpy(h_localOut.data(), d_out, localNumRows * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Gather results to rank 0 using MPI_Gatherv
    std::vector<double> h_out;
    std::vector<int> recvCounts, recvDispls;

    if (rank == 0) {
        h_out.resize(numRows);
        recvCounts.resize(numRanks);
        recvDispls.resize(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            index_t sr = (numRows * static_cast<index_t>(r)) / static_cast<index_t>(numRanks);
            index_t er = (numRows * static_cast<index_t>(r + 1)) / static_cast<index_t>(numRanks);
            recvCounts[r] = static_cast<int>(er - sr);
            recvDispls[r] = static_cast<int>(sr);
        }
    }

    MPI_CHECK(MPI_Gatherv(h_localOut.data(), static_cast<int>(localNumRows), MPI_DOUBLE,
                          h_out.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                          0, MPI_COMM_WORLD));

    // Rank 0 prints results
    int exitCode = 0;
    if (rank == 0) {
        auto durationCount = static_cast<long>(maxDurationMs);
        printf("Computation time: %ld ms\n", durationCount);

        const double gflops = (2.0 * nItems * iterations) / (maxDurationMs / 1000.0) / 1e9;
        const double avgTime = maxDurationMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    // Cleanup CUDA resources
    if (d_val) CUDA_CHECK(cudaFree(d_val));
    if (d_cols) CUDA_CHECK(cudaFree(d_cols));
    if (d_vec) CUDA_CHECK(cudaFree(d_vec));
    if (d_rowDelims) CUDA_CHECK(cudaFree(d_rowDelims));
    if (d_out) CUDA_CHECK(cudaFree(d_out));

    MPI_CHECK(MPI_Finalize());

    return exitCode;
}
