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
constexpr int CUDA_BLOCK_SIZE = 256;

// ****************************************************************************
// CUDA error checking macro
// ****************************************************************************
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// ****************************************************************************
// CUDA Kernel: CSR-based SpMV
//
// Each thread handles one row of the matrix, computing the dot product
// of that row with the input vector.
// ****************************************************************************
__global__ void spmvCudaKernel(const double* __restrict__ val,
                               const index_t* __restrict__ cols,
                               const index_t* __restrict__ rowDelimiters,
                               const double* __restrict__ vec,
                               const index_t numRows,
                               double* __restrict__ out) {
    index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows) return;

    double sum = 0.0;
    const index_t rowStart = rowDelimiters[row];
    const index_t rowEnd = rowDelimiters[row + 1];

    for (index_t j = rowStart; j < rowEnd; ++j) {
        sum += val[j] * vec[cols[j]];
    }
    out[row] = sum;
}

// ****************************************************************************
// Function: fill (OpenMP-parallelized)
//
// Purpose:
//   Initialize array with random values using thread-safe RNG
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num() + 1;
        #pragma omp for
        for (index_t i = 0; i < n; ++i) {
            A[i] = maxVal * (static_cast<double>(rand_r(&seed)) /
                             (static_cast<double>(RAND_MAX) + 1.0));
        }
    }
}

// ****************************************************************************
// Function: initRandomMatrix
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix using CSR format. Only called by rank 0.
// ****************************************************************************
void initRandomMatrix(index_t* cols, index_t* rowDelimiters,
                      const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;
    double prob = static_cast<double>(n) /
                  (static_cast<double>(dim) * static_cast<double>(dim));
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
// Function: spmvCpu (reference implementation)
//
// Purpose:
//   Sequential SpMV for computing reference solution on rank 0
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            t += val[j] * vec[cols[j]];
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
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                       i, ref, res);
                return false;
            }
        } else {
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e"
                       " (rel error: %.10e)\n",
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
// Function: runSpmvGpu
//
// Purpose:
//   Runs SpMV on GPU for a specified number of iterations.
//   GPU memory is expected to be pre-allocated and populated.
// ****************************************************************************
void runSpmvGpu(const double* d_val, const index_t* d_cols,
                const index_t* d_rowDelimiters, const double* d_vec,
                const index_t localRows, double* d_out,
                const index_t iterations) {
    const int gridSize = (localRows + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCudaKernel<<<gridSize, CUDA_BLOCK_SIZE>>>(
            d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int mpiRank, mpiNumRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiNumRanks);

    const index_t rank = static_cast<index_t>(mpiRank);
    const index_t numRanks = static_cast<index_t>(mpiNumRanks);

    // Set CUDA device based on MPI rank (for multi-GPU setups)
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices > 0) {
        int deviceId = mpiRank % numDevices;
        CUDA_CHECK(cudaSetDevice(deviceId));
    }

    // Parameters
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Rank 0 parses command line arguments and broadcasts
    if (rank == 0) {
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
    }

    // Broadcast parameters to all ranks
    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    const index_t nItems = (numRows * numRows) / sparsity;

    // Each rank determines its local row range
    const index_t rowsPerRank = numRows / numRanks;
    const index_t remainder = numRows % numRanks;
    const index_t localRows = rowsPerRank + (rank < remainder ? 1 : 0);

    // Compute localStartRow properly accounting for remainder distribution
    index_t localStartRow;
    if (rank < remainder) {
        localStartRow = rank * (rowsPerRank + 1);
    } else {
        localStartRow = remainder * (rowsPerRank + 1) + (rank - remainder) * rowsPerRank;
    }
    const index_t localEndRow = localStartRow + localRows;

    // Rank 0 holds and initializes the full data
    std::vector<double> h_val_full;
    std::vector<index_t> h_cols_full;
    std::vector<index_t> h_rowDelimiters_full;
    std::vector<double> h_vec_full;

    // Reference solution (compute before distribution)
    std::vector<double> h_reference;

    if (rank == 0) {
        h_val_full.resize(nItems);
        h_cols_full.resize(nItems);
        h_rowDelimiters_full.resize(numRows + 1);
        h_vec_full.resize(numRows);

        printf("Hybrid MPI+OpenMP+CUDA SpMV Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) /
                                        (static_cast<double>(numRows) * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
        printf("CUDA devices available: %d\n", numDevices);
        printf("OpenMP threads: %d\n", omp_get_max_threads());

        printf("Initializing data structures (rank 0)...\n");
        fill(h_vec_full.data(), numRows, maxVal);
        fill(h_val_full.data(), nItems, maxVal);
        srand(8675309);
        initRandomMatrix(h_cols_full.data(), h_rowDelimiters_full.data(), nItems, numRows);

        // Compute reference solution on rank 0 using the same data
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val_full.data(), h_cols_full.data(),
                    h_rowDelimiters_full.data(), h_vec_full.data(),
                    numRows, h_reference.data());
        }
    }

    // Broadcast the full rowDelimiters to all ranks (needed to determine local nnz)
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    if (rank == 0) {
        std::copy(h_rowDelimiters_full.begin(), h_rowDelimiters_full.end(),
                  h_rowDelimiters.begin());
    }
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1),
              MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Broadcast the full vector to all ranks (each row can reference any column)
    std::vector<double> h_vec(numRows);
    if (rank == 0) {
        std::copy(h_vec_full.begin(), h_vec_full.end(), h_vec.begin());
    }
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Determine local non-zero count
    const index_t localNnz = h_rowDelimiters[localEndRow] - h_rowDelimiters[localStartRow];

    // Prepare local rowDelimiters (adjusted to start from 0)
    std::vector<index_t> h_rowDelimiters_local(localRows + 1);
    #pragma omp parallel for
    for (index_t i = 0; i <= localRows; ++i) {
        h_rowDelimiters_local[i] =
            h_rowDelimiters[localStartRow + i] - h_rowDelimiters[localStartRow];
    }

    // Distribute val and cols arrays using MPI_Scatterv
    std::vector<int> sendCounts_val(numRanks);
    std::vector<int> displacements_val(numRanks);
    if (rank == 0) {
        int offset = 0;
        for (index_t r = 0; r < numRanks; ++r) {
            index_t rStart = (r < remainder)
                ? r * (rowsPerRank + 1)
                : remainder * (rowsPerRank + 1) + (r - remainder) * rowsPerRank;
            index_t rLen = rowsPerRank + (r < remainder ? 1 : 0);
            index_t rEnd = rStart + rLen;
            sendCounts_val[r] =
                static_cast<int>(h_rowDelimiters_full[rEnd] - h_rowDelimiters_full[rStart]);
            displacements_val[r] =
                static_cast<int>(h_rowDelimiters_full[rStart]);
            offset += rLen;
        }
    }

    std::vector<double> h_val_local(localNnz);
    std::vector<index_t> h_cols_local(localNnz);

    // Scatter val
    MPI_Scatterv(
        rank == 0 ? h_val_full.data() : nullptr,
        sendCounts_val.data(),
        displacements_val.data(),
        MPI_DOUBLE,
        h_val_local.data(),
        static_cast<int>(localNnz),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);

    // Scatter cols
    MPI_Scatterv(
        rank == 0 ? h_cols_full.data() : nullptr,
        sendCounts_val.data(),
        displacements_val.data(),
        MPI_UINT32_T,
        h_cols_local.data(),
        static_cast<int>(localNnz),
        MPI_UINT32_T,
        0,
        MPI_COMM_WORLD);

    // Free full matrix data on rank 0 (keep reference for later validation)
    if (rank == 0) {
        std::vector<double>().swap(h_val_full);
        std::vector<index_t>().swap(h_cols_full);
        std::vector<double>().swap(h_vec_full);
    }
    std::vector<index_t>().swap(h_rowDelimiters);

    // Allocate GPU memory for this rank's local data
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rowDelimiters_d = nullptr;

    if (localNnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, localNnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, localNnz * sizeof(index_t)));
    }
    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(&d_out, localRows * sizeof(double)));
    }
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters_d, (localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));

    // Copy local data to GPU
    if (localNnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_val, h_val_local.data(), localNnz * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, h_cols_local.data(), localNnz * sizeof(index_t),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters_d, h_rowDelimiters_local.data(),
                          (localRows + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Perform GPU-accelerated SpMV computation
    if (rank == 0) {
        printf("Computing SpMV on GPU (%d MPI ranks, %d CUDA devices)...\n",
               numRanks, numDevices);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSpmvGpu(d_val, d_cols, d_rowDelimiters_d, d_vec,
               localRows, d_out, iterations);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start);
    MPI_Barrier(MPI_COMM_WORLD);

    // Copy local results from GPU to host
    std::vector<double> h_out_local;
    if (localRows > 0) {
        h_out_local.resize(localRows);
        CUDA_CHECK(cudaMemcpy(h_out_local.data(), d_out, localRows * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    // Free GPU memory
    if (localNnz > 0) {
        CUDA_CHECK(cudaFree(d_val));
        CUDA_CHECK(cudaFree(d_cols));
    }
    CUDA_CHECK(cudaFree(d_rowDelimiters_d));
    CUDA_CHECK(cudaFree(d_vec));
    if (localRows > 0) {
        CUDA_CHECK(cudaFree(d_out));
    }

    // Gather all results to rank 0
    std::vector<double> h_out_full;
    std::vector<int> recvCounts(numRanks);
    std::vector<int> recvDispls(numRanks);

    if (rank == 0) {
        h_out_full.resize(numRows);
        int offset = 0;
        for (index_t r = 0; r < numRanks; ++r) {
            index_t rRows = rowsPerRank + (r < remainder ? 1 : 0);
            recvCounts[r] = static_cast<int>(rRows);
            recvDispls[r] = offset;
            offset += rRows;
        }
    }

    MPI_Gatherv(h_out_local.data(), static_cast<int>(localRows), MPI_DOUBLE,
                rank == 0 ? h_out_full.data() : nullptr,
                recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Rank 0 reports timing and performs validation
    if (rank == 0) {
        double elapsedSec = duration.count() / 1000.0;

        printf("Computation time: %ld ms\n", duration.count());

        const double gflops = (2.0 * nItems * iterations) / elapsedSec / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out_full, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out_full.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
