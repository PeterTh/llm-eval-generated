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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel: each thread computes one row of the local partition
__global__ void spmvKernel(const double* __restrict__ val,
                           const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec,
                           double* __restrict__ out,
                           const index_t localRows,
                           const index_t rowOffset) {
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < localRows) {
        index_t globalRow = i + rowOffset;
        double t = 0.0;
        index_t rowStart = rowDelimiters[globalRow];
        index_t rowEnd = rowDelimiters[globalRow + 1];
        for (index_t j = rowStart; j < rowEnd; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Function: fill
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: initRandomMatrix
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
// Function: spmvCpu (reference for validation)
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
// Function: verifyResults
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                return false;
            }
        } else {
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
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Assign GPU based on local rank
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount > 0) {
        // Use local rank for multi-GPU nodes
        int localRank = 0;
        const char* localRankEnv = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
        if (!localRankEnv) localRankEnv = getenv("MV2_COMM_WORLD_LOCAL_RANK");
        if (!localRankEnv) localRankEnv = getenv("LOCAL_RANK");
        if (localRankEnv) localRank = atoi(localRankEnv);
        CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    }

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
            if (rank == 0) printUsage(argv[0]);
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

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Parallelization: MPI + OpenMP + CUDA (hybrid)\n");
        printf("MPI ranks: %d\n", nprocs);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // All ranks allocate full CSR structure (needed for val/cols indexing)
    std::vector<double> h_val(nItems);
    std::vector<index_t> h_cols(nItems);
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out(numRows, 0.0);

    // Rank 0 initializes data, then broadcasts
    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    MPI_Bcast(h_val.data(), nItems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_cols.data(), nItems * sizeof(index_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), (numRows + 1) * sizeof(index_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute row distribution across MPI ranks
    index_t localRows = numRows / nprocs;
    index_t rowOffset = rank * localRows;
    // Last rank takes remaining rows
    if (rank == nprocs - 1) {
        localRows = numRows - rowOffset;
    }

    // For validation, compute reference solution on rank 0
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Allocate device memory for the full CSR + vector (each rank holds full data)
    double *d_val, *d_vec, *d_out;
    index_t *d_cols, *d_rowDelimiters;

    CUDA_CHECK(cudaMalloc(&d_val, nItems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, nItems * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (numRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, localRows * sizeof(double)));

    // Copy data to device (these don't change between iterations)
    CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), nItems * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), nItems * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(), (numRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    int gridSize = (localRows + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    // Prepare gather info
    std::vector<int> recvCounts(nprocs);
    std::vector<int> displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        index_t rRows = numRows / nprocs;
        index_t rOffset = r * rRows;
        if (r == nprocs - 1) rRows = numRows - rOffset;
        recvCounts[r] = static_cast<int>(rRows);
        displs[r] = static_cast<int>(rOffset);
    }

    std::vector<double> h_localOut(localRows);

    MPI_Barrier(MPI_COMM_WORLD);

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvKernel<<<gridSize, CUDA_BLOCK_SIZE>>>(d_val, d_cols, d_rowDelimiters,
                                                   d_vec, d_out, localRows, rowOffset);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy local results back to host
        CUDA_CHECK(cudaMemcpy(h_localOut.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost));

        // Gather results from all ranks to rank 0
        MPI_Gatherv(h_localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                     h_out.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long localDurationMs = static_cast<long long>(duration.count());
    long long globalDurationMs = 0;
    MPI_Reduce(&localDurationMs, &globalDurationMs, 1, MPI_LONG_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);

    // Free device memory
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    if (rank == 0) {
        printf("Computation time: %lld ms\n", globalDurationMs);

        const double gflops = (2.0 * nItems * iterations) / (globalDurationMs / 1000.0) / 1e9;
        const double avgTime = globalDurationMs / static_cast<double>(iterations);

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
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
