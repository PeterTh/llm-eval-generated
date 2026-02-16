#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// Function: fill
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: initRandomMatrix (same semantics)
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

// CPU SpMV over a row range using OpenMP for intra-node threading
void spmvCpuRange(const double* val, const index_t* cols, const index_t* rowDelimiters,
                  const double* vec, const index_t rowStart, const index_t rowCount, double* out) {
#pragma omp parallel for schedule(static)
    for (index_t ii = 0; ii < rowCount; ++ii) {
        index_t i = rowStart + ii;
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[ii] = t;
    }
}

// Verification (unchanged semantics)
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

#ifdef USE_CUDA
#include <cuda_runtime.h>

static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}

__global__ void spmvKernel(const double* val, const uint32_t* cols, const uint32_t* rowDelimiters,
                           const double* vec, uint32_t rowStart, uint32_t rowCount, double* out) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rowCount) return;
    uint32_t i = rowStart + tid;
    double t = 0.0;
    uint32_t r0 = rowDelimiters[i];
    uint32_t r1 = rowDelimiters[i + 1];
    for (uint32_t j = r0; j < r1; ++j) {
        t += val[j] * vec[cols[j]];
    }
    out[tid] = t;
}

// spmv using CUDA for a local partition
void spmvCuda(const double* h_val, const index_t* h_cols, const index_t* h_rowDelimiters,
              const double* h_vec, const index_t rowStart, const index_t rowCount, double* h_out, const index_t nItems, const index_t dim) {
    // Allocate device arrays (entire arrays for simplicity/performance), then launch kernel for rows [rowStart, rowStart+rowCount)
    double* d_val = nullptr;
    uint32_t* d_cols = nullptr;
    uint32_t* d_rowDel = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    checkCuda(cudaMalloc(&d_val, sizeof(double) * nItems), "cudaMalloc val");
    checkCuda(cudaMalloc(&d_cols, sizeof(uint32_t) * nItems), "cudaMalloc cols");
    checkCuda(cudaMalloc(&d_rowDel, sizeof(uint32_t) * (dim + 1)), "cudaMalloc rowDel");
    checkCuda(cudaMalloc(&d_vec, sizeof(double) * dim), "cudaMalloc vec");
    checkCuda(cudaMalloc(&d_out, sizeof(double) * rowCount), "cudaMalloc out");

    checkCuda(cudaMemcpy(d_val, h_val, sizeof(double) * nItems, cudaMemcpyHostToDevice), "cudaMemcpy val");
    checkCuda(cudaMemcpy(d_cols, h_cols, sizeof(uint32_t) * nItems, cudaMemcpyHostToDevice), "cudaMemcpy cols");
    checkCuda(cudaMemcpy(d_rowDel, h_rowDelimiters, sizeof(uint32_t) * (dim + 1), cudaMemcpyHostToDevice), "cudaMemcpy rowDel");
    checkCuda(cudaMemcpy(d_vec, h_vec, sizeof(double) * dim, cudaMemcpyHostToDevice), "cudaMemcpy vec");

    const int block = 256;
    int grid = (rowCount + block - 1) / block;
    spmvKernel<<<grid, block>>>(d_val, d_cols, d_rowDel, h_vec, rowStart, rowCount, d_out);
    checkCuda(cudaGetLastError(), "kernel launch");

    checkCuda(cudaMemcpy(h_out, d_out, sizeof(double) * rowCount, cudaMemcpyDeviceToHost), "cudaMemcpy outback");

    cudaFree(d_val);
    cudaFree(d_cols);
    cudaFree(d_rowDel);
    cudaFree(d_vec);
    cudaFree(d_out);
}
#endif

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

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResultsFlag = false;

    // Parse command line arguments (only rank 0 prints usage)
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
            printResultsFlag = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) printf("Unknown option: %s\n", argv[i]);
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    const index_t nItems = (numRows * numRows) / sparsity;

    if (worldRank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate on all ranks only the necessary arrays; rank 0 will initialize and broadcast
    std::vector<double> h_val(nItems);
    std::vector<index_t> h_cols(nItems);
    std::vector<index_t> h_rowDel(numRows + 1);
    std::vector<double> h_vec(numRows);

    if (worldRank == 0) {
        srand(8675309);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDel.data(), nItems, numRows);
    }

    // Broadcast arrays to all ranks
    MPI_Bcast(h_rowDel.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_cols.data(), nItems, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_val.data(), nItems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute reference on rank 0 if needed
    std::vector<double> h_reference;
    if (validate && worldRank == 0) {
        h_reference.resize(numRows);
        spmvCpuRange(h_val.data(), h_cols.data(), h_rowDel.data(), h_vec.data(), 0, numRows, h_reference.data());
    }

    // Partition rows among ranks
    index_t base = numRows / worldSize;
    index_t rem = numRows % worldSize;
    index_t rowStart = worldRank * base + std::min<index_t>(worldRank, rem);
    index_t rowCount = base + (worldRank < rem ? 1 : 0);

    std::vector<double> local_out(rowCount);

    // Warm-up and synchronization
    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
#ifdef USE_CUDA
        // Use CUDA for the local partition if available
        spmvCuda(h_val.data(), h_cols.data(), h_rowDel.data(), h_vec.data(), rowStart, rowCount, local_out.data(), nItems, numRows);
#else
        // CPU path with OpenMP
        spmvCpuRange(h_val.data(), h_cols.data(), h_rowDel.data(), h_vec.data(), rowStart, rowCount, local_out.data());
#endif
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Gather results to rank 0
    std::vector<int> recvcounts(worldSize), displs(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        index_t rs = r * base + std::min<index_t>(r, rem);
        index_t rc = base + (r < rem ? 1 : 0);
        recvcounts[r] = static_cast<int>(rc);
        displs[r] = static_cast<int>(rs);
    }

    std::vector<double> h_out;
    if (worldRank == 0) h_out.resize(numRows);

    MPI_Gatherv(local_out.data(), static_cast<int>(rowCount), MPI_DOUBLE,
                h_out.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
        printf("Computation time: %ld ms\n", duration.count());
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResultsFlag) print_results(h_out, "OutputVector");

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
