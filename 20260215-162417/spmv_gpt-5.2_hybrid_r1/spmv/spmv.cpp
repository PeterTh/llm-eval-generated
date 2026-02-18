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

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t _e = (call);                                                          \
        if (_e != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                \
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

__global__ void spmv_csr_vector_kernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                                      const index_t* __restrict__ rowPtr, const double* __restrict__ x,
                                      double* __restrict__ y, index_t numRowsLocal) {
    constexpr int WARP = 32;
    const int lane = threadIdx.x & (WARP - 1);
    const int warpInBlock = threadIdx.x / WARP;
    const int warpsPerBlock = blockDim.x / WARP;
    const index_t row = static_cast<index_t>(blockIdx.x * warpsPerBlock + warpInBlock);

    if (row >= numRowsLocal) return;

    const index_t rowStart = rowPtr[row];
    const index_t rowEnd = rowPtr[row + 1];

    double sum = 0.0;
    for (index_t jj = rowStart + static_cast<index_t>(lane); jj < rowEnd; jj += WARP) {
        sum += val[jj] * x[cols[jj]];
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    if (lane == 0) {
        y[row] = sum;
    }
}

static void pickCudaDeviceForRank(int rank) {
    int devCount = 0;
    cudaError_t e = cudaGetDeviceCount(&devCount);
    if (e != cudaSuccess || devCount <= 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % devCount));
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
    int validate_i = 0;
    int printResults_i = 0;

    int exitNow = 0;
    int exitCode = 0;

    if (rank == 0) {
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
                validate_i = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults_i = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                exitNow = 1;
                exitCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitNow = 1;
                exitCode = 1;
                break;
            }
        }
    }

    MPI_Bcast(&exitNow, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exitNow) {
        MPI_Finalize();
        return (rank == 0) ? exitCode : 0;
    }

    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_i, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const bool validate = (validate_i != 0);
    const bool printResults = (printResults_i != 0);

    // Calculate number of non-zero elements (global)
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads (rank 0): %d\n", omp_get_max_threads());
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    pickCudaDeviceForRank(rank);

    // Global data (root only)
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out;

    // Partition rows across ranks (simple block partitioning)
    const index_t rowStart = static_cast<index_t>((static_cast<uint64_t>(rank) * numRows) / size);
    const index_t rowEnd = static_cast<index_t>((static_cast<uint64_t>(rank + 1) * numRows) / size);
    const index_t localRows = rowEnd - rowStart;

    std::vector<int> rowCounts;
    std::vector<int> rowDispls;

    // Per-rank nnz metadata (root)
    std::vector<int> nnzCounts;
    std::vector<int> nnzDispls;

    // Flattened rowPtr segments for MPI_Scatterv (root)
    std::vector<int> rowPtrCounts;
    std::vector<int> rowPtrDispls;
    std::vector<index_t> flatRowPtr;

    int localNnz_i = 0;

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_out.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        rowCounts.resize(size);
        rowDispls.resize(size);
        nnzCounts.resize(size);
        nnzDispls.resize(size);
        rowPtrCounts.resize(size);
        rowPtrDispls.resize(size);

        int rpOff = 0;
        for (int r = 0; r < size; ++r) {
            const index_t rs = static_cast<index_t>((static_cast<uint64_t>(r) * numRows) / size);
            const index_t re = static_cast<index_t>((static_cast<uint64_t>(r + 1) * numRows) / size);
            const index_t lr = re - rs;

            const index_t nnzStart = h_rowDelimiters[rs];
            const index_t nnzEnd = h_rowDelimiters[re];
            const index_t lnnz = nnzEnd - nnzStart;

            rowCounts[r] = static_cast<int>(lr);
            rowDispls[r] = static_cast<int>(rs);

            nnzCounts[r] = static_cast<int>(lnnz);
            nnzDispls[r] = static_cast<int>(nnzStart);

            rowPtrCounts[r] = static_cast<int>(lr + 1);
            rowPtrDispls[r] = rpOff;
            rpOff += rowPtrCounts[r];
        }

        flatRowPtr.resize(static_cast<size_t>(rpOff));
#pragma omp parallel for schedule(static)
        for (int r = 0; r < size; ++r) {
            const index_t rs = static_cast<index_t>((static_cast<uint64_t>(r) * numRows) / size);
            const index_t re = static_cast<index_t>((static_cast<uint64_t>(r + 1) * numRows) / size);
            const index_t lr = re - rs;
            const index_t nnzStart = h_rowDelimiters[rs];

            const int off = rowPtrDispls[r];
            for (index_t i = 0; i <= lr; ++i) {
                flatRowPtr[static_cast<size_t>(off) + i] = h_rowDelimiters[rs + i] - nnzStart;
            }
        }
    }

    // Broadcast dense vector (all ranks need it)
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter local nnz count first
    MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_INT, &localNnz_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const index_t localNnz = static_cast<index_t>(localNnz_i);

    // Allocate local CSR
    std::vector<double> l_val(localNnz);
    std::vector<index_t> l_cols(localNnz);
    std::vector<index_t> l_rowPtr(localRows + 1);

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr, MPI_DOUBLE, l_val.data(), localNnz_i, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr, MPI_UINT32_T, l_cols.data(), localNnz_i, MPI_UINT32_T, 0,
                 MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? flatRowPtr.data() : nullptr, rank == 0 ? rowPtrCounts.data() : nullptr,
                 rank == 0 ? rowPtrDispls.data() : nullptr, MPI_UINT32_T, l_rowPtr.data(),
                 static_cast<int>(localRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // GPU buffers
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowPtr = nullptr;
    double* d_x = nullptr;
    double* d_y = nullptr;

    if (localNnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, sizeof(double) * static_cast<size_t>(localNnz)));
        CUDA_CHECK(cudaMalloc(&d_cols, sizeof(index_t) * static_cast<size_t>(localNnz)));
        CUDA_CHECK(cudaMemcpy(d_val, l_val.data(), sizeof(double) * static_cast<size_t>(localNnz), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, l_cols.data(), sizeof(index_t) * static_cast<size_t>(localNnz), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&d_rowPtr, sizeof(index_t) * static_cast<size_t>(localRows + 1)));
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(double) * static_cast<size_t>(numRows)));
    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(&d_y, sizeof(double) * static_cast<size_t>(localRows)));
    }

    CUDA_CHECK(cudaMemcpy(d_rowPtr, l_rowPtr.data(), sizeof(index_t) * static_cast<size_t>(localRows + 1),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_vec.data(), sizeof(double) * static_cast<size_t>(numRows), cudaMemcpyHostToDevice));

    // Optional reference (root only, full problem)
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows, h_reference.data());
    }

    if (rank == 0) {
        printf("Computing SpMV (MPI + OpenMP + CUDA)...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Time kernel loop (each rank), report max rank time for scalability
    const double t0 = MPI_Wtime();
    if (localRows > 0 && iterations > 0) {
        const int threadsPerBlock = 256;
        const int warpsPerBlock = threadsPerBlock / 32;
        const int blocks = static_cast<int>((localRows + static_cast<index_t>(warpsPerBlock) - 1) /
                                            static_cast<index_t>(warpsPerBlock));
        for (index_t iter = 0; iter < iterations; ++iter) {
            spmv_csr_vector_kernel<<<blocks, threadsPerBlock>>>(d_val, d_cols, d_rowPtr, d_x, d_y, localRows);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    const double t1 = MPI_Wtime();

    const double localTime = t1 - t0;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather result back to root
    std::vector<double> l_y(localRows);
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(l_y.data(), d_y, sizeof(double) * static_cast<size_t>(localRows), cudaMemcpyDeviceToHost));
    }

    if (rank == 0) {
        if (h_out.empty()) h_out.resize(numRows);
    }

    MPI_Gatherv(l_y.data(), static_cast<int>(localRows), MPI_DOUBLE, rank == 0 ? h_out.data() : nullptr,
                rank == 0 ? rowCounts.data() : nullptr, rank == 0 ? rowDispls.data() : nullptr, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    int finalExitCode = 0;
    if (rank == 0) {
        const double denom = (maxTime > 0.0) ? maxTime : 1e-30;
        printf("Computation time: %.3f ms\n", denom * 1000.0);
        const double gflops = (2.0 * static_cast<double>(nItems) * static_cast<double>(iterations)) / denom / 1e9;
        const double avgTimeMs = (iterations > 0) ? ((denom * 1000.0) / static_cast<double>(iterations)) : 0.0;
        printf("Average time per iteration: %.3f ms\n", avgTimeMs);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
                finalExitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                finalExitCode = 1;
            }
        }
    }

    if (d_val) CUDA_CHECK(cudaFree(d_val));
    if (d_cols) CUDA_CHECK(cudaFree(d_cols));
    if (d_rowPtr) CUDA_CHECK(cudaFree(d_rowPtr));
    if (d_x) CUDA_CHECK(cudaFree(d_x));
    if (d_y) CUDA_CHECK(cudaFree(d_y));

    MPI_Finalize();
    return (rank == 0) ? finalExitCode : 0;
}
