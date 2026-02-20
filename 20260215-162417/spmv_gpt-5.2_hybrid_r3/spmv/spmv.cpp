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

static inline void cudaCheck(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}
#define CUDA_CHECK(x) cudaCheck((x), __FILE__, __LINE__)

static int getLocalRankFallback(int rank) {
    const char* envs[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "MPI_LOCALRANKID",
    };
    for (const char* e : envs) {
        if (const char* v = std::getenv(e)) {
            return std::atoi(v);
        }
    }
    return rank;
}

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

__global__ void spmvCsrVectorKernel(const index_t numRows, const index_t* __restrict__ rowPtr,
                                   const index_t* __restrict__ colInd,
                                   const double* __restrict__ vals,
                                   const double* __restrict__ x,
                                   double* __restrict__ y) {
    const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if (static_cast<index_t>(warp) >= numRows) return;

    const index_t row = static_cast<index_t>(warp);
    const index_t start = rowPtr[row];
    const index_t end = rowPtr[row + 1];

    double sum = 0.0;
    for (index_t jj = start + static_cast<index_t>(lane); jj < end; jj += 32) {
        sum += vals[jj] * x[colInd[jj]];
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) y[row] = sum;
}

static void printUsage(const char* progName) {
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

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

    // Calculate number of non-zero elements (matches original semantics)
    const index_t nItems = (numRows * numRows) / sparsity;

    // 1D row partitioning across MPI ranks
    const index_t base = numRows / static_cast<index_t>(world);
    const index_t rem = numRows % static_cast<index_t>(world);
    const index_t startRow = static_cast<index_t>(rank) * base + static_cast<index_t>(rank < static_cast<int>(rem) ? rank : rem);
    const index_t localRows = base + (static_cast<index_t>(rank) < rem ? 1u : 0u);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", world);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
    }

    // Root builds the original global CSR to preserve exact semantics, then scatters by row.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);

    if (rank == 0) {
        if (rank == 0) printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);

        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast dense vector x to all ranks (random sparsity implies effectively global access).
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Root computes nnz counts/displacements per rank for CSR scatter
    std::vector<int> nnzCounts, nnzDispls, rowCounts, rowDispls, outCounts, outDispls;
    int localNnzInt = 0;

    if (rank == 0) {
        nnzCounts.resize(world);
        nnzDispls.resize(world);
        rowCounts.resize(world);
        rowDispls.resize(world);
        outCounts.resize(world);
        outDispls.resize(world);

        for (int r = 0; r < world; ++r) {
            const index_t rStart = static_cast<index_t>(r) * base + static_cast<index_t>(r < static_cast<int>(rem) ? r : rem);
            const index_t rRows = base + (static_cast<index_t>(r) < rem ? 1u : 0u);
            const index_t nnzStart = h_rowDelimiters[rStart];
            const index_t nnzEnd = h_rowDelimiters[rStart + rRows];

            nnzDispls[r] = static_cast<int>(nnzStart);
            nnzCounts[r] = static_cast<int>(nnzEnd - nnzStart);
            rowDispls[r] = static_cast<int>(rStart);
            rowCounts[r] = static_cast<int>(rRows + 1);
            outDispls[r] = static_cast<int>(rStart);
            outCounts[r] = static_cast<int>(rRows);
        }
    }

    MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_INT, &localNnzInt, 1, MPI_INT, 0,
                MPI_COMM_WORLD);
    const index_t localNnz = static_cast<index_t>(localNnzInt);

    std::vector<double> h_val_local(localNnz);
    std::vector<index_t> h_cols_local(localNnz);
    std::vector<index_t> h_rowPtr_local(localRows + 1);
    std::vector<double> h_out_local(localRows);

    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr, rank == 0 ? rowCounts.data() : nullptr,
                 rank == 0 ? rowDispls.data() : nullptr, MPI_UINT32_T, h_rowPtr_local.data(),
                 static_cast<int>(localRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t nnzOffset = h_rowPtr_local[0];
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < localRows + 1; ++i) {
        h_rowPtr_local[i] -= nnzOffset;
    }

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr, MPI_DOUBLE, h_val_local.data(),
                 static_cast<int>(localNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr, MPI_UINT32_T, h_cols_local.data(),
                 static_cast<int>(localNnz), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Select GPU
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    const int localRank = getLocalRankFallback(rank);
    const int dev = localRank % devCount;
    CUDA_CHECK(cudaSetDevice(dev));

    // Device allocations
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowPtr = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_rowPtr, sizeof(index_t) * static_cast<size_t>(localRows + 1)));
    CUDA_CHECK(cudaMalloc(&d_vec, sizeof(double) * static_cast<size_t>(numRows)));
    if (localNnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, sizeof(double) * static_cast<size_t>(localNnz)));
        CUDA_CHECK(cudaMalloc(&d_cols, sizeof(index_t) * static_cast<size_t>(localNnz)));
    }
    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * static_cast<size_t>(localRows)));
    }

    if (localNnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_val, h_val_local.data(), sizeof(double) * static_cast<size_t>(localNnz),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, h_cols_local.data(), sizeof(index_t) * static_cast<size_t>(localNnz),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rowPtr, h_rowPtr_local.data(), sizeof(index_t) * static_cast<size_t>(localRows + 1),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), sizeof(double) * static_cast<size_t>(numRows), cudaMemcpyHostToDevice));

    // Time the GPU SpMV over iterations; use global max over ranks for scalability reporting.
    if (rank == 0) printf("Computing SpMV...\n");

    const int threads = 256;
    float localMsF = 0.0f;

    MPI_Barrier(MPI_COMM_WORLD);
    if (localRows > 0) {
        cudaEvent_t evStart{}, evStop{};
        CUDA_CHECK(cudaEventCreate(&evStart));
        CUDA_CHECK(cudaEventCreate(&evStop));

        const int warpsPerBlock = threads / 32;
        const int blocks = static_cast<int>((localRows + static_cast<index_t>(warpsPerBlock) - 1) / static_cast<index_t>(warpsPerBlock));

        CUDA_CHECK(cudaEventRecord(evStart));
        for (index_t iter = 0; iter < iterations; ++iter) {
            spmvCsrVectorKernel<<<blocks, threads>>>(localRows, d_rowPtr, d_cols, d_val, d_vec, d_out);
        }
        CUDA_CHECK(cudaEventRecord(evStop));
        CUDA_CHECK(cudaEventSynchronize(evStop));

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&localMsF, evStart, evStop));
        CUDA_CHECK(cudaEventDestroy(evStart));
        CUDA_CHECK(cudaEventDestroy(evStop));
    }

    const double localMs = static_cast<double>(localMsF);
    double maxMs = 0.0;
    MPI_Allreduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // Global nnz for GFLOPS
    const uint64_t localNnz64 = static_cast<uint64_t>(localNnz);
    uint64_t globalNnz64 = 0;
    MPI_Allreduce(&localNnz64, &globalNnz64, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(h_out_local.data(), d_out, sizeof(double) * static_cast<size_t>(localRows),
                              cudaMemcpyDeviceToHost));
    }

    std::vector<double> h_out;
    if (rank == 0) h_out.resize(numRows);

    MPI_Gatherv(h_out_local.data(), static_cast<int>(localRows), MPI_DOUBLE, rank == 0 ? h_out.data() : nullptr,
                rank == 0 ? outCounts.data() : nullptr, rank == 0 ? outDispls.data() : nullptr, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(llround(maxMs)));
        const double avgTime = maxMs / static_cast<double>(iterations);
        const double gflops = (2.0 * static_cast<double>(globalNnz64) * static_cast<double>(iterations)) /
                              (maxMs / 1000.0) / 1e9;
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Computing reference solution...\n");
            std::vector<double> h_reference(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows, h_reference.data());

            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");

            if (d_val) CUDA_CHECK(cudaFree(d_val));
            if (d_cols) CUDA_CHECK(cudaFree(d_cols));
            if (d_rowPtr) CUDA_CHECK(cudaFree(d_rowPtr));
            if (d_vec) CUDA_CHECK(cudaFree(d_vec));
            if (d_out) CUDA_CHECK(cudaFree(d_out));

            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    if (d_val) CUDA_CHECK(cudaFree(d_val));
    if (d_cols) CUDA_CHECK(cudaFree(d_cols));
    if (d_rowPtr) CUDA_CHECK(cudaFree(d_rowPtr));
    if (d_vec) CUDA_CHECK(cudaFree(d_vec));
    if (d_out) CUDA_CHECK(cudaFree(d_out));

    MPI_Finalize();
    return 0;
}
