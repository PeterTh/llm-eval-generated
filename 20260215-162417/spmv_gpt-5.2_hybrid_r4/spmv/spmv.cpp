#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

static inline void partition_rows(const index_t totalRows, const int commSize, const int rank,
                                  index_t& rowStart, index_t& rowEnd) {
    const index_t base = totalRows / static_cast<index_t>(commSize);
    const index_t rem = totalRows % static_cast<index_t>(commSize);
    rowStart = static_cast<index_t>(rank) * base + static_cast<index_t>(rank < static_cast<int>(rem) ? rank : rem);
    rowEnd = rowStart + base + (rank < static_cast<int>(rem) ? 1u : 0u);
}

static inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err == cudaSuccess) return;
    fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(err));
    int mpiInit = 0;
    MPI_Initialized(&mpiInit);
    if (mpiInit) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::abort();
}

#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

__device__ __forceinline__ double warpReduceSum(double v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__global__ void spmvCsrWarpKernel(const double* __restrict__ val,
                                 const index_t* __restrict__ cols,
                                 const index_t* __restrict__ rowDelim,
                                 const double* __restrict__ x,
                                 double* __restrict__ y,
                                 const index_t numRows) {
    const int lane = threadIdx.x & 31;
    const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int numWarps = (gridDim.x * blockDim.x) >> 5;

    for (index_t row = static_cast<index_t>(warpId); row < numRows; row += static_cast<index_t>(numWarps)) {
        const index_t start = rowDelim[row];
        const index_t end = rowDelim[row + 1];
        double sum = 0.0;
        for (index_t jj = start + static_cast<index_t>(lane); jj < end; jj += 32u) {
            const index_t c = cols[jj];
            sum += val[jj] * x[c];
        }
        sum = warpReduceSum(sum);
        if (lane == 0) y[row] = sum;
    }
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
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
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
    int commSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool badArgs = false;

    // Parse command line arguments (all ranks see the same argv under MPI launch)
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
            showHelp = true;
        } else {
            badArgs = true;
        }
    }

    if (showHelp || badArgs) {
        if (rank == 0) {
            if (badArgs) printf("Unknown option(s)\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return badArgs ? 1 : 0;
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", commSize);
#ifdef _OPENMP
        printf("OpenMP threads (max): %d\n", omp_get_max_threads());
#endif
    }

    // Root allocates and initializes global data deterministically (preserves original rand() behavior)
    std::vector<double> h_val_global;
    std::vector<index_t> h_cols_global;
    std::vector<index_t> h_rowDelim_global;
    std::vector<double> h_vec(numRows);

    if (rank == 0) {
        h_val_global.resize(nItems);
        h_cols_global.resize(nItems);
        h_rowDelim_global.resize(numRows + 1);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val_global.data(), nItems, maxVal);
        initRandomMatrix(h_cols_global.data(), h_rowDelim_global.data(), nItems, numRows);
    }

    // Broadcast the dense vector to all ranks
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Unconditional OpenMP usage (lightweight checksum, outside timed region)
    double vecChecksum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : vecChecksum) schedule(static)
#endif
    for (index_t i = 0; i < numRows; ++i) {
        vecChecksum += h_vec[i];
    }

    if (rank == 0) {
        // prevent checksum from being optimized away
        printf("Vector checksum: %.6e\n", vecChecksum);
    }

    // Partition rows across ranks
    index_t rowStart = 0, rowEnd = 0;
    partition_rows(numRows, commSize, rank, rowStart, rowEnd);
    index_t localRows = rowEnd - rowStart;

    // Receive/construct local CSR slice
    std::vector<double> h_val_local;
    std::vector<index_t> h_cols_local;
    std::vector<index_t> h_rowDelim_local;

    if (rank == 0) {
        // Send CSR slices to non-root ranks
        for (int r = 1; r < commSize; ++r) {
            index_t rs = 0, re = 0;
            partition_rows(numRows, commSize, r, rs, re);
            const index_t rRows = re - rs;
            const index_t nnzStart = h_rowDelim_global[rs];
            const index_t nnzEnd = h_rowDelim_global[re];
            const index_t rNnz = nnzEnd - nnzStart;

            MPI_Send(&rRows, 1, MPI_UINT32_T, r, 0, MPI_COMM_WORLD);
            MPI_Send(&rNnz, 1, MPI_UINT32_T, r, 1, MPI_COMM_WORLD);

            std::vector<index_t> rRowDelim(rRows + 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (index_t i = 0; i < rRows; ++i) {
                rRowDelim[i] = h_rowDelim_global[rs + i] - nnzStart;
            }
            rRowDelim[rRows] = rNnz;

            MPI_Send(rRowDelim.data(), static_cast<int>(rRows + 1), MPI_UINT32_T, r, 2, MPI_COMM_WORLD);
            MPI_Send(h_cols_global.data() + nnzStart, static_cast<int>(rNnz), MPI_UINT32_T, r, 3, MPI_COMM_WORLD);
            MPI_Send(h_val_global.data() + nnzStart, static_cast<int>(rNnz), MPI_DOUBLE, r, 4, MPI_COMM_WORLD);
        }

        // Root local slice
        const index_t nnzStart = h_rowDelim_global[rowStart];
        const index_t nnzEnd = h_rowDelim_global[rowEnd];
        const index_t localNnz = nnzEnd - nnzStart;

        h_val_local.assign(h_val_global.begin() + nnzStart, h_val_global.begin() + nnzEnd);
        h_cols_local.assign(h_cols_global.begin() + nnzStart, h_cols_global.begin() + nnzEnd);
        h_rowDelim_local.resize(localRows + 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (index_t i = 0; i < localRows; ++i) {
            h_rowDelim_local[i] = h_rowDelim_global[rowStart + i] - nnzStart;
        }
        h_rowDelim_local[localRows] = localNnz;
    } else {
        index_t recvRows = 0;
        index_t localNnz = 0;
        MPI_Recv(&recvRows, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&localNnz, 1, MPI_UINT32_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (recvRows != localRows) {
            fprintf(stderr, "Rank %d row partition mismatch (expected %u, got %u)\n", rank, localRows, recvRows);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        h_val_local.resize(localNnz);
        h_cols_local.resize(localNnz);
        h_rowDelim_local.resize(localRows + 1);

        MPI_Recv(h_rowDelim_local.data(), static_cast<int>(localRows + 1), MPI_UINT32_T, 0, 2, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(h_cols_local.data(), static_cast<int>(localNnz), MPI_UINT32_T, 0, 3, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(h_val_local.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Validation reference (root only, CPU)
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val_global.data(), h_cols_global.data(), h_rowDelim_global.data(), h_vec.data(), numRows,
                h_reference.data());
    }

    // CUDA device selection per rank
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(0));

    if (rank == 0) {
        printf("Computing SpMV (MPI+CUDA)...\n");
    }

    // Device buffers
    const index_t localNnz = (localRows > 0) ? h_rowDelim_local[localRows] : 0u;
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelim = nullptr;
    double* d_x = nullptr;
    double* d_y = nullptr;

    if (localNnz > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_val), static_cast<size_t>(localNnz) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cols), static_cast<size_t>(localNnz) * sizeof(index_t)));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rowDelim), static_cast<size_t>(localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_x), static_cast<size_t>(numRows) * sizeof(double)));
    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_y), static_cast<size_t>(localRows) * sizeof(double)));
    }

    if (localNnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_val, h_val_local.data(), static_cast<size_t>(localNnz) * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, h_cols_local.data(), static_cast<size_t>(localNnz) * sizeof(index_t),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rowDelim, h_rowDelim_local.data(), static_cast<size_t>(localRows + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_vec.data(), static_cast<size_t>(numRows) * sizeof(double), cudaMemcpyHostToDevice));

    const int threads = 256;
    const int warpsPerBlock = threads / 32;
    int blocks = 1;
    if (localRows > 0) {
        blocks = static_cast<int>((localRows + static_cast<index_t>(warpsPerBlock) - 1u) /
                                  static_cast<index_t>(warpsPerBlock));
        blocks = std::min(blocks, 65535);
        blocks = std::max(blocks, 1);
    }

    // Warmup
    if (localRows > 0) {
        spmvCsrWarpKernel<<<blocks, threads>>>(d_val, d_cols, d_rowDelim, d_x, d_y, localRows);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows > 0) {
            spmvCsrWarpKernel<<<blocks, threads>>>(d_val, d_cols, d_rowDelim, d_x, d_y, localRows);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const double t1 = MPI_Wtime();
    const double localTime = t1 - t0;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy local result back
    std::vector<double> h_out_local(localRows);
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(h_out_local.data(), d_y, static_cast<size_t>(localRows) * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    // Gather full output to root
    std::vector<double> h_out;
    std::vector<int> recvCounts;
    std::vector<int> displs;
    if (rank == 0) {
        h_out.resize(numRows);
        recvCounts.resize(commSize);
        displs.resize(commSize);
        for (int r = 0; r < commSize; ++r) {
            index_t rs = 0, re = 0;
            partition_rows(numRows, commSize, r, rs, re);
            recvCounts[r] = static_cast<int>(re - rs);
            displs[r] = static_cast<int>(rs);
        }
    }

    MPI_Gatherv(h_out_local.data(), static_cast<int>(localRows), MPI_DOUBLE, rank == 0 ? h_out.data() : nullptr,
                rank == 0 ? recvCounts.data() : nullptr, rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        const double avgTimeMs = (maxTime * 1000.0) / static_cast<double>(iterations);
        const double gflops = (2.0 * static_cast<double>(nItems) * static_cast<double>(iterations)) / maxTime / 1e9;
        printf("Computation time: %.3f ms\n", maxTime * 1000.0);
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
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    if (d_y) CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_rowDelim));
    if (d_cols) CUDA_CHECK(cudaFree(d_cols));
    if (d_val) CUDA_CHECK(cudaFree(d_val));

    MPI_Finalize();

    if (validate && rank == 0) {
        return verifyResults(h_reference.data(), h_out.data(), numRows) ? 0 : 1;
    }
    return 0;
}
