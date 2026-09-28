// Hybrid MPI + OpenMP + CUDA implementation of the SpMV benchmark.
//
// Parallelization strategy:
//   * MPI   distributes the rows of the CSR matrix across ranks. The partition is
//           balanced by number of non-zeros (not by number of rows) so that every
//           rank receives roughly the same amount of work. Each rank only stores
//           its own slice of the matrix.
//   * CUDA  performs the actual SpMV. Every rank drives one GPU (selected by the
//           rank's index within its compute node) and runs a CSR "vector" kernel
//           with a sub-warp group per row; the group size is chosen at runtime
//           from the average number of non-zeros per row.
//   * OpenMP parallelizes all remaining host-side work (matrix rebasing, the
//           reference solution and result verification) over the CPU cores.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// CUDA launch configuration
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_BLOCKS = 65535;

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        const cudaError_t err__ = (call);                                                    \
        if (err__ != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, \
                    __LINE__);                                                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                    \
        }                                                                                    \
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
//   (OpenMP parallel over the rows; each row is summed in the original order)
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

// ****************************************************************************
// Kernel: spmvKernel
//
// Purpose:
//   CSR SpMV on the GPU. A group of GROUP threads (a power-of-two subset of a
//   warp) cooperates on one row: the group strides over the row's non-zeros for
//   fully coalesced accesses and reduces the partial products with warp shuffles.
//   The trip count of the outer loop is uniform across the grid so that every
//   thread of a warp participates in the shuffles.
// ****************************************************************************
template <int GROUP>
__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec, const index_t numRows,
                           double* __restrict__ out) {
    const index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = static_cast<int>(tid & (GROUP - 1));
    const index_t numGroups = (gridDim.x * blockDim.x) / GROUP;
    const index_t firstRow = tid / GROUP;
    const index_t numIter = (numRows + numGroups - 1) / numGroups;

    for (index_t it = 0; it < numIter; ++it) {
        const index_t row = firstRow + it * numGroups;
        double sum = 0.0;
        if (row < numRows) {
            const index_t end = rowDelimiters[row + 1];
            for (index_t j = rowDelimiters[row] + lane; j < end; j += GROUP) {
                sum += val[j] * __ldg(&vec[cols[j]]);
            }
        }
#pragma unroll
        for (int off = GROUP / 2; off > 0; off >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, off, GROUP);
        }
        if (lane == 0 && row < numRows) {
            out[row] = sum;
        }
    }
}

// Launches the kernel variant whose per-row thread group best matches the
// average number of non-zeros per row of this rank's slice.
static void launchSpmv(const double* d_val, const index_t* d_cols, const index_t* d_rowDelimiters,
                       const double* d_vec, const index_t numRows, double* d_out,
                       const int group, cudaStream_t stream) {
    if (numRows == 0) {
        return;
    }
    const long long needed =
        (static_cast<long long>(numRows) * group + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int blocks = static_cast<int>(needed > MAX_BLOCKS ? MAX_BLOCKS : needed);

    switch (group) {
        case 2:
            spmvKernel<2><<<blocks, BLOCK_SIZE, 0, stream>>>(d_val, d_cols, d_rowDelimiters, d_vec,
                                                             numRows, d_out);
            break;
        case 4:
            spmvKernel<4><<<blocks, BLOCK_SIZE, 0, stream>>>(d_val, d_cols, d_rowDelimiters, d_vec,
                                                             numRows, d_out);
            break;
        case 8:
            spmvKernel<8><<<blocks, BLOCK_SIZE, 0, stream>>>(d_val, d_cols, d_rowDelimiters, d_vec,
                                                             numRows, d_out);
            break;
        case 16:
            spmvKernel<16><<<blocks, BLOCK_SIZE, 0, stream>>>(d_val, d_cols, d_rowDelimiters, d_vec,
                                                              numRows, d_out);
            break;
        default:
            spmvKernel<32><<<blocks, BLOCK_SIZE, 0, stream>>>(d_val, d_cols, d_rowDelimiters, d_vec,
                                                              numRows, d_out);
            break;
    }
}

static int chooseGroupSize(const double avgNnzPerRow) {
    if (avgNnzPerRow <= 3.0) return 2;
    if (avgNnzPerRow <= 6.0) return 4;
    if (avgNnzPerRow <= 12.0) return 8;
    if (avgNnzPerRow <= 24.0) return 16;
    return 32;
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
    // Find the first mismatching index (if any) in parallel; report it like the
    // serial version would.
    index_t bad = size;
#pragma omp parallel for schedule(static) reduction(min : bad)
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                bad = std::min(bad, i);
            }
        } else {
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                bad = std::min(bad, i);
            }
        }
    }

    if (bad == size) {
        return true;
    }

    const double ref = reference[bad];
    const double res = result[bad];
    if (std::abs(ref) < 1e-10) {
        printf("Validation failed at index %u: reference %.10e, got %.10e\n", bad, ref, res);
    } else {
        printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
               bad, ref, res, std::abs((res - ref) / ref));
    }
    return false;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", numRanks, omp_get_max_threads());
    }

    // ------------------------------------------------------------------
    // Bind each rank to one GPU (round robin over the devices of its node)
    // ------------------------------------------------------------------
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr));  // establish the context up front

    // ------------------------------------------------------------------
    // Data generation (kept sequential on rank 0 to reproduce the exact
    // random sequence of the original benchmark) and distribution.
    // ------------------------------------------------------------------
    std::vector<double> h_val;            // Non-zero values (rank 0: full matrix)
    std::vector<index_t> h_cols;          // Column indices  (rank 0: full matrix)
    std::vector<index_t> h_rowDelimiters; // Row delimiters  (rank 0: full matrix)
    std::vector<double> h_vec(numRows);   // Dense vector (replicated on all ranks)
    std::vector<double> h_out;            // Output vector (rank 0)

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_out.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Row partition, balanced by number of non-zeros.
    std::vector<index_t> rowStart(numRanks + 1);
    std::vector<index_t> nnzStart(numRanks + 1);
    if (rank == 0) {
        rowStart[0] = 0;
        for (int r = 1; r < numRanks; ++r) {
            const index_t target =
                static_cast<index_t>((static_cast<uint64_t>(nItems) * r) / numRanks);
            // First row whose starting offset reaches the target nnz count
            index_t lo = rowStart[r - 1];
            index_t hi = numRows;
            while (lo < hi) {
                const index_t mid = lo + (hi - lo) / 2;
                if (h_rowDelimiters[mid] < target) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            rowStart[r] = lo;
        }
        rowStart[numRanks] = numRows;
        for (int r = 0; r <= numRanks; ++r) {
            nnzStart[r] = h_rowDelimiters[rowStart[r]];
        }
    }
    MPI_Bcast(rowStart.data(), numRanks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzStart.data(), numRanks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t myFirstRow = rowStart[rank];
    const index_t myRows = rowStart[rank + 1] - myFirstRow;
    const index_t myNnz = nnzStart[rank + 1] - nnzStart[rank];

    std::vector<int> rowCounts(numRanks), rowDispls(numRanks);
    std::vector<int> nnzCounts(numRanks), nnzDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        rowCounts[r] = static_cast<int>(rowStart[r + 1] - rowStart[r]);
        rowDispls[r] = static_cast<int>(rowStart[r]);
        nnzCounts[r] = static_cast<int>(nnzStart[r + 1] - nnzStart[r]);
        nnzDispls[r] = static_cast<int>(nnzStart[r]);
    }

    // Local CSR slice
    std::vector<double> l_val(myNnz);
    std::vector<index_t> l_cols(myNnz);
    std::vector<index_t> l_rowDelimiters(myRows + 1);
    std::vector<double> l_out(myRows);

    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr, rowCounts.data(), rowDispls.data(),
                 MPI_UINT32_T, l_rowDelimiters.data(), static_cast<int>(myRows), MPI_UINT32_T, 0,
                 MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 l_val.data(), static_cast<int>(myNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                 MPI_UINT32_T, l_cols.data(), static_cast<int>(myNnz), MPI_UINT32_T, 0,
                 MPI_COMM_WORLD);

    // Rebase the local row delimiters to the local value array
    const index_t base = nnzStart[rank];
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < myRows; ++i) {
        l_rowDelimiters[i] -= base;
    }
    l_rowDelimiters[myRows] = myNnz;

    // ------------------------------------------------------------------
    // Device buffers
    // ------------------------------------------------------------------
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_val, (myNnz ? myNnz : 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, (myNnz ? myNnz : 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (myRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, (numRows ? numRows : 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, (myRows ? myRows : 1) * sizeof(double)));

    if (myNnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_val, l_val.data(), myNnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(d_cols, l_cols.data(), myNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, l_rowDelimiters.data(), (myRows + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    const int group =
        chooseGroupSize(myRows > 0 ? static_cast<double>(myNnz) / static_cast<double>(myRows) : 1.0);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // ------------------------------------------------------------------
    // Perform SpMV computation
    // ------------------------------------------------------------------
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    // Warm up the kernel / JIT path outside of the measured region
    launchSpmv(d_val, d_cols, d_rowDelimiters, d_vec, myRows, d_out, group, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpmv(d_val, d_cols, d_rowDelimiters, d_vec, myRows, d_out, group, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Collect the distributed result on rank 0
    if (myRows > 0) {
        CUDA_CHECK(
            cudaMemcpy(l_out.data(), d_out, myRows * sizeof(double), cudaMemcpyDeviceToHost));
    }
    MPI_Gatherv(l_out.data(), static_cast<int>(myRows), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr, rowCounts.data(), rowDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &elapsedMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsedMs));

        // Calculate performance metrics
        const double seconds = elapsedMs / 1000.0;
        const double gflops = (2.0 * nItems * iterations) / seconds / 1e9;
        const double avgTime = elapsedMs / static_cast<double>(iterations);

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
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
