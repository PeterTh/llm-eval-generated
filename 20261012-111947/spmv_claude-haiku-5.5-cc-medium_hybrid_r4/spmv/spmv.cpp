#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr index_t SLICE_HEIGHT = 32;  // Rows per SELL slice; matches the CUDA warp size
constexpr unsigned THREADS_PER_BLOCK = 256;
constexpr index_t LOAD_BATCH = 32;    // Products loaded in flight per thread before they are summed

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err_));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
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
//   Computes sparse matrix-vector multiplication using CSR format on the host.
//   Rows are independent, so they are distributed over OpenMP threads. Each row
//   is still summed in the same order as the serial code.
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
// Struct: SellMatrix
//
// Purpose:
//   Sliced ELLPACK (SELL-32) storage of a rank's row block. Rows are grouped
//   into slices of SLICE_HEIGHT consecutive rows. Each slice is padded to its
//   longest row and stored column-major, so the lanes of a warp read
//   consecutive addresses. Padding entries are (0.0, 0); they add exactly
//   zero to a row sum and leave the result unchanged.
//
// Members:
//   sliceOff: offset of each slice in val/cols; size = number of slices + 1
//   val:      non-zero values, padded
//   cols:     column indices, padded
//
// ****************************************************************************
struct SellMatrix {
    std::vector<size_t> sliceOff;
    std::vector<double> val;
    std::vector<index_t> cols;
};

// ****************************************************************************
// Function: toSell
//
// Purpose:
//   Converts a CSR row block to SELL-32 storage. Slices are independent, so
//   they are filled in parallel with OpenMP.
//
// Arguments:
//   val: CSR non-zero values of the block
//   cols: CSR column indices of the block
//   rowPtr: CSR row delimiters of the block (rows + 1 entries, relative to 0)
//   rows: number of rows in the block
//
// ****************************************************************************
SellMatrix toSell(const double* val, const index_t* cols, const index_t* rowPtr, const index_t rows) {
    const index_t nSlices = (rows + SLICE_HEIGHT - 1) / SLICE_HEIGHT;

    // Longest row in each slice
    std::vector<index_t> width(nSlices, 0);
    #pragma omp parallel for schedule(static)
    for (index_t s = 0; s < nSlices; ++s) {
        const index_t first = s * SLICE_HEIGHT;
        const index_t last = std::min(rows, first + SLICE_HEIGHT);
        for (index_t r = first; r < last; ++r) {
            width[s] = std::max(width[s], rowPtr[r + 1] - rowPtr[r]);
        }
    }

    SellMatrix m;
    m.sliceOff.assign(nSlices + 1, 0);
    for (index_t s = 0; s < nSlices; ++s) {
        m.sliceOff[s + 1] = m.sliceOff[s] + static_cast<size_t>(width[s]) * SLICE_HEIGHT;
    }
    m.val.resize(m.sliceOff[nSlices]);
    m.cols.resize(m.sliceOff[nSlices]);

    #pragma omp parallel for schedule(static)
    for (index_t s = 0; s < nSlices; ++s) {
        for (index_t lane = 0; lane < SLICE_HEIGHT; ++lane) {
            const index_t r = s * SLICE_HEIGHT + lane;
            const index_t begin = r < rows ? rowPtr[r] : 0;
            const index_t len = r < rows ? rowPtr[r + 1] - rowPtr[r] : 0;
            for (index_t k = 0; k < width[s]; ++k) {
                const size_t idx = m.sliceOff[s] + static_cast<size_t>(k) * SLICE_HEIGHT + lane;
                const bool real = k < len;
                m.val[idx] = real ? val[begin + k] : 0.0;
                m.cols[idx] = real ? cols[begin + k] : 0;
            }
        }
    }
    return m;
}

// ****************************************************************************
// Kernel: spmvSellKernel
//
// Purpose:
//   Computes one SpMV on the local row block, one thread per row, using SELL-32
//   storage. Each row accumulates its entries in the same order as the CSR
//   reference, so the results are identical to the host computation. Products
//   are rounded separately from the adds (no FMA contraction), matching the host
//   rounding. Each row is a sequential sum, so LOAD_BATCH products are loaded
//   before they are added to keep many memory requests in flight per thread.
//
// Arguments:
//   val, cols: SELL-32 non-zero values and column indices
//   sliceOff: SELL-32 slice offsets
//   vec: dense input vector (full length)
//   rows: number of rows in the local block
//   out: local output vector
//
// ****************************************************************************
__global__ void spmvSellKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                               const size_t* __restrict__ sliceOff, const double* __restrict__ vec,
                               const index_t rows, double* __restrict__ out) {
    const index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const index_t slice = row / SLICE_HEIGHT;
    const index_t lane = row % SLICE_HEIGHT;
    const size_t begin = sliceOff[slice] + lane;
    const index_t width = static_cast<index_t>((sliceOff[slice + 1] - sliceOff[slice]) / SLICE_HEIGHT);

    double t = 0.0;
    index_t k = 0;
    for (; k + LOAD_BATCH <= width; k += LOAD_BATCH) {
        double p[LOAD_BATCH];
        #pragma unroll
        for (index_t u = 0; u < LOAD_BATCH; ++u) {
            const size_t idx = begin + static_cast<size_t>(k + u) * SLICE_HEIGHT;
            p[u] = __dmul_rn(val[idx], vec[cols[idx]]);
        }
        #pragma unroll
        for (index_t u = 0; u < LOAD_BATCH; ++u) {
            t = __dadd_rn(t, p[u]);
        }
    }
    for (; k < width; ++k) {
        const size_t idx = begin + static_cast<size_t>(k) * SLICE_HEIGHT;
        t = __dadd_rn(t, __dmul_rn(val[idx], vec[cols[idx]]));
    }
    out[row] = t;
}

// ****************************************************************************
// Function: deviceAlloc / toDevice
//
// Purpose:
//   Allocate device memory (at least one element, so empty blocks are valid)
//   and copy a host vector to the device.
//
// ****************************************************************************
template <typename T>
T* deviceAlloc(const size_t count) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(count, 1) * sizeof(T)));
    return d;
}

template <typename T>
T* toDevice(const std::vector<T>& host) {
    T* d = deviceAlloc<T>(host.size());
    if (!host.empty()) {
        CUDA_CHECK(cudaMemcpy(d, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    return d;
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

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   Runs the benchmark on all MPI ranks. Rank 0 generates the problem and
//   prints. Rows are split into contiguous blocks with roughly equal numbers of
//   non-zeros, one block per rank. Each rank runs its block on its GPU.
//
// ****************************************************************************
int runBenchmark(int argc, char** argv, const int rank, const int nprocs) {
    const bool root = rank == 0;

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
            if (root) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    // Bind each rank to one GPU on its node, round-robin over the devices present
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_free(&nodeComm);
    CUDA_CHECK(cudaSetDevice(nodeRank % deviceCount));

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (root) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Rank 0 generates the full problem. The RNG sequence is the same as in the serial code.
    std::vector<double> h_val;                          // Non-zero values (root only)
    std::vector<index_t> h_cols;                        // Column indices (root only)
    std::vector<index_t> h_rowDelimiters;               // Row delimiters (root only)
    std::vector<double> h_vec(numRows);                 // Dense vector (all ranks)

    if (root) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && root) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Partition rows into contiguous blocks with roughly equal numbers of non-zeros
    std::vector<index_t> rowBounds(nprocs + 1, 0);   // Rank r owns rows [rowBounds[r], rowBounds[r+1])
    std::vector<index_t> nnzBounds(nprocs + 1, 0);   // Rank r owns non-zeros [nnzBounds[r], nnzBounds[r+1])
    if (root) {
        rowBounds[nprocs] = numRows;
        for (int r = 1; r < nprocs; ++r) {
            const auto target = static_cast<index_t>(static_cast<uint64_t>(nItems) * r / nprocs);
            rowBounds[r] = static_cast<index_t>(
                std::lower_bound(h_rowDelimiters.begin(), h_rowDelimiters.end(), target) -
                h_rowDelimiters.begin());
        }
        for (int r = 0; r <= nprocs; ++r) {
            nnzBounds[r] = h_rowDelimiters[rowBounds[r]];
        }
    }
    MPI_Bcast(rowBounds.data(), nprocs + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzBounds.data(), nprocs + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Per-rank counts and displacements for the scatter and gather
    std::vector<int> rowsPerRank(nprocs), rowStart(nprocs), ptrPerRank(nprocs);
    std::vector<int> nnzPerRank(nprocs), nnzStart(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        rowsPerRank[r] = static_cast<int>(rowBounds[r + 1] - rowBounds[r]);
        rowStart[r] = static_cast<int>(rowBounds[r]);
        ptrPerRank[r] = rowsPerRank[r] + 1;
        nnzPerRank[r] = static_cast<int>(nnzBounds[r + 1] - nnzBounds[r]);
        nnzStart[r] = static_cast<int>(nnzBounds[r]);
    }

    // Distribute the local CSR block
    const index_t localRows = static_cast<index_t>(rowsPerRank[rank]);
    const index_t localNnz = static_cast<index_t>(nnzPerRank[rank]);
    std::vector<index_t> localRowPtr(localRows + 1);
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);

    MPI_Scatterv(root ? h_rowDelimiters.data() : nullptr, ptrPerRank.data(), rowStart.data(),
                 MPI_UINT32_T, localRowPtr.data(), static_cast<int>(localRows + 1), MPI_UINT32_T,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(root ? h_val.data() : nullptr, nnzPerRank.data(), nnzStart.data(), MPI_DOUBLE,
                 localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(root ? h_cols.data() : nullptr, nnzPerRank.data(), nnzStart.data(), MPI_UINT32_T,
                 localCols.data(), static_cast<int>(localNnz), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Make row delimiters local to this block
    const index_t localBase = localRowPtr[0];
    #pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= localRows; ++i) {
        localRowPtr[i] -= localBase;
    }

    // The full problem is no longer needed on rank 0
    if (root) {
        std::vector<double>().swap(h_val);
        std::vector<index_t>().swap(h_cols);
        std::vector<index_t>().swap(h_rowDelimiters);
    }

    // Build the device-side representation and upload it
    const SellMatrix sell = toSell(localVal.data(), localCols.data(), localRowPtr.data(), localRows);
    std::vector<double>().swap(localVal);
    std::vector<index_t>().swap(localCols);

    double* d_val = toDevice(sell.val);
    index_t* d_cols = toDevice(sell.cols);
    size_t* d_sliceOff = toDevice(sell.sliceOff);
    double* d_vec = toDevice(h_vec);
    double* d_out = deviceAlloc<double>(localRows);

    // Perform SpMV computation
    if (root) {
        printf("Computing SpMV...\n");
    }
    const unsigned blocks = (localRows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows > 0) {
            spmvSellKernel<<<blocks, THREADS_PER_BLOCK>>>(d_val, d_cols, d_sliceOff, d_vec, localRows, d_out);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy the local result back to the host
    std::vector<double> localOut(localRows);
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(localOut.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost));
    }

    if (root) {
        const auto durationMs = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (durationMs / 1000.0) / 1e9;
        const double avgTime = durationMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather the full result on rank 0 only if it is printed or validated
    std::vector<double> h_out;
    if (printResults || validate) {
        if (root) {
            h_out.resize(numRows);
        }
        MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    root ? h_out.data() : nullptr, rowsPerRank.data(), rowStart.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (printResults && root) {
        print_results(h_out, "OutputVector");
    }

    // Validation
    int status = 0;
    if (validate && root) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_sliceOff));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int status = runBenchmark(argc, argv, rank, nprocs);

    MPI_Finalize();
    return status;
}
