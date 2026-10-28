// Hybrid MPI + OpenMP + CUDA sparse matrix-vector multiplication benchmark.
//
// Decomposition:
//   - MPI: rows are partitioned across ranks, balanced by non-zero count.
//          Each rank owns a contiguous block of CSR rows. The matrix is
//          generated deterministically on every rank (identical to the serial
//          generator), so no matrix data is communicated. The dense vector is
//          constant across iterations, so the timed loop needs no communication.
//   - CUDA: each rank drives one GPU (selected by node-local rank). A warp-per-row
//          CSR kernel gives the GPU enough independent work to saturate memory
//          bandwidth even when a rank holds few rows; lanes read consecutive
//          non-zeros (coalesced) and reduce with warp shuffles.
//   - OpenMP: host-side work that is data-parallel and rank-local: the CPU
//          reference SpMV used for validation. The random matrix generator is
//          inherently sequential (it consumes one rand() stream) and is kept
//          serial so the generated matrix is identical to the original program.
//
// Numerics: each row is a sum of products. The GPU reduces each row's partial
// sums in a fixed warp-tree order, which differs from the serial left-to-right
// order only by floating-point reassociation. The result is deterministic and
// independent of the number of ranks or GPUs, and is checked against the serial
// reference with the same tolerance as before.

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr unsigned int WARP_SIZE = 32;
constexpr unsigned int CUDA_BLOCK_SIZE = 256;  // 8 warps per block

// ****************************************************************************
// Macro: CHECK_CUDA
//
// Purpose:
//   Abort all MPI ranks if a CUDA runtime call fails
//
// ****************************************************************************
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        const cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// ****************************************************************************
// Function: rootPrintf
//
// Purpose:
//   printf restricted to MPI rank 0 so that output is printed once
//
// ****************************************************************************
__attribute__((format(printf, 2, 3))) static void rootPrintf(int rank, const char* fmt, ...) {
    if (rank != 0) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
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
//   Computes sparse matrix-vector multiplication using CSR format on the host.
//   Rows are independent and each row is summed in order, so the result does
//   not depend on the number of OpenMP threads.
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
// Kernel: spmvCsrVectorKernel
//
// Purpose:
//   CSR SpMV on the device with one warp per row. Lanes stride over the row's
//   non-zeros (coalesced loads), then combine partial sums with a warp shuffle
//   reduction. Thread-per-row would leave most SMs idle when a rank holds few
//   rows, because each row's sum is long.
//
// Arguments:
//   val, cols, rowDelimiters: local CSR arrays (rowDelimiters rebased to 0)
//   vec: dense input vector (full length)
//   rows: number of local rows
//   out: local output vector
//
// ****************************************************************************
__global__ void spmvCsrVectorKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                                    const index_t* __restrict__ rowDelimiters,
                                    const double* __restrict__ vec, const index_t rows,
                                    double* __restrict__ out) {
    const uint64_t globalThread = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t row = globalThread / WARP_SIZE;
    if (row >= rows) {
        return;  // whole warp exits together, so the shuffles below are safe
    }
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const uint64_t end = rowDelimiters[row + 1];

    double t = 0.0;
    for (uint64_t j = rowDelimiters[row] + lane; j < end; j += WARP_SIZE) {
        t += val[j] * vec[cols[j]];
    }

    for (unsigned int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        t += __shfl_down_sync(0xffffffffu, t, offset);
    }
    if (lane == 0) {
        out[row] = t;
    }
}

// ****************************************************************************
// Function: deviceAlloc / copyToDevice
//
// Purpose:
//   Allocate device memory (at least one element, so zero-row ranks are valid)
//   and copy host vectors to it
//
// ****************************************************************************
template <typename T>
T* deviceAlloc(size_t count) {
    T* ptr = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr), std::max<size_t>(count, 1) * sizeof(T)));
    return ptr;
}

template <typename T>
void copyToDevice(T* dst, const std::vector<T>& src) {
    if (!src.empty()) {
        CHECK_CUDA(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
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

void printUsage(int rank, const char* progName) {
    rootPrintf(rank, "Usage: %s [options]\n", progName);
    rootPrintf(rank, "Options:\n");
    rootPrintf(rank, "  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    rootPrintf(rank, "  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    rootPrintf(rank, "  -i <num>     Number of iterations (default: 10)\n");
    rootPrintf(rank, "  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    rootPrintf(rank, "  -v           Enable validation\n");
    rootPrintf(rank, "  -r           Print results for external validation\n");
    rootPrintf(rank, "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
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

    // Parse command line arguments (identical on every rank)
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
            printUsage(rank, argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            rootPrintf(rank, "Unknown option: %s\n", argv[i]);
            printUsage(rank, argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    rootPrintf(rank, "Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    rootPrintf(rank, "Matrix size: %u x %u\n", numRows, numRows);
    rootPrintf(rank, "Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    rootPrintf(rank, "Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
    rootPrintf(rank, "Iterations: %u\n", iterations);
    rootPrintf(rank, "Max value: %.2f\n", maxVal);
    rootPrintf(rank, "Validation: %s\n", validate ? "enabled" : "disabled");

    // Bind each rank to one GPU on its node
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CHECK_CUDA(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CHECK_CUDA(cudaSetDevice(localRank % numDevices));

    // Row partition balanced by non-zeros. rowStart has numRanks+1 entries;
    // rank r owns rows [rowStart[r], rowStart[r+1]).
    std::vector<index_t> rowStart(numRanks + 1);
    std::vector<double> h_vec(numRows);
    std::vector<double> l_val;               // local non-zero values
    std::vector<index_t> l_cols;             // local column indices
    std::vector<index_t> l_rowDelimiters;    // local row delimiters (rebased to 0)

    rootPrintf(rank, "Initializing data structures...\n");
    {
        // Full matrix is generated on every rank with the same sequential RNG
        // stream, so all ranks agree on the matrix without communication.
        std::vector<double> h_val(nItems);                  // Non-zero values
        std::vector<index_t> h_cols(nItems);                // Column indices
        std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters

        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        rowStart[0] = 0;
        rowStart[numRanks] = numRows;
        for (int r = 1; r < numRanks; ++r) {
            const auto target = static_cast<index_t>((static_cast<uint64_t>(nItems) * r) / numRanks);
            const auto it = std::lower_bound(h_rowDelimiters.begin(), h_rowDelimiters.end(), target);
            rowStart[r] = std::min<index_t>(static_cast<index_t>(it - h_rowDelimiters.begin()), numRows);
        }

        // Extract this rank's CSR slice (OpenMP-parallel copy)
        const index_t rowBegin = rowStart[rank];
        const index_t rowEnd = rowStart[rank + 1];
        const index_t localRows = rowEnd - rowBegin;
        const index_t nnzBegin = h_rowDelimiters[rowBegin];
        const index_t localNnz = h_rowDelimiters[rowEnd] - nnzBegin;

        l_val.resize(localNnz);
        l_cols.resize(localNnz);
        l_rowDelimiters.resize(localRows + 1);

#pragma omp parallel for schedule(static)
        for (index_t k = 0; k < localNnz; ++k) {
            l_val[k] = h_val[nnzBegin + k];
            l_cols[k] = h_cols[nnzBegin + k];
        }
#pragma omp parallel for schedule(static)
        for (index_t i = 0; i <= localRows; ++i) {
            l_rowDelimiters[i] = h_rowDelimiters[rowBegin + i] - nnzBegin;
        }
    }

    const index_t localRows = rowStart[rank + 1] - rowStart[rank];

    // Stage the local CSR slice and the dense vector on the device
    double* d_val = deviceAlloc<double>(l_val.size());
    index_t* d_cols = deviceAlloc<index_t>(l_cols.size());
    index_t* d_rowDelimiters = deviceAlloc<index_t>(l_rowDelimiters.size());
    double* d_vec = deviceAlloc<double>(numRows);
    double* d_out = deviceAlloc<double>(localRows);
    copyToDevice(d_val, l_val);
    copyToDevice(d_cols, l_cols);
    copyToDevice(d_rowDelimiters, l_rowDelimiters);
    copyToDevice(d_vec, h_vec);
    // Zero the output so iterations == 0 yields the same zero vector as the original
    CHECK_CUDA(cudaMemset(d_out, 0, std::max<size_t>(localRows, 1) * sizeof(double)));
    CHECK_CUDA(cudaDeviceSynchronize());

    // For validation, compute reference solution (rank-local rows, OpenMP on host)
    std::vector<double> l_reference;
    if (validate) {
        rootPrintf(rank, "Computing reference solution...\n");
        l_reference.resize(localRows);
        spmvCpu(l_val.data(), l_cols.data(), l_rowDelimiters.data(), h_vec.data(), localRows,
                l_reference.data());
    }

    // Perform SpMV computation
    rootPrintf(rank, "Computing SpMV...\n");
    const uint64_t totalThreads = static_cast<uint64_t>(localRows) * WARP_SIZE;
    const auto numBlocks = static_cast<unsigned int>((totalThreads + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (localRows > 0) {
        for (index_t iter = 0; iter < iterations; ++iter) {
            spmvCsrVectorKernel<<<numBlocks, CUDA_BLOCK_SIZE>>>(d_val, d_cols, d_rowDelimiters, d_vec,
                                                                localRows, d_out);
        }
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    rootPrintf(rank, "Computation time: %ld ms\n", static_cast<long>(duration.count()));

    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
    const double avgTime = duration.count() / static_cast<double>(iterations);

    rootPrintf(rank, "Average time per iteration: %.3f ms\n", avgTime);
    rootPrintf(rank, "Performance: %.3f GFLOPS\n", gflops);

    // Retrieve local output and assemble the full vector on rank 0
    std::vector<double> l_out(localRows);
    if (localRows > 0) {
        CHECK_CUDA(cudaMemcpy(l_out.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost));
    }

    std::vector<int> counts(numRanks);
    std::vector<int> displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        counts[r] = static_cast<int>(rowStart[r + 1] - rowStart[r]);
        displs[r] = static_cast<int>(rowStart[r]);
    }

    std::vector<double> h_out(rank == 0 ? numRows : 0);
    MPI_Gatherv(l_out.data(), static_cast<int>(localRows), MPI_DOUBLE, h_out.data(), counts.data(),
                displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print results for external validation
    if (printResults) {
        if (rank == 0) {
            print_results(h_out, "OutputVector");
        }
    }

    int rc = 0;
    if (validate) {
        std::vector<double> h_reference(rank == 0 ? numRows : 0);
        MPI_Gatherv(l_reference.data(), static_cast<int>(localRows), MPI_DOUBLE, h_reference.data(),
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        int valid = 0;
        if (rank == 0) {
            printf("Validating result...\n");
            valid = verifyResults(h_reference.data(), h_out.data(), numRows) ? 1 : 0;
            printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        rc = valid ? 0 : 1;
    }

    CHECK_CUDA(cudaFree(d_val));
    CHECK_CUDA(cudaFree(d_cols));
    CHECK_CUDA(cudaFree(d_rowDelimiters));
    CHECK_CUDA(cudaFree(d_vec));
    CHECK_CUDA(cudaFree(d_out));

    MPI_Finalize();
    return rc;
}
