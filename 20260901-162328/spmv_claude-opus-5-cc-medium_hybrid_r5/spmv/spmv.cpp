// Hybrid MPI + OpenMP + CUDA implementation of the SpMV benchmark.
//
// Parallelization strategy:
//   * MPI   : the matrix rows are partitioned across ranks with a load balance
//             based on the number of non-zeros per rank. Each rank only stores
//             (and multiplies) its own row slice; the dense input vector is
//             replicated. Rank 0 generates the matrix (the RNG sequence of the
//             original code must be preserved bit-for-bit) and distributes the
//             slices.
//   * CUDA  : every rank drives one GPU (round robin over the GPUs visible on
//             its compute node) and executes the local SpMV with a CSR
//             "vector" kernel using a tunable number of threads per row.
//   * OpenMP: host side work (reference solution, packing/unpacking of the
//             distributed slices) is threaded.

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

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        const cudaError_t status__ = (call);                                                  \
        if (status__ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(status__),        \
                    __FILE__, __LINE__, cudaGetErrorString(status__));                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                     \
        }                                                                                     \
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
//   (OpenMP threaded over the rows; rows are independent)
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
// CUDA kernel: CSR SpMV, TPR (threads per row) cooperating threads per row.
// The partial products of one row are reduced within a sub-warp group using
// warp shuffles; rows are distributed in a grid stride fashion.
// ****************************************************************************
template <int TPR>
__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec, const index_t numRows,
                           double* __restrict__ out) {
    const unsigned int lane = threadIdx.x & (TPR - 1);
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int groupsPerGrid = (gridDim.x * blockDim.x) / TPR;

    for (index_t row = gid / TPR; row < numRows; row += groupsPerGrid) {
        const index_t rowStart = rowDelimiters[row];
        const index_t rowEnd = rowDelimiters[row + 1];

        double t = 0.0;
        for (index_t j = rowStart + lane; j < rowEnd; j += TPR) {
            t += val[j] * __ldg(&vec[cols[j]]);
        }

#pragma unroll
        for (int offset = TPR / 2; offset > 0; offset >>= 1) {
            t += __shfl_down_sync(0xffffffffu, t, offset, TPR);
        }

        if (lane == 0) {
            out[row] = t;
        }
    }
}

// Launch helper selecting the number of threads per row at compile time
static void launchSpmv(int threadsPerRow, int blocks, int blockSize, cudaStream_t stream,
                       const double* val, const index_t* cols, const index_t* rowDelimiters,
                       const double* vec, index_t numRows, double* out) {
    switch (threadsPerRow) {
        case 2:
            spmvKernel<2><<<blocks, blockSize, 0, stream>>>(val, cols, rowDelimiters, vec, numRows, out);
            break;
        case 4:
            spmvKernel<4><<<blocks, blockSize, 0, stream>>>(val, cols, rowDelimiters, vec, numRows, out);
            break;
        case 8:
            spmvKernel<8><<<blocks, blockSize, 0, stream>>>(val, cols, rowDelimiters, vec, numRows, out);
            break;
        case 16:
            spmvKernel<16><<<blocks, blockSize, 0, stream>>>(val, cols, rowDelimiters, vec, numRows, out);
            break;
        default:
            spmvKernel<32><<<blocks, blockSize, 0, stream>>>(val, cols, rowDelimiters, vec, numRows, out);
            break;
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
    bool ok = true;
#pragma omp parallel for schedule(static) reduction(&& : ok)
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                ok = false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i, ref, res, relError);
                ok = false;
            }
        }
    }
    return ok;
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

// Send/receive potentially huge buffers (element counts may exceed the range of
// the int used for MPI counts), split into chunks.
template <typename T>
static void sendChunked(const T* buf, uint64_t count, int dest, int tag, MPI_Comm comm) {
    constexpr uint64_t kChunk = 1u << 26;  // elements per message
    uint64_t offset = 0;
    while (offset < count) {
        const int n = static_cast<int>(std::min<uint64_t>(kChunk, count - offset));
        MPI_Send(buf + offset, n * static_cast<int>(sizeof(T)), MPI_BYTE, dest, tag, comm);
        offset += static_cast<uint64_t>(n);
    }
}

template <typename T>
static void recvChunked(T* buf, uint64_t count, int src, int tag, MPI_Comm comm) {
    constexpr uint64_t kChunk = 1u << 26;
    uint64_t offset = 0;
    while (offset < count) {
        const int n = static_cast<int>(std::min<uint64_t>(kChunk, count - offset));
        MPI_Recv(buf + offset, n * static_cast<int>(sizeof(T)), MPI_BYTE, src, tag, comm,
                 MPI_STATUS_IGNORE);
        offset += static_cast<uint64_t>(n);
    }
}

int main(int argc, char** argv) {
    int mpiThreadLevel = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiThreadLevel);

    int numRanks = 1;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
        printf("MPI ranks: %d, OpenMP threads per rank: %d\n", numRanks, omp_get_max_threads());
    }

    // ------------------------------------------------------------------
    // Pick a GPU: round robin over the devices visible on this node
    // ------------------------------------------------------------------
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA device available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));  // establish the context before timing

    // ------------------------------------------------------------------
    // Data generation on rank 0 (keeps the original RNG sequence)
    // ------------------------------------------------------------------
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);  // dense vector, replicated on all ranks
    std::vector<double> h_out;           // full output vector (rank 0)

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        h_out.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ------------------------------------------------------------------
    // Row partitioning, balanced by number of non-zeros
    // ------------------------------------------------------------------
    std::vector<index_t> rowStart(static_cast<size_t>(numRanks) + 1, 0);
    if (rank == 0) {
        for (int r = 0; r <= numRanks; ++r) {
            const uint64_t target = (static_cast<uint64_t>(nItems) * r) / numRanks;
            // First row whose starting offset is >= target
            index_t lo = 0, hi = numRows;
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
        rowStart[0] = 0;
        rowStart[numRanks] = numRows;
        // keep the partition monotone
        for (int r = 1; r <= numRanks; ++r) {
            if (rowStart[r] < rowStart[r - 1]) rowStart[r] = rowStart[r - 1];
        }
    }
    MPI_Bcast(rowStart.data(), numRanks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t myFirstRow = rowStart[rank];
    const index_t myRows = rowStart[rank + 1] - rowStart[rank];

    // ------------------------------------------------------------------
    // Distribute the CSR slices
    // ------------------------------------------------------------------
    std::vector<index_t> l_rowDelimiters(static_cast<size_t>(myRows) + 1, 0);
    std::vector<double> l_val;
    std::vector<index_t> l_cols;
    uint64_t myNnz = 0;

    if (rank == 0) {
        for (int r = 1; r < numRanks; ++r) {
            const index_t first = rowStart[r];
            const index_t rows = rowStart[r + 1] - rowStart[r];
            const index_t base = h_rowDelimiters[first];
            const uint64_t nnz = static_cast<uint64_t>(h_rowDelimiters[first + rows]) - base;

            std::vector<index_t> delims(static_cast<size_t>(rows) + 1);
#pragma omp parallel for schedule(static)
            for (index_t i = 0; i <= rows; ++i) {
                delims[i] = h_rowDelimiters[first + i] - base;
            }
            sendChunked(delims.data(), static_cast<uint64_t>(rows) + 1, r, 0, MPI_COMM_WORLD);
            sendChunked(h_val.data() + base, nnz, r, 1, MPI_COMM_WORLD);
            sendChunked(h_cols.data() + base, nnz, r, 2, MPI_COMM_WORLD);
        }

        const index_t base = h_rowDelimiters[myFirstRow];
        myNnz = static_cast<uint64_t>(h_rowDelimiters[myFirstRow + myRows]) - base;
        l_val.assign(h_val.begin() + base, h_val.begin() + base + myNnz);
        l_cols.assign(h_cols.begin() + base, h_cols.begin() + base + myNnz);
#pragma omp parallel for schedule(static)
        for (index_t i = 0; i <= myRows; ++i) {
            l_rowDelimiters[i] = h_rowDelimiters[myFirstRow + i] - base;
        }
    } else {
        recvChunked(l_rowDelimiters.data(), static_cast<uint64_t>(myRows) + 1, 0, 0, MPI_COMM_WORLD);
        myNnz = l_rowDelimiters[myRows];
        l_val.resize(myNnz);
        l_cols.resize(myNnz);
        recvChunked(l_val.data(), myNnz, 0, 1, MPI_COMM_WORLD);
        recvChunked(l_cols.data(), myNnz, 0, 2, MPI_COMM_WORLD);
    }

    // ------------------------------------------------------------------
    // Upload the local slice to the GPU
    // ------------------------------------------------------------------
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_vec, static_cast<size_t>(numRows) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), static_cast<size_t>(numRows) * sizeof(double),
                          cudaMemcpyHostToDevice));
    if (myRows > 0) {
        CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (static_cast<size_t>(myRows) + 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(myRows) * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_rowDelimiters, l_rowDelimiters.data(),
                              (static_cast<size_t>(myRows) + 1) * sizeof(index_t),
                              cudaMemcpyHostToDevice));
    }
    if (myNnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, myNnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, myNnz * sizeof(index_t)));
        CUDA_CHECK(cudaMemcpy(d_val, l_val.data(), myNnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, l_cols.data(), myNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    }

    // Kernel configuration: threads per row follows the average row length
    int threadsPerRow = 32;
    if (myRows > 0) {
        const uint64_t nnzPerRow = (myNnz + myRows - 1) / myRows;
        threadsPerRow = 2;
        while (threadsPerRow < 32 && static_cast<uint64_t>(threadsPerRow) * 2 <= nnzPerRow) {
            threadsPerRow *= 2;
        }
    }
    constexpr int kBlockSize = 256;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const uint64_t neededBlocks =
        (static_cast<uint64_t>(myRows) * threadsPerRow + kBlockSize - 1) / kBlockSize;
    const uint64_t maxBlocks = static_cast<uint64_t>(prop.multiProcessorCount) * 32;
    const int blocks = static_cast<int>(std::max<uint64_t>(1, std::min(neededBlocks, maxBlocks)));

    // ------------------------------------------------------------------
    // Perform SpMV computation
    // ------------------------------------------------------------------
    // For validation, compute reference solution (host, OpenMP)
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    // Warm-up launch (loads the kernel module / spins up the GPU); computes the
    // same result as the timed iterations
    if (myRows > 0) {
        launchSpmv(threadsPerRow, blocks, kBlockSize, 0, d_val, d_cols, d_rowDelimiters, d_vec,
                   myRows, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (myRows > 0) {
            launchSpmv(threadsPerRow, blocks, kBlockSize, 0, d_val, d_cols, d_rowDelimiters, d_vec,
                       myRows, d_out);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // ------------------------------------------------------------------
    // Collect the result on rank 0
    // ------------------------------------------------------------------
    std::vector<double> l_out(myRows);
    if (myRows > 0) {
        CUDA_CHECK(cudaMemcpy(l_out.data(), d_out, static_cast<size_t>(myRows) * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    if (rank == 0) {
#pragma omp parallel for schedule(static)
        for (index_t i = 0; i < myRows; ++i) {
            h_out[myFirstRow + i] = l_out[i];
        }
        for (int r = 1; r < numRanks; ++r) {
            const index_t rows = rowStart[r + 1] - rowStart[r];
            if (rows > 0) {
                recvChunked(h_out.data() + rowStart[r], rows, r, 3, MPI_COMM_WORLD);
            }
        }
    } else if (myRows > 0) {
        sendChunked(l_out.data(), myRows, 0, 3, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

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

    if (d_val) CUDA_CHECK(cudaFree(d_val));
    if (d_cols) CUDA_CHECK(cudaFree(d_cols));
    if (d_rowDelimiters) CUDA_CHECK(cudaFree(d_rowDelimiters));
    if (d_out) CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_vec));

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
