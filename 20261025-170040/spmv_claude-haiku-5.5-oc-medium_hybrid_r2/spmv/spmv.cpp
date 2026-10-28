// Hybrid MPI + OpenMP + CUDA sparse matrix-vector multiplication benchmark.
//
// Decomposition:
//   MPI:    rows are split into contiguous blocks, one per rank, with block
//           boundaries chosen so every rank owns a similar number of nonzeros.
//           Rank 0 generates the matrix and vector exactly as the sequential
//           original did (the rand() sequence defines the data), then
//           distributes the blocks. The vector is replicated on all ranks and
//           the output blocks are gathered back to rank 0.
//   CUDA:   each rank binds to one GPU (node-local rank modulo device count)
//           and computes its row block with a CSR kernel. The local matrix
//           block is uploaded once, before timing.
//   OpenMP: the host-side reference computation used for validation is
//           parallelized over the rows of each rank's block.

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int TAG_ROWDELIM = 1;
constexpr int TAG_VAL = 2;
constexpr int TAG_COLS = 3;
constexpr size_t MPI_CHUNK = size_t(1) << 30;  // max elements per MPI message
constexpr int CUDA_BLOCK = 256;

#define CUDA_CHECK(call)                                                               \
    do {                                                                               \
        cudaError_t err_ = (call);                                                     \
        if (err_ != cudaSuccess) {                                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
                    cudaGetErrorString(err_));                                         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                              \
        }                                                                              \
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
//   Computes sparse matrix-vector multiplication using CSR format.
//   Rows are independent, so they are distributed across OpenMP threads.
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
//   CSR SpMV with one thread per row. Each row is accumulated sequentially in
//   the same order as spmvCpu, so device results are bitwise identical to the
//   host reference. Rows are independent, so the kernel scales with rows.
//
// ****************************************************************************
__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec, const index_t rows,
                           double* __restrict__ out) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; row < rows;
         row += stride) {
        double t = 0.0;
        const index_t end = rowDelimiters[row + 1];
        for (index_t j = rowDelimiters[row]; j < end; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[row] = t;
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

// ****************************************************************************
// Function: computeRowPartition
//
// Purpose:
//   Splits rows into `parts` contiguous blocks with roughly equal nonzero
//   counts. Block p owns rows [bounds[p], bounds[p+1]).
//
// ****************************************************************************
std::vector<index_t> computeRowPartition(const index_t* rowDelimiters, const index_t dim,
                                         const index_t nnz, const int parts) {
    std::vector<index_t> bounds(parts + 1);
    bounds[0] = 0;
    bounds[parts] = dim;
    for (int p = 1; p < parts; ++p) {
        const index_t target = static_cast<index_t>(static_cast<uint64_t>(nnz) * p / parts);
        const index_t* it = std::lower_bound(rowDelimiters, rowDelimiters + dim + 1, target);
        index_t bound = static_cast<index_t>(it - rowDelimiters);
        bounds[p] = std::max(bound, bounds[p - 1]);
    }
    return bounds;
}

// Sends `count` elements in MPI-sized chunks. Requests are completed by the caller.
template <typename T>
void sendChunked(const T* buf, size_t count, MPI_Datatype type, int dst, int tag,
                 std::vector<MPI_Request>& reqs) {
    for (size_t off = 0; off < count; off += MPI_CHUNK) {
        const size_t n = std::min(MPI_CHUNK, count - off);
        MPI_Request req;
        MPI_Isend(buf + off, static_cast<int>(n), type, dst, tag, MPI_COMM_WORLD, &req);
        reqs.push_back(req);
    }
}

template <typename T>
void recvChunked(T* buf, size_t count, MPI_Datatype type, int src, int tag) {
    for (size_t off = 0; off < count; off += MPI_CHUNK) {
        const size_t n = std::min(MPI_CHUNK, count - off);
        MPI_Recv(buf + off, static_cast<int>(n), type, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

template <typename T>
void bcastChunked(T* buf, size_t count, MPI_Datatype type) {
    for (size_t off = 0; off < count; off += MPI_CHUNK) {
        const size_t n = std::min(MPI_CHUNK, count - off);
        MPI_Bcast(buf + off, static_cast<int>(n), type, 0, MPI_COMM_WORLD);
    }
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
// Function: uploadToDevice
//
// Purpose:
//   Allocates device memory for `count` elements and copies them from host.
//   The allocation is at least one element so that empty blocks are valid.
//
// ****************************************************************************
template <typename T>
T* uploadToDevice(const T* src, size_t count) {
    T* dst = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dst), std::max<size_t>(count, 1) * sizeof(T)));
    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    return dst;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    const bool root = (rank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank, so all ranks
    // take the same exit path)
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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

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

    // Generate the matrix and vector on rank 0 (sequential, same RNG order as the original)
    std::vector<double> h_vec(numRows);                 // Dense vector (replicated on all ranks)
    std::vector<double> h_val;                          // Non-zero values (root only)
    std::vector<index_t> h_cols;                        // Column indices (root only)
    std::vector<index_t> h_rowDelimiters;               // Row delimiters (root only)
    std::vector<index_t> bounds(numRanks + 1);          // Row block boundaries

    if (root) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        bounds = computeRowPartition(h_rowDelimiters.data(), numRows, nItems, numRanks);
    }
    MPI_Bcast(bounds.data(), numRanks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    bcastChunked(h_vec.data(), numRows, MPI_DOUBLE);

    // Distribute row blocks from rank 0
    const index_t rowBegin = bounds[rank];
    const index_t rows = bounds[rank + 1] - rowBegin;
    std::vector<index_t> localRowDelimiters(rows + 1);
    std::vector<double> recvVal;
    std::vector<index_t> recvCols;
    const double* localVal = nullptr;
    const index_t* localCols = nullptr;

    if (root) {
        std::vector<MPI_Request> reqs;
        for (int p = 1; p < numRanks; ++p) {
            const index_t r0 = bounds[p];
            const index_t r1 = bounds[p + 1];
            const index_t d0 = h_rowDelimiters[r0];
            const index_t d1 = h_rowDelimiters[r1];
            sendChunked(h_rowDelimiters.data() + r0, static_cast<size_t>(r1 - r0) + 1,
                        MPI_UINT32_T, p, TAG_ROWDELIM, reqs);
            sendChunked(h_val.data() + d0, d1 - d0, MPI_DOUBLE, p, TAG_VAL, reqs);
            sendChunked(h_cols.data() + d0, d1 - d0, MPI_UINT32_T, p, TAG_COLS, reqs);
        }

        // Rank 0 uses its portion of the generated arrays directly
        const index_t d0 = h_rowDelimiters[rowBegin];
        for (index_t i = 0; i <= rows; ++i) {
            localRowDelimiters[i] = h_rowDelimiters[rowBegin + i] - d0;
        }
        localVal = h_val.data() + d0;
        localCols = h_cols.data() + d0;

        if (!reqs.empty()) MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    } else {
        recvChunked(localRowDelimiters.data(), static_cast<size_t>(rows) + 1, MPI_UINT32_T, 0,
                    TAG_ROWDELIM);
        const index_t base = localRowDelimiters[0];
        for (index_t i = 0; i <= rows; ++i) {
            localRowDelimiters[i] -= base;
        }
        const size_t localNnz = localRowDelimiters[rows];
        recvVal.resize(localNnz);
        recvCols.resize(localNnz);
        recvChunked(recvVal.data(), localNnz, MPI_DOUBLE, 0, TAG_VAL);
        recvChunked(recvCols.data(), localNnz, MPI_UINT32_T, 0, TAG_COLS);
        localVal = recvVal.data();
        localCols = recvCols.data();
    }
    const index_t localNnz = localRowDelimiters[rows];

    // Bind this rank to a GPU (ranks on a node are spread round-robin over its devices)
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        if (root) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    // Upload local data to the device once (not part of the timed region)
    double* d_val = uploadToDevice(localVal, localNnz);
    index_t* d_cols = uploadToDevice(localCols, localNnz);
    index_t* d_rowDelimiters = uploadToDevice(localRowDelimiters.data(), static_cast<size_t>(rows) + 1);
    double* d_vec = uploadToDevice(h_vec.data(), numRows);
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), std::max<size_t>(rows, 1) * sizeof(double)));

    const size_t numBlocksWanted = (static_cast<size_t>(rows) + CUDA_BLOCK - 1) / CUDA_BLOCK;
    const int numBlocks = static_cast<int>(std::min<size_t>(numBlocksWanted, 1u << 20));

    auto launchSpmv = [&]() {
        if (rows == 0) return;
        spmvKernel<<<numBlocks, CUDA_BLOCK>>>(d_val, d_cols, d_rowDelimiters, d_vec, rows, d_out);
        CUDA_CHECK(cudaGetLastError());
    };

    // For validation, compute the reference solution (distributed; OpenMP within each rank)
    std::vector<double> localReference;
    std::vector<double> h_reference;
    if (validate) {
        if (root) printf("Computing reference solution...\n");
        localReference.resize(rows);
        spmvCpu(localVal, localCols, localRowDelimiters.data(), h_vec.data(), rows, localReference.data());
        if (root) h_reference.resize(numRows);
        std::vector<int> counts(numRanks), displs(numRanks);
        for (int p = 0; p < numRanks; ++p) {
            counts[p] = static_cast<int>(bounds[p + 1] - bounds[p]);
            displs[p] = static_cast<int>(bounds[p]);
        }
        MPI_Gatherv(localReference.data(), static_cast<int>(rows), MPI_DOUBLE, h_reference.data(),
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Perform SpMV computation. Untimed warm-up launch first so that one-time
    // CUDA initialization is not attributed to the measured iterations.
    if (root) printf("Computing SpMV...\n");
    launchSpmv();
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpmv();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    const double localMs = std::chrono::duration<double, std::milli>(end - start).count();
    double durationMs = 0.0;
    MPI_Reduce(&localMs, &durationMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the output vector on rank 0
    std::vector<double> localOut(rows);
    if (rows > 0) {
        CUDA_CHECK(cudaMemcpy(localOut.data(), d_out, static_cast<size_t>(rows) * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }
    std::vector<double> h_out(root ? numRows : 0);
    {
        std::vector<int> counts(numRanks), displs(numRanks);
        for (int p = 0; p < numRanks; ++p) {
            counts[p] = static_cast<int>(bounds[p + 1] - bounds[p]);
            displs[p] = static_cast<int>(bounds[p]);
        }
        MPI_Gatherv(localOut.data(), static_cast<int>(rows), MPI_DOUBLE, h_out.data(),
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (root) {
        printf("Computation time: %ld ms\n", static_cast<long>(durationMs));

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (durationMs / 1000.0) / 1e9;
        const double avgTime = durationMs / static_cast<double>(iterations);

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

    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));
    MPI_Finalize();
    return exitCode;
}
