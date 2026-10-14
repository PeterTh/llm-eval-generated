#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr index_t SLICE_HEIGHT = 32;      // Rows per SELL slice (one warp of rows)
constexpr unsigned THREADS_PER_BLOCK = 128;
constexpr index_t UNROLL = 8;             // Nonzeros loaded per batch inside the SpMV kernel
constexpr double LONG_ROW_FACTOR = 4.0;   // Rows with more than this multiple of the average row length are long
constexpr unsigned LONG_THREADS = 256;    // Threads per block for one long row
constexpr index_t LONG_CHUNK = 4096;      // Products of a long row staged in shared memory at a time

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err_));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// ****************************************************************************
// Function: blockBegin
//
// Purpose:
//   Splits n rows into contiguous blocks. Block p starts at blockBegin(n, parts, p)
//   and ends before blockBegin(n, parts, p+1).
//
// Arguments:
//   n:     number of rows
//   parts: number of blocks
//   part:  block index in [0, parts]
//
// ****************************************************************************
index_t blockBegin(const index_t n, const int parts, const int part) {
    return static_cast<index_t>(static_cast<uint64_t>(n) * static_cast<uint64_t>(part) /
                                static_cast<uint64_t>(parts));
}

// ****************************************************************************
// Function: fillRange
//
// Purpose:
//   Initialize array with random values. Draws the same n values as the serial
//   fill, but only stores those with indexes in [begin, end)
//
// Arguments:
//   A: pointer to the array to initialize; holds end-begin elements, A[i-begin] = value i
//   n: number of values drawn from the random number generator
//   maxVal: specifies range of random values [0, maxVal]
//   begin: first index stored in A
//   end: one past the last index stored in A
//
// ****************************************************************************
void fillRange(double* A, const index_t n, const double maxVal, const index_t begin, const index_t end) {
    for (index_t i = 0; i < n; ++i) {
        const double val = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
        if (i >= begin && i < end) {
            A[i - begin] = val;
        }
    }
}

// ****************************************************************************
// Function: initRandomMatrix
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix. The function encodes these positions in compressed sparse
//   row (CSR) format. Row delimiters are produced for every row, but column
//   indexes are only stored for rows in [rowBegin, rowEnd).
//
// Arguments:
//   cols:          array for column indexes of the local elements
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   n:             number of nonzero elements
//   dim:           number of rows/columns in the matrix
//   rowBegin:      first row stored locally
//   rowEnd:        one past the last row stored locally
//
// ****************************************************************************
void initRandomMatrix(std::vector<index_t>& cols, index_t* rowDelimiters, const index_t n, const index_t dim,
                      const index_t rowBegin, const index_t rowEnd) {
    index_t nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        const bool isLocal = i >= rowBegin && i < rowEnd;
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                if (isLocal) {
                    cols.push_back(j);
                }
                nnzAssigned++;
            }
        }
    }
    // Convention: put the number of non-zeroes at the end of the row delimiters array
    rowDelimiters[dim] = n;
}

// ****************************************************************************
// Function: buildSell
//
// Purpose:
//   Repacks the short CSR rows into sliced ELLPACK (SELL-32) layout for the GPU.
//   Rows are grouped into slices of SLICE_HEIGHT consecutive rows. Inside a slice,
//   the k-th nonzero of every row is stored contiguously, so a warp reads each
//   step as one coalesced segment. The nonzeros of each row keep their CSR order.
//   Rows longer than longThreshold are left out; they are handled separately, so
//   that a single long row does not pad every row of its slice.
//
// Arguments:
//   val: array holding the non-zero values in CSR order
//   cols: array of column indices for each element in CSR order
//   rowDelimiters: array of size nRows+1 holding indices to rows
//   nRows: number of rows
//   longThreshold: rows with more nonzeros than this are not packed
//   sliceOff: output array of size nSlices+1 holding the offset of each slice
//   sVal: output packed values
//   sCols: output packed column indices
//
// ****************************************************************************
void buildSell(const std::vector<double>& val, const std::vector<index_t>& cols,
               const std::vector<index_t>& rowDelimiters, const index_t nRows, const index_t longThreshold,
               std::vector<size_t>& sliceOff, std::vector<double>& sVal, std::vector<index_t>& sCols) {
    const index_t nSlices = (nRows + SLICE_HEIGHT - 1) / SLICE_HEIGHT;
    // Length of the row as stored in SELL: zero for long rows
    auto packedLen = [&](const index_t r) {
        const index_t len = rowDelimiters[r + 1] - rowDelimiters[r];
        return len > longThreshold ? index_t{0} : len;
    };

    sliceOff.assign(nSlices + 1, 0);
    for (index_t s = 0; s < nSlices; ++s) {
        index_t maxLen = 0;
        const index_t last = std::min(nRows, (s + 1) * SLICE_HEIGHT);
        for (index_t r = s * SLICE_HEIGHT; r < last; ++r) {
            maxLen = std::max(maxLen, packedLen(r));
        }
        sliceOff[s + 1] = sliceOff[s] + static_cast<size_t>(SLICE_HEIGHT) * maxLen;
    }

    sVal.assign(sliceOff[nSlices], 0.0);
    sCols.assign(sliceOff[nSlices], 0);

    #pragma omp parallel for schedule(static)
    for (index_t s = 0; s < nSlices; ++s) {
        const index_t last = std::min(nRows, (s + 1) * SLICE_HEIGHT);
        for (index_t r = s * SLICE_HEIGHT; r < last; ++r) {
            const index_t lane = r - s * SLICE_HEIGHT;
            const index_t len = packedLen(r);
            for (index_t k = 0; k < len; ++k) {
                const size_t dst = sliceOff[s] + static_cast<size_t>(k) * SLICE_HEIGHT + lane;
                const index_t src = rowDelimiters[r] + k;
                sVal[dst] = val[src];
                sCols[dst] = cols[src];
            }
        }
    }
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

// ****************************************************************************
// Kernel: spmvSellKernel
//
// Purpose:
//   Computes sparse matrix-vector multiplication for one short row per thread using
//   the SELL-32 layout built by buildSell. Threads of a warp handle the rows of one
//   slice, so each step reads val and cols with coalesced loads. Each row sums its
//   nonzeros in CSR order. Long rows are left to spmvLongRowKernel.
//
// Arguments:
//   val, cols: packed nonzero values and column indices (SELL-32 layout)
//   sliceOff: offset of each slice in val/cols
//   rowDelimiters: array of size nRows+1 holding indices to rows (local CSR)
//   longThreshold: rows with more nonzeros than this are long
//   vec: dense vector of size dim to be used for multiplication
//   nRows: number of local rows
//   out: output - result from the spmv calculation for the local rows
//
// ****************************************************************************
__global__ void spmvSellKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                               const size_t* __restrict__ sliceOff, const index_t* __restrict__ rowDelimiters,
                               const index_t longThreshold, const double* __restrict__ vec, const index_t nRows,
                               double* __restrict__ out) {
    const index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nRows) {
        return;
    }
    const index_t lane = row % SLICE_HEIGHT;
    const size_t base = sliceOff[row / SLICE_HEIGHT] + lane;
    const index_t len = rowDelimiters[row + 1] - rowDelimiters[row];
    if (len > longThreshold) {
        return;
    }

    // Products of UNROLL nonzeros are loaded before they are added, so many memory requests are in
    // flight per thread. The adds still happen in CSR order, so the row sum is unchanged
    double t = 0.0;
    index_t k = 0;
    for (; k + UNROLL <= len; k += UNROLL) {
        double p[UNROLL];
        #pragma unroll
        for (index_t u = 0; u < UNROLL; ++u) {
            const size_t idx = base + static_cast<size_t>(k + u) * SLICE_HEIGHT;
            p[u] = val[idx] * vec[cols[idx]];
        }
        #pragma unroll
        for (index_t u = 0; u < UNROLL; ++u) {
            t += p[u];
        }
    }
    for (; k < len; ++k) {
        const size_t idx = base + static_cast<size_t>(k) * SLICE_HEIGHT;
        t += val[idx] * vec[cols[idx]];
    }
    out[row] = t;
}

// ****************************************************************************
// Kernel: spmvLongRowKernel
//
// Purpose:
//   Computes sparse matrix-vector multiplication for the long rows, one block per
//   row. The products are computed in parallel and staged in shared memory, then
//   thread 0 adds them in CSR order, so the row sum matches the serial result.
//   Splitting a row across threads would change the summation order.
//
// Arguments:
//   val, cols: nonzero values and column indices of the long rows (CSR order)
//   longPtr: offsets of each long row in val/cols (size nLong+1)
//   longRow: local row index of each long row
//   vec: dense vector of size dim to be used for multiplication
//   out: output - result from the spmv calculation for the local rows
//
// ****************************************************************************
__global__ void spmvLongRowKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                                  const index_t* __restrict__ longPtr, const index_t* __restrict__ longRow,
                                  const double* __restrict__ vec, double* __restrict__ out) {
    __shared__ double products[LONG_CHUNK];
    const index_t id = blockIdx.x;
    const index_t begin = longPtr[id];
    const index_t len = longPtr[id + 1] - begin;

    double t = 0.0;
    for (index_t c = 0; c < len; c += LONG_CHUNK) {
        const index_t count = min(LONG_CHUNK, len - c);
        for (index_t i = threadIdx.x; i < count; i += blockDim.x) {
            const index_t idx = begin + c + i;
            products[i] = val[idx] * vec[cols[idx]];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (index_t i = 0; i < count; ++i) {
                t += products[i];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[longRow[id]] = t;
    }
}

// ****************************************************************************
// Function: toDevice
//
// Purpose:
//   Allocates device memory and copies a host vector to it
//
// Arguments:
//   host: source vector
//
// ****************************************************************************
template <typename T>
T* toDevice(const std::vector<T>& host) {
    T* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
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
//   offset: global index of the first element, used in messages
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size, const index_t offset) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", offset + i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       offset + i, ref, res, relError);
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
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

    // Ranks on the same node share its GPUs and CPU cores
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int ranksOnNode = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &ranksOnNode);
    MPI_Comm_free(&nodeComm);

    // Each rank drives one GPU. Ranks on a node are spread over its devices
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA device available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // Split the node's CPU cores between its ranks unless the user chose a thread count
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / ranksOnNode));
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

    // Rows [rowBegin, rowEnd) are owned by this rank
    const index_t rowBegin = blockBegin(numRows, numRanks, rank);
    const index_t rowEnd = blockBegin(numRows, numRanks, rank + 1);
    const index_t nLocal = rowEnd - rowBegin;

    // Allocate and initialize data structures. Every rank replays the same deterministic random
    // streams as the serial code and keeps only the part it owns, so no matrix data is communicated
    if (root) {
        printf("Initializing data structures...\n");
    }
    std::vector<index_t> rowDelimiters(numRows + 1);    // Row delimiters (all rows)
    std::vector<index_t> h_cols;                        // Column indices of local rows
    initRandomMatrix(h_cols, rowDelimiters.data(), nItems, numRows, rowBegin, rowEnd);

    const index_t valBegin = rowDelimiters[rowBegin];
    const index_t valEnd = rowDelimiters[rowEnd];
    std::vector<double> h_vec(numRows);                 // Dense vector (replicated on all ranks)
    std::vector<double> h_val(valEnd - valBegin);       // Non-zero values of local rows

    // The default generator state is the same as seed 1, so this reproduces the serial sequence
    srand(1);
    fillRange(h_vec.data(), numRows, maxVal, 0, numRows);
    fillRange(h_val.data(), nItems, maxVal, valBegin, valEnd);

    std::vector<index_t> h_rowPtr(nLocal + 1);          // Local row delimiters, relative to valBegin
    for (index_t i = 0; i <= nLocal; ++i) {
        h_rowPtr[i] = rowDelimiters[rowBegin + i] - valBegin;
    }

    // For validation, compute the local part of the reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (root) {
            printf("Computing reference solution...\n");
        }
        h_reference.resize(nLocal);
        spmvCpu(h_val.data(), h_cols.data(), h_rowPtr.data(), h_vec.data(), nLocal, h_reference.data());
    }

    // Rows much longer than the local average are long rows. They are kept in CSR order
    // in their own arrays, so that one long row cannot delay all rows of its slice
    const index_t nnzLocal = h_rowPtr[nLocal];
    const double avgRowLen = nLocal > 0 ? static_cast<double>(nnzLocal) / nLocal : 0.0;
    const index_t longThreshold = static_cast<index_t>(LONG_ROW_FACTOR * avgRowLen);
    std::vector<index_t> h_longRow;     // Local row index of each long row
    std::vector<index_t> h_longPtr(1, 0);
    std::vector<double> h_longVal;
    std::vector<index_t> h_longCols;
    for (index_t r = 0; r < nLocal; ++r) {
        if (h_rowPtr[r + 1] - h_rowPtr[r] > longThreshold) {
            h_longRow.push_back(r);
            h_longVal.insert(h_longVal.end(), h_val.begin() + h_rowPtr[r], h_val.begin() + h_rowPtr[r + 1]);
            h_longCols.insert(h_longCols.end(), h_cols.begin() + h_rowPtr[r], h_cols.begin() + h_rowPtr[r + 1]);
            h_longPtr.push_back(static_cast<index_t>(h_longVal.size()));
        }
    }
    const index_t nLong = static_cast<index_t>(h_longRow.size());

    // Move the local matrix to the GPU: short rows in SELL-32 layout, long rows in CSR
    std::vector<size_t> h_sliceOff;
    std::vector<double> h_sVal;
    std::vector<index_t> h_sCols;
    buildSell(h_val, h_cols, h_rowPtr, nLocal, longThreshold, h_sliceOff, h_sVal, h_sCols);

    double* d_val = toDevice(h_sVal);
    index_t* d_cols = toDevice(h_sCols);
    size_t* d_sliceOff = toDevice(h_sliceOff);
    index_t* d_rowPtr = toDevice(h_rowPtr);
    double* d_longVal = toDevice(h_longVal);
    index_t* d_longCols = toDevice(h_longCols);
    index_t* d_longPtr = toDevice(h_longPtr);
    index_t* d_longRow = toDevice(h_longRow);
    double* d_vec = toDevice(h_vec);
    double* d_out = toDevice(std::vector<double>(nLocal, 0.0));

    // Short and long rows run on separate streams so that long rows overlap the SELL kernel
    cudaStream_t shortStream;
    cudaStream_t longStream;
    CUDA_CHECK(cudaStreamCreate(&shortStream));
    CUDA_CHECK(cudaStreamCreate(&longStream));

    const unsigned blocks = (nLocal + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto runSpmv = [&]() {
        if (nLocal > 0) {
            spmvSellKernel<<<blocks, THREADS_PER_BLOCK, 0, shortStream>>>(
                d_val, d_cols, d_sliceOff, d_rowPtr, longThreshold, d_vec, nLocal, d_out);
        }
        if (nLong > 0) {
            spmvLongRowKernel<<<nLong, LONG_THREADS, 0, longStream>>>(
                d_longVal, d_longCols, d_longPtr, d_longRow, d_vec, d_out);
        }
        CUDA_CHECK(cudaGetLastError());
    };

    // Perform SpMV computation
    if (root) {
        printf("Computing SpMV...\n");
    }
    // Untimed warm-up: loads the kernel and fills the output, which every iteration overwrites
    if (iterations > 0) {
        runSpmv();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        runSpmv();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy the local part of the output back to the host
    std::vector<double> h_out(nLocal);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, nLocal * sizeof(double), cudaMemcpyDeviceToHost));

    if (root) {
        printf("Computation time: %ld ms\n", max_ms);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (max_ms / 1000.0) / 1e9;
        const double avgTime = max_ms / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<int> counts(numRanks);
        std::vector<int> displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            displs[r] = static_cast<int>(blockBegin(numRows, numRanks, r));
            counts[r] = static_cast<int>(blockBegin(numRows, numRanks, r + 1)) - displs[r];
        }
        std::vector<double> h_full(root ? numRows : 0);
        MPI_Gatherv(h_out.data(), static_cast<int>(nLocal), MPI_DOUBLE, h_full.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (root) {
            print_results(h_full, "OutputVector");
        }
    }

    int exitCode = 0;
    // Validation
    if (validate) {
        if (root) {
            printf("Validating result...\n");
        }
        int valid = verifyResults(h_reference.data(), h_out.data(), nLocal, rowBegin) ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (root) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        exitCode = valid ? 0 : 1;
    }

    CUDA_CHECK(cudaStreamDestroy(shortStream));
    CUDA_CHECK(cudaStreamDestroy(longStream));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_sliceOff));
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_longVal));
    CUDA_CHECK(cudaFree(d_longCols));
    CUDA_CHECK(cudaFree(d_longPtr));
    CUDA_CHECK(cudaFree(d_longRow));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    MPI_Finalize();
    return exitCode;
}
