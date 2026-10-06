#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// SELL-C-sigma layout parameters: slices of C rows stored column-major,
// rows sorted by length inside windows of SIGMA rows to minimize padding.
constexpr index_t SLICE = 32;
constexpr index_t SIGMA = 1024;
constexpr int BLOCK_SIZE = 64;
constexpr int UNROLL = 8;
// Rows longer than max(LONG_ROW_MIN, 2 * average) are processed one block per row
constexpr index_t LONG_ROW_MIN = 256;
constexpr int LONG_BLOCK = 256;
constexpr int LONG_UNROLL = 4;

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format
//   (OpenMP-parallel over rows; per-row summation order is unchanged)
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
#pragma omp parallel for schedule(dynamic, 256)
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
// Function: spmvSellKernel
//
// Purpose:
//   SpMV on a SELL-32 matrix: one thread per row, fully coalesced loads of
//   values/columns. Each row is summed sequentially in the original CSR order
//   with separate multiply and add rounding, so results are bit-identical
//   to the serial CPU code. The loop is software-pipelined in chunks of
//   UNROLL entries (values/columns loaded two chunks ahead, vector gathered
//   one chunk ahead) to keep enough memory requests in flight even when
//   there are few, long rows per GPU.
//
// ****************************************************************************
__global__ void __launch_bounds__(BLOCK_SIZE)
spmvSellKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
               const size_t* __restrict__ sliceOffsets, const index_t* __restrict__ rowLen,
               const index_t* __restrict__ rowIdx, const double* __restrict__ vec,
               const index_t numRows, double* __restrict__ out) {
    const index_t r = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (r >= numRows) return;

    const index_t s = r / SLICE;
    const size_t base = sliceOffsets[s] + (r % SLICE);
    // Chunks run over the (padded) slice width so the warp stays converged;
    // padded entries are valid memory and are simply not accumulated.
    const index_t width = static_cast<index_t>((sliceOffsets[s + 1] - sliceOffsets[s]) / SLICE);
    const double* v = val + base;
    const index_t* c = cols + base;
    const index_t len = rowLen[r];
    const index_t numChunks = width / UNROLL;

    double a0[UNROLL], x0[UNROLL], a1[UNROLL];
    index_t c1[UNROLL];
    if (numChunks > 0) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            a0[u] = __ldg(v + static_cast<size_t>(u) * SLICE);
            x0[u] = __ldg(vec + __ldg(c + static_cast<size_t>(u) * SLICE));
        }
    }
    if (numChunks > 1) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            a1[u] = __ldg(v + static_cast<size_t>(UNROLL + u) * SLICE);
            c1[u] = __ldg(c + static_cast<size_t>(UNROLL + u) * SLICE);
        }
    }

    double t = 0.0;
    for (index_t ch = 0; ch < numChunks; ++ch) {
        double a2[UNROLL], x1[UNROLL];
        index_t c2[UNROLL];
        if (ch + 2 < numChunks) {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const size_t k = static_cast<size_t>(ch + 2) * UNROLL + u;
                a2[u] = __ldg(v + k * SLICE);
                c2[u] = __ldg(c + k * SLICE);
            }
        }
        if (ch + 1 < numChunks) {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) x1[u] = __ldg(vec + c1[u]);
        }
        const index_t k0 = ch * UNROLL;
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            if (k0 + u < len) t = __dadd_rn(t, __dmul_rn(a0[u], x0[u]));
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            a0[u] = a1[u];
            x0[u] = x1[u];
            a1[u] = a2[u];
            c1[u] = c2[u];
        }
    }
    for (index_t k = numChunks * UNROLL; k < len; ++k) {
        const size_t o = static_cast<size_t>(k) * SLICE;
        t = __dadd_rn(t, __dmul_rn(__ldg(v + o), __ldg(vec + __ldg(c + o))));
    }
    out[rowIdx[r]] = t;
}

// ****************************************************************************
// Function: spmvLongRowKernel
//
// Purpose:
//   SpMV for exceptionally long CSR rows: one block per row. All threads
//   stream the row with coalesced loads and form the products into
//   double-buffered shared memory; thread 0 accumulates them sequentially
//   in the original order (bit-identical to the serial CPU code).
//
// ****************************************************************************
__global__ void __launch_bounds__(LONG_BLOCK)
spmvLongRowKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                  const size_t* __restrict__ rowPtr, const index_t* __restrict__ rowIdx,
                  const double* __restrict__ vec, double* __restrict__ out) {
    constexpr index_t TILE = LONG_BLOCK * LONG_UNROLL;
    __shared__ double prod[2][TILE];

    const size_t start = rowPtr[blockIdx.x];
    const index_t len = static_cast<index_t>(rowPtr[blockIdx.x + 1] - start);
    const double* v = val + start;
    const index_t* c = cols + start;

    double t = 0.0;
    int buf = 0;
    for (index_t k0 = 0; k0 < len; k0 += TILE) {
#pragma unroll
        for (int u = 0; u < LONG_UNROLL; ++u) {
            const index_t k = k0 + u * LONG_BLOCK + threadIdx.x;
            if (k < len) prod[buf][u * LONG_BLOCK + threadIdx.x] = __dmul_rn(__ldg(v + k), __ldg(vec + __ldg(c + k)));
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const index_t kend = min(TILE, len - k0);
            index_t kk = 0;
            for (; kk + 8 <= kend; kk += 8) {
                double p[8];
#pragma unroll
                for (int u = 0; u < 8; ++u) p[u] = prod[buf][kk + u];
#pragma unroll
                for (int u = 0; u < 8; ++u) t = __dadd_rn(t, p[u]);
            }
            for (; kk < kend; ++kk) t = __dadd_rn(t, prod[buf][kk]);
        }
        buf ^= 1;
    }
    if (threadIdx.x == 0) out[rowIdx[blockIdx.x]] = t;
}

// Point-to-point / broadcast helpers that handle messages larger than INT_MAX bytes
constexpr size_t MPI_CHUNK_BYTES = size_t(1) << 30;

void sendBig(const void* buf, size_t bytes, int dest, int tag) {
    const char* p = static_cast<const char*>(buf);
    for (size_t off = 0; off < bytes; off += MPI_CHUNK_BYTES) {
        const int n = static_cast<int>(std::min(MPI_CHUNK_BYTES, bytes - off));
        MPI_Send(p + off, n, MPI_BYTE, dest, tag, MPI_COMM_WORLD);
    }
}

void recvBig(void* buf, size_t bytes, int src, int tag) {
    char* p = static_cast<char*>(buf);
    for (size_t off = 0; off < bytes; off += MPI_CHUNK_BYTES) {
        const int n = static_cast<int>(std::min(MPI_CHUNK_BYTES, bytes - off));
        MPI_Recv(p + off, n, MPI_BYTE, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void bcastBig(void* buf, size_t bytes, int root) {
    char* p = static_cast<char*>(buf);
    for (size_t off = 0; off < bytes; off += MPI_CHUNK_BYTES) {
        const int n = static_cast<int>(std::min(MPI_CHUNK_BYTES, bytes - off));
        MPI_Bcast(p + off, n, MPI_BYTE, root, MPI_COMM_WORLD);
    }
}

// Device-resident SELL-32 representation of this rank's block of rows
struct DeviceSell {
    index_t numRows = 0;      // rows stored in SELL format
    index_t numLongRows = 0;  // exceptionally long rows stored in CSR format
    double* longVal = nullptr;
    index_t* longCols = nullptr;
    size_t* longPtr = nullptr;
    index_t* longRowIdx = nullptr;
    double* val = nullptr;
    index_t* cols = nullptr;
    size_t* sliceOffsets = nullptr;
    index_t* rowLen = nullptr;
    index_t* rowIdx = nullptr;
};

// ****************************************************************************
// Function: buildDeviceSell
//
// Purpose:
//   Converts a local CSR block (rowPtr relative to val/cols) to SELL-32-sigma
//   on the host with OpenMP and uploads it to the current device. Rows much
//   longer than average are kept in a separate CSR block for the long-row
//   kernel so they neither inflate slice padding nor serialize a warp.
//
// ****************************************************************************
DeviceSell buildDeviceSell(const index_t numLocalRows, const index_t* rowPtr, const index_t* cols,
                           const double* val) {
    DeviceSell d;
    if (numLocalRows == 0) return d;

    // Classify rows
    const double avgLen = static_cast<double>(rowPtr[numLocalRows] - rowPtr[0]) / numLocalRows;
    const index_t longThreshold = std::max<index_t>(LONG_ROW_MIN, static_cast<index_t>(2.0 * avgLen));
    std::vector<index_t> shortRows, longRows;
    shortRows.reserve(numLocalRows);
    for (index_t i = 0; i < numLocalRows; ++i) {
        if (rowPtr[i + 1] - rowPtr[i] > longThreshold) {
            longRows.push_back(i);
        } else {
            shortRows.push_back(i);
        }
    }

    // Long rows: compact CSR copy
    d.numLongRows = static_cast<index_t>(longRows.size());
    if (d.numLongRows > 0) {
        std::vector<size_t> lPtr(d.numLongRows + 1, 0);
        for (index_t i = 0; i < d.numLongRows; ++i)
            lPtr[i + 1] = lPtr[i] + (rowPtr[longRows[i] + 1] - rowPtr[longRows[i]]);
        const size_t lNnz = lPtr[d.numLongRows];
        std::vector<double> lVal(lNnz);
        std::vector<index_t> lCols(lNnz);
        for (index_t i = 0; i < d.numLongRows; ++i) {
            const index_t b = rowPtr[longRows[i]];
            std::copy(val + b, val + rowPtr[longRows[i] + 1], lVal.begin() + lPtr[i]);
            std::copy(cols + b, cols + rowPtr[longRows[i] + 1], lCols.begin() + lPtr[i]);
        }
        CUDA_CHECK(cudaMalloc(&d.longVal, lNnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d.longCols, lNnz * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&d.longPtr, lPtr.size() * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d.longRowIdx, longRows.size() * sizeof(index_t)));
        CUDA_CHECK(cudaMemcpy(d.longVal, lVal.data(), lNnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.longCols, lCols.data(), lNnz * sizeof(index_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.longPtr, lPtr.data(), lPtr.size() * sizeof(size_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.longRowIdx, longRows.data(), longRows.size() * sizeof(index_t),
                              cudaMemcpyHostToDevice));
    }

    const index_t m = static_cast<index_t>(shortRows.size());
    d.numRows = m;
    if (m == 0) return d;

    const index_t numSlices = (m + SLICE - 1) / SLICE;
    const size_t paddedRows = static_cast<size_t>(numSlices) * SLICE;

    // Sort rows by descending length inside each sigma-window
    std::vector<index_t> perm(paddedRows);
    const index_t numWindows = (m + SIGMA - 1) / SIGMA;
#pragma omp parallel for schedule(dynamic)
    for (index_t w = 0; w < numWindows; ++w) {
        const index_t b = w * SIGMA;
        const index_t e = std::min(m, b + SIGMA);
        std::copy(shortRows.begin() + b, shortRows.begin() + e, perm.begin() + b);
        std::stable_sort(perm.begin() + b, perm.begin() + e, [&](index_t x, index_t y) {
            return (rowPtr[x + 1] - rowPtr[x]) > (rowPtr[y + 1] - rowPtr[y]);
        });
    }

    std::vector<index_t> rowLen(paddedRows, 0);
    std::vector<index_t> rowIdx(paddedRows, 0);
    std::vector<size_t> sliceOffsets(numSlices + 1, 0);
#pragma omp parallel for schedule(static)
    for (index_t s = 0; s < numSlices; ++s) {
        index_t width = 0;
        for (index_t l = 0; l < SLICE; ++l) {
            const size_t r = static_cast<size_t>(s) * SLICE + l;
            if (r < m) {
                const index_t orig = perm[r];
                rowIdx[r] = orig;
                rowLen[r] = rowPtr[orig + 1] - rowPtr[orig];
                width = std::max(width, rowLen[r]);
            } else {
                rowIdx[r] = 0;
                rowLen[r] = 0;
            }
        }
        sliceOffsets[s + 1] = static_cast<size_t>(width) * SLICE;
    }
    for (index_t s = 0; s < numSlices; ++s) sliceOffsets[s + 1] += sliceOffsets[s];
    const size_t total = sliceOffsets[numSlices];

    double* sVal = nullptr;
    index_t* sCols = nullptr;
    CUDA_CHECK(cudaMallocHost(&sVal, total * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&sCols, total * sizeof(index_t)));
#pragma omp parallel for schedule(dynamic, 16)
    for (index_t s = 0; s < numSlices; ++s) {
        const size_t off = sliceOffsets[s];
        const index_t width = static_cast<index_t>((sliceOffsets[s + 1] - off) / SLICE);
        for (index_t l = 0; l < SLICE; ++l) {
            const size_t r = static_cast<size_t>(s) * SLICE + l;
            const index_t len = rowLen[r];
            const index_t start = (r < m) ? rowPtr[rowIdx[r]] : 0;
            for (index_t k = 0; k < len; ++k) {
                sVal[off + static_cast<size_t>(k) * SLICE + l] = val[start + k];
                sCols[off + static_cast<size_t>(k) * SLICE + l] = cols[start + k];
            }
            for (index_t k = len; k < width; ++k) {
                sVal[off + static_cast<size_t>(k) * SLICE + l] = 0.0;
                sCols[off + static_cast<size_t>(k) * SLICE + l] = 0;
            }
        }
    }

    CUDA_CHECK(cudaMalloc(&d.val, std::max<size_t>(total, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.cols, std::max<size_t>(total, 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d.sliceOffsets, (numSlices + 1) * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d.rowLen, paddedRows * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d.rowIdx, paddedRows * sizeof(index_t)));
    CUDA_CHECK(cudaMemcpy(d.val, sVal, total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.cols, sCols, total * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.sliceOffsets, sliceOffsets.data(), (numSlices + 1) * sizeof(size_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.rowLen, rowLen.data(), paddedRows * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.rowIdx, rowIdx.data(), paddedRows * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFreeHost(sVal));
    CUDA_CHECK(cudaFreeHost(sCols));
    return d;
}

void freeDeviceSell(DeviceSell& d) {
    cudaFree(d.longVal);
    cudaFree(d.longCols);
    cudaFree(d.longPtr);
    cudaFree(d.longRowIdx);
    cudaFree(d.val);
    cudaFree(d.cols);
    cudaFree(d.sliceOffsets);
    cudaFree(d.rowLen);
    cudaFree(d.rowIdx);
    d = DeviceSell{};
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

int run(int argc, char** argv, const int rank, const int nranks) {
    const bool root = (rank == 0);
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identically on every rank)
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
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    // Select a GPU based on the node-local rank
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int localRank = 0;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        int numDevices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&numDevices));
        if (numDevices < 1) {
            fprintf(stderr, "No CUDA device available on rank %d\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % numDevices));
        CUDA_CHECK(cudaFree(nullptr));  // establish context outside the timed region
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

    // Allocate and initialize data structures (full matrix on root only)
    std::vector<double> h_val;                          // Non-zero values
    std::vector<index_t> h_cols;                        // Column indices
    std::vector<index_t> h_rowDelimiters;               // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out;                          // Output vector

    if (root) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_out.resize(numRows);
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Partition rows among ranks with balanced non-zero counts
    std::vector<index_t> rowBounds(nranks + 1, 0);
    if (root) {
        rowBounds[nranks] = numRows;
        for (int r = 1; r < nranks; ++r) {
            const index_t target = static_cast<index_t>(
                (static_cast<uint64_t>(nItems) * static_cast<uint64_t>(r)) / nranks);
            const auto it = std::lower_bound(h_rowDelimiters.begin(),
                                             h_rowDelimiters.begin() + numRows, target);
            rowBounds[r] = std::max(rowBounds[r - 1],
                                    static_cast<index_t>(it - h_rowDelimiters.begin()));
        }
    }
    MPI_Bcast(rowBounds.data(), nranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    bcastBig(h_vec.data(), static_cast<size_t>(numRows) * sizeof(double), 0);

    const index_t myRow0 = rowBounds[rank];
    const index_t myRows = rowBounds[rank + 1] - myRow0;

    // Distribute CSR blocks and build the device representation
    DeviceSell dmat;
    if (root) {
        for (int r = 1; r < nranks; ++r) {
            const index_t b = rowBounds[r], e = rowBounds[r + 1];
            const index_t nb = h_rowDelimiters[b], ne = h_rowDelimiters[e];
            sendBig(h_rowDelimiters.data() + b, (static_cast<size_t>(e - b) + 1) * sizeof(index_t), r, 1);
            sendBig(h_cols.data() + nb, static_cast<size_t>(ne - nb) * sizeof(index_t), r, 2);
            sendBig(h_val.data() + nb, static_cast<size_t>(ne - nb) * sizeof(double), r, 3);
        }
        std::vector<index_t> localPtr(h_rowDelimiters.begin(),
                                      h_rowDelimiters.begin() + myRows + 1);
        dmat = buildDeviceSell(myRows, localPtr.data(), h_cols.data(), h_val.data());
    } else {
        std::vector<index_t> localPtr(static_cast<size_t>(myRows) + 1);
        recvBig(localPtr.data(), localPtr.size() * sizeof(index_t), 0, 1);
        const index_t nb = localPtr[0];
        const size_t localNnz = localPtr[myRows] - nb;
        for (auto& p : localPtr) p -= nb;
        std::vector<index_t> localCols(localNnz);
        std::vector<double> localVal(localNnz);
        recvBig(localCols.data(), localNnz * sizeof(index_t), 0, 2);
        recvBig(localVal.data(), localNnz * sizeof(double), 0, 3);
        dmat = buildDeviceSell(myRows, localPtr.data(), localCols.data(), localVal.data());
    }

    double* d_vec = nullptr;
    double* d_out = nullptr;
    double* h_localOut = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vec, std::max<size_t>(numRows, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max<size_t>(myRows, 1) * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_localOut, std::max<size_t>(myRows, 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), static_cast<size_t>(numRows) * sizeof(double),
                          cudaMemcpyHostToDevice));
    // SELL rows and long rows run concurrently on separate streams
    cudaStream_t stream, longStream;
    cudaEvent_t longDone;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&longStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&longDone, cudaEventDisableTiming));
    const unsigned int numBlocks = (dmat.numRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto launchSpmv = [&]() {
        if (dmat.numLongRows > 0) {
            spmvLongRowKernel<<<dmat.numLongRows, LONG_BLOCK, 0, longStream>>>(
                dmat.longVal, dmat.longCols, dmat.longPtr, dmat.longRowIdx, d_vec, d_out);
        }
        if (numBlocks > 0) {
            spmvSellKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(
                dmat.val, dmat.cols, dmat.sliceOffsets, dmat.rowLen, dmat.rowIdx, d_vec,
                dmat.numRows, d_out);
        }
    };
    // Join the long-row stream into the main stream
    auto joinStreams = [&]() {
        CUDA_CHECK(cudaEventRecord(longDone, longStream));
        CUDA_CHECK(cudaStreamWaitEvent(stream, longDone, 0));
    };

    std::vector<int> recvCounts, displs;
    if (root) {
        recvCounts.resize(nranks);
        displs.resize(nranks);
        for (int r = 0; r < nranks; ++r) {
            recvCounts[r] = static_cast<int>(rowBounds[r + 1] - rowBounds[r]);
            displs[r] = static_cast<int>(rowBounds[r]);
        }
    }

    // For validation, compute reference solution (OpenMP on root)
    std::vector<double> h_reference;
    if (validate && root) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Untimed warm-up launch (module load, clock ramp-up, TLB warm-up)
    launchSpmv();
    CUDA_CHECK(cudaGetLastError());
    joinStreams();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Perform SpMV computation
    if (root) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpmv();
    }
    CUDA_CHECK(cudaGetLastError());
    joinStreams();
    if (iterations > 0 && myRows > 0) {
        CUDA_CHECK(cudaMemcpyAsync(h_localOut, d_out, static_cast<size_t>(myRows) * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (iterations > 0) {
        // Assemble the distributed result vector on root
        MPI_Gatherv(h_localOut, static_cast<int>(myRows), MPI_DOUBLE,
                    root ? h_out.data() : nullptr, recvCounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    CUDA_CHECK(cudaEventDestroy(longDone));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaStreamDestroy(longStream));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_localOut));
    freeDeviceSell(dmat);

    if (!root) return 0;

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / (elapsedMs / 1000.0) / 1e9;
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int rc = run(argc, argv, rank, nranks);

    // Propagate the root's exit status to all ranks
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
