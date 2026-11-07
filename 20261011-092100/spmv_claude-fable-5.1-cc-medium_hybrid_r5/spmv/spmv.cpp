// Sparse Matrix-Vector Multiplication (SpMV) benchmark
//
// Hybrid parallel implementation:
//   * MPI    : the CSR matrix is partitioned by rows (balanced by non-zeros)
//              across ranks; every rank owns a contiguous block of rows.
//   * CUDA   : each rank drives one or more GPUs; its block of rows is split
//              again (balanced by non-zeros) across those GPUs and the SpMV
//              itself runs as a CSR "vector" kernel (several threads per row).
//   * OpenMP : one host thread per GPU launches and synchronizes that GPU's
//              work concurrently; host-side preprocessing, the reference
//              solution, and verification are also OpenMP parallel.
//
// Matrix generation is sequential (it depends on the ordering of rand()
// calls) and therefore happens on rank 0 exactly as in the original program;
// the data is then distributed with MPI.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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
constexpr int BLOCK_SIZE = 256;
// Maximum number of elements per MPI point-to-point message (keeps counts < INT_MAX)
constexpr size_t MPI_CHUNK_ELEMS = size_t(1) << 28;

// ****************************************************************************
// Error handling helpers
// ****************************************************************************
#define CUDA_CHECK(call)                                                                \
    do {                                                                                \
        cudaError_t err__ = (call);                                                     \
        if (err__ != cudaSuccess) {                                                     \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err__),    \
                    __FILE__, __LINE__, cudaGetErrorString(err__));                     \
            MPI_Abort(MPI_COMM_WORLD, 1);                                               \
        }                                                                               \
    } while (0)

#define MPI_CHECK(call)                                                                 \
    do {                                                                                \
        int err__ = (call);                                                             \
        if (err__ != MPI_SUCCESS) {                                                     \
            char msg__[MPI_MAX_ERROR_STRING];                                           \
            int len__ = 0;                                                              \
            MPI_Error_string(err__, msg__, &len__);                                     \
            fprintf(stderr, "MPI error at %s:%d: %s\n", __FILE__, __LINE__, msg__);     \
            MPI_Abort(MPI_COMM_WORLD, 1);                                               \
        }                                                                               \
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
//   Computes sparse matrix-vector multiplication using CSR format on the host
//   (OpenMP parallel over rows). Used for the validation reference.
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
// Kernel: spmvCsrKernel
//
// Purpose:
//   CSR "vector" SpMV kernel: TPR consecutive threads of a warp cooperate on
//   one row, striding over its non-zeros, then reduce with warp shuffles.
//   TPR == 1 degenerates to the classic scalar (thread-per-row) kernel.
//
// ****************************************************************************
template <int TPR>
__global__ void __launch_bounds__(BLOCK_SIZE)
spmvCsrKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
              const index_t* __restrict__ rowPtr, const double* __restrict__ vec,
              const index_t numRows, double* __restrict__ out) {
    const unsigned long long tid =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long row = tid / TPR;
    const int lane = static_cast<int>(threadIdx.x % TPR);
    const bool active = row < numRows;

    double t = 0.0;
    if (active) {
        const index_t start = rowPtr[row];
        const index_t end = rowPtr[row + 1];
        for (index_t j = start + lane; j < end; j += TPR) {
            t += val[j] * __ldg(&vec[cols[j]]);
        }
    }

    if constexpr (TPR > 1) {
        // All threads of the warp reach this point (no early return above).
#pragma unroll
        for (int off = TPR / 2; off > 0; off >>= 1) {
            t += __shfl_down_sync(0xffffffffu, t, off, TPR);
        }
    }

    if (active && lane == 0) {
        out[row] = t;
    }
}

// ****************************************************************************
// Function: partitionRows
//
// Purpose:
//   Splits rows [0, numRows) into `parts` contiguous ranges so that the work
//   (non-zeros + one unit per row) is balanced. Returns parts+1 boundaries.
//
// ****************************************************************************
std::vector<index_t> partitionRows(const index_t* rowPtr, const index_t numRows, const int parts) {
    std::vector<index_t> bounds(parts + 1);
    bounds[0] = 0;
    bounds[parts] = numRows;
    const uint64_t totalCost = static_cast<uint64_t>(rowPtr[numRows]) + numRows;
    for (int p = 1; p < parts; ++p) {
        const uint64_t target = totalCost * static_cast<uint64_t>(p) / static_cast<uint64_t>(parts);
        // Binary search for the first row r with cost(r) = rowPtr[r] + r >= target
        index_t lo = bounds[p - 1];
        index_t hi = numRows;
        while (lo < hi) {
            const index_t mid = lo + (hi - lo) / 2;
            const uint64_t cost = static_cast<uint64_t>(rowPtr[mid]) + mid;
            if (cost < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        bounds[p] = lo;
    }
    return bounds;
}

// ****************************************************************************
// Function: chooseThreadsPerRow
//
// Purpose:
//   Picks the number of cooperating threads per row from the average row
//   length (standard CSR-vector heuristic).
//
// ****************************************************************************
int chooseThreadsPerRow(const index_t nnz, const index_t rows) {
    if (rows == 0) return 1;
    const double avg = static_cast<double>(nnz) / static_cast<double>(rows);
    if (avg <= 1.0) return 1;
    if (avg <= 2.0) return 2;
    if (avg <= 4.0) return 4;
    if (avg <= 8.0) return 8;
    if (avg <= 16.0) return 16;
    return 32;
}

// ****************************************************************************
// Per-GPU state for the rows handled by one device of this rank
// ****************************************************************************
struct GpuWork {
    int device = 0;
    index_t rowBegin = 0;   // local row index (relative to this rank's block)
    index_t rowEnd = 0;
    index_t nnzBegin = 0;   // local nnz index (relative to this rank's block)
    index_t nnzEnd = 0;
    int threadsPerRow = 1;
    cudaStream_t stream = nullptr;
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowPtr = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    index_t rows() const { return rowEnd - rowBegin; }
    index_t nnz() const { return nnzEnd - nnzBegin; }
};

void launchSpmv(const GpuWork& g) {
    const index_t rows = g.rows();
    if (rows == 0) return;
    const unsigned long long threads = static_cast<unsigned long long>(rows) * g.threadsPerRow;
    const unsigned int blocks = static_cast<unsigned int>((threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
    switch (g.threadsPerRow) {
        case 1:
            spmvCsrKernel<1><<<blocks, BLOCK_SIZE, 0, g.stream>>>(g.d_val, g.d_cols, g.d_rowPtr, g.d_vec, rows, g.d_out);
            break;
        case 2:
            spmvCsrKernel<2><<<blocks, BLOCK_SIZE, 0, g.stream>>>(g.d_val, g.d_cols, g.d_rowPtr, g.d_vec, rows, g.d_out);
            break;
        case 4:
            spmvCsrKernel<4><<<blocks, BLOCK_SIZE, 0, g.stream>>>(g.d_val, g.d_cols, g.d_rowPtr, g.d_vec, rows, g.d_out);
            break;
        case 8:
            spmvCsrKernel<8><<<blocks, BLOCK_SIZE, 0, g.stream>>>(g.d_val, g.d_cols, g.d_rowPtr, g.d_vec, rows, g.d_out);
            break;
        case 16:
            spmvCsrKernel<16><<<blocks, BLOCK_SIZE, 0, g.stream>>>(g.d_val, g.d_cols, g.d_rowPtr, g.d_vec, rows, g.d_out);
            break;
        default:
            spmvCsrKernel<32><<<blocks, BLOCK_SIZE, 0, g.stream>>>(g.d_val, g.d_cols, g.d_rowPtr, g.d_vec, rows, g.d_out);
            break;
    }
}

// ****************************************************************************
// Chunked point-to-point helpers (robust for element counts beyond INT_MAX)
// ****************************************************************************
template <typename T>
void sendChunked(const T* buf, size_t count, int dest, int tag) {
    for (size_t off = 0; off < count; off += MPI_CHUNK_ELEMS) {
        const int n = static_cast<int>(std::min(MPI_CHUNK_ELEMS, count - off));
        MPI_CHECK(MPI_Send(buf + off, n * static_cast<int>(sizeof(T)), MPI_BYTE, dest, tag, MPI_COMM_WORLD));
    }
    if (count == 0) {
        MPI_CHECK(MPI_Send(buf, 0, MPI_BYTE, dest, tag, MPI_COMM_WORLD));
    }
}

template <typename T>
void recvChunked(T* buf, size_t count, int src, int tag) {
    for (size_t off = 0; off < count; off += MPI_CHUNK_ELEMS) {
        const int n = static_cast<int>(std::min(MPI_CHUNK_ELEMS, count - off));
        MPI_CHECK(MPI_Recv(buf + off, n * static_cast<int>(sizeof(T)), MPI_BYTE, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    if (count == 0) {
        MPI_CHECK(MPI_Recv(buf, 0, MPI_BYTE, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
}

// Distribute contiguous slices of a rank-0 array: rank p receives [offs[p], offs[p+1]).
template <typename T>
void scatterSlices(const T* full, const std::vector<uint64_t>& offs, T* local, int rank, int size, int tag) {
    if (rank == 0) {
        const size_t myCount = static_cast<size_t>(offs[1] - offs[0]);
        if (myCount > 0) std::memcpy(local, full + offs[0], myCount * sizeof(T));
        for (int p = 1; p < size; ++p) {
            sendChunked(full + offs[p], static_cast<size_t>(offs[p + 1] - offs[p]), p, tag);
        }
    } else {
        recvChunked(local, static_cast<size_t>(offs[rank + 1] - offs[rank]), 0, tag);
    }
}

// Inverse of scatterSlices.
template <typename T>
void gatherSlices(const T* local, const std::vector<uint64_t>& offs, T* full, int rank, int size, int tag) {
    if (rank == 0) {
        const size_t myCount = static_cast<size_t>(offs[1] - offs[0]);
        if (myCount > 0) std::memcpy(full + offs[0], local, myCount * sizeof(T));
        for (int p = 1; p < size; ++p) {
            recvChunked(full + offs[p], static_cast<size_t>(offs[p + 1] - offs[p]), p, tag);
        }
    } else {
        sendChunked(local, static_cast<size_t>(offs[rank + 1] - offs[rank]), 0, tag);
    }
}

template <typename T>
void bcastChunked(T* buf, size_t count) {
    for (size_t off = 0; off < count; off += MPI_CHUNK_ELEMS) {
        const int n = static_cast<int>(std::min(MPI_CHUNK_ELEMS, count - off));
        MPI_CHECK(MPI_Bcast(buf + off, n * static_cast<int>(sizeof(T)), MPI_BYTE, 0, MPI_COMM_WORLD));
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
    // Find the first failing index in parallel, then report it (same message as
    // the sequential version).
    index_t firstBad = size;
#pragma omp parallel for schedule(static) reduction(min : firstBad)
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        bool ok;
        if (std::abs(ref) < 1e-10) {
            ok = !(std::abs(res) > MAX_RELATIVE_ERROR);
        } else {
            ok = !(std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR);
        }
        if (!ok && i < firstBad) firstBad = i;
    }
    if (firstBad == size) return true;

    const index_t i = firstBad;
    const double ref = reference[i];
    const double res = result[i];
    if (std::abs(ref) < 1e-10) {
        // For very small values, check absolute error
        printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
    } else {
        // Check relative error
        const double relError = std::abs((res - ref) / ref);
        printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
               i, ref, res, relError);
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
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank = 0, size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

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

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    // ------------------------------------------------------------------
    // GPU assignment: ranks sharing a node split that node's GPUs evenly
    // ------------------------------------------------------------------
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int localRank = 0, localSize = 1;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_size(nodeComm, &localSize));
    MPI_CHECK(MPI_Comm_free(&nodeComm));

    int deviceCount = 0;
    const cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount <= 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available (%s)\n", rank,
                devErr == cudaSuccess ? "count is 0" : cudaGetErrorString(devErr));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> myDevices;
    if (localSize <= deviceCount) {
        for (int d = localRank; d < deviceCount; d += localSize) myDevices.push_back(d);
    } else {
        myDevices.push_back(localRank % deviceCount);  // oversubscribed: share GPUs
    }
    const int numGpus = static_cast<int>(myDevices.size());

    int totalGpus = 0;
    MPI_CHECK(MPI_Reduce(&numGpus, &totalGpus, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: MPI ranks: %d, total GPUs: %d, OpenMP threads/rank: %d\n",
               size, totalGpus, omp_get_max_threads());
    }

    // ------------------------------------------------------------------
    // Allocate and initialize data structures (generation on rank 0 only,
    // exactly as in the sequential program)
    // ------------------------------------------------------------------
    std::vector<double> h_val;             // Non-zero values (rank 0: full)
    std::vector<index_t> h_cols;           // Column indices  (rank 0: full)
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters (all ranks)
    std::vector<double> h_vec(numRows);    // Dense vector (all ranks)
    std::vector<double> h_out;             // Output vector (rank 0: full)

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_out.resize(numRows);
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    bcastChunked(h_rowDelimiters.data(), h_rowDelimiters.size());
    bcastChunked(h_vec.data(), h_vec.size());

    // Row partition across ranks (balanced by nnz; identical on all ranks)
    const std::vector<index_t> rankRows = partitionRows(h_rowDelimiters.data(), numRows, size);
    std::vector<uint64_t> rankRowOffs(size + 1), rankNnzOffs(size + 1);
    for (int p = 0; p <= size; ++p) {
        rankRowOffs[p] = rankRows[p];
        rankNnzOffs[p] = h_rowDelimiters[rankRows[p]];
    }
    const index_t myRowBegin = rankRows[rank];
    const index_t myRows = rankRows[rank + 1] - rankRows[rank];
    const index_t myNnz = h_rowDelimiters[rankRows[rank + 1]] - h_rowDelimiters[myRowBegin];

    // Local CSR block
    std::vector<double> l_val(myNnz);
    std::vector<index_t> l_cols(myNnz);
    std::vector<index_t> l_rowPtr(static_cast<size_t>(myRows) + 1);
    std::vector<double> l_out(myRows);

    scatterSlices(h_val.data(), rankNnzOffs, l_val.data(), rank, size, 100);
    scatterSlices(h_cols.data(), rankNnzOffs, l_cols.data(), rank, size, 101);

    {
        const index_t base = h_rowDelimiters[myRowBegin];
#pragma omp parallel for schedule(static)
        for (index_t r = 0; r <= myRows; ++r) {
            l_rowPtr[r] = h_rowDelimiters[myRowBegin + r] - base;
        }
    }

    // Split this rank's block across its GPUs and set up device state
    const std::vector<index_t> gpuRows = partitionRows(l_rowPtr.data(), myRows, numGpus);
    std::vector<GpuWork> gpus(numGpus);
    for (int g = 0; g < numGpus; ++g) {
        gpus[g].device = myDevices[g];
        gpus[g].rowBegin = gpuRows[g];
        gpus[g].rowEnd = gpuRows[g + 1];
        gpus[g].nnzBegin = l_rowPtr[gpuRows[g]];
        gpus[g].nnzEnd = l_rowPtr[gpuRows[g + 1]];
        gpus[g].threadsPerRow = chooseThreadsPerRow(gpus[g].nnz(), gpus[g].rows());
    }

#pragma omp parallel num_threads(numGpus)
    {
        GpuWork& g = gpus[omp_get_thread_num()];
        CUDA_CHECK(cudaSetDevice(g.device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking));
        const size_t rows = g.rows();
        const size_t nnz = g.nnz();
        CUDA_CHECK(cudaMalloc(&g.d_val, std::max<size_t>(nnz, 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&g.d_cols, std::max<size_t>(nnz, 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&g.d_rowPtr, (rows + 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&g.d_vec, std::max<size_t>(numRows, 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&g.d_out, std::max<size_t>(rows, 1) * sizeof(double)));

        // Row pointers relative to this GPU's first non-zero
        std::vector<index_t> rp(rows + 1);
        for (size_t r = 0; r <= rows; ++r) rp[r] = l_rowPtr[g.rowBegin + r] - g.nnzBegin;

        if (nnz > 0) {
            CUDA_CHECK(cudaMemcpyAsync(g.d_val, l_val.data() + g.nnzBegin, nnz * sizeof(double), cudaMemcpyHostToDevice, g.stream));
            CUDA_CHECK(cudaMemcpyAsync(g.d_cols, l_cols.data() + g.nnzBegin, nnz * sizeof(index_t), cudaMemcpyHostToDevice, g.stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(g.d_rowPtr, rp.data(), (rows + 1) * sizeof(index_t), cudaMemcpyHostToDevice, g.stream));
        if (numRows > 0) {
            CUDA_CHECK(cudaMemcpyAsync(g.d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice, g.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(g.stream));
    }

    // For validation, compute reference solution (on the host, rank 0)
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
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(numGpus)
    {
        GpuWork& g = gpus[omp_get_thread_num()];
        CUDA_CHECK(cudaSetDevice(g.device));
        for (index_t iter = 0; iter < iterations; ++iter) {
            launchSpmv(g);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(g.stream));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Report the slowest rank's time (the benchmark completes when all ranks do)
    long localMs = static_cast<long>(localDuration.count());
    long maxMs = localMs;
    MPI_CHECK(MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD));

    // Collect the result: device -> host slice -> rank 0
#pragma omp parallel num_threads(numGpus)
    {
        GpuWork& g = gpus[omp_get_thread_num()];
        CUDA_CHECK(cudaSetDevice(g.device));
        if (g.rows() > 0) {
            CUDA_CHECK(cudaMemcpyAsync(l_out.data() + g.rowBegin, g.d_out, g.rows() * sizeof(double), cudaMemcpyDeviceToHost, g.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(g.stream));
    }
    gatherSlices(l_out.data(), rankRowOffs, h_out.data(), rank, size, 102);

    int rc = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (maxMs / 1000.0) / 1e9;
        const double avgTime = maxMs / static_cast<double>(iterations);

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
                rc = 1;
            }
        }
    }
    MPI_CHECK(MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD));

    // Cleanup
#pragma omp parallel num_threads(numGpus)
    {
        GpuWork& g = gpus[omp_get_thread_num()];
        CUDA_CHECK(cudaSetDevice(g.device));
        CUDA_CHECK(cudaFree(g.d_val));
        CUDA_CHECK(cudaFree(g.d_cols));
        CUDA_CHECK(cudaFree(g.d_rowPtr));
        CUDA_CHECK(cudaFree(g.d_vec));
        CUDA_CHECK(cudaFree(g.d_out));
        CUDA_CHECK(cudaStreamDestroy(g.stream));
    }

    MPI_Finalize();
    return rc;
}
