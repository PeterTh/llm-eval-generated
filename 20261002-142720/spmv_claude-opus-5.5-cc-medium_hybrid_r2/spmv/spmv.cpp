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
    // Rows are independent; each row keeps the sequential summation order
#pragma omp parallel for schedule(dynamic, 64)
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
// CUDA / MPI helpers
// ****************************************************************************
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// ****************************************************************************
// Kernel: spmvKernel
//
// Purpose:
//   CSR SpMV where a group of TPR threads cooperates on one row. Groups of up
//   to a warp reduce via shuffles; larger groups (up to a full block) finish
//   the reduction through shared memory.
// ****************************************************************************
template <int TPR, int BLOCK>
__global__ void __launch_bounds__(BLOCK)
spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
           const index_t* __restrict__ rowDelimiters, const double* __restrict__ vec,
           const index_t rows, double* __restrict__ out) {
    static_assert(BLOCK % TPR == 0, "BLOCK must be a multiple of TPR");
    constexpr int ROWS_PER_BLOCK = BLOCK / TPR;
    constexpr int WARP_GROUP = TPR < 32 ? TPR : 32;

    const index_t row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / TPR;
    const int lane = threadIdx.x % TPR;

    double sum = 0.0;
    if (row < rows) {
        const index_t begin = rowDelimiters[row];
        const index_t end = rowDelimiters[row + 1];
        for (index_t j = begin + lane; j < end; j += TPR) {
            sum += val[j] * __ldg(&vec[cols[j]]);
        }
    }

#pragma unroll
    for (int offset = WARP_GROUP / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset, WARP_GROUP);
    }

    if constexpr (TPR > 32) {
        constexpr int WARPS_PER_ROW = TPR / 32;
        __shared__ double partial[BLOCK / 32];
        const int warp = threadIdx.x / 32;
        if ((threadIdx.x & 31) == 0) partial[warp] = sum;
        __syncthreads();
        if (lane == 0) {
            sum = 0.0;
#pragma unroll
            for (int w = 0; w < WARPS_PER_ROW; ++w) sum += partial[warp + w];
        }
    }

    if (lane == 0 && row < rows) {
        out[row] = sum;
    }
}

template <int TPR, int BLOCK>
static void launchSpmv(const double* val, const index_t* cols, const index_t* rowDelimiters,
                       const double* vec, index_t rows, double* out, cudaStream_t stream) {
    constexpr int ROWS_PER_BLOCK = BLOCK / TPR;
    const unsigned grid = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    spmvKernel<TPR, BLOCK><<<grid, BLOCK, 0, stream>>>(val, cols, rowDelimiters, vec, rows, out);
}

// Pick the number of threads per row from the average row length
static void spmvGpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
                    const double* vec, index_t rows, double avgNnzPerRow, double* out,
                    cudaStream_t stream) {
    if (rows == 0) return;
    if (avgNnzPerRow <= 4)         launchSpmv<2, 128>(val, cols, rowDelimiters, vec, rows, out, stream);
    else if (avgNnzPerRow <= 8)    launchSpmv<4, 128>(val, cols, rowDelimiters, vec, rows, out, stream);
    else if (avgNnzPerRow <= 16)   launchSpmv<8, 128>(val, cols, rowDelimiters, vec, rows, out, stream);
    else if (avgNnzPerRow <= 32)   launchSpmv<16, 128>(val, cols, rowDelimiters, vec, rows, out, stream);
    else if (avgNnzPerRow <= 128)  launchSpmv<32, 128>(val, cols, rowDelimiters, vec, rows, out, stream);
    else                           launchSpmv<64, 128>(val, cols, rowDelimiters, vec, rows, out, stream);
}

// Point-to-point transfer of arbitrarily large buffers (MPI counts are int)
constexpr size_t MPI_CHUNK_BYTES = size_t(1) << 30;

template <typename T>
static void sendLarge(const T* buf, size_t count, int dest, int tag) {
    const char* p = reinterpret_cast<const char*>(buf);
    size_t bytes = count * sizeof(T);
    while (bytes > 0) {
        const size_t n = std::min(bytes, MPI_CHUNK_BYTES);
        MPI_Send(p, static_cast<int>(n), MPI_BYTE, dest, tag, MPI_COMM_WORLD);
        p += n;
        bytes -= n;
    }
}

template <typename T>
static void recvLarge(T* buf, size_t count, int src, int tag) {
    char* p = reinterpret_cast<char*>(buf);
    size_t bytes = count * sizeof(T);
    while (bytes > 0) {
        const size_t n = std::min(bytes, MPI_CHUNK_BYTES);
        MPI_Recv(p, static_cast<int>(n), MPI_BYTE, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        p += n;
        bytes -= n;
    }
}

// Per-GPU state of this rank
struct GpuWorker {
    int device = 0;
    index_t rowBegin = 0;  // global first row
    index_t rows = 0;      // number of rows
    double avgNnz = 0.0;
    cudaStream_t stream = nullptr;
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    cudaGraphExec_t batch = nullptr;  // GRAPH_BATCH back-to-back SpMV launches
};

// Iterations captured per CUDA graph (amortizes kernel launch overhead)
constexpr index_t GRAPH_BATCH = 32;

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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
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

    // ------------------------------------------------------------------------
    // GPU assignment: GPUs of a node are shared among the node-local ranks.
    // A rank owning several GPUs drives each with its own OpenMP thread.
    // ------------------------------------------------------------------------
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices < 1) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> myDevices;
    if (localSize >= numDevices) {
        myDevices.push_back(localRank % numDevices);
    } else {
        for (int d = localRank; d < numDevices; d += localSize) myDevices.push_back(d);
    }
    const int myGpus = static_cast<int>(myDevices.size());

    // Global list of workers (GPUs) in rank order
    std::vector<int> gpusPerRank(nranks);
    MPI_Allgather(&myGpus, 1, MPI_INT, gpusPerRank.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> firstWorker(nranks + 1, 0);
    for (int r = 0; r < nranks; ++r) firstWorker[r + 1] = firstWorker[r] + gpusPerRank[r];
    const int numWorkers = firstWorker[nranks];

    // ------------------------------------------------------------------------
    // Data generation on the root (the rand() sequence is inherently serial)
    // ------------------------------------------------------------------------
    std::vector<double> h_val;            // Non-zero values
    std::vector<index_t> h_cols;          // Column indices
    std::vector<index_t> h_rowDelimiters; // Row delimiters
    std::vector<double> h_vec(numRows);   // Dense vector
    std::vector<double> h_out;            // Output vector (root only)
    std::vector<index_t> workerRowBounds(numWorkers + 1, 0);

    if (root) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_out.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        // Split rows into contiguous blocks with balanced (nnz + rows) work
        const double totalWork = static_cast<double>(nItems) + numRows;
        workerRowBounds[0] = 0;
        workerRowBounds[numWorkers] = numRows;
        index_t lo = 0;
        for (int w = 1; w < numWorkers; ++w) {
            const double target = totalWork * w / numWorkers;
            index_t a = lo, b = numRows;
            while (a < b) {
                const index_t mid = a + (b - a) / 2;
                if (static_cast<double>(h_rowDelimiters[mid]) + mid < target) a = mid + 1;
                else b = mid;
            }
            workerRowBounds[w] = a;
            lo = a;
        }
    }

    MPI_Bcast(workerRowBounds.data(), numWorkers + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ------------------------------------------------------------------------
    // Distribute each rank's contiguous row block
    // ------------------------------------------------------------------------
    const index_t myRowBegin = workerRowBounds[firstWorker[rank]];
    const index_t myRowEnd = workerRowBounds[firstWorker[rank + 1]];
    const index_t myRows = myRowEnd - myRowBegin;

    std::vector<index_t> l_rowDelimitersBuf;
    std::vector<index_t> l_colsBuf;
    std::vector<double> l_valBuf;
    const index_t* l_rowDelimiters;  // global nnz offsets for rows [myRowBegin, myRowEnd]
    const index_t* l_cols;           // indexed by (global offset - l_nnzBase)
    const double* l_val;
    index_t l_nnzBase;

    if (root) {
        for (int r = 1; r < nranks; ++r) {
            const index_t rb = workerRowBounds[firstWorker[r]];
            const index_t re = workerRowBounds[firstWorker[r + 1]];
            const index_t nb = h_rowDelimiters[rb];
            const size_t nnz = static_cast<size_t>(h_rowDelimiters[re]) - nb;
            sendLarge(h_rowDelimiters.data() + rb, static_cast<size_t>(re - rb) + 1, r, 0);
            sendLarge(h_cols.data() + nb, nnz, r, 1);
            sendLarge(h_val.data() + nb, nnz, r, 2);
        }
        l_rowDelimiters = h_rowDelimiters.data() + myRowBegin;
        l_nnzBase = h_rowDelimiters[myRowBegin];
        l_cols = h_cols.data() + l_nnzBase;
        l_val = h_val.data() + l_nnzBase;
    } else {
        l_rowDelimitersBuf.resize(static_cast<size_t>(myRows) + 1);
        recvLarge(l_rowDelimitersBuf.data(), l_rowDelimitersBuf.size(), 0, 0);
        l_nnzBase = l_rowDelimitersBuf[0];
        const size_t nnz = static_cast<size_t>(l_rowDelimitersBuf[myRows]) - l_nnzBase;
        l_colsBuf.resize(nnz);
        l_valBuf.resize(nnz);
        recvLarge(l_colsBuf.data(), nnz, 0, 1);
        recvLarge(l_valBuf.data(), nnz, 0, 2);
        l_rowDelimiters = l_rowDelimitersBuf.data();
        l_cols = l_colsBuf.data();
        l_val = l_valBuf.data();
    }

    // ------------------------------------------------------------------------
    // Upload per-GPU partitions (one OpenMP thread per GPU)
    // ------------------------------------------------------------------------
    std::vector<GpuWorker> workers(myGpus);
    std::vector<double> l_out(std::max<index_t>(myRows, 1));
    CUDA_CHECK(cudaHostRegister(l_out.data(), l_out.size() * sizeof(double), cudaHostRegisterDefault));

#pragma omp parallel for num_threads(myGpus) schedule(static, 1)
    for (int g = 0; g < myGpus; ++g) {
        GpuWorker& w = workers[g];
        w.device = myDevices[g];
        w.rowBegin = workerRowBounds[firstWorker[rank] + g];
        w.rows = workerRowBounds[firstWorker[rank] + g + 1] - w.rowBegin;
        const index_t lr = w.rowBegin - myRowBegin;  // first row within the rank block
        const index_t nb = l_rowDelimiters[lr];
        const size_t nnz = static_cast<size_t>(l_rowDelimiters[lr + w.rows]) - nb;
        w.avgNnz = w.rows ? static_cast<double>(nnz) / w.rows : 0.0;

        // Row delimiters rebased to the start of this GPU's nnz range
        std::vector<index_t> rp(static_cast<size_t>(w.rows) + 1);
        for (index_t i = 0; i <= w.rows; ++i) rp[i] = l_rowDelimiters[lr + i] - nb;

        CUDA_CHECK(cudaSetDevice(w.device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&w.d_val, std::max<size_t>(nnz, 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&w.d_cols, std::max<size_t>(nnz, 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&w.d_rowDelimiters, rp.size() * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&w.d_vec, std::max<size_t>(numRows, 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&w.d_out, std::max<size_t>(w.rows, 1) * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(w.d_val, l_val + (nb - l_nnzBase), nnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w.d_cols, l_cols + (nb - l_nnzBase), nnz * sizeof(index_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w.d_rowDelimiters, rp.data(), rp.size() * sizeof(index_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w.d_vec, h_vec.data(), static_cast<size_t>(numRows) * sizeof(double), cudaMemcpyHostToDevice));

        // Warm-up launch (module load, clocks); result is overwritten later
        spmvGpu(w.d_val, w.d_cols, w.d_rowDelimiters, w.d_vec, w.rows, w.avgNnz, w.d_out, w.stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(w.stream));

        if (w.rows > 0 && iterations >= GRAPH_BATCH) {
            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamBeginCapture(w.stream, cudaStreamCaptureModeThreadLocal));
            for (index_t k = 0; k < GRAPH_BATCH; ++k) {
                spmvGpu(w.d_val, w.d_cols, w.d_rowDelimiters, w.d_vec, w.rows, w.avgNnz, w.d_out, w.stream);
            }
            CUDA_CHECK(cudaStreamEndCapture(w.stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&w.batch, graph, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            CUDA_CHECK(cudaGraphUpload(w.batch, w.stream));
            CUDA_CHECK(cudaStreamSynchronize(w.stream));
        }
    }

    // Non-root ranks no longer need their host copies
    l_rowDelimitersBuf = {};
    l_colsBuf = {};
    l_valBuf = {};

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && root) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Gather layout: each rank returns its contiguous block of output rows
    std::vector<int> recvCounts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        displs[r] = static_cast<int>(workerRowBounds[firstWorker[r]]);
        recvCounts[r] = static_cast<int>(workerRowBounds[firstWorker[r + 1]] - workerRowBounds[firstWorker[r]]);
    }

    // Perform SpMV computation
    if (root) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(myGpus) schedule(static, 1)
    for (int g = 0; g < myGpus; ++g) {
        GpuWorker& w = workers[g];
        CUDA_CHECK(cudaSetDevice(w.device));
        index_t iter = 0;
        if (w.batch) {
            for (; iter + GRAPH_BATCH <= iterations; iter += GRAPH_BATCH) {
                CUDA_CHECK(cudaGraphLaunch(w.batch, w.stream));
            }
        }
        for (; iter < iterations; ++iter) {
            spmvGpu(w.d_val, w.d_cols, w.d_rowDelimiters, w.d_vec, w.rows, w.avgNnz, w.d_out, w.stream);
        }
        CUDA_CHECK(cudaGetLastError());
        if (w.rows > 0) {
            CUDA_CHECK(cudaMemcpyAsync(l_out.data() + (w.rowBegin - myRowBegin), w.d_out,
                                       static_cast<size_t>(w.rows) * sizeof(double),
                                       cudaMemcpyDeviceToHost, w.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(w.stream));
    }

    MPI_Gatherv(l_out.data(), static_cast<int>(myRows), MPI_DOUBLE,
                root ? h_out.data() : nullptr, recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Release GPU resources
    for (GpuWorker& w : workers) {
        CUDA_CHECK(cudaSetDevice(w.device));
        cudaFree(w.d_val);
        cudaFree(w.d_cols);
        cudaFree(w.d_rowDelimiters);
        cudaFree(w.d_vec);
        cudaFree(w.d_out);
        if (w.batch) cudaGraphExecDestroy(w.batch);
        cudaStreamDestroy(w.stream);
    }
    cudaHostUnregister(l_out.data());

    int exitCode = 0;
    if (root) {
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

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
