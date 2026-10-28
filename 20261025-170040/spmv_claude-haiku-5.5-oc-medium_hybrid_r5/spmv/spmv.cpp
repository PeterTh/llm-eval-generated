// Hybrid MPI + OpenMP + CUDA sparse matrix-vector multiplication benchmark.
//
// Decomposition:
//   * MPI:    rows are block-partitioned across ranks. The dense vector is
//             replicated (broadcast once); each rank owns a contiguous block
//             of matrix rows and the matching block of the output. No
//             communication happens inside the timed iterations.
//   * CUDA:   each rank drives one GPU (localRank % deviceCount). The leading
//             rows of the local block are computed on the GPU using a sliced
//             ELLPACK layout (slices of 32 rows, column-major within a slice)
//             so that warp accesses to values and column indices are coalesced.
//   * OpenMP: the trailing rows of the local block are computed on host
//             threads concurrently with the GPU kernel.
//
// Every output row is accumulated in the same left-to-right order as the
// original serial code with separate (non-fused) multiply and add operations,
// on both the host and the device, so results are bit-identical regardless of
// which resource computed a row.

#include <mpi.h>
#include <omp.h>
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

// The trailing 1/CPU_ROW_DIVISOR of each rank's rows is computed by OpenMP host
// threads concurrently with the GPU; the leading rows are computed on the GPU.
constexpr index_t CPU_ROW_DIVISOR = 16;

// Minimum nonzeros per host thread for the CPU share. Keeps the OpenMP team
// small enough that fork/join overhead does not dominate a small CPU share.
constexpr uint64_t NNZ_PER_CPU_THREAD = 32768;

// Rows per sliced-ELL slice; equal to the warp size so slices map to warps.
constexpr index_t SLICE_HEIGHT = 32;

// Threads per CUDA block (a multiple of SLICE_HEIGHT).
constexpr unsigned int CUDA_BLOCK_SIZE = 128;

// ****************************************************************************
// Error handling helpers
// ****************************************************************************
#define MPI_CHECK(call)                                                        \
    do {                                                                       \
        const int err_ = (call);                                               \
        if (err_ != MPI_SUCCESS) {                                             \
            fprintf(stderr, "MPI error at %s:%d\n", __FILE__, __LINE__);       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t err_ = (call);                                       \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
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
//   Serial reference implementation of sparse matrix-vector multiplication
//   using CSR format. Used only for validation.
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
// Function: spmvCpuOmp
//
// Purpose:
//   OpenMP-parallel CSR SpMV over the row range [rowBegin, rowEnd). Each row is
//   accumulated sequentially, so the result matches the device kernel exactly.
//
// Arguments:
//   val, cols, rowDelimiters: local CSR arrays (rowDelimiters indexes val/cols)
//   vec: dense input vector
//   rowBegin, rowEnd: half-open row range to compute
//   out: output vector (indexed by row)
//   numThreads: OpenMP team size for this region
//
// ****************************************************************************
void spmvCpuOmp(const double* val, const index_t* cols, const index_t* rowDelimiters,
                const double* vec, const index_t rowBegin, const index_t rowEnd, double* out,
                const int numThreads) {
#pragma omp parallel for schedule(static) num_threads(numThreads)
    for (index_t i = rowBegin; i < rowEnd; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Sliced ELLPACK representation for the GPU
// ****************************************************************************
struct EllSlices {
    std::vector<double> val;        // slice-major, then column-major within slice
    std::vector<index_t> cols;      // same layout as val
    std::vector<uint64_t> sliceOff; // offset of each slice in val/cols (size numSlices+1)
    std::vector<index_t> rowLen;    // number of nonzeros in each row
};

// ****************************************************************************
// Function: buildEll
//
// Purpose:
//   Converts the first numRows rows of a CSR matrix into sliced ELLPACK.
//   Entry k of row r in slice s is stored at sliceOff[s] + k*SLICE_HEIGHT +
//   (r % SLICE_HEIGHT), so consecutive lanes of a warp access consecutive
//   memory. Padding slots are zero and are never read (rowLen bounds the loop).
//
// ****************************************************************************
EllSlices buildEll(const double* csrVal, const index_t* csrCols, const index_t* rowDelimiters,
                   const index_t numRows) {
    EllSlices e;
    const index_t numSlices = (numRows + SLICE_HEIGHT - 1) / SLICE_HEIGHT;

    e.rowLen.resize(numRows);
    for (index_t r = 0; r < numRows; ++r) {
        e.rowLen[r] = rowDelimiters[r + 1] - rowDelimiters[r];
    }

    e.sliceOff.assign(numSlices + 1, 0);
    for (index_t s = 0; s < numSlices; ++s) {
        const index_t first = s * SLICE_HEIGHT;
        const index_t last = std::min<index_t>(first + SLICE_HEIGHT, numRows);
        index_t width = 0;
        for (index_t r = first; r < last; ++r) {
            width = std::max(width, e.rowLen[r]);
        }
        e.sliceOff[s + 1] = e.sliceOff[s] + static_cast<uint64_t>(SLICE_HEIGHT) * width;
    }

    const uint64_t total = e.sliceOff[numSlices];
    e.val.assign(total, 0.0);
    e.cols.assign(total, 0);

#pragma omp parallel for schedule(dynamic, 4)
    for (index_t s = 0; s < numSlices; ++s) {
        const index_t first = s * SLICE_HEIGHT;
        const index_t last = std::min<index_t>(first + SLICE_HEIGHT, numRows);
        for (index_t r = first; r < last; ++r) {
            const index_t lane = r - first;
            const index_t begin = rowDelimiters[r];
            for (index_t k = 0; k < e.rowLen[r]; ++k) {
                const uint64_t dst = e.sliceOff[s] + static_cast<uint64_t>(k) * SLICE_HEIGHT + lane;
                e.val[dst] = csrVal[begin + k];
                e.cols[dst] = csrCols[begin + k];
            }
        }
    }
    return e;
}

// ****************************************************************************
// CUDA kernel: one thread per row over sliced ELLPACK data.
// Accumulation order per row matches the host kernel (k ascending). The
// intrinsics force round-to-nearest multiply and add with no contraction.
// ****************************************************************************
__global__ void spmvEllKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                              const uint64_t* __restrict__ sliceOff,
                              const index_t* __restrict__ rowLen,
                              const double* __restrict__ x, const index_t numRows,
                              double* __restrict__ y) {
    const index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows) {
        return;
    }
    const index_t lane = row % SLICE_HEIGHT;
    const uint64_t base = sliceOff[row / SLICE_HEIGHT] + lane;
    const index_t len = rowLen[row];

    double t = 0.0;
    for (index_t k = 0; k < len; ++k) {
        const uint64_t idx = base + static_cast<uint64_t>(k) * SLICE_HEIGHT;
        t = __dadd_rn(t, __dmul_rn(val[idx], __ldg(&x[cols[idx]])));
    }
    y[row] = t;
}

// ****************************************************************************
// Class: GpuEllSpmv
//
// Purpose:
//   Owns the device copy of a rank's GPU row block (matrix in sliced ELL, and
//   the replicated input vector) and the stream used for kernel launches.
//
// ****************************************************************************
class GpuEllSpmv {
  public:
    GpuEllSpmv(const EllSlices& ell, const std::vector<double>& x, const index_t rows)
        : rows_(rows) {
        CUDA_CHECK(cudaStreamCreate(&stream_));
        if (rows_ == 0) {
            return;
        }
        d_val_ = allocAndCopy(ell.val.data(), ell.val.size());
        d_cols_ = allocAndCopy(ell.cols.data(), ell.cols.size());
        d_sliceOff_ = allocAndCopy(ell.sliceOff.data(), ell.sliceOff.size());
        d_rowLen_ = allocAndCopy(ell.rowLen.data(), ell.rowLen.size());
        d_x_ = allocAndCopy(x.data(), x.size());
        CUDA_CHECK(cudaMalloc(&d_y_, std::max<size_t>(rows_, 1) * sizeof(double)));
    }

    ~GpuEllSpmv() {
        cudaFree(d_val_);
        cudaFree(d_cols_);
        cudaFree(d_sliceOff_);
        cudaFree(d_rowLen_);
        cudaFree(d_x_);
        cudaFree(d_y_);
        cudaStreamDestroy(stream_);
    }

    GpuEllSpmv(const GpuEllSpmv&) = delete;
    GpuEllSpmv& operator=(const GpuEllSpmv&) = delete;

    // Asynchronously enqueue one SpMV over the GPU row block.
    void launch() {
        if (rows_ == 0) {
            return;
        }
        const unsigned int blocks = (rows_ + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        spmvEllKernel<<<blocks, CUDA_BLOCK_SIZE, 0, stream_>>>(d_val_, d_cols_, d_sliceOff_,
                                                             d_rowLen_, d_x_, rows_, d_y_);
        CUDA_CHECK(cudaGetLastError());
    }

    // Block until all enqueued work has completed.
    void sync() {
        if (rows_ != 0) {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
    }

    // Copy the GPU row block result into dst[0..rows).
    void download(double* dst) {
        if (rows_ != 0) {
            CUDA_CHECK(cudaMemcpy(dst, d_y_, rows_ * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }

  private:
    template <typename T>
    static T* allocAndCopy(const T* src, const size_t n) {
        T* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(n, 1) * sizeof(T)));
        if (n != 0) {
            CUDA_CHECK(cudaMemcpy(d, src, n * sizeof(T), cudaMemcpyHostToDevice));
        }
        return d;
    }

    index_t rows_;
    cudaStream_t stream_ = nullptr;
    double* d_val_ = nullptr;
    index_t* d_cols_ = nullptr;
    uint64_t* d_sliceOff_ = nullptr;
    index_t* d_rowLen_ = nullptr;
    double* d_x_ = nullptr;
    double* d_y_ = nullptr;
};

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

// First row owned by rank r when dim rows are split into nranks contiguous blocks.
index_t rowBegin(const index_t dim, const int r, const int nranks) {
    return static_cast<index_t>((static_cast<uint64_t>(dim) * static_cast<uint64_t>(r)) /
                                static_cast<uint64_t>(nranks));
}

struct Options {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
};

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   Runs the distributed, hybrid SpMV benchmark. All ranks participate; rank 0
//   performs input generation, reporting, and validation.
//
// Returns:
//   Process exit status (identical on all ranks).
//
// ****************************************************************************
int runBenchmark(const Options& opt, const int rank, const int nranks) {
    const MPI_Comm comm = MPI_COMM_WORLD;
    const index_t numRows = opt.numRows;
    const index_t sparsity = opt.sparsity;
    const index_t iterations = opt.iterations;
    const double maxVal = opt.maxVal;
    const bool validate = opt.validate;

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    // One MPI rank per GPU: node-local rank selects the device.
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int localRank = 0;
    int localSize = 1;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_size(nodeComm, &localSize));

    // Split host cores among node-local ranks unless the user set threads.
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(comm, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Rank 0 generates the full problem, in the same RNG order as the serial code.
    std::vector<index_t> rowDelimiters(numRows + 1);
    std::vector<double> vec(numRows);
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<double> h_reference;
    std::vector<double> h_out;

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_out.resize(numRows);

        printf("Initializing data structures...\n");
        fill(vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), rowDelimiters.data(), nItems, numRows);

        // For validation, compute reference solution
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), rowDelimiters.data(),
                    vec.data(), numRows, h_reference.data());
        }
    }

    // Replicate the vector and row delimiters; row blocks are then scattered.
    MPI_CHECK(MPI_Bcast(rowDelimiters.data(), numRows + 1, MPI_UINT32_T, 0, comm));
    MPI_CHECK(MPI_Bcast(vec.data(), numRows, MPI_DOUBLE, 0, comm));

    std::vector<int> rowCounts(nranks), rowDispls(nranks);
    std::vector<int> nnzCounts(nranks), nnzDispls(nranks);
    for (int r = 0; r < nranks; ++r) {
        const index_t b = rowBegin(numRows, r, nranks);
        const index_t e = rowBegin(numRows, r + 1, nranks);
        rowCounts[r] = static_cast<int>(e - b);
        rowDispls[r] = static_cast<int>(b);
        nnzDispls[r] = static_cast<int>(rowDelimiters[b]);
        nnzCounts[r] = static_cast<int>(rowDelimiters[e] - rowDelimiters[b]);
    }

    const index_t nLocal = static_cast<index_t>(rowCounts[rank]);
    const index_t nnzBase = static_cast<index_t>(nnzDispls[rank]);
    const int nnzLocal = nnzCounts[rank];

    std::vector<double> l_val(nnzLocal);
    std::vector<index_t> l_cols(nnzLocal);
    MPI_CHECK(MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                           MPI_DOUBLE, l_val.data(), nnzLocal, MPI_DOUBLE, 0, comm));
    MPI_CHECK(MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                           MPI_UINT32_T, l_cols.data(), nnzLocal, MPI_UINT32_T, 0, comm));

    // Release the full matrix on rank 0 unless validation still needs it.
    if (rank == 0 && !validate) {
        std::vector<double>().swap(h_val);
        std::vector<index_t>().swap(h_cols);
    }

    // Local CSR row delimiters, relative to this rank's first nonzero.
    std::vector<index_t> l_rd(nLocal + 1);
    for (index_t i = 0; i <= nLocal; ++i) {
        l_rd[i] = rowDelimiters[rowDispls[rank] + i] - nnzBase;
    }

    // Leading rows -> GPU, trailing rows -> OpenMP host threads.
    const index_t gpuRows = nLocal - nLocal / CPU_ROW_DIVISOR;
    const EllSlices ell = buildEll(l_val.data(), l_cols.data(), l_rd.data(), gpuRows);
    GpuEllSpmv gpu(ell, vec, gpuRows);

    std::vector<double> l_out(nLocal);

    // Size the host team to the CPU share's work, capped by available threads.
    const uint64_t cpuNnz = static_cast<uint64_t>(l_rd[nLocal]) - l_rd[gpuRows];
    const int cpuThreads = static_cast<int>(std::clamp<uint64_t>(
        cpuNnz / NNZ_PER_CPU_THREAD, 1, static_cast<uint64_t>(omp_get_max_threads())));

    auto runOnce = [&]() {
        gpu.launch();
        spmvCpuOmp(l_val.data(), l_cols.data(), l_rd.data(), vec.data(), gpuRows, nLocal,
                   l_out.data(), cpuThreads);
        gpu.sync();
    };

    // Untimed warm-up: loads the kernel and initializes device state.
    if (iterations > 0) {
        runOnce();
    }

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_CHECK(MPI_Barrier(comm));
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        runOnce();
    }

    auto end = std::chrono::high_resolution_clock::now();
    const long long localNs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    long long maxNs = 0;
    MPI_CHECK(MPI_Reduce(&localNs, &maxNs, 1, MPI_LONG_LONG, MPI_MAX, 0, comm));

    // Collect the GPU block result, then gather the full output on rank 0.
    if (iterations > 0) {
        gpu.download(l_out.data());
    }
    MPI_CHECK(MPI_Gatherv(l_out.data(), static_cast<int>(nLocal), MPI_DOUBLE,
                          rank == 0 ? h_out.data() : nullptr, rowCounts.data(), rowDispls.data(),
                          MPI_DOUBLE, 0, comm));

    int status = 0;
    if (rank == 0) {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::nanoseconds(maxNs));
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (opt.printResults) {
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
            }
            status = valid ? 0 : 1;
        }
    }

    // Keep the exit status identical on every rank.
    if (validate) {
        MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, comm));
    }

    MPI_CHECK(MPI_Comm_free(&nodeComm));
    return status;
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank = 0;
    int nranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    Options opt;
    bool help = false;
    const char* unknownOpt = nullptr;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            opt.numRows = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            opt.sparsity = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            opt.iterations = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            opt.maxVal = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            opt.validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            opt.printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            unknownOpt = argv[i];
            break;
        }
    }

    int status = 0;
    if (unknownOpt != nullptr) {
        if (rank == 0) {
            printf("Unknown option: %s\n", unknownOpt);
            printUsage(argv[0]);
        }
        status = 1;
    } else if (help) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
    } else {
        status = runBenchmark(opt, rank, nranks);
    }

    MPI_CHECK(MPI_Finalize());
    return status;
}
