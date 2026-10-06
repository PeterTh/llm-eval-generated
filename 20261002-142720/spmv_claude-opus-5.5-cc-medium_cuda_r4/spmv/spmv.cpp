#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(err_));                          \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// ****************************************************************************
// Class: GlibcRand
//
// Purpose:
//   Inlined, lock-free replica of glibc's rand() (TYPE_3 additive feedback
//   generator). It yields exactly the same sequence as srand(seed)/rand() on
//   glibc, but much faster. Its equivalence with the C library's rand() is
//   verified at runtime (see fastRandMatchesLibc); if it does not match, the
//   program falls back to the library rand().
//
// ****************************************************************************
class GlibcRand {
  public:
    explicit GlibcRand(unsigned int seed) {
        // srandom_r(): fill the table with a Park-Miller LCG
        uint32_t r[DEG];
        int32_t word = static_cast<int32_t>(seed == 0 ? 1 : seed);
        r[0] = static_cast<uint32_t>(word);
        for (int i = 1; i < DEG; ++i) {
            const int32_t hi = word / 127773;
            const int32_t lo = word % 127773;
            word = 16807 * lo - 2836 * hi;
            if (word < 0) word += 2147483647;
            r[i] = static_cast<uint32_t>(word);
        }
        // Linearize the ring buffer (front pointer at index SEP, rear at 0):
        // hist_[k] holds x[t-31+k], and x[t] = x[t-31] + x[t-3].
        for (int k = 0; k < DEG; ++k) hist_[k] = r[(SEP + k) % DEG];
        // srandom_r() discards the first 10*DEG outputs
        int discard[10 * DEG];
        generate(discard, 10 * DEG);
    }

    // Produce the next 'count' rand() outputs
    void generate(int* out, size_t count) {
        constexpr size_t BLOCK = 4096;
        uint32_t buf[DEG + BLOCK];
        while (count > 0) {
            const size_t m = std::min(count, BLOCK);
            for (int k = 0; k < DEG; ++k) buf[k] = hist_[k];
            // Keep the three most recent values in registers (lag SEP = 3)
            uint32_t a = buf[DEG - 3], b = buf[DEG - 2], c = buf[DEG - 1];
            size_t i = 0;
            for (; i + 3 <= m; i += 3) {
                a += buf[i];
                b += buf[i + 1];
                c += buf[i + 2];
                buf[i + DEG] = a;
                buf[i + DEG + 1] = b;
                buf[i + DEG + 2] = c;
                out[i] = static_cast<int>(a >> 1);
                out[i + 1] = static_cast<int>(b >> 1);
                out[i + 2] = static_cast<int>(c >> 1);
            }
            for (; i < m; ++i) {
                const uint32_t x = buf[i] + buf[i + DEG - SEP];
                buf[i + DEG] = x;
                out[i] = static_cast<int>(x >> 1);
            }
            for (int k = 0; k < DEG; ++k) hist_[k] = buf[m + k];
            out += m;
            count -= m;
        }
    }

    inline int next() {
        int v;
        generate(&v, 1);
        return v;
    }

  private:
    static constexpr int DEG = 31;
    static constexpr int SEP = 3;
    uint32_t hist_[DEG];
};

static bool fastRandMatchesLibc() {
    if (RAND_MAX != 2147483647) return false;
    const unsigned int seeds[] = {1u, 8675309u};
    for (unsigned int seed : seeds) {
        srand(seed);
        GlibcRand g(seed);
        for (int i = 0; i < 4096; ++i) {
            if (g.next() != rand()) return false;
        }
    }
    return true;
}

// Fast path: identical to fill() below, using an inlined generator.
void fillFast(GlibcRand& rng, double* A, const index_t n, const double maxVal) {
    constexpr index_t BLOCK = 4096;
    int buf[BLOCK];
    for (index_t i0 = 0; i0 < n; i0 += BLOCK) {
        const index_t m = std::min(BLOCK, n - i0);
        rng.generate(buf, m);
        for (index_t k = 0; k < m; ++k) {
            A[i0 + k] = maxVal * (buf[k] / (static_cast<double>(RAND_MAX) + 1.0));
        }
    }
}

// Fast path: identical to initRandomMatrix() below, using an inlined generator.
// The test "rand()/RAND_MAX <= prob" is replaced by the exactly equivalent
// integer test "rand() <= threshold" (the double division is monotonic).
// Rows in which neither the "n values cap" nor the "fill remaining" rule can
// take effect are processed with a branch-free loop.
void initRandomMatrixFast(index_t* cols, index_t* rowDelimiters, const index_t n,
                          const index_t dim) {
    index_t nnzAssigned = 0;
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Largest k with k / RAND_MAX <= prob (or -1 if none)
    int64_t lo = -1, hi = RAND_MAX;
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo + 1) / 2;
        if (static_cast<double>(mid) / RAND_MAX <= prob) lo = mid; else hi = mid - 1;
    }
    const int64_t threshold = lo;

    GlibcRand rng(8675309);
    std::vector<int> rowRand(dim);
    // The fast row path relies on dim*dim not overflowing index_t
    const bool noOverflow = static_cast<uint64_t>(dim) * dim <= UINT32_MAX;

    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        if (fillRemaining) {
            // All remaining entries are assigned; random values are irrelevant
            for (index_t j = 0; j < dim; ++j) cols[nnzAssigned++] = j;
            continue;
        }
        rng.generate(rowRand.data(), dim);
        const uint64_t minEntriesLeft = static_cast<uint64_t>(dim) * dim -
                                        static_cast<uint64_t>(i) * dim - dim + 1;
        const bool safeRow = noOverflow &&
                             static_cast<uint64_t>(nnzAssigned) + dim <= n &&
                             minEntriesLeft > static_cast<uint64_t>(n - nnzAssigned);
        if (safeRow) {
            index_t nnz = nnzAssigned;
            for (index_t j = 0; j < dim; ++j) {
                cols[nnz] = j;
                nnz += (rowRand[j] <= threshold) ? 1 : 0;
            }
            nnzAssigned = nnz;
            continue;
        }
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            if ((nnzAssigned < n && rowRand[j] <= threshold) || fillRemaining) {
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    rowDelimiters[dim] = n;
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
//   CSR SpMV on the GPU. Each row is processed by a group of VS consecutive
//   threads (VS is a power of two <= 32) that read the row's values/columns
//   in a coalesced fashion and combine their partial sums with warp shuffles.
//   rowDelimiters holds global offsets; 'base' is the global offset of the
//   first non-zero stored on this device.
//
// ****************************************************************************
template <int VS>
__global__ void __launch_bounds__(256)
spmvCsrVectorKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                    const index_t* __restrict__ rowDelimiters, const double* __restrict__ vec,
                    const index_t numRows, const index_t base, double* __restrict__ out) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = tid / VS;
    const unsigned lane = static_cast<unsigned>(tid) & (VS - 1);

    double t = 0.0;
    if (row < numRows) {
        const index_t start = __ldg(rowDelimiters + row) - base;
        const index_t end = __ldg(rowDelimiters + row + 1) - base;
        // Align the start to a VS boundary for better coalescing
        index_t j = (start & ~static_cast<index_t>(VS - 1)) + lane;
        if (j >= start && j < end) {
            t += __ldcs(val + j) * __ldg(vec + __ldcs(cols + j));
        }
        j += VS;
#pragma unroll 4
        for (; j < end; j += VS) {
            t += __ldcs(val + j) * __ldg(vec + __ldcs(cols + j));
        }
    }
#pragma unroll
    for (int offset = VS / 2; offset > 0; offset >>= 1) {
        t += __shfl_xor_sync(0xffffffffu, t, offset);
    }
    if (row < numRows && lane == 0) {
        out[row] = t;
    }
}

// Per-GPU slice of the CSR matrix
struct DeviceSlice {
    int device = 0;
    cudaStream_t stream = nullptr;
    index_t rowBegin = 0, rowEnd = 0;  // rows [rowBegin, rowEnd)
    index_t nnzBegin = 0, nnzEnd = 0;  // non-zeros [nnzBegin, nnzEnd)
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
};

template <int VS>
static void launchSpmv(const DeviceSlice& s) {
    const index_t rows = s.rowEnd - s.rowBegin;
    if (rows == 0) return;
    constexpr int threads = 256;
    const size_t totalThreads = static_cast<size_t>(rows) * VS;
    const unsigned blocks = static_cast<unsigned>((totalThreads + threads - 1) / threads);
    spmvCsrVectorKernel<VS><<<blocks, threads, 0, s.stream>>>(
        s.d_val, s.d_cols, s.d_rowDelimiters, s.d_vec, rows, s.nnzBegin, s.d_out);
}

static void launchSpmvVS(const DeviceSlice& s, int vs) {
    switch (vs) {
        case 1: launchSpmv<1>(s); break;
        case 2: launchSpmv<2>(s); break;
        case 4: launchSpmv<4>(s); break;
        case 8: launchSpmv<8>(s); break;
        case 16: launchSpmv<16>(s); break;
        default: launchSpmv<32>(s); break;
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
    printf("Iterations: %u\n", iterations);
    printf("Max value: %.2f\n", maxVal);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate and initialize data structures
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out(numRows);                 // Output vector

    // Initialize the CUDA driver and create the contexts in the background
    // while the host generates the input data. Rows are later partitioned
    // across the selected GPUs by non-zero count.
    constexpr size_t MIN_NNZ_PER_DEVICE = size_t(1) << 22;
    int numDevices = 0;
    std::thread cudaInitThread([&numDevices, nItems, numRows]() {
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount < 1) {
            fprintf(stderr, "No CUDA device found\n");
            exit(EXIT_FAILURE);
        }
        int n = static_cast<int>(std::min<size_t>(
            static_cast<size_t>(deviceCount),
            std::max<size_t>(1, static_cast<size_t>(nItems) / MIN_NNZ_PER_DEVICE)));
        n = std::max(1, std::min<int>(n, std::max<index_t>(numRows, 1)));
        for (int d = n - 1; d >= 0; --d) {
            CUDA_CHECK(cudaSetDevice(d));
            CUDA_CHECK(cudaFree(nullptr));
        }
        numDevices = n;
    });

    printf("Initializing data structures...\n");
    if (fastRandMatchesLibc()) {
        // rand() is unseeded at this point in the original, i.e. seed 1
        GlibcRand rng(1);
        fillFast(rng, h_vec.data(), numRows, maxVal);
        fillFast(rng, h_val.data(), nItems, maxVal);
        initRandomMatrixFast(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    } else {
        srand(1);  // equivalent to the unseeded initial state
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    cudaInitThread.join();

    // Threads per row: power of two near the average row length (max one warp)
    const double avgRowLen = numRows ? static_cast<double>(nItems) / numRows : 0.0;
    int vs = 1;
    while (vs < 32 && vs * 2 <= avgRowLen) vs *= 2;

    std::vector<DeviceSlice> slices(numDevices);
    for (int d = 0; d < numDevices; ++d) {
        DeviceSlice& s = slices[d];
        s.device = d;
        const uint64_t target = static_cast<uint64_t>(nItems) * d / numDevices;
        s.rowBegin = (d == 0) ? 0
            : static_cast<index_t>(std::lower_bound(h_rowDelimiters.begin(),
                                                    h_rowDelimiters.begin() + numRows,
                                                    static_cast<index_t>(target)) -
                                   h_rowDelimiters.begin());
    }
    for (int d = 0; d < numDevices; ++d) {
        DeviceSlice& s = slices[d];
        s.rowEnd = (d + 1 < numDevices) ? slices[d + 1].rowBegin : numRows;
    }
    // Upload each slice to its GPU (one host thread per GPU, then warm up)
    auto setupDevice = [&](DeviceSlice& s) {
        s.nnzBegin = h_rowDelimiters[s.rowBegin];
        s.nnzEnd = h_rowDelimiters[s.rowEnd];
        const index_t rows = s.rowEnd - s.rowBegin;
        const size_t nnz = s.nnzEnd - s.nnzBegin;

        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&s.d_val, std::max<size_t>(nnz, 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.d_cols, std::max<size_t>(nnz, 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&s.d_rowDelimiters, (static_cast<size_t>(rows) + 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&s.d_vec, std::max<size_t>(numRows, 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.d_out, std::max<size_t>(rows, 1) * sizeof(double)));
        CUDA_CHECK(cudaMemcpyAsync(s.d_val, h_val.data() + s.nnzBegin, nnz * sizeof(double),
                                   cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_cols, h_cols.data() + s.nnzBegin, nnz * sizeof(index_t),
                                   cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_rowDelimiters, h_rowDelimiters.data() + s.rowBegin,
                                   (static_cast<size_t>(rows) + 1) * sizeof(index_t),
                                   cudaMemcpyHostToDevice, s.stream));
        CUDA_CHECK(cudaMemcpyAsync(s.d_vec, h_vec.data(), static_cast<size_t>(numRows) * sizeof(double),
                                   cudaMemcpyHostToDevice, s.stream));
        // Warm up (loads the kernel module); does not change the results
        launchSpmvVS(s, vs);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
    };
    if (numDevices == 1) {
        setupDevice(slices[0]);
    } else {
        std::vector<std::thread> setupThreads;
        for (auto& s : slices) setupThreads.emplace_back(setupDevice, std::ref(s));
        for (auto& t : setupThreads) t.join();
    }
    CUDA_CHECK(cudaSetDevice(slices[0].device));

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        for (auto& s : slices) {
            if (numDevices > 1) CUDA_CHECK(cudaSetDevice(s.device));
            launchSpmvVS(s, vs);
        }
    }
    for (auto& s : slices) {
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaGetLastError());
        if (iterations > 0) {
            CUDA_CHECK(cudaMemcpyAsync(h_out.data() + s.rowBegin, s.d_out,
                                       static_cast<size_t>(s.rowEnd - s.rowBegin) * sizeof(double),
                                       cudaMemcpyDeviceToHost, s.stream));
        }
    }
    for (auto& s : slices) {
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
    const double avgTime = duration.count() / static_cast<double>(iterations);
    
    printf("Average time per iteration: %.3f ms\n", avgTime);
    printf("Performance: %.3f GFLOPS\n", gflops);

    for (auto& s : slices) {
        CUDA_CHECK(cudaSetDevice(s.device));
        cudaFree(s.d_val);
        cudaFree(s.d_cols);
        cudaFree(s.d_rowDelimiters);
        cudaFree(s.d_vec);
        cudaFree(s.d_out);
        cudaStreamDestroy(s.stream);
    }
    
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
