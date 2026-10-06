#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

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
// Class: FastRand
//
// Purpose:
//   Lock-free reimplementation of the glibc rand() generator (TYPE_3 additive
//   feedback generator) producing the identical sequence for a given seed.
//   It is only used after verifying that it matches the C library's rand();
//   otherwise rand() itself is used.
//
// ****************************************************************************
class FastRand {
  public:
    explicit FastRand(unsigned int seed) {
        int32_t r[34];
        r[0] = static_cast<int32_t>(seed == 0 ? 1 : seed);
        for (int i = 1; i < 31; ++i) {
            const int64_t hi = r[i - 1] / 127773;
            const int64_t lo = r[i - 1] % 127773;
            int64_t word = 16807 * lo - 2836 * hi;
            if (word < 0) word += 2147483647;
            r[i] = static_cast<int32_t>(word);
        }
        for (int i = 31; i < 34; ++i) r[i] = r[i - 31];
        for (int i = 0; i < 34; ++i) ring[i] = static_cast<uint32_t>(r[i]);
        pos = 0;  // ring[pos] holds r[k-34] for the next value r[k]
        for (int i = 34; i < 344; ++i) advance();
    }

    inline int next() { return static_cast<int>(advance() >> 1); }

  private:
    uint32_t ring[34];
    unsigned pos;

    // r[k] = r[k-31] + r[k-3]
    inline uint32_t advance() {
        unsigned i31 = pos + 3;
        if (i31 >= 34) i31 -= 34;
        unsigned i3 = pos + 31;
        if (i3 >= 34) i3 -= 34;
        const uint32_t v = ring[i31] + ring[i3];
        ring[pos] = v;
        if (++pos == 34) pos = 0;
        return v;
    }
};

// Returns true if FastRand reproduces the C library rand() sequence.
// Leaves the C library generator freshly seeded with 'seed'.
static bool fastRandMatchesLibc(unsigned int seed) {
    FastRand fr(seed);
    srand(seed);
    bool match = true;
    for (int i = 0; i < 4096 && match; ++i) {
        match = (fr.next() == rand());
    }
    srand(seed);
    return match;
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
    constexpr unsigned int seed = 8675309;
    const bool useFast = fastRandMatchesLibc(seed);  // also performs srand(seed)
    FastRand fastRand(seed);

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
            const int r = useFast ? fastRand.next() : rand();
            double randVal = static_cast<double>(r) / RAND_MAX;
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
// Function: spmvKernel
//
// Purpose:
//   CSR SpMV on the GPU. Each row is processed by a group of TPR consecutive
//   threads (TPR = 1..32, a power of two chosen from the average row length);
//   partial sums are combined with warp shuffles. Rows longer than
//   longRowThreshold are skipped here and handled by spmvLongRowKernel.
//
// ****************************************************************************
template <int TPR>
__global__ void __launch_bounds__(256)
spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
           const index_t* __restrict__ rowDelimiters, const double* __restrict__ vec,
           const index_t dim, const index_t longRowThreshold, double* __restrict__ out) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & (TPR - 1);
    const index_t row = tid / TPR;
    if (row >= dim) return;  // whole TPR groups exit together

    const index_t start = __ldg(&rowDelimiters[row]);
    const index_t end = __ldg(&rowDelimiters[row + 1]);
    if (end - start > longRowThreshold) return;

    double t = 0.0;
    for (index_t j = start + lane; j < end; j += TPR) {
        t += __ldg(&val[j]) * __ldg(&vec[__ldg(&cols[j])]);
    }
    if constexpr (TPR > 1) {
        const unsigned mask = (TPR == 32) ? 0xffffffffu
                                          : (((1u << TPR) - 1u) << (threadIdx.x & 31 & ~(TPR - 1)));
#pragma unroll
        for (int off = TPR / 2; off > 0; off >>= 1) {
            t += __shfl_down_sync(mask, t, off, TPR);
        }
    }
    if (lane == 0) out[row] = t;
}

// ****************************************************************************
// Function: spmvLongRowKernel
//
// Purpose:
//   Processes long rows (e.g. the densely filled tail of the generated
//   matrix) with one thread block per row.
//
// ****************************************************************************
template <int BS>
__global__ void __launch_bounds__(BS)
spmvLongRowKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                  const index_t* __restrict__ rowDelimiters, const double* __restrict__ vec,
                  const index_t* __restrict__ rowList, double* __restrict__ out) {
    __shared__ double warpSums[BS / 32];
    const index_t row = rowList[blockIdx.x];
    const index_t start = rowDelimiters[row];
    const index_t end = rowDelimiters[row + 1];

    double t = 0.0;
    for (index_t j = start + threadIdx.x; j < end; j += BS) {
        t += __ldg(&val[j]) * __ldg(&vec[__ldg(&cols[j])]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
    if ((threadIdx.x & 31) == 0) warpSums[threadIdx.x >> 5] = t;
    __syncthreads();
    if (threadIdx.x < 32) {
        t = threadIdx.x < BS / 32 ? warpSums[threadIdx.x] : 0.0;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
        if (threadIdx.x == 0) out[row] = t;
    }
}

// Launches the row-group kernel with the requested threads-per-row.
static void launchSpmv(int tpr, const double* val, const index_t* cols, const index_t* rowDelimiters,
                       const double* vec, index_t dim, index_t longRowThreshold, double* out,
                       cudaStream_t stream) {
    constexpr int BS = 256;
    const unsigned rowsPerBlock = BS / tpr;
    const unsigned blocks = static_cast<unsigned>((static_cast<uint64_t>(dim) + rowsPerBlock - 1) / rowsPerBlock);
    if (blocks == 0) return;
#define SPMV_CASE(T)                                                                                   \
    case T:                                                                                            \
        spmvKernel<T><<<blocks, BS, 0, stream>>>(val, cols, rowDelimiters, vec, dim, longRowThreshold, out); \
        break;
    switch (tpr) {
        SPMV_CASE(1)
        SPMV_CASE(2)
        SPMV_CASE(4)
        SPMV_CASE(8)
        SPMV_CASE(16)
        default:
        SPMV_CASE(32)
    }
#undef SPMV_CASE
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

    printf("Initializing data structures...\n");
    fill(h_vec.data(), numRows, maxVal);
    fill(h_val.data(), nItems, maxVal);
    initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Choose the work decomposition: threads per row from the average row length,
    // and a list of exceptionally long rows that get a whole block each.
    const double avgRowLen = numRows > 0 ? static_cast<double>(nItems) / numRows : 0.0;
    int tpr = 1;
    while (tpr < 32 && tpr * 2 <= avgRowLen) tpr *= 2;
    constexpr int LONG_BS = 256;
    const index_t longRowThreshold = std::max<index_t>(static_cast<index_t>(tpr) * 32, 1024);
    std::vector<index_t> h_longRows;
    for (index_t i = 0; i < numRows; ++i) {
        if (h_rowDelimiters[i + 1] - h_rowDelimiters[i] > longRowThreshold) h_longRows.push_back(i);
    }

    // Allocate device memory and upload data
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rowDelimiters = nullptr, *d_longRows = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val, std::max<size_t>(nItems, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, std::max<size_t>(nItems, 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (static_cast<size_t>(numRows) + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, std::max<size_t>(numRows, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max<size_t>(numRows, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_longRows, std::max<size_t>(h_longRows.size(), 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), static_cast<size_t>(nItems) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), static_cast<size_t>(nItems) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(), (static_cast<size_t>(numRows) + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), static_cast<size_t>(numRows) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_longRows, h_longRows.data(), h_longRows.size() * sizeof(index_t), cudaMemcpyHostToDevice));

    cudaStream_t mainStream, longStream;
    cudaEvent_t longDone;
    CUDA_CHECK(cudaStreamCreateWithFlags(&mainStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&longStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&longDone, cudaEventDisableTiming));

    // Each SpMV: short rows on the main stream, long rows concurrently on a second stream
    auto runSpmv = [&]() {
        launchSpmv(tpr, d_val, d_cols, d_rowDelimiters, d_vec, numRows, longRowThreshold, d_out, mainStream);
        if (!h_longRows.empty()) {
            spmvLongRowKernel<LONG_BS><<<static_cast<unsigned>(h_longRows.size()), LONG_BS, 0, longStream>>>(
                d_val, d_cols, d_rowDelimiters, d_vec, d_longRows, d_out);
        }
    };

    // Untimed warm-up (module load, caches)
    runSpmv();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        runSpmv();
    }
    if (!h_longRows.empty()) {
        CUDA_CHECK(cudaEventRecord(longDone, longStream));
        CUDA_CHECK(cudaStreamWaitEvent(mainStream, longDone, 0));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_out.data(), d_out, static_cast<size_t>(numRows) * sizeof(double),
                               cudaMemcpyDeviceToHost, mainStream));
    CUDA_CHECK(cudaStreamSynchronize(mainStream));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
    const double avgTime = duration.count() / static_cast<double>(iterations);
    
    printf("Average time per iteration: %.3f ms\n", avgTime);
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    CUDA_CHECK(cudaEventDestroy(longDone));
    CUDA_CHECK(cudaStreamDestroy(longStream));
    CUDA_CHECK(cudaStreamDestroy(mainStream));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_longRows));

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
