#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#endif

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// Granularity used for NUMA-aware first touch; matches the transparent huge page size
constexpr size_t TOUCH_GRANULARITY = 2 * 1024 * 1024;

// Row ranges assigned to each thread, balanced by number of non-zeroes.
// Shared by the first-touch, the reference and the timed kernel so that every
// thread always works on the rows whose data it faulted in.
static std::vector<index_t> g_rowStart;
static int g_numThreads = 1;

// ****************************************************************************
// Function: allocArray
//
// Purpose:
//   Allocates an uninitialized, huge-page friendly array. Memory is left
//   untouched so that the pages can be first-touched (and hence NUMA-placed)
//   by the thread that will later work on them.
//
// ****************************************************************************
template <typename T>
struct AlignedDeleter {
    void operator()(T* p) const { std::free(p); }
};

template <typename T>
using AlignedArray = std::unique_ptr<T[], AlignedDeleter<T>>;

template <typename T>
AlignedArray<T> allocArray(const size_t n) {
    const size_t bytes = std::max<size_t>(TOUCH_GRANULARITY, ((n * sizeof(T) + TOUCH_GRANULARITY - 1) / TOUCH_GRANULARITY) * TOUCH_GRANULARITY);
    void* p = std::aligned_alloc(TOUCH_GRANULARITY, bytes);
    if (p == nullptr) {
        printf("Failed to allocate %zu bytes\n", bytes);
        exit(1);
    }
#ifdef MADV_HUGEPAGE
    madvise(p, bytes, MADV_HUGEPAGE);
#endif
    return AlignedArray<T>(static_cast<T*>(p));
}

// ****************************************************************************
// Function: bindThreads
//
// Purpose:
//   Pins the OpenMP threads to distinct CPUs, spread over all CPUs the process
//   is allowed to run on. Stable placement is what makes the NUMA-aware first
//   touch below pay off. Skipped if the user expressed a placement policy or if
//   the run is oversubscribed.
//
// ****************************************************************************
void bindThreads(const int numThreads) {
#ifdef __linux__
    if (getenv("OMP_PROC_BIND") != nullptr || getenv("OMP_PLACES") != nullptr) {
        return;
    }
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return;
    }
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &allowed)) {
            cpus.push_back(c);
        }
    }
    if (static_cast<int>(cpus.size()) < numThreads) {
        return;
    }

#pragma omp parallel num_threads(numThreads)
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        const int cpu = cpus[(tid * cpus.size()) / static_cast<size_t>(numThreads)];
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu, &mask);
        pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
    }
#else
    (void)numThreads;
#endif
}

// ****************************************************************************
// Function: touchBlocked / touchInterleaved
//
// Purpose:
//   First-touch helpers. touchBlocked hands each thread the contiguous block of
//   elements it will later read (matching the non-zero balanced partitioning),
//   while touchInterleaved distributes pages round-robin over the threads, which
//   approximates a NUMA interleaved placement for the randomly accessed vector.
//
// ****************************************************************************
template <typename T>
void touchBlocked(T* data, const size_t n, const int numThreads) {
#pragma omp parallel num_threads(numThreads)
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        const size_t begin = (n * tid) / static_cast<size_t>(numThreads);
        const size_t end = (n * (tid + 1)) / static_cast<size_t>(numThreads);
        for (size_t i = begin; i < end; ++i) {
            data[i] = T{};
        }
    }
}

template <typename T>
void touchInterleaved(T* data, const size_t n, const int numThreads) {
    const size_t pageElems = TOUCH_GRANULARITY / sizeof(T);
    const size_t numPages = (n + pageElems - 1) / pageElems;
#pragma omp parallel for num_threads(numThreads) schedule(static, 1)
    for (size_t p = 0; p < numPages; ++p) {
        const size_t begin = p * pageElems;
        const size_t end = std::min(n, begin + pageElems);
        for (size_t i = begin; i < end; ++i) {
            data[i] = T{};
        }
    }
}

// ****************************************************************************
// Function: partitionRows
//
// Purpose:
//   Splits the rows into one contiguous range per thread such that every thread
//   receives (close to) the same number of non-zero elements. This keeps the
//   memory bound kernel balanced even when the row lengths vary.
//
// ****************************************************************************
void partitionRows(const index_t* rowDelimiters, const index_t dim, const index_t nnz, const int numThreads) {
    g_rowStart.assign(static_cast<size_t>(numThreads) + 1, dim);
    g_rowStart[0] = 0;
    for (int t = 1; t < numThreads; ++t) {
        const index_t target =
            static_cast<index_t>((static_cast<uint64_t>(nnz) * static_cast<uint64_t>(t)) / static_cast<uint64_t>(numThreads));
        const index_t* pos = std::lower_bound(rowDelimiters, rowDelimiters + dim + 1, target);
        g_rowStart[t] = static_cast<index_t>(pos - rowDelimiters);
    }
    g_rowStart[numThreads] = dim;
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
// Kernel for a single thread's row range. The accumulation order within a row is
// kept identical to the serial version, so results are bit-for-bit reproducible.
static inline void spmvRows(const double* __restrict val, const index_t* __restrict cols,
                            const index_t* __restrict rowDelimiters, const double* __restrict vec,
                            const index_t rowBegin, const index_t rowEnd, double* __restrict out) {
    for (index_t i = rowBegin; i < rowEnd; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
    (void)dim;
#pragma omp parallel num_threads(g_numThreads)
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        spmvRows(val, cols, rowDelimiters, vec, g_rowStart[tid], g_rowStart[tid + 1], out);
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

    g_numThreads = omp_get_max_threads();
    if (g_numThreads > static_cast<int>(numRows)) {
        g_numThreads = static_cast<int>(numRows);
    }
    if (g_numThreads < 1) {
        g_numThreads = 1;
    }
    printf("OpenMP threads: %d\n", g_numThreads);
    bindThreads(g_numThreads);

    // Allocate and initialize data structures
    auto h_val = allocArray<double>(nItems);                   // Non-zero values
    auto h_cols = allocArray<index_t>(nItems);                 // Column indices
    auto h_rowDelimiters = allocArray<index_t>(numRows + 1u);  // Row delimiters
    auto h_vec = allocArray<double>(numRows);                  // Dense vector
    auto h_out = allocArray<double>(numRows);                  // Output vector

    // NUMA-aware first touch: the streamed matrix arrays are distributed in the
    // same blocks the threads will process, while the randomly indexed dense
    // vector is spread page-wise over all threads.
    touchBlocked(h_val.get(), nItems, g_numThreads);
    touchBlocked(h_cols.get(), nItems, g_numThreads);
    touchBlocked(h_rowDelimiters.get(), static_cast<size_t>(numRows) + 1, g_numThreads);
    touchInterleaved(h_vec.get(), numRows, g_numThreads);

    printf("Initializing data structures...\n");
    fill(h_vec.get(), numRows, maxVal);
    fill(h_val.get(), nItems, maxVal);
    initRandomMatrix(h_cols.get(), h_rowDelimiters.get(), nItems, numRows);

    // Balance the rows over the threads by number of non-zeroes
    partitionRows(h_rowDelimiters.get(), numRows, nItems, g_numThreads);

    // First touch the output vector according to the final row partitioning
#pragma omp parallel num_threads(g_numThreads)
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        for (index_t i = g_rowStart[tid]; i < g_rowStart[tid + 1]; ++i) {
            h_out[i] = 0.0;
        }
    }

    // For validation, compute reference solution
    AlignedArray<double> h_reference;
    if (validate) {
        printf("Computing reference solution...\n");
        h_reference = allocArray<double>(numRows);
        touchBlocked(h_reference.get(), numRows, g_numThreads);
        spmvCpu(h_val.get(), h_cols.get(), h_rowDelimiters.get(),
                h_vec.get(), numRows, h_reference.get());
    }

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // A single parallel region spans all iterations so that fork/join overhead is
    // paid once instead of per iteration.
#pragma omp parallel num_threads(g_numThreads)
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        const index_t rowBegin = g_rowStart[tid];
        const index_t rowEnd = g_rowStart[tid + 1];
        for (index_t iter = 0; iter < iterations; ++iter) {
            spmvRows(h_val.get(), h_cols.get(), h_rowDelimiters.get(),
                     h_vec.get(), rowBegin, rowEnd, h_out.get());
#if defined(__GNUC__)
            // Keep every iteration observable, the inputs are loop invariant
            asm volatile("" ::: "memory");
#endif
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
    const double avgTime = duration.count() / static_cast<double>(iterations);
    
    printf("Average time per iteration: %.3f ms\n", avgTime);
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        const std::vector<double> results(h_out.get(), h_out.get() + numRows);
        print_results(results, "OutputVector");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.get(), h_out.get(), numRows);

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
