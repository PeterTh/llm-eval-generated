#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>
#include <sched.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Allocator that leaves elements uninitialized on value-construction. This lets
// us control the NUMA first-touch placement of the large arrays: pages are
// touched in parallel (below) with the same static partitioning used by the
// SpMV kernel, instead of being zeroed serially by std::vector's constructor.
template <typename T>
struct default_init_allocator : std::allocator<T> {
    using std::allocator<T>::allocator;
    template <typename U>
    struct rebind { using other = default_init_allocator<U>; };
    template <typename U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible<U>::value) {
        ::new (static_cast<void*>(p)) U;
    }
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        std::allocator_traits<std::allocator<T>>::construct(
            static_cast<std::allocator<T>&>(*this), p, std::forward<Args>(args)...);
    }
};

template <typename T>
using numa_vector = std::vector<T, default_init_allocator<T>>;

// Minimum number of non-zeros per thread; below this, the per-region barrier
// cost outweighs the work, so the number of threads is reduced accordingly.
constexpr index_t MIN_NNZ_PER_THREAD = 8192;

// ****************************************************************************
// Function: spmvThreadCount
//
// Purpose:
//   Number of OpenMP threads used for the SpMV kernel and for the matching
//   NUMA first-touch of its arrays, scaled with the amount of work.
//
// ****************************************************************************
int spmvThreadCount(const index_t nnz) {
    const int maxThreads = omp_get_max_threads();
    const index_t workThreads = nnz / MIN_NNZ_PER_THREAD;
    if (workThreads < 1) return 1;
    if (workThreads < static_cast<index_t>(maxThreads)) return static_cast<int>(workThreads);
    return maxThreads;
}

// ****************************************************************************
// Function: pinThreads
//
// Purpose:
//   Pin each OpenMP worker thread to a distinct CPU from the process's allowed
//   CPU set. Unbound threads on a fully occupied machine get migrated by the
//   OS scheduler, which turns each barrier into a multi-millisecond stall.
//   Pinning also keeps the thread->row mapping (and therefore the NUMA
//   first-touch placement) stable across parallel regions. Skipped when the
//   user has already requested a binding policy via OMP_PROC_BIND/OMP_PLACES.
//
// ****************************************************************************
void pinThreads() {
    if (omp_get_proc_bind() != omp_proc_bind_false) return;

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
    const int numAllowed = CPU_COUNT(&allowed);
    if (numAllowed <= 0) return;

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int target = tid % numAllowed;  // tid-th allowed CPU
        int seen = 0;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &allowed)) continue;
            if (seen == target) {
                cpu_set_t one;
                CPU_ZERO(&one);
                CPU_SET(cpu, &one);
                sched_setaffinity(0, sizeof(one), &one);
                break;
            }
            ++seen;
        }
    }
}

// ****************************************************************************
// Function: firstTouch
//
// Purpose:
//   Touch (zero) an uninitialized array in parallel using a static schedule so
//   that each page is physically allocated on the NUMA node of the thread that
//   will later process that range in the parallel SpMV kernel.
//
// ****************************************************************************
template <typename T>
void firstTouch(T* A, const index_t n, const int numThreads) {
#pragma omp parallel for schedule(static) num_threads(numThreads)
    for (index_t i = 0; i < n; ++i) {
        A[i] = T{};
    }
}

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
    const int numThreads = spmvThreadCount(rowDelimiters[dim]);
    // Rows are independent; the random matrix has a near-uniform number of
    // non-zeros per row, so a static schedule gives balanced work, no
    // scheduling overhead, and a stable row->thread mapping that matches the
    // NUMA first-touch placement of val/cols/out. Accumulation order within a
    // row is unchanged, so results are bitwise identical to the serial code.
#pragma omp parallel for schedule(static) num_threads(numThreads)
    for (index_t i = 0; i < dim; ++i) {
        const index_t rowStart = rowDelimiters[i];
        const index_t rowEnd = rowDelimiters[i + 1];
        double t = 0.0;
        for (index_t j = rowStart; j < rowEnd; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[i] = t;
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
    numa_vector<double> h_val(nItems);                  // Non-zero values
    numa_vector<index_t> h_cols(nItems);                // Column indices
    numa_vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    numa_vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out(numRows);                 // Output vector

    printf("Initializing data structures...\n");
    pinThreads();
    const int numThreads = spmvThreadCount(nItems);
    printf("OpenMP threads: %d (max %d)\n", numThreads, omp_get_max_threads());

    // NUMA-aware first touch (parallel, same partitioning as the SpMV kernel),
    // before the deterministic serial fill.
    firstTouch(h_val.data(), nItems, numThreads);
    firstTouch(h_cols.data(), nItems, numThreads);
    firstTouch(h_rowDelimiters.data(), numRows + 1, numThreads);
    firstTouch(h_vec.data(), numRows, numThreads);
    firstTouch(h_out.data(), numRows, numThreads);

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

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_out.data());
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
