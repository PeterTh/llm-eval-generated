#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#include <sched.h>

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
// Function: bindThreadsToCores
//
// Purpose:
//   Pins each OpenMP thread to a distinct hardware thread, so that the NUMA
//   placement established by first touch stays valid for the whole run.
//   The pinning is restricted to the CPUs the process is already allowed to
//   run on (e.g. when started under taskset) and is skipped entirely if the
//   OpenMP runtime has been given an explicit affinity policy.
//
// ****************************************************************************
void bindThreadsToCores() {
    if (getenv("OMP_PROC_BIND") != nullptr || getenv("OMP_PLACES") != nullptr ||
        getenv("GOMP_CPU_AFFINITY") != nullptr) {
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
    if (cpus.empty()) {
        return;
    }

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        // Spread threads evenly over the available CPUs
        const size_t slot = (static_cast<size_t>(tid) * cpus.size()) /
                            static_cast<size_t>(nthreads > 0 ? nthreads : 1);
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpus[slot < cpus.size() ? slot : cpus.size() - 1], &mask);
        sched_setaffinity(0, sizeof(mask), &mask);
    }
}

// ****************************************************************************
// Function: threadRowRange
//
// Purpose:
//   Computes the half-open range of rows [begin, end) handled by the calling
//   thread. Rows are split such that every thread gets (nearly) the same
//   number of non-zeros, which both balances the load and keeps the val/cols
//   stream accesses contiguous per thread.
//
//   The partitioning depends only on the row delimiters and the thread count,
//   so it is identical for the first-touch pass and for every SpMV call.
//
// ****************************************************************************
static inline index_t rowSplitPoint(const index_t* rowDelimiters, const index_t dim,
                                    const int t, const int nthreads) {
    if (t <= 0) return 0;
    if (t >= nthreads) return dim;

    const uint64_t nnz = rowDelimiters[dim];
    const index_t target = static_cast<index_t>((nnz * static_cast<uint64_t>(t)) /
                                                static_cast<uint64_t>(nthreads));
    // First row r with rowDelimiters[r] >= target
    index_t lo = 0;
    index_t hi = dim;
    while (lo < hi) {
        const index_t mid = lo + (hi - lo) / 2;
        if (rowDelimiters[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static inline void threadRowRange(const index_t* rowDelimiters, const index_t dim,
                                  index_t& begin, index_t& end) {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    begin = rowSplitPoint(rowDelimiters, dim, tid, nthreads);
    end = rowSplitPoint(rowDelimiters, dim, tid + 1, nthreads);
}

// ****************************************************************************
// Function: spmvRows
//
// Purpose:
//   Sequential SpMV over a row range; the per-row summation order is the same
//   as in the original serial code, so results are bit-identical.
//
// ****************************************************************************
static inline void spmvRows(const double* __restrict val, const index_t* __restrict cols,
                            const index_t* __restrict rowDelimiters,
                            const double* __restrict vec, const index_t begin,
                            const index_t end, double* __restrict out) {
    for (index_t i = begin; i < end; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
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
#pragma omp parallel
    {
        index_t begin, end;
        threadRowRange(rowDelimiters, dim, begin, end);
        spmvRows(val, cols, rowDelimiters, vec, begin, end, out);
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

    bindThreadsToCores();
    printf("OpenMP threads: %d\n", omp_get_max_threads());

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

    // NUMA-aware placement: replicate the matrix into buffers whose pages are
    // first touched by the very thread that will read them during the SpMV.
    const size_t valBytes = ((sizeof(double) * nItems + 63) / 64) * 64;
    const size_t colBytes = ((sizeof(index_t) * nItems + 63) / 64) * 64;
    const size_t outBytes = ((sizeof(double) * numRows + 63) / 64) * 64;
    auto* d_val = static_cast<double*>(aligned_alloc(64, valBytes));
    auto* d_cols = static_cast<index_t*>(aligned_alloc(64, colBytes));
    auto* d_out = static_cast<double*>(aligned_alloc(64, outBytes));
    if (d_val == nullptr || d_cols == nullptr || d_out == nullptr) {
        printf("Allocation failed\n");
        return 1;
    }

    const double* srcVal = h_val.data();
    const index_t* srcCols = h_cols.data();
    const index_t* rowDelimiters = h_rowDelimiters.data();
    const double* vec = h_vec.data();

#pragma omp parallel
    {
        index_t begin, end;
        threadRowRange(rowDelimiters, numRows, begin, end);
        for (index_t i = begin; i < end; ++i) {
            d_out[i] = 0.0;
        }
        for (index_t j = rowDelimiters[begin]; j < rowDelimiters[end]; ++j) {
            d_val[j] = srcVal[j];
            d_cols[j] = srcCols[j];
        }
    }

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
        index_t begin, end;
        threadRowRange(rowDelimiters, numRows, begin, end);
        for (index_t iter = 0; iter < iterations; ++iter) {
            spmvRows(d_val, d_cols, rowDelimiters, vec, begin, end, d_out);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::copy(d_out, d_out + numRows, h_out.begin());
    free(d_val);
    free(d_cols);
    free(d_out);
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
