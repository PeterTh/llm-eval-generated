#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>
#include <pthread.h>
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
// Uses an orphaned OpenMP worksharing loop: when called from within a parallel
// region the rows are distributed across the team; when called outside one it
// executes with a team of a single thread.
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
#pragma omp for schedule(static)
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

    // Allocate data structures (default-initialized so no pages are touched yet)
    std::unique_ptr<double[]> h_val(new double[nItems]);                 // Non-zero values
    std::unique_ptr<index_t[]> h_cols(new index_t[nItems]);              // Column indices
    std::unique_ptr<index_t[]> h_rowDelimiters(new index_t[numRows + 1]);  // Row delimiters
    std::unique_ptr<double[]> h_vec(new double[numRows]);                // Dense vector
    std::unique_ptr<double[]> h_out(new double[numRows]);                // Output vector

    // Size the thread team to the amount of work so small problems are not
    // dominated by fork/join and barrier overhead; capped by the OpenMP
    // runtime's thread limit (e.g. OMP_NUM_THREADS).
    const index_t nnzPerThread = 2048;
    int numThreads = static_cast<int>(nItems / nnzPerThread);
    if (numThreads < 1) numThreads = 1;
    if (numThreads > omp_get_max_threads()) numThreads = omp_get_max_threads();

    // Determine the CPUs this process may run on, for explicit thread pinning.
    // libgomp disables affinity entirely unless OMP_PROC_BIND/OMP_PLACES is
    // set, and without pinning threads migrate across NUMA nodes and lose the
    // locality established by the first-touch pass below.
    cpu_set_t processMask;
    std::vector<int> allowedCpus;
    if (sched_getaffinity(0, sizeof(processMask), &processMask) == 0) {
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (CPU_ISSET(c, &processMask)) allowedCpus.push_back(c);
        }
    }

    // Parallel first-touch so pages are distributed across NUMA nodes,
    // matching the static partitioning used by the compute loop. The
    // subsequent serial initialization overwrites the values but does not
    // move the pages. Threads are pinned here first so the pages stay local
    // to the threads for the rest of the run (the OpenMP runtime reuses the
    // same pool threads for subsequent same-sized teams).
#pragma omp parallel num_threads(numThreads)
    {
        if (!allowedCpus.empty()) {
            const int tid = omp_get_thread_num();
            cpu_set_t threadMask;
            CPU_ZERO(&threadMask);
            CPU_SET(allowedCpus[static_cast<size_t>(tid) % allowedCpus.size()], &threadMask);
            pthread_setaffinity_np(pthread_self(), sizeof(threadMask), &threadMask);
        }
#pragma omp barrier
#pragma omp for schedule(static) nowait
        for (index_t i = 0; i < nItems; ++i) {
            h_val[i] = 0.0;
            h_cols[i] = 0;
        }
#pragma omp for schedule(static) nowait
        for (index_t i = 0; i < numRows; ++i) {
            h_vec[i] = 0.0;
            h_out[i] = 0.0;
            h_rowDelimiters[i] = 0;
        }
    }
    h_rowDelimiters[numRows] = 0;

    printf("Initializing data structures...\n");
    fill(h_vec.get(), numRows, maxVal);
    fill(h_val.get(), nItems, maxVal);
    initRandomMatrix(h_cols.get(), h_rowDelimiters.get(), nItems, numRows);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.get(), h_cols.get(), h_rowDelimiters.get(), 
                h_vec.get(), numRows, h_reference.data());
    }

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // Single parallel region for all iterations to avoid per-iteration
    // fork/join overhead; the worksharing loop inside spmvCpu distributes
    // rows across the team, with an implicit barrier between iterations.
#pragma omp parallel num_threads(numThreads)
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.get(), h_cols.get(), h_rowDelimiters.get(),
                h_vec.get(), numRows, h_out.get());
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
        print_results(std::vector<double>(h_out.get(), h_out.get() + numRows), "OutputVector");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), h_out.get(), numRows);

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
