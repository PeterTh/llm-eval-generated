#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <utility>
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
    // Rows are independent; when called from within a parallel region the
    // work is shared across the team, otherwise this runs on a single thread.
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

// ****************************************************************************
// Function: physicalCoreRepresentatives
//
// Purpose:
//   Returns one logical CPU id per physical core (from sysfs topology),
//   ordered by (package, core). SpMV is memory-bandwidth-bound, so running
//   more than one thread per physical core only adds contention. Returns an
//   empty vector if the topology cannot be determined.
//
// ****************************************************************************
std::vector<int> physicalCoreRepresentatives() {
    std::vector<int> reps;
    std::set<std::pair<int, int>> seen;
    for (int cpu = 0;; ++cpu) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        FILE* f = fopen(path, "r");
        if (!f) {
            break;
        }
        int coreId = -1;
        const bool ok = fscanf(f, "%d", &coreId) == 1;
        fclose(f);
        if (!ok) {
            return {};
        }
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        int pkg = 0;
        f = fopen(path, "r");
        if (f) {
            if (fscanf(f, "%d", &pkg) != 1) {
                pkg = 0;
            }
            fclose(f);
        }
        if (seen.insert({pkg, coreId}).second) {
            reps.push_back(cpu);
        }
    }
    return reps;
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

    // Pick the team size. SpMV is memory-bound, so SMT oversubscription hurts:
    // default to one thread per physical core. Also cap the team so each
    // thread has enough nonzeros to amortize scheduling/barrier overhead on
    // small problems. An explicit OMP_NUM_THREADS wins.
    const std::vector<int> coreCpus = physicalCoreRepresentatives();
    int numThreads = omp_get_max_threads();
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        if (!coreCpus.empty() && static_cast<int>(coreCpus.size()) < numThreads) {
            numThreads = static_cast<int>(coreCpus.size());
        }
        constexpr index_t minNnzPerThread = 8192;
        const index_t workLimit = nItems / minNnzPerThread;
        if (workLimit < static_cast<index_t>(numThreads)) {
            numThreads = workLimit > 0 ? static_cast<int>(workLimit) : 1;
        }
    }
    omp_set_num_threads(numThreads);

    // Pin each thread to its own physical core, spread across packages, so
    // threads stay next to the data they first-touch below. Skipped when the
    // user asked for their own binding or requested more threads than cores.
    if (getenv("OMP_PROC_BIND") == nullptr && !coreCpus.empty() &&
        numThreads <= static_cast<int>(coreCpus.size())) {
        const size_t nCores = coreCpus.size();
#pragma omp parallel
        {
            const size_t tid = static_cast<size_t>(omp_get_thread_num());
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(coreCpus[tid * nCores / static_cast<size_t>(numThreads)], &mask);
            sched_setaffinity(0, sizeof(mask), &mask);
        }
    }

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
    printf("Iterations: %u\n", iterations);
    printf("Max value: %.2f\n", maxVal);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate data structures. The large CSR arrays are left uninitialized
    // so their pages can be first-touched in parallel below for NUMA locality.
    std::unique_ptr<double[]> h_val(new double[nItems]);    // Non-zero values
    std::unique_ptr<index_t[]> h_cols(new index_t[nItems]); // Column indices
    std::vector<index_t> h_rowDelimiters(numRows + 1);      // Row delimiters
    std::vector<double> h_vec(numRows);                     // Dense vector
    std::vector<double> h_out(numRows);                     // Output vector

    // First-touch the CSR arrays with the same static partitioning the
    // compute loop uses (nonzeros are distributed uniformly across rows),
    // so each thread's slice ends up on its local NUMA node.
#pragma omp parallel for schedule(static)
    for (index_t k = 0; k < nItems; ++k) {
        h_val[k] = 0.0;
        h_cols[k] = 0;
    }

    printf("Initializing data structures...\n");
    fill(h_vec.data(), numRows, maxVal);
    fill(h_val.get(), nItems, maxVal);
    initRandomMatrix(h_cols.get(), h_rowDelimiters.data(), nItems, numRows);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.get(), h_cols.get(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // Keep one parallel region alive across all iterations to avoid
    // per-iteration fork/join overhead; spmvCpu's "omp for" (with its
    // implicit barrier) distributes the rows and separates iterations.
#pragma omp parallel
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.get(), h_cols.get(), h_rowDelimiters.data(),
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
