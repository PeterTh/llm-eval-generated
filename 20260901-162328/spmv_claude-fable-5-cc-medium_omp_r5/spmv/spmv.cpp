#include <omp.h>
#ifdef __linux__
#include <sched.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <utility>
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
    // Rows are independent; nonzeros are near-uniformly distributed across rows,
    // so a static schedule balances load with minimal scheduling overhead. The
    // static partition also matches the first-touch initialization in main, so
    // each thread reads and writes memory on its own NUMA node.
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

#ifdef __linux__
// Reads one integer topology attribute for a CPU from sysfs; -1 on failure.
static int readTopologyId(const int cpu, const char* attribute) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, attribute);
    FILE* f = fopen(path, "r");
    if (!f) {
        return -1;
    }
    int id = -1;
    if (fscanf(f, "%d", &id) != 1) {
        id = -1;
    }
    fclose(f);
    return id;
}
#endif

// ****************************************************************************
// Function: pinThreads
//
// Purpose:
//   Pins each OpenMP worker thread to one CPU out of those the process is
//   allowed to run on. Without pinning, threads migrate across NUMA nodes
//   and lose the locality established by first-touch page placement. CPUs
//   are ordered so that threads occupy distinct physical cores (alternating
//   between packages to use the memory bandwidth of every NUMA node) before
//   any SMT siblings are used. Skipped when the user already requested a
//   binding policy via the standard environment variables.
//
// ****************************************************************************
void pinThreads() {
#ifdef __linux__
    if (getenv("OMP_PROC_BIND") || getenv("OMP_PLACES") || getenv("GOMP_CPU_AFFINITY")) {
        return;
    }
    cpu_set_t procMask;
    if (sched_getaffinity(0, sizeof(procMask), &procMask) != 0) {
        return;
    }
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &procMask)) {
            cpus.push_back(c);
        }
    }
    if (cpus.empty()) {
        return;
    }

    // Group allowed CPUs into SMT sibling lists per (package, core).
    std::map<std::pair<int, int>, std::vector<int>> coreCpus;
    bool haveTopology = true;
    for (const int c : cpus) {
        const int pkg = readTopologyId(c, "physical_package_id");
        const int core = readTopologyId(c, "core_id");
        if (pkg < 0 || core < 0) {
            haveTopology = false;
            break;
        }
        coreCpus[{pkg, core}].push_back(c);
    }

    std::vector<int> order;
    if (haveTopology) {
        std::map<int, std::vector<std::vector<int>>> pkgCores;
        size_t maxCores = 0;
        size_t maxSiblings = 0;
        for (const auto& [key, siblings] : coreCpus) {
            pkgCores[key.first].push_back(siblings);
            maxSiblings = std::max(maxSiblings, siblings.size());
        }
        for (const auto& [pkg, cores] : pkgCores) {
            maxCores = std::max(maxCores, cores.size());
        }
        // Emit one CPU per physical core, filling a whole package before
        // moving to the next (small teams stay on one package and avoid
        // cross-socket traffic on shared data); repeat for each further SMT
        // sibling only after all physical cores are listed.
        for (size_t s = 0; s < maxSiblings; ++s) {
            for (const auto& [pkg, cores] : pkgCores) {
                for (size_t k = 0; k < maxCores; ++k) {
                    if (k < cores.size() && s < cores[k].size()) {
                        order.push_back(cores[k][s]);
                    }
                }
            }
        }
    } else {
        order = cpus;
    }

#pragma omp parallel
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(order[tid % order.size()], &mask);
        sched_setaffinity(0, sizeof(mask), &mask);
    }
#endif
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

    // Allocate data structures. The large arrays use default-initialized (untouched)
    // storage so pages can be first-touched in parallel below, distributing them
    // across NUMA nodes to match the static partition of the SpMV loop.
    std::unique_ptr<double[]> h_val(new double[nItems]);        // Non-zero values
    std::unique_ptr<index_t[]> h_cols(new index_t[nItems]);     // Column indices
    std::vector<index_t> h_rowDelimiters(numRows + 1);          // Row delimiters
    std::vector<double> h_vec(numRows);                         // Dense vector
    std::unique_ptr<double[]> h_out(new double[numRows]);       // Output vector

    // First-touch pages with the same schedule/binding as the compute loop.
    pinThreads();
#pragma omp parallel
    {
#pragma omp for schedule(static) nowait
        for (index_t i = 0; i < nItems; ++i) {
            h_val[i] = 0.0;
            h_cols[i] = 0;
        }
#pragma omp for schedule(static)
        for (index_t i = 0; i < numRows; ++i) {
            h_out[i] = 0.0;
        }
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

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.get(), h_cols.get(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_out.get());
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
