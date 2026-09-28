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

// Number of threads used for every parallel region. Fixed once at startup so that
// the row partitioning used for NUMA-aware first touch matches the one used during
// the actual computation.
static int gNumThreads = 1;

// ****************************************************************************
// Function: parseCpuList
//
// Purpose:
//   Parses a sysfs CPU list of the form "0-63,128-191" into the given vector.
//   Returns false if the file could not be read.
//
// ****************************************************************************
static bool parseCpuList(const char* path, std::vector<int>& cpus) {
    FILE* f = fopen(path, "r");
    if (f == nullptr) return false;
    char buf[4096];
    const bool ok = fgets(buf, sizeof(buf), f) != nullptr;
    fclose(f);
    if (!ok) return false;

    const char* p = buf;
    while (*p != '\0') {
        char* endp = nullptr;
        const long first = strtol(p, &endp, 10);
        if (endp == p) break;
        p = endp;
        long last = first;
        if (*p == '-') {
            last = strtol(p + 1, &endp, 10);
            p = endp;
        }
        for (long c = first; c <= last; ++c) {
            cpus.push_back(static_cast<int>(c));
        }
        if (*p == ',') ++p; else break;
    }
    return true;
}

// ****************************************************************************
// Function: pinThreads
//
// Purpose:
//   SpMV is memory bound, so both bandwidth and the NUMA-aware data placement
//   done by first touch depend on threads staying on the core they started on.
//   If the user did not request an OpenMP binding policy, threads are pinned
//   here: cores are handed out round robin over the NUMA nodes (to use the
//   bandwidth of all memory controllers) and SMT siblings are only used once
//   every physical core is taken.
//
//   Any explicit OMP_PROC_BIND setting is left untouched.
//
// ****************************************************************************
static void pinThreads(const int nthreads) {
    if (omp_get_proc_bind() != omp_proc_bind_false || getenv("OMP_PROC_BIND") != nullptr) return;

    cpu_set_t allowedSet;
    CPU_ZERO(&allowedSet);
    if (sched_getaffinity(0, sizeof(allowedSet), &allowedSet) != 0) return;

    std::vector<int> allowed;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &allowedSet)) allowed.push_back(c);
    }
    // Do not interfere when the process is oversubscribed
    if (allowed.empty() || nthreads > static_cast<int>(allowed.size())) return;

    // NUMA node of every allowed CPU (0 if the topology cannot be read)
    std::vector<int> nodeOf(CPU_SETSIZE, 0);
    int numNodes = 0;
    for (int n = 0; n < 1024; ++n) {
        char path[256];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", n);
        std::vector<int> nodeCpus;
        if (!parseCpuList(path, nodeCpus)) break;
        for (const int c : nodeCpus) {
            if (c >= 0 && c < CPU_SETSIZE) nodeOf[c] = n;
        }
        numNodes = n + 1;
    }
    if (numNodes == 0) numNodes = 1;

    // SMT rank of every allowed CPU: its position within the sibling list
    int maxRank = 0;
    std::vector<int> smtRankOf(CPU_SETSIZE, 0);
    for (const int c : allowed) {
        char path[256];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        std::vector<int> siblings;
        if (!parseCpuList(path, siblings)) continue;
        std::sort(siblings.begin(), siblings.end());
        const auto it = std::find(siblings.begin(), siblings.end(), c);
        if (it != siblings.end()) {
            smtRankOf[c] = static_cast<int>(it - siblings.begin());
            maxRank = std::max(maxRank, smtRankOf[c]);
        }
    }

    // Last level cache domain of every allowed CPU, identified by its lowest CPU.
    // Spreading over these domains gives access to all cache slices and memory
    // controllers even when fewer threads than cores are used.
    std::vector<int> domainOf(CPU_SETSIZE, -1);
    for (const int c : allowed) {
        for (int level = 3; level >= 0 && domainOf[c] < 0; --level) {
            char path[256];
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/shared_cpu_list", c, level);
            std::vector<int> shared;
            if (parseCpuList(path, shared) && !shared.empty()) {
                domainOf[c] = *std::min_element(shared.begin(), shared.end());
            }
        }
        if (domainOf[c] < 0) domainOf[c] = nodeOf[c];
    }

    // Domains grouped per NUMA node, in ascending order
    std::vector<std::vector<int>> domainsPerNode(numNodes);
    for (const int c : allowed) {
        auto& d = domainsPerNode[nodeOf[c]];
        if (std::find(d.begin(), d.end(), domainOf[c]) == d.end()) d.push_back(domainOf[c]);
    }
    for (auto& d : domainsPerNode) {
        std::sort(d.begin(), d.end());
    }

    // Visit the domains round robin over the NUMA nodes
    std::vector<int> domainOrder;
    for (size_t i = 0;; ++i) {
        bool any = false;
        for (int n = 0; n < numNodes; ++n) {
            if (i < domainsPerNode[n].size()) {
                domainOrder.push_back(domainsPerNode[n][i]);
                any = true;
            }
        }
        if (!any) break;
    }

    // Order: one CPU per domain at a time, physical cores before SMT siblings
    std::vector<int> order;
    order.reserve(allowed.size());
    for (int rank = 0; rank <= maxRank; ++rank) {
        std::vector<std::vector<int>> perDomain(domainOrder.size());
        for (const int c : allowed) {
            if (smtRankOf[c] != rank) continue;
            const auto it = std::find(domainOrder.begin(), domainOrder.end(), domainOf[c]);
            if (it == domainOrder.end()) continue;
            perDomain[static_cast<size_t>(it - domainOrder.begin())].push_back(c);
        }
        for (size_t i = 0;; ++i) {
            bool any = false;
            for (auto& cpus : perDomain) {
                if (i < cpus.size()) {
                    order.push_back(cpus[i]);
                    any = true;
                }
            }
            if (!any) break;
        }
    }
    if (order.size() < allowed.size()) order = allowed;

#pragma omp parallel num_threads(nthreads)
    {
        const int cpu = order[static_cast<size_t>(omp_get_thread_num())];
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
}

// ****************************************************************************
// Function: rowRangeForThread
//
// Purpose:
//   Computes the range of rows [begin, end) a thread is responsible for. Rows
//   are distributed such that every thread gets a (nearly) equal share of the
//   non-zero elements, which is what determines the amount of work / memory
//   traffic in SpMV. The partitioning is a pure function of the row delimiters
//   and the team size, so all parallel regions agree on it.
//
// Arguments:
//   rowDelimiters: array of size dim+1 holding indices to rows
//   dim:           number of rows in the matrix
//   tid:           thread id
//   nthreads:      size of the thread team
//   begin/end:     output - row range assigned to this thread
//
// ****************************************************************************
static inline void rowRangeForThread(const index_t* rowDelimiters, const index_t dim, const int tid,
                                     const int nthreads, index_t& begin, index_t& end) {
    const index_t nnz = rowDelimiters[dim];
    const auto split = [&](const int t) -> index_t {
        if (t == 0) return 0;
        if (t >= nthreads) return dim;
        // First row whose starting offset reaches this thread's share of non-zeroes
        const index_t target = static_cast<index_t>((static_cast<uint64_t>(nnz) * static_cast<uint64_t>(t)) /
                                                    static_cast<uint64_t>(nthreads));
        const index_t* pos = std::lower_bound(rowDelimiters, rowDelimiters + dim + 1, target);
        return static_cast<index_t>(pos - rowDelimiters);
    };
    begin = split(tid);
    end = split(tid + 1);
    if (end < begin) end = begin;
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
#pragma omp parallel num_threads(gNumThreads)
    {
        index_t rowBegin, rowEnd;
        rowRangeForThread(rowDelimiters, dim, omp_get_thread_num(), omp_get_num_threads(), rowBegin, rowEnd);

        // The accumulation order within a row is kept identical to the serial
        // version, so results are bit-for-bit reproducible.
        for (index_t i = rowBegin; i < rowEnd; ++i) {
            double t = 0.0;
            for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
                const auto col = cols[j];
                t += val[j] * vec[col];
            }
            out[i] = t;
        }
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

    gNumThreads = omp_get_max_threads();
    printf("OpenMP threads: %d\n", gNumThreads);
    pinThreads(gNumThreads);

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

    // Move the working set into memory that is first touched by the thread which
    // will later operate on it, so that the pages end up on the correct NUMA node.
    const auto allocLocal = [](const size_t bytes) -> void* {
        const size_t rounded = std::max<size_t>(64, ((bytes + 63) / 64) * 64);
        void* p = std::aligned_alloc(64, rounded);
        if (p == nullptr) {
            printf("Allocation of %zu bytes failed\n", rounded);
            exit(1);
        }
        return p;
    };

    auto* d_val = static_cast<double*>(allocLocal(sizeof(double) * nItems));
    auto* d_cols = static_cast<index_t*>(allocLocal(sizeof(index_t) * nItems));
    auto* d_rowDelimiters = static_cast<index_t*>(allocLocal(sizeof(index_t) * (numRows + 1)));
    auto* d_vec = static_cast<double*>(allocLocal(sizeof(double) * numRows));
    auto* d_out = static_cast<double*>(allocLocal(sizeof(double) * numRows));

#pragma omp parallel num_threads(gNumThreads)
    {
        index_t rowBegin, rowEnd;
        rowRangeForThread(h_rowDelimiters.data(), numRows, omp_get_thread_num(), omp_get_num_threads(),
                          rowBegin, rowEnd);

        for (index_t i = rowBegin; i < rowEnd; ++i) {
            d_rowDelimiters[i] = h_rowDelimiters[i];
            d_vec[i] = h_vec[i];
            d_out[i] = 0.0;
        }
        for (index_t j = h_rowDelimiters[rowBegin]; j < h_rowDelimiters[rowEnd]; ++j) {
            d_val[j] = h_val[j];
            d_cols[j] = h_cols[j];
        }
    }
    d_rowDelimiters[numRows] = nItems;

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(d_val, d_cols, d_rowDelimiters, d_vec, numRows, h_reference.data());
    }

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::copy(d_out, d_out + numRows, h_out.begin());
    std::free(d_val);
    std::free(d_cols);
    std::free(d_rowDelimiters);
    std::free(d_vec);
    std::free(d_out);
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
