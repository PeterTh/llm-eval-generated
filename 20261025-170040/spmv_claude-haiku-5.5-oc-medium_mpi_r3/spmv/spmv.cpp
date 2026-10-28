#include <mpi.h>

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

// ****************************************************************************
// Function: fillWindow
//
// Purpose:
//   Draws n random values exactly as the serial fill() does (same rand() call
//   sequence), but stores only the values whose global index lies in
//   [keepBegin, keepEnd). This lets each rank keep just its share of the data
//   while reproducing the global random sequence bit-for-bit.
//
// Arguments:
//   dst: receives value with global index g at dst[g - keepBegin]
//   n: total number of values in the serial sequence
//   maxVal: specifies range of random values [0, maxVal]
//   keepBegin, keepEnd: global index window to store
//
// ****************************************************************************
void fillWindow(double* dst, const index_t n, const double maxVal,
                const index_t keepBegin, const index_t keepEnd) {
    for (index_t i = 0; i < n; ++i) {
        const double v = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
        if (i >= keepBegin && i < keepEnd) {
            dst[i - keepBegin] = v;
        }
    }
}

// ****************************************************************************
// Function: initRandomMatrixRows
//
// Purpose:
//   Reproduces the serial initRandomMatrix() decisions for every row, but only
//   records column indices (global numbering) for rows [rowBegin, rowEnd).
//   The random stream is consumed identically on every rank, so the matrix
//   structure is the same regardless of the number of ranks.
//
// Arguments:
//   cols:       appended with global column indices of owned nonzeros
//   rowPtr:     appended with local row offsets (size = rowEnd-rowBegin+1)
//   n:          total number of nonzero elements
//   dim:        number of rows/columns in the matrix
//   rowBegin, rowEnd: owned row range
//   nnzBegin, nnzEnd: global nonzero index range of the owned rows
//
// ****************************************************************************
void initRandomMatrixRows(std::vector<index_t>& cols, std::vector<index_t>& rowPtr,
                          const index_t n, const index_t dim,
                          const index_t rowBegin, const index_t rowEnd,
                          index_t& nnzBegin, index_t& nnzEnd) {
    index_t nnzAssigned = 0;
    nnzBegin = 0;
    nnzEnd = 0;

    // Figure out the probability that a nonzero should be assigned
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        if (i == rowBegin) nnzBegin = nnzAssigned;
        if (i == rowEnd) nnzEnd = nnzAssigned;
        const bool owned = (i >= rowBegin && i < rowEnd);
        if (owned) rowPtr.push_back(static_cast<index_t>(cols.size()));

        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                if (owned) cols.push_back(j);
                nnzAssigned++;
            }
        }
    }
    // Convention: the end of the last row is the number of non-zeroes
    if (rowBegin == dim) nnzBegin = n;
    if (rowEnd == dim) nnzEnd = n;
    rowPtr.push_back(static_cast<index_t>(cols.size()));
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
//   vec: dense vector to be used for multiplication
//   dim: number of rows in the (local) matrix
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
// Function: spmvRows
//
// Purpose:
//   Computes the SpMV result for a subset of local rows. Local column indices
//   refer to the compact vector [owned entries | ghost entries].
//
// ****************************************************************************
void spmvRows(const double* val, const index_t* cols, const index_t* rowDelimiters,
              const double* vec, const std::vector<index_t>& rows, double* out) {
    for (const index_t i : rows) {
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
//   rowOffset: global index of element 0 (used for diagnostics)
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t rowOffset) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                       rowOffset + i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       rowOffset + i, ref, res, relError);
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

// Row ownership: rank p owns rows [starts[p], starts[p+1]).
// Block partition; the first (dim % nprocs) ranks get one extra row.
std::vector<index_t> makeRowPartition(const index_t dim, const int nprocs) {
    std::vector<index_t> starts(nprocs + 1);
    const index_t base = dim / static_cast<index_t>(nprocs);
    const index_t rem = dim % static_cast<index_t>(nprocs);
    for (int p = 0; p <= nprocs; ++p) {
        const index_t up = static_cast<index_t>(p);
        starts[p] = up * base + std::min(up, rem);
    }
    return starts;
}

// Owner rank of global row/column c (c < dim).
int ownerOf(const std::vector<index_t>& starts, const index_t c) {
    return static_cast<int>(std::upper_bound(starts.begin(), starts.end(), c) - starts.begin()) - 1;
}

// Communication plan for the x-vector halo exchange.
struct HaloPlan {
    std::vector<int> recvCount, recvDispl;  // per owner rank; ghosts stored after owned entries
    std::vector<int> sendCount, sendDispl;  // per requesting rank; owned entries to pack
    std::vector<index_t> sendIdx;           // owned-local indices, grouped by requester
    std::vector<double> sendBuf;            // packed values, same layout as sendIdx
    std::vector<int> recvPeers, sendPeers;  // ranks with nonzero traffic
    index_t numGhosts = 0;
};

// Builds the halo plan and remaps the global column indices in `cols` to the
// compact local numbering [0, localRows) = owned, [localRows, ...) = ghosts.
void buildHaloPlan(MPI_Comm comm, const int nprocs, const int rank,
                   const std::vector<index_t>& starts, std::vector<index_t>& cols,
                   HaloPlan& plan) {
    const index_t rowBegin = starts[rank];
    const index_t rowEnd = starts[rank + 1];
    const index_t localRows = rowEnd - rowBegin;

    // Unique remote columns, sorted => grouped by owner in ascending rank order
    std::vector<index_t> ghosts;
    for (const index_t c : cols) {
        if (c < rowBegin || c >= rowEnd) ghosts.push_back(c);
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());

    // Ghost counts per owner, and the owner-local index of each request
    plan.recvCount.assign(nprocs, 0);
    plan.recvDispl.assign(nprocs, 0);
    std::vector<index_t> reqIdx(ghosts.size());
    for (size_t k = 0; k < ghosts.size(); ++k) {
        const int owner = ownerOf(starts, ghosts[k]);
        plan.recvCount[owner]++;
        reqIdx[k] = ghosts[k] - starts[owner];
    }
    for (int p = 1; p < nprocs; ++p) {
        plan.recvDispl[p] = plan.recvDispl[p - 1] + plan.recvCount[p - 1];
    }
    plan.numGhosts = static_cast<index_t>(ghosts.size());

    // Tell each owner which of its entries this rank needs
    plan.sendCount.assign(nprocs, 0);
    plan.sendDispl.assign(nprocs, 0);
    MPI_Alltoall(plan.recvCount.data(), 1, MPI_INT, plan.sendCount.data(), 1, MPI_INT, comm);
    for (int p = 1; p < nprocs; ++p) {
        plan.sendDispl[p] = plan.sendDispl[p - 1] + plan.sendCount[p - 1];
    }
    const int sendTotal = plan.sendDispl[nprocs - 1] + plan.sendCount[nprocs - 1];
    plan.sendIdx.assign(sendTotal, 0);
    plan.sendBuf.assign(sendTotal, 0.0);
    MPI_Alltoallv(reqIdx.data(), plan.recvCount.data(), plan.recvDispl.data(), MPI_UINT32_T,
                  plan.sendIdx.data(), plan.sendCount.data(), plan.sendDispl.data(), MPI_UINT32_T,
                  comm);

    // Remap column indices to compact local numbering
    for (index_t& c : cols) {
        if (c >= rowBegin && c < rowEnd) {
            c -= rowBegin;
        } else {
            const auto it = std::lower_bound(ghosts.begin(), ghosts.end(), c);
            c = localRows + static_cast<index_t>(it - ghosts.begin());
        }
    }

    plan.recvPeers.clear();
    plan.sendPeers.clear();
    for (int p = 0; p < nprocs; ++p) {
        if (plan.recvCount[p] > 0) plan.recvPeers.push_back(p);
        if (plan.sendCount[p] > 0) plan.sendPeers.push_back(p);
    }
}

// Entry point for the benchmark logic, executed identically on all ranks
// (only rank 0 produces console output). Returns the process exit code.
int runBenchmark(int argc, char** argv, MPI_Comm comm, int rank, int nprocs) {
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
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI processes: %d\n", nprocs);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    // Row partition and this rank's share of the problem
    const std::vector<index_t> starts = makeRowPartition(numRows, nprocs);
    const index_t rowBegin = starts[rank];
    const index_t rowEnd = starts[rank + 1];
    const index_t localRows = rowEnd - rowBegin;

    std::vector<int> rowCounts(nprocs), rowDispls(nprocs);
    for (int p = 0; p < nprocs; ++p) {
        rowCounts[p] = static_cast<int>(starts[p + 1] - starts[p]);
        rowDispls[p] = static_cast<int>(starts[p]);
    }

    // Local matrix structure (global column numbering for now)
    std::vector<index_t> cols;
    std::vector<index_t> rowPtr;
    index_t nnzBegin = 0, nnzEnd = 0;
    initRandomMatrixRows(cols, rowPtr, nItems, numRows, rowBegin, rowEnd, nnzBegin, nnzEnd);

    // Dense vector and values: reproduce the serial fill() sequence. The
    // serial code starts from the C-standard initial seed (equivalent to srand(1)).
    srand(1);
    std::vector<double> x(localRows);
    std::vector<double> val(nnzEnd - nnzBegin);
    fillWindow(x.data(), numRows, maxVal, rowBegin, rowEnd);
    fillWindow(val.data(), nItems, maxVal, nnzBegin, nnzEnd);

    // For validation, compute the reference solution for owned rows using the
    // full vector (independent of the halo exchange used in the timed loop)
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        std::vector<double> xFull(numRows);
        MPI_Allgatherv(x.data(), static_cast<int>(localRows), MPI_DOUBLE,
                       xFull.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE, comm);
        h_reference.resize(localRows);
        spmvCpu(val.data(), cols.data(), rowPtr.data(), xFull.data(), localRows, h_reference.data());
    }

    // Communication plan and local column remapping
    HaloPlan plan;
    buildHaloPlan(comm, nprocs, rank, starts, cols, plan);

    // Compact vector: owned entries followed by ghost entries
    x.resize(localRows + plan.numGhosts);

    // Rows without any ghost reference can be computed while messages are in flight
    std::vector<index_t> interiorRows, boundaryRows;
    for (index_t i = 0; i < localRows; ++i) {
        bool external = false;
        for (index_t j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
            if (cols[j] >= localRows) {
                external = true;
                break;
            }
        }
        (external ? boundaryRows : interiorRows).push_back(i);
    }

    std::vector<double> out(localRows);
    std::vector<MPI_Request> reqs(plan.recvPeers.size() + plan.sendPeers.size());

    if (rank == 0) printf("Computing SpMV...\n");

    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        int numReqs = 0;
        for (const int p : plan.recvPeers) {
            MPI_Irecv(x.data() + localRows + plan.recvDispl[p], plan.recvCount[p], MPI_DOUBLE,
                      p, 0, comm, &reqs[numReqs++]);
        }
        for (size_t k = 0; k < plan.sendIdx.size(); ++k) {
            plan.sendBuf[k] = x[plan.sendIdx[k]];
        }
        for (const int p : plan.sendPeers) {
            MPI_Isend(plan.sendBuf.data() + plan.sendDispl[p], plan.sendCount[p], MPI_DOUBLE,
                      p, 0, comm, &reqs[numReqs++]);
        }

        spmvRows(val.data(), cols.data(), rowPtr.data(), x.data(), interiorRows, out.data());

        MPI_Waitall(numReqs, reqs.data(), MPI_STATUSES_IGNORE);

        spmvRows(val.data(), cols.data(), rowPtr.data(), x.data(), boundaryRows, out.data());
    }

    auto end = std::chrono::high_resolution_clock::now();

    // The slowest rank determines the computation time
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double maxSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(maxSeconds));
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation (gathered in row order on rank 0)
    if (printResults) {
        std::vector<double> full(rank == 0 ? numRows : 0);
        MPI_Gatherv(out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    full.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(full, "OutputVector");
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localOk = verifyResults(h_reference.data(), out.data(), localRows, rowBegin) ? 1 : 0;
        int allOk = 0;
        MPI_Allreduce(&localOk, &allOk, 1, MPI_INT, MPI_LAND, comm);

        if (rank == 0) {
            printf("Validation: %s\n", allOk ? "PASSED" : "FAILED");
        }
        return allOk ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int rc = runBenchmark(argc, argv, MPI_COMM_WORLD, rank, nprocs);

    MPI_Finalize();
    return rc;
}
