#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
static_assert(sizeof(index_t) == sizeof(unsigned int), "index_t must match MPI_UNSIGNED");

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr unsigned int MATRIX_SEED = 8675309;
// srand(1) is the implicit initial seed of rand(); the original program never seeds
// before generating the vector and values, so this reproduces the same sequence.
constexpr unsigned int VECTOR_SEED = 1;
constexpr int HALO_TAG = 1;

// ****************************************************************************
// Function: randomValue
//
// Purpose:
//   Produce one random value in [0, maxVal] (same expression as the original
//   fill routine, so the RNG consumption and values are identical)
//
// ****************************************************************************
inline double randomValue(const double maxVal) {
    return maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

// ****************************************************************************
// Function: forEachNonzero
//
// Purpose:
//   Enumerates the random sparsity pattern in row-major order, calling
//   visit(i, j) for each nonzero (i,j). This reproduces exactly the pattern
//   of the original initRandomMatrix routine. It is deterministic, so every
//   rank can run it independently and obtain the same pattern.
//
// Arguments:
//   n:     number of nonzero elements
//   dim:   number of rows/columns in the matrix
//   visit: callable invoked as visit(row, col)
//
// ****************************************************************************
template <typename Visit>
void forEachNonzero(const index_t n, const index_t dim, Visit visit) {
    index_t nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(MATRIX_SEED);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                visit(i, j);
                nnzAssigned++;
            }
        }
    }
}

// ****************************************************************************
// Function: computeRowPartition
//
// Purpose:
//   Splits the rows into nprocs contiguous blocks with approximately equal
//   numbers of nonzeros. Rank p owns rows [bounds[p], bounds[p+1]).
//
// Arguments:
//   rowDelim: CSR row delimiters (size dim+1)
//   dim:      number of rows
//   nprocs:   number of MPI ranks
//
// ****************************************************************************
std::vector<index_t> computeRowPartition(const std::vector<index_t>& rowDelim, const index_t dim, const int nprocs) {
    std::vector<index_t> bounds(nprocs + 1, 0);
    const uint64_t nnz = rowDelim[dim];
    bounds[0] = 0;
    bounds[nprocs] = dim;
    for (int p = 1; p < nprocs; ++p) {
        const auto target = static_cast<index_t>(nnz * static_cast<uint64_t>(p) / static_cast<uint64_t>(nprocs));
        const auto row = static_cast<index_t>(
            std::lower_bound(rowDelim.begin(), rowDelim.begin() + dim + 1, target) - rowDelim.begin());
        bounds[p] = std::min(row, dim);
    }
    return bounds;
}

// Returns the rank owning global index col under the partition bounds.
int ownerOf(const index_t col, const std::vector<index_t>& bounds) {
    return static_cast<int>(std::upper_bound(bounds.begin(), bounds.end(), col) - bounds.begin()) - 1;
}

// ****************************************************************************
// Function: spmvRow
//
// Purpose:
//   Computes one row of the CSR sparse matrix-vector product
//
// ****************************************************************************
inline void spmvRow(const double* val, const index_t* cols, const index_t* rowDelimiters,
                    const double* vec, const index_t row, double* out) {
    double t = 0.0;
    for (index_t j = rowDelimiters[row]; j < rowDelimiters[row + 1]; ++j) {
        const auto col = cols[j];
        t += val[j] * vec[col];
    }
    out[row] = t;
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
        spmvRow(val, cols, rowDelimiters, vec, i, out);
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
//   offset: global row index of element 0 (for diagnostics)
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size, const index_t offset) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", offset + i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       offset + i, ref, res, relError);
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
    MPI_Init(&argc, &argv);
    const MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    const bool root = (rank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    int earlyExit = -1;
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
            if (root) printUsage(argv[0]);
            earlyExit = 0;
            break;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            earlyExit = 1;
            break;
        }
    }
    if (earlyExit >= 0) {
        MPI_Finalize();
        return earlyExit;
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (root) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    // Pass 1: row structure of the pattern (every rank replays the generator; only
    // row counts are stored, so memory stays O(numRows) per rank)
    std::vector<index_t> rowDelim(numRows + 1, 0);
    forEachNonzero(nItems, numRows, [&](index_t i, index_t) { rowDelim[i + 1]++; });
    for (index_t i = 0; i < numRows; ++i) {
        rowDelim[i + 1] += rowDelim[i];
    }

    // Contiguous row partition balanced by nonzeros
    const std::vector<index_t> bounds = computeRowPartition(rowDelim, numRows, nprocs);
    const index_t rowBegin = bounds[rank];
    const index_t rowEnd = bounds[rank + 1];
    const index_t localRows = rowEnd - rowBegin;
    const index_t nnzBegin = rowDelim[rowBegin];
    const index_t nnzEnd = rowDelim[rowEnd];
    const index_t localNnz = nnzEnd - nnzBegin;

    // Local CSR row delimiters (rebased to 0)
    std::vector<index_t> localDelim(localRows + 1);
    for (index_t i = 0; i <= localRows; ++i) {
        localDelim[i] = rowDelim[rowBegin + i] - nnzBegin;
    }

    // Dense vector and nonzero values: same RNG sequence as the original program
    // (vector first, then values, from the initial seed). Only owned parts are kept.
    std::vector<double> xOwned(localRows);
    std::vector<double> val(localNnz);
    std::vector<double> vecFull;  // full vector, only needed for the validation reference
    if (validate) {
        vecFull.resize(numRows);
    }
    srand(VECTOR_SEED);
    for (index_t i = 0; i < numRows; ++i) {
        const double v = randomValue(maxVal);
        if (i >= rowBegin && i < rowEnd) {
            xOwned[i - rowBegin] = v;
        }
        if (validate) {
            vecFull[i] = v;
        }
    }
    for (index_t k = 0; k < nItems; ++k) {
        const double v = randomValue(maxVal);
        if (k >= nnzBegin && k < nnzEnd) {
            val[k - nnzBegin] = v;
        }
    }

    // Pass 2: column indices of the locally owned rows (global column ids)
    std::vector<index_t> cols(localNnz);
    index_t cursor = 0;
    forEachNonzero(nItems, numRows, [&](index_t i, index_t j) {
        if (i >= rowBegin && i < rowEnd) {
            cols[cursor++] = j;
        }
    });

    // For validation, compute the local part of the reference solution
    std::vector<double> reference;
    if (validate) {
        if (root) printf("Computing reference solution...\n");
        reference.resize(localRows);
        spmvCpu(val.data(), cols.data(), localDelim.data(), vecFull.data(), localRows, reference.data());
        std::vector<double>().swap(vecFull);
    }

    // Halo setup: find the remote vector entries this rank needs (sorted, unique).
    // Because the partition is contiguous, the list is already grouped by owner rank.
    std::vector<index_t> remoteCols;
    for (index_t j = 0; j < localNnz; ++j) {
        if (cols[j] < rowBegin || cols[j] >= rowEnd) {
            remoteCols.push_back(cols[j]);
        }
    }
    std::sort(remoteCols.begin(), remoteCols.end());
    remoteCols.erase(std::unique(remoteCols.begin(), remoteCols.end()), remoteCols.end());

    std::vector<int> recvCounts(nprocs, 0);
    std::vector<int> recvDispl(nprocs, 0);
    std::vector<int> sendCounts(nprocs, 0);
    std::vector<int> sendDispl(nprocs, 0);
    for (const index_t c : remoteCols) {
        recvCounts[ownerOf(c, bounds)]++;
    }
    for (int p = 1; p < nprocs; ++p) {
        recvDispl[p] = recvDispl[p - 1] + recvCounts[p - 1];
    }
    // Tell each owner how many entries it must send us
    MPI_Alltoall(recvCounts.data(), 1, MPI_INT, sendCounts.data(), 1, MPI_INT, comm);
    for (int p = 1; p < nprocs; ++p) {
        sendDispl[p] = sendDispl[p - 1] + sendCounts[p - 1];
    }
    const int totalSend = sendDispl[nprocs - 1] + sendCounts[nprocs - 1];

    // Send the requested global indices to their owners; convert to local on arrival
    std::vector<index_t> sendIdx(totalSend);
    {
        std::vector<index_t> requested(totalSend);
        MPI_Alltoallv(remoteCols.data(), recvCounts.data(), recvDispl.data(), MPI_UNSIGNED,
                      requested.data(), sendCounts.data(), sendDispl.data(), MPI_UNSIGNED, comm);
        for (int k = 0; k < totalSend; ++k) {
            sendIdx[k] = requested[k] - rowBegin;
        }
    }

    // Remap column indices: owned entries -> [0, localRows), halo entries -> [localRows, ...)
    for (index_t j = 0; j < localNnz; ++j) {
        const index_t c = cols[j];
        if (c >= rowBegin && c < rowEnd) {
            cols[j] = c - rowBegin;
        } else {
            const auto pos = std::lower_bound(remoteCols.begin(), remoteCols.end(), c) - remoteCols.begin();
            cols[j] = localRows + static_cast<index_t>(pos);
        }
    }

    // Rows that touch no halo entry can be computed while messages are in flight
    std::vector<index_t> interiorRows;
    std::vector<index_t> boundaryRows;
    for (index_t r = 0; r < localRows; ++r) {
        bool needsHalo = false;
        for (index_t j = localDelim[r]; j < localDelim[r + 1]; ++j) {
            if (cols[j] >= localRows) {
                needsHalo = true;
                break;
            }
        }
        (needsHalo ? boundaryRows : interiorRows).push_back(r);
    }

    // Local input vector: owned block followed by the halo region
    std::vector<double> xloc(localRows + remoteCols.size());
    std::copy(xOwned.begin(), xOwned.end(), xloc.begin());
    std::vector<double>().swap(xOwned);

    std::vector<double> out(localRows);
    std::vector<double> sendBuf(totalSend);
    std::vector<MPI_Request> reqs;
    reqs.reserve(2 * static_cast<size_t>(nprocs));

    // Perform SpMV computation
    if (root) printf("Computing SpMV...\n");
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        // Pack values requested by other ranks
        for (int k = 0; k < totalSend; ++k) {
            sendBuf[k] = xloc[sendIdx[k]];
        }

        reqs.clear();
        for (int p = 0; p < nprocs; ++p) {
            if (recvCounts[p] > 0) {
                MPI_Request r;
                MPI_Irecv(xloc.data() + localRows + recvDispl[p], recvCounts[p], MPI_DOUBLE, p, HALO_TAG, comm, &r);
                reqs.push_back(r);
            }
        }
        for (int p = 0; p < nprocs; ++p) {
            if (sendCounts[p] > 0) {
                MPI_Request r;
                MPI_Isend(sendBuf.data() + sendDispl[p], sendCounts[p], MPI_DOUBLE, p, HALO_TAG, comm, &r);
                reqs.push_back(r);
            }
        }

        for (const index_t r : interiorRows) {
            spmvRow(val.data(), cols.data(), localDelim.data(), xloc.data(), r, out.data());
        }

        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

        for (const index_t r : boundaryRows) {
            spmvRow(val.data(), cols.data(), localDelim.data(), xloc.data(), r, out.data());
        }
    }

    const double localElapsed = MPI_Wtime() - t0;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    const long durationMs = static_cast<long>(elapsed * 1000.0);

    if (root) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (durationMs / 1000.0) / 1e9;
        const double avgTime = durationMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<int> counts(nprocs);
        std::vector<int> displs(nprocs);
        for (int p = 0; p < nprocs; ++p) {
            counts[p] = static_cast<int>(bounds[p + 1] - bounds[p]);
            displs[p] = static_cast<int>(bounds[p]);
        }
        std::vector<double> fullOut(root ? numRows : 0);
        MPI_Gatherv(out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    fullOut.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
        if (root) {
            print_results(fullOut, "OutputVector");
        }
    }

    // Validation
    if (validate) {
        if (root) printf("Validating result...\n");
        const int localValid = verifyResults(reference.data(), out.data(), localRows, rowBegin) ? 1 : 0;
        int globalValid = 0;
        MPI_Allreduce(&localValid, &globalValid, 1, MPI_INT, MPI_LAND, comm);

        if (root) {
            printf(globalValid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        }
        MPI_Finalize();
        return globalValid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
