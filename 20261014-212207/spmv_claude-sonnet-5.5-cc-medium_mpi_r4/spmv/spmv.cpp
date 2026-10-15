#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// glibc-compatible rand() with O(log k) jump-ahead.
//
// glibc's rand() (TYPE_3) is the additive feedback generator
//   r[i] = r[i-31] + r[i-3] (mod 2^32),   output k = r[k+344] >> 1
// which is linear, so any position of the stream can be reached by computing
// x^k mod (x^31 - x^28 - 1). This lets every rank generate exactly the slice of
// the original sequential random stream that it needs.
// ****************************************************************************
class JumpRand {
public:
    // Position the generator so that next() returns the k-th output of srand(seed)
    void seek(uint32_t seed, uint64_t k) {
        uint32_t base[64];
        int32_t w = static_cast<int32_t>(seed ? seed : 1);
        base[0] = static_cast<uint32_t>(w);
        for (int i = 1; i < 31; ++i) {
            const int64_t hi = w / 127773, lo = w % 127773;
            int64_t word = 16807 * lo - 2836 * hi;
            if (word < 0) word += 2147483647;
            w = static_cast<int32_t>(word);
            base[i] = static_cast<uint32_t>(w);
        }
        for (int i = 31; i < 34; ++i) base[i] = base[i - 31];
        for (int i = 34; i < 64; ++i) base[i] = base[i - 31] + base[i - 3];

        // Window r[W .. W+30] with W = k + 344 - 31
        const uint64_t W = k + 344 - 31;
        uint32_t c[31];
        xPow(W - 3, c);
        for (int j = 0; j < 31; ++j) {
            uint32_t v = 0;
            for (int m = 0; m < 31; ++m) v += c[m] * base[3 + m + j];
            ring_[j] = v;
        }
        i_ = 0;
        j_ = 28;
    }

    inline uint32_t next() {
        const uint32_t v = ring_[i_] + ring_[j_];
        ring_[i_] = v;
        if (++i_ == 31) i_ = 0;
        if (++j_ == 31) j_ = 0;
        return v >> 1;
    }

private:
    static void mulmod(const uint32_t* a, const uint32_t* b, uint32_t* out) {
        uint32_t t[61] = {0};
        for (int i = 0; i < 31; ++i)
            for (int j = 0; j < 31; ++j) t[i + j] += a[i] * b[j];
        for (int k = 60; k >= 31; --k) {  // x^k = x^(k-3) + x^(k-31)
            t[k - 3] += t[k];
            t[k - 31] += t[k];
        }
        for (int i = 0; i < 31; ++i) out[i] = t[i];
    }
    static void xPow(uint64_t e, uint32_t* out) {
        uint32_t result[31] = {0}, sq[31] = {0}, tmp[31];
        result[0] = 1;
        sq[1] = 1;
        while (e) {
            if (e & 1) {
                mulmod(result, sq, tmp);
                memcpy(result, tmp, sizeof(tmp));
            }
            mulmod(sq, sq, tmp);
            memcpy(sq, tmp, sizeof(tmp));
            e >>= 1;
        }
        memcpy(out, result, sizeof(result));
    }

    uint32_t ring_[31];
    int i_ = 0, j_ = 28;
};

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values taken from the srand(1) default stream,
//   starting at stream position `offset`
//
// Arguments:
//   A: pointer to the array to initialize
//   n: number of elements in the array
//   maxVal: specifies range of random values [0, maxVal]
//   offset: position in the random stream of the first element
//
// ****************************************************************************
void fill(double* A, const uint64_t n, const double maxVal, const uint64_t offset) {
    JumpRand rng;
    rng.seek(1, offset);
    for (uint64_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rng.next() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: initRandomMatrix
//
// Purpose:
//   Generates the rows [rowBegin, rowEnd) of the same random CSR matrix the
//   sequential algorithm would produce (RNG seeded with 8675309, one random
//   number per matrix entry in row-major order, exactly n nonzeros).
//
//   The sequential algorithm carries a dependency (nnzAssigned) between rows,
//   which only matters at the cap (nnz == n) or in the fill-remaining tail.
//   Each rank first generates its rows assuming neither applies and counts its
//   hits; the counts are then chained across ranks. Ranks whose range could be
//   affected by the cap/tail are regenerated exactly with the true start count.
//
// Outputs (local):
//   cols:    column indices of local nonzeros
//   rowPtr:  local row delimiters (size localRows + 1, starting at 0)
//   nnzStart: global index of the first local nonzero
//
// ****************************************************************************
void initRandomMatrix(std::vector<index_t>& cols, std::vector<index_t>& rowPtr, uint64_t& nnzStart,
                      const uint64_t n, const uint64_t dim, const uint64_t rowBegin,
                      const uint64_t rowEnd, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const uint64_t localRows = rowEnd - rowBegin;
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));
    const uint64_t total = dim * dim;
    const uint64_t posBegin = rowBegin * dim;

    // randVal <= prob  <=>  r <= thr  (monotonic in r); find thr with the exact same expression
    int64_t lo = -1, hi = RAND_MAX;
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo + 1) / 2;
        if (static_cast<double>(mid) / RAND_MAX <= prob) lo = mid; else hi = mid - 1;
    }
    const int64_t thr = lo;

    JumpRand rng;
    rng.seek(8675309, posBegin);

    // Pass 1: uncapped generation
    cols.clear();
    cols.resize(dim + 1024);
    rowPtr.assign(localRows + 1, 0);
    uint64_t cnt = 0;
    for (uint64_t i = 0; i < localRows; ++i) {
        if (cols.size() < cnt + dim) cols.resize(2 * (cnt + dim));
        index_t* out = cols.data() + cnt;
        uint32_t c = 0;
        for (uint64_t j = 0; j < dim; ++j) {
            const int64_t r = rng.next();
            out[c] = static_cast<index_t>(j);
            c += (r <= thr);
        }
        cnt += c;
        rowPtr[i + 1] = static_cast<index_t>(cnt);
    }

    std::vector<uint64_t> counts(size);
    MPI_Allgather(&cnt, 1, MPI_UINT64_T, counts.data(), 1, MPI_UINT64_T, comm);

    // Chain the true start offsets across ranks
    uint64_t T = 0, myStart = 0;
    for (int r = 0; r < size; ++r) {
        const uint64_t rEndPos = (dim * (r + 1) / size) * dim;
        const bool clean = (T + counts[r] <= n) && (rEndPos - (T + counts[r]) <= total - n);
        uint64_t trueCount = counts[r];
        if (r == rank) myStart = T;
        if (!clean) {
            if (r == rank) {
                // Exact sequential simulation of this rank's range starting with nnz = T
                rng.seek(8675309, posBegin);
                uint64_t nnz = T;
                cnt = 0;
                cols.clear();
                bool fillRemaining = false;
                for (uint64_t i = 0; i < localRows; ++i) {
                    for (uint64_t j = 0; j < dim; ++j) {
                        const uint64_t pos = posBegin + i * dim + j;
                        if (total - pos <= n - nnz) fillRemaining = true;
                        const int64_t rv = rng.next();
                        if ((nnz < n && rv <= thr) || fillRemaining) {
                            cols.push_back(static_cast<index_t>(j));
                            ++cnt;
                            ++nnz;
                        }
                    }
                    rowPtr[i + 1] = static_cast<index_t>(cnt);
                }
                trueCount = cnt;
            }
            MPI_Bcast(&trueCount, 1, MPI_UINT64_T, r, comm);
        }
        T += trueCount;
    }
    nnzStart = myStart;
    cols.resize(rowPtr[localRows]);
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
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t globalOffset) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", globalOffset + i, ref,
                       res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       globalOffset + i, ref, res, relError);
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

static int run(int argc, char** argv, const int rank, const int nprocs) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank parses identically)
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
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nprocs);
        printf("Initializing data structures...\n");
    }

    // Block row distribution
    std::vector<int> rowCounts(nprocs), rowDispls(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        const uint64_t b = static_cast<uint64_t>(numRows) * r / nprocs;
        const uint64_t e = static_cast<uint64_t>(numRows) * (r + 1) / nprocs;
        rowDispls[r] = static_cast<int>(b);
        rowCounts[r] = static_cast<int>(e - b);
    }
    const index_t rowBegin = rowDispls[rank];
    const index_t localRows = rowCounts[rank];

    // The dense vector is replicated on every rank, so iterations need no communication
    std::vector<double> h_vec(numRows);
    fill(h_vec.data(), numRows, maxVal, 0);

    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;  // local, starts at 0
    uint64_t nnzStart = 0;
    initRandomMatrix(h_cols, h_rowDelimiters, nnzStart, nItems, numRows, rowBegin,
                     static_cast<uint64_t>(rowBegin) + localRows, MPI_COMM_WORLD);

    std::vector<double> h_val(h_cols.size());
    fill(h_val.data(), h_val.size(), maxVal, static_cast<uint64_t>(numRows) + nnzStart);
    std::vector<double> h_out(localRows);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        h_reference.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), localRows, h_out.data());
    }

    double elapsed = MPI_Wtime() - t0;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (ms / 1000.0) / 1e9;
        const double avgTime = ms / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> full;
        if (rank == 0) full.resize(numRows);
        MPI_Gatherv(h_out.data(), localRows, MPI_DOUBLE, full.data(), rowCounts.data(),
                    rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(full, "OutputVector");
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int valid = verifyResults(h_reference.data(), h_out.data(), localRows, rowBegin) ? 1 : 0;
        int allValid = 0;
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf(allValid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        return allValid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const int rc = run(argc, argv, rank, nprocs);
    MPI_Finalize();
    return rc;
}
