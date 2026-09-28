#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// Class: RandStream
//
// Purpose:
//   Reproduces the exact sequence of values returned by the C library rand()
//   (glibc TYPE_3 additive feedback generator, degree 31, separation 3) while
//   additionally supporting O(log n) random access into the stream.
//
//   The distributed data generation below needs to produce byte-identical
//   input data to the sequential version, but every rank must be able to
//   start generating in the middle of the stream without stepping through all
//   preceding values. The underlying recurrence
//
//       r[k] = r[k-31] + r[k-3]   (mod 2^32),   rand() -> r[k] >> 1
//
//   is linear, so the state can be advanced by an arbitrary number of steps
//   by exponentiating the 31x31 transition matrix over Z_(2^32).
//
//   The implementation is validated against the platform rand() at startup;
//   if the platform generator differs, the class transparently falls back to
//   using rand() itself (skipping then means stepping the stream, which is
//   correct but not parallel-friendly).
//
// ****************************************************************************
class RandStream {
  public:
    static constexpr int DEG = 31;  // degree of the recurrence
    static constexpr int SEP = 3;   // separation

    // Enable/disable the fast (random access) implementation globally.
    static bool& useFast() {
        static bool fast = true;
        return fast;
    }

    void seed(unsigned s) {
        m_seed = s;
        m_pos = 0;
        if (useFast()) {
            init();
        } else {
            srand(s);
        }
    }

    // Position the stream such that the next call to next() returns the
    // value that the target'th (0-based) rand() call would have returned.
    void skipTo(uint64_t target) {
        if (target < m_pos) {
            // Restart the stream; only ever needed for backwards seeks.
            const unsigned s = m_seed;
            seed(s);
        }
        const uint64_t delta = target - m_pos;
        if (delta == 0) { return; }
        if (useFast()) {
            jump(delta);
        } else {
            for (uint64_t i = 0; i < delta; ++i) { rand(); }
        }
        m_pos = target;
    }

    inline int next() {
        ++m_pos;
        if (!useFast()) { return rand(); }
        const int p = m_p;
        const uint32_t v = (m_buf[p] += m_buf[m_back[p]]);
        m_p = (p + 1 == DEG) ? 0 : p + 1;
        return static_cast<int>(v >> 1);
    }

    // Check that the fast generator matches the platform rand().
    static bool validateAgainstLibc() {
        static const unsigned seeds[] = {1u, 8675309u};
        for (const unsigned s : seeds) {
            RandStream rs;
            rs.m_seed = s;
            rs.m_pos = 0;
            rs.init();
            srand(s);
            for (int i = 0; i < 1024; ++i) {
                if (rs.next() != rand()) { return false; }
            }
        }
        return true;
    }

  private:
    using Mat = std::vector<uint32_t>;  // DEG x DEG, row major

    // Initial table as produced by glibc's __initstate_r(), followed by the
    // 10 * DEG discarded outputs.
    void init() {
        int32_t r[DEG];
        r[0] = static_cast<int32_t>(m_seed == 0 ? 1u : m_seed);
        for (int i = 1; i < DEG; ++i) {
            r[i] = static_cast<int32_t>((16807LL * static_cast<int64_t>(r[i - 1])) % 2147483647LL);
        }
        for (int i = 0; i < DEG; ++i) { m_buf[i] = static_cast<uint32_t>(r[i]); }
        for (int i = 0; i < DEG; ++i) { m_back[i] = (i + DEG - SEP) % DEG; }
        // The first generated element is written to slot SEP, i.e. it plays
        // the role of index DEG + SEP in the recurrence.
        m_p = SEP;
        for (int i = 0; i < 10 * DEG; ++i) { next(); }
        m_pos = 0;
    }

    // Advance the state by delta steps using matrix exponentiation.
    void jump(uint64_t delta) {
        // Recurrence index of the most recently produced value.
        uint64_t idx = IDX0 + m_pos;

        // Unpack circular buffer into window w[k] = r[idx - k].
        Mat w(DEG);
        for (int k = 0; k < DEG; ++k) {
            w[k] = m_buf[static_cast<int>((idx - static_cast<uint64_t>(k)) % DEG)];
        }

        // Transition matrix: r[i+1] = r[i-30] + r[i-2] -> row 0 = e[30] + e[2],
        // remaining rows shift the window.
        Mat base(DEG * DEG, 0u);
        base[0 * DEG + (DEG - 1)] += 1u;
        base[0 * DEG + (SEP - 1)] += 1u;
        for (int i = 1; i < DEG; ++i) { base[i * DEG + (i - 1)] = 1u; }

        Mat acc(DEG * DEG, 0u);
        for (int i = 0; i < DEG; ++i) { acc[i * DEG + i] = 1u; }

        Mat tmp(DEG * DEG);
        uint64_t e = delta;
        while (e != 0) {
            if ((e & 1u) != 0) {
                matmul(acc, base, tmp);
                acc.swap(tmp);
            }
            e >>= 1;
            if (e != 0) {
                matmul(base, base, tmp);
                base.swap(tmp);
            }
        }

        Mat nw(DEG, 0u);
        for (int i = 0; i < DEG; ++i) {
            uint32_t s = 0;
            const uint32_t* row = &acc[i * DEG];
            for (int j = 0; j < DEG; ++j) { s += row[j] * w[j]; }
            nw[i] = s;
        }

        // Repack.
        idx += delta;
        for (int k = 0; k < DEG; ++k) {
            m_buf[static_cast<int>((idx - static_cast<uint64_t>(k)) % DEG)] = nw[k];
        }
        m_p = static_cast<int>((idx + 1) % DEG);
    }

    static void matmul(const Mat& a, const Mat& b, Mat& c) {
        std::fill(c.begin(), c.end(), 0u);
        for (int i = 0; i < DEG; ++i) {
            for (int k = 0; k < DEG; ++k) {
                const uint32_t aik = a[i * DEG + k];
                if (aik == 0u) { continue; }
                const uint32_t* brow = &b[k * DEG];
                uint32_t* crow = &c[i * DEG];
                for (int j = 0; j < DEG; ++j) { crow[j] += aik * brow[j]; }
            }
        }
    }

    // Recurrence index of the last value produced during initialization
    // (DEG + SEP - 1 initial entries plus 10 * DEG discarded outputs).
    static constexpr uint64_t IDX0 = DEG + SEP - 1 + 10 * DEG;

    uint32_t m_buf[DEG] = {};
    int m_back[DEG] = {};
    int m_p = 0;             // slot written by the next call to next()
    uint64_t m_pos = 0;      // number of values consumed since seeding
    unsigned m_seed = 1;
};

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values, taken from the given position of the
//   random stream (matching the sequential generation order).
//
// Arguments:
//   A: pointer to the array to initialize
//   n: number of elements in the array
//   maxVal: specifies range of random values [0, maxVal]
//   rng: random stream, positioned at the first value to consume
//
// ****************************************************************************
void fill(double* A, const uint64_t n, const double maxVal, RandStream& rng) {
    for (uint64_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rng.next() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: initRandomMatrixLocal
//
// Purpose:
//   Distributed version of the original initRandomMatrix(). Assigns random
//   positions to a given number of elements in a square matrix and encodes
//   them in compressed sparse row (CSR) format; every rank only stores the
//   rows it owns.
//
//   The sequential algorithm walks all dim*dim matrix entries in row major
//   order, drawing one random value per entry and accepting it with
//   probability prob, until n entries have been assigned. If the remaining
//   entries are exactly as many as the still missing nonzeros, all remaining
//   entries are taken ("fillRemaining").
//
//   This is reproduced exactly as follows:
//     1) Every rank draws the random values for its own row block (using
//        random access into the stream) and counts the entries it would
//        accept, ignoring the global termination logic.
//     2) The per-rank counts are gathered, giving each rank the exact number
//        of nonzeros assigned before its own block, and identifying the
//        single block in which the termination logic kicks in (both
//        termination conditions are monotone in the entry index).
//     3) That block is regenerated with the full original logic; blocks after
//        it are either empty (nonzero budget exhausted) or completely full
//        (fillRemaining).
//
// Arguments:
//   cols:          output, column indexes of the locally owned elements
//   rowDelimiters: output, size localRows+1, local indices into cols
//   globalOffset:  output, index of the first local nonzero in the global
//                  nonzero enumeration
//   n:             number of nonzero elements (global)
//   dim:           number of rows/columns in the matrix
//   rowBegin:      first row owned by this rank
//   rowEnd:        one past the last row owned by this rank
//   comm:          communicator
//
// ****************************************************************************
void initRandomMatrixLocal(std::vector<index_t>& cols, std::vector<index_t>& rowDelimiters,
                           uint64_t& globalOffset, const index_t n, const index_t dim,
                           const index_t rowBegin, const index_t rowEnd, MPI_Comm comm) {
    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &numRanks);

    const index_t localRows = rowEnd - rowBegin;
    const uint64_t entries = static_cast<uint64_t>(dim) * static_cast<uint64_t>(dim);
    const uint64_t kBegin = static_cast<uint64_t>(rowBegin) * dim;
    const uint64_t kEnd = static_cast<uint64_t>(rowEnd) * dim;

    // Figure out the probability that a nonzero should be assigned
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    std::vector<index_t> rowNnz(localRows, 0);
    cols.clear();
    cols.reserve(static_cast<size_t>(static_cast<double>(kEnd - kBegin) * prob * 1.05) + 64);

    // Largest random value v for which (double)v / RAND_MAX <= prob holds.
    // The mapping is monotone, so testing the integer against this threshold
    // is exactly equivalent to the original floating point comparison, but
    // takes a division and a conversion out of the innermost loop.
    int64_t threshold = -1;
    {
        int64_t lo = 0;
        int64_t hi = RAND_MAX;
        while (lo <= hi) {
            const int64_t mid = lo + (hi - lo) / 2;
            if (static_cast<double>(mid) / RAND_MAX <= prob) {
                threshold = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
    }

    // --- Step 1: unconstrained generation of the local row block ------------
    RandStream rng;
    rng.seed(8675309);
    rng.skipTo(kBegin);

    uint64_t localAccepted = 0;
    for (index_t i = 0; i < localRows; ++i) {
        index_t rowCount = 0;
        for (index_t j = 0; j < dim; ++j) {
            if (rng.next() <= threshold) {
                cols.push_back(j);
                ++rowCount;
            }
        }
        rowNnz[i] = rowCount;
        localAccepted += rowCount;
    }

    // --- Step 2: global prefix information ---------------------------------
    uint64_t myCount = localAccepted;
    std::vector<uint64_t> allCounts(numRanks);
    MPI_Allgather(&myCount, 1, MPI_UINT64_T, allCounts.data(), 1, MPI_UINT64_T, comm);

    uint64_t prefix = 0;
    int trigger = 0;
    for (int r = 0; r < numRanks; ++r) {
        const uint64_t rBegin = static_cast<uint64_t>(dim) * ((static_cast<uint64_t>(dim) * r) / numRanks);
        const uint64_t rEnd = static_cast<uint64_t>(dim) * ((static_cast<uint64_t>(dim) * (r + 1)) / numRanks);
        if (rEnd == rBegin) { continue; }  // rank owns no rows
        trigger = r;                       // last non-empty block seen so far
        const uint64_t accEnd = prefix + allCounts[r];
        const bool budgetDone = accEnd >= n;
        const bool fillDone =
            static_cast<int64_t>(entries - rEnd) <= static_cast<int64_t>(n) - static_cast<int64_t>(accEnd);
        if (budgetDone || fillDone) {
            trigger = r;
            break;
        }
        prefix = accEnd;
    }
    // prefix now holds the number of nonzeros assigned before block 'trigger'.

    uint64_t myOffset = 0;
    uint64_t triggerInfo[2] = {0, 0};  // {fillRemaining, nnz assigned at end of trigger block}

    if (rank < trigger) {
        // Unconstrained result is exact; recompute own prefix.
        uint64_t off = 0;
        for (int r = 0; r < rank; ++r) { off += allCounts[r]; }
        myOffset = off;
    } else if (rank == trigger) {
        // --- Step 3a: apply the original termination logic to this block ----
        //
        // Neither termination condition depends on the random values, only on
        // how many entries have been accepted/rejected so far:
        //
        //   fillRemaining  <=>  rejected >= dim*dim - n
        //   budget reached <=>  accepted == n
        //
        // The unconstrained result of step 1 is therefore valid up to the
        // first entry at which one of them holds, and no random numbers have
        // to be drawn again.
        myOffset = prefix;
        const index_t dimSquared = dim * dim;
        const uint64_t rejectionsToFill = dimSquared - n;

        uint64_t nnzAssigned = prefix;
        uint64_t rejected = kBegin - prefix;

        // Rows that are provably unaffected keep their step 1 result.
        index_t firstDirty = 0;
        size_t keptNnz = 0;
        while (firstDirty < localRows) {
            const uint64_t accEndRow = nnzAssigned + rowNnz[firstDirty];
            const uint64_t rejEndRow = rejected + (dim - rowNnz[firstDirty]);
            if (rejEndRow >= rejectionsToFill || accEndRow > n) { break; }
            nnzAssigned = accEndRow;
            rejected = rejEndRow;
            keptNnz += rowNnz[firstDirty];
            ++firstDirty;
        }

        // Replay the remaining rows entry by entry, using the step 1 result
        // as the outcome of the random draws.
        const std::vector<index_t> tail(cols.begin() + static_cast<ptrdiff_t>(keptNnz), cols.end());
        cols.resize(keptNnz);
        bool fillRemaining = false;
        size_t tailPos = 0;
        for (index_t i = firstDirty; i < localRows; ++i) {
            const index_t drawn = rowNnz[i];
            const index_t* rowTail = tail.data() + tailPos;
            tailPos += drawn;
            index_t next = 0;  // index into the accepted columns of this row
            index_t rowCount = 0;
            for (index_t j = 0; j < dim; ++j) {
                if (rejected >= rejectionsToFill) { fillRemaining = true; }
                const bool drawnHit = (next < drawn) && (rowTail[next] == j);
                if (drawnHit) { ++next; }
                if ((nnzAssigned < n && drawnHit) || fillRemaining) {
                    cols.push_back(j);
                    ++nnzAssigned;
                    ++rowCount;
                } else {
                    ++rejected;
                }
            }
            rowNnz[i] = rowCount;
        }
        triggerInfo[0] = fillRemaining ? 1 : 0;
        triggerInfo[1] = nnzAssigned;
    }

    MPI_Bcast(triggerInfo, 2, MPI_UINT64_T, trigger, comm);

    if (rank > trigger) {
        // --- Step 3b: deterministic tail ----------------------------------
        cols.clear();
        if (triggerInfo[0] == 0) {
            // Nonzero budget exhausted: nothing left for this rank.
            myOffset = triggerInfo[1];
            std::fill(rowNnz.begin(), rowNnz.end(), 0);
        } else {
            // fillRemaining: every remaining entry is a nonzero.
            const uint64_t triggerEnd =
                static_cast<uint64_t>(dim) * ((static_cast<uint64_t>(dim) * (trigger + 1)) / numRanks);
            myOffset = triggerInfo[1] + (kBegin - triggerEnd);
            cols.resize(static_cast<size_t>(localRows) * dim);
            for (index_t i = 0; i < localRows; ++i) {
                index_t* dst = cols.data() + static_cast<size_t>(i) * dim;
                for (index_t j = 0; j < dim; ++j) { dst[j] = j; }
                rowNnz[i] = dim;
            }
        }
    }

    // Local row delimiters (local indices into the local cols/val arrays)
    rowDelimiters.resize(static_cast<size_t>(localRows) + 1);
    rowDelimiters[0] = 0;
    for (index_t i = 0; i < localRows; ++i) { rowDelimiters[i + 1] = rowDelimiters[i] + rowNnz[i]; }

    globalOffset = myOffset;
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format for the
//   locally owned rows of the matrix.
//
// Arguments:
//   val: array holding the non-zero values for the local rows
//   cols: array of column indices for each local element
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   vec: dense vector of size dim to be used for multiplication
//   dim: number of local rows
//   out: output - result from the spmv calculation for the local rows
//
// ****************************************************************************
void spmvCpu(const double* __restrict val, const index_t* __restrict cols,
             const index_t* __restrict rowDelimiters, const double* __restrict vec,
             const index_t dim, double* __restrict out) {
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
//   indexOffset: global index of the first local element (for reporting)
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t indexOffset) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i + indexOffset, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i + indexOffset, ref, res, relError);
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

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    const bool isRoot = (rank == 0);

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
            if (isRoot) { printUsage(argv[0]); }
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (isRoot) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
    }

    // The random number generation used for the input data is reproduced with
    // a random-access capable implementation of the C library generator, so
    // that all ranks can generate their share of the data independently.
    // Verify that the reimplementation matches this platform.
    if (!RandStream::validateAgainstLibc()) {
        RandStream::useFast() = false;
        if (isRoot) {
            printf("Note: platform rand() differs from the reference implementation, "
                   "falling back to sequential stream generation.\n");
        }
    }

    // Row-block distribution of the matrix and the output vector. The number
    // of nonzeros per row is uniformly distributed, so equally sized row
    // blocks also balance the nonzeros.
    auto rowBeginOf = [&](int r) {
        return static_cast<index_t>((static_cast<uint64_t>(numRows) * r) / numRanks);
    };
    const index_t rowBegin = rowBeginOf(rank);
    const index_t rowEnd = rowBeginOf(rank + 1);
    const index_t localRows = rowEnd - rowBegin;

    // Allocate and initialize data structures
    std::vector<index_t> h_cols;                        // Column indices (local rows)
    std::vector<index_t> h_rowDelimiters;               // Row delimiters (local rows)
    std::vector<double> h_vec(numRows);                 // Dense vector (replicated)
    std::vector<double> h_out(localRows);               // Output vector (local rows)

    if (isRoot) { printf("Initializing data structures...\n"); }

    uint64_t nnzOffset = 0;
    initRandomMatrixLocal(h_cols, h_rowDelimiters, nnzOffset, nItems, numRows, rowBegin, rowEnd,
                          MPI_COMM_WORLD);
    const index_t localNnz = h_rowDelimiters.empty() ? 0 : h_rowDelimiters[localRows];

    // Non-zero values: drawn from the same stream as the dense vector, right
    // after it; each rank only materializes its own slice.
    std::vector<double> h_val(localNnz);
    {
        RandStream rng;
        rng.seed(1);
        fill(h_vec.data(), numRows, maxVal, rng);
        rng.skipTo(static_cast<uint64_t>(numRows) + nnzOffset);
        fill(h_val.data(), localNnz, maxVal, rng);
    }

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (isRoot) { printf("Computing reference solution...\n"); }
        h_reference.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    // Perform SpMV computation
    if (isRoot) { printf("Computing SpMV...\n"); }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), localRows, h_out.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long long elapsedMs = duration.count();
    MPI_Allreduce(MPI_IN_PLACE, &elapsedMs, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (isRoot) {
        printf("Computation time: %lld ms\n", elapsedMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (elapsedMs / 1000.0) / 1e9;
        const double avgTime = elapsedMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displs;
        if (isRoot) {
            counts.resize(numRanks);
            displs.resize(numRanks);
            for (int r = 0; r < numRanks; ++r) {
                counts[r] = static_cast<int>(rowBeginOf(r + 1) - rowBeginOf(r));
                displs[r] = static_cast<int>(rowBeginOf(r));
            }
        }
        std::vector<double> gathered(isRoot ? numRows : 0);
        MPI_Gatherv(h_out.data(), static_cast<int>(localRows), MPI_DOUBLE, gathered.data(),
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (isRoot) { print_results(gathered, "OutputVector"); }
    }

    // Validation
    if (validate) {
        if (isRoot) { printf("Validating result...\n"); }
        int valid = verifyResults(h_reference.data(), h_out.data(), localRows, rowBegin) ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (valid != 0) {
            if (isRoot) { printf("Validation: PASSED\n"); }
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) { printf("Validation: FAILED\n"); }
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
