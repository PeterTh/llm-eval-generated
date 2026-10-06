// Hybrid MPI + OpenMP + CUDA SpMV benchmark.
//
//   - MPI:    the matrix rows are block-distributed over ranks; every rank owns
//             a contiguous row block, the matching CSR slice and a full copy of
//             the (read-only) dense input vector. Results are gathered on rank 0.
//   - CUDA:   each rank drives one GPU (round-robin over the node-local GPUs).
//             The local rows are stored in a SELL-32 layout (column-major within
//             slices of 32 rows) so that one thread per row reads fully
//             coalesced data while summing in exactly the same order as the
//             sequential CSR code (bit-identical results).
//   - OpenMP: host-side setup (matrix generation, layout conversion, reference
//             computation) is multithreaded. The glibc rand() sequence is
//             reproduced with a jump-ahead generator so the random matrix can be
//             generated in parallel while staying identical to the original.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int SLICE = 32;          // SELL slice height (= warp size)
constexpr index_t LONG_ROW = 0xFFFFFFFFu;  // rowLen marker for rows handled separately

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

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
//   (OpenMP-parallel over rows; per-row summation order is unchanged)
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

// ****************************************************************************
// glibc rand() (TYPE_3 additive feedback generator) with jump-ahead.
//
//   r[0] = seed, r[i] = 16807 * r[i-1] mod (2^31-1)       for 1 <= i < 31
//   r[i] = r[i-31]                                        for 31 <= i < 34
//   r[i] = r[i-31] + r[i-3]  (mod 2^32)                   for i >= 34
//   k-th output of rand() = r[k + 344] >> 1
//
// The recurrence is linear over Z/2^32, so r[m] can be expressed as a linear
// combination of a 31-element window via x^m mod (x^31 - x^28 - 1).
// The implementation is checked against the C library at runtime; if it does
// not match, the original sequential generator is used instead.
// ****************************************************************************
namespace glibc_rand {

constexpr int DEG = 31;

struct Poly {
    uint32_t c[DEG];
};

inline Poly mulMod(const Poly& a, const Poly& b) {
    uint32_t t[2 * DEG - 1] = {};
    for (int i = 0; i < DEG; ++i) {
        for (int j = 0; j < DEG; ++j) {
            t[i + j] += a.c[i] * b.c[j];
        }
    }
    // x^31 = x^28 + 1
    for (int d = 2 * DEG - 2; d >= DEG; --d) {
        t[d - 3] += t[d];
        t[d - DEG] += t[d];
    }
    Poly r;
    for (int i = 0; i < DEG; ++i) r.c[i] = t[i];
    return r;
}

inline Poly mulX(const Poly& a) {
    Poly r;
    const uint32_t top = a.c[DEG - 1];
    r.c[0] = top;
    for (int i = 1; i < DEG; ++i) r.c[i] = a.c[i - 1];
    r.c[28] += top;
    return r;
}

inline Poly powX(uint64_t e) {
    Poly result{};
    result.c[0] = 1;
    Poly base{};
    base.c[1] = 1;
    while (e) {
        if (e & 1) result = mulMod(result, base);
        e >>= 1;
        if (e) base = mulMod(base, base);
    }
    return result;
}

// Window r[3..33] for a given seed (all later values follow from the recurrence)
struct Seed {
    uint32_t base[DEG];
    explicit Seed(unsigned int seed) {
        int32_t r[34];
        r[0] = static_cast<int32_t>(seed == 0 ? 1 : seed);
        long int word = r[0];
        for (int i = 1; i < DEG; ++i) {
            long int hi = word / 127773;
            long int lo = word % 127773;
            word = 16807 * lo - 2836 * hi;
            if (word < 0) word += 2147483647;
            r[i] = static_cast<int32_t>(word);
        }
        for (int i = DEG; i < 34; ++i) r[i] = r[i - DEG];
        for (int j = 0; j < DEG; ++j) base[j] = static_cast<uint32_t>(r[3 + j]);
    }
};

// Generator positioned so that the first call of next() yields output #k
struct Gen {
    uint32_t buf[DEG];
    int pos = 0;

    Gen(const Seed& s, uint64_t k) {
        const uint64_t t = k + 313;  // buf[l] = r[t + l]
        Poly q = powX(t - 3);
        for (int l = 0; l < DEG; ++l) {
            uint32_t v = 0;
            for (int j = 0; j < DEG; ++j) v += q.c[j] * s.base[j];
            buf[l] = v;
            q = mulX(q);
        }
    }

    inline int next() {
        const int p3 = pos >= 3 ? pos - 3 : pos + 28;
        const uint32_t v = buf[pos] + buf[p3];
        buf[pos] = v;
        pos = (pos == DEG - 1) ? 0 : pos + 1;
        return static_cast<int>(v >> 1);
    }
};

// Verify against the C library implementation (modifies the rand() state)
bool matchesLibc() {
    if (RAND_MAX != 2147483647) return false;
    const unsigned int seeds[2] = {1u, 8675309u};
    for (unsigned int seed : seeds) {
        const Seed s(seed);
        Gen g0(s, 0);
        Gen g1(s, 4093);
        srand(seed);
        for (int k = 0; k < 20000; ++k) {
            const int ref = rand();
            if (g0.next() != ref) return false;
            if (k >= 4093 && g1.next() != ref) return false;
        }
    }
    return true;
}

}  // namespace glibc_rand

// ****************************************************************************
// Local (per-rank) portion of the problem
// ****************************************************************************
struct LocalProblem {
    index_t rowBegin = 0;             // first global row owned by this rank
    index_t nRows = 0;                // number of rows owned by this rank
    std::vector<index_t> rowPtr;      // local CSR delimiters (size nRows + 1)
    std::vector<index_t> cols;        // local column indices
    std::vector<double> val;          // local non-zero values
    std::vector<double> vec;          // full dense vector
};

// Fallback: run the original sequential initialization and extract the local part.
static void initSequential(LocalProblem& lp, const index_t numRows, const index_t nItems,
                           const double maxVal) {
    std::vector<double> val(nItems);
    std::vector<index_t> cols(nItems);
    std::vector<index_t> rowDelimiters(numRows + 1);
    lp.vec.resize(numRows);

    srand(1);  // state of rand() at program start
    fill(lp.vec.data(), numRows, maxVal);
    fill(val.data(), nItems, maxVal);
    initRandomMatrix(cols.data(), rowDelimiters.data(), nItems, numRows);

    const index_t g0 = rowDelimiters[lp.rowBegin];
    const index_t g1 = rowDelimiters[lp.rowBegin + lp.nRows];
    lp.rowPtr.resize(lp.nRows + 1);
    for (index_t i = 0; i <= lp.nRows; ++i) lp.rowPtr[i] = rowDelimiters[lp.rowBegin + i] - g0;
    lp.cols.assign(cols.begin() + g0, cols.begin() + g1);
    lp.val.assign(val.begin() + g0, val.begin() + g1);
}

// Parallel initialization producing exactly the same data as initSequential.
static void initParallel(LocalProblem& lp, const index_t numRows, const index_t nItems,
                         const double maxVal, const int nRanks) {
    using namespace glibc_rand;
    const index_t dim = numRows;
    const index_t n = nItems;
    const index_t rowBegin = lp.rowBegin;
    const index_t rowEnd = lp.rowBegin + lp.nRows;
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));
    const Seed seedMatrix(8675309u);
    const Seed seedValues(1u);

    // ---- Phase 1: speculative generation of the local rows (no cap / fill-remaining logic)
    const int nThreads = omp_get_max_threads();
    std::vector<std::vector<index_t>> threadCols(nThreads);
    std::vector<index_t> threadRowBegin(nThreads + 1);
    for (int t = 0; t <= nThreads; ++t) {
        threadRowBegin[t] = rowBegin + static_cast<index_t>(
            (static_cast<uint64_t>(lp.nRows) * t) / nThreads);
    }
    std::vector<index_t> allCounts(dim);
    std::vector<index_t> localCounts(lp.nRows);

#pragma omp parallel num_threads(nThreads)
    {
        const int t = omp_get_thread_num();
        const index_t a = threadRowBegin[t];
        const index_t b = threadRowBegin[t + 1];
        if (a < b) {
            auto& buf = threadCols[t];
            buf.reserve(static_cast<size_t>(static_cast<double>(b - a) * dim * prob * 1.05) + 64);
            Gen g(seedMatrix, static_cast<uint64_t>(a) * dim);
            for (index_t i = a; i < b; ++i) {
                index_t cnt = 0;
                for (index_t j = 0; j < dim; ++j) {
                    const double randVal = static_cast<double>(g.next()) / RAND_MAX;
                    if (randVal <= prob) {
                        buf.push_back(j);
                        ++cnt;
                    }
                }
                localCounts[i - rowBegin] = cnt;
            }
        }
    }

    // ---- Phase 2: share row counts, find the prefix of rows where the speculation is exact
    {
        std::vector<int> recvCounts(nRanks), displs(nRanks);
        for (int r = 0; r < nRanks; ++r) {
            const index_t rb = static_cast<index_t>((static_cast<uint64_t>(dim) * r) / nRanks);
            const index_t re = static_cast<index_t>((static_cast<uint64_t>(dim) * (r + 1)) / nRanks);
            recvCounts[r] = static_cast<int>(re - rb);
            displs[r] = static_cast<int>(rb);
        }
        MPI_Allgatherv(localCounts.data(), static_cast<int>(lp.nRows), MPI_UINT32_T,
                       allCounts.data(), recvCounts.data(), displs.data(), MPI_UINT32_T,
                       MPI_COMM_WORLD);
    }

    std::vector<index_t> delim(dim + 1);
    index_t u = dim;  // first row that must be generated exactly
    uint64_t S = 0;
    for (index_t i = 0; i < dim; ++i) {
        const uint64_t c = allCounts[i];
        // Row is safe if neither the nnz cap nor the "fill remaining" rule can trigger in it.
        const uint64_t entriesLeftAtLast = static_cast<uint64_t>(dim - i - 1) * dim + 1;
        if (S + c > n || entriesLeftAtLast <= n - S) {
            u = i;
            break;
        }
        delim[i] = static_cast<index_t>(S);
        S += c;
    }
    if (u < dim) delim[u] = static_cast<index_t>(S);
    delim[dim] = n;

    // ---- Phase 3: exact (sequential) generation of the tail rows, if any are needed here
    std::vector<index_t> tailCols;
    if (u < dim && rowEnd > u) {
        Gen g(seedMatrix, static_cast<uint64_t>(u) * dim);
        index_t nnzAssigned = static_cast<index_t>(S);
        bool fillRemaining = false;
        for (index_t i = u; i < rowEnd; ++i) {
            delim[i] = nnzAssigned;
            const bool own = i >= rowBegin;
            for (index_t j = 0; j < dim; ++j) {
                index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
                index_t needToAssign = n - nnzAssigned;
                if (numEntriesLeft <= needToAssign) {
                    fillRemaining = true;
                }
                double randVal = static_cast<double>(g.next()) / RAND_MAX;
                if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                    if (own) tailCols.push_back(j);
                    nnzAssigned++;
                }
            }
        }
        if (rowEnd < dim) delim[rowEnd] = nnzAssigned;
    }

    // ---- Assemble local CSR
    const index_t g0 = delim[rowBegin];
    const index_t g1 = delim[rowEnd];
    const index_t localNnz = g1 - g0;
    lp.rowPtr.resize(lp.nRows + 1);
    for (index_t i = 0; i <= lp.nRows; ++i) lp.rowPtr[i] = delim[rowBegin + i] - g0;
    lp.cols.resize(localNnz);
    lp.val.resize(localNnz);

#pragma omp parallel num_threads(nThreads)
    {
        const int t = omp_get_thread_num();
        const index_t a = threadRowBegin[t];
        const index_t b = std::min(threadRowBegin[t + 1], u);
        if (a < b) {
            const index_t off = delim[a] - g0;
            const index_t len = delim[b] - delim[a];
            std::copy(threadCols[t].begin(), threadCols[t].begin() + len, lp.cols.begin() + off);
        }
        std::vector<index_t>().swap(threadCols[t]);
    }
    if (!tailCols.empty()) {
        const index_t off = delim[std::max(u, rowBegin)] - g0;
        std::copy(tailCols.begin(), tailCols.end(), lp.cols.begin() + off);
    }

    // ---- Values: rand() outputs numRows + k (seed 1) for global nnz index k
#pragma omp parallel num_threads(nThreads)
    {
        const int t = omp_get_thread_num();
        const index_t a = static_cast<index_t>((static_cast<uint64_t>(localNnz) * t) / nThreads);
        const index_t b = static_cast<index_t>((static_cast<uint64_t>(localNnz) * (t + 1)) / nThreads);
        if (a < b) {
            Gen g(seedValues, static_cast<uint64_t>(numRows) + g0 + a);
            for (index_t k = a; k < b; ++k) {
                lp.val[k] = maxVal * (g.next() / (static_cast<double>(RAND_MAX) + 1.0));
            }
        }
    }

    // ---- Dense vector: rand() outputs 0 .. numRows-1 (seed 1)
    lp.vec.resize(numRows);
    {
        Gen g(seedValues, 0);
        for (index_t i = 0; i < numRows; ++i) {
            lp.vec[i] = maxVal * (g.next() / (static_cast<double>(RAND_MAX) + 1.0));
        }
    }
}

// ****************************************************************************
// SELL-32 device kernel. Element j of local row r is stored at
// sliceOff[r/32] + j*32 + r%32. One thread block processes one slice of 32 rows:
//   - LOAD_WARPS warps gather val[j] * vec[col[j]] for a stage of STAGE_COLS
//     columns of all 32 rows (coalesced loads, many independent loads in flight)
//     into a double-buffered shared memory tile,
//   - warp 0 accumulates the products of the previous stage, one lane per row,
//     strictly in column order.
// The per-row summation order is therefore identical to the sequential CSR
// code, and __dmul_rn/__dadd_rn prevent FMA contraction, so the results are
// bit-identical to the CPU reference.
// ****************************************************************************
constexpr int LOAD_WARPS = 7;
constexpr int LOADS_PER_LANE = 4;
constexpr int STAGE_COLS = LOAD_WARPS * LOADS_PER_LANE;
constexpr int BLOCK_SIZE = (LOAD_WARPS + 1) * SLICE;

__global__ void __launch_bounds__(BLOCK_SIZE)
spmvSellKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
               const unsigned long long* __restrict__ sliceOff,
               const index_t* __restrict__ rowLen, const double* __restrict__ vec,
               const index_t nRows, double* __restrict__ out) {
    __shared__ double prod[2][STAGE_COLS][SLICE];

    const index_t slice = blockIdx.x;
    const int lane = threadIdx.x % SLICE;
    const int warp = threadIdx.x / SLICE;
    const index_t row = slice * SLICE + lane;
    const index_t rl = row < nRows ? rowLen[row] : 0;
    const index_t len = rl == LONG_ROW ? 0 : rl;  // long rows are handled on the host
    const unsigned long long sBegin = sliceOff[slice];
    const index_t width = static_cast<index_t>((sliceOff[slice + 1] - sBegin) / SLICE);
    const index_t nStages = (width + STAGE_COLS - 1) / STAGE_COLS;
    const double* v = val + sBegin + lane;
    const index_t* c = cols + sBegin + lane;

    double t = 0.0;
    for (index_t s = 0; s <= nStages; ++s) {
        if (warp > 0) {
            if (s < nStages) {
                const int k0 = (warp - 1) * LOADS_PER_LANE;
                const index_t j0 = s * STAGE_COLS + k0;
                index_t cc[LOADS_PER_LANE];
                double vv[LOADS_PER_LANE];
#pragma unroll
                for (int u = 0; u < LOADS_PER_LANE; ++u) {
                    const index_t j = j0 + u;
                    const size_t o = static_cast<size_t>(j) * SLICE;
                    cc[u] = j < len ? __ldg(c + o) : 0;
                    vv[u] = j < len ? __ldg(v + o) : 0.0;
                }
#pragma unroll
                for (int u = 0; u < LOADS_PER_LANE; ++u) {
                    const double x = (j0 + u < len) ? __ldg(vec + cc[u]) : 0.0;
                    prod[s & 1][k0 + u][lane] = __dmul_rn(vv[u], x);
                }
            }
        } else if (s > 0) {
            // Padding entries hold +0.0, adding them leaves t unchanged.
            const index_t j0 = (s - 1) * STAGE_COLS;
            const int kEnd = len > j0 ? static_cast<int>(min(len - j0, static_cast<index_t>(STAGE_COLS))) : 0;
            const double(*p)[SLICE] = prod[(s - 1) & 1];
            for (int k = 0; k < kEnd; ++k) {
                t = __dadd_rn(t, p[k][lane]);
            }
        }
        __syncthreads();
    }
    if (warp == 0 && row < nRows && rl != LONG_ROW) out[row] = t;
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

static int finish(int code) {
    MPI_Finalize();
    return code;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    const bool root = (rank == 0);

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
            if (root) printUsage(argv[0]);
            return finish(0);
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return finish(1);
        }
    }

    // Select the GPU for this rank (round-robin over node-local GPUs)
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0, localSize = 1;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_size(nodeComm, &localSize);
        MPI_Comm_free(&nodeComm);
        // Share the node's cores between the ranks on it unless set explicitly
        if (getenv("OMP_NUM_THREADS") == nullptr) {
            const int hw = static_cast<int>(std::thread::hardware_concurrency());
            const int perRank = std::max(1, (hw > 0 ? hw : omp_get_num_procs()) / localSize);
            omp_set_num_threads(std::max(1, std::min(omp_get_num_procs(), perRank)));
        }
        int nDevices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&nDevices));
        if (nDevices == 0) {
            fprintf(stderr, "No CUDA device found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % nDevices));
        CUDA_CHECK(cudaFree(nullptr));  // create context early
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
    }

    // Row block distribution
    LocalProblem lp;
    std::vector<int> rowCounts(nRanks), rowDispls(nRanks);
    for (int r = 0; r < nRanks; ++r) {
        const index_t rb = static_cast<index_t>((static_cast<uint64_t>(numRows) * r) / nRanks);
        const index_t re = static_cast<index_t>((static_cast<uint64_t>(numRows) * (r + 1)) / nRanks);
        rowCounts[r] = static_cast<int>(re - rb);
        rowDispls[r] = static_cast<int>(rb);
    }
    lp.rowBegin = static_cast<index_t>(rowDispls[rank]);
    lp.nRows = static_cast<index_t>(rowCounts[rank]);

    if (root) printf("Initializing data structures...\n");
    // The parallel generator requires dim*dim to fit in index_t (no wrap-around)
    const bool fastInit = numRows <= 65535 && glibc_rand::matchesLibc();
    if (fastInit) {
        initParallel(lp, numRows, nItems, maxVal, nRanks);
    } else {
        initSequential(lp, numRows, nItems, maxVal);
    }

    // For validation, compute reference solution (local rows, gathered on rank 0)
    std::vector<double> h_reference;
    if (validate) {
        if (root) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
        }
        std::vector<double> localRef(lp.nRows);
        spmvCpu(lp.val.data(), lp.cols.data(), lp.rowPtr.data(), lp.vec.data(), lp.nRows,
                localRef.data());
        MPI_Gatherv(localRef.data(), static_cast<int>(lp.nRows), MPI_DOUBLE, h_reference.data(),
                    rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Separate exceptionally long rows (more than ~2x the average length, at least 512;
    // e.g. the dense tail rows produced by the "fill remaining" rule). Their strictly
    // sequential summation is latency-bound, which is far cheaper on the host CPU
    // (OpenMP, overlapped with the asynchronous GPU kernel) than on the GPU.
    const double avgLen = lp.nRows > 0 ? static_cast<double>(lp.rowPtr[lp.nRows]) / lp.nRows : 0.0;
    const index_t longThreshold = static_cast<index_t>(std::min(2.0 * avgLen + 512.0, 4.0e9));
    std::vector<index_t> longRowIdx;
    std::vector<unsigned long long> longPtr(1, 0);
    for (index_t r = 0; r < lp.nRows; ++r) {
        const index_t len = lp.rowPtr[r + 1] - lp.rowPtr[r];
        if (len > longThreshold) {
            longRowIdx.push_back(r);
            longPtr.push_back(longPtr.back() + len);
        }
    }
    const index_t nLong = static_cast<index_t>(longRowIdx.size());
    std::vector<double> longVal(longPtr.back());
    std::vector<index_t> longCols(longPtr.back());
    for (index_t k = 0; k < nLong; ++k) {
        const index_t p0 = lp.rowPtr[longRowIdx[k]];
        const index_t len = static_cast<index_t>(longPtr[k + 1] - longPtr[k]);
        std::copy(lp.val.begin() + p0, lp.val.begin() + p0 + len, longVal.begin() + longPtr[k]);
        std::copy(lp.cols.begin() + p0, lp.cols.begin() + p0 + len, longCols.begin() + longPtr[k]);
    }

    // Convert the remaining local CSR rows to SELL-32
    const index_t nSlices = (lp.nRows + SLICE - 1) / SLICE;
    std::vector<unsigned long long> sliceOff(nSlices + 1);
    std::vector<index_t> rowLen(lp.nRows);
#pragma omp parallel for schedule(static)
    for (index_t r = 0; r < lp.nRows; ++r) rowLen[r] = lp.rowPtr[r + 1] - lp.rowPtr[r];
    for (index_t k = 0; k < nLong; ++k) rowLen[longRowIdx[k]] = 0;
    sliceOff[0] = 0;
    for (index_t s = 0; s < nSlices; ++s) {
        index_t width = 0;
        const index_t rEnd = std::min<index_t>((s + 1) * SLICE, lp.nRows);
        for (index_t r = s * SLICE; r < rEnd; ++r) width = std::max(width, rowLen[r]);
        sliceOff[s + 1] = sliceOff[s] + static_cast<unsigned long long>(width) * SLICE;
    }
    const size_t sellSize = std::max<size_t>(sliceOff[nSlices], 1);
    std::vector<double> sellVal(sellSize, 0.0);
    std::vector<index_t> sellCols(sellSize, 0);
#pragma omp parallel for schedule(dynamic, 16)
    for (index_t s = 0; s < nSlices; ++s) {
        const index_t rEnd = std::min<index_t>((s + 1) * SLICE, lp.nRows);
        for (index_t r = s * SLICE; r < rEnd; ++r) {
            const size_t base = sliceOff[s] + (r % SLICE);
            const index_t p0 = lp.rowPtr[r];
            for (index_t j = 0; j < rowLen[r]; ++j) {
                sellVal[base + static_cast<size_t>(j) * SLICE] = lp.val[p0 + j];
                sellCols[base + static_cast<size_t>(j) * SLICE] = lp.cols[p0 + j];
            }
        }
    }
    for (index_t k = 0; k < nLong; ++k) rowLen[longRowIdx[k]] = LONG_ROW;
    std::vector<double>().swap(lp.val);
    std::vector<index_t>().swap(lp.cols);

    // Device buffers
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rowLen = nullptr;
    unsigned long long* d_sliceOff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val, sellSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, sellSize * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_sliceOff, (nSlices + 1) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_rowLen, std::max<size_t>(lp.nRows, 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, std::max<size_t>(numRows, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max<size_t>(lp.nRows, 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_val, sellVal.data(), sellSize * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, sellCols.data(), sellSize * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sliceOff, sliceOff.data(), (nSlices + 1) * sizeof(unsigned long long),
                          cudaMemcpyHostToDevice));
    if (lp.nRows > 0) {
        CUDA_CHECK(cudaMemcpy(d_rowLen, rowLen.data(), lp.nRows * sizeof(index_t),
                              cudaMemcpyHostToDevice));
    }
    if (numRows > 0) {
        CUDA_CHECK(cudaMemcpy(d_vec, lp.vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));
    }
    std::vector<double>().swap(sellVal);
    std::vector<index_t>().swap(sellCols);


    double* h_localOut = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_localOut, std::max<size_t>(lp.nRows, 1) * sizeof(double)));
    std::vector<double> h_out(root ? numRows : 0);  // Output vector (rank 0)

    const unsigned int gridSize = nSlices;
    std::vector<double> longOut(nLong);
    auto launch = [&]() {
        if (lp.nRows > 0) {
            spmvSellKernel<<<gridSize, BLOCK_SIZE>>>(d_val, d_cols, d_sliceOff, d_rowLen, d_vec,
                                                     lp.nRows, d_out);
        }
    };

    // Warm-up (module load / caches), not timed
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Perform SpMV computation
    if (root) printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        launch();  // asynchronous
        // Long rows on the host while the GPU works (same summation order as spmvCpu)
#pragma omp parallel for schedule(dynamic, 1) if (nLong > 1)
        for (index_t k = 0; k < nLong; ++k) {
            double t = 0.0;
            for (unsigned long long j = longPtr[k]; j < longPtr[k + 1]; ++j) {
                t += longVal[j] * lp.vec[longCols[j]];
            }
            longOut[k] = t;
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (lp.nRows > 0) {
        CUDA_CHECK(cudaMemcpy(h_localOut, d_out, lp.nRows * sizeof(double), cudaMemcpyDeviceToHost));
    }
    for (index_t k = 0; k < nLong; ++k) h_localOut[longRowIdx[k]] = longOut[k];
    MPI_Gatherv(h_localOut, static_cast<int>(lp.nRows), MPI_DOUBLE, h_out.data(), rowCounts.data(),
                rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    CUDA_CHECK(cudaFreeHost(h_localOut));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_sliceOff));
    CUDA_CHECK(cudaFree(d_rowLen));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    int exitCode = 0;
    if (root) {
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
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    return finish(exitCode);
}
