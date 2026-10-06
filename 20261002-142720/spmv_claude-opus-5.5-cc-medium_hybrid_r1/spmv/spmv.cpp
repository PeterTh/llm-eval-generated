#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr uint32_t GLIBC_RAND_MAX = 2147483647u;  // RAND_MAX of glibc rand()
constexpr unsigned SLICE = 32;                    // SELL slice height (= warp size)
constexpr unsigned SLICE_SHIFT = 5;
constexpr int BLOCK_SIZE = 128;
constexpr index_t LONG_ROW = 0xffffffffu;         // rowLen marker for rows handled separately

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// ****************************************************************************
// Class: GlibcRand
//
// Purpose:
//   Bit-exact reimplementation of glibc's rand() (TYPE_3 additive feedback
//   generator, r[i] = r[i-31] + r[i-3]) with O(log n) jump-ahead, so that
//   independent threads/ranks can produce any slice of the sequence that
//   the original serial code obtains from srand()/rand().
//
//   Output number k (0-based) after srand(seed) equals r[k + 344] >> 1.
//
// ****************************************************************************
class GlibcRand {
  public:
    GlibcRand(uint32_t seed, uint64_t outputIndex) {
        uint32_t init[64];
        int32_t word = static_cast<int32_t>(seed == 0 ? 1 : seed);
        init[0] = static_cast<uint32_t>(word);
        for (int i = 1; i < 31; ++i) {
            const int32_t hi = word / 127773;
            const int32_t lo = word % 127773;
            word = 16807 * lo - 2836 * hi;
            if (word < 0) word += 2147483647;
            init[i] = static_cast<uint32_t>(word);
        }
        for (int i = 31; i < 34; ++i) init[i] = init[i - 31];
        for (int i = 34; i < 64; ++i) init[i] = init[i - 31] + init[i - 3];

        // We need r[t-31 .. t-1] with t = outputIndex + 344. Since the
        // recurrence r[s+31] = r[s+28] + r[s] holds for s >= 3, use the
        // polynomial x^k mod (x^31 - x^28 - 1) with base r[3..33].
        const uint64_t k = outputIndex + 344 - 34;
        uint32_t c[31];
        polyPowX(k, c);
        for (int m = 0; m < 31; ++m) {
            uint32_t s = 0;
            for (int j = 0; j < 31; ++j) s += c[j] * init[3 + m + j];
            buf_[m] = s;  // r[t-31+m] = sum_j c_j r[3+m+j]
        }
        pos_ = 31;
    }

    // Returns the next value exactly as glibc rand() would.
    inline uint32_t next() {
        // buffer index i&31 holds r[i]; r[i-31] at (i+1)&31, r[i-3] at (i-3)&31
        const uint32_t v = buf_[(pos_ + 1) & 31] + buf_[(pos_ - 3) & 31];
        buf_[pos_ & 31] = v;
        ++pos_;
        return v >> 1;
    }

  private:
    uint32_t buf_[32];
    uint32_t pos_;

    static void polyMulMod(const uint32_t* a, const uint32_t* b, uint32_t* out) {
        uint32_t p[61] = {0};
        for (int i = 0; i < 31; ++i) {
            if (a[i] == 0) continue;
            for (int j = 0; j < 31; ++j) p[i + j] += a[i] * b[j];
        }
        for (int d = 60; d >= 31; --d) {
            p[d - 3] += p[d];
            p[d - 31] += p[d];
        }
        for (int i = 0; i < 31; ++i) out[i] = p[i];
    }

    static void polyPowX(uint64_t k, uint32_t* res) {
        uint32_t base[31] = {0};
        for (int i = 0; i < 31; ++i) res[i] = 0;
        res[0] = 1;
        base[1] = 1;
        uint32_t tmp[31];
        while (k) {
            if (k & 1) {
                polyMulMod(res, base, tmp);
                memcpy(res, tmp, sizeof(tmp));
            }
            k >>= 1;
            if (k) {
                polyMulMod(base, base, tmp);
                memcpy(base, tmp, sizeof(tmp));
            }
        }
    }
};

// ****************************************************************************
// Function: fillRange
//
// Purpose:
//   Produces elements [offset, offset+count) of the sequence that the
//   original fill() writes when it is called after `skip` previous rand()
//   calls on the default seed (1). Parallelized with OpenMP via jump-ahead.
//
// ****************************************************************************
void fillRange(double* A, const uint64_t skip, const uint64_t count, const double maxVal) {
#pragma omp parallel
    {
        const uint64_t nt = omp_get_num_threads();
        const uint64_t t = omp_get_thread_num();
        const uint64_t b = count * t / nt;
        const uint64_t e = count * (t + 1) / nt;
        if (b < e) {
            GlibcRand rng(1, skip + b);
            for (uint64_t i = b; i < e; ++i) {
                A[i] = maxVal * (rng.next() / (static_cast<double>(GLIBC_RAND_MAX) + 1.0));
            }
        }
    }
}

// ****************************************************************************
// Function: initRandomMatrixLocal
//
// Purpose:
//   Builds rows [rowBegin, rowEnd) of the CSR matrix that the original
//   serial initRandomMatrix() generates (identical result). Rows are first
//   generated in parallel ignoring the global "nnz cap" / "fill remaining"
//   rules; afterwards the exact global state is checked and, from the first
//   row where those rules kick in (typically only the last few rows), the
//   generation is redone with the exact serial semantics.
//
// Arguments:
//   cols:      output - column indexes of the local nonzeros
//   rowPtr:    output - local row delimiters (size localRows+1, from 0)
//   n:         number of nonzero elements (global)
//   dim:       number of rows/columns in the matrix
//
// ****************************************************************************
void initRandomMatrixLocal(std::vector<index_t>& cols, std::vector<uint64_t>& rowPtr,
                           const index_t n, const index_t dim, const index_t rowBegin,
                           const index_t rowEnd, MPI_Comm comm) {
    const index_t localRows = rowEnd - rowBegin;
    const uint32_t seed = 8675309;

    // Figure out the probability that a nonzero should be assigned
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));
    // Integer threshold: largest x with x / RAND_MAX <= prob (exactly as the double compare)
    int64_t thr;
    {
        int64_t lo = -1, hi = GLIBC_RAND_MAX;  // invariant: pass(lo) (or lo=-1), !pass(hi+1)
        if (static_cast<double>(GLIBC_RAND_MAX) / GLIBC_RAND_MAX <= prob) {
            lo = GLIBC_RAND_MAX;
        } else {
            while (hi - lo > 1) {
                const int64_t mid = lo + (hi - lo) / 2;
                if (static_cast<double>(mid) / GLIBC_RAND_MAX <= prob) lo = mid; else hi = mid;
            }
        }
        thr = lo;
    }
    const uint32_t thrU = static_cast<uint32_t>(thr < 0 ? 0 : thr);
    const bool anyHit = thr >= 0;

    // ---- Phase 1: uncapped parallel generation of the local rows ----
    std::vector<uint64_t> rowCount(localRows + 1, 0);
    const int maxThreads = omp_get_max_threads();
    std::vector<std::vector<index_t>> threadCols(maxThreads);
    std::vector<index_t> threadRowBegin(maxThreads + 1, localRows);

#pragma omp parallel
    {
        const index_t nt = omp_get_num_threads();
        const index_t t = omp_get_thread_num();
        const index_t rb = static_cast<index_t>(static_cast<uint64_t>(localRows) * t / nt);
        const index_t re = static_cast<index_t>(static_cast<uint64_t>(localRows) * (t + 1) / nt);
        threadRowBegin[t] = rb;
        if (t == nt - 1) threadRowBegin[nt] = localRows;
        auto& my = threadCols[t];
        my.reserve(static_cast<size_t>((re - rb) * (static_cast<double>(dim) * prob * 1.05 + 16)));
        if (rb < re) {
            GlibcRand rng(seed, static_cast<uint64_t>(rowBegin + rb) * dim);
            for (index_t r = rb; r < re; ++r) {
                const size_t before = my.size();
                for (index_t j = 0; j < dim; ++j) {
                    const uint32_t v = rng.next();
                    if (anyHit && v <= thrU) my.push_back(j);
                }
                rowCount[r + 1] = my.size() - before;
            }
        }
#pragma omp barrier
#pragma omp single
        {
            for (index_t r = 0; r < localRows; ++r) rowCount[r + 1] += rowCount[r];
        }
    }
    rowPtr.swap(rowCount);  // rowPtr[r] = local prefix (uncapped)
    const uint64_t localNnz = rowPtr[localRows];
    cols.resize(localNnz);
    {
        const int nt = static_cast<int>(threadCols.size());
#pragma omp parallel for schedule(static, 1)
        for (int t = 0; t < nt; ++t) {
            if (!threadCols[t].empty()) {
                const uint64_t off = rowPtr[threadRowBegin[t]];
                memcpy(cols.data() + off, threadCols[t].data(), threadCols[t].size() * sizeof(index_t));
            }
            std::vector<index_t>().swap(threadCols[t]);
        }
    }

    // Global nnz offset of my first row (assuming no capping before me)
    uint64_t globalStart = 0;
    MPI_Exscan(&localNnz, &globalStart, 1, MPI_UINT64_T, MPI_SUM, comm);
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) globalStart = 0;

    // ---- Phase 2: find the first row where the serial cap/fill rules deviate ----
    const index_t D = dim * dim;  // same (wrapping) arithmetic as the original
    uint64_t firstDirty = UINT64_MAX;
#pragma omp parallel for schedule(static) reduction(min : firstDirty)
    for (index_t r = 0; r < localRows; ++r) {
        const index_t i = rowBegin + r;
        const index_t rowBase = i * dim;
        index_t nnz = static_cast<index_t>(globalStart + rowPtr[r]);
        index_t p = 0;
        bool dirty = false;
        const uint64_t hb = rowPtr[r], he = rowPtr[r + 1];
        for (uint64_t h = hb; h <= he && !dirty; ++h) {
            // segment [p, q]; q is the next hit column (check happens before the decision)
            const bool isHit = h < he;
            const index_t q = isHit ? cols[h] : dim - 1;
            if (!isHit && p > q) break;  // empty final segment
            const index_t need = n - nnz;
            const index_t leftP = D - (rowBase + p);
            const index_t leftQ = D - (rowBase + q);
            if (leftP < q - p || leftQ <= need) dirty = true;  // fillRemaining triggers
            if (isHit) {
                if (nnz >= n) dirty = true;  // cap reached: hit would not be assigned
                ++nnz;
                p = q + 1;
            }
        }
        if (dirty && i < firstDirty) firstDirty = i;
    }

    // Global first dirty row along with its exact starting nnz
    uint64_t key = UINT64_MAX;
    if (firstDirty != UINT64_MAX) {
        const index_t r = static_cast<index_t>(firstDirty) - rowBegin;
        key = (firstDirty << 32) | static_cast<uint32_t>(globalStart + rowPtr[r]);
    }
    uint64_t gkey;
    MPI_Allreduce(&key, &gkey, 1, MPI_UINT64_T, MPI_MIN, comm);
    if (gkey == UINT64_MAX) return;

    const index_t d = static_cast<index_t>(gkey >> 32);
    if (rowEnd <= d) return;  // my rows are unaffected

    // ---- Phase 3: exact serial redo from row d up to the end of my range ----
    index_t nnz = static_cast<index_t>(gkey & 0xffffffffu);
    const index_t keepFrom = d > rowBegin ? d : rowBegin;
    rowPtr.resize(localRows + 1);
    cols.resize(rowPtr[keepFrom - rowBegin]);
    GlibcRand rng(seed, static_cast<uint64_t>(d) * dim);
    bool fillRemaining = false;
    bool rngLive = true;  // once fill/cap is reached, random values never matter again
    for (index_t i = d; i < rowEnd; ++i) {
        const bool keep = i >= rowBegin;
        if (keep) rowPtr[i - rowBegin] = cols.size();
        for (index_t j = 0; j < dim; ++j) {
            const index_t numEntriesLeft = D - ((i * dim) + j);
            const index_t needToAssign = n - nnz;
            if (numEntriesLeft <= needToAssign) fillRemaining = true;
            bool assign;
            if (fillRemaining) {
                assign = true;
            } else if (nnz < n) {
                assign = anyHit && rng.next() <= thrU;
            } else {
                assign = false;
            }
            if (rngLive && (fillRemaining || nnz >= n)) rngLive = false;
            if (assign) {
                if (keep) cols.push_back(j);
                ++nnz;
            }
        }
        (void)rngLive;
    }
    rowPtr[localRows] = cols.size();
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
void spmvCpu(const double* val, const index_t* cols, const uint64_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (uint64_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Function: spmvSellKernel
//
// Purpose:
//   GPU SpMV on a SELL-32 layout: one thread per row, element k of the 32
//   rows of a slice stored contiguously, giving fully coalesced loads while
//   keeping the sequential per-row summation order of the CPU code.
//   Rows flagged LONG_ROW are computed on the host (see run()).
//
// ****************************************************************************
__global__ void __launch_bounds__(BLOCK_SIZE)
spmvSellKernel(const index_t rows, const uint64_t* __restrict__ sliceOffsets,
               const index_t* __restrict__ rowLen, const index_t* __restrict__ cols,
               const double* __restrict__ val, const double* __restrict__ vec,
               double* __restrict__ out) {
    const index_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const index_t len = rowLen[r];
    if (len == LONG_ROW) return;
    const index_t* c = cols + sliceOffsets[r >> SLICE_SHIFT] + (r & (SLICE - 1));
    const double* v = val + sliceOffsets[r >> SLICE_SHIFT] + (r & (SLICE - 1));
    // Separate (non-fused) multiply and add to round exactly like the CPU code
    double t = 0.0;
    index_t k = 0;
    constexpr index_t U = 8;
    for (; k + U <= len; k += U) {
        index_t cc[U];
        double vv[U], xx[U];
#pragma unroll
        for (index_t u = 0; u < U; ++u) cc[u] = __ldcs(c + u * SLICE);
#pragma unroll
        for (index_t u = 0; u < U; ++u) vv[u] = __ldcs(v + u * SLICE);
#pragma unroll
        for (index_t u = 0; u < U; ++u) xx[u] = __ldg(vec + cc[u]);
#pragma unroll
        for (index_t u = 0; u < U; ++u) t = __dadd_rn(t, __dmul_rn(vv[u], xx[u]));
        c += U * SLICE;
        v += U * SLICE;
    }
    for (; k < len; ++k) {
        t = __dadd_rn(t, __dmul_rn(__ldcs(v), __ldg(vec + __ldcs(c))));
        c += SLICE;
        v += SLICE;
    }
    out[r] = t;
}

// ****************************************************************************
// Function: csrToSellKernel
//
// Purpose:
//   Scatters the (device-resident) local CSR matrix into the SELL-32 layout.
//
// ****************************************************************************
__global__ void csrToSellKernel(const index_t rows, const uint64_t* __restrict__ csrPtr,
                                const index_t* __restrict__ csrCols, const double* __restrict__ csrVal,
                                const uint64_t* __restrict__ sliceOffsets, const index_t* __restrict__ rowLen,
                                index_t* __restrict__ sellCols, double* __restrict__ sellVal) {
    const index_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const index_t len = rowLen[r];
    if (len == LONG_ROW) return;
    const uint64_t base = sliceOffsets[r >> SLICE_SHIFT] + (r & (SLICE - 1));
    const uint64_t src = csrPtr[r];
    for (index_t k = 0; k < len; ++k) {
        sellCols[base + static_cast<uint64_t>(k) * SLICE] = csrCols[src + k];
        sellVal[base + static_cast<uint64_t>(k) * SLICE] = csrVal[src + k];
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

int run(int argc, char** argv, const int rank, const int nranks) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
    const bool root = rank == 0;

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
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
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
        fflush(stdout);
    }

    // Row-block decomposition across ranks
    const index_t rowBegin = static_cast<index_t>(static_cast<uint64_t>(numRows) * rank / nranks);
    const index_t rowEnd = static_cast<index_t>(static_cast<uint64_t>(numRows) * (rank + 1) / nranks);
    const index_t localRows = rowEnd - rowBegin;
    std::vector<int> recvCounts(nranks), displs(nranks);
    for (int p = 0; p < nranks; ++p) {
        const index_t b = static_cast<index_t>(static_cast<uint64_t>(numRows) * p / nranks);
        const index_t e = static_cast<index_t>(static_cast<uint64_t>(numRows) * (p + 1) / nranks);
        recvCounts[p] = static_cast<int>(e - b);
        displs[p] = static_cast<int>(b);
    }

    // Select GPU: one per rank within each node
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "No CUDA device found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));
    CUDA_CHECK(cudaFree(nullptr));  // create context early

    if (root) {
        printf("Initializing data structures...\n");
        fflush(stdout);
    }

    // Dense vector (full copy per rank; rand() calls 0 .. numRows-1)
    std::vector<double> h_vec(numRows);
    fillRange(h_vec.data(), 0, numRows, maxVal);

    // Local rows of the matrix structure
    std::vector<index_t> h_cols;
    std::vector<uint64_t> h_rowPtr;
    initRandomMatrixLocal(h_cols, h_rowPtr, nItems, numRows, rowBegin, rowEnd, MPI_COMM_WORLD);
    const uint64_t localNnz = h_rowPtr[localRows];
    uint64_t globalStart = 0;
    MPI_Exscan(&localNnz, &globalStart, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (root) globalStart = 0;

    // Local nonzero values (rand() calls numRows + globalStart ...)
    std::vector<double> h_val(localNnz);
    fillRange(h_val.data(), static_cast<uint64_t>(numRows) + globalStart, localNnz, maxVal);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (root) printf("Computing reference solution...\n");
        std::vector<double> localRef(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowPtr.data(), h_vec.data(), localRows, localRef.data());
        if (root) h_reference.resize(numRows);
        MPI_Gatherv(localRef.data(), static_cast<int>(localRows), MPI_DOUBLE, h_reference.data(),
                    recvCounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Split rows: regular rows go to a SELL-32 layout on the GPU (thread per
    // row). The rare rows that are much longer than average (e.g. the dense
    // rows the generator's "fill remaining" rule may create at the end) form
    // a long serial dependency chain that GPUs execute slowly in FP64; they
    // are computed by OpenMP threads on the host, overlapped with the GPU.
    const double avgLen = localRows ? static_cast<double>(localNnz) / localRows : 0.0;
    const uint64_t longThreshold = static_cast<uint64_t>(std::max(256.0, 2.0 * avgLen));
    const index_t numSlices = (localRows + SLICE - 1) / SLICE;
    std::vector<uint64_t> h_sliceOff(numSlices + 1, 0);
    std::vector<index_t> h_rowLen(static_cast<size_t>(numSlices) * SLICE, 0);
    std::vector<index_t> h_longIds;
#pragma omp parallel for schedule(static)
    for (index_t s = 0; s < numSlices; ++s) {
        uint64_t maxLen = 0;
        for (index_t r = s * SLICE; r < (s + 1) * SLICE && r < localRows; ++r) {
            const uint64_t len = h_rowPtr[r + 1] - h_rowPtr[r];
            if (len > longThreshold) {
                h_rowLen[r] = LONG_ROW;
            } else {
                h_rowLen[r] = static_cast<index_t>(len);
                if (len > maxLen) maxLen = len;
            }
        }
        h_sliceOff[s + 1] = maxLen * SLICE;
    }
    for (index_t s = 0; s < numSlices; ++s) h_sliceOff[s + 1] += h_sliceOff[s];
    const uint64_t sellSize = h_sliceOff[numSlices];

    std::vector<uint64_t> h_longPtr(1, 0);
    for (index_t r = 0; r < localRows; ++r) {
        if (h_rowLen[r] == LONG_ROW) {
            h_longIds.push_back(r);
            h_longPtr.push_back(h_longPtr.back() + (h_rowPtr[r + 1] - h_rowPtr[r]));
        }
    }
    const index_t numLong = static_cast<index_t>(h_longIds.size());
    const uint64_t longNnz = h_longPtr.back();
    std::vector<index_t> h_longCols(longNnz);
    std::vector<double> h_longVal(longNnz);
    for (index_t l = 0; l < numLong; ++l) {
        const uint64_t src = h_rowPtr[h_longIds[l]];
        const uint64_t len = h_longPtr[l + 1] - h_longPtr[l];
        memcpy(h_longCols.data() + h_longPtr[l], h_cols.data() + src, len * sizeof(index_t));
        memcpy(h_longVal.data() + h_longPtr[l], h_val.data() + src, len * sizeof(double));
    }

    auto devAlloc = [](auto** ptr, size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr), std::max<size_t>(count, 1) * sizeof(**ptr)));
    };
    auto upload = [](auto* dst, const auto& src) {
        if (!src.empty())
            CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(src[0]), cudaMemcpyHostToDevice));
    };

    index_t* d_cols = nullptr;
    double* d_val = nullptr;
    uint64_t* d_sliceOff = nullptr;
    index_t* d_rowLen = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    devAlloc(&d_cols, sellSize);
    devAlloc(&d_val, sellSize);
    devAlloc(&d_sliceOff, h_sliceOff.size());
    devAlloc(&d_rowLen, h_rowLen.size());
    devAlloc(&d_vec, numRows);
    devAlloc(&d_out, localRows);
    upload(d_sliceOff, h_sliceOff);
    upload(d_rowLen, h_rowLen);
    upload(d_vec, h_vec);

    const unsigned gridSize = (localRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    {
        // Upload CSR and convert to SELL on the device (padding stays zero)
        uint64_t* d_csrPtr = nullptr;
        index_t* d_csrCols = nullptr;
        double* d_csrVal = nullptr;
        devAlloc(&d_csrPtr, h_rowPtr.size());
        devAlloc(&d_csrCols, h_cols.size());
        devAlloc(&d_csrVal, h_val.size());
        upload(d_csrPtr, h_rowPtr);
        upload(d_csrCols, h_cols);
        upload(d_csrVal, h_val);
        CUDA_CHECK(cudaMemset(d_cols, 0, std::max<uint64_t>(sellSize, 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMemset(d_val, 0, std::max<uint64_t>(sellSize, 1) * sizeof(double)));
        if (gridSize > 0) {
            csrToSellKernel<<<gridSize, BLOCK_SIZE>>>(localRows, d_csrPtr, d_csrCols, d_csrVal, d_sliceOff,
                                                      d_rowLen, d_cols, d_val);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_csrPtr));
        CUDA_CHECK(cudaFree(d_csrCols));
        CUDA_CHECK(cudaFree(d_csrVal));
    }
    // Host matrix copies are only needed for the reference solution
    std::vector<index_t>().swap(h_cols);
    std::vector<double>().swap(h_val);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<double> h_longOut(numLong);

    // One SpMV: asynchronous GPU launch for the regular rows, long rows on the host
    auto spmvHybrid = [&]() {
        if (gridSize > 0) {
            spmvSellKernel<<<gridSize, BLOCK_SIZE, 0, stream>>>(localRows, d_sliceOff, d_rowLen, d_cols, d_val,
                                                                d_vec, d_out);
        }
        if (numLong > 0) {
#pragma omp parallel for schedule(dynamic, 1) if (numLong > 1)
            for (index_t l = 0; l < numLong; ++l) {
                double t = 0.0;
                for (uint64_t j = h_longPtr[l]; j < h_longPtr[l + 1]; ++j) {
                    t += h_longVal[j] * h_vec[h_longCols[j]];
                }
                h_longOut[l] = t;
            }
        }
    };

    // Warm-up launch (module load, clocks); result is overwritten by the timed loop
    spmvHybrid();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> h_localOut(localRows);
    std::vector<double> h_out(root ? numRows : 0);  // Output vector (on root)

    // Perform SpMV computation
    if (root) {
        printf("Computing SpMV...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvHybrid();
    }
    CUDA_CHECK(cudaGetLastError());
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpyAsync(h_localOut.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost,
                                   stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (index_t l = 0; l < numLong; ++l) h_localOut[h_longIds[l]] = h_longOut[l];
    MPI_Gatherv(h_localOut.data(), static_cast<int>(localRows), MPI_DOUBLE, h_out.data(),
                recvCounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    CUDA_CHECK(cudaStreamDestroy(stream));
    for (void* p : {static_cast<void*>(d_cols), static_cast<void*>(d_val), static_cast<void*>(d_sliceOff),
                    static_cast<void*>(d_rowLen), static_cast<void*>(d_vec), static_cast<void*>(d_out)}) {
        CUDA_CHECK(cudaFree(p));
    }

    if (!root) return 0;

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

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int rc = run(argc, argv, rank, nranks);
    fflush(stdout);
    // Propagate root's exit status to all ranks
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
