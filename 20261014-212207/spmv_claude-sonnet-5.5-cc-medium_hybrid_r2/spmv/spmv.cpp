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
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// ****************************************************************************
// Random number generation
//
// The original benchmark draws from glibc's rand() in one long sequential
// stream. To parallelize while producing the identical matrix and vectors we
// reimplement the glibc TYPE_3 generator (additive feedback, r[k] = r[k-3] +
// r[k-31]) and jump ahead to arbitrary stream positions using polynomial
// arithmetic modulo x^31 - x^28 - 1 (coefficients in Z_2^32).
// ****************************************************************************
struct GlibcRand {
    uint32_t b[31];
    int p, q;

    // Position the generator so that next() returns the m-th (0-based) output
    // of rand() after srand(seed).
    void init(uint32_t seed, uint64_t m) {
        uint32_t s[64];
        if (seed == 0) seed = 1;
        s[0] = seed;
        for (int i = 1; i < 31; ++i) {
            s[i] = static_cast<uint32_t>((16807LL * static_cast<int32_t>(s[i - 1])) % 2147483647LL);
        }
        for (int i = 31; i < 34; ++i) s[i] = s[i - 31];
        for (int i = 34; i < 64; ++i) s[i] = s[i - 31] + s[i - 3];

        // c = x^e mod (x^31 - x^28 - 1), e = (344 + m - 31) - 3
        uint64_t e = 310 + m;
        uint32_t res[31] = {0}, base[31] = {0};
        res[0] = 1;
        base[1] = 1;
        while (e) {
            if (e & 1) mulmod(res, base);
            e >>= 1;
            if (e) mulmod(base, base);
        }
        for (int j = 0; j < 31; ++j) {
            uint32_t acc = 0;
            for (int i = 0; i < 31; ++i) acc += res[i] * s[3 + i + j];
            b[j] = acc;
        }
        p = 0;
        q = 28;
    }

    inline uint32_t next() {
        const uint32_t v = b[p] + b[q];
        b[p] = v;
        if (++p == 31) p = 0;
        if (++q == 31) q = 0;
        return v >> 1;
    }

   private:
    static void mulmod(uint32_t* a, const uint32_t* c) {
        uint32_t t[61] = {0};
        for (int i = 0; i < 31; ++i)
            for (int j = 0; j < 31; ++j) t[i + j] += a[i] * c[j];
        for (int d = 60; d >= 31; --d) {
            t[d - 3] += t[d];
            t[d - 31] += t[d];
        }
        for (int i = 0; i < 31; ++i) a[i] = t[i];
    }
};

constexpr uint32_t FILL_SEED = 1;  // default seed of rand() without srand()
constexpr uint32_t MATRIX_SEED = 8675309;

// ****************************************************************************
// Function: fillRange
//
// Purpose:
//   Fills out[0..count) with the values the original fill() would produce at
//   stream positions [first, first + count) of the default rand() stream.
// ****************************************************************************
static void fillRange(double* out, uint64_t first, uint64_t count, const double maxVal) {
    constexpr uint64_t BLOCK = 1 << 16;
    const int64_t nBlocks = static_cast<int64_t>((count + BLOCK - 1) / BLOCK);
#pragma omp parallel for schedule(static)
    for (int64_t blk = 0; blk < nBlocks; ++blk) {
        const uint64_t lo = static_cast<uint64_t>(blk) * BLOCK;
        const uint64_t hi = std::min(count, lo + BLOCK);
        GlibcRand rng;
        rng.init(FILL_SEED, first + lo);
        for (uint64_t i = lo; i < hi; ++i) {
            out[i] = maxVal * (rng.next() / (static_cast<double>(RAND_MAX) + 1.0));
        }
    }
}

// ****************************************************************************
// Distributed random CSR matrix generation
//
// Semantics of the original initRandomMatrix: one rand() draw per matrix cell
// in row-major order; cell is a non-zero if draw/RAND_MAX <= prob, subject to
// the cap of n non-zeros and the "fill remaining" rule at the end.
//
// Rows are split among MPI ranks and each rank splits its rows into chunks
// processed by OpenMP threads. Every chunk first counts/records candidates
// assuming no cap/fill rule is active. A replicated sequential pass over the
// chunk counts then determines for each chunk whether that assumption holds;
// the (rare) chunks where the cap or fill rule may trigger are re-run exactly
// with the original logic, and chunks after a terminal state are trivial.
// ****************************************************************************
namespace {

enum ChunkMode : int { MODE_CANDIDATES = 0, MODE_EXACT = 1, MODE_ALL = 2, MODE_NONE = 3 };

struct Chunk {
    uint64_t rowBegin = 0, rowEnd = 0;  // global rows
    std::vector<index_t> rowCnt;        // per-row nnz counts
    std::vector<index_t> cols;          // column indices (row-major)
    uint64_t candCount = 0;             // unconstrained candidate count
    uint64_t startNnz = 0;              // global nnz offset at chunk start
    uint64_t count = 0;                 // final nnz in chunk
    int mode = MODE_CANDIDATES;
};

inline uint64_t blockBegin(uint64_t total, int parts, int idx) {
    return total * static_cast<uint64_t>(idx) / static_cast<uint64_t>(parts);
}

void scanCandidates(Chunk& ch, index_t dim, double prob) {
    const uint64_t rows = ch.rowEnd - ch.rowBegin;
    ch.rowCnt.assign(rows, 0);
    ch.cols.clear();
    ch.cols.reserve(static_cast<size_t>(prob * static_cast<double>(rows) * dim * 1.05) + 64);
    GlibcRand rng;
    rng.init(MATRIX_SEED, ch.rowBegin * dim);
    for (uint64_t i = 0; i < rows; ++i) {
        index_t cnt = 0;
        for (index_t j = 0; j < dim; ++j) {
            const double randVal = static_cast<double>(rng.next()) / RAND_MAX;
            if (randVal <= prob) {
                ch.cols.push_back(j);
                ++cnt;
            }
        }
        ch.rowCnt[i] = cnt;
    }
    ch.candCount = ch.cols.size();
}

// Exact replay of the original logic for one chunk starting from nnz = startNnz
// (fill rule not yet active). Returns end state.
void runExact(Chunk& ch, index_t dim, index_t n, double prob, uint64_t startNnz,
              uint64_t& endNnz, int& fillFlag) {
    const uint64_t rows = ch.rowEnd - ch.rowBegin;
    const uint64_t total = static_cast<uint64_t>(dim) * dim;
    ch.rowCnt.assign(rows, 0);
    ch.cols.clear();
    GlibcRand rng;
    rng.init(MATRIX_SEED, ch.rowBegin * dim);
    uint64_t nnz = startNnz;
    bool fill = false;
    for (uint64_t i = 0; i < rows; ++i) {
        index_t cnt = 0;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t pos = (ch.rowBegin + i) * dim + j;
            const uint64_t left = total - pos;
            const uint64_t need = n - nnz;
            if (left <= need) fill = true;
            const double randVal = static_cast<double>(rng.next()) / RAND_MAX;
            if ((nnz < n && randVal <= prob) || fill) {
                ch.cols.push_back(j);
                ++nnz;
                ++cnt;
            }
        }
        ch.rowCnt[i] = cnt;
    }
    endNnz = nnz;
    fillFlag = fill ? 1 : 0;
}

}  // namespace

// ****************************************************************************
// CUDA kernel: CSR SpMV with T threads cooperating on each row
// ****************************************************************************
template <int T>
__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowPtr, const double* __restrict__ vec,
                           const index_t nRows, double* __restrict__ out) {
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = gid / T;
    const int lane = static_cast<int>(gid % T);
    const bool valid = row < nRows;
    index_t beg = 0, end = 0;
    if (valid) {
        beg = rowPtr[row];
        end = rowPtr[row + 1];
    }
    double t = 0.0;
    for (index_t j = beg + lane; j < end; j += T) {
        t += val[j] * __ldg(&vec[cols[j]]);
    }
#pragma unroll
    for (int off = T / 2; off > 0; off >>= 1) {
        t += __shfl_down_sync(0xffffffffu, t, off, T);
    }
    if (valid && lane == 0) out[row] = t;
}

template <int T>
static void launchSpmv(const double* val, const index_t* cols, const index_t* rowPtr,
                       const double* vec, index_t nRows, double* out, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    const size_t threads = static_cast<size_t>(nRows) * T;
    const size_t grid = (threads + BLOCK - 1) / BLOCK;
    if (grid == 0) return;
    spmvKernel<T><<<static_cast<unsigned>(grid), BLOCK, 0, stream>>>(val, cols, rowPtr, vec, nRows, out);
}

static void launchSpmvAuto(int tpr, const double* val, const index_t* cols, const index_t* rowPtr,
                           const double* vec, index_t nRows, double* out, cudaStream_t s) {
    switch (tpr) {
        case 1: launchSpmv<1>(val, cols, rowPtr, vec, nRows, out, s); break;
        case 2: launchSpmv<2>(val, cols, rowPtr, vec, nRows, out, s); break;
        case 4: launchSpmv<4>(val, cols, rowPtr, vec, nRows, out, s); break;
        case 8: launchSpmv<8>(val, cols, rowPtr, vec, nRows, out, s); break;
        case 16: launchSpmv<16>(val, cols, rowPtr, vec, nRows, out, s); break;
        default: launchSpmv<32>(val, cols, rowPtr, vec, nRows, out, s); break;
    }
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Reference sparse matrix-vector multiplication using CSR format (OpenMP).
//   Operates on a local block of rows; rowDelimiters are local offsets.
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t nRows, double* out) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(nRows); ++i) {
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
//   (indexOffset converts local indices to global row numbers in messages)
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const uint64_t indexOffset = 0) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        const unsigned long long gi = indexOffset + i;
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %llu: reference %.10e, got %.10e\n", gi, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %llu: reference %.10e, got %.10e (rel error: %.10e)\n",
                       gi, ref, res, relError);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
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
        printf("Initializing data structures...\n");
        fflush(stdout);
    }

    // Select a GPU per rank (round-robin among ranks on the same node)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank = 0;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

    const uint64_t dim = numRows;
    const uint64_t total = dim * dim;
    const double prob = static_cast<double>(nItems) / (static_cast<double>(numRows) * static_cast<double>(numRows));

    // Row partition among ranks (row blocks)
    const uint64_t myRowBegin = blockBegin(dim, nprocs, rank);
    const uint64_t myRowEnd = blockBegin(dim, nprocs, rank + 1);
    const index_t myRows = static_cast<index_t>(myRowEnd - myRowBegin);

    // Dense vector, replicated on every rank
    std::vector<double> h_vec(numRows);
    fillRange(h_vec.data(), 0, numRows, maxVal);

    // ---- Matrix structure ----
    const int K = std::max(16, 4 * omp_get_max_threads());  // chunks per rank
    const int C = K * nprocs;
    std::vector<Chunk> chunks(C);
    for (int c = 0; c < C; ++c) {
        const int r = c / K, k = c % K;
        const uint64_t rb = blockBegin(dim, nprocs, r), re = blockBegin(dim, nprocs, r + 1);
        chunks[c].rowBegin = rb + blockBegin(re - rb, K, k);
        chunks[c].rowEnd = rb + blockBegin(re - rb, K, k + 1);
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (int k = 0; k < K; ++k) {
        scanCandidates(chunks[rank * K + k], numRows, prob);
    }

    {
        std::vector<uint64_t> cnt(C);
        for (int k = 0; k < K; ++k) cnt[rank * K + k] = chunks[rank * K + k].candCount;
        MPI_Allgather(MPI_IN_PLACE, K, MPI_UINT64_T, cnt.data(), K, MPI_UINT64_T, MPI_COMM_WORLD);
        for (int c = 0; c < C; ++c) chunks[c].candCount = cnt[c];
    }

    // Replicated sequential resolution of chunk modes
    {
        uint64_t nnz = 0;
        bool fill = false;
        const uint64_t n = nItems;
        for (int c = 0; c < C; ++c) {
            Chunk& ch = chunks[c];
            const uint64_t rows = ch.rowEnd - ch.rowBegin;
            ch.startNnz = nnz;
            if (fill) {
                ch.mode = MODE_ALL;
                ch.count = rows * dim;
            } else if (nnz >= n) {
                ch.mode = MODE_NONE;
                ch.count = 0;
            } else if (rows == 0) {
                ch.mode = MODE_NONE;
                ch.count = 0;
            } else {
                const uint64_t cellsEnd = ch.rowEnd * dim;
                const bool capSafe = nnz + ch.candCount <= n;
                const bool fillSafe = static_cast<int64_t>(cellsEnd) - static_cast<int64_t>(nnz + ch.candCount) <
                                      static_cast<int64_t>(total - n);
                if (capSafe && fillSafe) {
                    ch.mode = MODE_CANDIDATES;
                    ch.count = ch.candCount;
                } else {
                    ch.mode = MODE_EXACT;
                    const int owner = c / K;
                    uint64_t st[2] = {0, 0};
                    if (owner == rank) {
                        int f = 0;
                        runExact(ch, numRows, nItems, prob, nnz, st[0], f);
                        st[1] = static_cast<uint64_t>(f);
                    }
                    MPI_Bcast(st, 2, MPI_UINT64_T, owner, MPI_COMM_WORLD);
                    ch.count = st[0] - nnz;
                    fill = st[1] != 0;
                }
            }
            nnz += ch.count;
        }
    }

    // Assemble the local CSR block
    const uint64_t sliceBegin = chunks[rank * K].startNnz;
    const uint64_t sliceEnd = chunks[rank * K + K - 1].startNnz + chunks[rank * K + K - 1].count;
    const uint64_t localNnz = sliceEnd - sliceBegin;

    std::vector<index_t> h_rowPtr(static_cast<size_t>(myRows) + 1);
    std::vector<index_t> h_cols(localNnz);
    std::vector<double> h_val(localNnz);
    std::vector<double> h_out(myRows);

#pragma omp parallel for schedule(dynamic, 1)
    for (int k = 0; k < K; ++k) {
        Chunk& ch = chunks[rank * K + k];
        const uint64_t rows = ch.rowEnd - ch.rowBegin;
        const uint64_t rowOff = ch.rowBegin - myRowBegin;
        uint64_t off = ch.startNnz - sliceBegin;
        const uint64_t base = off;
        if (ch.mode == MODE_ALL) {
            for (uint64_t i = 0; i < rows; ++i) {
                h_rowPtr[rowOff + i] = static_cast<index_t>(off);
                for (index_t j = 0; j < numRows; ++j) h_cols[off + j] = j;
                off += numRows;
            }
        } else if (ch.mode == MODE_NONE) {
            for (uint64_t i = 0; i < rows; ++i) h_rowPtr[rowOff + i] = static_cast<index_t>(off);
        } else {
            for (uint64_t i = 0; i < rows; ++i) {
                h_rowPtr[rowOff + i] = static_cast<index_t>(off);
                off += ch.rowCnt[i];
            }
            if (!ch.cols.empty()) memcpy(h_cols.data() + base, ch.cols.data(), ch.cols.size() * sizeof(index_t));
        }
        std::vector<index_t>().swap(ch.cols);
        std::vector<index_t>().swap(ch.rowCnt);
    }
    h_rowPtr[myRows] = static_cast<index_t>(localNnz);

    // Matrix values: stream continues after the vector
    fillRange(h_val.data(), dim + sliceBegin, localNnz, maxVal);

    // ---- Device setup ----
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rowPtr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val, std::max<size_t>(localNnz, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, std::max<size_t>(localNnz, 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowPtr, (static_cast<size_t>(myRows) + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, std::max<size_t>(numRows, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max<size_t>(myRows, 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowPtr, h_rowPtr.data(), (static_cast<size_t>(myRows) + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), static_cast<size_t>(numRows) * sizeof(double), cudaMemcpyHostToDevice));

    // Threads per row: power of two closest to the average row length
    const double avgRow = myRows ? static_cast<double>(localNnz) / myRows : 0.0;
    int tpr = 1;
    while (tpr < 32 && tpr * 1.5 < avgRow) tpr <<= 1;

    // For validation, compute reference solution (each rank checks its rows)
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
            fflush(stdout);
        }
        h_reference.resize(myRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowPtr.data(), h_vec.data(), myRows, h_reference.data());
    }

    // Warm-up of the device context/kernel (does not alter results)
    launchSpmvAuto(tpr, d_val, d_cols, d_rowPtr, d_vec, myRows, d_out, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform SpMV computation
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computing SpMV...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpmvAuto(tpr, d_val, d_cols, d_rowPtr, d_vec, myRows, d_out, 0);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double seconds = std::chrono::duration<double>(end - start).count();
    MPI_Bcast(&seconds, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    long long ms = duration.count();
    MPI_Bcast(&ms, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, static_cast<size_t>(myRows) * sizeof(double), cudaMemcpyDeviceToHost));

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(ms));

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / seconds / 1e9;
        const double avgTime = seconds * 1000.0 / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<int> counts(nprocs), displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            counts[r] = static_cast<int>(blockBegin(dim, nprocs, r + 1) - blockBegin(dim, nprocs, r));
            displs[r] = static_cast<int>(blockBegin(dim, nprocs, r));
        }
        std::vector<double> full;
        if (rank == 0) full.resize(numRows);
        MPI_Gatherv(h_out.data(), static_cast<int>(myRows), MPI_DOUBLE, full.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(full, "OutputVector");
    }

    int exitCode = 0;
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        int ok = verifyResults(h_reference.data(), h_out.data(), myRows, myRowBegin) ? 1 : 0;
        int allOk = 0;
        MPI_Allreduce(&ok, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf(allOk ? "Validation: PASSED\n" : "Validation: FAILED\n");
        exitCode = allOk ? 0 : 1;
    }

    cudaFree(d_val);
    cudaFree(d_cols);
    cudaFree(d_rowPtr);
    cudaFree(d_vec);
    cudaFree(d_out);
    MPI_Finalize();
    return exitCode;
}
