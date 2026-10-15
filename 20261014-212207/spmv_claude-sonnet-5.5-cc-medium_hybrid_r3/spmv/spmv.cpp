#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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
// Random number stream
//
// The original program uses glibc rand(). To obtain exactly the same matrix and
// values while generating them in parallel, glibc's TYPE_3 additive lagged
// Fibonacci generator (r[i] = r[i-3] + r[i-31], output = r[i] >> 1) is
// reimplemented here together with a jump-ahead (x^N mod x^31 - x^28 - 1 over
// Z/2^32), so any rank/thread can start at an arbitrary position of the stream.
// ****************************************************************************
constexpr int LFG_N = 31;

struct LfgBase {
    uint32_t s[LFG_N];  // s[j] = r[3 + j] of the glibc state after seeding
    explicit LfgBase(uint32_t seed) {
        int32_t r[344];
        if (seed == 0) seed = 1;
        r[0] = static_cast<int32_t>(seed);
        for (int i = 1; i < 31; ++i) {
            int64_t hi = r[i - 1] / 127773;
            int64_t lo = r[i - 1] % 127773;
            int64_t word = 16807 * lo - 2836 * hi;
            if (word < 0) word += 2147483647;
            r[i] = static_cast<int32_t>(word);
        }
        uint32_t u[344];
        for (int i = 0; i < 31; ++i) u[i] = static_cast<uint32_t>(r[i]);
        for (int i = 31; i < 34; ++i) u[i] = u[i - 31];
        for (int i = 34; i < 344; ++i) u[i] = u[i - 31] + u[i - 3];
        for (int j = 0; j < LFG_N; ++j) s[j] = u[3 + j];
    }
};

static void polyMulMod(const uint32_t* a, const uint32_t* b, uint32_t* out) {
    uint32_t c[2 * LFG_N - 1] = {0};
    for (int i = 0; i < LFG_N; ++i)
        for (int j = 0; j < LFG_N; ++j) c[i + j] += a[i] * b[j];
    for (int d = 2 * LFG_N - 2; d >= LFG_N; --d) {  // x^31 = x^28 + 1
        c[d - 3] += c[d];
        c[d - LFG_N] += c[d];
    }
    for (int i = 0; i < LFG_N; ++i) out[i] = c[i];
}

struct Lfg {
    uint32_t w[LFG_N];
    int p;
    // Position the generator so that the next call to next() returns the
    // rand() value with index 'pos' (0-based) after srand(seed).
    Lfg(const LfgBase& base, uint64_t pos) : p(0) {
        uint64_t e = 310 + pos;
        uint32_t result[LFG_N] = {0}, sq[LFG_N] = {0};
        result[0] = 1;
        sq[1] = 1;  // x
        while (e) {
            if (e & 1) polyMulMod(result, sq, result);
            e >>= 1;
            if (e) polyMulMod(sq, sq, sq);
        }
        for (int j = 0; j < LFG_N; ++j) {
            uint32_t acc = 0;
            for (int m = 0; m < LFG_N; ++m) acc += result[m] * base.s[m];
            w[j] = acc;
            const uint32_t t = result[LFG_N - 1];  // multiply by x
            for (int m = LFG_N - 1; m > 0; --m) result[m] = result[m - 1];
            result[0] = t;
            result[28] += t;
        }
    }
    inline uint32_t next() {
        const int q = (p + 28 >= LFG_N) ? p - 3 : p + 28;
        const uint32_t v = w[p] + w[q];
        w[p] = v;
        p = (p + 1 == LFG_N) ? 0 : p + 1;
        return v >> 1;
    }
};

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values (identical to the serial rand() based
//   fill after the default seed, starting at stream offset 'offset')
//
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal, const uint64_t offset) {
    static const LfgBase base(1);
    constexpr index_t CH = 1 << 16;
    const index_t nch = (n + CH - 1) / CH;
#pragma omp parallel for schedule(static)
    for (index_t c = 0; c < nch; ++c) {
        const index_t b = c * CH, e = std::min(n, b + CH);
        Lfg g(base, offset + b);
        for (index_t i = b; i < e; ++i) {
            A[i] = maxVal * (g.next() / (static_cast<double>(RAND_MAX) + 1.0));
        }
    }
}

// ****************************************************************************
// Matrix generation (same sequential semantics as the original
// initRandomMatrix, parallelized over chunks of rows)
//
// State carried along the (row-major) entry sequence: number of assigned
// nonzeros and the "fillRemaining" flag.
// ****************************************************************************
struct GenState {
    index_t nnz;
    bool fill;
};

struct MatrixGen {
    index_t n, dim;
    double prob;
    LfgBase base;
    MatrixGen(index_t n_, index_t dim_) : n(n_), dim(dim_), base(8675309) {
        prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));
    }

    // Number of random draws with randVal <= prob in rows [r0, r1), ignoring the cap/fill logic
    index_t countRaw(index_t r0, index_t r1) const {
        Lfg g(base, static_cast<uint64_t>(r0) * dim);
        const uint64_t total = static_cast<uint64_t>(r1 - r0) * dim;
        index_t c = 0;
        for (uint64_t k = 0; k < total; ++k) {
            const double randVal = static_cast<double>(g.next()) / RAND_MAX;
            c += (randVal <= prob);
        }
        return c;
    }

    // Exact state machine of the original loop over rows [r0, r1). If cols != nullptr,
    // writes column indices (offset by colBase) and row delimiters (global values) into rowDel
    // (indexed relative to rowBase).
    GenState run(index_t r0, index_t r1, GenState st, index_t* cols, index_t colBase,
                 index_t* rowDel, index_t rowBase) const {
        const bool write = cols != nullptr;
        Lfg g(base, static_cast<uint64_t>(r0) * dim);
        for (index_t i = r0; i < r1; ++i) {
            if (write) rowDel[i - rowBase] = st.nnz;
            if (st.fill) {  // every remaining entry is assigned, no randomness needed
                if (write)
                    for (index_t j = 0; j < dim; ++j) cols[st.nnz - colBase + j] = j;
                st.nnz += dim;
                continue;
            }
            if (st.nnz >= n) {  // nothing more can be assigned (and fill can never trigger)
                if (write)
                    for (index_t k = i + 1; k < r1; ++k) rowDel[k - rowBase] = st.nnz;
                return st;
            }
            for (index_t j = 0; j < dim; ++j) {
                const index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
                const index_t needToAssign = n - st.nnz;
                if (numEntriesLeft <= needToAssign) st.fill = true;
                const double randVal = static_cast<double>(g.next()) / RAND_MAX;
                if ((st.nnz < n && randVal <= prob) || st.fill) {
                    if (write) cols[st.nnz - colBase] = j;
                    st.nnz++;
                }
            }
        }
        return st;
    }
};

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
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t globalOffset) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", globalOffset + i, ref, res);
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


// ****************************************************************************
// GPU kernel: CSR SpMV with TPR threads cooperating on each row
// ****************************************************************************
template <int TPR>
__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDel, const double* __restrict__ vec,
                           const index_t rows, double* __restrict__ out) {
    const index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t row = tid / TPR;
    const int lane = threadIdx.x % TPR;
    double t = 0.0;
    if (row < rows) {
        const index_t end = rowDel[row + 1];
        for (index_t j = rowDel[row] + lane; j < end; j += TPR) {
            t += val[j] * __ldg(&vec[cols[j]]);
        }
    }
#pragma unroll
    for (int o = TPR / 2; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o, TPR);
    if (lane == 0 && row < rows) out[row] = t;
}

template <int TPR>
static void launchSpmv(const double* val, const index_t* cols, const index_t* rowDel,
                       const double* vec, index_t rows, double* out) {
    constexpr int BLOCK = 128;
    const uint64_t threads = static_cast<uint64_t>(rows) * TPR;
    const unsigned blocks = static_cast<unsigned>((threads + BLOCK - 1) / BLOCK);
    if (blocks > 0) spmvKernel<TPR><<<blocks, BLOCK>>>(val, cols, rowDel, vec, rows, out);
}

static void spmvGpu(int tpr, const double* val, const index_t* cols, const index_t* rowDel,
                    const double* vec, index_t rows, double* out) {
    switch (tpr) {
        case 1: launchSpmv<1>(val, cols, rowDel, vec, rows, out); break;
        case 2: launchSpmv<2>(val, cols, rowDel, vec, rows, out); break;
        case 4: launchSpmv<4>(val, cols, rowDel, vec, rows, out); break;
        case 8: launchSpmv<8>(val, cols, rowDel, vec, rows, out); break;
        case 16: launchSpmv<16>(val, cols, rowDel, vec, rows, out); break;
        default: launchSpmv<32>(val, cols, rowDel, vec, rows, out); break;
    }
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
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // One GPU per rank (round-robin over the GPUs of the node)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank = 0, ndev = 0;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
        CUDA_CHECK(cudaFree(nullptr));
    }

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

    // ---- Work decomposition: chunks of rows, contiguous chunk ranges per rank ----
    const int nthreads = omp_get_max_threads();
    index_t chunkRows = numRows / (static_cast<index_t>(nranks) * nthreads * 8);
    if (numRows > 0) chunkRows = std::min<index_t>(chunkRows, (1u << 22) / numRows);
    chunkRows = std::max<index_t>(chunkRows, 1);
    const index_t nChunks = (numRows + chunkRows - 1) / chunkRows;
    std::vector<index_t> chunkLoOf(nranks + 1);
    for (int r = 0; r <= nranks; ++r)
        chunkLoOf[r] = static_cast<index_t>(static_cast<uint64_t>(nChunks) * r / nranks);
    const index_t cLo = chunkLoOf[rank], cHi = chunkLoOf[rank + 1];
    auto rowOf = [&](index_t c) { return std::min<uint64_t>(static_cast<uint64_t>(c) * chunkRows, numRows); };
    const index_t rowLo = static_cast<index_t>(rowOf(cLo)), rowHi = static_cast<index_t>(rowOf(cHi));
    const index_t localRows = rowHi - rowLo;

    // ---- Pass 1: count raw hits per chunk (parallel across ranks and threads) ----
    MatrixGen gen(nItems, numRows);
    std::vector<index_t> rawCount(nChunks, 0);
#pragma omp parallel for schedule(dynamic, 1)
    for (index_t c = cLo; c < cHi; ++c)
        rawCount[c] = gen.countRaw(static_cast<index_t>(rowOf(c)), static_cast<index_t>(rowOf(c + 1)));
    {
        std::vector<int> cnt(nranks), dsp(nranks);
        for (int r = 0; r < nranks; ++r) {
            cnt[r] = static_cast<int>(chunkLoOf[r + 1] - chunkLoOf[r]);
            dsp[r] = static_cast<int>(chunkLoOf[r]);
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, rawCount.data(), cnt.data(), dsp.data(),
                       MPI_UNSIGNED, MPI_COMM_WORLD);
    }

    // ---- Resolve exact start state of every chunk (sequential chain, cheap) ----
    // Chunks where the cap (nnz < n) or fillRemaining logic could matter are re-run exactly.
    std::vector<GenState> startState(nChunks + 1);
    {
        GenState st{0, false};
        const bool noWrap = numRows < 65536;  // dim*dim fits in index_t
        for (index_t c = 0; c < nChunks; ++c) {
            startState[c] = st;
            const index_t r0 = static_cast<index_t>(rowOf(c)), r1 = static_cast<index_t>(rowOf(c + 1));
            if (st.fill) {
                st.nnz += (r1 - r0) * numRows;
            } else if (st.nnz >= nItems) {
                // nothing further is assigned
            } else if (noWrap &&
                       numRows * numRows - (r1 * numRows - 1) > nItems - st.nnz &&
                       static_cast<uint64_t>(st.nnz) + rawCount[c] <= nItems) {
                st.nnz += rawCount[c];
            } else {
                st = gen.run(r0, r1, st, nullptr, 0, nullptr, 0);
            }
        }
        startState[nChunks] = st;
    }
    const index_t nnzLo = startState[cLo].nnz;
    const index_t nnzHi = startState[cHi].nnz;
    const index_t localNnz = nnzHi - nnzLo;

    // ---- Pass 2: generate local CSR (columns, row delimiters, values) and the vector ----
    std::vector<double> h_val(localNnz);
    std::vector<index_t> h_cols(localNnz);
    std::vector<index_t> h_rowDelimiters(localRows + 1);
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out(localRows);

    fill(h_vec.data(), numRows, maxVal, 0);
    {
        static const LfgBase valBase(1);
#pragma omp parallel for schedule(dynamic, 1)
        for (index_t c = cLo; c < cHi; ++c) {
            const index_t r0 = static_cast<index_t>(rowOf(c)), r1 = static_cast<index_t>(rowOf(c + 1));
            gen.run(r0, r1, startState[c], h_cols.data(), nnzLo, h_rowDelimiters.data(), rowLo);
            const index_t b = startState[c].nnz - nnzLo, e = startState[c + 1].nnz - nnzLo;
            // values: stream index numRows + global nonzero index
            Lfg g(valBase, static_cast<uint64_t>(numRows) + nnzLo + b);
            for (index_t k = b; k < e; ++k)
                h_val[k] = maxVal * (g.next() / (static_cast<double>(RAND_MAX) + 1.0));
        }
    }
    for (index_t i = 0; i < localRows; ++i) h_rowDelimiters[i] -= nnzLo;
    h_rowDelimiters[localRows] = localNnz;

    // For validation, compute reference solution for local rows
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
            fflush(stdout);
        }
        h_reference.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), localRows,
                h_reference.data());
    }

    // ---- Device data ----
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rowDel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val, sizeof(double) * std::max<index_t>(localNnz, 1)));
    CUDA_CHECK(cudaMalloc(&d_cols, sizeof(index_t) * std::max<index_t>(localNnz, 1)));
    CUDA_CHECK(cudaMalloc(&d_rowDel, sizeof(index_t) * (localRows + 1)));
    CUDA_CHECK(cudaMalloc(&d_vec, sizeof(double) * std::max<index_t>(numRows, 1)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * std::max<index_t>(localRows, 1)));
    CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), sizeof(double) * localNnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), sizeof(index_t) * localNnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDel, h_rowDelimiters.data(), sizeof(index_t) * (localRows + 1),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), sizeof(double) * numRows, cudaMemcpyHostToDevice));

    const double avgRow = localRows ? static_cast<double>(localNnz) / localRows : 0.0;
    const int tpr = avgRow >= 48 ? 32 : avgRow >= 24 ? 16 : avgRow >= 12 ? 8 : avgRow >= 6 ? 4
                    : avgRow >= 3 ? 2 : 1;

    // Warm-up (result is identical every iteration)
    spmvGpu(tpr, d_val, d_cols, d_rowDel, d_vec, localRows, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (rank == 0) {
        printf("Computing SpMV...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvGpu(tpr, d_val, d_cols, d_rowDel, d_vec, localRows, d_out);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    const double local_duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(double) * localRows, cudaMemcpyDeviceToHost));

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration_ms);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration_ms / 1000.0) / 1e9;
        const double avgTime = duration_ms / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> all;
        std::vector<int> cnt(nranks), dsp(nranks);
        if (rank == 0) all.resize(numRows);
        for (int r = 0; r < nranks; ++r) {
            dsp[r] = static_cast<int>(rowOf(chunkLoOf[r]));
            cnt[r] = static_cast<int>(rowOf(chunkLoOf[r + 1])) - dsp[r];
        }
        MPI_Gatherv(h_out.data(), static_cast<int>(localRows), MPI_DOUBLE, all.data(), cnt.data(),
                    dsp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(all, "OutputVector");
    }

    cudaFree(d_val);
    cudaFree(d_cols);
    cudaFree(d_rowDel);
    cudaFree(d_vec);
    cudaFree(d_out);

    int status = 0;
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int ok = verifyResults(h_reference.data(), h_out.data(), localRows, rowLo) ? 1 : 0;
        int allOk = 0;
        MPI_Allreduce(&ok, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf(allOk ? "Validation: PASSED\n" : "Validation: FAILED\n");
        status = allOk ? 0 : 1;
    }

    MPI_Finalize();
    return status;
}
