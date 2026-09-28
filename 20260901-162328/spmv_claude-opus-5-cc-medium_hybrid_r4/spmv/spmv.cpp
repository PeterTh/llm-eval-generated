#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        const cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                         \
        }                                                                                         \
    } while (0)

// ****************************************************************************
// Reproducible random number generation
//
// The reference implementation draws all of its random numbers from the C
// library's rand().  In order to distribute (and parallelize) the data setup
// across MPI ranks, GPUs and OpenMP threads while producing bit-identical
// input data, glibc's TYPE_3 additive-feedback generator is re-implemented
// here together with a jump-ahead facility.
//
// State:   s_j = r[k-31+j],  j = 0..30
// Step:    v = s_0 + s_28 (mod 2^32);  shift;  s_30 = v;  output = (v>>1)&INT_MAX
// The step is a linear map over Z_{2^32}, so an arbitrary number of draws can
// be skipped by raising the corresponding 31x31 matrix to the desired power.
// ****************************************************************************

using u32 = uint32_t;

struct RState {
    u32 s[31];
};

// Fast sequential generator (circular buffer form of the state above)
struct Rng {
    u32 a[31];
    int p;

    explicit Rng(const RState& st) {
        for (int j = 0; j < 31; ++j) {
            a[j] = st.s[j];
        }
        p = 0;
    }

    inline int next() {
        const u32 v = a[p] + a[p >= 3 ? p - 3 : p + 28];
        a[p] = v;
        p = (p + 1 == 31) ? 0 : p + 1;
        return static_cast<int>((v >> 1) & 0x7fffffffu);
    }
};

struct JumpMat {
    u32 m[31][31];
};

static void matIdentity(JumpMat& r) {
    memset(&r, 0, sizeof(JumpMat));
    for (int i = 0; i < 31; ++i) {
        r.m[i][i] = 1;
    }
}

static void matStep(JumpMat& r) {
    memset(&r, 0, sizeof(JumpMat));
    for (int i = 0; i < 30; ++i) {
        r.m[i][i + 1] = 1;
    }
    r.m[30][0] = 1;
    r.m[30][28] += 1;
}

static void matMul(const JumpMat& a, const JumpMat& b, JumpMat& c) {
    JumpMat t;
    memset(&t, 0, sizeof(JumpMat));
    for (int i = 0; i < 31; ++i) {
        for (int k = 0; k < 31; ++k) {
            const u32 aik = a.m[i][k];
            if (aik == 0) {
                continue;
            }
            for (int j = 0; j < 31; ++j) {
                t.m[i][j] += aik * b.m[k][j];
            }
        }
    }
    c = t;
}

static void matApply(const JumpMat& a, const RState& in, RState& out) {
    RState t;
    for (int i = 0; i < 31; ++i) {
        u32 acc = 0;
        for (int j = 0; j < 31; ++j) {
            acc += a.m[i][j] * in.s[j];
        }
        t.s[i] = acc;
    }
    out = t;
}

// Matrix corresponding to skipping `d` draws
static JumpMat jumpMatrix(uint64_t d) {
    JumpMat result;
    matIdentity(result);
    JumpMat base;
    matStep(base);
    while (d > 0) {
        if (d & 1ull) {
            matMul(result, base, result);
        }
        d >>= 1;
        if (d > 0) {
            matMul(base, base, base);
        }
    }
    return result;
}

static void rngSkip(RState& st, uint64_t d) {
    if (d == 0) {
        return;
    }
    const JumpMat j = jumpMatrix(d);
    matApply(j, st, st);
}

// Equivalent to srand(seed) followed by the state glibc uses for the first draw
static RState rngSeed(u32 seed) {
    int32_t r[31];
    r[0] = seed != 0 ? static_cast<int32_t>(seed) : 1;
    for (int i = 1; i < 31; ++i) {
        const int64_t hi = r[i - 1] / 127773;
        const int64_t lo = r[i - 1] % 127773;
        int64_t w = 16807 * lo - 2836 * hi;
        if (w < 0) {
            w += 2147483647;
        }
        r[i] = static_cast<int32_t>(w);
    }
    RState st;
    // The first value is written to slot 3 with slot 0 as addend
    for (int j = 0; j < 31; ++j) {
        st.s[j] = static_cast<u32>(r[(3 + j) % 31]);
    }
    // glibc discards the first 310 outputs
    rngSkip(st, 310);
    return st;
}

// ****************************************************************************
// Function: fillRange
//
// Purpose:
//   Initialize a range of a (conceptually global) array with random values,
//   identical to the reference `fill()` but starting at an arbitrary draw
//   index and parallelized with OpenMP.
//
// Arguments:
//   A: pointer to the first element to initialize
//   base: generator state at the very beginning of the random stream
//   firstDraw: index of the random draw corresponding to A[0]
//   n: number of elements to initialize
//   maxVal: specifies range of random values [0, maxVal]
//
// ****************************************************************************
static void fillRange(double* A, const RState& base, const uint64_t firstDraw, const uint64_t n,
                      const double maxVal) {
    if (n == 0) {
        return;
    }

    int nThreads = omp_get_max_threads();
    if (static_cast<uint64_t>(nThreads) > n / 4096 + 1) {
        nThreads = static_cast<int>(n / 4096 + 1);
    }
    const uint64_t chunk = (n + nThreads - 1) / nThreads;

    // Derive per-thread start states sequentially (cheap: one 31x31 mat-vec each)
    std::vector<RState> starts(nThreads);
    starts[0] = base;
    rngSkip(starts[0], firstDraw);
    if (nThreads > 1) {
        const JumpMat jc = jumpMatrix(chunk);
        for (int t = 1; t < nThreads; ++t) {
            matApply(jc, starts[t - 1], starts[t]);
        }
    }

#pragma omp parallel for num_threads(nThreads) schedule(static, 1)
    for (int t = 0; t < nThreads; ++t) {
        const uint64_t begin = static_cast<uint64_t>(t) * chunk;
        const uint64_t end = begin + chunk < n ? begin + chunk : n;
        Rng rng(starts[t]);
        for (uint64_t i = begin; i < end; ++i) {
            A[i] = maxVal * (rng.next() / (static_cast<double>(RAND_MAX) + 1.0));
        }
    }
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format for a range
//   of rows (used for the reference solution of the locally owned rows).
//
// Arguments:
//   val: array holding the non-zero values for the local rows
//   cols: array of column indices for each local element
//   rowDelimiters: array of size dim+1 holding local indices to rows
//   vec: dense vector to be used for multiplication
//   dim: number of local rows
//   out: output - result from the spmv calculation
//
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
#pragma omp parallel for schedule(dynamic, 256)
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
// CUDA kernels
// ****************************************************************************

// One warp per row: good when there are enough non-zeros per row to keep the
// memory accesses of a warp coalesced.
__global__ void spmvWarpKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                               const index_t* __restrict__ rowDelimiters,
                               const double* __restrict__ vec, const index_t dim,
                               double* __restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const unsigned int warpId = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned int warpCount = (gridDim.x * blockDim.x) >> 5;

    for (index_t row = warpId; row < dim; row += warpCount) {
        const index_t begin = rowDelimiters[row];
        const index_t end = rowDelimiters[row + 1];
        double t = 0.0;
        for (index_t j = begin + lane; j < end; j += 32) {
            t += val[j] * __ldg(&vec[cols[j]]);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            t += __shfl_down_sync(0xffffffffu, t, off);
        }
        if (lane == 0) {
            out[row] = t;
        }
    }
}

// One thread per row: better for very short rows, where a whole warp would be
// mostly idle.
__global__ void spmvScalarKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                                 const index_t* __restrict__ rowDelimiters,
                                 const double* __restrict__ vec, const index_t dim,
                                 double* __restrict__ out) {
    const unsigned int stride = gridDim.x * blockDim.x;
    for (index_t row = blockIdx.x * blockDim.x + threadIdx.x; row < dim; row += stride) {
        const index_t begin = rowDelimiters[row];
        const index_t end = rowDelimiters[row + 1];
        double t = 0.0;
        for (index_t j = begin; j < end; ++j) {
            t += val[j] * __ldg(&vec[cols[j]]);
        }
        out[row] = t;
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
//   globalOffset: index of the first element within the global vector
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
                printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                       globalOffset + i, ref, res);
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

// Split [0, dim) into `parts` contiguous row ranges with a balanced number of
// non-zeros per part.  bounds has `parts + 1` entries.
static void partitionByNnz(const index_t* rowDelimiters, const index_t dim, const int parts,
                           std::vector<index_t>& bounds) {
    bounds.assign(parts + 1, dim);
    bounds[0] = 0;
    // The delimiters may be a slice of a larger array, so work relative to the first entry
    const index_t base = dim > 0 ? rowDelimiters[0] : 0;
    const uint64_t total = dim > 0 ? rowDelimiters[dim] - base : 0;
    index_t row = 0;
    for (int p = 1; p < parts; ++p) {
        const uint64_t target = (total * static_cast<uint64_t>(p)) / static_cast<uint64_t>(parts);
        while (row < dim && rowDelimiters[row] - base < target) {
            ++row;
        }
        // Keep the partition monotone and leave room for the remaining parts
        index_t lo = bounds[p - 1];
        index_t candidate = row < lo ? lo : row;
        const index_t maxStart = dim - static_cast<index_t>(parts - p) < lo
                                     ? lo
                                     : dim - static_cast<index_t>(parts - p);
        if (candidate > maxStart) {
            candidate = maxStart;
        }
        bounds[p] = candidate;
    }
    bounds[parts] = dim;
}

// Per-device state
struct DeviceCtx {
    int device = 0;
    index_t rowBegin = 0;
    index_t rowEnd = 0;
    index_t nnzBegin = 0;
    index_t nnzEnd = 0;
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    cudaStream_t stream = nullptr;
    int blocks = 0;
    int threads = 256;
    bool useWarpKernel = true;
};

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    // ------------------------------------------------------------------
    // Determine which GPUs this rank drives.  Ranks sharing a node split the
    // node's GPUs evenly, so the whole node is used regardless of how many
    // ranks are started per node.
    // ------------------------------------------------------------------
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<int> myDevices;
    if (localSize >= deviceCount) {
        myDevices.push_back(localRank % deviceCount);
    } else {
        // Fewer ranks than GPUs: this rank drives several GPUs
        const int begin = (deviceCount * localRank) / localSize;
        const int end = (deviceCount * (localRank + 1)) / localSize;
        for (int d = begin; d < end; ++d) {
            myDevices.push_back(d);
        }
    }
    const int numDevices = static_cast<int>(myDevices.size());

    // Number of distinct GPUs used across all nodes
    // Every GPU of a node is used, no matter how the ranks are distributed
    int totalDevices = localRank == 0 ? deviceCount : 0;
    MPI_Allreduce(MPI_IN_PLACE, &totalDevices, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPUs: %d\n", numRanks,
               omp_get_max_threads(), totalDevices);
    }

    if (numRows == 0) {
        if (rank == 0) {
            fprintf(stderr, "Empty problem\n");
        }
        MPI_Finalize();
        return 1;
    }

    // ------------------------------------------------------------------
    // Distributed generation of the sparse matrix structure.
    //
    // Every rank computes the number of non-zeros of a slice of the rows,
    // the counts are combined, and the exact row delimiters are derived from
    // them.  Afterwards every rank materializes only the rows it owns.
    // ------------------------------------------------------------------
    if (rank == 0) {
        printf("Initializing data structures...\n");
    }

    const double prob = static_cast<double>(nItems) /
                        (static_cast<double>(numRows) * static_cast<double>(numRows));

    // Generator state at the beginning of every row of the generation loop
    const RState matBase = rngSeed(8675309);
    std::vector<RState> rowState(numRows);
    {
        const JumpMat jRow = jumpMatrix(numRows);
        rowState[0] = matBase;
        for (index_t i = 1; i < numRows; ++i) {
            matApply(jRow, rowState[i - 1], rowState[i]);
        }
    }

    std::vector<index_t> rowCounts(numRows, 0);
    {
        const index_t share = (numRows + numRanks - 1) / static_cast<index_t>(numRanks);
        const index_t begin = share * static_cast<index_t>(rank) < numRows
                                  ? share * static_cast<index_t>(rank)
                                  : numRows;
        const index_t end = begin + share < numRows ? begin + share : numRows;
#pragma omp parallel for schedule(static)
        for (index_t i = begin; i < end; ++i) {
            Rng rng(rowState[i]);
            index_t count = 0;
            for (index_t j = 0; j < numRows; ++j) {
                const double randVal = static_cast<double>(rng.next()) / RAND_MAX;
                if (randVal <= prob) {
                    ++count;
                }
            }
            rowCounts[i] = count;
        }
        MPI_Allreduce(MPI_IN_PLACE, rowCounts.data(), static_cast<int>(numRows), MPI_UINT32_T,
                      MPI_SUM, MPI_COMM_WORLD);
    }

    // Exact row delimiters.  As long as neither the "all remaining entries must
    // be filled" rule nor the cap at nItems can trigger within a row, the row's
    // size is simply its raw hit count.  The remaining tail (a handful of rows
    // at the very end) is replayed exactly as in the reference implementation.
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<index_t> tailCols;
    index_t tailRow = numRows;
    index_t tailBase = 0;
    {
        const bool noOverflow =
            static_cast<uint64_t>(numRows) * static_cast<uint64_t>(numRows) <= 0xffffffffull;
        index_t cum = 0;
        index_t i = 0;
        for (; i < numRows; ++i) {
            h_rowDelimiters[i] = cum;
            const index_t entriesLeftAtRowEnd =
                (numRows * numRows) - ((i * numRows) + (numRows - 1));
            const bool safe = noOverflow &&
                              (static_cast<uint64_t>(cum) + rowCounts[i] <= nItems) &&
                              (entriesLeftAtRowEnd > nItems - cum);
            if (!safe) {
                break;
            }
            cum += rowCounts[i];
        }
        tailRow = i;
        tailBase = cum;

        // Replay of the original assignment loop for the tail rows
        index_t nnzAssigned = cum;
        bool fillRemaining = false;
        for (; i < numRows; ++i) {
            h_rowDelimiters[i] = nnzAssigned;
            if (nnzAssigned >= nItems && !fillRemaining) {
                continue;
            }
            Rng rng(rowState[i]);
            for (index_t j = 0; j < numRows; ++j) {
                const index_t numEntriesLeft = (numRows * numRows) - ((i * numRows) + j);
                const index_t needToAssign = nItems - nnzAssigned;
                if (numEntriesLeft <= needToAssign) {
                    fillRemaining = true;
                }
                const double randVal = static_cast<double>(rng.next()) / RAND_MAX;
                if ((nnzAssigned < nItems && randVal <= prob) || fillRemaining) {
                    if (nnzAssigned >= nItems) {
                        break;  // matrix parameters overflow index_t; stop safely
                    }
                    tailCols.push_back(j);
                    nnzAssigned++;
                }
                if (nnzAssigned >= nItems && !fillRemaining) {
                    break;
                }
            }
        }
        // Convention: put the number of non-zeroes at the end of the row delimiters array
        h_rowDelimiters[numRows] = nItems;
    }

    // Row ranges per rank (balanced by non-zeros) and per device within a rank
    std::vector<index_t> rankBounds;
    partitionByNnz(h_rowDelimiters.data(), numRows, numRanks, rankBounds);
    const index_t myRowBegin = rankBounds[rank];
    const index_t myRowEnd = rankBounds[rank + 1];
    const index_t myRows = myRowEnd - myRowBegin;
    const index_t myNnzBegin = h_rowDelimiters[myRowBegin];
    const index_t myNnzEnd = h_rowDelimiters[myRowEnd];
    const index_t myNnz = myNnzEnd - myNnzBegin;

    // Local CSR data
    std::vector<double> h_val(myNnz);
    std::vector<index_t> h_cols(myNnz);
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out(rank == 0 ? numRows : myRows);
    double* myOut = h_out.data() + (rank == 0 ? myRowBegin : 0);

    // Dense vector and matrix values come from the default (seed 1) stream:
    // first numRows draws for the vector, then nItems draws for the values.
    const RState fillBase = rngSeed(1);
    fillRange(h_vec.data(), fillBase, 0, numRows, maxVal);
    fillRange(h_val.data(), fillBase, static_cast<uint64_t>(numRows) + myNnzBegin, myNnz, maxVal);

    // Column indices of the locally owned rows
#pragma omp parallel for schedule(dynamic, 16)
    for (index_t i = myRowBegin; i < myRowEnd; ++i) {
        index_t* dst = h_cols.data() + (h_rowDelimiters[i] - myNnzBegin);
        const index_t count = h_rowDelimiters[i + 1] - h_rowDelimiters[i];
        if (i < tailRow) {
            Rng rng(rowState[i]);
            index_t k = 0;
            for (index_t j = 0; j < numRows; ++j) {
                const double randVal = static_cast<double>(rng.next()) / RAND_MAX;
                if (randVal <= prob) {
                    dst[k++] = j;
                }
            }
        } else {
            memcpy(dst, tailCols.data() + (h_rowDelimiters[i] - tailBase),
                   count * sizeof(index_t));
        }
    }
    rowState.clear();
    rowState.shrink_to_fit();
    tailCols.clear();
    tailCols.shrink_to_fit();

    // ------------------------------------------------------------------
    // Device setup
    // ------------------------------------------------------------------
    std::vector<index_t> devBounds;
    partitionByNnz(h_rowDelimiters.data() + myRowBegin, myRows, numDevices, devBounds);

    std::vector<DeviceCtx> ctx(numDevices);
    const double nnzPerRow = myRows > 0 ? static_cast<double>(myNnz) / myRows : 0.0;

    for (int d = 0; d < numDevices; ++d) {
        DeviceCtx& c = ctx[d];
        c.device = myDevices[d];
        c.rowBegin = myRowBegin + devBounds[d];
        c.rowEnd = myRowBegin + devBounds[d + 1];
        c.nnzBegin = h_rowDelimiters[c.rowBegin];
        c.nnzEnd = h_rowDelimiters[c.rowEnd];
        const index_t rows = c.rowEnd - c.rowBegin;
        const index_t nnz = c.nnzEnd - c.nnzBegin;

        CUDA_CHECK(cudaSetDevice(c.device));
        CUDA_CHECK(cudaStreamCreate(&c.stream));
        CUDA_CHECK(cudaMalloc(&c.d_val, (nnz ? nnz : 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.d_cols, (nnz ? nnz : 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&c.d_rowDelimiters, (rows + 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&c.d_vec, numRows * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.d_out, (rows ? rows : 1) * sizeof(double)));

        // Row delimiters rebased to the device-local non-zero indexing
        std::vector<index_t> localDelims(rows + 1);
        for (index_t i = 0; i <= rows; ++i) {
            localDelims[i] = h_rowDelimiters[c.rowBegin + i] - c.nnzBegin;
        }
        CUDA_CHECK(cudaMemcpyAsync(c.d_val, h_val.data() + (c.nnzBegin - myNnzBegin),
                                   nnz * sizeof(double), cudaMemcpyHostToDevice, c.stream));
        CUDA_CHECK(cudaMemcpyAsync(c.d_cols, h_cols.data() + (c.nnzBegin - myNnzBegin),
                                   nnz * sizeof(index_t), cudaMemcpyHostToDevice, c.stream));
        CUDA_CHECK(cudaMemcpy(c.d_rowDelimiters, localDelims.data(), (rows + 1) * sizeof(index_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(c.d_vec, h_vec.data(), numRows * sizeof(double),
                                   cudaMemcpyHostToDevice, c.stream));
        CUDA_CHECK(cudaStreamSynchronize(c.stream));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, c.device));
        c.useWarpKernel = nnzPerRow >= 6.0;
        c.threads = 256;
        const int perBlockRows = c.useWarpKernel ? c.threads / 32 : c.threads;
        int blocks = static_cast<int>((rows + perBlockRows - 1) / perBlockRows);
        const int maxBlocks = prop.multiProcessorCount * 32;
        if (blocks > maxBlocks) {
            blocks = maxBlocks;
        }
        c.blocks = blocks > 0 ? blocks : 1;
    }

    // Reference solution for the locally owned rows
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        h_reference.resize(myRows);
        std::vector<index_t> localDelims(myRows + 1);
        for (index_t i = 0; i <= myRows; ++i) {
            localDelims[i] = h_rowDelimiters[myRowBegin + i] - myNnzBegin;
        }
        spmvCpu(h_val.data(), h_cols.data(), localDelims.data(), h_vec.data(), myRows,
                h_reference.data());
    }

    // Gather setup for the output vector
    std::vector<int> recvCounts, displs;
    if (rank == 0) {
        recvCounts.resize(numRanks);
        displs.resize(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            recvCounts[r] = static_cast<int>(rankBounds[r + 1] - rankBounds[r]);
            displs[r] = static_cast<int>(rankBounds[r]);
        }
    }

    // ------------------------------------------------------------------
    // SpMV
    // ------------------------------------------------------------------
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    // Warm-up (kernel load / clock ramp), not part of the measurement
    for (int d = 0; iterations > 0 && d < numDevices; ++d) {
        DeviceCtx& c = ctx[d];
        const index_t rows = c.rowEnd - c.rowBegin;
        CUDA_CHECK(cudaSetDevice(c.device));
        if (rows > 0) {
            if (c.useWarpKernel) {
                spmvWarpKernel<<<c.blocks, c.threads, 0, c.stream>>>(
                    c.d_val, c.d_cols, c.d_rowDelimiters, c.d_vec, rows, c.d_out);
            } else {
                spmvScalarKernel<<<c.blocks, c.threads, 0, c.stream>>>(
                    c.d_val, c.d_cols, c.d_rowDelimiters, c.d_vec, rows, c.d_out);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(c.stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        for (int d = 0; d < numDevices; ++d) {
            DeviceCtx& c = ctx[d];
            const index_t rows = c.rowEnd - c.rowBegin;
            if (rows == 0) {
                continue;
            }
            CUDA_CHECK(cudaSetDevice(c.device));
            if (c.useWarpKernel) {
                spmvWarpKernel<<<c.blocks, c.threads, 0, c.stream>>>(
                    c.d_val, c.d_cols, c.d_rowDelimiters, c.d_vec, rows, c.d_out);
            } else {
                spmvScalarKernel<<<c.blocks, c.threads, 0, c.stream>>>(
                    c.d_val, c.d_cols, c.d_rowDelimiters, c.d_vec, rows, c.d_out);
            }
        }
    }

    // Copy the final result back and assemble it on rank 0
    for (int d = 0; iterations > 0 && d < numDevices; ++d) {
        DeviceCtx& c = ctx[d];
        const index_t rows = c.rowEnd - c.rowBegin;
        if (rows == 0) {
            continue;
        }
        CUDA_CHECK(cudaSetDevice(c.device));
        CUDA_CHECK(cudaMemcpyAsync(myOut + (c.rowBegin - myRowBegin), c.d_out,
                                   rows * sizeof(double), cudaMemcpyDeviceToHost, c.stream));
    }
    for (int d = 0; d < numDevices; ++d) {
        CUDA_CHECK(cudaSetDevice(ctx[d].device));
        CUDA_CHECK(cudaStreamSynchronize(ctx[d].stream));
    }

    if (rank == 0) {
        MPI_Gatherv(MPI_IN_PLACE, static_cast<int>(myRows), MPI_DOUBLE, h_out.data(),
                    recvCounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(h_out.data(), static_cast<int>(myRows), MPI_DOUBLE, nullptr, nullptr, nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (elapsedMs / 1000.0) / 1e9;
        const double avgTime = elapsedMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }
    }

    // Validation
    int valid = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        valid = verifyResults(h_reference.data(), myOut, myRows, myRowBegin) ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    for (int d = 0; d < numDevices; ++d) {
        DeviceCtx& c = ctx[d];
        CUDA_CHECK(cudaSetDevice(c.device));
        CUDA_CHECK(cudaFree(c.d_val));
        CUDA_CHECK(cudaFree(c.d_cols));
        CUDA_CHECK(cudaFree(c.d_rowDelimiters));
        CUDA_CHECK(cudaFree(c.d_vec));
        CUDA_CHECK(cudaFree(c.d_out));
        CUDA_CHECK(cudaStreamDestroy(c.stream));
    }
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();

    return (validate && !valid) ? 1 : 0;
}
