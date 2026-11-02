#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
constexpr double MAX_RELATIVE_ERROR = 0.02;

static int world_rank = 0;

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", world_rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

// Keep the original random number sequence, including the matrix's separate seed.
static void fill(double* a, index_t n, double maxVal) {
    for (index_t i = 0; i < n; ++i)
        a[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

static void initRandomMatrix(index_t* cols, index_t* rows, index_t n, index_t dim) {
    index_t assigned = 0;
    const uint64_t total = static_cast<uint64_t>(dim) * dim;
    const double probability = total ? static_cast<double>(n) / static_cast<double>(total) : 0.0;
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rows[i] = assigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t remaining = total - (static_cast<uint64_t>(i) * dim + j);
            if (remaining <= static_cast<uint64_t>(n - assigned)) fillRemaining = true;
            const double random = static_cast<double>(rand()) / RAND_MAX;
            if ((assigned < n && random <= probability) || fillRemaining) cols[assigned++] = j;
        }
    }
    rows[dim] = n;
}

// Chunk transfers because MPI's count argument is an int while CSR offsets are uint32_t.
template <typename T>
static void sendChunks(const T* data, uint64_t count, MPI_Datatype type, int dest, int tag) {
    while (count) {
        const int chunk = static_cast<int>(std::min<uint64_t>(count, INT32_MAX));
        MPI_Send(data, chunk, type, dest, tag, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}
template <typename T>
static void recvChunks(T* data, uint64_t count, MPI_Datatype type, int source, int tag) {
    while (count) {
        const int chunk = static_cast<int>(std::min<uint64_t>(count, INT32_MAX));
        MPI_Recv(data, chunk, type, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}
template <typename T>
static void broadcastChunks(T* data, uint64_t count, MPI_Datatype type) {
    while (count) {
        const int chunk = static_cast<int>(std::min<uint64_t>(count, INT32_MAX));
        MPI_Bcast(data, chunk, type, 0, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

// A thread per row avoids wasting lanes for very short rows.
__global__ void spmvShort(const double* values, const index_t* cols, const index_t* rows,
                          const double* vec, index_t nrows, double* out) {
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < nrows) {
        double sum = 0.0;
        for (index_t j = rows[row]; j < rows[row + 1]; ++j) sum += values[j] * vec[cols[j]];
        out[row] = sum;
    }
}

// One warp cooperates on each longer row, making matrix loads coalesced.
__global__ void spmvWarp(const double* values, const index_t* cols, const index_t* rows,
                         const double* vec, index_t nrows, double* out) {
    const uint64_t thread = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t row = thread / 32;
    const unsigned lane = thread % 32;
    if (row < nrows) {
        double sum = 0.0;
        const uint64_t end = rows[row + 1];
        for (uint64_t j = static_cast<uint64_t>(rows[row]) + lane; j < end; j += 32)
            sum += values[j] * vec[cols[j]];
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) out[row] = sum;
    }
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool parseIndex(const char* s, index_t& value) {
    if (!*s || *s == '-') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long x = strtoull(s, &end, 10);
    if (errno || *end || x > std::numeric_limits<index_t>::max()) return false;
    value = static_cast<index_t>(x);
    return true;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int worldSize = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) {
        if (!world_rank) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false, badArgs = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "-s") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            index_t value = 0;
            if (!parseIndex(argv[++i], value)) badArgs = true;
            if (option == 'n') numRows = value;
            else if (option == 's') sparsity = value;
            else iterations = value;
        } else if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* arg = argv[++i];
            maxVal = strtod(arg, &end);
            if (errno || end == arg || *end || !std::isfinite(maxVal)) badArgs = true;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else badArgs = true;
    }
    const uint64_t total = static_cast<uint64_t>(numRows) * numRows;
    if (!sparsity || (sparsity && total / sparsity > std::numeric_limits<index_t>::max())) badArgs = true;
    if (help || badArgs) {
        if (!world_rank) {
            if (badArgs) fprintf(stderr, "Invalid arguments or matrix exceeds 32-bit CSR capacity\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return badArgs ? 1 : 0;
    }
    const index_t nItems = static_cast<index_t>(total / sparsity);
    if (!world_rank) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               total ? 100.0 * (1.0 - static_cast<double>(nItems) / total) : 100.0);
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    std::vector<double> vec(numRows), allValues;
    std::vector<index_t> allCols, allRows, bounds(static_cast<size_t>(worldSize) + 1), offsets(static_cast<size_t>(worldSize) + 1);
    if (!world_rank) {
        allValues.resize(nItems); allCols.resize(nItems); allRows.resize(static_cast<size_t>(numRows) + 1);
        fill(vec.data(), numRows, maxVal);
        fill(allValues.data(), nItems, maxVal);
        initRandomMatrix(allCols.data(), allRows.data(), nItems, numRows);
        // Balance both row overhead and nonzero work, including empty rows.
        const uint64_t work = static_cast<uint64_t>(numRows) + nItems;
        for (int p = 1; p < worldSize; ++p) {
            const uint64_t target = (work / worldSize) * p + (work % worldSize) * p / worldSize;
            index_t lo = bounds[p - 1], hi = numRows;
            while (lo < hi) {
                const index_t mid = lo + (hi - lo) / 2;
                if (static_cast<uint64_t>(mid) + allRows[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            bounds[p] = lo;
        }
        bounds[worldSize] = numRows;
        for (int p = 0; p <= worldSize; ++p) offsets[p] = allRows[bounds[p]];
    }
    MPI_Bcast(bounds.data(), worldSize + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(offsets.data(), worldSize + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    broadcastChunks(vec.data(), numRows, MPI_DOUBLE);
    const index_t localRows = bounds[world_rank + 1] - bounds[world_rank];
    const index_t localNnz = offsets[world_rank + 1] - offsets[world_rank];
    std::vector<index_t> rows(static_cast<size_t>(localRows) + 1), cols(localNnz);
    std::vector<double> values(localNnz), out(localRows, 0.0);
    if (!world_rank) {
        for (int p = 1; p < worldSize; ++p) {
            const uint64_t countRows = static_cast<uint64_t>(bounds[p + 1]) - bounds[p] + 1;
            const uint64_t countNnz = static_cast<uint64_t>(offsets[p + 1]) - offsets[p];
            sendChunks(allRows.data() + bounds[p], countRows, MPI_UINT32_T, p, 1);
            if (countNnz) {
                sendChunks(allCols.data() + offsets[p], countNnz, MPI_UINT32_T, p, 2);
                sendChunks(allValues.data() + offsets[p], countNnz, MPI_DOUBLE, p, 3);
            }
        }
        std::copy_n(allRows.data(), static_cast<size_t>(localRows) + 1, rows.data());
        if (localNnz) {
            std::copy_n(allCols.data(), localNnz, cols.data());
            std::copy_n(allValues.data(), localNnz, values.data());
        }
        std::vector<index_t>().swap(allRows); std::vector<index_t>().swap(allCols);
        std::vector<double>().swap(allValues);
    } else {
        recvChunks(rows.data(), static_cast<uint64_t>(localRows) + 1, MPI_UINT32_T, 0, 1);
        recvChunks(cols.data(), localNnz, MPI_UINT32_T, 0, 2);
        recvChunks(values.data(), localNnz, MPI_DOUBLE, 0, 3);
    }
    const index_t base = offsets[world_rank];
    #pragma omp parallel for if(localRows > 10000) schedule(static)
    for (uint64_t i = 0; i <= localRows; ++i) rows[i] -= base;

    std::vector<double> reference;
    if (validate) {
        if (!world_rank) printf("Computing reference solution...\n");
        reference.resize(localRows);
        #pragma omp parallel for if(localRows > 1000) schedule(static)
        for (uint64_t row = 0; row < localRows; ++row) {
            double sum = 0.0;
            for (index_t j = rows[row]; j < rows[row + 1]; ++j) sum += values[j] * vec[cols[j]];
            reference[row] = sum;
        }
    }

    double *dValues = nullptr, *dVec = nullptr, *dOut = nullptr;
    index_t *dCols = nullptr, *dRows = nullptr;
    if (localRows) {
        CUDA_CHECK(cudaMalloc(&dRows, (static_cast<size_t>(localRows) + 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&dOut, static_cast<size_t>(localRows) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dVec, static_cast<size_t>(numRows) * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dRows, rows.data(), (static_cast<size_t>(localRows) + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dVec, vec.data(), static_cast<size_t>(numRows) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dOut, 0, static_cast<size_t>(localRows) * sizeof(double)));
        if (localNnz) {
            CUDA_CHECK(cudaMalloc(&dValues, static_cast<size_t>(localNnz) * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&dCols, static_cast<size_t>(localNnz) * sizeof(index_t)));
            CUDA_CHECK(cudaMemcpy(dValues, values.data(), static_cast<size_t>(localNnz) * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dCols, cols.data(), static_cast<size_t>(localNnz) * sizeof(index_t), cudaMemcpyHostToDevice));
        }
    }
    if (!world_rank) printf("Computing SpMV...\n");
    constexpr unsigned block = 256;
    const bool shortRows = static_cast<uint64_t>(localNnz) <= static_cast<uint64_t>(localRows) * 4;
    const unsigned grid = shortRows ? static_cast<unsigned>((static_cast<uint64_t>(localRows) + block - 1) / block)
                                    : static_cast<unsigned>((static_cast<uint64_t>(localRows) + block / 32 - 1) / (block / 32));
    auto launch = [&]() {
        if (shortRows) spmvShort<<<grid, block>>>(dValues, dCols, dRows, dVec, localRows, dOut);
        else spmvWarp<<<grid, block>>>(dValues, dCols, dRows, dVec, localRows, dOut);
    };
    // Exclude CUDA module loading and first-launch setup from the measured iterations.
    if (localRows && iterations) {
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (localRows) {
        for (index_t i = 0; i < iterations; ++i) launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (localRows && (validate || printResults))
        CUDA_CHECK(cudaMemcpy(out.data(), dOut, static_cast<size_t>(localRows) * sizeof(double), cudaMemcpyDeviceToHost));
    if (!world_rank) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        printf("Average time per iteration: %.3f ms\n", iterations ? maxElapsed * 1000.0 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", maxElapsed > 0 ? (2.0 * nItems * iterations) / maxElapsed / 1e9 : 0.0);
    }
    if (printResults) {
        if (!world_rank) {
            std::vector<double> result(numRows);
            std::copy(out.begin(), out.end(), result.begin());
            for (int p = 1; p < worldSize; ++p) {
                const index_t count = bounds[p + 1] - bounds[p];
                if (count) recvChunks(result.data() + bounds[p], count, MPI_DOUBLE, p, 4);
            }
            print_results(result, "OutputVector");
        } else sendChunks(out.data(), localRows, MPI_DOUBLE, 0, 4);
    }
    int localValid = 1;
    if (validate) {
        if (!world_rank) printf("Validating result...\n");
        for (index_t i = 0; i < localRows; ++i) {
            const double ref = reference[i], res = out[i];
            if ((std::abs(ref) < 1e-10 && std::abs(res) > MAX_RELATIVE_ERROR) ||
                (std::abs(ref) >= 1e-10 && std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR)) {
                fprintf(stderr, "Validation failed at index %u: reference %.10e, got %.10e\n", bounds[world_rank] + i, ref, res);
                localValid = 0;
                break;
            }
        }
    }
    int valid = 0;
    MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!world_rank && validate) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    if (dValues) CUDA_CHECK(cudaFree(dValues));
    if (dCols) CUDA_CHECK(cudaFree(dCols));
    if (dRows) CUDA_CHECK(cudaFree(dRows));
    if (dVec) CUDA_CHECK(cudaFree(dVec));
    if (dOut) CUDA_CHECK(cudaFree(dOut));
    MPI_Finalize();
    return valid ? 0 : 1;
}
