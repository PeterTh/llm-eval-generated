#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <climits>
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

static void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", world_rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cuda_check((call), #call)

// MPI's counts are int even though CSR offsets can cover the full uint32_t range.
template <typename T>
void send_chunks(const T* data, uint64_t count, MPI_Datatype type, int peer, int tag) {
    while (count) {
        int chunk = static_cast<int>(std::min<uint64_t>(count, INT_MAX));
        MPI_Send(data, chunk, type, peer, tag, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}
template <typename T>
void recv_chunks(T* data, uint64_t count, MPI_Datatype type, int peer, int tag) {
    while (count) {
        int chunk = static_cast<int>(std::min<uint64_t>(count, INT_MAX));
        MPI_Recv(data, chunk, type, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}
template <typename T>
void bcast_chunks(T* data, uint64_t count, MPI_Datatype type) {
    while (count) {
        int chunk = static_cast<int>(std::min<uint64_t>(count, INT_MAX));
        MPI_Bcast(data, chunk, type, 0, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

void fill(double* a, index_t n, double maxVal) {
    for (index_t i = 0; i < n; ++i)
        a[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

void initRandomMatrix(index_t* cols, index_t* rows, index_t n, index_t dim) {
    index_t assigned = 0;
    const uint64_t cells = uint64_t(dim) * dim;
    const double prob = static_cast<double>(n) / static_cast<double>(cells);
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rows[i] = assigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t left = cells - (uint64_t(i) * dim + j);
            if (left <= uint64_t(n - assigned)) fillRemaining = true;
            const double random = static_cast<double>(rand()) / RAND_MAX;
            if ((assigned < n && random <= prob) || fillRemaining) cols[assigned++] = j;
        }
    }
    rows[dim] = n;
}

// A thread per row for short rows; a warp per row for longer rows. The latter
// coalesces matrix loads and shares the row's dot product across the warp.
__global__ void spmv_scalar(const double* values, const index_t* columns,
                            const index_t* rows, const double* x, index_t nrows, double* y) {
    for (uint64_t row = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         row < nrows; row += uint64_t(blockDim.x) * gridDim.x) {
        double sum = 0.0;
        for (index_t j = rows[row]; j < rows[row + 1]; ++j)
            sum += values[j] * x[columns[j]];
        y[row] = sum;
    }
}
__global__ void spmv_warp(const double* values, const index_t* columns,
                          const index_t* rows, const double* x, index_t nrows, double* y) {
    const unsigned lane = threadIdx.x & 31;
    const uint64_t first = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const uint64_t stride = uint64_t(blockDim.x) * gridDim.x / 32;
    for (uint64_t row = first; row < nrows; row += stride) {
        double sum = 0.0;
        const index_t end = rows[row + 1];
        for (uint64_t j = uint64_t(rows[row]) + lane; j < end; j += 32)
            sum += values[j] * x[columns[j]];
        for (int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) y[row] = sum;
    }
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

template <typename T> T* device_alloc(size_t count) {
    T* ptr = nullptr;
    if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    return ptr;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (!world_rank) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numRows = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) sparsity = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) maxVal = atof(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!world_rank) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!world_rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    const uint64_t cells = uint64_t(numRows) * numRows;
    if (!numRows || !sparsity || cells / sparsity > std::numeric_limits<index_t>::max()) {
        if (!world_rank) fprintf(stderr, "Matrix dimension and sparsity must be nonzero, and the nonzero count must fit in 32 bits.\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(cells / sparsity);
    if (!world_rank) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, 100.0 * (1.0 - double(nItems) / cells));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &localComm);
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    std::vector<double> x(numRows), allValues;
    std::vector<index_t> allColumns, allRows;
    std::vector<index_t> boundaries(static_cast<size_t>(ranks) + 1);
    if (!world_rank) {
        allValues.resize(nItems);
        allColumns.resize(nItems);
        allRows.resize(static_cast<size_t>(numRows) + 1);
        // The original starts from rand's default seed; MPI/CUDA setup may
        // have used the process-wide generator before we reach this point.
        srand(1);
        fill(x.data(), numRows, maxVal);
        fill(allValues.data(), nItems, maxVal);
        initRandomMatrix(allColumns.data(), allRows.data(), nItems, numRows);
        // Account for both per-row work and nonzeros when choosing boundaries.
        const uint64_t totalWork = uint64_t(numRows) * 32 + nItems;
        for (int p = 1; p < ranks; ++p) {
            const uint64_t target = (totalWork / ranks) * p + (totalWork % ranks) * p / ranks;
            index_t lo = boundaries[p - 1], hi = numRows;
            while (lo < hi) {
                const index_t mid = lo + (hi - lo) / 2;
                if (uint64_t(mid) * 32 + allRows[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            boundaries[p] = lo;
        }
        boundaries[ranks] = numRows;
    }
    MPI_Bcast(boundaries.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    bcast_chunks(x.data(), numRows, MPI_DOUBLE);
    const index_t firstRow = boundaries[world_rank];
    const index_t localRows = boundaries[world_rank + 1] - firstRow;
    std::vector<index_t> rows(static_cast<size_t>(localRows) + 1), columns;
    std::vector<double> values;
    if (!world_rank) {
        for (int p = 1; p < ranks; ++p) {
            index_t begin = boundaries[p], end = boundaries[p + 1];
            index_t first = allRows[begin], count = allRows[end] - first;
            send_chunks(allRows.data() + begin, uint64_t(end) - begin + 1, MPI_UINT32_T, p, 1);
            if (count) {
                send_chunks(allColumns.data() + first, count, MPI_UINT32_T, p, 2);
                send_chunks(allValues.data() + first, count, MPI_DOUBLE, p, 3);
            }
        }
        std::copy_n(allRows.data(), static_cast<size_t>(localRows) + 1, rows.data());
        index_t count = rows.back();
        columns.assign(allColumns.begin(), allColumns.begin() + count);
        values.assign(allValues.begin(), allValues.begin() + count);
        std::vector<index_t>().swap(allRows);
        std::vector<index_t>().swap(allColumns);
        std::vector<double>().swap(allValues);
    } else {
        recv_chunks(rows.data(), uint64_t(localRows) + 1, MPI_UINT32_T, 0, 1);
        const index_t count = rows.back() - rows.front();
        columns.resize(count);
        values.resize(count);
        if (count) {
            recv_chunks(columns.data(), count, MPI_UINT32_T, 0, 2);
            recv_chunks(values.data(), count, MPI_DOUBLE, 0, 3);
        }
    }
    const index_t offset = rows.front();
    const int hostThreads = static_cast<int>(std::min<uint64_t>(
        omp_get_max_threads(), std::max<uint64_t>(1, (uint64_t(localRows) + 255) / 256)));
    #pragma omp parallel for schedule(static) num_threads(hostThreads)
    for (uint64_t i = 0; i <= localRows; ++i) rows[i] -= offset;

    std::vector<double> reference;
    if (validate) {
        if (!world_rank) printf("Computing reference solution...\n");
        reference.resize(localRows);
        #pragma omp parallel for schedule(dynamic, 64) num_threads(hostThreads)
        for (uint64_t i = 0; i < localRows; ++i) {
            double sum = 0.0;
            for (index_t j = rows[i]; j < rows[i + 1]; ++j) sum += values[j] * x[columns[j]];
            reference[i] = sum;
        }
    }

    double *dValues = device_alloc<double>(values.size()), *dX = device_alloc<double>(x.size()),
           *dY = device_alloc<double>(localRows);
    index_t *dColumns = device_alloc<index_t>(columns.size()), *dRows = device_alloc<index_t>(rows.size());
    if (!values.empty()) {
        CUDA_CHECK(cudaMemcpy(dValues, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dColumns, columns.data(), columns.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(dRows, rows.data(), rows.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dX, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice));
    if (localRows) CUDA_CHECK(cudaMemset(dY, 0, size_t(localRows) * sizeof(double)));
    constexpr int block = 256;
    const bool useWarp = localRows && values.size() / localRows >= 8;
    const uint64_t threadsNeeded = uint64_t(localRows) * (useWarp ? 32 : 1);
    const int blocks = static_cast<int>(std::min<uint64_t>((threadsNeeded + block - 1) / block, 65535));
    // Initialize the CUDA kernel before timing so module loading and JIT work
    // are not charged to the first multiplication.
    if (blocks && iterations) {
        if (useWarp) spmv_warp<<<blocks, block>>>(dValues, dColumns, dRows, dX, localRows, dY);
        else spmv_scalar<<<blocks, block>>>(dValues, dColumns, dRows, dX, localRows, dY);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!world_rank) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (blocks) {
        for (index_t i = 0; i < iterations; ++i) {
            if (useWarp) spmv_warp<<<blocks, block>>>(dValues, dColumns, dRows, dX, localRows, dY);
            else spmv_scalar<<<blocks, block>>>(dValues, dColumns, dRows, dX, localRows, dY);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<double> out(localRows);
    if (localRows && (validate || printResults))
        CUDA_CHECK(cudaMemcpy(out.data(), dY, size_t(localRows) * sizeof(double), cudaMemcpyDeviceToHost));
    if (!world_rank) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        printf("Average time per iteration: %.3f ms\n", iterations ? maxElapsed * 1000 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", maxElapsed > 0 ? 2.0 * nItems * iterations / maxElapsed / 1e9 : 0.0);
    }
    if (printResults) {
        if (!world_rank) {
            std::vector<double> allOut(numRows);
            std::copy(out.begin(), out.end(), allOut.begin());
            for (int p = 1; p < ranks; ++p) {
                index_t count = boundaries[p + 1] - boundaries[p];
                if (count) recv_chunks(allOut.data() + boundaries[p], count, MPI_DOUBLE, p, 4);
            }
            print_results(allOut, "OutputVector");
        } else if (localRows) send_chunks(out.data(), localRows, MPI_DOUBLE, 0, 4);
    }
    int valid = 1;
    if (validate) {
        if (!world_rank) printf("Validating result...\n");
        for (index_t i = 0; i < localRows; ++i) {
            double ref = reference[i], res = out[i];
            if ((std::abs(ref) < 1e-10 && std::abs(res) > MAX_RELATIVE_ERROR) ||
                (std::abs(ref) >= 1e-10 && std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR)) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", firstRow + i, ref, res);
                valid = 0;
                break;
            }
        }
        int allValid;
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        valid = allValid;
        if (!world_rank) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    CUDA_CHECK(cudaFree(dValues)); CUDA_CHECK(cudaFree(dColumns)); CUDA_CHECK(cudaFree(dRows));
    CUDA_CHECK(cudaFree(dX)); CUDA_CHECK(cudaFree(dY));
    MPI_Finalize();
    return valid ? 0 : 1;
}
