#include <mpi.h>
#include <cuda_runtime.h>

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

static void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cuda_check((call), #call)

// Preserve the original random number sequence and matrix construction.
static void fill(double* a, index_t n, double maxVal) {
    for (index_t i = 0; i < n; ++i)
        a[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

static void initRandomMatrix(index_t* cols, index_t* rows, index_t n, index_t dim) {
    index_t assigned = 0;
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * dim);
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rows[i] = assigned;
        for (index_t j = 0; j < dim; ++j) {
            const index_t left = dim * dim - (i * dim + j);
            const index_t needed = n - assigned;
            if (left <= needed) fillRemaining = true;
            const double random = static_cast<double>(rand()) / RAND_MAX;
            if ((assigned < n && random <= prob) || fillRemaining) cols[assigned++] = j;
        }
    }
    rows[dim] = n;
}

// One thread per row is efficient for very short rows; a warp per row exposes
// more memory parallelism for the longer rows typical of this benchmark.
__global__ void spmv_short(const double* val, const index_t* cols, const index_t* rows,
                           const double* vec, index_t nrows, double* out) {
    for (uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < nrows; row += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        double sum = 0.0;
        for (index_t j = rows[row]; j < rows[row + 1]; ++j) sum += val[j] * vec[cols[j]];
        out[row] = sum;
    }
}

__global__ void spmv_warp(const double* val, const index_t* cols, const index_t* rows,
                          const double* vec, index_t nrows, double* out) {
    const unsigned lane = threadIdx.x & 31;
    const uint64_t first = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x / 32;
    for (uint64_t row = first; row < nrows; row += stride) {
        double sum = 0.0;
        const index_t end = rows[row + 1];
        for (uint64_t j = static_cast<uint64_t>(rows[row]) + lane; j < end; j += 32)
            sum += val[j] * vec[cols[j]];
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) out[row] = sum;
    }
}

// MPI counts are int, while CSR arrays may contain up to UINT32_MAX items.
template <typename T>
static void send_array(const T* data, uint64_t count, MPI_Datatype type, int peer, int tag) {
    while (count) {
        const int part = static_cast<int>(std::min<uint64_t>(count, INT_MAX));
        MPI_Send(data, part, type, peer, tag, MPI_COMM_WORLD);
        data += part; count -= part;
    }
}
template <typename T>
static void recv_array(T* data, uint64_t count, MPI_Datatype type, int peer, int tag) {
    while (count) {
        const int part = static_cast<int>(std::min<uint64_t>(count, INT_MAX));
        MPI_Recv(data, part, type, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += part; count -= part;
    }
}
template <typename T>
static void broadcast_array(T* data, uint64_t count, MPI_Datatype type) {
    while (count) {
        const int part = static_cast<int>(std::min<uint64_t>(count, INT_MAX));
        MPI_Bcast(data, part, type, 0, MPI_COMM_WORLD);
        data += part; count -= part;
    }
}
static index_t row_boundary(index_t rows, int rank, int size) {
    return static_cast<index_t>(static_cast<uint64_t>(rows) * rank / size);
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
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
            if (!rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    const uint64_t square = static_cast<uint64_t>(numRows) * numRows;
    if (!numRows || !sparsity || square > std::numeric_limits<index_t>::max()) {
        if (!rank) fprintf(stderr, "Matrix dimension and sparsity must be positive, and dimension squared must fit in 32 bits.\n");
        MPI_Finalize(); return 1;
    }
    const index_t nItems = static_cast<index_t>(square / sparsity);
    if (!rank) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, 100.0 * (1.0 - static_cast<double>(nItems) / square));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    // Local rank selects the GPU, respecting CUDA_VISIBLE_DEVICES and per-rank masks.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, deviceCount;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    const index_t rowBegin = row_boundary(numRows, rank, size);
    const index_t rowEnd = row_boundary(numRows, rank + 1, size);
    const index_t localRows = rowEnd - rowBegin;
    std::vector<double> vec(numRows), vals;
    std::vector<index_t> cols, rows(static_cast<size_t>(localRows) + 1);
    if (!rank) {
        std::vector<double> allVals(nItems);
        std::vector<index_t> allCols(nItems), allRows(static_cast<size_t>(numRows) + 1);
        // MPI initialization may itself use rand(); the original program starts
        // with the C library's default seed of 1.
        srand(1);
        fill(vec.data(), numRows, maxVal);
        fill(allVals.data(), nItems, maxVal);
        initRandomMatrix(allCols.data(), allRows.data(), nItems, numRows);
        for (int p = 1; p < size; ++p) {
            const index_t begin = row_boundary(numRows, p, size);
            const index_t end = row_boundary(numRows, p + 1, size);
            const index_t base = allRows[begin], count = allRows[end] - base;
            MPI_Send(&count, 1, MPI_UINT32_T, p, 0, MPI_COMM_WORLD);
            send_array(allRows.data() + begin, static_cast<uint64_t>(end - begin) + 1, MPI_UINT32_T, p, 1);
            if (count) {
                send_array(allCols.data() + base, count, MPI_UINT32_T, p, 2);
                send_array(allVals.data() + base, count, MPI_DOUBLE, p, 3);
            }
        }
        std::copy_n(allRows.data() + rowBegin, static_cast<size_t>(localRows) + 1, rows.data());
        cols.assign(allCols.begin() + allRows[rowBegin], allCols.begin() + allRows[rowEnd]);
        vals.assign(allVals.begin() + allRows[rowBegin], allVals.begin() + allRows[rowEnd]);
    } else {
        index_t count;
        MPI_Recv(&count, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        recv_array(rows.data(), static_cast<uint64_t>(localRows) + 1, MPI_UINT32_T, 0, 1);
        cols.resize(count); vals.resize(count);
        if (count) {
            recv_array(cols.data(), count, MPI_UINT32_T, 0, 2);
            recv_array(vals.data(), count, MPI_DOUBLE, 0, 3);
        }
    }
    broadcast_array(vec.data(), numRows, MPI_DOUBLE);
    const index_t base = rows[0];
    #pragma omp parallel for schedule(static) if(localRows > 4096)
    for (uint64_t i = 0; i <= localRows; ++i) rows[i] -= base;

    std::vector<double> reference;
    if (validate) {
        if (!rank) printf("Computing reference solution...\n");
        reference.resize(localRows);
        #pragma omp parallel for schedule(static) if(localRows > 256)
        for (uint64_t i = 0; i < localRows; ++i) {
            double sum = 0.0;
            for (index_t j = rows[i]; j < rows[i + 1]; ++j) sum += vals[j] * vec[cols[j]];
            reference[i] = sum;
        }
    }

    double *dVals = nullptr, *dVec = nullptr, *dOut = nullptr;
    index_t *dCols = nullptr, *dRows = nullptr;
    if (localRows) {
        CUDA_CHECK(cudaMalloc(&dRows, rows.size() * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&dVec, vec.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dOut, static_cast<size_t>(localRows) * sizeof(double)));
        if (!vals.empty()) {
            CUDA_CHECK(cudaMalloc(&dVals, vals.size() * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&dCols, cols.size() * sizeof(index_t)));
            CUDA_CHECK(cudaMemcpy(dVals, vals.data(), vals.size() * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dCols, cols.data(), cols.size() * sizeof(index_t), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(dRows, rows.data(), rows.size() * sizeof(index_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dVec, vec.data(), vec.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dOut, 0, static_cast<size_t>(localRows) * sizeof(double)));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    constexpr unsigned threads = 256;
    const bool shortRows = vals.size() <= static_cast<uint64_t>(localRows) * 8;
    const uint64_t work = shortRows ? localRows : static_cast<uint64_t>(localRows) * 32;
    const unsigned blocks = static_cast<unsigned>(std::min<uint64_t>((work + threads - 1) / threads, 65535));
    // Exclude first-launch module loading and JIT compilation from the benchmark.
    if (localRows && iterations) {
        if (shortRows) spmv_short<<<blocks, threads>>>(dVals, dCols, dRows, dVec, localRows, dOut);
        else spmv_warp<<<blocks, threads>>>(dVals, dCols, dRows, dVec, localRows, dOut);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    if (!rank) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (localRows) {
        for (index_t iter = 0; iter < iterations; ++iter) {
            if (shortRows) spmv_short<<<blocks, threads>>>(dVals, dCols, dRows, dVec, localRows, dOut);
            else spmv_warp<<<blocks, threads>>>(dVals, dCols, dRows, dVec, localRows, dOut);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        printf("Average time per iteration: %.3f ms\n", iterations ? duration * 1000.0 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", duration > 0 ? 2.0 * nItems * iterations / duration / 1e9 : 0.0);
    }
    std::vector<double> out(localRows, 0.0);
    if (localRows && (validate || printResults))
        CUDA_CHECK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(double), cudaMemcpyDeviceToHost));
    if (printResults) {
        if (!rank) {
            std::vector<double> allOut(numRows);
            std::copy(out.begin(), out.end(), allOut.begin() + rowBegin);
            for (int p = 1; p < size; ++p) {
                const index_t begin = row_boundary(numRows, p, size), end = row_boundary(numRows, p + 1, size);
                recv_array(allOut.data() + begin, end - begin, MPI_DOUBLE, p, 4);
            }
            print_results(allOut, "OutputVector");
        } else send_array(out.data(), localRows, MPI_DOUBLE, 0, 4);
    }
    int valid = 1;
    if (validate) {
        if (!rank) printf("Validating result...\n");
        for (index_t i = 0; i < localRows; ++i) {
            const double ref = reference[i], res = out[i];
            const bool bad = std::abs(ref) < 1e-10 ? std::abs(res) > MAX_RELATIVE_ERROR
                              : std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR;
            if (bad) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", rowBegin + i, ref, res);
                valid = 0; break;
            }
        }
        int allValid;
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        valid = allValid;
        if (!rank) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    if (dVals) CUDA_CHECK(cudaFree(dVals));
    if (dCols) CUDA_CHECK(cudaFree(dCols));
    if (dRows) CUDA_CHECK(cudaFree(dRows));
    if (dVec) CUDA_CHECK(cudaFree(dVec));
    if (dOut) CUDA_CHECK(cudaFree(dOut));
    MPI_Finalize();
    return valid ? 0 : 1;
}
