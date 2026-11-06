#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;
constexpr double MAX_RELATIVE_ERROR = 0.02;

// Keep the original sequential random-number stream and its initialization order.
void fill(double* a, index_t n, double maxVal) {
    for (index_t i = 0; i < n; ++i)
        a[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

void initRandomMatrix(index_t* cols, index_t* rowDelimiters, index_t n, index_t dim) {
    index_t assigned = 0;
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * dim);
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = assigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t left = static_cast<uint64_t>(dim) * dim -
                                  (static_cast<uint64_t>(i) * dim + j);
            if (left <= static_cast<uint64_t>(n - assigned)) fillRemaining = true;
            const double random = static_cast<double>(rand()) / RAND_MAX;
            if ((assigned < n && random <= prob) || fillRemaining) cols[assigned++] = j;
        }
    }
    rowDelimiters[dim] = n;
}

void spmvCpu(const double* val, const index_t* cols, const index_t* rows,
             const double* vec, index_t nrows, double* out) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(nrows); ++i) {
        double sum = 0.0;
        for (index_t j = rows[i]; j < rows[i + 1]; ++j) sum += val[j] * vec[cols[j]];
        out[i] = sum;
    }
}

bool verifyResults(const double* reference, const double* result, index_t size, index_t rowStart) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i], res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", rowStart + i, ref, res);
                return false;
            }
        } else {
            const double error = std::abs((res - ref) / ref);
            if (error > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n", rowStart + i, ref, res, error);
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

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    int r; MPI_Comm_rank(MPI_COMM_WORLD, &r); \
    fprintf(stderr, "Rank %d: %s failed: %s\n", r, #call, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// A warp cooperatively processes a row for sufficiently long rows. Each lane
// reads consecutive CSR entries and the partial sums are reduced in the warp.
__global__ void spmvWarp(const double* __restrict__ val, const index_t* __restrict__ cols,
                         const index_t* __restrict__ rows, const double* __restrict__ vec,
                         index_t nrows, double* __restrict__ out) {
    const uint64_t thread = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t row = thread / 32;
    const unsigned lane = threadIdx.x & 31;
    if (row >= nrows) return;
    double sum = 0.0;
    for (uint64_t j = static_cast<uint64_t>(rows[row]) + lane; j < rows[row + 1]; j += 32)
        sum += val[j] * vec[cols[j]];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) out[row] = sum;
}

__global__ void spmvScalar(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rows, const double* __restrict__ vec,
                           index_t nrows, double* __restrict__ out) {
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    double sum = 0.0;
    for (index_t j = rows[row]; j < rows[row + 1]; ++j) sum += val[j] * vec[cols[j]];
    out[row] = sum;
}

// MPI counts are int; split transfers so all representable CSR arrays work.
template <typename T>
void transfer(T* data, uint64_t count, MPI_Datatype type, int peer, int tag, bool send) {
    constexpr uint64_t chunkLimit = 1u << 28;
    uint64_t offset = 0;
    while (offset < count) {
        const int chunk = static_cast<int>(std::min(count - offset, chunkLimit));
        if (send) MPI_Send(data + offset, chunk, type, peer, tag, MPI_COMM_WORLD);
        else MPI_Recv(data + offset, chunk, type, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        offset += chunk;
    }
}

template <typename T>
void broadcast(T* data, uint64_t count, MPI_Datatype type) {
    constexpr uint64_t chunkLimit = 1u << 28;
    for (uint64_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(count - offset, chunkLimit));
        MPI_Bcast(data + offset, chunk, type, 0, MPI_COMM_WORLD);
        offset += chunk;
    }
}

template <typename T>
T* deviceAlloc(size_t count) {
    T* p = nullptr;
    if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)));
    return p;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numRows = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) sparsity = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) maxVal = atof(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    const uint64_t matrixEntries = static_cast<uint64_t>(numRows) * numRows;
    if (!sparsity || matrixEntries / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Invalid sparsity or matrix has too many nonzero entries for 32-bit CSR indices\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(matrixEntries / sparsity);
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / static_cast<double>(matrixEntries)));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

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

    std::vector<double> vec(numRows), allVal;
    std::vector<index_t> allCols, allRows, boundaries(static_cast<size_t>(ranks) + 1);
    if (rank == 0) {
        allVal.resize(nItems);
        allCols.resize(nItems);
        allRows.resize(static_cast<size_t>(numRows) + 1);
        fill(vec.data(), numRows, maxVal);
        fill(allVal.data(), nItems, maxVal);
        initRandomMatrix(allCols.data(), allRows.data(), nItems, numRows);
        boundaries[0] = 0;
        for (int r = 1; r < ranks; ++r) {
            if (nItems == 0) boundaries[r] = static_cast<index_t>(static_cast<uint64_t>(numRows) * r / ranks);
            else {
                const index_t target = static_cast<index_t>(static_cast<uint64_t>(nItems) * r / ranks);
                boundaries[r] = static_cast<index_t>(std::lower_bound(allRows.begin(), allRows.end(), target) - allRows.begin());
            }
        }
        boundaries[ranks] = numRows;
    }
    broadcast(vec.data(), numRows, MPI_DOUBLE);
    broadcast(boundaries.data(), static_cast<uint64_t>(ranks) + 1, MPI_UINT32_T);
    const index_t rowStart = boundaries[rank], localRows = boundaries[rank + 1] - rowStart;
    std::vector<index_t> rows(static_cast<size_t>(localRows) + 1);
    std::vector<index_t> cols;
    std::vector<double> val;
    if (rank == 0) {
        for (int r = 0; r < ranks; ++r) {
            const index_t firstRow = boundaries[r], endRow = boundaries[r + 1];
            const index_t first = allRows[firstRow], count = allRows[endRow] - first;
            if (r == 0) {
                std::copy(allRows.begin() + firstRow, allRows.begin() + static_cast<size_t>(endRow) + 1, rows.begin());
                if (count) {
                    cols.assign(allCols.begin() + first, allCols.begin() + static_cast<size_t>(first) + count);
                    val.assign(allVal.begin() + first, allVal.begin() + static_cast<size_t>(first) + count);
                }
            } else {
                transfer(allRows.data() + firstRow, static_cast<uint64_t>(endRow) - firstRow + 1, MPI_UINT32_T, r, 10, true);
                if (count) {
                    transfer(allCols.data() + first, count, MPI_UINT32_T, r, 11, true);
                    transfer(allVal.data() + first, count, MPI_DOUBLE, r, 12, true);
                }
            }
        }
        std::vector<index_t>().swap(allRows);
        std::vector<index_t>().swap(allCols);
        std::vector<double>().swap(allVal);
    } else {
        transfer(rows.data(), static_cast<uint64_t>(localRows) + 1, MPI_UINT32_T, 0, 10, false);
        const index_t count = rows.back() - rows.front();
        cols.resize(count);
        val.resize(count);
        transfer(cols.data(), count, MPI_UINT32_T, 0, 11, false);
        transfer(val.data(), count, MPI_DOUBLE, 0, 12, false);
    }
    const index_t offset = rows[0];
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i <= static_cast<int64_t>(localRows); ++i) rows[i] -= offset;

    std::vector<double> reference, out(localRows, 0.0);
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        reference.resize(localRows);
        spmvCpu(val.data(), cols.data(), rows.data(), vec.data(), localRows, reference.data());
    }
    double* dVal = deviceAlloc<double>(val.size());
    index_t* dCols = deviceAlloc<index_t>(cols.size());
    index_t* dRows = deviceAlloc<index_t>(rows.size());
    double* dVec = deviceAlloc<double>(vec.size());
    double* dOut = deviceAlloc<double>(out.size());
    if (!val.empty()) CUDA_CHECK(cudaMemcpy(dVal, val.data(), val.size() * sizeof(double), cudaMemcpyHostToDevice));
    if (!cols.empty()) CUDA_CHECK(cudaMemcpy(dCols, cols.data(), cols.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRows, rows.data(), rows.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    if (!vec.empty()) CUDA_CHECK(cudaMemcpy(dVec, vec.data(), vec.size() * sizeof(double), cudaMemcpyHostToDevice));
    if (!out.empty()) CUDA_CHECK(cudaMemset(dOut, 0, out.size() * sizeof(double)));
    constexpr unsigned blockSize = 256;
    const auto launch = [&]() {
        if (localRows) {
            if (val.size() / localRows >= 16) {
                const unsigned blocks = static_cast<unsigned>((static_cast<uint64_t>(localRows) + blockSize / 32 - 1) / (blockSize / 32));
                spmvWarp<<<blocks, blockSize>>>(dVal, dCols, dRows, dVec, localRows, dOut);
            } else {
                const unsigned blocks = static_cast<unsigned>((static_cast<uint64_t>(localRows) + blockSize - 1) / blockSize);
                spmvScalar<<<blocks, blockSize>>>(dVal, dCols, dRows, dVec, localRows, dOut);
            }
        }
    };
    if (iterations) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!out.empty()) CUDA_CHECK(cudaMemcpy(out.data(), dOut, out.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dVal)); CUDA_CHECK(cudaFree(dCols)); CUDA_CHECK(cudaFree(dRows));
    CUDA_CHECK(cudaFree(dVec)); CUDA_CHECK(cudaFree(dOut));
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        printf("Average time per iteration: %.3f ms\n", maxElapsed * 1000 / iterations);
        printf("Performance: %.3f GFLOPS\n", 2.0 * nItems * iterations / maxElapsed / 1e9);
    }
    if (printResults) {
        if (rank == 0) {
            std::vector<double> allOut(numRows);
            if (!out.empty()) std::copy(out.begin(), out.end(), allOut.begin() + rowStart);
            for (int r = 1; r < ranks; ++r) {
                const index_t count = boundaries[r + 1] - boundaries[r];
                if (count) transfer(allOut.data() + boundaries[r], count, MPI_DOUBLE, r, 20, false);
            }
            print_results(allOut, "OutputVector");
        } else transfer(out.data(), localRows, MPI_DOUBLE, 0, 20, true);
    }
    int localValid = 1, valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        localValid = verifyResults(reference.data(), out.data(), localRows, rowStart) ? 1 : 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
