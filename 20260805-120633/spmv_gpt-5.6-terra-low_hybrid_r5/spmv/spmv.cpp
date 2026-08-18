#include <algorithm>
#include <chrono>
#include <cmath>
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
constexpr int THREADS_PER_BLOCK = 256;

#define CUDA_CHECK(call) do { \
    const cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

void fill(double* A, index_t n, double maxVal) {
    for (index_t i = 0; i < n; ++i) A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

void initRandomMatrix(index_t* cols, index_t* rows, index_t n, index_t dim) {
    index_t assigned = 0;
    const double probability = static_cast<double>(n) / (static_cast<double>(dim) * dim);
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rows[i] = assigned;
        for (index_t j = 0; j < dim; ++j) {
            const index_t left = dim * dim - (i * dim + j);
            if (left <= n - assigned) fillRemaining = true;
            if ((assigned < n && static_cast<double>(rand()) / RAND_MAX <= probability) || fillRemaining)
                cols[assigned++] = j;
        }
    }
    rows[dim] = n;
}

void spmvCpu(const double* val, const index_t* cols, const index_t* rows, const double* vec, index_t dim, double* out) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(dim); ++i) {
        double sum = 0.0;
        for (index_t j = rows[i]; j < rows[i + 1]; ++j) sum += val[j] * vec[cols[j]];
        out[i] = sum;
    }
}

__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rows, const double* __restrict__ vec,
                           double* __restrict__ out, index_t localRows) {
    const index_t row = blockIdx.x;
    if (row >= localRows) return;
    double sum = 0.0;
    for (index_t j = rows[row] + threadIdx.x; j < rows[row + 1]; j += blockDim.x)
        sum += val[j] * vec[cols[j]];
    __shared__ double partial[THREADS_PER_BLOCK];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = partial[0];
}

bool verifyResults(const double* reference, const double* result, index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i], res = result[i];
        const double error = std::abs(ref) < 1e-10 ? std::abs(res) : std::abs((res - ref) / ref);
        if (error > MAX_RELATIVE_ERROR) {
            std::printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
            return false;
        }
    }
    return true;
}

void printUsage(const char* prog) {
    std::printf("Usage: %s [options]\n  -n <num>  Matrix dimension (default: 1024)\n"
                "  -s <num>  One nonzero per N entries (default: 10)\n  -i <num>  Iterations (default: 10)\n"
                "  -m <val>  Maximum element value (default: 1.0)\n  -v        Enable validation\n"
                "  -r        Print results for external validation\n  -h        Show this help\n", prog);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { if (!rank) std::fprintf(stderr, "No CUDA devices available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    if (rank == 0) for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) numRows = static_cast<index_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) sparsity = static_cast<index_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = static_cast<index_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-m") && i + 1 < argc) maxVal = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { std::fprintf(stderr, "Unknown option: %s\n", argv[i]); printUsage(argv[0]); MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (!sparsity || !numRows) { if (!rank) std::fprintf(stderr, "-n and -s must be positive\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    const index_t nItems = (numRows * numRows) / sparsity;
    std::vector<int> rowCounts(ranks), rowOffsets(ranks), nnzCounts(ranks), nnzOffsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        rowOffsets[r] = static_cast<int>((static_cast<uint64_t>(numRows) * r) / ranks);
        const int end = static_cast<int>((static_cast<uint64_t>(numRows) * (r + 1)) / ranks);
        rowCounts[r] = end - rowOffsets[r];
    }
    std::vector<double> val, vec(numRows), out, reference;
    std::vector<index_t> cols, rows(numRows + 1);
    if (!rank) {
        val.resize(nItems); cols.resize(nItems); out.resize(numRows);
        std::printf("Sparse Matrix-Vector Multiplication (MPI + OpenMP + CUDA) Benchmark\nMatrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\nNon-zero elements: %u (%.2f%% sparse)\nIterations: %u\nMax value: %.2f\nValidation: %s\nInitializing data structures...\n",
                    numRows, numRows, sparsity, nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)), iterations, maxVal, validate ? "enabled" : "disabled");
        fill(vec.data(), numRows, maxVal); fill(val.data(), nItems, maxVal); initRandomMatrix(cols.data(), rows.data(), nItems, numRows);
        for (int r = 0; r < ranks; ++r) { nnzOffsets[r] = static_cast<int>(rows[rowOffsets[r]]); nnzCounts[r] = static_cast<int>(rows[rowOffsets[r] + rowCounts[r]] - rows[rowOffsets[r]]); }
        if (validate) { std::printf("Computing reference solution...\n"); reference.resize(numRows); spmvCpu(val.data(), cols.data(), rows.data(), vec.data(), numRows, reference.data()); }
    }
    MPI_Bcast(nnzCounts.data(), ranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzOffsets.data(), ranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const int localRows = rowCounts[rank], localNnz = nnzCounts[rank];
    std::vector<double> localVal(localNnz), localOut(localRows);
    std::vector<index_t> localCols(localNnz), localRowDelimiters(localRows + 1);
    MPI_Scatterv(rank ? nullptr : val.data(), nnzCounts.data(), nnzOffsets.data(), MPI_DOUBLE, localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank ? nullptr : cols.data(), nnzCounts.data(), nnzOffsets.data(), MPI_UINT32_T, localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(rows.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    for (int i = 0; i <= localRows; ++i) localRowDelimiters[i] = rows[rowOffsets[rank] + i] - nnzOffsets[rank];

    double *dVal, *dVec, *dOut; index_t *dCols, *dRows;
    CUDA_CHECK(cudaMalloc(&dVal, std::max(1, localNnz) * static_cast<int>(sizeof(double)))); CUDA_CHECK(cudaMalloc(&dCols, std::max(1, localNnz) * static_cast<int>(sizeof(index_t))));
    CUDA_CHECK(cudaMalloc(&dRows, (localRows + 1) * static_cast<int>(sizeof(index_t)))); CUDA_CHECK(cudaMalloc(&dVec, numRows * sizeof(double))); CUDA_CHECK(cudaMalloc(&dOut, std::max(1, localRows) * static_cast<int>(sizeof(double))));
    CUDA_CHECK(cudaMemcpy(dVal, localVal.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dCols, localCols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dRows, localRowDelimiters.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dVec, vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));
    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank) std::printf("Computing SpMV...\n");
    const auto start = std::chrono::steady_clock::now();
    for (index_t it = 0; it < iterations; ++it) if (localRows) { spmvKernel<<<localRows, THREADS_PER_BLOCK>>>(dVal, dCols, dRows, dVec, dOut, localRows); CUDA_CHECK(cudaGetLastError()); }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double localMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    double elapsedMs = 0; MPI_Reduce(&localMs, &elapsedMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(localOut.data(), dOut, localRows * sizeof(double), cudaMemcpyDeviceToHost));
    MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE, rank ? nullptr : out.data(), rowCounts.data(), rowOffsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(dVal)); CUDA_CHECK(cudaFree(dCols)); CUDA_CHECK(cudaFree(dRows)); CUDA_CHECK(cudaFree(dVec)); CUDA_CHECK(cudaFree(dOut)); MPI_Comm_free(&localComm);
    int status = 0;
    if (!rank) { std::printf("Computation time: %.3f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n", elapsedMs, elapsedMs / iterations, (2.0 * nItems * iterations) / (elapsedMs / 1000.0) / 1e9); if (printResults) print_results(out, "OutputVector"); if (validate) { const bool ok = verifyResults(reference.data(), out.data(), numRows); std::printf("Validation: %s\n", ok ? "PASSED" : "FAILED"); status = ok ? 0 : 1; } }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Finalize(); return status;
}
