#include <chrono>
#include <climits>
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

#define CUDA_CHECK(call) do { \
    const cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error_)); \
    } \
} while (0)

void fill(double* a, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i)
        a[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * dim);
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            const index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) fillRemaining = true;
            const double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) cols[nnzAssigned++] = j;
        }
    }
    rowDelimiters[dim] = n;
}

void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
#pragma omp parallel for schedule(static)
    for (int64_t ii = 0; ii < static_cast<int64_t>(dim); ++ii) {
        const index_t i = static_cast<index_t>(ii);
        double sum = 0.0;
#pragma omp simd reduction(+:sum)
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) sum += val[j] * vec[cols[j]];
        out[i] = sum;
    }
}

__global__ void spmvCsrKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                              const index_t* __restrict__ rowDelimiters, const double* __restrict__ vec,
                              const index_t rowBegin, const index_t localRows, double* __restrict__ out) {
    const index_t localRow = blockIdx.x * blockDim.x + threadIdx.x;
    if (localRow >= localRows) return;
    const index_t row = rowBegin + localRow;
    double sum = 0.0;
    const index_t base = rowDelimiters[rowBegin];
    for (index_t j = rowDelimiters[row]; j < rowDelimiters[row + 1]; ++j)
        sum += val[j - base] * vec[cols[j - base]];
    out[localRow] = sum;
}

bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i], res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                std::printf("Validation failed at index %u: reference %.10e, got %.10e\\n", i, ref, res);
                return false;
            }
        } else if (std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR) {
            std::printf("Validation failed at index %u: reference %.10e, got %.10e\\n", i, ref, res);
            return false;
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\\n", progName);
    std::printf("  -n <num>  Matrix dimension (default: 1024)\\n  -s <num>  One nonzero per N entries (default: 10)\\n");
    std::printf("  -i <num>  Iterations (default: 10)\\n  -m <val>  Maximum value (default: 1.0)\\n");
    std::printf("  -v        Enable validation\\n  -r        Print results\\n  -h        Show help\\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) numRows = static_cast<index_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) sparsity = static_cast<index_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = static_cast<index_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-m") && i + 1 < argc) maxVal = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (numRows == 0 || sparsity == 0 || iterations == 0 ||
        static_cast<uint64_t>(numRows) * numRows / sparsity > INT_MAX) {
        if (!rank) std::fprintf(stderr, "Invalid dimensions, sparsity, or iteration count\\n");
        MPI_Finalize(); return 1;
    }
    const index_t nItems = static_cast<index_t>(static_cast<uint64_t>(numRows) * numRows / sparsity);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { if (!rank) std::fprintf(stderr, "No CUDA device available\\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    std::vector<double> val, sourceVal, vec(numRows), out, reference;
    std::vector<index_t> cols, sourceCols, rowDelimiters(numRows + 1);
    if (!rank) {
        sourceVal.resize(nItems); sourceCols.resize(nItems);
        fill(vec.data(), numRows, maxVal); fill(sourceVal.data(), nItems, maxVal);
        initRandomMatrix(sourceCols.data(), rowDelimiters.data(), nItems, numRows);
        if (validate) { reference.resize(numRows); spmvCpu(sourceVal.data(), sourceCols.data(), rowDelimiters.data(), vec.data(), numRows, reference.data()); }
        std::printf("Sparse Matrix-Vector Multiplication (MPI + OpenMP + CUDA) Benchmark\\nMatrix size: %u x %u\\nSparsity: 1 out of %u entries is non-zero\\nNon-zero elements: %u (%.2f%% sparse)\\nIterations: %u\\nMax value: %.2f\\nValidation: %s\\n",
                    numRows, numRows, sparsity, nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (static_cast<double>(numRows) * numRows)), iterations, maxVal, validate ? "enabled" : "disabled");
    }
    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t rowBegin = static_cast<index_t>((static_cast<uint64_t>(rank) * numRows) / ranks);
    const index_t rowEnd = static_cast<index_t>((static_cast<uint64_t>(rank + 1) * numRows) / ranks);
    const index_t localRows = rowEnd - rowBegin;
    const index_t localNnz = rowDelimiters[rowEnd] - rowDelimiters[rowBegin];
    cols.resize(localNnz); val.resize(localNnz); out.resize(localRows);
    std::vector<int> counts(ranks), displs(ranks), rowCounts(ranks), rowDispls(ranks);
    for (int p = 0; p < ranks; ++p) {
        const index_t begin = static_cast<index_t>((static_cast<uint64_t>(p) * numRows) / ranks);
        const index_t end = static_cast<index_t>((static_cast<uint64_t>(p + 1) * numRows) / ranks);
        counts[p] = static_cast<int>(rowDelimiters[end] - rowDelimiters[begin]); displs[p] = static_cast<int>(rowDelimiters[begin]);
        rowCounts[p] = static_cast<int>(end - begin); rowDispls[p] = static_cast<int>(begin);
    }
    MPI_Scatterv(rank ? nullptr : sourceVal.data(), counts.data(), displs.data(), MPI_DOUBLE, val.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank ? nullptr : sourceCols.data(), counts.data(), displs.data(), MPI_UINT32_T, cols.data(), static_cast<int>(localNnz), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    double *dVal = nullptr, *dVec = nullptr, *dOut = nullptr;
    index_t *dCols = nullptr, *dRows = nullptr;
    CUDA_CHECK(cudaMalloc(&dVec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dRows, (numRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dOut, std::max<index_t>(localRows, 1) * sizeof(double)));
    if (localNnz) { CUDA_CHECK(cudaMalloc(&dVal, localNnz * sizeof(double))); CUDA_CHECK(cudaMalloc(&dCols, localNnz * sizeof(index_t))); }
    CUDA_CHECK(cudaMemcpy(dVec, vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRows, rowDelimiters.data(), (numRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    if (localNnz) { CUDA_CHECK(cudaMemcpy(dVal, val.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dCols, cols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice)); }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    constexpr int threads = 256;
    for (index_t iter = 0; iter < iterations; ++iter)
        if (localRows) spmvCsrKernel<<<(localRows + threads - 1) / threads, threads>>>(dVal, dCols, dRows, dVec, rowBegin, localRows, dOut);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float localMs = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&localMs, start, stop));
    if (localRows) CUDA_CHECK(cudaMemcpy(out.data(), dOut, localRows * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> globalOut(rank == 0 ? numRows : 0);
    MPI_Gatherv(out.data(), static_cast<int>(localRows), MPI_DOUBLE, rank ? nullptr : globalOut.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    float elapsedMs = 0.0f; MPI_Reduce(&localMs, &elapsedMs, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        std::printf("Computation time: %.3f ms\\nAverage time per iteration: %.3f ms\\nPerformance: %.3f GFLOPS\\n", elapsedMs, elapsedMs / iterations, (2.0 * nItems * iterations) / (elapsedMs / 1000.0) / 1e9);
        if (printResults) print_results(globalOut, "OutputVector");
        if (validate) { const bool ok = verifyResults(reference.data(), globalOut.data(), numRows); std::printf("Validation: %s\\n", ok ? "PASSED" : "FAILED"); exitCode = ok ? 0 : 1; }
    }
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(dVal); cudaFree(dCols); cudaFree(dVec); cudaFree(dRows); cudaFree(dOut);
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
