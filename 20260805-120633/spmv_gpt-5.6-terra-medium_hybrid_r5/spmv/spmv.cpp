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

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t cudaStatus = (call);                                            \
        if (cudaStatus != cudaSuccess) {                                                  \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                         cudaGetErrorString(cudaStatus));                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

void fill(double* A, index_t n, double maxVal) {
    for (index_t i = 0; i < n; ++i)
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

void initRandomMatrix(index_t* cols, index_t* rowDelimiters, index_t n, index_t dim) {
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
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining)
                cols[nnzAssigned++] = j;
        }
    }
    rowDelimiters[dim] = n;
}

void spmvCpu(const double* val, const index_t* cols, const index_t* rows,
             const double* vec, index_t dim, double* out) {
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < dim; ++i) {
        double sum = 0.0;
        for (index_t j = rows[i]; j < rows[i + 1]; ++j) sum += val[j] * vec[cols[j]];
        out[i] = sum;
    }
}

__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rows, const double* __restrict__ vec,
                           double* __restrict__ out, index_t rowCount) {
    constexpr int warpSize = 32;
    constexpr int warpsPerBlock = 8;
    const int lane = threadIdx.x % warpSize;
    const index_t row = blockIdx.x * warpsPerBlock + threadIdx.x / warpSize;
    if (row >= rowCount) return;
    double sum = 0.0;
    for (index_t j = rows[row] + lane; j < rows[row + 1]; j += warpSize) sum += val[j] * vec[cols[j]];
    for (int offset = warpSize / 2; offset; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (!lane) out[row] = sum;
}

bool verifyResults(const double* reference, const double* result, index_t size) {
    for (index_t i = 0; i < size; ++i) {
        if (std::abs(reference[i]) < 1e-10) {
            if (std::abs(result[i]) > MAX_RELATIVE_ERROR) {
                std::printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, reference[i], result[i]);
                return false;
            }
        } else if (std::abs((result[i] - reference[i]) / reference[i]) > MAX_RELATIVE_ERROR) {
            std::printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, reference[i], result[i]);
            return false;
        }
    }
    return true;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>  Matrix dimension (default: 1024)\n"
                "  -s <num>  1 out of N entries is non-zero (default: 10)\n"
                "  -i <num>  Iterations (default: 10)\n  -m <val>  Maximum element value (default: 1.0)\n"
                "  -v        Enable validation\n  -r        Print results for external validation\n  -h        Show this help\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) numRows = static_cast<index_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) sparsity = static_cast<index_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = static_cast<index_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-m") && i + 1 < argc) maxVal = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (numRows == 0 || sparsity == 0 || iterations == 0) {
        if (!rank) std::fprintf(stderr, "-n, -s, and -i must be positive\n");
        MPI_Finalize(); return 1;
    }

    const index_t nItems = (numRows * numRows) / sparsity;
    std::vector<int> rowCounts(ranks), rowDispls(ranks), nnzCounts(ranks), nnzDispls(ranks);
    for (int r = 0; r < ranks; ++r) {
        rowCounts[r] = static_cast<int>(numRows / ranks + (static_cast<index_t>(r) < numRows % ranks));
        rowDispls[r] = r ? rowDispls[r - 1] + rowCounts[r - 1] : 0;
    }

    std::vector<double> h_val, h_vec, h_out, h_reference;
    std::vector<index_t> h_cols, h_rows;
    if (!rank) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\nMatrix size: %u x %u\n", numRows, numRows);
        std::printf("Sparsity: 1 out of %u entries is non-zero\nNon-zero elements: %u (%.2f%% sparse)\nIterations: %u\nMax value: %.2f\nValidation: %s\n",
                    sparsity, nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)), iterations, maxVal, validate ? "enabled" : "disabled");
        h_val.resize(nItems); h_cols.resize(nItems); h_rows.resize(numRows + 1); h_vec.resize(numRows); h_out.resize(numRows);
        std::printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal); fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rows.data(), nItems, numRows);
        for (int r = 0; r < ranks; ++r) {
            nnzDispls[r] = static_cast<int>(h_rows[rowDispls[r]]);
            nnzCounts[r] = static_cast<int>(h_rows[rowDispls[r] + rowCounts[r]] - h_rows[rowDispls[r]]);
        }
        if (validate) { std::printf("Computing reference solution...\n"); h_reference.resize(numRows); spmvCpu(h_val.data(), h_cols.data(), h_rows.data(), h_vec.data(), numRows, h_reference.data()); }
    }

    MPI_Bcast(nnzCounts.data(), ranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzDispls.data(), ranks, MPI_INT, 0, MPI_COMM_WORLD);
    const int localRows = rowCounts[rank], localNnz = nnzCounts[rank];
    std::vector<double> localVal(localNnz), localOut(localRows), localVec(numRows);
    std::vector<index_t> localCols(localNnz), localRowsCsr(localRows + 1);
    MPI_Scatterv(rank ? nullptr : h_val.data(), nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE, localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank ? nullptr : h_cols.data(), nnzCounts.data(), nnzDispls.data(), MPI_UNSIGNED, localCols.data(), localNnz, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    std::vector<int> rowDelimiterCounts(ranks), rowDelimiterDispls(ranks);
    for (int r = 0; r < ranks; ++r) { rowDelimiterCounts[r] = rowCounts[r] + 1; rowDelimiterDispls[r] = rowDispls[r]; }
    MPI_Scatterv(rank ? nullptr : h_rows.data(), rowDelimiterCounts.data(), rowDelimiterDispls.data(), MPI_UNSIGNED, localRowsCsr.data(), localRows + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#pragma omp parallel for schedule(static)
    for (int i = 0; i <= localRows; ++i) localRowsCsr[i] -= static_cast<index_t>(nnzDispls[rank]);
    MPI_Bcast(rank ? localVec.data() : h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (!rank) localVec.swap(h_vec); // Root also uses the common local vector below.

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, deviceCount;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { if (!rank) std::fprintf(stderr, "No CUDA devices available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    double *dVal, *dVec, *dOut; index_t *dCols, *dRows;
    CUDA_CHECK(cudaMalloc(&dVal, std::max<size_t>(1, localVal.size()) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCols, std::max<size_t>(1, localCols.size()) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dRows, localRowsCsr.size() * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dVec, localVec.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dOut, std::max<size_t>(1, localOut.size()) * sizeof(double)));
    if (localNnz) { CUDA_CHECK(cudaMemcpy(dVal, localVal.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dCols, localCols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice)); }
    CUDA_CHECK(cudaMemcpy(dRows, localRowsCsr.data(), localRowsCsr.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVec, localVec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    constexpr int threads = 256;
    const int blocks = (localRows + 7) / 8;
    for (index_t iter = 0; iter < iterations; ++iter) {
        if (blocks) spmvKernel<<<blocks, threads>>>(dVal, dCols, dRows, dVec, dOut, localRows);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double localSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (localRows) CUDA_CHECK(cudaMemcpy(localOut.data(), dOut, localRows * sizeof(double), cudaMemcpyDeviceToHost));
    MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE, rank ? nullptr : h_out.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(dVal)); CUDA_CHECK(cudaFree(dCols)); CUDA_CHECK(cudaFree(dRows)); CUDA_CHECK(cudaFree(dVec)); CUDA_CHECK(cudaFree(dOut));
    int result = 0;
    if (!rank) {
        const double milliseconds = elapsedSeconds * 1000.0;
        std::printf("Computation time: %.3f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n", milliseconds, milliseconds / iterations, (2.0 * nItems * iterations) / elapsedSeconds / 1e9);
        if (printResults) print_results(h_out, "OutputVector");
        if (validate) { std::printf("Validating result...\n"); result = verifyResults(h_reference.data(), h_out.data(), numRows) ? 0 : 1; std::printf("Validation: %s\n", result ? "FAILED" : "PASSED"); }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
