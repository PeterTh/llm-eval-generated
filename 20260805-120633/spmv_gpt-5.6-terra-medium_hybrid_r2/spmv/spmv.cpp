#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int THREADS_PER_ROW = 256;

#define CUDA_CHECK(call) do { \
    const cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error_)); \
    } \
} while (0)

void fill(double* a, index_t n, double maxVal) {
    for (index_t i = 0; i < n; ++i)
        a[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

void initRandomMatrix(index_t* cols, index_t* rowDelimiters, index_t n, index_t dim) {
    index_t nnzAssigned = 0;
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * dim);
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const index_t entriesLeft = dim * dim - (i * dim + j);
            if (entriesLeft <= n - nnzAssigned) fillRemaining = true;
            const double randomValue = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randomValue <= prob) || fillRemaining)
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

__global__ void spmvKernel(const double* __restrict__ val,
                           const index_t* __restrict__ cols,
                           const index_t* __restrict__ rows,
                           const double* __restrict__ vec,
                           double* __restrict__ out, index_t localRows) {
    const index_t row = blockIdx.x;
    if (row >= localRows) return;
    double sum = 0.0;
    for (index_t j = rows[row] + threadIdx.x; j < rows[row + 1]; j += blockDim.x)
        sum += val[j] * vec[cols[j]];
    __shared__ double partial[THREADS_PER_ROW];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = partial[0];
}

bool verifyResults(const double* reference, const double* result, index_t size) {
    for (index_t i = 0; i < size; ++i) {
        if (std::abs(reference[i]) < 1e-10) {
            if (std::abs(result[i]) > MAX_RELATIVE_ERROR) return false;
        } else if (std::abs((result[i] - reference[i]) / reference[i]) > MAX_RELATIVE_ERROR) return false;
    }
    return true;
}

void printUsage(const char* p) {
    printf("Usage: %s [options]\n", p);
    printf("  -n <num>     Number of rows/columns (default: 1024)\n"
           "  -s <num>     One out of N entries is non-zero (default: 10)\n"
           "  -i <num>     Number of iterations (default: 10)\n"
           "  -m <val>     Maximum element value (default: 1.0)\n"
           "  -v           Enable validation\n  -r           Print results\n  -h           Show help\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    int optionError = 0;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (!strcmp(argv[i], "-n") && i + 1 < argc) numRows = static_cast<index_t>(atoi(argv[++i]));
            else if (!strcmp(argv[i], "-s") && i + 1 < argc) sparsity = static_cast<index_t>(atoi(argv[++i]));
            else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = static_cast<index_t>(atoi(argv[++i]));
            else if (!strcmp(argv[i], "-m") && i + 1 < argc) maxVal = atof(argv[++i]);
            else if (!strcmp(argv[i], "-v")) validate = true;
            else if (!strcmp(argv[i], "-r")) printResults = true;
            else if (!strcmp(argv[i], "-h")) { printUsage(argv[0]); MPI_Finalize(); return 0; }
            else { printf("Unknown option: %s\n", argv[i]); optionError = 1; }
        }
        if (!numRows || !sparsity || !iterations || numRows > std::numeric_limits<index_t>::max() / numRows) optionError = 1;
    }
    MPI_Bcast(&optionError, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (optionError) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 1; }
    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (ranks > static_cast<int>(numRows)) {
        if (rank == 0) fprintf(stderr, "The number of MPI ranks must not exceed the number of rows.\n");
        MPI_Finalize();
        return 1;
    }

    const index_t nItems = (numRows * numRows) / sparsity;
    const index_t firstRow = static_cast<index_t>((static_cast<uint64_t>(rank) * numRows) / ranks);
    const index_t lastRow = static_cast<index_t>((static_cast<uint64_t>(rank + 1) * numRows) / ranks);
    const index_t localRows = lastRow - firstRow;
    std::vector<double> vec(numRows), localVal;
    std::vector<index_t> localCols, localRowDelimiters(localRows + 1);
    std::vector<double> result(rank == 0 ? numRows : 0), reference;
    std::vector<int> rowCounts(ranks), rowDispls(ranks), nnzCounts(ranks), nnzDispls(ranks);

    std::vector<double> values;
    std::vector<index_t> cols, rowDelimiters;
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\nMatrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\nNon-zero elements: %u (%.2f%% sparse)\nIterations: %u\nMax value: %.2f\nValidation: %s\n",
               numRows, numRows, sparsity, nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)), iterations, maxVal, validate ? "enabled" : "disabled");
        values.resize(nItems); cols.resize(nItems); rowDelimiters.resize(numRows + 1);
        fill(vec.data(), numRows, maxVal); fill(values.data(), nItems, maxVal);
        initRandomMatrix(cols.data(), rowDelimiters.data(), nItems, numRows);
        if (validate) { reference.resize(numRows); spmvCpu(values.data(), cols.data(), rowDelimiters.data(), vec.data(), numRows, reference.data()); }
        for (int r = 0; r < ranks; ++r) {
            const index_t begin = static_cast<index_t>((static_cast<uint64_t>(r) * numRows) / ranks);
            const index_t end = static_cast<index_t>((static_cast<uint64_t>(r + 1) * numRows) / ranks);
            rowCounts[r] = static_cast<int>(end - begin); rowDispls[r] = static_cast<int>(begin);
            nnzCounts[r] = static_cast<int>(rowDelimiters[end] - rowDelimiters[begin]);
            nnzDispls[r] = static_cast<int>(rowDelimiters[begin]);
        }
    }
    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int localNnz = 0;
    MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_INT, &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    localVal.resize(localNnz); localCols.resize(localNnz);
    MPI_Scatterv(rank == 0 ? values.data() : nullptr, rank == 0 ? nnzCounts.data() : nullptr, rank == 0 ? nnzDispls.data() : nullptr, MPI_DOUBLE, localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? cols.data() : nullptr, rank == 0 ? nnzCounts.data() : nullptr, rank == 0 ? nnzDispls.data() : nullptr, MPI_UINT32_T, localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (index_t i = 0; i <= localRows; ++i) localRowDelimiters[i] = rowDelimiters[i];
        for (int r = 1; r < ranks; ++r) {
            const index_t begin = static_cast<index_t>((static_cast<uint64_t>(r) * numRows) / ranks);
            const index_t end = static_cast<index_t>((static_cast<uint64_t>(r + 1) * numRows) / ranks);
            std::vector<index_t> offsets(end - begin + 1);
            for (index_t i = 0; i <= end - begin; ++i) offsets[i] = rowDelimiters[begin + i] - rowDelimiters[begin];
            MPI_Send(offsets.data(), static_cast<int>(offsets.size()), MPI_UINT32_T, r, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(localRowDelimiters.data(), static_cast<int>(localRows + 1), MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0; CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { if (rank == 0) fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);
    double *dVal, *dVec, *dOut; index_t *dCols, *dRows;
    CUDA_CHECK(cudaMalloc(&dVal, localNnz * sizeof(double))); CUDA_CHECK(cudaMalloc(&dCols, localNnz * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dRows, (localRows + 1) * sizeof(index_t))); CUDA_CHECK(cudaMalloc(&dVec, numRows * sizeof(double))); CUDA_CHECK(cudaMalloc(&dOut, localRows * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dVal, localVal.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dCols, localCols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRows, localRowDelimiters.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dVec, vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> localOut(localRows);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (index_t it = 0; it < iterations; ++it) spmvKernel<<<localRows, THREADS_PER_ROW>>>(dVal, dCols, dRows, dVec, dOut, localRows);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaMemcpy(localOut.data(), dOut, localRows * sizeof(double), cudaMemcpyDeviceToHost));
    const auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count(), maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE, rank == 0 ? result.data() : nullptr, rank == 0 ? rowCounts.data() : nullptr, rank == 0 ? rowDispls.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n", maxElapsed, maxElapsed / iterations, (2.0 * nItems * iterations) / (maxElapsed / 1000.0) / 1e9);
        if (printResults) print_results(result, "OutputVector");
        if (validate) printf("Validation: %s\n", verifyResults(reference.data(), result.data(), numRows) ? "PASSED" : "FAILED");
    }
    CUDA_CHECK(cudaFree(dVal)); CUDA_CHECK(cudaFree(dCols)); CUDA_CHECK(cudaFree(dRows)); CUDA_CHECK(cudaFree(dVec)); CUDA_CHECK(cudaFree(dOut));
    const bool valid = !validate || rank != 0 || verifyResults(reference.data(), result.data(), numRows);
    MPI_Finalize();
    return valid ? 0 : 1;
}
