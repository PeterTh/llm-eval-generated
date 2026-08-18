#include <algorithm>
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

#define CUDA_CHECK(call) do {                                                     \
    const cudaError_t error_ = (call);                                            \
    if (error_ != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
                cudaGetErrorString(error_));                                      \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error_));                       \
    }                                                                              \
} while (0)

void fill(double* a, index_t n, double maxVal) {
    for (index_t i = 0; i < n; ++i)
        a[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
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

void spmvCpu(const double* val, const index_t* cols, const index_t* rows,
             const double* vec, index_t dim, double* out) {
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
    const index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < localRows) {
        double sum = 0.0;
        for (index_t j = rows[row]; j < rows[row + 1]; ++j) sum += val[j] * vec[cols[j]];
        out[row] = sum;
    }
}

bool verifyResults(const double* reference, const double* result, index_t n) {
    for (index_t i = 0; i < n; ++i) {
        const double error = std::abs(reference[i]) < 1e-10 ? std::abs(result[i]) :
            std::abs((result[i] - reference[i]) / reference[i]);
        if (error > MAX_RELATIVE_ERROR) {
            printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, reference[i], result[i]);
            return false;
        }
    }
    return true;
}

void usage(const char* prog) {
    printf("Usage: %s [-n rows] [-s sparsity] [-i iterations] [-m max] [-v] [-r] [-h]\n", prog);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    int validate = 0, printResults = 0, badArgs = 0;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (!strcmp(argv[i], "-n") && i + 1 < argc) numRows = static_cast<index_t>(atoi(argv[++i]));
            else if (!strcmp(argv[i], "-s") && i + 1 < argc) sparsity = static_cast<index_t>(atoi(argv[++i]));
            else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = static_cast<index_t>(atoi(argv[++i]));
            else if (!strcmp(argv[i], "-m") && i + 1 < argc) maxVal = atof(argv[++i]);
            else if (!strcmp(argv[i], "-v")) validate = 1;
            else if (!strcmp(argv[i], "-r")) printResults = 1;
            else if (!strcmp(argv[i], "-h")) { usage(argv[0]); MPI_Finalize(); return 0; }
            else badArgs = 1;
        }
        if (!numRows || !sparsity || static_cast<uint64_t>(numRows) * numRows > std::numeric_limits<index_t>::max()) badArgs = 1;
        if (badArgs) usage(argv[0]);
    }
    MPI_Bcast(&badArgs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (badArgs) { MPI_Finalize(); return 1; }
    MPI_Bcast(&numRows, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { if (!rank) fprintf(stderr, "No CUDA device found.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    const index_t nItems = static_cast<index_t>((static_cast<uint64_t>(numRows) * numRows) / sparsity);
    const index_t firstRow = static_cast<index_t>((static_cast<uint64_t>(numRows) * rank) / ranks);
    const index_t lastRow = static_cast<index_t>((static_cast<uint64_t>(numRows) * (rank + 1)) / ranks);
    const index_t localRows = lastRow - firstRow;
    std::vector<double> vec(numRows), localVal;
    std::vector<index_t> localCols, localRowOffsets(localRows + 1);
    std::vector<double> output(localRows), reference, globalOutput;
    std::vector<int> rowCounts(ranks), rowDisplacements(ranks), nnzCounts(ranks), nnzDisplacements(ranks);
    std::vector<double> values;
    std::vector<index_t> columns, rowOffsets;
    if (rank == 0) {
        values.resize(nItems); columns.resize(nItems); rowOffsets.resize(numRows + 1);
        fill(vec.data(), numRows, maxVal); fill(values.data(), nItems, maxVal);
        initRandomMatrix(columns.data(), rowOffsets.data(), nItems, numRows);
        for (int r = 0; r < ranks; ++r) {
            const index_t begin = static_cast<index_t>((static_cast<uint64_t>(numRows) * r) / ranks);
            const index_t end = static_cast<index_t>((static_cast<uint64_t>(numRows) * (r + 1)) / ranks);
            rowCounts[r] = static_cast<int>(end - begin); rowDisplacements[r] = static_cast<int>(begin);
            nnzCounts[r] = static_cast<int>(rowOffsets[end] - rowOffsets[begin]);
            nnzDisplacements[r] = static_cast<int>(rowOffsets[begin]);
        }
        if (validate) { reference.resize(numRows); spmvCpu(values.data(), columns.data(), rowOffsets.data(), vec.data(), numRows, reference.data()); }
        if (printResults || validate) globalOutput.resize(numRows);
        printf("Sparse Matrix-Vector Multiplication (MPI + OpenMP + CUDA)\nMatrix size: %u x %u, non-zeros: %u, ranks: %d\n", numRows, numRows, nItems, ranks);
    }
    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const int localNnz = [&] { int n = 0; MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_INT, &n, 1, MPI_INT, 0, MPI_COMM_WORLD); return n; }();
    int localBase = 0;
    MPI_Scatter(rank == 0 ? nnzDisplacements.data() : nullptr, 1, MPI_INT, &localBase, 1, MPI_INT, 0, MPI_COMM_WORLD);
    localVal.resize(localNnz); localCols.resize(localNnz);
    MPI_Scatterv(rank == 0 ? values.data() : nullptr, rank == 0 ? nnzCounts.data() : nullptr, rank == 0 ? nnzDisplacements.data() : nullptr, MPI_DOUBLE, localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? columns.data() : nullptr, rank == 0 ? nnzCounts.data() : nullptr, rank == 0 ? nnzDisplacements.data() : nullptr, MPI_UNSIGNED, localCols.data(), localNnz, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    std::vector<index_t> globalLocalRows(localRows + 1);
    MPI_Scatterv(rank == 0 ? rowOffsets.data() : nullptr, rank == 0 ? rowCounts.data() : nullptr, rank == 0 ? rowDisplacements.data() : nullptr, MPI_UNSIGNED, globalLocalRows.data(), static_cast<int>(localRows), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    // Send the final delimiter separately because Scatterv distributes row starts only.
    for (index_t i = 0; i < localRows; ++i) localRowOffsets[i] = globalLocalRows[i] - static_cast<index_t>(localBase);
    localRowOffsets[localRows] = static_cast<index_t>(localNnz);

    double *dVal, *dVec, *dOut; index_t *dCols, *dRows;
    CUDA_CHECK(cudaMalloc(&dVal, std::max<size_t>(1, localVal.size()) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCols, std::max<size_t>(1, localCols.size()) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dRows, localRowOffsets.size() * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dVec, vec.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dOut, std::max<size_t>(1, output.size()) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dVal, localVal.data(), localVal.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCols, localCols.data(), localCols.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRows, localRowOffsets.data(), localRowOffsets.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVec, vec.data(), vec.size() * sizeof(double), cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    MPI_Barrier(MPI_COMM_WORLD);
    cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    if (localRows) {
        for (index_t iter = 0; iter < iterations; ++iter)
            spmvKernel<<<(localRows + threads - 1) / threads, threads>>>(dVal, dCols, dRows, dVec, dOut, localRows);
    }
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float localMilliseconds = 0; CUDA_CHECK(cudaEventElapsedTime(&localMilliseconds, start, stop));
    CUDA_CHECK(cudaMemcpy(output.data(), dOut, output.size() * sizeof(double), cudaMemcpyDeviceToHost));
    double elapsed = localMilliseconds, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (printResults || validate)
        MPI_Gatherv(output.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    rank == 0 ? globalOutput.data() : nullptr,
                    rank == 0 ? rowCounts.data() : nullptr,
                    rank == 0 ? rowDisplacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int resultCode = 0;
    if (rank == 0) {
        const double gflops = maxElapsed > 0 ? 2.0 * nItems * iterations / (maxElapsed * 1.e6) : 0.0;
        printf("Computation time: %.3f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n", maxElapsed, iterations ? maxElapsed / iterations : 0.0, gflops);
        if (printResults) print_results(globalOutput, "OutputVector");
        if (validate) {
            const bool valid = verifyResults(reference.data(), globalOutput.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            resultCode = valid ? 0 : 1;
        }
    }
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dVal)); CUDA_CHECK(cudaFree(dCols)); CUDA_CHECK(cudaFree(dRows)); CUDA_CHECK(cudaFree(dVec)); CUDA_CHECK(cudaFree(dOut));
    MPI_Bcast(&resultCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return resultCode;
}
