#include <cuda_runtime.h>
#include <mpi.h>

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

[[noreturn]] static void cudaFail(cudaError_t error, const char* expression,
                                  const char* file, int line) {
    std::fprintf(stderr, "CUDA error on rank: %s (%s) at %s:%d\n", expression,
                 cudaGetErrorString(error), file, line);
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}
#define CUDA_CHECK(expr) do { const cudaError_t e = (expr); if (e != cudaSuccess) cudaFail(e, #expr, __FILE__, __LINE__); } while (0)

void fill(double* values, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i)
        values[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;
    const double probability = static_cast<double>(n) / (static_cast<double>(dim) * dim);
    srand(8675309);
    bool fillRemaining = false;
    for (index_t row = 0; row < dim; ++row) {
        rowDelimiters[row] = nnzAssigned;
        for (index_t col = 0; col < dim; ++col) {
            const index_t entriesLeft = dim * dim - (row * dim + col);
            if (entriesLeft <= n - nnzAssigned) fillRemaining = true;
            const double randomValue = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randomValue <= probability) || fillRemaining)
                cols[nnzAssigned++] = col;
        }
    }
    rowDelimiters[dim] = n;
}

void spmvCpu(const double* val, const index_t* cols, const index_t* rows,
             const double* vec, index_t dim, double* out) {
#pragma omp parallel for schedule(static)
    for (index_t row = 0; row < dim; ++row) {
        double sum = 0.0;
        for (index_t j = rows[row]; j < rows[row + 1]; ++j)
            sum += val[j] * vec[cols[j]];
        out[row] = sum;
    }
}

__global__ void spmvCsrKernel(const double* __restrict__ values,
                              const index_t* __restrict__ columns,
                              const index_t* __restrict__ rowDelimiters,
                              const double* __restrict__ vector,
                              index_t firstRow, index_t localRows,
                              index_t valueOffset, double* __restrict__ output) {
    for (index_t localRow = blockIdx.x * blockDim.x + threadIdx.x;
         localRow < localRows;
         localRow += blockDim.x * gridDim.x) {
        const index_t row = firstRow + localRow;
        double sum = 0.0;
        for (index_t j = rowDelimiters[row] - valueOffset;
             j < rowDelimiters[row + 1] - valueOffset; ++j)
            sum += values[j] * vector[columns[j]];
        output[localRow] = sum;
    }
}

bool verifyResults(const double* reference, const double* result, index_t size) {
    for (index_t i = 0; i < size; ++i) {
        if (std::abs(reference[i]) < 1e-10) {
            if (std::abs(result[i]) > MAX_RELATIVE_ERROR) return false;
        } else if (std::abs((result[i] - reference[i]) / reference[i]) > MAX_RELATIVE_ERROR) {
            return false;
        }
    }
    return true;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("  -n <num>     Number of rows/columns (default: 1024)\n"
                "  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n"
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
    if (!numRows || !sparsity || !iterations) {
        if (!rank) std::fprintf(stderr, "-n, -s, and -i must be positive\n");
        MPI_Finalize(); return 1;
    }
    const index_t nItems = static_cast<index_t>((static_cast<uint64_t>(numRows) * numRows) / sparsity);
    if (nItems > static_cast<index_t>(std::numeric_limits<int>::max())) {
        if (!rank) std::fprintf(stderr, "Matrix is too large for this MPI implementation\n");
        MPI_Finalize(); return 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    const index_t firstRow = static_cast<index_t>((static_cast<uint64_t>(rank) * numRows) / ranks);
    const index_t lastRow = static_cast<index_t>((static_cast<uint64_t>(rank + 1) * numRows) / ranks);
    const index_t localRows = lastRow - firstRow;
    std::vector<index_t> rowDelimiters(numRows + 1);
    std::vector<double> values;
    std::vector<index_t> columns;
    std::vector<double> vector(numRows);
    std::vector<double> reference, output;

    if (!rank) {
        values.resize(nItems); columns.resize(nItems); output.resize(numRows);
        if (!rank) {
            std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\nMatrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\nNon-zero elements: %u (%.2f%% sparse)\nIterations: %u\nMax value: %.2f\nValidation: %s\nInitializing data structures...\n",
                        numRows, numRows, sparsity, nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (static_cast<double>(numRows) * numRows)), iterations, maxVal, validate ? "enabled" : "disabled");
        }
        fill(vector.data(), numRows, maxVal);
        fill(values.data(), nItems, maxVal);
        initRandomMatrix(columns.data(), rowDelimiters.data(), nItems, numRows);
        if (validate) { reference.resize(numRows); spmvCpu(values.data(), columns.data(), rowDelimiters.data(), vector.data(), numRows, reference.data()); }
    }
    MPI_Bcast(rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const index_t valueOffset = rowDelimiters[firstRow];
    const int localNnz = static_cast<int>(rowDelimiters[lastRow] - valueOffset);
    std::vector<double> localValues(localNnz);
    std::vector<index_t> localColumns(localNnz);
    std::vector<int> nnzCounts(ranks), nnzDisplacements(ranks), rowCounts(ranks), rowDisplacements(ranks);
    // Every rank independently builds the identical communication plan.  This
    // host-side work is intentionally threaded; CUDA owns the numerical kernel.
#pragma omp parallel for schedule(static)
    for (int p = 0; p < ranks; ++p) {
        const index_t begin = static_cast<index_t>((static_cast<uint64_t>(p) * numRows) / ranks);
        const index_t end = static_cast<index_t>((static_cast<uint64_t>(p + 1) * numRows) / ranks);
        nnzCounts[p] = static_cast<int>(rowDelimiters[end] - rowDelimiters[begin]);
        nnzDisplacements[p] = static_cast<int>(rowDelimiters[begin]);
        rowCounts[p] = static_cast<int>(end - begin);
        rowDisplacements[p] = static_cast<int>(begin);
    }
    MPI_Scatterv(rank ? nullptr : values.data(), nnzCounts.data(), nnzDisplacements.data(), MPI_DOUBLE, localValues.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank ? nullptr : columns.data(), nnzCounts.data(), nnzDisplacements.data(), MPI_UINT32_T, localColumns.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    double *dValues = nullptr, *dVector = nullptr, *dOutput = nullptr;
    index_t *dColumns = nullptr, *dRows = nullptr;
    // cudaMalloc(0) is not portable; ranks with no assigned work still join all MPI calls.
    const size_t valueAllocation = std::max<size_t>(1, localNnz);
    const size_t outputAllocation = std::max<size_t>(1, localRows);
    CUDA_CHECK(cudaMalloc(&dValues, valueAllocation * sizeof(*dValues)));
    CUDA_CHECK(cudaMalloc(&dColumns, valueAllocation * sizeof(*dColumns)));
    CUDA_CHECK(cudaMalloc(&dRows, static_cast<size_t>(numRows + 1) * sizeof(*dRows)));
    CUDA_CHECK(cudaMalloc(&dVector, static_cast<size_t>(numRows) * sizeof(*dVector)));
    CUDA_CHECK(cudaMalloc(&dOutput, outputAllocation * sizeof(*dOutput)));
    CUDA_CHECK(cudaMemcpy(dValues, localValues.data(), static_cast<size_t>(localNnz) * sizeof(*dValues), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dColumns, localColumns.data(), static_cast<size_t>(localNnz) * sizeof(*dColumns), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRows, rowDelimiters.data(), static_cast<size_t>(numRows + 1) * sizeof(*dRows), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVector, vector.data(), static_cast<size_t>(numRows) * sizeof(*dVector), cudaMemcpyHostToDevice));
    std::vector<double> localOutput(localRows);

    constexpr int threads = 256;
    const int blocks = std::max(1, std::min<int>((localRows + threads - 1) / threads, 65535));
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (index_t iteration = 0; iteration < iterations; ++iteration)
        spmvCsrKernel<<<blocks, threads>>>(dValues, dColumns, dRows, dVector, firstRow, localRows, valueOffset, dOutput);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(localOutput.data(), dOutput, static_cast<size_t>(localRows) * sizeof(*dOutput), cudaMemcpyDeviceToHost));
    const auto end = std::chrono::steady_clock::now();
    const double localMilliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    double elapsedMilliseconds = 0.0;
    MPI_Reduce(&localMilliseconds, &elapsedMilliseconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localOutput.data(), static_cast<int>(localRows), MPI_DOUBLE, rank ? nullptr : output.data(), rowCounts.data(), rowDisplacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(dOutput)); CUDA_CHECK(cudaFree(dVector)); CUDA_CHECK(cudaFree(dRows));
    CUDA_CHECK(cudaFree(dColumns)); CUDA_CHECK(cudaFree(dValues));
    if (!rank) {
        std::printf("Computation time: %.3f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n",
                    elapsedMilliseconds, elapsedMilliseconds / iterations,
                    (2.0 * nItems * iterations) / (elapsedMilliseconds / 1000.0) / 1e9);
        if (printResults) print_results(output, "OutputVector");
        if (validate) std::printf("Validation: %s\n", verifyResults(reference.data(), output.data(), numRows) ? "PASSED" : "FAILED");
    }
    const bool passed = !validate || (!rank && verifyResults(reference.data(), output.data(), numRows));
    int failure = passed ? 0 : 1;
    MPI_Bcast(&failure, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return failure;
}
