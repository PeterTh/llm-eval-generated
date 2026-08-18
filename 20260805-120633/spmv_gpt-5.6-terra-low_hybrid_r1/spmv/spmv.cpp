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

#define CUDA_CHECK(call) do {                                                    \
    const cudaError_t status_ = (call);                                          \
    if (status_ != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(status_));                                    \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status_));                     \
    }                                                                            \
} while (0)

void fill(double* a, index_t n, double maxVal) {
    for (index_t i = 0; i < n; ++i)
        a[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

void initRandomMatrix(index_t* cols, index_t* row, index_t n, index_t dim) {
    index_t assigned = 0;
    const double probability = static_cast<double>(n) / (static_cast<double>(dim) * dim);
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        row[i] = assigned;
        for (index_t j = 0; j < dim; ++j) {
            const index_t left = dim * dim - (i * dim + j);
            if (left <= n - assigned) fillRemaining = true;
            const double randomValue = static_cast<double>(rand()) / RAND_MAX;
            if ((assigned < n && randomValue <= probability) || fillRemaining)
                cols[assigned++] = j;
        }
    }
    row[dim] = n;
}

void spmvCpu(const double* val, const index_t* cols, const index_t* row,
             const double* vec, index_t dim, double* out) {
    #pragma omp parallel for schedule(static)
    for (index_t i = 0; i < dim; ++i) {
        double sum = 0.0;
        for (index_t j = row[i]; j < row[i + 1]; ++j) sum += val[j] * vec[cols[j]];
        out[i] = sum;
    }
}

__global__ void spmvCsrKernel(const double* __restrict__ val,
                              const index_t* __restrict__ cols,
                              const index_t* __restrict__ row,
                              const double* __restrict__ vec,
                              double* __restrict__ out, index_t rows) {
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        double sum = 0.0;
        for (index_t j = row[i]; j < row[i + 1]; ++j) sum += val[j] * vec[cols[j]];
        out[i] = sum;
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

void printUsage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  -n <num>  Matrix dimension (default: 1024)\n"
           "  -s <num>  One nonzero per N entries (default: 10)\n"
           "  -i <num>  Iterations (default: 10)\n"
           "  -m <val>  Maximum input value (default: 1.0)\n"
           "  -v        Enable validation\n  -r        Print results\n  -h        Show help\n");
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
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numRows = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) sparsity = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) maxVal = atof(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (!numRows || !sparsity || !iterations) {
        if (!rank) fprintf(stderr, "-n, -s, and -i must be non-zero\n");
        MPI_Finalize(); return 1;
    }
    const index_t nItems = (numRows * numRows) / sparsity;
    const index_t firstRow = static_cast<index_t>((static_cast<uint64_t>(rank) * numRows) / ranks);
    const index_t lastRow = static_cast<index_t>((static_cast<uint64_t>(rank + 1) * numRows) / ranks);
    const index_t localRows = lastRow - firstRow;

    // Rank zero owns initialization; CSR row blocks and the dense vector are distributed.
    std::vector<double> values, vec(numRows);
    std::vector<index_t> cols, globalRow;
    if (rank == 0) {
        values.resize(nItems); cols.resize(nItems); globalRow.resize(numRows + 1);
        fill(vec.data(), numRows, maxVal);
        fill(values.data(), nItems, maxVal);
        initRandomMatrix(cols.data(), globalRow.data(), nItems, numRows);
    }
    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<double> reference;
    if (validate && rank == 0) {
        reference.resize(numRows);
        spmvCpu(values.data(), cols.data(), globalRow.data(), vec.data(), numRows, reference.data());
    }
    std::vector<int> rowCounts(ranks), rowOffsets(ranks), nnzCounts(ranks), nnzOffsets(ranks);
    if (rank == 0) for (int r = 0; r < ranks; ++r) {
        const index_t begin = static_cast<index_t>((static_cast<uint64_t>(r) * numRows) / ranks);
        const index_t end = static_cast<index_t>((static_cast<uint64_t>(r + 1) * numRows) / ranks);
        rowCounts[r] = static_cast<int>(end - begin); rowOffsets[r] = static_cast<int>(begin);
        nnzOffsets[r] = static_cast<int>(globalRow[begin]);
        nnzCounts[r] = static_cast<int>(globalRow[end] - globalRow[begin]);
    }
    int firstNnzInt = 0, localNnzInt = 0;
    MPI_Scatter(nnzOffsets.data(), 1, MPI_INT, &firstNnzInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nnzCounts.data(), 1, MPI_INT, &localNnzInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const index_t firstNnz = static_cast<index_t>(firstNnzInt);
    const index_t localNnz = static_cast<index_t>(localNnzInt);
    std::vector<index_t> localRow(localRows + 1);
    MPI_Scatterv(globalRow.data(), rowCounts.data(), rowOffsets.data(), MPI_UNSIGNED,
                 localRow.data(), static_cast<int>(localRows), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    for (index_t i = 0; i < localRows; ++i) localRow[i] -= firstNnz;
    localRow[localRows] = localNnz;
    std::vector<double> localValues(localNnz);
    std::vector<index_t> localCols(localNnz);
    MPI_Scatterv(values.data(), nnzCounts.data(), nnzOffsets.data(), MPI_DOUBLE, localValues.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(cols.data(), nnzCounts.data(), nnzOffsets.data(), MPI_UNSIGNED, localCols.data(), static_cast<int>(localNnz), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    values.clear(); cols.clear(); globalRow.clear();

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { if (!rank) fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    double *dVal, *dVec, *dOut;
    index_t *dCols, *dRow;
    CUDA_CHECK(cudaMalloc(&dVal, std::max<size_t>(1, localNnz) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCols, std::max<size_t>(1, localNnz) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dRow, (localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dVec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dOut, std::max<size_t>(1, localRows) * sizeof(double)));
    if (localNnz) { CUDA_CHECK(cudaMemcpy(dVal, localValues.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dCols, localCols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice)); }
    CUDA_CHECK(cudaMemcpy(dRow, localRow.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVec, vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int blocks = (localRows + threads - 1) / threads;
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (index_t iter = 0; iter < iterations; ++iter)
        if (blocks) spmvCsrKernel<<<blocks, threads>>>(dVal, dCols, dRow, dVec, dOut, localRows);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float localMs = 0; CUDA_CHECK(cudaEventElapsedTime(&localMs, start, stop));
    double maxMs = 0, localMsDouble = localMs;
    MPI_Reduce(&localMsDouble, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<double> localOut(localRows);
    if (localRows) CUDA_CHECK(cudaMemcpy(localOut.data(), dOut, localRows * sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> output;
    if (rank == 0 && (validate || printResults)) output.resize(numRows);
    if (validate || printResults) MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE, output.data(), rowCounts.data(), rowOffsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int validationPassed = 1;
    if (!rank) {
        printf("Sparse Matrix-Vector Multiplication (MPI + OpenMP + CUDA)\nMatrix size: %u x %u\nMPI ranks: %d, CUDA devices per node: %d\n", numRows, numRows, ranks, deviceCount);
        printf("Computation time: %.3f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n", maxMs, maxMs / iterations, (2.0 * nItems * iterations) / (maxMs / 1000.0) / 1e9);
        if (printResults) print_results(output, "OutputVector");
        if (validate) {
            validationPassed = verifyResults(reference.data(), output.data(), numRows) ? 1 : 0;
            printf("Validation: %s\n", validationPassed ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&validationPassed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dVal)); CUDA_CHECK(cudaFree(dCols)); CUDA_CHECK(cudaFree(dRow)); CUDA_CHECK(cudaFree(dVec)); CUDA_CHECK(cudaFree(dOut));
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return validationPassed ? 0 : 1;
}
