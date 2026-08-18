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

#define CUDA_CHECK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

void fill(double* a, index_t n, double maxVal) {
    // Kept serial to preserve the benchmark's deterministic rand() stream.
    for (index_t i = 0; i < n; ++i) a[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

void initRandomMatrix(index_t* cols, index_t* rows, index_t n, index_t dim) {
    index_t assigned = 0;
    const double probability = static_cast<double>(n) / (static_cast<double>(dim) * dim);
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rows[i] = assigned;
        for (index_t j = 0; j < dim; ++j) {
            const index_t entriesLeft = dim * dim - (i * dim + j);
            if (entriesLeft <= n - assigned) fillRemaining = true;
            if ((assigned < n && static_cast<double>(rand()) / RAND_MAX <= probability) || fillRemaining)
                cols[assigned++] = j;
        }
    }
    rows[dim] = n;
}

void spmvCpu(const double* val, const index_t* cols, const index_t* rows, const double* vec, index_t dim, double* out) {
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
    const index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rowCount) return;
    double sum = 0.0;
    for (index_t j = rows[row]; j < rows[row + 1]; ++j) sum += val[j] * __ldg(vec + cols[j]);
    out[row] = sum;
}

bool verifyResults(const double* ref, const double* result, index_t n) {
    for (index_t i = 0; i < n; ++i) {
        if (std::abs(ref[i]) < 1e-10) {
            if (std::abs(result[i]) > MAX_RELATIVE_ERROR) return false;
        } else if (std::abs((result[i] - ref[i]) / ref[i]) > MAX_RELATIVE_ERROR) return false;
    }
    return true;
}

void printUsage(const char* p) {
    printf("Usage: %s [options]\n  -n <num>  matrix dimension (default: 1024)\n"
           "  -s <num>  one nonzero per N entries (default: 10)\n  -i <num>  iterations (default: 10)\n"
           "  -m <val>  maximum value (default: 1.0)\n  -v        validate\n  -r        print results\n  -h        help\n", p);
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
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!numRows || !sparsity || !iterations) { if (!rank) fprintf(stderr, "-n, -s and -i must be positive\n"); MPI_Finalize(); return 1; }
    const index_t nItems = (numRows * numRows) / sparsity;
    const index_t firstRow = static_cast<index_t>((static_cast<uint64_t>(rank) * numRows) / ranks);
    const index_t lastRow = static_cast<index_t>((static_cast<uint64_t>(rank + 1) * numRows) / ranks);
    const index_t localRows = lastRow - firstRow;

    std::vector<double> values, vec(numRows), output;
    std::vector<index_t> cols, rowPtrs;
    std::vector<int> valueCounts(ranks), valueDispls(ranks), rowCounts(ranks), rowDispls(ranks);
    std::vector<double> reference;
    if (!rank) {
        values.resize(nItems); cols.resize(nItems); rowPtrs.resize(numRows + 1); output.resize(numRows);
        fill(vec.data(), numRows, maxVal); fill(values.data(), nItems, maxVal);
        initRandomMatrix(cols.data(), rowPtrs.data(), nItems, numRows);
        if (validate) { reference.resize(numRows); spmvCpu(values.data(), cols.data(), rowPtrs.data(), vec.data(), numRows, reference.data()); }
        for (int r = 0; r < ranks; ++r) {
            const index_t begin = static_cast<index_t>((static_cast<uint64_t>(r) * numRows) / ranks);
            const index_t end = static_cast<index_t>((static_cast<uint64_t>(r + 1) * numRows) / ranks);
            valueDispls[r] = static_cast<int>(rowPtrs[begin]); valueCounts[r] = static_cast<int>(rowPtrs[end] - rowPtrs[begin]);
            rowDispls[r] = static_cast<int>(begin); rowCounts[r] = static_cast<int>(end - begin + 1);
        }
        printf("Sparse Matrix-Vector Multiplication (MPI + OpenMP + CUDA)\nMatrix size: %u x %u\nNon-zero elements: %u\nMPI ranks: %d, OpenMP threads/rank: %d\n", numRows, numRows, nItems, ranks, omp_get_max_threads());
    }
    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const int localNnz = rank == 0 ? valueCounts[0] : 0;
    int receivedNnz = localNnz;
    MPI_Scatter(valueCounts.data(), 1, MPI_INT, &receivedNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<double> localVal(receivedNnz);
    std::vector<index_t> localCols(receivedNnz), localPtrs(localRows + 1);
    MPI_Scatterv(rank ? nullptr : values.data(), valueCounts.data(), valueDispls.data(), MPI_DOUBLE, localVal.data(), receivedNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank ? nullptr : cols.data(), valueCounts.data(), valueDispls.data(), MPI_UNSIGNED, localCols.data(), receivedNnz, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank ? nullptr : rowPtrs.data(), rowCounts.data(), rowDispls.data(), MPI_UNSIGNED, localPtrs.data(), static_cast<int>(localRows + 1), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    const index_t base = localPtrs[0];
    #pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= localRows; ++i) localPtrs[i] -= base;

    int deviceCount = 0; CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { if (!rank) fprintf(stderr, "No CUDA devices available\n"); MPI_Abort(MPI_COMM_WORLD, 3); }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));
    double *dVal, *dVec, *dOut; index_t *dCols, *dPtrs;
    CUDA_CHECK(cudaMalloc(&dVal, std::max<size_t>(1, receivedNnz) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCols, std::max<size_t>(1, receivedNnz) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dPtrs, (localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dVec, numRows * sizeof(double))); CUDA_CHECK(cudaMalloc(&dOut, std::max<index_t>(1, localRows) * sizeof(double)));
    if (receivedNnz) { CUDA_CHECK(cudaMemcpy(dVal, localVal.data(), receivedNnz * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dCols, localCols.data(), receivedNnz * sizeof(index_t), cudaMemcpyHostToDevice)); }
    CUDA_CHECK(cudaMemcpy(dPtrs, localPtrs.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dVec, vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize()); MPI_Barrier(MPI_COMM_WORLD);
    cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); CUDA_CHECK(cudaEventRecord(start));
    constexpr int block = 256;
    for (index_t it = 0; it < iterations; ++it) {
        if (localRows) spmvKernel<<<(localRows + block - 1) / block, block>>>(dVal, dCols, dPtrs, dVec, dOut, localRows);
    }
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float localMs = 0; CUDA_CHECK(cudaEventElapsedTime(&localMs, start, stop));
    std::vector<double> localOut(localRows); if (localRows) CUDA_CHECK(cudaMemcpy(localOut.data(), dOut, localRows * sizeof(double), cudaMemcpyDeviceToHost));
    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE, rank ? nullptr : output.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double maxMs = 0; MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int result = 0;
    if (!rank) { printf("Computation time: %.3f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n", maxMs, maxMs / iterations, (2.0 * nItems * iterations) / (maxMs / 1000.0) / 1e9); if (printResults) print_results(output, "OutputVector"); if (validate) { result = verifyResults(reference.data(), output.data(), numRows) ? 0 : 1; printf("Validation: %s\n", result ? "FAILED" : "PASSED"); } }
    CUDA_CHECK(cudaFree(dVal)); CUDA_CHECK(cudaFree(dCols)); CUDA_CHECK(cudaFree(dPtrs)); CUDA_CHECK(cudaFree(dVec)); CUDA_CHECK(cudaFree(dOut)); CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Finalize(); return result;
}
