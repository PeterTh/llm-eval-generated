#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call) do { \
    cudaError_t error__ = (call); \
    if (error__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error__)); \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error__)); \
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
            const index_t left = (dim * dim) - ((i * dim) + j);
            const index_t needed = n - nnzAssigned;
            if (left <= needed) fillRemaining = true;
            const double randomValue = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randomValue <= prob) || fillRemaining)
                cols[nnzAssigned++] = j;
        }
    }
    rowDelimiters[dim] = n;
}

// A warp owns a row. This avoids the poor utilization of one thread per row
// for the benchmark's moderately long sparse rows.
__global__ void spmvKernel(const double* __restrict__ val,
                           const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec,
                           double* __restrict__ out,
                           index_t firstRow, index_t rowCount) {
    const unsigned lane = threadIdx.x & 31u;
    const index_t localRow = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (localRow >= rowCount) return;
    const index_t row = firstRow + localRow;
    const index_t begin = rowDelimiters[row];
    const index_t end = rowDelimiters[row + 1];
    double sum = 0.0;
    for (index_t j = begin + lane; j < end; j += 32)
        sum += val[j] * vec[cols[j]];
    #pragma unroll
    for (unsigned offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) out[row] = sum;
}

void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, index_t dim, double* out) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(dim); ++i) {
        double sum = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j)
            sum += val[j] * vec[cols[j]];
        out[i] = sum;
    }
}

bool verifyResults(const double* reference, const double* result, index_t size) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (int i = 0; i < static_cast<int>(size); ++i) {
        const double ref = reference[i], res = result[i];
        const bool okay = std::abs(ref) < 1e-10
            ? std::abs(res) <= MAX_RELATIVE_ERROR
            : std::abs((res - ref) / ref) <= MAX_RELATIVE_ERROR;
        if (!okay) valid = 0;
    }
    if (!valid) {
        for (index_t i = 0; i < size; ++i) {
            const double ref = reference[i], res = result[i];
            const double error = std::abs(ref) < 1e-10 ? std::abs(res) : std::abs((res - ref) / ref);
            if (error > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                break;
            }
        }
    }
    return valid != 0;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Number of rows/columns (default: 1024)\n");
    printf("  -s <num>     Sparsity: 1 out of N entries non-zero (default: 10)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -m <val>     Maximum element value (default: 1.0)\n");
    printf("  -v           Enable validation\n  -r           Print results\n  -h           Show this help\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);

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
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }

    const index_t nItems = (numRows * numRows) / sparsity;
    std::vector<double> val(nItems), vec(numRows), out(numRows, 0.0);
    std::vector<index_t> cols(nItems), rowDelimiters(numRows + 1);
    if (rank == 0) {
        fill(vec.data(), numRows, maxVal);
        fill(val.data(), nItems, maxVal);
        initRandomMatrix(cols.data(), rowDelimiters.data(), nItems, numRows);
    }
    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(val.data(), static_cast<int>(nItems), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(cols.data(), static_cast<int>(nItems), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { fprintf(stderr, "No CUDA device available on MPI rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    const index_t firstRow = (static_cast<uint64_t>(numRows) * rank) / world;
    const index_t lastRow = (static_cast<uint64_t>(numRows) * (rank + 1)) / world;
    const index_t rowCount = lastRow - firstRow;
    double *dVal = nullptr, *dVec = nullptr, *dOut = nullptr;
    index_t *dCols = nullptr, *dRows = nullptr;
    CUDA_CHECK(cudaMalloc(&dVal, nItems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCols, nItems * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dRows, (numRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dVec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dOut, numRows * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dVal, val.data(), nItems * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCols, cols.data(), nItems * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRows, rowDelimiters.data(), (numRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVec, vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> reference;
    if (validate) { reference.resize(numRows); spmvCpu(val.data(), cols.data(), rowDelimiters.data(), vec.data(), numRows, reference.data()); }
    MPI_Barrier(MPI_COMM_WORLD);
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(begin));
    if (iterations > 0 && rowCount > 0) {
        const int threads = 256;
        const int blocks = (static_cast<int>(rowCount) * 32 + threads - 1) / threads;
        for (index_t iter = 0; iter < iterations; ++iter)
            spmvKernel<<<blocks, threads>>>(dVal, dCols, dRows, dVec, dOut, firstRow, rowCount);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float localMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&localMs, begin, end));
    if (iterations > 0 && rowCount > 0)
        CUDA_CHECK(cudaMemcpy(out.data() + firstRow, dOut + firstRow, rowCount * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> globalOut;
    if (rank == 0) globalOut.resize(numRows);
    // Collect only the final row blocks; this avoids an all-rank output buffer.
    if (rank == 0) {
        for (int r = 1; r < world; ++r) { const index_t beginRow = (static_cast<uint64_t>(numRows) * r) / world; const index_t count = (static_cast<uint64_t>(numRows) * (r + 1)) / world - beginRow; MPI_Recv(globalOut.data() + beginRow, static_cast<int>(count), MPI_DOUBLE, r, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE); }
        std::copy(out.begin() + firstRow, out.begin() + firstRow + rowCount,
                  globalOut.begin() + firstRow);
    } else MPI_Send(out.data() + firstRow, static_cast<int>(rowCount), MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);

    float maxMs = 0.0f; MPI_Reduce(&localMs, &maxMs, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\nMatrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\nNon-zero elements: %u\nIterations: %u\nMax value: %.2f\nValidation: %s\n", numRows, numRows, sparsity, nItems, iterations, maxVal, validate ? "enabled" : "disabled");
        printf("Computation time: %.3f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n", maxMs, iterations ? maxMs / iterations : 0.0, iterations && maxMs > 0 ? (2.0 * nItems * iterations) / (maxMs / 1000.0) / 1e9 : 0.0);
        if (printResults) print_results(globalOut, "OutputVector");
        if (validate) { printf("Validating result...\n"); printf("Validation: %s\n", verifyResults(reference.data(), globalOut.data(), numRows) ? "PASSED" : "FAILED"); }
    }
    CUDA_CHECK(cudaFree(dVal)); CUDA_CHECK(cudaFree(dCols)); CUDA_CHECK(cudaFree(dRows)); CUDA_CHECK(cudaFree(dVec)); CUDA_CHECK(cudaFree(dOut));
    CUDA_CHECK(cudaEventDestroy(begin)); CUDA_CHECK(cudaEventDestroy(end));
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return 0;
}
