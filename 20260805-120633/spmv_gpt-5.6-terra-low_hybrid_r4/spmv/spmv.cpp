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

static void cudaCheck(cudaError_t error, const char* where, int rank) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA error at %s: %s\n", rank, where,
                     cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

void fill(double* a, index_t n, double maxVal) {
    // rand() is deliberately consumed serially, preserving the original data set.
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

bool verifyResults(const double* reference, const double* result, index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i], res = result[i];
        if ((std::abs(ref) < 1e-10 && std::abs(res) > MAX_RELATIVE_ERROR) ||
            (std::abs(ref) >= 1e-10 && std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR)) {
            std::printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
            return false;
        }
    }
    return true;
}

void printUsage(const char* prog) {
    std::printf("Usage: %s [options]\n  -n <num>  Matrix dimension (default: 1024)\n"
                "  -s <num>  One nonzero per N entries (default: 10)\n"
                "  -i <num>  Iterations (default: 10)\n  -m <val>  Maximum value (default: 1.0)\n"
                "  -v        Enable validation\n  -r        Print results\n  -h        Show help\n", prog);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    index_t dim = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) dim = static_cast<index_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) sparsity = static_cast<index_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = static_cast<index_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-m") && i + 1 < argc) maxVal = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!dim || !sparsity || !iterations || static_cast<uint64_t>(dim) * dim / sparsity > std::numeric_limits<index_t>::max()) {
        if (!rank) std::fprintf(stderr, "Invalid dimensions, sparsity, or iteration count\n");
        MPI_Finalize(); return 1;
    }
    const index_t nnz = static_cast<uint64_t>(dim) * dim / sparsity;
    std::vector<double> val(rank == 0 ? nnz : 0), vec(dim), out(rank == 0 ? dim : 0), reference;
    std::vector<index_t> cols(rank == 0 ? nnz : 0), rows(rank == 0 ? dim + 1 : 0);
    if (!rank) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\nMatrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\nNon-zero elements: %u (%.2f%% sparse)\nIterations: %u\nMax value: %.2f\nValidation: %s\nInitializing data structures...\n",
                    dim, dim, sparsity, nnz, 100.0 * (1.0 - static_cast<double>(nnz) / (static_cast<double>(dim) * dim)), iterations, maxVal, validate ? "enabled" : "disabled");
        fill(vec.data(), dim, maxVal); fill(val.data(), nnz, maxVal); initRandomMatrix(cols.data(), rows.data(), nnz, dim);
        if (validate) { std::printf("Computing reference solution...\n"); reference.resize(dim); spmvCpu(val.data(), cols.data(), rows.data(), vec.data(), dim, reference.data()); }
    }
    MPI_Bcast(vec.data(), dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const index_t firstRow = static_cast<uint64_t>(rank) * dim / ranks;
    const index_t lastRow = static_cast<uint64_t>(rank + 1) * dim / ranks;
    const index_t localRows = lastRow - firstRow;
    std::vector<index_t> localRowsOffsets(localRows + 1);
    // CSR segments are distributed once; only the dense input vector is replicated.
    std::vector<int> nnzCounts, nnzDispls, rowCounts, rowDispls;
    if (!rank) {
        nnzCounts.resize(ranks); nnzDispls.resize(ranks); rowCounts.resize(ranks); rowDispls.resize(ranks);
        for (int r = 0; r < ranks; ++r) {
            const index_t begin = static_cast<uint64_t>(r) * dim / ranks;
            const index_t end = static_cast<uint64_t>(r + 1) * dim / ranks;
            nnzDispls[r] = rows[begin]; nnzCounts[r] = rows[end] - rows[begin];
            rowDispls[r] = begin; rowCounts[r] = end - begin + 1;
        }
    }
    MPI_Scatterv(rank ? nullptr : rows.data(), rank ? nullptr : rowCounts.data(), rank ? nullptr : rowDispls.data(), MPI_UINT32_T,
                 localRowsOffsets.data(), localRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t firstNnz = localRowsOffsets.front();
    const index_t localNnz = localRowsOffsets.back() - firstNnz;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i <= static_cast<int64_t>(localRows); ++i) localRowsOffsets[i] -= firstNnz;
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    MPI_Scatterv(rank ? nullptr : val.data(), rank ? nullptr : nnzCounts.data(), rank ? nullptr : nnzDispls.data(), MPI_DOUBLE,
                 localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank ? nullptr : cols.data(), rank ? nullptr : nnzCounts.data(), rank ? nullptr : nnzDispls.data(), MPI_UINT32_T,
                 localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (!deviceCount) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    cudaCheck(cudaSetDevice(rank % deviceCount), "cudaSetDevice", rank);
    double *dVal, *dVec, *dOut; index_t *dCols, *dRows;
    cudaCheck(cudaMalloc(&dVal, static_cast<size_t>(localNnz) * sizeof(double)), "cudaMalloc val", rank);
    cudaCheck(cudaMalloc(&dCols, static_cast<size_t>(localNnz) * sizeof(index_t)), "cudaMalloc cols", rank);
    cudaCheck(cudaMalloc(&dRows, (localRows + 1ULL) * sizeof(index_t)), "cudaMalloc rows", rank);
    cudaCheck(cudaMalloc(&dVec, static_cast<size_t>(dim) * sizeof(double)), "cudaMalloc vec", rank);
    cudaCheck(cudaMalloc(&dOut, static_cast<size_t>(localRows) * sizeof(double)), "cudaMalloc out", rank);
    cudaCheck(cudaMemcpy(dVal, localVal.data(), static_cast<size_t>(localNnz) * sizeof(double), cudaMemcpyHostToDevice), "copy val", rank);
    cudaCheck(cudaMemcpy(dCols, localCols.data(), static_cast<size_t>(localNnz) * sizeof(index_t), cudaMemcpyHostToDevice), "copy cols", rank);
    cudaCheck(cudaMemcpy(dRows, localRowsOffsets.data(), (localRows + 1ULL) * sizeof(index_t), cudaMemcpyHostToDevice), "copy rows", rank);
    cudaCheck(cudaMemcpy(dVec, vec.data(), static_cast<size_t>(dim) * sizeof(double), cudaMemcpyHostToDevice), "copy vec", rank);
    std::vector<double> localOut(localRows);
    if (!rank) std::printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    constexpr int threads = 256;
    for (index_t it = 0; it < iterations; ++it) spmvKernel<<<(localRows + threads - 1) / threads, threads>>>(dVal, dCols, dRows, dVec, dOut, localRows);
    cudaCheck(cudaGetLastError(), "kernel launch", rank);
    cudaCheck(cudaMemcpy(localOut.data(), dOut, static_cast<size_t>(localRows) * sizeof(double), cudaMemcpyDeviceToHost), "copy output", rank);
    const double localSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double seconds = 0; MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<int> recvCounts, displs;
    if (!rank) { recvCounts.resize(ranks); displs.resize(ranks); for (int r = 0; r < ranks; ++r) { displs[r] = static_cast<uint64_t>(r) * dim / ranks; recvCounts[r] = static_cast<uint64_t>(r + 1) * dim / ranks - displs[r]; } }
    MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE, rank ? nullptr : out.data(), rank ? nullptr : recvCounts.data(), rank ? nullptr : displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    cudaFree(dOut); cudaFree(dVec); cudaFree(dRows); cudaFree(dCols); cudaFree(dVal);
    int result = 0;
    if (!rank) {
        const double ms = seconds * 1000.0;
        std::printf("Computation time: %.3f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n", ms, ms / iterations, (2.0 * nnz * iterations) / seconds / 1e9);
        if (printResults) print_results(out, "OutputVector");
        if (validate) { std::printf("Validating result...\n"); result = verifyResults(reference.data(), out.data(), dim) ? 0 : 1; std::printf("Validation: %s\n", result ? "FAILED" : "PASSED"); }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
