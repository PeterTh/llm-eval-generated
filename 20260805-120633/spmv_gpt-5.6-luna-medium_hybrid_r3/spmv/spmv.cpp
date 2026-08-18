#include <cuda_runtime.h>
#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
constexpr double MAX_RELATIVE_ERROR = 0.02;

void fill(double* a, index_t n, double maxVal) {
    // Keep the original RNG stream and therefore the original input values.
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
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i], res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                return false;
            }
        } else {
            const double relativeError = std::abs((res - ref) / ref);
            if (relativeError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i, ref, res, relativeError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("  -n <num>  Number of rows/columns (default: 1024)\n");
    printf("  -s <num>  Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    printf("  -i <num>  Number of iterations (default: 10)\n");
    printf("  -m <val>  Maximum matrix/vector value (default: 1.0)\n");
    printf("  -v        Enable validation\n  -r        Print results\n  -h        Show this help\n");
}

static void cudaCheck(cudaError_t error, const char* operation, int rank) {
    if (error != cudaSuccess) {
        fprintf(stderr, "MPI rank %d: CUDA %s failed: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__device__ __forceinline__ double warpSum(double value) {
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

// One warp owns one row. This avoids atomics and gives long rows parallel
// memory access while retaining coalesced access to the vector for each row.
__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec, double* __restrict__ out,
                           index_t rows) {
    constexpr int warpsPerBlock = 8;
    const int lane = threadIdx.x & 31;
    const index_t row = blockIdx.x * warpsPerBlock + threadIdx.x / 32;
    double sum = 0.0;
    if (row < rows) {
        const index_t begin = rowDelimiters[row], end = rowDelimiters[row + 1];
        for (index_t j = begin + lane; j < end; j += 32)
            sum += val[j] * vec[cols[j]];
    }
    sum = warpSum(sum);
    if (lane == 0 && row < rows) out[row] = sum;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false, showHelp = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numRows = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) sparsity = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) maxVal = atof(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) showHelp = true;
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (showHelp) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
    if (sparsity == 0 || numRows == 0) { if (rank == 0) fprintf(stderr, "n and sparsity must be non-zero\n"); MPI_Finalize(); return 1; }

    const index_t nItems = (numRows * numRows) / sparsity;
    const index_t rowStart = static_cast<index_t>((static_cast<uint64_t>(rank) * numRows) / world);
    const index_t rowEnd = static_cast<index_t>((static_cast<uint64_t>(rank + 1) * numRows) / world);
    const index_t localRows = rowEnd - rowStart;

    std::vector<double> globalVal, globalVec;
    std::vector<index_t> globalCols, globalDelimiters;
    if (rank == 0) {
        globalVal.resize(nItems); globalCols.resize(nItems); globalDelimiters.resize(numRows + 1);
        globalVec.resize(numRows);
        fill(globalVec.data(), numRows, maxVal);
        fill(globalVal.data(), nItems, maxVal);
        initRandomMatrix(globalCols.data(), globalDelimiters.data(), nItems, numRows);
    }
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> rowCounts(world), rowStarts(world), nnzCounts(world), nnzStarts(world);
    for (int r = 0; r < world; ++r) {
        const index_t first = static_cast<index_t>((static_cast<uint64_t>(r) * numRows) / world);
        const index_t last = static_cast<index_t>((static_cast<uint64_t>(r + 1) * numRows) / world);
        rowCounts[r] = static_cast<int>(last - first + 1);
        rowStarts[r] = static_cast<int>(first);
        nnzStarts[r] = static_cast<int>(rank == 0 ? globalDelimiters[first] : 0);
    }
    if (rank == 0) {
        for (int r = 0; r < world; ++r) {
            const index_t first = static_cast<index_t>((static_cast<uint64_t>(r) * numRows) / world);
            const index_t last = static_cast<index_t>((static_cast<uint64_t>(r + 1) * numRows) / world);
            nnzCounts[r] = static_cast<int>(globalDelimiters[last] - globalDelimiters[first]);
            nnzStarts[r] = static_cast<int>(globalDelimiters[first]);
        }
    }
    MPI_Bcast(nnzCounts.data(), world, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzStarts.data(), world, MPI_INT, 0, MPI_COMM_WORLD);
    const index_t localNnz = static_cast<index_t>(nnzCounts[rank]);
    std::vector<double> val(localNnz), vec(numRows), output(localRows);
    std::vector<index_t> cols(localNnz), delimiters(localRows + 1);
    MPI_Bcast(rank == 0 ? globalVec.data() : vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) vec = globalVec;
    MPI_Scatterv(rank == 0 ? globalVal.data() : nullptr, nnzCounts.data(), nnzStarts.data(), MPI_DOUBLE,
                 val.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalCols.data() : nullptr, nnzCounts.data(), nnzStarts.data(), MPI_UINT32_T,
                 cols.data(), static_cast<int>(localNnz), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalDelimiters.data() : nullptr, rowCounts.data(), rowStarts.data(), MPI_UINT32_T,
                 delimiters.data(), static_cast<int>(localRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    for (index_t& d : delimiters) d -= static_cast<index_t>(nnzStarts[rank]);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (MPI + OpenMP + CUDA)\nMatrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\nNon-zero elements: %u (%.2f%% sparse)\nIterations: %u\nMax value: %.2f\nValidation: %s\n",
               numRows, numRows, sparsity, nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (static_cast<double>(numRows) * numRows)), iterations, maxVal, validate ? "enabled" : "disabled");
    }
    std::vector<double> reference, gathered;
    if (validate && rank == 0) { reference.resize(numRows); spmvCpu(globalVal.data(), globalCols.data(), globalDelimiters.data(), globalVec.data(), numRows, reference.data()); }
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "get device count", rank);
    if (deviceCount <= 0) { fprintf(stderr, "MPI rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(rank % deviceCount), "select device", rank);
    double *dVal = nullptr, *dVec = nullptr, *dOut = nullptr;
    index_t *dCols = nullptr, *dDelimiters = nullptr;
    cudaCheck(cudaMalloc(&dVal, sizeof(double) * (localNnz ? localNnz : 1)), "allocate values", rank);
    cudaCheck(cudaMalloc(&dCols, sizeof(index_t) * (localNnz ? localNnz : 1)), "allocate columns", rank);
    cudaCheck(cudaMalloc(&dDelimiters, sizeof(index_t) * (localRows + 1)), "allocate delimiters", rank);
    cudaCheck(cudaMalloc(&dVec, sizeof(double) * numRows), "allocate vector", rank);
    cudaCheck(cudaMalloc(&dOut, sizeof(double) * (localRows ? localRows : 1)), "allocate output", rank);
    cudaCheck(cudaMemcpy(dVal, val.data(), sizeof(double) * localNnz, cudaMemcpyHostToDevice), "copy values", rank);
    cudaCheck(cudaMemcpy(dCols, cols.data(), sizeof(index_t) * localNnz, cudaMemcpyHostToDevice), "copy columns", rank);
    cudaCheck(cudaMemcpy(dDelimiters, delimiters.data(), sizeof(index_t) * (localRows + 1), cudaMemcpyHostToDevice), "copy delimiters", rank);
    cudaCheck(cudaMemcpy(dVec, vec.data(), sizeof(double) * numRows, cudaMemcpyHostToDevice), "copy vector", rank);
    cudaCheck(cudaDeviceSynchronize(), "initial synchronize", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (index_t iter = 0; iter < iterations; ++iter)
        spmvKernel<<<localRows ? (localRows + 7) / 8 : 1, 256>>>(dVal, dCols, dDelimiters, dVec, dOut, localRows);
    cudaCheck(cudaGetLastError(), "launch SpMV", rank);
    cudaCheck(cudaDeviceSynchronize(), "complete SpMV", rank);
    const auto end = std::chrono::high_resolution_clock::now();
    cudaCheck(cudaMemcpy(output.data(), dOut, sizeof(double) * localRows, cudaMemcpyDeviceToHost), "copy output", rank);
    const double localMs = std::chrono::duration<double, std::milli>(end - start).count();
    double elapsedMs = 0.0;
    MPI_Reduce(&localMs, &elapsedMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    bool valid = true;
    if (rank == 0) {
        gathered.resize(numRows);
    }
    std::vector<int> gatherCounts(world), gatherStarts(world);
    for (int r = 0; r < world; ++r) { gatherStarts[r] = static_cast<int>((static_cast<uint64_t>(r) * numRows) / world); gatherCounts[r] = static_cast<int>((static_cast<uint64_t>(r + 1) * numRows) / world) - gatherStarts[r]; }
    MPI_Gatherv(output.data(), static_cast<int>(localRows), MPI_DOUBLE, rank == 0 ? gathered.data() : nullptr, gatherCounts.data(), gatherStarts.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double seconds = elapsedMs / 1000.0;
        printf("Computation time: %.3f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n", elapsedMs, elapsedMs / iterations, seconds > 0.0 ? (2.0 * nItems * iterations) / seconds / 1e9 : 0.0);
        if (printResults) print_results(gathered, "OutputVector");
        if (validate) { printf("Validating result...\n"); valid = verifyResults(reference.data(), gathered.data(), numRows); printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); }
    }
    cudaFree(dVal); cudaFree(dCols); cudaFree(dDelimiters); cudaFree(dVec); cudaFree(dOut);
    int result = (rank == 0 && validate && !valid) ? 1 : 0;
    MPI_Finalize();
    return result;
}
