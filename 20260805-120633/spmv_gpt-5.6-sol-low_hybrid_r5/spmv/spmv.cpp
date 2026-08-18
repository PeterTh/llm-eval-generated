#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t e_ = (call);                                                    \
    if (e_ != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
                cudaGetErrorString(e_));                                        \
        MPI_Abort(MPI_COMM_WORLD, 2);                                           \
    }                                                                           \
} while (0)

void fill(double* a, index_t n, double maxVal) {
    for (index_t i = 0; i < n; ++i)
        a[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

void initRandomMatrix(index_t* cols, index_t* rows, index_t n, index_t dim) {
    index_t assigned = 0;
    const double probability = static_cast<double>(n) /
                               (static_cast<double>(dim) * dim);
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rows[i] = assigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t left = static_cast<uint64_t>(dim) * dim -
                                  (static_cast<uint64_t>(i) * dim + j);
            if (left <= n - assigned) fillRemaining = true;
            const double r = static_cast<double>(rand()) / RAND_MAX;
            if ((assigned < n && r <= probability) || fillRemaining)
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
        for (index_t j = rows[i]; j < rows[i + 1]; ++j)
            sum += val[j] * vec[cols[j]];
        out[i] = sum;
    }
}

// A warp cooperatively evaluates one row.  Grid-striding rows keeps occupancy
// high for both short and irregular rows, while shuffle reduction avoids shared memory.
__global__ void spmvCsrWarp(const double* __restrict__ val,
                            const index_t* __restrict__ cols,
                            const index_t* __restrict__ rows,
                            const double* __restrict__ vec,
                            index_t numRows, double* __restrict__ out) {
    const unsigned lane = threadIdx.x & 31u;
    index_t row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const index_t stride = (gridDim.x * blockDim.x) >> 5;
    for (; row < numRows; row += stride) {
        double sum = 0.0;
        for (index_t j = rows[row] + lane; j < rows[row + 1]; j += 32)
            sum += val[j] * __ldg(vec + cols[j]);
        for (int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) out[row] = sum;
    }
}

bool verifyResults(const double* ref, const double* result, index_t n) {
    for (index_t i = 0; i < n; ++i) {
        const double err = std::abs(result[i] - ref[i]);
        if ((std::abs(ref[i]) < 1e-10 && err > MAX_RELATIVE_ERROR) ||
            (std::abs(ref[i]) >= 1e-10 && err / std::abs(ref[i]) > MAX_RELATIVE_ERROR)) {
            printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                   i, ref[i], result[i]);
            return false;
        }
    }
    return true;
}

void printUsage(const char* p) {
    printf("Usage: %s [-n rows] [-s sparsity] [-i iterations] [-m max] [-v] [-r] [-h]\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    bool argsOk = true, help = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numRows = static_cast<index_t>(strtoul(argv[++i], nullptr, 10));
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) sparsity = static_cast<index_t>(strtoul(argv[++i], nullptr, 10));
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = static_cast<index_t>(strtoul(argv[++i], nullptr, 10));
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) maxVal = strtod(argv[++i], nullptr);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else argsOk = false;
    }
    if (help || !argsOk) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argsOk ? 0 : 1;
    }
    const uint64_t totalSlots = static_cast<uint64_t>(numRows) * numRows;
    if (!numRows || !sparsity || !iterations || totalSlots / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Invalid or oversized benchmark parameters\n");
        MPI_Finalize(); return 1;
    }
    const index_t nItems = static_cast<index_t>(totalSlots / sparsity);

    // Bind ranks round-robin to GPUs local to each node.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { if (rank == 0) fprintf(stderr, "No CUDA devices found\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    std::vector<int> rowCounts(ranks), rowDispls(ranks), nnzCounts(ranks), nnzDispls(ranks);
    for (int r = 0; r < ranks; ++r) {
        const uint64_t first = static_cast<uint64_t>(r) * numRows / ranks;
        const uint64_t last = static_cast<uint64_t>(r + 1) * numRows / ranks;
        rowDispls[r] = static_cast<int>(first);
        rowCounts[r] = static_cast<int>(last - first);
    }
    const index_t localRows = static_cast<index_t>(rowCounts[rank]);
    std::vector<double> val, vec(numRows), fullOut, reference;
    std::vector<index_t> cols, rows;
    if (rank == 0) {
        val.resize(nItems); cols.resize(nItems); rows.resize(numRows + 1);
        srand(1); fill(vec.data(), numRows, maxVal); fill(val.data(), nItems, maxVal);
        initRandomMatrix(cols.data(), rows.data(), nItems, numRows);
        for (int r = 0; r < ranks; ++r) {
            nnzDispls[r] = static_cast<int>(rows[rowDispls[r]]);
            nnzCounts[r] = static_cast<int>(rows[rowDispls[r] + rowCounts[r]] - rows[rowDispls[r]]);
        }
        if (validate) { reference.resize(numRows); spmvCpu(val.data(), cols.data(), rows.data(), vec.data(), numRows, reference.data()); }
        printf("Sparse Matrix-Vector Multiplication (MPI + OpenMP + CUDA)\nMatrix size: %u x %u\nNon-zero elements: %u\nMPI ranks: %d, OpenMP threads/rank: %d\nIterations: %u\n",
               numRows, numRows, nItems, ranks, omp_get_max_threads(), iterations);
    }
    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzCounts.data(), ranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzDispls.data(), ranks, MPI_INT, 0, MPI_COMM_WORLD);

    const int localNnz = nnzCounts[rank];
    std::vector<double> localVal(localNnz), localOut(localRows);
    std::vector<index_t> localCols(localNnz), localRowsPtr(localRows + 1);
    MPI_Scatterv(rank == 0 ? val.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? cols.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_UINT32_T,
                 localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    std::vector<int> ptrCounts = rowCounts, ptrDispls(ranks);
    int totalPtrs = 0;
    for (int r = 0; r < ranks; ++r) { ++ptrCounts[r]; ptrDispls[r] = totalPtrs; totalPtrs += ptrCounts[r]; }
    std::vector<index_t> packedRows(rank == 0 ? totalPtrs : 0);
    if (rank == 0) {
#pragma omp parallel for schedule(static)
        for (int r = 0; r < ranks; ++r)
            for (int i = 0; i <= rowCounts[r]; ++i)
                packedRows[ptrDispls[r] + i] = rows[rowDispls[r] + i] - static_cast<index_t>(nnzDispls[r]);
    }
    MPI_Scatterv(rank == 0 ? packedRows.data() : nullptr, ptrCounts.data(), ptrDispls.data(), MPI_UINT32_T,
                 localRowsPtr.data(), static_cast<int>(localRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    double *dVal, *dVec, *dOut; index_t *dCols, *dRows;
    CUDA_CHECK(cudaMalloc(&dVal, std::max<size_t>(1, localNnz) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCols, std::max<size_t>(1, localNnz) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dRows, (localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dVec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dOut, std::max<index_t>(1, localRows) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dVal, localVal.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCols, localCols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRows, localRowsPtr.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVec, vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int blocks = std::max(1, std::min(65535, (static_cast<int>(localRows) * 32 + threads - 1) / threads));
    spmvCsrWarp<<<blocks, threads>>>(dVal, dCols, dRows, dVec, localRows, dOut); // warm-up/JIT
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t i = 0; i < iterations; ++i)
        spmvCsrWarp<<<blocks, threads>>>(dVal, dCols, dRows, dVec, localRows, dOut);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const bool needOutput = validate || printResults;
    if (needOutput) {
        CUDA_CHECK(cudaMemcpy(localOut.data(), dOut, localRows * sizeof(double), cudaMemcpyDeviceToHost));
        if (rank == 0) fullOut.resize(numRows);
        MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    rank == 0 ? fullOut.data() : nullptr, rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    int status = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n",
               maxElapsed * 1000.0, maxElapsed * 1000.0 / iterations,
               2.0 * nItems * iterations / maxElapsed / 1e9);
        if (printResults) print_results(fullOut, "OutputVector");
        if (validate) { const bool ok = verifyResults(reference.data(), fullOut.data(), numRows); printf("Validation: %s\n", ok ? "PASSED" : "FAILED"); status = ok ? 0 : 1; }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(dVal)); CUDA_CHECK(cudaFree(dCols)); CUDA_CHECK(cudaFree(dRows));
    CUDA_CHECK(cudaFree(dVec)); CUDA_CHECK(cudaFree(dOut));
    MPI_Finalize();
    return status;
}
