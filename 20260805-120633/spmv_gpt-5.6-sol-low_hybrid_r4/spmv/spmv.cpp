#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

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

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t error_ = (call);                                                \
    if (error_ != cudaSuccess) {                                                \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                     cudaGetErrorString(error_));                               \
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
            const index_t needed = n - assigned;
            if (left <= needed) fillRemaining = true;
            const double random = static_cast<double>(rand()) / RAND_MAX;
            if ((assigned < n && random <= probability) || fillRemaining)
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

// One warp processes one row.  This exposes parallelism both between rows and
// within long rows while keeping accesses to the CSR value arrays coalesced.
__global__ void spmvKernel(const double* __restrict__ val,
                           const index_t* __restrict__ cols,
                           const index_t* __restrict__ rows,
                           const double* __restrict__ vec,
                           index_t rowCount, double* __restrict__ out) {
    const unsigned lane = threadIdx.x & 31u;
    const index_t row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= rowCount) return;
    double sum = 0.0;
    for (index_t j = rows[row] + lane; j < rows[row + 1]; j += 32)
        sum += val[j] * __ldg(vec + cols[j]);
    for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) out[row] = sum;
}

bool verifyResults(const double* reference, const double* result, index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i], res = result[i];
        const double error = std::abs(ref) < 1e-10 ? std::abs(res)
                                                   : std::abs((res - ref) / ref);
        if (error > MAX_RELATIVE_ERROR) {
            std::printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                        i, ref, res);
            return false;
        }
    }
    return true;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("  -n <num>  matrix rows/columns (default 1024)\n"
                "  -s <num>  one out of N entries is nonzero (default 10)\n"
                "  -i <num>  iterations (default 10)\n"
                "  -m <val>  maximum element value (default 1.0)\n"
                "  -v        validate\n  -r        print results\n"
                "  -h        show help\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 3);

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false, help = false, badArgs = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) numRows = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) sparsity = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-m") && i + 1 < argc) maxVal = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else badArgs = true;
    }
    if (help || badArgs || !numRows || !sparsity || !iterations) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return badArgs || !numRows || !sparsity || !iterations;
    }
    const uint64_t nnz64 = static_cast<uint64_t>(numRows) * numRows / sparsity;
    if (nnz64 > std::numeric_limits<index_t>::max() || nnz64 > INT_MAX) {
        if (rank == 0) std::fprintf(stderr, "Matrix has too many nonzeros for 32-bit CSR/MPI counts\n");
        MPI_Finalize(); return 1;
    }
    const index_t nItems = static_cast<index_t>(nnz64);

    std::vector<int> rowCounts(ranks), rowStarts(ranks), rowBoundaryCounts(ranks);
    for (int p = 0; p < ranks; ++p) {
        const uint64_t begin = static_cast<uint64_t>(numRows) * p / ranks;
        const uint64_t end = static_cast<uint64_t>(numRows) * (p + 1) / ranks;
        rowStarts[p] = static_cast<int>(begin);
        rowCounts[p] = static_cast<int>(end - begin);
        rowBoundaryCounts[p] = rowCounts[p] + 1;
    }
    const index_t localRows = rowCounts[rank];
    std::vector<double> hVal, hVec(numRows), hOut;
    std::vector<index_t> hCols, hRows;
    if (rank == 0) {
        hVal.resize(nItems); hCols.resize(nItems); hRows.resize(numRows + 1); hOut.resize(numRows);
        fill(hVec.data(), numRows, maxVal); fill(hVal.data(), nItems, maxVal);
        initRandomMatrix(hCols.data(), hRows.data(), nItems, numRows);
        std::printf("Sparse Matrix-Vector Multiplication (MPI + OpenMP + CUDA)\n"
                    "Matrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\n"
                    "Non-zero elements: %u\nIterations: %u\nMPI ranks: %d, OpenMP threads/rank: %d\n",
                    numRows, numRows, sparsity, nItems, iterations, ranks, omp_get_max_threads());
    }
    MPI_Bcast(hVec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> nnzCounts(ranks), nnzStarts(ranks);
    if (rank == 0) {
#pragma omp parallel for schedule(static)
        for (int p = 0; p < ranks; ++p) {
            nnzStarts[p] = static_cast<int>(hRows[rowStarts[p]]);
            nnzCounts[p] = static_cast<int>(hRows[rowStarts[p] + rowCounts[p]] - hRows[rowStarts[p]]);
        }
    }
    MPI_Bcast(nnzCounts.data(), ranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzStarts.data(), ranks, MPI_INT, 0, MPI_COMM_WORLD);
    const int localNnz = nnzCounts[rank];
    std::vector<double> localVal(localNnz), localOut(localRows);
    std::vector<index_t> localCols(localNnz), localRow(localRows + 1);
    MPI_Scatterv(rank == 0 ? hVal.data() : nullptr, nnzCounts.data(), nnzStarts.data(), MPI_DOUBLE,
                 localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? hCols.data() : nullptr, nnzCounts.data(), nnzStarts.data(), MPI_UINT32_T,
                 localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? hRows.data() : nullptr, rowBoundaryCounts.data(), rowStarts.data(), MPI_UINT32_T,
                 localRow.data(), localRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i <= localRows; ++i) localRow[i] -= nnzStarts[rank];

    int deviceCount = 0; CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { if (rank == 0) std::fprintf(stderr, "No CUDA device found\n"); MPI_Abort(MPI_COMM_WORLD, 4); }
    int localRank = 0; MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank); CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    double *dVal = nullptr, *dVec = nullptr, *dOut = nullptr;
    index_t *dCols = nullptr, *dRows = nullptr;
    // CUDA does not require zero-sized allocations to succeed; allocate a
    // harmless sentinel element for empty partitions (more ranks than rows).
    CUDA_CHECK(cudaMalloc(&dVal, static_cast<size_t>(localNnz ? localNnz : 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCols, static_cast<size_t>(localNnz ? localNnz : 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dRows, (localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dVec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dOut, static_cast<size_t>(localRows ? localRows : 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dVal, localVal.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCols, localCols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRows, localRow.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVec, hVec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int blocks = static_cast<int>((static_cast<uint64_t>(localRows) * 32 + threads - 1) / threads);
    // Launching a zero-block grid is invalid. Empty ranks still participate in
    // all collectives and timing but have no GPU work.
    if (localRows)
        spmvKernel<<<blocks, threads>>>(dVal, dCols, dRows, dVec, localRows, dOut); // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (localRows)
        for (index_t iter = 0; iter < iterations; ++iter)
            spmvKernel<<<blocks, threads>>>(dVal, dCols, dRows, dVec, localRows, dOut);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (validate || printResults) {
        CUDA_CHECK(cudaMemcpy(localOut.data(), dOut, localRows * sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE, rank == 0 ? hOut.data() : nullptr,
                    rowCounts.data(), rowStarts.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    int exitCode = 0;
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n",
                    elapsed * 1e3, elapsed * 1e3 / iterations,
                    2.0 * nItems * iterations / elapsed / 1e9);
        if (printResults) print_results(hOut, "OutputVector");
        if (validate) {
            std::vector<double> reference(numRows);
            spmvCpu(hVal.data(), hCols.data(), hRows.data(), hVec.data(), numRows, reference.data());
            const bool valid = verifyResults(reference.data(), hOut.data(), numRows);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaFree(dVal); cudaFree(dCols); cudaFree(dRows); cudaFree(dVec); cudaFree(dOut);
    MPI_Finalize();
    return exitCode;
}
