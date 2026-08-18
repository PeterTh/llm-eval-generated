#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
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
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
                cudaGetErrorString(error_));                                    \
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
    for (long long i = 0; i < static_cast<long long>(dim); ++i) {
        double sum = 0.0;
        for (index_t j = rows[i]; j < rows[i + 1]; ++j)
            sum += val[j] * vec[cols[j]];
        out[i] = sum;
    }
}

// One warp owns a row.  Lanes cooperatively process long rows and use a
// shuffle reduction, avoiding a block-wide synchronization for each row.
__global__ void spmvCsrWarp(const double* __restrict__ val,
                            const index_t* __restrict__ cols,
                            const index_t* __restrict__ rows,
                            const double* __restrict__ vec,
                            index_t nrows, double* __restrict__ out) {
    constexpr unsigned FULL = 0xffffffffu;
    const unsigned lane = threadIdx.x & 31u;
    const index_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const index_t stride = (gridDim.x * blockDim.x) >> 5;
    for (index_t row = warp; row < nrows; row += stride) {
        double sum = 0.0;
        for (index_t j = rows[row] + lane; j < rows[row + 1]; j += 32)
            sum += val[j] * __ldg(vec + cols[j]);
        for (int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(FULL, sum, offset);
        if (lane == 0) out[row] = sum;
    }
}

bool verifyResults(const double* ref, const double* result, index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double error = std::abs(result[i] - ref[i]);
        if ((std::abs(ref[i]) < 1e-10 && error > MAX_RELATIVE_ERROR) ||
            (std::abs(ref[i]) >= 1e-10 && error / std::abs(ref[i]) > MAX_RELATIVE_ERROR)) {
            printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                   i, ref[i], result[i]);
            return false;
        }
    }
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("  -n <num>  Matrix rows/columns (default: 1024)\n"
           "  -s <num>  One out of N entries is non-zero (default: 10)\n"
           "  -i <num>  Iterations (default: 10)\n"
           "  -m <val>  Maximum element value (default: 1.0)\n"
           "  -v        Validate result\n  -r        Print result\n"
           "  -h        Show help\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    bool bad = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numRows = strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) sparsity = strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) maxVal = strtod(argv[++i], nullptr);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else bad = true;
    }
    if (help || bad || !numRows || !sparsity || !iterations) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return bad || !numRows || !sparsity || !iterations;
    }

    const uint64_t totalNnz64 = static_cast<uint64_t>(numRows) * numRows / sparsity;
    if (totalNnz64 > std::numeric_limits<index_t>::max() ||
        totalNnz64 > std::numeric_limits<int>::max()) {
        if (rank == 0) fprintf(stderr, "Problem has too many nonzeros for 32-bit CSR/MPI counts\n");
        MPI_Finalize(); return 1;
    }
    const index_t nItems = static_cast<index_t>(totalNnz64);

    // Contiguous, near-equal row ownership gives ordered output and cheap Gatherv.
    std::vector<int> rowCounts(ranks), rowDispls(ranks);
    for (int r = 0; r < ranks; ++r) {
        const uint64_t begin = static_cast<uint64_t>(numRows) * r / ranks;
        const uint64_t end = static_cast<uint64_t>(numRows) * (r + 1) / ranks;
        rowDispls[r] = static_cast<int>(begin);
        rowCounts[r] = static_cast<int>(end - begin);
    }
    const index_t localRows = rowCounts[rank];

    std::vector<double> values, vec(numRows), output;
    std::vector<index_t> cols, rows;
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (MPI + OpenMP + CUDA)\n"
               "Matrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\n"
               "Non-zero elements: %u\nIterations: %u\nMPI ranks: %d, OpenMP threads/rank: %d\n",
               numRows, numRows, sparsity, nItems, iterations, ranks, omp_get_max_threads());
        values.resize(nItems); cols.resize(nItems); rows.resize(numRows + 1); output.resize(numRows);
        fill(vec.data(), numRows, maxVal);
        fill(values.data(), nItems, maxVal);
        initRandomMatrix(cols.data(), rows.data(), nItems, numRows);
    }
    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> nnzCounts(ranks), nnzDispls(ranks);
    if (rank == 0) {
        for (int r = 0; r < ranks; ++r) {
            nnzDispls[r] = static_cast<int>(rows[rowDispls[r]]);
            nnzCounts[r] = static_cast<int>(rows[rowDispls[r] + rowCounts[r]] - rows[rowDispls[r]]);
        }
    }
    MPI_Bcast(nnzCounts.data(), ranks, MPI_INT, 0, MPI_COMM_WORLD);
    const int localNnz = nnzCounts[rank];
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz), localRow(localRows + 1);
    MPI_Scatterv(rank == 0 ? values.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? cols.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_UINT32_T,
                 localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        // Send each rank its row pointers; overlaps at partition boundaries are intentional.
        for (int r = 1; r < ranks; ++r)
            MPI_Send(rows.data() + rowDispls[r], rowCounts[r] + 1, MPI_UINT32_T, r, 1, MPI_COMM_WORLD);
        std::copy_n(rows.data(), localRows + 1, localRow.data());
    } else {
        MPI_Recv(localRow.data(), localRows + 1, MPI_UINT32_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    const index_t rowBase = localRow[0];
#pragma omp parallel for schedule(static)
    for (long long i = 0; i <= static_cast<long long>(localRows); ++i) localRow[i] -= rowBase;

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { if (rank == 0) fprintf(stderr, "No CUDA devices available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    int localRank = 0;
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    double *dVal = nullptr, *dVec = nullptr, *dOut = nullptr;
    index_t *dCols = nullptr, *dRows = nullptr;
    CUDA_CHECK(cudaMalloc(&dVal, std::max<size_t>(1, localNnz) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCols, std::max<size_t>(1, localNnz) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dRows, (localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dVec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dOut, std::max<size_t>(1, localRows) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dVal, localVal.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCols, localCols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRows, localRow.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVec, vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int blocks = std::max(1, std::min(65535, (static_cast<int>(localRows) * 32 + threads - 1) / threads));
    spmvCsrWarp<<<blocks, threads>>>(dVal, dCols, dRows, dVec, localRows, dOut); // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t i = 0; i < iterations; ++i)
        spmvCsrWarp<<<blocks, threads>>>(dVal, dCols, dRows, dVec, localRows, dOut);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    CUDA_CHECK(cudaGetLastError());
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> localOut(localRows);
    CUDA_CHECK(cudaMemcpy(localOut.data(), dOut, localRows * sizeof(double), cudaMemcpyDeviceToHost));
    MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE, rank == 0 ? output.data() : nullptr,
                rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int status = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n",
               maxElapsed * 1e3, maxElapsed * 1e3 / iterations,
               2.0 * nItems * iterations / maxElapsed / 1e9);
        if (printResults) print_results(output, "OutputVector");
        if (validate) {
            std::vector<double> reference(numRows);
            spmvCpu(values.data(), cols.data(), rows.data(), vec.data(), numRows, reference.data());
            const bool valid = verifyResults(reference.data(), output.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaFree(dVal); cudaFree(dCols); cudaFree(dRows); cudaFree(dVec); cudaFree(dOut);
    MPI_Finalize();
    return status;
}
