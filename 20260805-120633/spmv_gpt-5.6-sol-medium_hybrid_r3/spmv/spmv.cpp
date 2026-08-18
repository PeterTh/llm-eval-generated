#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

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

static void cudaCheck(cudaError_t status, const char* operation, MPI_Comm comm) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        std::fprintf(stderr, "Rank %d: CUDA error in %s: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(comm, EXIT_FAILURE);
    }
}

void fill(double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n,
                      const index_t dim) {
    index_t nnzAssigned = 0;
    const double prob = static_cast<double>(n) /
                        (static_cast<double>(dim) * static_cast<double>(dim));
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t entriesLeft = static_cast<uint64_t>(dim) * dim -
                                         (static_cast<uint64_t>(i) * dim + j);
            const index_t needToAssign = n - nnzAssigned;
            if (entriesLeft <= needToAssign) fillRemaining = true;
            const double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                cols[nnzAssigned++] = j;
            }
        }
    }
    rowDelimiters[dim] = n;
}

// Each warp owns a row.  The grid-stride warp loop keeps launch overhead and
// occupancy good for both small and very large matrices.
__global__ void spmvCsrWarpKernel(const double* __restrict__ val,
                                  const index_t* __restrict__ cols,
                                  const index_t* __restrict__ row,
                                  const double* __restrict__ vec,
                                  index_t numRows,
                                  double* __restrict__ out) {
    const unsigned lane = threadIdx.x & 31u;
    index_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const index_t warpStride = (gridDim.x * blockDim.x) >> 5;
    for (; warp < numRows; warp += warpStride) {
        double sum = 0.0;
        for (index_t j = row[warp] + lane; j < row[warp + 1]; j += 32) {
            sum += val[j] * __ldg(vec + cols[j]);
        }
        for (int offset = 16; offset; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) out[warp] = sum;
    }
}

void spmvCpu(const double* val, const index_t* cols, const index_t* row,
             const double* vec, const index_t dim, double* out) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(dim); ++i) {
        double sum = 0.0;
        for (index_t j = row[i]; j < row[i + 1]; ++j) sum += val[j] * vec[cols[j]];
        out[i] = sum;
    }
}

bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i], res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                std::printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                return false;
            }
        } else {
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                std::printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                            i, ref, res, relError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n  -n <num>     Number of rows/columns (default: 1024)\n");
    std::printf("  -s <num>     Sparsity: 1 out of N is non-zero (default: 10)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -m <val>     Maximum element value (default: 1.0)\n");
    std::printf("  -v           Enable validation\n  -r           Print results\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false, help = false;
    int parseOk = 1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) numRows = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) sparsity = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        else if (!std::strcmp(argv[i], "-m") && i + 1 < argc) maxVal = std::strtod(argv[++i], nullptr);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else { if (rank == 0) std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]); parseOk = 0; }
    }
    if (help || !parseOk) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return help ? 0 : 1;
    }
    const uint64_t square = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nnz64 = sparsity ? square / sparsity : 0;
    if (!numRows || !sparsity || !iterations || nnz64 > std::numeric_limits<index_t>::max() ||
        nnz64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "Invalid problem size (MPI CSR counts must fit in 32 bits)\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nnz64);

    // Select GPUs by node-local rank so separate nodes independently use all devices.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", MPI_COMM_WORLD);
    if (!deviceCount) {
        if (rank == 0) std::fprintf(stderr, "No CUDA device is available\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", MPI_COMM_WORLD);
    MPI_Comm_free(&localComm);

    std::vector<int> rowCounts(ranks), rowDispls(ranks);
    for (int r = 0; r < ranks; ++r) {
        const uint64_t begin = static_cast<uint64_t>(numRows) * r / ranks;
        const uint64_t end = static_cast<uint64_t>(numRows) * (r + 1) / ranks;
        rowDispls[r] = static_cast<int>(begin);
        rowCounts[r] = static_cast<int>(end - begin);
    }
    const index_t localRows = static_cast<index_t>(rowCounts[rank]);

    std::vector<double> hVal, hVec(numRows), hOut;
    std::vector<index_t> hCols, hRow;
    if (rank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (MPI + OpenMP + CUDA)\n");
        std::printf("Matrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\n", numRows, numRows, sparsity);
        std::printf("Non-zero elements: %u (%.2f%% sparse)\nIterations: %u\nMax value: %.2f\nValidation: %s\n",
                    nItems, 100.0 * (1.0 - static_cast<double>(nItems) / square),
                    iterations, maxVal, validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\nInitializing data structures...\n",
                    ranks, omp_get_max_threads());
        hVal.resize(nItems); hCols.resize(nItems); hRow.resize(numRows + 1); hOut.resize(numRows);
        fill(hVec.data(), numRows, maxVal);
        fill(hVal.data(), nItems, maxVal);
        initRandomMatrix(hCols.data(), hRow.data(), nItems, numRows);
    }
    MPI_Bcast(hVec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> nnzCounts(ranks), nnzDispls(ranks), delimiterCounts(ranks), delimiterDispls(ranks);
    if (rank == 0) {
        for (int r = 0; r < ranks; ++r) {
            const index_t first = hRow[rowDispls[r]], last = hRow[rowDispls[r] + rowCounts[r]];
            nnzDispls[r] = static_cast<int>(first); nnzCounts[r] = static_cast<int>(last - first);
            delimiterDispls[r] = rowDispls[r]; delimiterCounts[r] = rowCounts[r] + 1;
        }
    }
    int localNnz = 0;
    MPI_Scatter(nnzCounts.data(), 1, MPI_INT, &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<double> localVal(localNnz), localOut(localRows);
    std::vector<index_t> localCols(localNnz), localRow(localRows + 1);
    MPI_Scatterv(rank == 0 ? hVal.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? hCols.data() : nullptr, nnzCounts.data(), nnzDispls.data(), MPI_UINT32_T,
                 localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? hRow.data() : nullptr, delimiterCounts.data(), delimiterDispls.data(), MPI_UINT32_T,
                 localRow.data(), static_cast<int>(localRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t rowBase = localRow[0];
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i <= static_cast<int64_t>(localRows); ++i) localRow[i] -= rowBase;

    std::vector<double> reference;
    if (rank == 0 && validate) {
        std::printf("Computing OpenMP reference solution...\n");
        reference.resize(numRows);
        spmvCpu(hVal.data(), hCols.data(), hRow.data(), hVec.data(), numRows, reference.data());
    }

    double *dVal = nullptr, *dVec = nullptr, *dOut = nullptr;
    index_t *dCols = nullptr, *dRow = nullptr;
    cudaCheck(cudaMalloc(&dVal, std::max<size_t>(1, localVal.size()) * sizeof(double)), "cudaMalloc values", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&dCols, std::max<size_t>(1, localCols.size()) * sizeof(index_t)), "cudaMalloc columns", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&dRow, localRow.size() * sizeof(index_t)), "cudaMalloc rows", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&dVec, hVec.size() * sizeof(double)), "cudaMalloc vector", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&dOut, std::max<size_t>(1, localOut.size()) * sizeof(double)), "cudaMalloc output", MPI_COMM_WORLD);
    if (localNnz) {
        cudaCheck(cudaMemcpy(dVal, localVal.data(), localVal.size() * sizeof(double), cudaMemcpyHostToDevice), "copy values", MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(dCols, localCols.data(), localCols.size() * sizeof(index_t), cudaMemcpyHostToDevice), "copy columns", MPI_COMM_WORLD);
    }
    cudaCheck(cudaMemcpy(dRow, localRow.data(), localRow.size() * sizeof(index_t), cudaMemcpyHostToDevice), "copy rows", MPI_COMM_WORLD);
    cudaCheck(cudaMemcpy(dVec, hVec.data(), hVec.size() * sizeof(double), cudaMemcpyHostToDevice), "copy vector", MPI_COMM_WORLD);

    constexpr int threads = 256;
    const int blocks = localRows
        ? std::min(65535, std::max(1, (static_cast<int>(localRows) + 7) / 8)) : 0;
    // Warm up the context and kernel so the benchmark measures steady-state
    // SpMV rather than CUDA lazy initialization/JIT costs.
    if (localRows) {
        spmvCsrWarpKernel<<<blocks, threads>>>(dVal, dCols, dRow, dVec, localRows, dOut);
        cudaCheck(cudaGetLastError(), "SpMV warm-up launch", MPI_COMM_WORLD);
    }
    cudaCheck(cudaDeviceSynchronize(), "SpMV warm-up", MPI_COMM_WORLD);

    if (rank == 0) std::printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    if (localRows) {
        for (index_t iter = 0; iter < iterations; ++iter)
            spmvCsrWarpKernel<<<blocks, threads>>>(dVal, dCols, dRow, dVec, localRows, dOut);
    }
    cudaCheck(cudaGetLastError(), "SpMV kernel launch", MPI_COMM_WORLD);
    cudaCheck(cudaDeviceSynchronize(), "SpMV kernel execution", MPI_COMM_WORLD);
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (localRows) cudaCheck(cudaMemcpy(localOut.data(), dOut, localOut.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy output", MPI_COMM_WORLD);
    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                rank == 0 ? hOut.data() : nullptr, rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    cudaFree(dVal); cudaFree(dCols); cudaFree(dRow); cudaFree(dVec); cudaFree(dOut);
    int success = 1;
    if (rank == 0) {
        const double milliseconds = seconds * 1000.0;
        const double gflops = 2.0 * nItems * iterations / seconds / 1e9;
        std::printf("Computation time: %.3f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n",
                    milliseconds, milliseconds / iterations, gflops);
        if (printResults) print_results(hOut, "OutputVector");
        if (validate) {
            std::printf("Validating result...\n");
            success = verifyResults(reference.data(), hOut.data(), numRows) ? 1 : 0;
            std::printf("Validation: %s\n", success ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return success ? 0 : 1;
}
