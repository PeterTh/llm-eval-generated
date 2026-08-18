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
constexpr int THREADS_PER_BLOCK = 256;

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        const cudaError_t error_ = (call);                                                  \
        if (error_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(error_));                                            \
            MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error_));                            \
        }                                                                                   \
    } while (0)

void fill(double* values, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i)
        values[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * dim);
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            const index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) fillRemaining = true;
            const double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining)
                cols[nnzAssigned++] = j;
        }
    }
    rowDelimiters[dim] = n;
}

void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < dim; ++i) {
        double sum = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j)
            sum += val[j] * vec[cols[j]];
        out[i] = sum;
    }
}

__global__ void spmvCsrWarpKernel(const double* __restrict__ val,
                                  const index_t* __restrict__ cols,
                                  const index_t* __restrict__ rows,
                                  const double* __restrict__ vec,
                                  double* __restrict__ out, index_t localRows) {
    const unsigned lane = threadIdx.x & 31u;
    const index_t row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= localRows) return;

    double sum = 0.0;
    for (index_t j = rows[row] + lane; j < rows[row + 1]; j += 32)
        sum += val[j] * vec[cols[j]];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) out[row] = sum;
}

bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                return false;
            }
        } else if (std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR) {
            printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                   i, ref, res, std::abs((res - ref) / ref));
            return false;
        }
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -n <num>     Number of rows/columns in the matrix (default: 1024)\n"
           "  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n"
           "  -i <num>     Number of iterations (default: 10)\n"
           "  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n"
           "  -v           Enable validation\n  -r           Print results for external validation\n"
           "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    int parseOk = 1;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numRows = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) sparsity = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) maxVal = atof(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } parseOk = 0; }
    }
    if (!parseOk || !numRows || !sparsity || !iterations) {
        if (rank == 0 && parseOk) fprintf(stderr, "-n, -s, and -i must be positive\n");
        MPI_Finalize(); return 1;
    }
    const uint64_t nnz64 = (static_cast<uint64_t>(numRows) * numRows) / sparsity;
    if (nnz64 > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Matrix has too many non-zero elements for this benchmark\n");
        MPI_Finalize(); return 1;
    }
    const index_t nItems = static_cast<index_t>(nnz64);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { if (rank == 0) fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    int valid = 1;
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\nMatrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\nNon-zero elements: %u (%.2f%% sparse)\n",
               sparsity, nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (static_cast<double>(numRows) * numRows)));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\nMPI ranks: %d, CUDA devices visible per rank: %d\n",
               iterations, maxVal, validate ? "enabled" : "disabled", ranks, deviceCount);
    }

    std::vector<double> h_val(nItems), h_vec(numRows), h_out(numRows);
    std::vector<index_t> h_cols(nItems), h_rows(numRows + 1);
    if (rank == 0) printf("Initializing data structures...\n");
    // Every rank deliberately generates the same deterministic CSR input, avoiding a root bottleneck.
    fill(h_vec.data(), numRows, maxVal);
    fill(h_val.data(), nItems, maxVal);
    initRandomMatrix(h_cols.data(), h_rows.data(), nItems, numRows);

    const index_t baseRows = numRows / ranks;
    const index_t remainder = numRows % ranks;
    const index_t localRows = baseRows + (static_cast<index_t>(rank) < remainder);
    const index_t firstRow = static_cast<index_t>(rank) * baseRows + std::min<index_t>(rank, remainder);
    const index_t nnzBegin = h_rows[firstRow];
    const index_t nnzEnd = h_rows[firstRow + localRows];
    const index_t localNnz = nnzEnd - nnzBegin;
    std::vector<index_t> localRowsDelim(localRows + 1);
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= localRows; ++i) localRowsDelim[i] = h_rows[firstRow + i] - nnzBegin;
    std::vector<double> localOut(localRows);

    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rows = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val, std::max<size_t>(1, localNnz) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, std::max<size_t>(1, localNnz) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rows, (static_cast<size_t>(localRows) + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, static_cast<size_t>(numRows) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max<size_t>(1, localRows) * sizeof(double)));
    if (localNnz) {
        CUDA_CHECK(cudaMemcpy(d_val, h_val.data() + nnzBegin, localNnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data() + nnzBegin, localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rows, localRowsDelim.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), static_cast<size_t>(numRows) * sizeof(double), cudaMemcpyHostToDevice));

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    const dim3 block(THREADS_PER_BLOCK);
    const dim3 grid((static_cast<unsigned long long>(localRows) * 32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows) spmvCsrWarpKernel<<<grid, block>>>(d_val, d_cols, d_rows, d_vec, d_out, localRows);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double localSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (localRows) CUDA_CHECK(cudaMemcpy(localOut.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<int> recvCounts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        recvCounts[r] = static_cast<int>(baseRows + (static_cast<index_t>(r) < remainder));
        displacements[r] = r * static_cast<int>(baseRows) + std::min<int>(r, remainder);
    }
    MPI_Allgatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE, h_out.data(), recvCounts.data(),
                   displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Average time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n",
               seconds * 1000.0 / iterations, (2.0 * nItems * iterations) / seconds / 1e9);
        if (printResults) print_results(h_out, "OutputVector");
        if (validate) {
            printf("Computing reference solution...\n");
            std::vector<double> reference(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rows.data(), h_vec.data(), numRows, reference.data());
            printf("Validating result...\n");
            valid = verifyResults(reference.data(), h_out.data(), numRows) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaFree(d_out); cudaFree(d_vec); cudaFree(d_rows); cudaFree(d_cols); cudaFree(d_val);
    MPI_Finalize();
    return valid ? 0 : 1;
}
