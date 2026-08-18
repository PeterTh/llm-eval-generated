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

void fill(double* a, index_t n, double maxVal) {
    // Keep the original serial rand() stream, which is part of the benchmark's
    // input semantics. OpenMP is used for the independent host-side work below.
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

__global__ void spmvKernel(const double* __restrict__ val,
                           const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec, double* __restrict__ out,
                           index_t rows) {
    const index_t row = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    double sum = 0.0;
    for (index_t j = rowDelimiters[row]; j < rowDelimiters[row + 1]; ++j)
        sum += val[j] * vec[cols[j]];
    out[row] = sum;
}

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

bool verifyResults(const double* reference, const double* result, index_t size) {
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

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\nOptions:\n", progName);
    std::printf("  -n <num>     Number of rows/columns (default: 1024)\n");
    std::printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -m <val>     Maximum value (default: 1.0)\n");
    std::printf("  -v           Enable validation\n  -r           Print results\n  -h           Show this help\n");
}

int main(int argc, char** argv) {
    int mpiProvided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiProvided);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) numRows = static_cast<index_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) sparsity = static_cast<index_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = static_cast<index_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-m") && i + 1 < argc) maxVal = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (sparsity == 0 || numRows == 0 || iterations == 0 ||
        static_cast<uint64_t>(numRows) * numRows > std::numeric_limits<index_t>::max() * 10ULL) {
        if (rank == 0) std::fprintf(stderr, "Invalid benchmark dimensions or parameters\n");
        MPI_Finalize(); return 1;
    }
    const index_t nItems = static_cast<index_t>((static_cast<uint64_t>(numRows) * numRows) / sparsity);
    const index_t rowBegin = static_cast<index_t>((static_cast<uint64_t>(rank) * numRows) / world);
    const index_t rowEnd = static_cast<index_t>((static_cast<uint64_t>(rank + 1) * numRows) / world);
    const index_t localRows = rowEnd - rowBegin;

    std::vector<double> h_vec(numRows);
    std::vector<double> allVal;
    std::vector<index_t> allCols;
    std::vector<index_t> rowDelimiters(numRows + 1);
    if (rank == 0) {
        allVal.resize(nItems);
        allCols.resize(nItems);
        fill(h_vec.data(), numRows, maxVal);
        fill(allVal.data(), nItems, maxVal);
        initRandomMatrix(allCols.data(), rowDelimiters.data(), nItems, numRows);
    }
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t nnzBegin = rowDelimiters[rowBegin], nnzEnd = rowDelimiters[rowEnd];
    std::vector<int> nnzCounts(world), nnzDisplacements(world);
    for (int r = 0; r < world; ++r) {
        const index_t first = static_cast<index_t>((static_cast<uint64_t>(r) * numRows) / world);
        const index_t last = static_cast<index_t>((static_cast<uint64_t>(r + 1) * numRows) / world);
        nnzCounts[r] = static_cast<int>(rowDelimiters[last] - rowDelimiters[first]);
        nnzDisplacements[r] = static_cast<int>(rowDelimiters[first]);
    }
    std::vector<double> val(nnzEnd - nnzBegin);
    std::vector<index_t> cols(nnzEnd - nnzBegin);
    MPI_Scatterv(rank == 0 ? allVal.data() : nullptr, nnzCounts.data(), nnzDisplacements.data(), MPI_DOUBLE,
                 val.data(), nnzCounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? allCols.data() : nullptr, nnzCounts.data(), nnzDisplacements.data(), MPI_UINT32_T,
                 cols.data(), nnzCounts[rank], MPI_UINT32_T, 0, MPI_COMM_WORLD);
    std::vector<index_t> localRowsDelim(localRows + 1);
    for (index_t r = 0; r <= localRows; ++r) localRowsDelim[r] = rowDelimiters[rowBegin + r] - nnzBegin;
    std::vector<double> localOut(localRows), h_out(numRows);
    // An unconditional OpenMP phase keeps host preparation scalable as well.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(localRows); ++i) localOut[i] = 0.0;

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    cudaCheck(cudaSetDevice(rank % deviceCount), "cudaSetDevice");
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rowDelim = nullptr;
    // Allocate at least one element for empty row partitions (possible when
    // the MPI size exceeds the number of rows).
    cudaCheck(cudaMalloc(&d_val, std::max<size_t>(1, val.size()) * sizeof(double)), "cudaMalloc(val)");
    cudaCheck(cudaMalloc(&d_cols, std::max<size_t>(1, cols.size()) * sizeof(index_t)), "cudaMalloc(cols)");
    cudaCheck(cudaMalloc(&d_rowDelim, std::max<size_t>(1, localRowsDelim.size()) * sizeof(index_t)), "cudaMalloc(row delimiters)");
    cudaCheck(cudaMalloc(&d_vec, std::max<size_t>(1, h_vec.size()) * sizeof(double)), "cudaMalloc(vector)");
    cudaCheck(cudaMalloc(&d_out, std::max<size_t>(1, localOut.size()) * sizeof(double)), "cudaMalloc(output)");
    if (!val.empty()) cudaCheck(cudaMemcpy(d_val, val.data(), val.size() * sizeof(double), cudaMemcpyHostToDevice), "copy values");
    if (!cols.empty()) cudaCheck(cudaMemcpy(d_cols, cols.data(), cols.size() * sizeof(index_t), cudaMemcpyHostToDevice), "copy columns");
    cudaCheck(cudaMemcpy(d_rowDelim, localRowsDelim.data(), localRowsDelim.size() * sizeof(index_t), cudaMemcpyHostToDevice), "copy row delimiters");
    cudaCheck(cudaMemcpy(d_vec, h_vec.data(), h_vec.size() * sizeof(double), cudaMemcpyHostToDevice), "copy vector");

    if (rank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (MPI %d + OpenMP + CUDA)\nMatrix size: %u x %u\nNon-zero elements: %u\nIterations: %u\nValidation: %s\n",
                    world, numRows, numRows, nItems, iterations, validate ? "enabled" : "disabled");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows != 0) {
            const dim3 block(256), grid((localRows + block.x - 1) / block.x);
            spmvKernel<<<grid, block>>>(d_val, d_cols, d_rowDelim, d_vec, d_out, localRows);
        }
    }
    cudaCheck(cudaGetLastError(), "SpMV kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "SpMV execution");
    cudaCheck(cudaMemcpy(localOut.data(), d_out, localOut.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy output");
    const auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<int> counts(world), displacements(world);
    for (int r = 0; r < world; ++r) { counts[r] = static_cast<int>((static_cast<uint64_t>(r + 1) * numRows) / world - (static_cast<uint64_t>(r) * numRows) / world); displacements[r] = static_cast<int>((static_cast<uint64_t>(r) * numRows) / world); }
    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE, h_out.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double milliseconds = seconds * 1000.0;
        std::printf("Computation time: %.0f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n",
                    milliseconds, milliseconds / iterations, (2.0 * nItems * iterations) / seconds / 1e9);
        if (printResults) print_results(h_out, "OutputVector");
        if (validate) {
            std::vector<double> reference(numRows);
            spmvCpu(allVal.data(), allCols.data(), rowDelimiters.data(), h_vec.data(), numRows, reference.data());
            const bool valid = verifyResults(reference.data(), h_out.data(), numRows);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) { cudaFree(d_val); cudaFree(d_cols); cudaFree(d_rowDelim); cudaFree(d_vec); cudaFree(d_out); MPI_Finalize(); return 1; }
        }
    }
    cudaFree(d_val); cudaFree(d_cols); cudaFree(d_rowDelim); cudaFree(d_vec); cudaFree(d_out);
    MPI_Finalize();
    return 0;
}
