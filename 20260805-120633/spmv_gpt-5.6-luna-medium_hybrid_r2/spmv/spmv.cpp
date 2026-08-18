#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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

static void dieCuda(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    }
}

void fill(double* a, index_t n, double maxVal) {
    // Initialization is parallelized independently of the accelerator work.
    #pragma omp parallel
    {
        unsigned int state = 8675309u + 101u * static_cast<unsigned int>(omp_get_thread_num());
        #pragma omp for
        for (index_t i = 0; i < n; ++i) {
            state = 1664525u * state + 1013904223u;
            a[i] = maxVal * (static_cast<double>(state) / 4294967296.0);
        }
    }
}

void initRandomMatrix(index_t* cols, index_t* rowDelimiters, index_t n, index_t dim) {
    index_t assigned = 0;
    const double prob = static_cast<double>(n) /
                        (static_cast<double>(dim) * static_cast<double>(dim));
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = assigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t position = static_cast<uint64_t>(i) * dim + j;
            const uint64_t left = static_cast<uint64_t>(dim) * dim - position;
            if (left <= n - assigned) fillRemaining = true;
            const double r = static_cast<double>(rand()) / RAND_MAX;
            if (assigned < n && (r <= prob || fillRemaining)) cols[assigned++] = j;
        }
    }
    rowDelimiters[dim] = n;
}

void spmvCpu(const double* val, const index_t* cols, const index_t* rows,
             const double* vec, index_t dim, double* out) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(dim); ++i) {
        double sum = 0.0;
        for (index_t j = rows[i]; j < rows[i + 1]; ++j) sum += val[j] * vec[cols[j]];
        out[i] = sum;
    }
}

__global__ void spmvKernel(const double* val, const index_t* cols, const index_t* rows,
                           const double* vec, index_t rowCount, double* out) {
    const index_t row = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rowCount) return;
    double sum = 0.0;
    for (index_t j = rows[row]; j < rows[row + 1]; ++j) sum += val[j] * vec[cols[j]];
    out[row] = sum;
}

bool verifyResults(const double* reference, const double* result, index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i], res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                return false;
            }
        } else if (std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR) {
            printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
            return false;
        }
    }
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n  -n <num> matrix dimension (default: 1024)\n"
           "  -s <num> one nonzero in N entries (default: 10)\n"
           "  -i <num> iterations (default: 10)\n  -m <val> maximum value (default: 1.0)\n"
           "  -v validate  -r print output  -h help\n", name);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numRows = static_cast<index_t>(strtoul(argv[++i], nullptr, 10));
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) sparsity = static_cast<index_t>(strtoul(argv[++i], nullptr, 10));
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = static_cast<index_t>(strtoul(argv[++i], nullptr, 10));
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) maxVal = atof(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (sparsity == 0 || numRows == 0 || iterations == 0) { MPI_Finalize(); return 1; }
    const index_t nItems = static_cast<index_t>((static_cast<uint64_t>(numRows) * numRows) / sparsity);

    std::vector<double> val(nItems), vec(numRows);
    std::vector<index_t> cols(nItems), rows(numRows + 1);
    fill(vec.data(), numRows, maxVal);
    fill(val.data(), nItems, maxVal);
    initRandomMatrix(cols.data(), rows.data(), nItems, numRows);
    MPI_Bcast(vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(val.data(), nItems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(cols.data(), nItems, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(rows.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t first = static_cast<index_t>((static_cast<uint64_t>(rank) * numRows) / world);
    const index_t last = static_cast<index_t>((static_cast<uint64_t>(rank + 1) * numRows) / world);
    const index_t localRows = last - first;
    const index_t localNnz = rows[last] - rows[first];
    std::vector<double> localVal(localNnz), localOut(localRows);
    std::vector<index_t> localCols(localNnz), localRowsIdx(localRows + 1);
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(localRows + 1); ++i) localRowsIdx[i] = rows[first + i] - rows[first];
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(localNnz); ++i) {
        localVal[i] = val[rows[first] + i];
        localCols[i] = cols[rows[first] + i];
    }

    int deviceCount = 0;
    dieCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) { fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    dieCuda(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");

    double *dVal = nullptr, *dVec = nullptr, *dOut = nullptr;
    index_t *dCols = nullptr, *dRows = nullptr;
    dieCuda(cudaMalloc(&dVal, localNnz * sizeof(double)), "cudaMalloc val");
    dieCuda(cudaMalloc(&dCols, localNnz * sizeof(index_t)), "cudaMalloc cols");
    dieCuda(cudaMalloc(&dRows, (localRows + 1) * sizeof(index_t)), "cudaMalloc rows");
    dieCuda(cudaMalloc(&dVec, numRows * sizeof(double)), "cudaMalloc vec");
    dieCuda(cudaMalloc(&dOut, localRows * sizeof(double)), "cudaMalloc out");
    dieCuda(cudaMemcpy(dVal, localVal.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice), "copy val");
    dieCuda(cudaMemcpy(dCols, localCols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice), "copy cols");
    dieCuda(cudaMemcpy(dRows, localRowsIdx.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice), "copy rows");
    dieCuda(cudaMemcpy(dVec, vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice), "copy vec");

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    constexpr int blockSize = 256;
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvKernel<<<(localRows + blockSize - 1) / blockSize, blockSize>>>(dVal, dCols, dRows, dVec, localRows, dOut);
        dieCuda(cudaGetLastError(), "spmvKernel");
    }
    dieCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    const auto end = std::chrono::high_resolution_clock::now();
    const double localMs = std::chrono::duration<double, std::milli>(end - start).count();
    double elapsedMs = 0.0;
    MPI_Reduce(&localMs, &elapsedMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    dieCuda(cudaMemcpy(localOut.data(), dOut, localRows * sizeof(double), cudaMemcpyDeviceToHost), "copy output");

    std::vector<int> counts(world), displs(world);
    for (int r = 0; r < world; ++r) {
        counts[r] = static_cast<int>((static_cast<uint64_t>(r + 1) * numRows) / world -
                                     (static_cast<uint64_t>(r) * numRows) / world);
        displs[r] = static_cast<int>((static_cast<uint64_t>(r) * numRows) / world);
    }
    std::vector<double> output(rank == 0 ? numRows : 0);
    MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE, output.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<double> reference;
        if (validate) { reference.resize(numRows); spmvCpu(val.data(), cols.data(), rows.data(), vec.data(), numRows, reference.data()); }
        printf("Sparse Matrix-Vector Multiplication (MPI/OpenMP/CUDA)\nMatrix size: %u x %u\nNon-zero elements: %u\nIterations: %u\n", numRows, numRows, nItems, iterations);
        const double avg = elapsedMs / iterations;
        printf("Computation time: %.3f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n", elapsedMs, avg, (2.0 * nItems * iterations) / (elapsedMs / 1000.0) / 1e9);
        if (printResults) print_results(output, "OutputVector");
        if (validate) { const bool ok = verifyResults(reference.data(), output.data(), numRows); printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            if (!ok) { cudaFree(dVal); cudaFree(dCols); cudaFree(dRows); cudaFree(dVec); cudaFree(dOut); MPI_Finalize(); return 1; } }
    }
    cudaFree(dVal); cudaFree(dCols); cudaFree(dRows); cudaFree(dVec); cudaFree(dOut);
    MPI_Finalize();
    return 0;
}
