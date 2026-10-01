#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;
constexpr double MAX_RELATIVE_ERROR = 0.02;

void fill(double* a, index_t n, double maxVal) {
    for (index_t i = 0; i < n; ++i)
        a[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

// Keep the original random sequence so the generated benchmark is reproducible.
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, index_t n, index_t dim) {
    index_t nnzAssigned = 0;
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * dim);
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t numEntriesLeft = uint64_t(dim) * dim - (uint64_t(i) * dim + j);
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
             const double* vec, index_t dim, double* out) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j)
            t += val[j] * vec[cols[j]];
        out[i] = t;
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
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i, ref, res, relError);
                return false;
            }
        }
    }
    return true;
}

// One thread per row is efficient for short rows and preserves summation order.
__global__ void spmvThread(const double* val, const index_t* cols, const index_t* row,
                           const double* vec, index_t nrows, double* out) {
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrows) return;
    double sum = 0.0;
    for (index_t j = row[i]; j < row[i + 1]; ++j)
        sum += val[j] * vec[cols[j]];
    out[i] = sum;
}

// A warp cooperates on each longer row; every lane traverses consecutive entries.
__global__ void spmvWarp(const double* val, const index_t* cols, const index_t* row,
                         const double* vec, index_t nrows, double* out) {
    const index_t i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const unsigned lane = threadIdx.x & 31;
    if (i >= nrows) return;
    double sum = 0.0;
    for (index_t j = row[i] + lane; j < row[i + 1]; j += 32)
        sum += val[j] * vec[cols[j]];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) out[i] = sum;
}

void cudaCheck(cudaError_t error, int rank, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    bool help = false, badOption = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            numRows = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc)
            sparsity = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
            iterations = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)
            maxVal = atof(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) help = true;
        else { badOption = true; if (rank == 0) printf("Unknown option: %s\n", argv[i]); }
    }
    if (help || badOption) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return badOption ? 1 : 0;
    }
    const uint64_t square = uint64_t(numRows) * numRows;
    if (!numRows || !sparsity || !iterations || numRows > INT_MAX ||
        square / sparsity > INT_MAX || nranks > INT_MAX / 2) {
        if (rank == 0) fprintf(stderr, "Invalid size, sparsity, iteration count, or MPI rank count\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(square / sparsity);
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / square));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal,
               validate ? "enabled" : "disabled");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), rank, "query CUDA devices");
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), rank, "select CUDA device");

    std::vector<double> h_val, h_vec(numRows), h_out;
    std::vector<index_t> h_cols, h_rowDelimiters;
    std::vector<int> rowCounts, rowDispls, nnzCounts, nnzDispls, delimCounts;
    std::vector<double> h_reference;
    if (rank == 0) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(size_t(numRows) + 1);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
        rowCounts.resize(nranks);
        rowDispls.resize(nranks);
        nnzCounts.resize(nranks);
        nnzDispls.resize(nranks);
        delimCounts.resize(nranks);
        // Find row boundaries close to equal nonzero counts. Empty partitions are valid.
        index_t previous = 0;
        for (int p = 0; p < nranks; ++p) {
            const index_t end = p == nranks - 1 ? numRows :
                static_cast<index_t>(std::lower_bound(h_rowDelimiters.begin() + previous,
                    h_rowDelimiters.end(), uint64_t(nItems) * (p + 1) / nranks) - h_rowDelimiters.begin());
            rowDispls[p] = static_cast<int>(previous);
            rowCounts[p] = static_cast<int>(end - previous);
            nnzDispls[p] = static_cast<int>(h_rowDelimiters[previous]);
            nnzCounts[p] = static_cast<int>(h_rowDelimiters[end] - h_rowDelimiters[previous]);
            delimCounts[p] = rowCounts[p] + 1;
            previous = end;
        }
    }
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int localRows = 0, localNnz = 0;
    MPI_Scatter(rank == 0 ? rowCounts.data() : nullptr, 1, MPI_INT,
                &localRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(rank == 0 ? nnzCounts.data() : nullptr, 1, MPI_INT,
                &localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<index_t> localCols(localNnz), localRow(size_t(localRows) + 1);
    std::vector<double> localVal(localNnz), localOut(localRows);
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr, MPI_DOUBLE,
                 localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDispls.data() : nullptr, MPI_UINT32_T,
                 localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 rank == 0 ? delimCounts.data() : nullptr,
                 rank == 0 ? rowDispls.data() : nullptr, MPI_UINT32_T,
                 localRow.data(), localRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t base = localRow[0];
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i <= localRows; ++i) localRow[i] -= base;

    double *d_val, *d_vec, *d_out;
    index_t *d_cols, *d_row;
    cudaCheck(cudaMalloc(&d_val, size_t(std::max(localNnz, 1)) * sizeof(double)), rank, "allocate values");
    cudaCheck(cudaMalloc(&d_cols, size_t(std::max(localNnz, 1)) * sizeof(index_t)), rank, "allocate columns");
    cudaCheck(cudaMalloc(&d_row, size_t(localRows + 1) * sizeof(index_t)), rank, "allocate row offsets");
    cudaCheck(cudaMalloc(&d_vec, size_t(numRows) * sizeof(double)), rank, "allocate vector");
    cudaCheck(cudaMalloc(&d_out, size_t(std::max(localRows, 1)) * sizeof(double)), rank, "allocate output");
    cudaCheck(cudaMemcpy(d_val, localVal.data(), size_t(localNnz) * sizeof(double), cudaMemcpyHostToDevice), rank, "copy values");
    cudaCheck(cudaMemcpy(d_cols, localCols.data(), size_t(localNnz) * sizeof(index_t), cudaMemcpyHostToDevice), rank, "copy columns");
    cudaCheck(cudaMemcpy(d_row, localRow.data(), size_t(localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice), rank, "copy row offsets");
    cudaCheck(cudaMemcpy(d_vec, h_vec.data(), size_t(numRows) * sizeof(double), cudaMemcpyHostToDevice), rank, "copy vector");

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    constexpr int threads = 256;
    const bool useWarp = localRows && localNnz / localRows >= 32;
    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows) {
            if (useWarp)
                spmvWarp<<<(localRows + threads / 32 - 1) / (threads / 32), threads>>>(
                    d_val, d_cols, d_row, d_vec, localRows, d_out);
            else
                spmvThread<<<(localRows + threads - 1) / threads, threads>>>(
                    d_val, d_cols, d_row, d_vec, localRows, d_out);
        }
    }
    cudaCheck(cudaGetLastError(), rank, "launch SpMV kernel");
    cudaCheck(cudaDeviceSynchronize(), rank, "complete SpMV kernels");
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    cudaCheck(cudaMemcpy(localOut.data(), d_out, size_t(localRows) * sizeof(double), cudaMemcpyDeviceToHost), rank, "copy output");
    if (rank == 0) h_out.resize(numRows);
    MPI_Gatherv(localOut.data(), localRows, MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr,
                rank == 0 ? rowCounts.data() : nullptr,
                rank == 0 ? rowDispls.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    cudaFree(d_val); cudaFree(d_cols); cudaFree(d_row); cudaFree(d_vec); cudaFree(d_out);

    int status = 0;
    if (rank == 0) {
        const double ms = seconds * 1000.0;
        printf("Computation time: %.3f ms\n", ms);
        printf("Average time per iteration: %.3f ms\n", ms / iterations);
        printf("Performance: %.3f GFLOPS\n", 2.0 * nItems * iterations / seconds / 1e9);
        if (printResults) print_results(h_out, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
