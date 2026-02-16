#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>

#if USE_CUDA
#include <cuda_runtime.h>
#endif

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

#if USE_CUDA
// CUDA error checking
#define CUDA_CALL(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    MPI_Abort(MPI_COMM_WORLD, -1); } } while(0)
#endif

// ****************************************************************************
// Function: fill
// Purpose: Initialize array with random values (threaded with OpenMP)
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
    #pragma omp parallel
    {
        unsigned int seed = 1234u + (unsigned int)omp_get_thread_num();
        #pragma omp for schedule(static)
        for (index_t i = 0; i < n; ++i) {
            A[i] = maxVal * (rand_r(&seed) / (static_cast<double>(RAND_MAX) + 1.0));
        }
    }
}

// ****************************************************************************
// Function: initRandomMatrix (serial to preserve determinism)
// ****************************************************************************
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) fillRemaining = true;
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    rowDelimiters[dim] = n;
}

// ****************************************************************************
// CPU reference SpMV (parallelized with OpenMP)
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
    #pragma omp parallel for schedule(static)
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Function: verifyResults
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
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

#if USE_CUDA
// CUDA kernel: compute one row per thread (each thread computes dot product for its row)
extern "C" __global__ void spmvKernel(const double* __restrict__ val,
                                       const unsigned int* __restrict__ cols,
                                       const unsigned int* __restrict__ rowDelimiters,
                                       const double* __restrict__ vec,
                                       unsigned int startRow, unsigned int endRow,
                                       double* __restrict__ out) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = gid + startRow;
    if (row >= endRow) return;
    unsigned int rstart = rowDelimiters[row];
    unsigned int rend = rowDelimiters[row + 1];
    double s = 0.0;
    for (unsigned int j = rstart; j < rend; ++j) {
        s += val[j] * vec[cols[j]];
    }
    out[row - startRow] = s;
}
#endif

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse options on rank 0 and broadcast
    if (mpiRank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numRows = static_cast<index_t>(atoi(argv[++i]));
            else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) sparsity = static_cast<index_t>(atoi(argv[++i]));
            else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = static_cast<index_t>(atoi(argv[++i]));
            else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) maxVal = atof(argv[++i]);
            else if (strcmp(argv[i], "-v") == 0) validate = true;
            else if (strcmp(argv[i], "-r") == 0) printResults = true;
            else if (strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); MPI_Finalize(); return 0; }
        }
    }
    // Broadcast configuration
    MPI_Bcast(&numRows, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    const index_t nItems = (numRows * numRows) / sparsity;

    if (mpiRank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate host arrays (all ranks hold full matrix for simplicity)
    std::vector<double> h_val(nItems);
    std::vector<index_t> h_cols(nItems);
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out_local; h_out_local.resize(numRows); // will use only local portion

    if (mpiRank == 0) printf("Initializing data structures...\n");
    fill(h_vec.data(), numRows, maxVal);
    fill(h_val.data(), nItems, maxVal);
    initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

    // Reference on rank 0
    std::vector<double> h_reference;
    if (validate && mpiRank == 0) {
        if (mpiRank == 0) printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows, h_reference.data());
    }

    // Determine row partition for this MPI rank
    int base = numRows / mpiSize;
    int rem = numRows % mpiSize;
    int startRow = mpiRank < rem ? mpiRank * (base + 1) : mpiRank * base + rem;
    int localNumRows = mpiRank < rem ? base + 1 : base;
    int endRow = startRow + localNumRows;

    if (mpiRank == 0) printf("Computing SpMV using MPI+OpenMP+CUDA...\n");

#if USE_CUDA
    // Initialize CUDA device (one GPU per MPI rank assumed)
    int deviceCount = 0; CUDA_CALL(cudaGetDeviceCount(&deviceCount));
    int deviceId = 0;
    if (deviceCount > 0) deviceId = mpiRank % deviceCount;
    CUDA_CALL(cudaSetDevice(deviceId));

    // Device pointers
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    unsigned int *d_cols = nullptr, *d_rowDelims = nullptr;

    // Allocate and copy global arrays to device
    CUDA_CALL(cudaMalloc((void**)&d_val, sizeof(double) * nItems));
    CUDA_CALL(cudaMalloc((void**)&d_cols, sizeof(unsigned int) * nItems));
    CUDA_CALL(cudaMalloc((void**)&d_rowDelims, sizeof(unsigned int) * (numRows + 1)));
    CUDA_CALL(cudaMalloc((void**)&d_vec, sizeof(double) * numRows));
    CUDA_CALL(cudaMalloc((void**)&d_out, sizeof(double) * localNumRows));

    // copy once
    CUDA_CALL(cudaMemcpy(d_val, h_val.data(), sizeof(double) * nItems, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_cols, h_cols.data(), sizeof(unsigned int) * nItems, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_rowDelims, h_rowDelimiters.data(), sizeof(unsigned int) * (numRows + 1), cudaMemcpyHostToDevice));
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

#if USE_CUDA
    const int threadsPerBlock = 256;
    const int blocks = (localNumRows + threadsPerBlock - 1) / threadsPerBlock;

    for (index_t iter = 0; iter < iterations; ++iter) {
        // copy vector each iteration (assume vector may change)
        CUDA_CALL(cudaMemcpy(d_vec, h_vec.data(), sizeof(double) * numRows, cudaMemcpyHostToDevice));
        // launch kernel to compute rows [startRow, endRow)
        spmvKernel<<<blocks, threadsPerBlock>>>(d_val, d_cols, d_rowDelims, d_vec,
                                                static_cast<unsigned int>(startRow), static_cast<unsigned int>(endRow), d_out);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());
        // copy local output back (into h_out_local at position startRow..endRow-1)
        CUDA_CALL(cudaMemcpy(h_out_local.data() + startRow, d_out, sizeof(double) * localNumRows, cudaMemcpyDeviceToHost));
    }

#else
    // Fallback: use threaded CPU compute per rank
    for (index_t iter = 0; iter < iterations; ++iter) {
        // compute only local rows
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows, h_out_local.data());
    }
#endif

    double t1 = MPI_Wtime();
    double elapsed = t1 - t0; // seconds

    // Gather results to rank 0
    std::vector<int> recvCounts(mpiSize), displs(mpiSize);
    int localCount = localNumRows;
    MPI_Gather(&localCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpiRank == 0) {
        displs[0] = 0;
        for (int r = 1; r < mpiSize; ++r) displs[r] = displs[r-1] + recvCounts[r-1];
    }

    std::vector<double> h_out_global;
    if (mpiRank == 0) h_out_global.resize(numRows);

    // send each rank's local contiguous block (h_out_local[startRow .. endRow-1])
    MPI_Gatherv(h_out_local.data() + startRow, localNumRows, MPI_DOUBLE,
                h_out_global.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Compute total GFLOPS on rank 0
    if (mpiRank == 0) {
        double gflops = (2.0 * static_cast<double>(nItems) * static_cast<double>(iterations)) / elapsed / 1e9;
        printf("Computation time: %.3f s\n", elapsed);
        printf("Average time per iteration: %.3f s\n", elapsed / static_cast<double>(iterations));
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) print_results(h_out_global, "OutputVector");

        if (validate) {
            printf("Validating result...\n");
            bool valid = verifyResults(h_reference.data(), h_out_global.data(), numRows);
            if (valid) { printf("Validation: PASSED\n"); }
            else { printf("Validation: FAILED\n"); }
        }
    }

#if USE_CUDA
    // cleanup
    CUDA_CALL(cudaFree(d_val)); CUDA_CALL(cudaFree(d_cols)); CUDA_CALL(cudaFree(d_rowDelims));
    CUDA_CALL(cudaFree(d_vec)); CUDA_CALL(cudaFree(d_out));
#endif

    MPI_Finalize();
    return 0;
}
