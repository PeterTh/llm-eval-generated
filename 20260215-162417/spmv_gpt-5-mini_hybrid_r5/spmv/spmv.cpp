#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values (parallelized with OpenMP)
//
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
    // Parallel fill for performance; deterministic seed per thread not required for benchmark
#pragma omp parallel
    {
        unsigned int tid = omp_get_thread_num();
        unsigned int nthreads = omp_get_num_threads();
        unsigned int chunk = (n + nthreads - 1) / nthreads;
        unsigned int start = tid * chunk;
        unsigned int end = std::min<index_t>(start + chunk, n);
        unsigned int local_seed = 8675309 + tid;
        for (index_t i = start; i < end; ++i) {
            // simple LCG per thread for reproducible-ish values
            local_seed = (1103515245u * local_seed + 12345u);
            A[i] = maxVal * (double)(local_seed & 0xFFFF) / 65536.0;
        }
    }
}

// Keep initRandomMatrix single-threaded for deterministic CSR layout
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
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    rowDelimiters[dim] = n;
}

// CPU reference implementation (used for validation)
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

// CUDA kernel: each thread computes one row of the local CSR matrix
extern "C" __global__ void spmv_kernel(const double* __restrict__ d_val,
                                        const unsigned int* __restrict__ d_cols,
                                        const unsigned int* __restrict__ d_rowDelims,
                                        const double* __restrict__ d_vec,
                                        unsigned int numRows,
                                        double* __restrict__ d_out) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows) return;
    unsigned int start = d_rowDelims[row];
    unsigned int end = d_rowDelims[row + 1];
    double sum = 0.0;
    for (unsigned int j = start; j < end; ++j) {
        sum += d_val[j] * d_vec[d_cols[j]];
    }
    d_out[row] = sum;
}

int main(int argc, char** argv) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            maxVal = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    const index_t nItems = (numRows * numRows) / sparsity;

    int mpiRank = 0, mpiSize = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

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

    // Allocate and initialize data structures on rank 0 then distribute
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out_local;
    std::vector<double> h_out_global;

    if (mpiRank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast vector to all ranks
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast row delimiters and cols/vals as needed by scattering
    if (mpiRank != 0) {
        h_rowDelimiters.resize(numRows + 1);
    }
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    if (mpiRank != 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
    }
    MPI_Bcast(h_cols.data(), nItems, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_val.data(), nItems, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Determine local row distribution
    index_t base = numRows / mpiSize;
    index_t rem = numRows % mpiSize;
    index_t localRows = base + (mpiRank < rem ? 1 : 0);
    index_t startRow = mpiRank * base + std::min<index_t>(mpiRank, rem);
    index_t endRow = startRow + localRows;

    // Determine local nnz
    index_t local_nnz = h_rowDelimiters[endRow] - h_rowDelimiters[startRow];

    // Prepare local CSR (compact)
    std::vector<double> local_val(local_nnz);
    std::vector<index_t> local_cols(local_nnz);
    std::vector<index_t> local_rowDelims(localRows + 1);

    for (index_t i = 0; i < localRows + 1; ++i) {
        local_rowDelims[i] = h_rowDelimiters[startRow + i] - h_rowDelimiters[startRow];
    }
    for (index_t i = 0; i < local_nnz; ++i) {
        local_cols[i] = h_cols[h_rowDelimiters[startRow] + i];
        local_val[i] = h_val[h_rowDelimiters[startRow] + i];
    }

    h_out_local.assign(localRows, 0.0);
    if (mpiRank == 0) h_out_global.assign(numRows, 0.0);

    // Prepare device memory
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    unsigned int *d_cols = nullptr, *d_rowDel = nullptr;
    cudaSetDevice(0);

    cudaMalloc(&d_val, sizeof(double) * local_nnz);
    cudaMalloc(&d_cols, sizeof(unsigned int) * local_nnz);
    cudaMalloc(&d_rowDel, sizeof(unsigned int) * (localRows + 1));
    cudaMalloc(&d_vec, sizeof(double) * numRows);
    cudaMalloc(&d_out, sizeof(double) * localRows);

    // Copy static arrays to device once
    cudaMemcpy(d_val, local_val.data(), sizeof(double) * local_nnz, cudaMemcpyHostToDevice);
    // convert cols and rowDelims to unsigned int on host for device copy
    std::vector<unsigned int> local_cols_ui(local_nnz);
    for (index_t i = 0; i < local_nnz; ++i) local_cols_ui[i] = static_cast<unsigned int>(local_cols[i]);
    std::vector<unsigned int> local_rowDelims_ui(localRows + 1);
    for (index_t i = 0; i < localRows + 1; ++i) local_rowDelims_ui[i] = static_cast<unsigned int>(local_rowDelims[i]);

    cudaMemcpy(d_cols, local_cols_ui.data(), sizeof(unsigned int) * local_nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowDel, local_rowDelims_ui.data(), sizeof(unsigned int) * (localRows + 1), cudaMemcpyHostToDevice);
    // copy vector
    cudaMemcpy(d_vec, h_vec.data(), sizeof(double) * numRows, cudaMemcpyHostToDevice);

    MPI_Barrier(MPI_COMM_WORLD);
    double local_start = MPI_Wtime();
    // Run iterations: launch CUDA kernel each iteration
    const int threads = 256;
    const int blocks = (localRows + threads - 1) / threads;
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmv_kernel<<<blocks, threads>>>(d_val, d_cols, d_rowDel, d_vec, (unsigned int)localRows, d_out);
        cudaMemcpy(h_out_local.data(), d_out, sizeof(double) * localRows, cudaMemcpyDeviceToHost);
    }
    double local_end = MPI_Wtime();
    double local_duration = local_end - local_start;

    // Reduce to get max time across ranks
    double max_time = 0.0;
    MPI_Reduce(&local_duration, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather outputs to root for printing/validation
    std::vector<int> recvcounts(mpiSize);
    std::vector<int> displs(mpiSize);
    for (int r = 0; r < mpiSize; ++r) {
        int rows_r = base + (r < rem ? 1 : 0);
        recvcounts[r] = rows_r;
        displs[r] = (r == 0) ? 0 : displs[r-1] + recvcounts[r-1];
    }
    MPI_Gatherv(h_out_local.data(), localRows, MPI_DOUBLE,
                (mpiRank==0? h_out_global.data(): nullptr), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (mpiRank == 0) {
        long ms = static_cast<long>(max_time * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double gflops = (2.0 * nItems * iterations) / (max_time) / 1e9;
        const double avgTime = (max_time * 1000.0) / static_cast<double>(iterations);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) print_results(h_out_global, "OutputVector");

        if (validate) {
            printf("Computing reference solution for validation...\n");
            // Root already has full matrix and vector; compute reference using CPU (multithreaded)
            std::vector<double> reference(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows, reference.data());
            printf("Validating result...\n");
            const bool valid = verifyResults(reference.data(), h_out_global.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // Cleanup
    cudaFree(d_val);
    cudaFree(d_cols);
    cudaFree(d_rowDel);
    cudaFree(d_vec);
    cudaFree(d_out);

    MPI_Finalize();
    return 0;
}
