#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA SpMV kernel: one thread per row
__global__ void spmv_kernel(const double* __restrict__ val,
                            const index_t* __restrict__ cols,
                            const index_t* __restrict__ rowDelimiters,
                            const double* __restrict__ vec,
                            const index_t nrows,
                            double* __restrict__ out) {
    index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    double t = 0.0;
    for (index_t j = rowDelimiters[row]; j < rowDelimiters[row + 1]; ++j) {
        t += val[j] * vec[cols[j]];
    }
    out[row] = t;
}

void fill(double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

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

// CPU reference with OpenMP for validation
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
    bool valid = true;
    #pragma omp parallel for schedule(static)
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                #pragma omp critical
                {
                    if (valid) {
                        printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                        valid = false;
                    }
                }
            }
        } else {
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                #pragma omp critical
                {
                    if (valid) {
                        printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                               i, ref, res, relError);
                        valid = false;
                    }
                }
            }
        }
    }
    return valid;
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

// Helper: compute row partition for a given rank
static void getRowPartition(index_t numRows, int nprocs, int r,
                            index_t& row_start, index_t& local_nrows) {
    index_t base = numRows / static_cast<index_t>(nprocs);
    index_t extra = numRows % static_cast<index_t>(nprocs);
    row_start = static_cast<index_t>(r) * base + std::min(static_cast<index_t>(r), extra);
    local_nrows = base + (static_cast<index_t>(r) < extra ? 1 : 0);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks)
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d\n", nprocs, omp_get_max_threads());
    }

    // Rank 0 initializes all data (preserving sequential RNG behavior)
    std::vector<double> h_val(rank == 0 ? nItems : 0);
    std::vector<index_t> h_cols(rank == 0 ? nItems : 0);
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast shared data to all ranks
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Compute reference solution on rank 0 for validation (OpenMP-parallel)
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Compute per-rank row and nnz partitioning
    index_t row_start, local_nrows;
    getRowPartition(numRows, nprocs, rank, row_start, local_nrows);
    index_t row_end = row_start + local_nrows;
    index_t local_nnz_offset = h_rowDelimiters[row_start];
    index_t local_nnz = h_rowDelimiters[row_end] - local_nnz_offset;

    std::vector<int> nnz_counts(nprocs), nnz_displs(nprocs);
    std::vector<int> row_counts(nprocs), row_displs(nprocs);
    for (int r = 0; r < nprocs; r++) {
        index_t rs, rn;
        getRowPartition(numRows, nprocs, r, rs, rn);
        row_counts[r] = static_cast<int>(rn);
        row_displs[r] = static_cast<int>(rs);
        nnz_counts[r] = static_cast<int>(h_rowDelimiters[rs + rn] - h_rowDelimiters[rs]);
        nnz_displs[r] = static_cast<int>(h_rowDelimiters[rs]);
    }

    // Distribute val and cols via MPI
    std::vector<double> local_val(local_nnz);
    std::vector<index_t> local_cols(local_nnz);

    MPI_Scatterv(h_val.data(), nnz_counts.data(), nnz_displs.data(), MPI_DOUBLE,
                 local_val.data(), static_cast<int>(local_nnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(h_cols.data(), nnz_counts.data(), nnz_displs.data(), MPI_UNSIGNED,
                 local_cols.data(), static_cast<int>(local_nnz), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Build local row delimiters (zero-based for local nnz data)
    std::vector<index_t> local_rowDelimiters(local_nrows + 1);
    #pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= local_nrows; ++i) {
        local_rowDelimiters[i] = h_rowDelimiters[row_start + i] - local_nnz_offset;
    }

    // Select GPU device (round-robin across available GPUs)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    // Allocate and populate GPU memory
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rowDelimiters = nullptr;

    if (local_nnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, local_nnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, local_nnz * sizeof(index_t)));
        CUDA_CHECK(cudaMemcpy(d_val, local_val.data(), local_nnz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, local_cols.data(), local_nnz * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    if (local_nrows > 0) {
        CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (local_nrows + 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&d_out, local_nrows * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_rowDelimiters, local_rowDelimiters.data(),
                              (local_nrows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    // Kernel launch configuration
    constexpr int THREADS_PER_BLOCK = 256;
    int blocks = (static_cast<int>(local_nrows) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Warm up GPU
    if (local_nrows > 0 && blocks > 0) {
        spmv_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_val, d_cols, d_rowDelimiters, d_vec, local_nrows, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed computation
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (local_nrows > 0 && blocks > 0) {
            spmv_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_val, d_cols, d_rowDelimiters,
                                                        d_vec, local_nrows, d_out);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy results back from GPU
    std::vector<double> local_out(local_nrows);
    if (local_nrows > 0) {
        CUDA_CHECK(cudaMemcpy(local_out.data(), d_out, local_nrows * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Gather results on rank 0
    std::vector<double> h_out(numRows);
    MPI_Gatherv(local_out.data(), static_cast<int>(local_nrows), MPI_DOUBLE,
                h_out.data(), row_counts.data(), row_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int result = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                result = 1;
            }
        }
    }

    // Cleanup GPU memory
    if (d_val) cudaFree(d_val);
    if (d_cols) cudaFree(d_cols);
    if (d_rowDelimiters) cudaFree(d_rowDelimiters);
    if (d_vec) cudaFree(d_vec);
    if (d_out) cudaFree(d_out);

    MPI_Finalize();
    return result;
}
