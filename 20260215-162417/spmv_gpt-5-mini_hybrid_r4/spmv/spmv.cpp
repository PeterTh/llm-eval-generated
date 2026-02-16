#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// Optionally include CUDA headers and runtime symbols
#if USE_CUDA
#include <cuda_runtime.h>
#endif

// CPU utility functions (unchanged semantics)
void fill(double* A, const index_t n, const double maxVal) {
    // Parallelize initialization with OpenMP
    #pragma omp parallel for schedule(static)
    for (index_t i = 0; i < n; ++i) {
        unsigned int seed = 8675309u + static_cast<unsigned int>(i);
        A[i] = maxVal * (rand_r(&seed) / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            bool fillRemaining = (numEntriesLeft <= needToAssign);
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    rowDelimiters[dim] = n;
}

void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
    // Use OpenMP for intra-node parallelism
    #pragma omp parallel for schedule(dynamic)
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

#if USE_CUDA
// CUDA kernel: one thread per row (local row index)
extern "C" __global__ void spmv_csr_kernel(const double* __restrict__ d_val,
                                            const index_t* __restrict__ d_cols,
                                            const index_t* __restrict__ d_rowDelimiters,
                                            const double* __restrict__ d_vec,
                                            double* __restrict__ d_out,
                                            index_t localRows) {
    index_t rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= localRows) return;
    index_t rowStart = d_rowDelimiters[rid];
    index_t rowEnd = d_rowDelimiters[rid + 1];
    double sum = 0.0;
    for (index_t jj = rowStart; jj < rowEnd; ++jj) {
        sum += d_val[jj] * d_vec[d_cols[jj]];
    }
    d_out[rid] = sum;
}
#endif

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only on rank 0, then broadcast)
    if (world_rank == 0) {
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
                if (world_rank == 0) printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            }
        }
    }
    // Broadcast configuration to all ranks
    MPI_Bcast(&numRows, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int v = validate ? 1 : 0;
    int r = printResults ? 1 : 0;
    MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&r, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (v != 0);
    printResults = (r != 0);

    const index_t nItems = (numRows * numRows) / sparsity;

    if (world_rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Root prepares full CSR and vector
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);

    if (world_rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        printf("Initializing data structures on rank 0...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast vector to all ranks
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Determine row distribution
    std::vector<int> rowsPerRank(world_size, 0);
    for (int rnk = 0; rnk < world_size; ++rnk) {
        rowsPerRank[rnk] = numRows / world_size + (rnk < (numRows % world_size) ? 1 : 0);
    }
    std::vector<int> rowOffset(world_size, 0);
    for (int i = 1; i < world_size; ++i) rowOffset[i] = rowOffset[i-1] + rowsPerRank[i-1];

    int localRows = rowsPerRank[world_rank];
    int localRowStart = rowOffset[world_rank];

    // Prepare sendcounts and displacements for nnz per rank
    std::vector<int> nnzPerRank(world_size, 0);
    std::vector<int> nnzOffset(world_size, 0);
    if (world_rank == 0) {
        for (int rnk = 0; rnk < world_size; ++rnk) {
            int rs = rowOffset[rnk];
            int re = rs + rowsPerRank[rnk];
            nnzPerRank[rnk] = static_cast<int>(h_rowDelimiters[re] - h_rowDelimiters[rs]);
            if (rnk > 0) nnzOffset[rnk] = nnzOffset[rnk-1] + nnzPerRank[rnk-1];
        }
    }

    // Scatter nnz counts to all ranks
    MPI_Scatter(nnzPerRank.data(), 1, MPI_INT, MPI_IN_PLACE, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Now local nnz
    int localNnz = 0;
    if (world_rank == 0) localNnz = nnzPerRank[0];
    MPI_Bcast(&localNnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate local CSR arrays
    std::vector<double> loc_val(localNnz);
    std::vector<index_t> loc_cols(localNnz);
    std::vector<index_t> loc_rowDelimiters(localRows + 1);

    // Prepare recvcounts and displs for values and cols
    std::vector<int> recvcounts(world_size,0), displs(world_size,0);
    if (world_rank == 0) {
        for (int i = 0; i < world_size; ++i) { recvcounts[i] = nnzPerRank[i]; displs[i] = nnzOffset[i]; }
    }

    // Scatter values and cols
    MPI_Scatterv(h_val.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                 loc_val.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(h_cols.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED,
                 loc_cols.data(), localNnz, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Prepare and scatter adjusted row delimiters (each rank gets local rows+1 entries)
    if (world_rank == 0) {
        for (int rnk = 0; rnk < world_size; ++rnk) {
            int rs = rowOffset[rnk];
            int rr = rowsPerRank[rnk];
            // copy and adjust
            for (int i = 0; i <= rr; ++i) {
                loc_rowDelimiters[i] = static_cast<index_t>(h_rowDelimiters[rs + i] - h_rowDelimiters[rs]);
            }
            if (rnk == 0) {
                // already stored in loc_rowDelimiters for rank 0
            } else {
                MPI_Send(loc_rowDelimiters.data(), rr + 1, MPI_UNSIGNED, rnk, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(loc_rowDelimiters.data(), localRows + 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Allocate local output
    std::vector<double> loc_out(localRows);

    // For validation, root computes reference serially
    std::vector<double> h_reference;
    if (validate && world_rank == 0) {
        h_reference.resize(numRows);
        // Compute on CPU using multithreaded spmvCpu for accuracy
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows, h_reference.data());
    }

    #if USE_CUDA
    // Choose device based on rank to avoid oversubscription
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    int device = world_rank % (deviceCount > 0 ? deviceCount : 1);
    cudaSetDevice(device);

    // Copy local CSR to device
    double* d_val = nullptr; index_t* d_cols = nullptr; index_t* d_rowDel = nullptr; double* d_vec = nullptr; double* d_out = nullptr;
    if (localNnz > 0) cudaMalloc(&d_val, sizeof(double) * localNnz);
    if (localNnz > 0) cudaMalloc(&d_cols, sizeof(index_t) * localNnz);
    cudaMalloc(&d_rowDel, sizeof(index_t) * (localRows + 1));
    cudaMalloc(&d_vec, sizeof(double) * numRows);
    cudaMalloc(&d_out, sizeof(double) * localRows);

    if (localNnz > 0) cudaMemcpy(d_val, loc_val.data(), sizeof(double) * localNnz, cudaMemcpyHostToDevice);
    if (localNnz > 0) cudaMemcpy(d_cols, loc_cols.data(), sizeof(index_t) * localNnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowDel, loc_rowDelimiters.data(), sizeof(index_t) * (localRows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec.data(), sizeof(double) * numRows, cudaMemcpyHostToDevice);

    // Warm-up kernel
    int threadsPerBlock = 128;
    int numBlocks = (localRows + threadsPerBlock - 1) / threadsPerBlock;
    spmv_csr_kernel<<<numBlocks, threadsPerBlock>>>(d_val, d_cols, d_rowDel, d_vec, d_out, localRows);
    cudaDeviceSynchronize();

    // Timed iterations (all ranks do the same number of iterations)
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmv_csr_kernel<<<numBlocks, threadsPerBlock>>>(d_val, d_cols, d_rowDel, d_vec, d_out, localRows);
    }
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // Copy back local output
    cudaMemcpy(loc_out.data(), d_out, sizeof(double) * localRows, cudaMemcpyDeviceToHost);

    // Cleanup device memory
    if (d_val) cudaFree(d_val);
    if (d_cols) cudaFree(d_cols);
    if (d_rowDel) cudaFree(d_rowDel);
    if (d_vec) cudaFree(d_vec);
    if (d_out) cudaFree(d_out);
    #else
    // CPU fallback: use OpenMP-accelerated spmv for local rows
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(loc_val.data(), loc_cols.data(), loc_rowDelimiters.data(), h_vec.data(), localRows, loc_out.data());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    #endif

    // Gather outputs to root
    std::vector<int> recvrows(world_size,0), recvdispl(world_size,0);
    if (world_rank == 0) {
        for (int i = 0; i < world_size; ++i) { recvrows[i] = rowsPerRank[i]; recvdispl[i] = rowOffset[i]; }
    }
    std::vector<double> h_out;
    if (world_rank == 0) h_out.resize(numRows);
    MPI_Gatherv(loc_out.data(), localRows, MPI_DOUBLE,
                h_out.data(), recvrows.data(), recvdispl.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) print_results(h_out, "OutputVector");

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) { printf("Validation: PASSED\n"); }
            else { printf("Validation: FAILED\n"); }
        }
    }

    MPI_Finalize();
    return 0;
}
