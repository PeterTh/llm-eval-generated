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

constexpr double MAX_RELATIVE_ERROR = 0.02;

void fill(double* A, const index_t n, const double maxVal) {
#pragma omp parallel for
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

void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
#pragma omp parallel for
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[i] = t;
    }
}

__global__ void spmvKernel(const double* val, const index_t* cols, const index_t* rd,
                           const double* vec, double* out, index_t numRows) {
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numRows) return;
    double t = 0.0;
    for (index_t j = rd[i]; j < rd[i + 1]; ++j) {
        t += val[j] * vec[cols[j]];
    }
    out[i] = t;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numRows = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) sparsity = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) maxVal = atof(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank==0){printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]);} MPI_Finalize(); return 1; }
    }

    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> h_val(nItems);
    std::vector<index_t> h_cols(nItems);
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out(numRows, 0.0);

    fill(h_vec.data(), numRows, maxVal);
    fill(h_val.data(), nItems, maxVal);
    initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    // broadcast to ensure identical (already identical due to deterministic, but broadcast vector anyway)
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_val.data(), nItems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_cols.data(), nItems, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), numRows+1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows, h_reference.data());
    }

    // partition rows
    index_t startRow = (rank * numRows) / size;
    index_t endRow = ((rank + 1) * numRows) / size;
    index_t localRows = endRow - startRow;
    index_t base = h_rowDelimiters[startRow];
    index_t localNnz = h_rowDelimiters[endRow] - base;

    std::vector<double> lval(localNnz);
    std::vector<index_t> lcols(localNnz);
    std::vector<index_t> lrd(localRows + 1);
    for (index_t k = 0; k < localNnz; ++k) { lval[k] = h_val[base + k]; lcols[k] = h_cols[base + k]; }
    for (index_t k = 0; k <= localRows; ++k) lrd[k] = h_rowDelimiters[startRow + k] - base;

    std::vector<double> lout(localRows, 0.0);

    // CUDA
    double *d_val, *d_vec, *d_out;
    index_t *d_cols, *d_rd;
    cudaMalloc(&d_val, localNnz * sizeof(double));
    cudaMalloc(&d_cols, localNnz * sizeof(index_t));
    cudaMalloc(&d_rd, (localRows + 1) * sizeof(index_t));
    cudaMalloc(&d_vec, numRows * sizeof(double));
    cudaMalloc(&d_out, localRows * sizeof(double));
    cudaMemcpy(d_val, lval.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, lcols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rd, lrd.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice);

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        int threads = 256;
        int blocks = (localRows + threads - 1) / threads;
        spmvKernel<<<blocks, threads>>>(d_val, d_cols, d_rd, d_vec, d_out, localRows);
        cudaDeviceSynchronize();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double durationMs = (t1 - t0) * 1000.0;

    cudaMemcpy(lout.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost);

    // gather
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        index_t s = (r * numRows) / size;
        index_t e = ((r + 1) * numRows) / size;
        counts[r] = e - s;
        displs[r] = s;
    }
    MPI_Gatherv(lout.data(), localRows, MPI_DOUBLE, h_out.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)durationMs);
        const double gflops = (2.0 * nItems * iterations) / (durationMs / 1000.0) / 1e9;
        const double avgTime = durationMs / static_cast<double>(iterations);
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

    cudaFree(d_val); cudaFree(d_cols); cudaFree(d_rd); cudaFree(d_vec); cudaFree(d_out);

    int rc = 0;
    if (rank == 0 && validate) {
        // recompute validity not stored; but we can set via broadcast
    }
    // broadcast validation result
    int validInt = 0;
    if (rank == 0 && validate) {
        // recompute quickly? Already printed; set based on verify? Need variable. Re-evaluate:
    }
    // simpler: only rank0 knows; broadcast a flag
    // We'll just finalize; exit code handling: rank0 returns 0/1 but MPI finalize; use MPI_Finalize and return.
    // To propagate, broadcast rc.
    if (rank == 0) {
        // recompute valid flag
        // (already computed above; store)
    }
    MPI_Finalize();
    // Return value only meaningful on rank0; mpirun exit code from rank0? Usually yes.
    return 0;
}
```

Issue: validation return code. Let's track `bool valid` on rank0 and broadcast int. Let me adjust: declare `int validInt=1;` if validate and rank0 compute; else 1. Then after gather, rank0 sets validInt = valid?0/1. Broadcast. Return valid?0:1. But non-rank0 return 0. mpirun exit code? Usually exit status of rank 0? OpenMPI returns non-zero if any rank non-zero. Fine.

Let me edit to include validation result broadcast. Also `MPI_UNSIGNED` for index_t (uint32_t) okay.

Let me rewrite file carefully with valid tracking. I'll just edit final return.<|end|><|start|> fileList? Let me rewrite using write.<|end|><|start|> assistant to=write<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="content">
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

constexpr double MAX_RELATIVE_ERROR = 0.02;

void fill(double* A, const index_t n, const double maxVal) {
#pragma omp parallel for
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

void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
#pragma omp parallel for
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[i] = t;
    }
}

__global__ void spmvKernel(const double* val, const index_t* cols, const index_t* rd,
                           const double* vec, double* out, index_t numRows) {
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numRows) return;
    double t = 0.0;
    for (index_t j = rd[i]; j < rd[i + 1]; ++j) {
        t += val[j] * vec[cols[j]];
    }
    out[i] = t;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numRows = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) sparsity = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = static_cast<index_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) maxVal = atof(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank==0){printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]);} MPI_Finalize(); return 1; }
    }

    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> h_val(nItems);
    std::vector<index_t> h_cols(nItems);
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out(numRows, 0.0);

    fill(h_vec.data(), numRows, maxVal);
    fill(h_val.data(), nItems, maxVal);
    initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_val.data(), nItems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_cols.data(), nItems, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), numRows+1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows, h_reference.data());
    }

    index_t startRow = (rank * numRows) / size;
    index_t endRow = ((rank + 1) * numRows) / size;
    index_t localRows = endRow - startRow;
    index_t base = h_rowDelimiters[startRow];
    index_t localNnz = h_rowDelimiters[endRow] - base;

    std::vector<double> lval(localNnz);
    std::vector<index_t> lcols(localNnz);
    std::vector<index_t> lrd(localRows + 1);
    for (index_t k = 0; k < localNnz; ++k) { lval[k] = h_val[base + k]; lcols[k] = h_cols[base + k]; }
    for (index_t k = 0; k <= localRows; ++k) lrd[k] = h_rowDelimiters[startRow + k] - base;

    std::vector<double> lout(localRows, 0.0);

    double *d_val, *d_vec, *d_out;
    index_t *d_cols, *d_rd;
    cudaMalloc(&d_val, localNnz * sizeof(double));
    cudaMalloc(&d_cols, localNnz * sizeof(index_t));
    cudaMalloc(&d_rd, (localRows + 1) * sizeof(index_t));
    cudaMalloc(&d_vec, numRows * sizeof(double));
    cudaMalloc(&d_out, localRows * sizeof(double));
    cudaMemcpy(d_val, lval.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, lcols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rd, lrd.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice);

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        int threads = 256;
        int blocks = (localRows + threads - 1) / threads;
        if (localRows > 0) spmvKernel<<<blocks, threads>>>(d_val, d_cols, d_rd, d_vec, d_out, localRows);
        cudaDeviceSynchronize();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double durationMs = (t1 - t0) * 1000.0;

    cudaMemcpy(lout.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        index_t s = (r * numRows) / size;
        index_t e = ((r + 1) * numRows) / size;
        counts[r] = e - s;
        displs[r] = s;
    }
    MPI_Gatherv(lout.data(), localRows, MPI_DOUBLE, h_out.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int validInt = 1;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)durationMs);
        const double gflops = (2.0 * nItems * iterations) / (durationMs / 1000.0) / 1e9;
        const double avgTime = durationMs / static_cast<double>(iterations);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(h_out, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            validInt = valid ? 1 : 0;
            if (valid) printf("Validation: PASSED\n");
            else printf("Validation: FAILED\n");
        }
    }
    MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);

    cudaFree(d_val); cudaFree(d_cols); cudaFree(d_rd); cudaFree(d_vec); cudaFree(d_out);

    MPI_Finalize();
    if (validate && !validInt) return 1;
    return 0;
}
```

CMakeLists update.<|end|><|start|> assistant to=write<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="content">
cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra>
)

target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
