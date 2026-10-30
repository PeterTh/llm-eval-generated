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

void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
    #pragma omp parallel for
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
    bool ok = true;
    #pragma omp parallel for reduction(&&:ok)
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                ok = false;
            }
        } else {
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n", i, ref, res, relError);
                ok = false;
            }
        }
    }
    return ok;
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

__global__ void spmvKernel(const double* val, const index_t* cols, const index_t* rowDelimiters,
                           const double* vec, index_t startRow, index_t numRowsLocal, double* out) {
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numRowsLocal) {
        double t = 0.0;
        for (index_t j = rowDelimiters[startRow + i]; j < rowDelimiters[startRow + i + 1]; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[i] = t;
    }
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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
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

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    MPI_Bcast(h_val.data(), nItems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_cols.data(), nItems, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows, h_reference.data());
    }

    // partition rows
    index_t base = numRows / size;
    index_t rem = numRows % size;
    index_t startRow = (rank < rem) ? rank * (base + 1) : rem * (base + 1) + (rank - rem) * base;
    index_t localRows = (rank < rem) ? base + 1 : base;

    // device
    double *d_val, *d_vec, *d_out;
    index_t *d_cols, *d_rowDelim;
    cudaMalloc(&d_val, nItems * sizeof(double));
    cudaMalloc(&d_cols, nItems * sizeof(index_t));
    cudaMalloc(&d_rowDelim, (numRows + 1) * sizeof(index_t));
    cudaMalloc(&d_vec, numRows * sizeof(double));
    cudaMalloc(&d_out, localRows * sizeof(double));
    cudaMemcpy(d_val, h_val.data(), nItems * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, h_cols.data(), nItems * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowDelim, h_rowDelimiters.data(), (numRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (localRows + threads - 1) / threads;

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvKernel<<<blocks, threads>>>(d_val, d_cols, d_rowDelim, d_vec, startRow, localRows, d_out);
        cudaDeviceSynchronize();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    cudaMemcpy(h_out.data() + startRow, d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost);

    // gather all
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        index_t br = numRows / size;
        index_t rr = numRows % size;
        index_t lr = (r < rr) ? br + 1 : br;
        counts[r] = lr;
        displs[r] = (r < rr) ? r * (br + 1) : rr * (br + 1) + (r - rr) * br;
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_out.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    int rc = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\\n", duration.count());
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

    if (validate) {
            printf("Validating result...\\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\\n");
            } else {
                printf("Validation: FAILED\\n");
                rc = 1;
            }
    }

    }

    MPI_Finalize();
    return (rank == 0) ? rc : 0;
}
