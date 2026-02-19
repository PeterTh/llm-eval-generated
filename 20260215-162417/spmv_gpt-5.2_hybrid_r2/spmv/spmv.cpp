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

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

static inline void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

static inline void checkMpi(int err, const char* what) {
    if (err != MPI_SUCCESS) {
        char buf[MPI_MAX_ERROR_STRING] = {};
        int len = 0;
        MPI_Error_string(err, buf, &len);
        fprintf(stderr, "MPI error (%s): %s\n", what, buf);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

static inline index_t rowStartForRank(index_t numRows, int size, int rank) {
    const index_t base = numRows / static_cast<index_t>(size);
    const index_t rem = numRows % static_cast<index_t>(size);
    return rank < static_cast<int>(rem) ? (base + 1U) * static_cast<index_t>(rank)
                                        : (base + 1U) * rem + base * (static_cast<index_t>(rank) - rem);
}

static inline index_t rowEndForRank(index_t numRows, int size, int rank) {
    return rowStartForRank(numRows, size, rank + 1);
}

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values
//
// NOTE: Kept serial to preserve original rand() stream semantics.
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: initRandomMatrix
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix. The function encodes these positions in compressed sparse
//   row (CSR) format.
// ****************************************************************************
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
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

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format (OpenMP-parallel)
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

__global__ void spmvCsrKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                              const index_t* __restrict__ rowDelims, const double* __restrict__ vec,
                              double* __restrict__ out, index_t numRowsLocal) {
    const index_t row = static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) +
                        static_cast<index_t>(threadIdx.x);
    if (row >= numRowsLocal) return;

    const index_t start = rowDelims[row];
    const index_t end = rowDelims[row + 1];

    double sum = 0.0;
    for (index_t jj = start; jj < end; ++jj) {
        sum += val[jj] * vec[cols[jj]];
    }
    out[row] = sum;
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
    int provided = 0;
    checkMpi(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided), "MPI_Init_thread");

    int rank = 0;
    int size = 1;
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &size), "MPI_Comm_size");

    // Bind each MPI rank to a GPU on the node.
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found.\n");
        MPI_Finalize();
        return 2;
    }
    checkCuda(cudaSetDevice(rank % deviceCount), "cudaSetDevice");

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks see same argv)
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

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP max threads: %d\n", omp_get_max_threads());
        printf("CUDA devices (this node): %d\n", deviceCount);
    }

    const index_t rowStart = rowStartForRank(numRows, size, rank);
    const index_t rowEnd = rowEndForRank(numRows, size, rank);
    const index_t localRows = rowEnd - rowStart;

    // Root initializes the full problem identically to the original code and scatters CSR rows.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec;
    std::vector<double> h_reference;

    std::vector<int> nnzSendCounts;
    std::vector<int> nnzDispls;
    std::vector<int> rowSendCounts;
    std::vector<int> rowDispls;

    if (rank == 0) {
        if (nItems > static_cast<index_t>(std::numeric_limits<int>::max()) ||
            numRows > static_cast<index_t>(std::numeric_limits<int>::max())) {
            fprintf(stderr, "Problem size too large for MPI counts (int).\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_vec.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        nnzSendCounts.resize(size);
        nnzDispls.resize(size);
        rowSendCounts.resize(size);
        rowDispls.resize(size);
        for (int r = 0; r < size; ++r) {
            const index_t rs = rowStartForRank(numRows, size, r);
            const index_t re = rowEndForRank(numRows, size, r);
            const index_t lnnz = h_rowDelimiters[re] - h_rowDelimiters[rs];
            nnzSendCounts[r] = static_cast<int>(lnnz);
            nnzDispls[r] = static_cast<int>(h_rowDelimiters[rs]);
            rowSendCounts[r] = static_cast<int>((re - rs) + 1U);
            rowDispls[r] = static_cast<int>(rs);
        }

        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows,
                    h_reference.data());
        }
    }

    // Broadcast input vector to all ranks.
    if (rank != 0) h_vec.resize(numRows);
    checkMpi(MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD),
             "MPI_Bcast(vec)");

    // Scatter CSR row delimiters (localRows + 1) and then shift to local nnz base.
    std::vector<index_t> h_rowDelimsLocal(localRows + 1);
#if defined(MPI_UINT32_T)
    MPI_Datatype mpi_index_t = MPI_UINT32_T;
#else
    MPI_Datatype mpi_index_t = MPI_UNSIGNED;
#endif

    checkMpi(MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                          rank == 0 ? rowSendCounts.data() : nullptr,
                          rank == 0 ? rowDispls.data() : nullptr, mpi_index_t,
                          h_rowDelimsLocal.data(), static_cast<int>(localRows + 1U), mpi_index_t, 0,
                          MPI_COMM_WORLD),
             "MPI_Scatterv(rowDelims)");

    const index_t baseNnz = h_rowDelimsLocal[0];
    for (index_t i = 0; i < localRows + 1U; ++i) {
        h_rowDelimsLocal[i] -= baseNnz;
    }

    // Determine local nnz and scatter val/cols.
    const index_t localNnz = h_rowDelimsLocal[localRows];
    if (localNnz > static_cast<index_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "Rank %d local nnz too large for MPI counts (int).\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    std::vector<double> h_valLocal(localNnz);
    std::vector<index_t> h_colsLocal(localNnz);

    checkMpi(MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                          rank == 0 ? nnzSendCounts.data() : nullptr,
                          rank == 0 ? nnzDispls.data() : nullptr, MPI_DOUBLE,
                          h_valLocal.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0,
                          MPI_COMM_WORLD),
             "MPI_Scatterv(val)");
    checkMpi(MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                          rank == 0 ? nnzSendCounts.data() : nullptr,
                          rank == 0 ? nnzDispls.data() : nullptr, mpi_index_t,
                          h_colsLocal.data(), static_cast<int>(localNnz), mpi_index_t, 0,
                          MPI_COMM_WORLD),
             "MPI_Scatterv(cols)");

    // Device allocations and copies.
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelims = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    const size_t allocNnz = (localNnz == 0) ? 1 : static_cast<size_t>(localNnz);
    const size_t allocRows = (localRows == 0) ? 1 : static_cast<size_t>(localRows);

    checkCuda(cudaMalloc(&d_val, sizeof(double) * allocNnz), "cudaMalloc(d_val)");
    checkCuda(cudaMalloc(&d_cols, sizeof(index_t) * allocNnz), "cudaMalloc(d_cols)");
    checkCuda(cudaMalloc(&d_rowDelims, sizeof(index_t) * static_cast<size_t>(localRows + 1U)),
              "cudaMalloc(d_rowDelims)");
    checkCuda(cudaMalloc(&d_vec, sizeof(double) * static_cast<size_t>(numRows)), "cudaMalloc(d_vec)");
    checkCuda(cudaMalloc(&d_out, sizeof(double) * allocRows), "cudaMalloc(d_out)");

    checkCuda(cudaMemcpy(d_val, h_valLocal.data(), sizeof(double) * static_cast<size_t>(localNnz),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(val)");
    checkCuda(cudaMemcpy(d_cols, h_colsLocal.data(), sizeof(index_t) * static_cast<size_t>(localNnz),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(cols)");
    checkCuda(cudaMemcpy(d_rowDelims, h_rowDelimsLocal.data(),
                         sizeof(index_t) * static_cast<size_t>(localRows + 1U),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(rowDelims)");
    checkCuda(cudaMemcpy(d_vec, h_vec.data(), sizeof(double) * static_cast<size_t>(numRows),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(vec)");

    // Perform SpMV computation (GPU) and time using CUDA events; use max time across ranks.
    if (rank == 0) printf("Computing SpMV...\n");

    cudaEvent_t evStart{}, evStop{};
    checkCuda(cudaEventCreate(&evStart), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&evStop), "cudaEventCreate(stop)");

    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(before)");

    const int threads = 256;
    const int blocks = (localRows == 0)
                           ? 1
                           : static_cast<int>((localRows + static_cast<index_t>(threads) - 1U) /
                                              static_cast<index_t>(threads));

    checkCuda(cudaEventRecord(evStart), "cudaEventRecord(start)");
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCsrKernel<<<blocks, threads>>>(d_val, d_cols, d_rowDelims, d_vec, d_out, localRows);
    }
    checkCuda(cudaEventRecord(evStop), "cudaEventRecord(stop)");
    checkCuda(cudaEventSynchronize(evStop), "cudaEventSynchronize(stop)");
    checkCuda(cudaGetLastError(), "kernel launch");

    float localMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&localMs, evStart, evStop), "cudaEventElapsedTime");

    double localMsD = static_cast<double>(localMs);
    double maxMs = 0.0;
    checkMpi(MPI_Reduce(&localMsD, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
             "MPI_Reduce(maxTime)");

    // Copy final local output back.
    std::vector<double> h_outLocal(localRows);
    checkCuda(cudaMemcpy(h_outLocal.data(), d_out, sizeof(double) * static_cast<size_t>(localRows),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(out)");

    // Gather output vector to root.
    std::vector<double> h_out;
    std::vector<int> outCounts;
    std::vector<int> outDispls;
    if (rank == 0) {
        h_out.resize(numRows);
        outCounts.resize(size);
        outDispls.resize(size);
        for (int r = 0; r < size; ++r) {
            const index_t rs = rowStartForRank(numRows, size, r);
            const index_t re = rowEndForRank(numRows, size, r);
            outCounts[r] = static_cast<int>(re - rs);
            outDispls[r] = static_cast<int>(rs);
        }
    }

    checkMpi(MPI_Gatherv(h_outLocal.data(), static_cast<int>(localRows), MPI_DOUBLE,
                         rank == 0 ? h_out.data() : nullptr,
                         rank == 0 ? outCounts.data() : nullptr,
                         rank == 0 ? outDispls.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD),
             "MPI_Gatherv(out)");

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxMs);
        const double seconds = maxMs / 1000.0;
        const double gflops = (2.0 * static_cast<double>(nItems) * static_cast<double>(iterations)) / seconds /
                              1e9;
        const double avgTime = maxMs / static_cast<double>(iterations);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");

            checkCuda(cudaFree(d_val), "cudaFree(d_val)");
            checkCuda(cudaFree(d_cols), "cudaFree(d_cols)");
            checkCuda(cudaFree(d_rowDelims), "cudaFree(d_rowDelims)");
            checkCuda(cudaFree(d_vec), "cudaFree(d_vec)");
            checkCuda(cudaFree(d_out), "cudaFree(d_out)");
            checkCuda(cudaEventDestroy(evStart), "cudaEventDestroy(start)");
            checkCuda(cudaEventDestroy(evStop), "cudaEventDestroy(stop)");

            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    checkCuda(cudaFree(d_val), "cudaFree(d_val)");
    checkCuda(cudaFree(d_cols), "cudaFree(d_cols)");
    checkCuda(cudaFree(d_rowDelims), "cudaFree(d_rowDelims)");
    checkCuda(cudaFree(d_vec), "cudaFree(d_vec)");
    checkCuda(cudaFree(d_out), "cudaFree(d_out)");
    checkCuda(cudaEventDestroy(evStart), "cudaEventDestroy(start)");
    checkCuda(cudaEventDestroy(evStop), "cudaEventDestroy(stop)");

    MPI_Finalize();
    return 0;
}
