#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// ****************************************************************************
// Kernel: spmvCsrKernel
//
// Purpose:
//   CUDA kernel for CSR-format SpMV — each thread processes one row
//
// ****************************************************************************
__global__ void spmvCsrKernel(const double* val, const index_t* cols,
                               const index_t* rowDelimiters,
                               const double* vec, index_t numRows, double* out) {
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numRows) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values (serial — rand is not thread-safe)
//
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
//   matrix using CSR format.
//
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
//   OpenMP-parallel CPU reference SpMV using CSR format
//
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
#ifndef __CUDACC__
    #pragma omp parallel for
#endif
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
//
// Purpose:
//   Verifies correctness by comparing to reference solution
//
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

// ****************************************************************************
// Function: main
//
// Purpose:
//   Hybrid MPI + OpenMP + CUDA SpMV benchmark.
//   - MPI: distributes rows across ranks (block distribution)
//   - OpenMP: used in CPU reference computation
//   - CUDA: each rank computes its local rows on GPU
//
// ****************************************************************************
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int numRanks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const index_t nItems = (numRows * numRows) / sparsity;

    // ---- Row distribution across MPI ranks (block distribution) ----
    const index_t rowsPerRank = numRows / numRanks;
    const index_t rem = numRows % numRanks;

    std::vector<index_t> rankRowStart(numRanks), rankRowCount(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        rankRowCount[r] = rowsPerRank + (static_cast<index_t>(r) < rem ? 1 : 0);
        rankRowStart[r] = (r == 0) ? 0 : (rankRowStart[r - 1] + rankRowCount[r - 1]);
    }

    const index_t localRowCount = rankRowCount[rank];

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
        printf("OpenMP threads available: %d\n", omp_get_max_threads());

        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        printf("CUDA devices available: %d\n", deviceCount);
    }

    // ---- Rank 0 generates the full matrix ----
    std::vector<double> fullVal, fullVec;
    std::vector<index_t> fullCols, fullRowDelimiters;

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fullVal.resize(nItems);
        fullCols.resize(nItems);
        fullRowDelimiters.resize(numRows + 1);
        fullVec.resize(numRows);

        fill(fullVec.data(), numRows, maxVal);
        fill(fullVal.data(), nItems, maxVal);
        initRandomMatrix(fullCols.data(), fullRowDelimiters.data(), nItems, numRows);
    }

    // ---- Scatter / gather metadata ----
    std::vector<int> rowSendCounts(numRanks, 0), rowSendDispls(numRanks, 0);
    std::vector<int> valSendCounts(numRanks, 0), valSendDispls(numRanks, 0);
    std::vector<int> gatherCounts(numRanks, 0), gatherDispls(numRanks, 0);

    if (rank == 0) {
        index_t rowDisp = 0;
        for (int r = 0; r < numRanks; ++r) {
            const index_t rStart = rankRowStart[r];
            const index_t rCount = rankRowCount[r];
            const index_t rEnd = rStart + rCount;

            rowSendCounts[r] = static_cast<int>(rCount + 1);
            rowSendDispls[r] = static_cast<int>(rowDisp);
            rowDisp += rCount + 1;

            const index_t rNnz = fullRowDelimiters[rEnd] - fullRowDelimiters[rStart];
            valSendCounts[r] = static_cast<int>(rNnz);
            valSendDispls[r] = static_cast<int>(fullRowDelimiters[rStart]);

            gatherCounts[r] = static_cast<int>(rCount);
            gatherDispls[r] = static_cast<int>(rStart);
        }
    }

    MPI_Bcast(rowSendCounts.data(), numRanks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rowSendDispls.data(), numRanks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(valSendCounts.data(), numRanks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(valSendDispls.data(), numRanks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(gatherCounts.data(), numRanks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(gatherDispls.data(), numRanks, MPI_INT, 0, MPI_COMM_WORLD);

    const index_t localNnz = static_cast<index_t>(valSendCounts[rank]);

    // ---- Scatter row delimiters (adjusted for local indexing) ----
    std::vector<index_t> localRowDelimiters(localRowCount + 1);

    if (rank == 0) {
        const index_t totalRowSend = static_cast<index_t>(
            rowSendDispls[numRanks - 1] + rowSendCounts[numRanks - 1]);
        std::vector<index_t> rowDelSendBuf(totalRowSend);

        for (int r = 0; r < numRanks; ++r) {
            const index_t rStart = rankRowStart[r];
            const index_t rCount = rankRowCount[r];
            const index_t base = fullRowDelimiters[rStart];
            for (index_t i = 0; i <= rCount; ++i) {
                rowDelSendBuf[rowSendDispls[r] + i] = fullRowDelimiters[rStart + i] - base;
            }
        }

        MPI_Scatterv(rowDelSendBuf.data(), rowSendCounts.data(), rowSendDispls.data(),
                     MPI_UINT32_T,
                     localRowDelimiters.data(), static_cast<int>(localRowCount + 1),
                     MPI_UINT32_T, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UINT32_T,
                     localRowDelimiters.data(), static_cast<int>(localRowCount + 1),
                     MPI_UINT32_T, 0, MPI_COMM_WORLD);
    }

    // ---- Scatter values and column indices ----
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);

    if (rank == 0) {
        MPI_Scatterv(fullVal.data(), valSendCounts.data(), valSendDispls.data(),
                     MPI_DOUBLE,
                     localVal.data(), static_cast<int>(localNnz),
                     MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(fullCols.data(), valSendCounts.data(), valSendDispls.data(),
                     MPI_UINT32_T,
                     localCols.data(), static_cast<int>(localNnz),
                     MPI_UINT32_T, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                     localVal.data(), static_cast<int>(localNnz),
                     MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UINT32_T,
                     localCols.data(), static_cast<int>(localNnz),
                     MPI_UINT32_T, 0, MPI_COMM_WORLD);
    }

    // ---- Broadcast the dense vector (all ranks need the full vector) ----
    std::vector<double> h_vec;
    if (rank == 0) {
        h_vec = std::move(fullVec);
    } else {
        h_vec.resize(numRows);
    }
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ---- CUDA device setup ----
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    // ---- Allocate GPU memory ----
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    if (localNnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, localNnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, localNnz * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (localRowCount + 1) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&d_out, localRowCount * sizeof(double)));
    }
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));

    // ---- Copy data from host to device ----
    if (localNnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_val, localVal.data(), localNnz * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, localCols.data(), localNnz * sizeof(index_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rowDelimiters, localRowDelimiters.data(),
                              (localRowCount + 1) * sizeof(index_t),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double),
                          cudaMemcpyHostToDevice));

    // ---- CUDA kernel launch configuration ----
    constexpr int blockSize = 256;
    const int gridSize = (localRowCount > 0)
                             ? static_cast<int>((localRowCount + blockSize - 1) / blockSize)
                             : 0;

    // ---- Synchronise ranks before benchmarking ----
    MPI_Barrier(MPI_COMM_WORLD);

    // ---- Benchmark: iterated GPU SpMV ----
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRowCount > 0) {
            spmvCsrKernel<<<gridSize, blockSize>>>(d_val, d_cols, d_rowDelimiters,
                                                    d_vec, localRowCount, d_out);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();

    // ---- Copy results back to host ----
    std::vector<double> localOut(localRowCount, 0.0);
    if (localRowCount > 0) {
        CUDA_CHECK(cudaMemcpy(localOut.data(), d_out, localRowCount * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    // ---- Gather results on rank 0 ----
    std::vector<double> h_out;
    if (rank == 0) {
        h_out.resize(numRows);
    }

    MPI_Gatherv(localOut.data(), static_cast<int>(localRowCount), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr,
                gatherCounts.data(), gatherDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // ---- Rank 0: report and validate ----
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        printf("Computation time: %ld ms\n", duration.count());

        const double elapsedSec = static_cast<double>(duration.count()) / 1000.0;
        const double gflops =
            (2.0 * static_cast<double>(nItems) * static_cast<double>(iterations)) /
            elapsedSec / 1e9;
        const double avgTime =
            static_cast<double>(duration.count()) / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Computing reference solution...\n");
            std::vector<double> h_reference(numRows);
            spmvCpu(fullVal.data(), fullCols.data(), fullRowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());

            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                CUDA_CHECK(cudaFree(d_vec));
                if (localNnz > 0) {
                    CUDA_CHECK(cudaFree(d_val));
                    CUDA_CHECK(cudaFree(d_cols));
                    CUDA_CHECK(cudaFree(d_rowDelimiters));
                    CUDA_CHECK(cudaFree(d_out));
                }
                MPI_Finalize();
                return 1;
            }
        }
    }

    // ---- Cleanup ----
    CUDA_CHECK(cudaFree(d_vec));
    if (localNnz > 0) {
        CUDA_CHECK(cudaFree(d_val));
        CUDA_CHECK(cudaFree(d_cols));
        CUDA_CHECK(cudaFree(d_rowDelimiters));
        CUDA_CHECK(cudaFree(d_out));
    }

    MPI_Finalize();
    return 0;
}
