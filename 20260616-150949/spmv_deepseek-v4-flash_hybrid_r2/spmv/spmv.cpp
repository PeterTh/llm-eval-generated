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

// CUDA error checking macro
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n",                 \
                    rank, __FILE__, __LINE__, cudaGetErrorString(err));            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Forward declaration of rank variable for CUDA_CHECK macro
static int rank = 0;

// ****************************************************************************
// CUDA Kernel: CSR SpMV (one thread per row)
// ****************************************************************************
__global__ void spmvCsrKernel(const double* __restrict__ val,
                              const index_t* __restrict__ cols,
                              const index_t* __restrict__ rowDelimiters,
                              const double* __restrict__ vec,
                              index_t numRows, double* __restrict__ out) {
    index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows) return;

    double sum = 0.0;
    const index_t rowStart = rowDelimiters[row];
    const index_t rowEnd = rowDelimiters[row + 1];

    for (index_t j = rowStart; j < rowEnd; ++j) {
        sum += val[j] * vec[cols[j]];
    }
    out[row] = sum;
}

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values (serial, deterministic).
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
//   matrix. The function encodes these positions in compressed sparse
//   row (CSR) format. (Serial — called only by rank 0.)
//
// ****************************************************************************
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n,
                      const index_t dim) {
    index_t nnzAssigned = 0;
    const uint64_t dim64 = static_cast<uint64_t>(dim);
    const uint64_t n64 = static_cast<uint64_t>(n);
    double prob =
        static_cast<double>(n64) / (static_cast<double>(dim64) * static_cast<double>(dim64));
    srand(8675309);

    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        const uint64_t i64 = static_cast<uint64_t>(i);
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t j64 = static_cast<uint64_t>(j);
            uint64_t numEntriesLeft = (dim64 * dim64) - ((i64 * dim64) + j64);
            uint64_t needToAssign = n64 - nnzAssigned;
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
// Function: spmvCpu
//
// Purpose:
//   Reference serial CSR SpMV (used for validation on rank 0).
//
// ****************************************************************************
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

// ****************************************************************************
// Function: verifyResults
//
// Purpose:
//   Verifies correctness by comparing to reference solution.
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i,
                       ref, res);
                return false;
            }
        } else {
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e "
                       "(rel error: %.10e)\n",
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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nRanks;
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

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

    // Total non-zero elements (use 64-bit intermediate to avoid overflow)
    const index_t nItems = static_cast<index_t>(
        (static_cast<uint64_t>(numRows) * static_cast<uint64_t>(numRows)) /
        static_cast<uint64_t>(sparsity));

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nRanks);
        printf("Hybrid mode: MPI + OpenMP + CUDA\n");

        // Detect GPUs
        int numGpus = 0;
        cudaGetDeviceCount(&numGpus);
        printf("GPUs detected: %d\n", numGpus);
    }

    // -----------------------------------------------------------------------
    // Row distribution across MPI ranks
    // -----------------------------------------------------------------------
    std::vector<index_t> rowsPerRank(nRanks);
    std::vector<index_t> displs(nRanks);
    {
        index_t offset = 0;
        for (int r = 0; r < nRanks; ++r) {
            rowsPerRank[r] = numRows / nRanks + (static_cast<index_t>(r) < (numRows % nRanks) ? 1 : 0);
            displs[r] = offset;
            offset += rowsPerRank[r];
        }
    }
    const index_t localRows = rowsPerRank[rank];
    const index_t localStart = displs[rank];

    // -----------------------------------------------------------------------
    // Rank 0: allocate and generate the full matrix
    // -----------------------------------------------------------------------
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec;

    if (rank == 0) {
        // Allocate extra room for overlapping row-delimiter distribution
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + static_cast<index_t>(nRanks));
        h_vec.resize(numRows);

        printf("Initializing data structures (rank 0)...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // -----------------------------------------------------------------------
    // Scatter row delimiters (each rank gets localRows+1 entries).
    // Adjacent ranks share one boundary element, so we set the send
    // displacement to the row offset and allocate extra space on the root.
    // -----------------------------------------------------------------------
    std::vector<int> rDelimCounts(nRanks), rDelimDispls(nRanks);
    if (rank == 0) {
        for (int r = 0; r < nRanks; ++r) {
            rDelimCounts[r] = static_cast<int>(rowsPerRank[r] + 1);
            rDelimDispls[r] = static_cast<int>(displs[r]);
        }
    }

    std::vector<index_t> localRowDelimiters(localRows + 1);
    MPI_Datatype mpiIndexT = MPI_UINT32_T;
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr, rDelimCounts.data(),
                 rDelimDispls.data(), mpiIndexT, localRowDelimiters.data(),
                 static_cast<int>(localRows + 1), mpiIndexT, 0, MPI_COMM_WORLD);

    // Number of non-zeros for this rank's rows
    const index_t localNnz =
        localRowDelimiters[localRows] - localRowDelimiters[0];
    const index_t baseOffset = localRowDelimiters[0];

    // Adjust row delimiters to be local (zero-based)
    for (auto& d : localRowDelimiters) d -= baseOffset;

    // -----------------------------------------------------------------------
    // Scatter values and column indices
    // -----------------------------------------------------------------------
    std::vector<int> valCounts(nRanks), valDispls(nRanks);
    if (rank == 0) {
        for (int r = 0; r < nRanks; ++r) {
            const index_t rEnd = displs[r] + rowsPerRank[r];
            const index_t rNnz  = h_rowDelimiters[rEnd] - h_rowDelimiters[displs[r]];
            valCounts[r] = static_cast<int>(rNnz);
            valDispls[r] = static_cast<int>(h_rowDelimiters[displs[r]]);
        }
    }

    std::vector<double> localVal(localNnz);
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, valCounts.data(),
                 valDispls.data(), MPI_DOUBLE, localVal.data(),
                 static_cast<int>(localNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<index_t> localCols(localNnz);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, valCounts.data(),
                 valDispls.data(), mpiIndexT, localCols.data(),
                 static_cast<int>(localNnz), mpiIndexT, 0, MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // Broadcast the full dense vector to all ranks
    // -----------------------------------------------------------------------
    std::vector<double> localVec;
    if (rank == 0) {
        localVec = h_vec;
    } else {
        localVec.resize(numRows);
    }
    MPI_Bcast(localVec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // Validation: rank 0 computes reference solution on CPU before timing
    // -----------------------------------------------------------------------
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution on rank 0...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // -----------------------------------------------------------------------
    // CUDA setup: select GPU and allocate device memory
    // -----------------------------------------------------------------------
    int numGpus = 0;
    cudaGetDeviceCount(&numGpus);
    if (numGpus == 0) {
        fprintf(stderr, "Rank %d: No CUDA-capable devices found!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % numGpus));

    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rowDelimiters = nullptr;

    if (localNnz > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, localNnz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, localNnz * sizeof(index_t)));
    }
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(&d_out, localRows * sizeof(double)));
    }

    // Copy data to device
    if (localNnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_val, localVal.data(), localNnz * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, localCols.data(), localNnz * sizeof(index_t),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, localRowDelimiters.data(),
                          (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, localVec.data(), numRows * sizeof(double),
                          cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // GPU warm-up kernel launch
    // -----------------------------------------------------------------------
    if (rank == 0) printf("Warming up GPU...\n");
    const int blockSize = 256;
    const int gridSize = (localRows + blockSize - 1) / blockSize;
    if (localRows > 0) {
        spmvCsrKernel<<<gridSize, blockSize>>>(
            d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // -----------------------------------------------------------------------
    // Timed SpMV iterations on GPU
    // -----------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing SpMV on GPU...\n");

    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows > 0) {
            spmvCsrKernel<<<gridSize, blockSize>>>(
                d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDurationMs = duration.count();
    long globalDurationMs = 0;
    MPI_Reduce(&localDurationMs, &globalDurationMs, 1, MPI_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // Copy result back from device
    // -----------------------------------------------------------------------
    std::vector<double> localOut(localRows);
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(localOut.data(), d_out, localRows * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    // -----------------------------------------------------------------------
    // Gather all results to rank 0
    // -----------------------------------------------------------------------
    std::vector<double> h_out;
    std::vector<int> recvCounts(nRanks), recvDispls(nRanks);
    if (rank == 0) {
        h_out.resize(numRows);
        int cum = 0;
        for (int r = 0; r < nRanks; ++r) {
            recvCounts[r] = static_cast<int>(rowsPerRank[r]);
            recvDispls[r] = cum;
            cum += recvCounts[r];
        }
    }

    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr, recvCounts.data(),
                recvDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // Rank 0: report and validate
    // -----------------------------------------------------------------------
    if (rank == 0) {
        printf("Computation time: %ld ms\n", globalDurationMs);

        const double gflops =
            (2.0 * nItems * iterations) / (globalDurationMs / 1000.0) / 1e9;
        const double avgTime = globalDurationMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid =
                verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    if (localNnz > 0) {
        CUDA_CHECK(cudaFree(d_val));
        CUDA_CHECK(cudaFree(d_cols));
    }
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    if (localRows > 0) CUDA_CHECK(cudaFree(d_out));

    MPI_Finalize();
    return 0;
}
