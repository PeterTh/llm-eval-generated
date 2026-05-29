#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <mpi.h>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// CUDA kernel: SpMV in CSR format
// Each thread block handles one row; threads within a block cooperatively
// accumulate the dot product for that row.
// ****************************************************************************
__global__ void spmvCudaKernel(const double* val, const index_t* cols,
                                const index_t* rowDelimiters,
                                const double* vec, const index_t dim, double* out) {
    const index_t row = blockIdx.x;
    if (row >= dim) return;

    const index_t start = rowDelimiters[row];
    const index_t end   = rowDelimiters[row + 1];
    double sum = 0.0;

    for (index_t j = start + threadIdx.x; j < end; j += blockDim.x) {
        sum += val[j] * vec[cols[j]];
    }

    // Block-level reduction using shared memory
    __shared__ double sdata[256];
    const int tid = threadIdx.x;
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = sdata[0];
    }
}

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values (OpenMP parallelized)
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
#pragma omp parallel for schedule(static)
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
//   Sequential by design (deterministic with fixed seed).
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
//   Computes sparse matrix-vector multiplication using CSR format
//   OpenMP parallelized across rows.
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
//
// Purpose:
//   Verifies correctness of results by comparing to reference solution
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

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments on rank 0
    if (rank == 0) {
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
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast parameters from rank 0 to all ranks
    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    const index_t nItems = (numRows * numRows) / sparsity;

    // Print configuration (rank 0 only)
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Parallelization: Hybrid MPI (%d ranks) + OpenMP + CUDA\n", numRanks);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // -----------------------------------------------------------------------
    // Phase 1: Rank 0 generates the full CSR matrix, then distributes rows.
    // -----------------------------------------------------------------------
    std::vector<double> full_val(nItems);
    std::vector<index_t> full_cols(nItems);
    std::vector<index_t> full_rowDelimiters(numRows + 1);
    std::vector<double> full_vec(numRows);

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(full_vec.data(), numRows, maxVal);
        fill(full_val.data(), nItems, maxVal);
        initRandomMatrix(full_cols.data(), full_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast the full vector (needed by all ranks for SpMV)
    MPI_Bcast(full_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute row distribution: each rank gets a contiguous block of rows.
    // We need to send rowDelimiters to every rank so they know their local
    // offset into the global val/cols arrays.
    // Strategy: broadcast full_rowDelimiters, then each rank computes its
    // local row range and builds local CSR structures.
    MPI_Bcast(full_rowDelimiters.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Compute local row range for this rank
    const index_t rowsPerRank = numRows / numRanks;
    const index_t remainder   = numRows % numRanks;
    index_t localRowStart = rowsPerRank * rank + (rank < static_cast<int>(remainder) ? rank : static_cast<index_t>(remainder));
    index_t localRowEnd;
    if (rank + 1 < numRanks) {
        localRowEnd = rowsPerRank * (rank + 1) + (rank + 1 < static_cast<int>(remainder) ? rank + 1 : static_cast<index_t>(remainder));
    } else {
        localRowEnd = numRows;
    }
    const index_t localRows = localRowEnd - localRowStart;

    // Compute local nnz count
    index_t localNnz = 0;
    for (index_t i = localRowStart; i < localRowEnd; ++i) {
        localNnz += full_rowDelimiters[i + 1] - full_rowDelimiters[i];
    }

    // Allocate local CSR structures
    std::vector<double> local_val(localNnz);
    std::vector<index_t> local_cols(localNnz);
    std::vector<index_t> local_rowDelimiters(localRows + 1);

    // Build local CSR: rows are renumbered 0..localRows-1
    // Rank 0 extracts from its own arrays; other ranks need data from rank 0.
    if (rank == 0) {
        index_t offset = 0;
        local_rowDelimiters[0] = 0;
        for (index_t i = 0; i < localRows; ++i) {
            const index_t globalRow = localRowStart + i;
            const index_t rowStart = full_rowDelimiters[globalRow];
            const index_t rowEnd   = full_rowDelimiters[globalRow + 1];
            for (index_t j = rowStart; j < rowEnd; ++j) {
                local_val[offset] = full_val[j];
                local_cols[offset] = full_cols[j];
                ++offset;
            }
            local_rowDelimiters[i + 1] = offset;
        }
    } else {
        // Receive local data from rank 0 via MPI
        // First, send localNnz to rank 0 so it knows how much to send
        index_t sendNnz = localNnz;
        MPI_Send(&sendNnz, 1, MPI_UINT32_T, 0, rank, MPI_COMM_WORLD);
        MPI_Recv(&localNnz, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Receive val and cols
        MPI_Recv(local_val.data(), localNnz, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_cols.data(), localNnz, MPI_UINT32_T, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_rowDelimiters.data(), localRows + 1, MPI_UINT32_T, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Rank 0 sends data to other ranks
    if (rank == 0) {
        for (int r = 1; r < numRanks; ++r) {
            index_t recvNnz = 0;
            MPI_Recv(&recvNnz, 1, MPI_UINT32_T, r, r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Compute the nnz for rank r
            index_t rStart = rowsPerRank * r + (r < static_cast<int>(remainder) ? r : static_cast<index_t>(remainder));
            index_t rEnd;
            if (r + 1 < numRanks) {
                rEnd = rowsPerRank * (r + 1) + (r + 1 < static_cast<int>(remainder) ? r + 1 : static_cast<index_t>(remainder));
            } else {
                rEnd = numRows;
            }
            const index_t rRows = rEnd - rStart;

            index_t rNnz = 0;
            for (index_t i = rStart; i < rEnd; ++i) {
                rNnz += full_rowDelimiters[i + 1] - full_rowDelimiters[i];
            }

            // Build temporary arrays for rank r
            std::vector<double> r_val(rNnz);
            std::vector<index_t> r_cols(rNnz);
            std::vector<index_t> r_rowDel(rRows + 1);

            index_t offset = 0;
            r_rowDel[0] = 0;
            for (index_t i = 0; i < rRows; ++i) {
                const index_t globalRow = rStart + i;
                const index_t rowStart = full_rowDelimiters[globalRow];
                const index_t rowEnd   = full_rowDelimiters[globalRow + 1];
                for (index_t j = rowStart; j < rowEnd; ++j) {
                    r_val[offset] = full_val[j];
                    r_cols[offset] = full_cols[j];
                    ++offset;
                }
                r_rowDel[i + 1] = offset;
            }

            MPI_Send(&rNnz, 1, MPI_UINT32_T, r, 0, MPI_COMM_WORLD);
            MPI_Send(r_val.data(), rNnz, MPI_DOUBLE, r, r, MPI_COMM_WORLD);
            MPI_Send(r_cols.data(), rNnz, MPI_UINT32_T, r, r, MPI_COMM_WORLD);
            MPI_Send(r_rowDel.data(), rRows + 1, MPI_UINT32_T, r, r, MPI_COMM_WORLD);
        }
    }

    // Barrier to ensure all ranks have local data
    MPI_Barrier(MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // Phase 2: For validation, rank 0 computes the reference on CPU (OpenMP).
    // -----------------------------------------------------------------------
    std::vector<double> h_reference;
    if (rank == 0 && validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(full_val.data(), full_cols.data(), full_rowDelimiters.data(),
                full_vec.data(), numRows, h_reference.data());
    }

    // -----------------------------------------------------------------------
    // Phase 3: CUDA SpMV on each rank.
    // -----------------------------------------------------------------------
    // Allocate device memory
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    cudaMalloc(&d_val, localNnz * sizeof(double));
    cudaMalloc(&d_cols, localNnz * sizeof(index_t));
    cudaMalloc(&d_rowDelimiters, (localRows + 1) * sizeof(index_t));
    cudaMalloc(&d_vec, numRows * sizeof(double));
    cudaMalloc(&d_out, localRows * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_val, local_val.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, local_cols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowDelimiters, local_rowDelimiters.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, full_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice);

    // Local output buffer on host
    std::vector<double> local_out(localRows);

    // CUDA stream for async transfers
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        const int threadsPerBlock = 256;
        const int blocksPerGrid   = static_cast<int>(localRows);
        spmvCudaKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out);
        cudaMemcpyAsync(local_out.data(), d_out, localRows * sizeof(double),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Cleanup CUDA resources
    cudaStreamDestroy(stream);
    cudaFree(d_val);
    cudaFree(d_cols);
    cudaFree(d_rowDelimiters);
    cudaFree(d_vec);
    cudaFree(d_out);

    // -----------------------------------------------------------------------
    // Phase 4: Gather results to rank 0 and compute timing.
    // -----------------------------------------------------------------------
    std::vector<double> h_out(numRows);
    if (rank == 0) {
        // Copy own local results
        std::memcpy(h_out.data() + localRowStart, local_out.data(), localRows * sizeof(double));

        // Receive results from other ranks
        for (int r = 1; r < numRanks; ++r) {
            index_t rStart = rowsPerRank * r + (r < static_cast<int>(remainder) ? r : static_cast<index_t>(remainder));
            index_t rEnd;
            if (r + 1 < numRanks) {
                rEnd = rowsPerRank * (r + 1) + (r + 1 < static_cast<int>(remainder) ? r + 1 : static_cast<index_t>(remainder));
            } else {
                rEnd = numRows;
            }
            const index_t rRows = rEnd - rStart;

            MPI_Recv(h_out.data() + rStart, rRows, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_out.data(), localRows, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
