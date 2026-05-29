#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// MPI headers
#include <mpi.h>

// CUDA headers
#include <cuda_runtime.h>

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// CUDA kernel: SpMV in CSR format
// Each block handles one row; threads within a block reduce partial sums.
// All arrays (val, cols, rowDelimiters) are local to this rank's partition.
// vec is the full global vector.
__global__ void spmvCudaKernel(const double* val, const index_t* cols,
                                const index_t* rowDelimiters, const double* vec,
                                const index_t localRowCount,
                                double* out) {
    const index_t tid = threadIdx.x;
    const index_t bid = blockIdx.x;

    const index_t localRow = bid;
    if (localRow >= localRowCount) return;

    const index_t rowStart = rowDelimiters[localRow];
    const index_t rowEnd   = rowDelimiters[localRow + 1];
    const index_t nnz      = rowEnd - rowStart;

    // Shared memory for partial reduction
    extern __shared__ double sdata[];

    double sum = 0.0;
    // Coarse-grained reduction: each thread processes multiple elements
    for (index_t k = tid; k < nnz; k += blockDim.x) {
        sum += val[rowStart + k] * vec[cols[rowStart + k]];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Tree-based reduction in shared memory
    for (index_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result
    if (tid == 0) {
        out[localRow] = sdata[0];
    }
}

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values
//
// Arguments:
//   A: pointer to the array to initialize
//   n: number of elements in the array
//   maxVal: specifies range of random values [0, maxVal]
//
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
//
// Arguments:
//   cols:          array for column indexes of elements (size should be = n)
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   n:             number of nonzero elements
//   dim:           number of rows/columns in the matrix
//
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
                // Assign (i,j) a value
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    // Convention: put the number of non-zeroes at the end of the row delimiters array
    rowDelimiters[dim] = n;
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format with OpenMP
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   vec: dense vector of size dim to be used for multiplication
//   dim: number of rows/columns in the matrix
//   out: output - result from the spmv calculation
//
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
//
// Arguments:
//   reference: array holding the reference result vector
//   result: array holding the result vector to verify
//   size: number of elements per vector
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    bool failed = false;
#pragma omp parallel for schedule(static) reduction(||:failed)
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                if (!failed) {
                    printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                }
                failed = true;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                if (!failed) {
                    printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                           i, ref, res, relError);
                }
                failed = true;
            }
        }
    }
    return !failed;
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
    int mpiSize, mpiRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    // Detect available GPUs per rank
    int cudaDeviceCount = 0;
    cudaGetDeviceCount(&cudaDeviceCount);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse, but only rank 0 prints)
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
            if (mpiRank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (mpiRank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Hybrid: MPI (%d ranks) + CUDA (%d GPUs) + OpenMP\n", mpiSize, cudaDeviceCount);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate and initialize data structures on rank 0
    std::vector<double> h_val(nItems);
    std::vector<index_t> h_cols(nItems);
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out(numRows, 0.0);

    if (mpiRank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // For validation, compute reference solution on rank 0
    std::vector<double> h_reference;
    if (validate && mpiRank == 0) {
        printf("Computing reference solution (OpenMP)...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // ---- MPI partitioning ----
    // Distribute rows across MPI ranks (contiguous partitioning)
    const index_t rowsPerRank = numRows / static_cast<index_t>(mpiSize);
    const index_t remainder   = numRows % static_cast<index_t>(mpiSize);
    const index_t myRowStart  = rowsPerRank * static_cast<index_t>(mpiRank) + static_cast<index_t>(mpiRank < static_cast<int>(remainder) ? mpiRank : remainder);
    const index_t myRowEnd    = rowsPerRank * static_cast<index_t>(mpiRank + 1) + static_cast<index_t>(mpiRank + 1 < static_cast<int>(remainder) ? mpiRank + 1 : remainder);
    const index_t myRowCount  = myRowEnd - myRowStart;

    // Broadcast row delimiters from rank 0 so all ranks can compute their partition sizes
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Number of local nnz entries (from rowDelimiters)
    const index_t myNnzStart = h_rowDelimiters[myRowStart];
    const index_t myNnzEnd   = h_rowDelimiters[myRowEnd];
    const index_t myNnz      = myNnzEnd - myNnzStart;

    // Local CSR data on each rank
    std::vector<double>  myVal(myNnz);
    std::vector<index_t>  myCols(myNnz);
    std::vector<index_t>  myRowDelimiters(myRowCount + 1);

    // Scatter val from rank 0
    // Build send counts (all ranks can compute these now)
    std::vector<int> sendcounts(mpiSize);
    std::vector<int> displs(mpiSize);
    for (int r = 0; r < mpiSize; ++r) {
        const index_t rStart = rowsPerRank * static_cast<index_t>(r) + static_cast<index_t>(r < static_cast<int>(remainder) ? r : remainder);
        const index_t rEnd   = rowsPerRank * static_cast<index_t>(r + 1) + static_cast<index_t>((r + 1) < static_cast<int>(remainder) ? r + 1 : remainder);
        sendcounts[r] = static_cast<int>(h_rowDelimiters[rEnd] - h_rowDelimiters[rStart]);
        displs[r]     = static_cast<int>(h_rowDelimiters[rStart]);
    }

    MPI_Scatterv(h_val.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 myVal.data(), static_cast<int>(myNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter cols
    MPI_Scatterv(h_cols.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 myCols.data(), static_cast<int>(myNnz), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Scatter local row delimiters
    std::vector<int> rdCounts(mpiSize);
    std::vector<int> rdDispls(mpiSize);
    for (int r = 0; r < mpiSize; ++r) {
        const index_t rStart = rowsPerRank * static_cast<index_t>(r) + static_cast<index_t>(r < static_cast<int>(remainder) ? r : remainder);
        const index_t rEnd   = rowsPerRank * static_cast<index_t>(r + 1) + static_cast<index_t>((r + 1) < static_cast<int>(remainder) ? r + 1 : remainder);
        rdCounts[r]   = static_cast<int>((rEnd - rStart) + 1);
        rdDispls[r]   = static_cast<int>(rStart);
    }
    MPI_Scatterv(h_rowDelimiters.data(), rdCounts.data(), rdDispls.data(), MPI_UNSIGNED,
                 myRowDelimiters.data(), static_cast<int>(myRowCount + 1), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Convert global row delimiter indices to local indices
    for (index_t i = 0; i <= myRowCount; ++i) {
        myRowDelimiters[i] -= myNnzStart;
    }

    // Broadcast the full input vector to all ranks
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ---- CUDA setup ----
    // Select GPU: round-robin assignment across ranks
    int deviceId = mpiRank % cudaDeviceCount;
    cudaSetDevice(deviceId);

    // Allocate device memory
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    if (myNnz > 0) {
        cudaMalloc(&d_val, myNnz * sizeof(double));
        cudaMemcpy(d_val, myVal.data(), myNnz * sizeof(double), cudaMemcpyHostToDevice);
    }
    if (myNnz > 0) {
        cudaMalloc(&d_cols, myNnz * sizeof(index_t));
        cudaMemcpy(d_cols, myCols.data(), myNnz * sizeof(index_t), cudaMemcpyHostToDevice);
    }
    {
        cudaMalloc(&d_rowDelimiters, (myRowCount + 1) * sizeof(index_t));
        cudaMemcpy(d_rowDelimiters, myRowDelimiters.data(), (myRowCount + 1) * sizeof(index_t), cudaMemcpyHostToDevice);
    }
    {
        cudaMalloc(&d_vec, numRows * sizeof(double));
        cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice);
    }
    {
        cudaMalloc(&d_out, myRowCount * sizeof(double));
    }

    // Local output buffer on host
    std::vector<double> myOut(myRowCount, 0.0);

    // CUDA kernel launch parameters
    const index_t threadsPerBlock = 256;
    const index_t blocksPerGrid   = (myRowCount > 0) ? myRowCount : 1;
    const size_t sharedMemSize    = threadsPerBlock * sizeof(double);

    // Warm-up iteration
    if (myRowCount > 0) {
        spmvCudaKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_val, d_cols, d_rowDelimiters, d_vec, myRowCount, d_out);
        cudaDeviceSynchronize();
    }

    // ---- Benchmark ----
    if (mpiRank == 0) {
        printf("Computing SpMV (Hybrid: MPI+CUDA+OpenMP)...\n");
    }

    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (myRowCount > 0) {
            spmvCudaKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
                d_val, d_cols, d_rowDelimiters, d_vec, myRowCount, d_out);
        }
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy result from GPU to host
    if (myRowCount > 0) {
        cudaMemcpy(myOut.data(), d_out, myRowCount * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // Gather results back to rank 0
    if (mpiRank == 0) {
        std::vector<int> gatherCounts(mpiSize);
        std::vector<int> gatherDispls(mpiSize);
        for (int r = 0; r < mpiSize; ++r) {
            const index_t rStart = rowsPerRank * static_cast<index_t>(r) + static_cast<index_t>(r < static_cast<int>(remainder) ? r : remainder);
            const index_t rEnd   = rowsPerRank * static_cast<index_t>(r + 1) + static_cast<index_t>((r + 1) < static_cast<int>(remainder) ? r + 1 : remainder);
            gatherCounts[r]   = static_cast<int>(rEnd - rStart);
            gatherDispls[r]   = static_cast<int>(rStart);
        }
        // Collect all local outputs into h_out
        std::vector<double> allOuts(numRows);
        // Gather from rank 0 itself
        for (index_t i = 0; i < myRowCount; ++i) {
            allOuts[myRowStart + i] = myOut[i];
        }
        MPI_Gatherv(myOut.data(), static_cast<int>(myRowCount), MPI_DOUBLE,
                    allOuts.data(), gatherCounts.data(), gatherDispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        // Copy gathered results
        for (index_t i = 0; i < numRows; ++i) {
            h_out[i] = allOuts[i];
        }
    } else {
        MPI_Gatherv(myOut.data(), static_cast<int>(myRowCount), MPI_DOUBLE,
                    NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Performance reporting (all ranks contribute timing, rank 0 reports)
    double localDurationMs = duration.count();
    double globalDurationMs;
    MPI_Reduce(&localDurationMs, &globalDurationMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mpiRank == 0) {
        const double gflops = (2.0 * nItems * iterations) / (globalDurationMs / 1000.0) / 1e9;
        const double avgTime = globalDurationMs / static_cast<double>(iterations);

        printf("Computation time: %.1f ms\n", globalDurationMs);
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
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // Cleanup CUDA
    if (d_val)           cudaFree(d_val);
    if (d_cols)          cudaFree(d_cols);
    if (d_rowDelimiters) cudaFree(d_rowDelimiters);
    if (d_vec)           cudaFree(d_vec);
    if (d_out)           cudaFree(d_out);

    MPI_Finalize();

    return 0;
}
