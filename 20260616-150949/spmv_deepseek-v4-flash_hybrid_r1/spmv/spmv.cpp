#include <chrono>
#include <cmath>
#include <cstdint>
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
constexpr int BLOCK_DIM = 256;

// ****************************************************************************
// CUDA error checking macro
// ****************************************************************************
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        MPI_Abort(MPI_COMM_WORLD, code);
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
//   Computes sparse matrix-vector multiplication using CSR format
//   Used for reference solution computation.
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
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                return false;
            }
        } else {
            // Check relative error
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

// ****************************************************************************
// CUDA Kernel: CSR SpMV
//
// Purpose:
//   Computes CSR SpMV on GPU with one thread per row.
//   Uses local (zero-based) rowDelimiters referencing local val/cols arrays.
//
// Arguments:
//   d_val:  local non-zero values
//   d_cols: local column indices
//   d_rowDelimiters: local row delimiters (relative to d_val/d_cols start)
//   d_vec:  full dense vector (broadcast to all ranks)
//   numRows: number of local rows
//   d_out:  local output vector
// ****************************************************************************
__global__ void spmvCsrKernel(
    const double* __restrict__ d_val,
    const index_t* __restrict__ d_cols,
    const index_t* __restrict__ d_rowDelimiters,
    const double* __restrict__ d_vec,
    const index_t numRows,
    double* __restrict__ d_out)
{
    const index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows) return;

    double sum = 0.0;
    const index_t rowStart = d_rowDelimiters[row];
    const index_t rowEnd = d_rowDelimiters[row + 1];
    for (index_t j = rowStart; j < rowEnd; ++j) {
        sum += d_val[j] * d_vec[d_cols[j]];
    }
    d_out[row] = sum;
}

// ****************************************************************************
// Function: printUsage
//
// Purpose:
//   Prints usage information
// ****************************************************************************
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
//   Hybrid MPI+OpenMP+CUDA parallel SpMV benchmark.
//   - MPI: distributes matrix rows across processes
//   - CUDA: each rank computes its row block on GPU
//   - OpenMP: parallelizes host-side data operations
// ****************************************************************************
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Initialize CUDA device: round-robin assignment across available GPUs
    int nDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    if (nDevices == 0) {
        fprintf(stderr, "Error: No CUDA-capable devices found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = rank % nDevices;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // Parse command-line arguments on rank 0, then broadcast to all ranks
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    int validate = 0;
    int printResults = 0;

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
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
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

    // Broadcast parameters to all ranks
    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate number of non-zero elements (use 64-bit to avoid overflow)
    const index_t nItems = static_cast<index_t>((static_cast<uint64_t>(numRows) * numRows) / sparsity);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (static_cast<double>(numRows) * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", nprocs);
        printf("CUDA device: %s (rank %% %d = %d)\n", prop.name, nDevices, device);
        printf("OpenMP max threads: %d\n", omp_get_max_threads());
    }

    // Rank 0 allocates and initializes full data
    std::vector<double> h_val_vec;
    std::vector<index_t> h_cols_vec;
    std::vector<index_t> h_rowDelimiters_vec;
    std::vector<double> h_vec;

    if (rank == 0) {
        printf("Initializing data structures...\n");
        h_val_vec.resize(nItems);
        h_cols_vec.resize(nItems);
        h_rowDelimiters_vec.resize(numRows + 1);
        h_vec.resize(numRows);

        fill(h_vec.data(), numRows, maxVal);
        fill(h_val_vec.data(), nItems, maxVal);
        initRandomMatrix(h_cols_vec.data(), h_rowDelimiters_vec.data(), nItems, numRows);
    }

    // Compute reference solution on rank 0 (CPU, serial) for validation
    std::vector<double> h_reference;
    if (rank == 0 && validate) {
        printf("Computing reference solution (CPU)...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val_vec.data(), h_cols_vec.data(), h_rowDelimiters_vec.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // -------------------------------------------------------------------------
    // Block distribution of rows across MPI ranks
    // -------------------------------------------------------------------------
    const index_t baseRows = numRows / nprocs;
    const index_t remRows = numRows % nprocs;
    const index_t localRows = baseRows + (static_cast<index_t>(rank) < remRows ? 1 : 0);

    // Prepare MPI Scatterv parameters on rank 0
    std::vector<int> rowSndCnt(nprocs, 0);
    std::vector<int> rowSndDpl(nprocs, 0);
    std::vector<int> nzSndCnt(nprocs, 0);
    std::vector<int> nzSndDpl(nprocs, 0);

    if (rank == 0) {
        for (int r = 0; r < nprocs; ++r) {
            const index_t rRows     = baseRows + (static_cast<index_t>(r) < remRows ? 1 : 0);
            const index_t rStart    = static_cast<index_t>(r) * baseRows + (static_cast<index_t>(r) < remRows ? static_cast<index_t>(r) : remRows);
            const index_t rEnd      = rStart + rRows;

            rowSndCnt[r] = static_cast<int>(rRows + 1);
            rowSndDpl[r] = static_cast<int>(rStart);

            const index_t nzStart = h_rowDelimiters_vec[rStart];
            const index_t nzEnd   = h_rowDelimiters_vec[rEnd];
            nzSndCnt[r] = static_cast<int>(nzEnd - nzStart);
            nzSndDpl[r] = static_cast<int>(nzStart);
        }
    }

    // Scatter row delimiters (global absolute indices)
    std::vector<index_t> localRawRowDel(localRows + 1);
    MPI_Scatterv(
        rank == 0 ? h_rowDelimiters_vec.data() : nullptr,
        rowSndCnt.data(), rowSndDpl.data(),
        MPI_UINT32_T,
        localRawRowDel.data(), static_cast<int>(localRows + 1),
        MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Convert to local (zero-based) row delimiters using OpenMP parallel for
    const index_t baseOffset = localRawRowDel[0];
    const index_t localNz = localRawRowDel[localRows] - baseOffset;
    std::vector<index_t> localRowDel(localRows + 1);
    #pragma omp parallel for
    for (index_t i = 0; i <= localRows; ++i) {
        localRowDel[i] = localRawRowDel[i] - baseOffset;
    }

    // Scatter val (non-zero values) to each rank
    std::vector<double> localVal(localNz);
    MPI_Scatterv(
        rank == 0 ? h_val_vec.data() : nullptr,
        nzSndCnt.data(), nzSndDpl.data(),
        MPI_DOUBLE,
        localVal.data(), static_cast<int>(localNz),
        MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter cols (column indices) to each rank
    std::vector<index_t> localCols(localNz);
    MPI_Scatterv(
        rank == 0 ? h_cols_vec.data() : nullptr,
        nzSndCnt.data(), nzSndDpl.data(),
        MPI_UINT32_T,
        localCols.data(), static_cast<int>(localNz),
        MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Broadcast full vec vector to all ranks
    if (rank != 0) h_vec.resize(numRows);
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allocate host output buffer for local results
    std::vector<double> localOut(localRows);

    // -------------------------------------------------------------------------
    // GPU memory allocation and data transfer
    // -------------------------------------------------------------------------
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDel = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    if (localNz > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, localNz * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cols, localNz * sizeof(index_t)));
    }
    CUDA_CHECK(cudaMalloc(&d_rowDel, (localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(&d_out, localRows * sizeof(double)));
    }

    if (localNz > 0) {
        CUDA_CHECK(cudaMemcpy(d_val, localVal.data(), localNz * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, localCols.data(), localNz * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rowDel, localRowDel.data(), (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // SpMV computation: timed GPU kernel loop
    // -------------------------------------------------------------------------
    if (rank == 0) {
        printf("Computing SpMV (MPI+CUDA+OpenMP)...\n");
    }

    // Warm-up kernel to avoid cold-start bias
    if (localRows > 0) {
        const dim3 gridDim((localRows + BLOCK_DIM - 1) / BLOCK_DIM);
        spmvCsrKernel<<<gridDim, BLOCK_DIM>>>(d_val, d_cols, d_rowDel, d_vec, localRows, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed computation iterations
    MPI_Barrier(MPI_COMM_WORLD);
    auto tStart = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows > 0) {
            const dim3 gridDim((localRows + BLOCK_DIM - 1) / BLOCK_DIM);
            spmvCsrKernel<<<gridDim, BLOCK_DIM>>>(d_val, d_cols, d_rowDel, d_vec, localRows, d_out);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto tEnd = std::chrono::high_resolution_clock::now();
    const double localDurationMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    double globalDurationMs = 0.0;
    MPI_Reduce(&localDurationMs, &globalDurationMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy results from GPU to host
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(localOut.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // -------------------------------------------------------------------------
    // Gather distributed results to rank 0
    // -------------------------------------------------------------------------
    std::vector<int> outCnt(nprocs, 0);
    std::vector<int> outDpl(nprocs, 0);
    if (rank == 0) {
        index_t offset = 0;
        for (int r = 0; r < nprocs; ++r) {
            const index_t rRows = baseRows + (static_cast<index_t>(r) < remRows ? 1 : 0);
            outCnt[r] = static_cast<int>(rRows);
            outDpl[r] = static_cast<int>(offset);
            offset += rRows;
        }
    }

    std::vector<double> h_out;
    if (rank == 0) h_out.resize(numRows);
    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr,
                outCnt.data(), outDpl.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // -------------------------------------------------------------------------
    // Report results on rank 0
    // -------------------------------------------------------------------------
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", globalDurationMs);

        const double gflops = (2.0 * nItems * iterations) / (globalDurationMs / 1000.0) / 1e9;
        const double avgTime = globalDurationMs / static_cast<double>(iterations);

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
            }

            // Cleanup GPU memory
            if (d_val)  CUDA_CHECK(cudaFree(d_val));
            if (d_cols) CUDA_CHECK(cudaFree(d_cols));
            if (d_rowDel) CUDA_CHECK(cudaFree(d_rowDel));
            if (d_vec)  CUDA_CHECK(cudaFree(d_vec));
            if (d_out)  CUDA_CHECK(cudaFree(d_out));

            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    // Cleanup GPU memory
    if (d_val)  CUDA_CHECK(cudaFree(d_val));
    if (d_cols) CUDA_CHECK(cudaFree(d_cols));
    if (d_rowDel) CUDA_CHECK(cudaFree(d_rowDel));
    if (d_vec)  CUDA_CHECK(cudaFree(d_vec));
    if (d_out)  CUDA_CHECK(cudaFree(d_out));

    MPI_Finalize();
    return 0;
}
