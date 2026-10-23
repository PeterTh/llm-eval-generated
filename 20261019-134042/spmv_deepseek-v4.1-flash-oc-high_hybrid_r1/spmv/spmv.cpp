#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

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
// Function: spmvCsrWarpKernel
//
// Purpose:
//   CUDA SpMV kernel using one warp per CSR row. The lanes stride through the
//   row, which keeps the sparse value/column streams coalesced and exposes
//   enough parallelism to fill a modern accelerator. The per-warp reduction
//   produces results with the same meaning as the original CPU code within
//   floating-point round-off.
//
// ****************************************************************************
__global__ void spmvCsrWarpKernel(const double* __restrict__ val,
                                 const index_t* __restrict__ cols,
                                 const index_t* __restrict__ rowDelimiters,
                                 const double* __restrict__ vec,
                                 const index_t numRows,
                                 double* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const index_t row = static_cast<index_t>(
        (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
    if (row >= numRows) {
        return;
    }

    const index_t start = rowDelimiters[row];
    const index_t end = rowDelimiters[row + 1];
    double sum = 0.0;
    #pragma unroll 4
    for (index_t j = start + static_cast<index_t>(lane); j < end; j += 32) {
        sum += val[j] * vec[cols[j]];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        out[row] = sum;
    }
}

static void checkCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank parses the same command line)
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Contiguous row partition across MPI ranks.
    const index_t rowsPerRank = numRows / static_cast<index_t>(numRanks);
    const index_t remainder = numRows % static_cast<index_t>(numRanks);
    const index_t rankId = static_cast<index_t>(rank);
    const index_t rowStart = rankId * rowsPerRank + std::min(rankId, remainder);
    const index_t rowEnd = rowStart + rowsPerRank + (rankId < remainder ? 1u : 0u);
    const index_t localRows = rowEnd - rowStart;

    // Rank 0 generates the complete, canonically seeded matrix. It is then
    // scattered so that each rank only stores the rows it owns.
    std::vector<double> h_vec(numRows);
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> fullVal;
    std::vector<index_t> fullCols;

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fullVal.resize(nItems);
        fullCols.resize(nItems);
        fill(h_vec.data(), numRows, maxVal);
        fill(fullVal.data(), nItems, maxVal);
        initRandomMatrix(fullCols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Per-rank row and non-zero counts used by MPI_Scatterv.
    std::vector<int> nnzCounts(numRanks);
    std::vector<int> nnzDispls(numRanks);
    std::vector<int> rowCounts(numRanks);
    std::vector<int> rowDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        const index_t rr = static_cast<index_t>(r);
        const index_t rs = rr * rowsPerRank + std::min(rr, remainder);
        const index_t re = rs + rowsPerRank + (rr < remainder ? 1u : 0u);
        rowCounts[r] = static_cast<int>(re - rs);
        rowDispls[r] = static_cast<int>(rs);
        nnzCounts[r] = static_cast<int>(h_rowDelimiters[re] - h_rowDelimiters[rs]);
        nnzDispls[r] = static_cast<int>(h_rowDelimiters[rs]);
    }

    const index_t localNnz = h_rowDelimiters[rowEnd] - h_rowDelimiters[rowStart];
    const index_t nnzBase = h_rowDelimiters[rowStart];

    std::vector<double> h_val(localNnz);
    std::vector<index_t> h_cols(localNnz);

    MPI_Scatterv(rank == 0 ? fullVal.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                 MPI_DOUBLE, h_val.data(), static_cast<int>(localNnz), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? fullCols.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                 MPI_UINT32_T, h_cols.data(), static_cast<int>(localNnz), MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    // Release the globally replicated copy on rank 0.
    if (rank == 0) {
        std::vector<double>().swap(fullVal);
        std::vector<index_t>().swap(fullCols);
    }

    // Local, zero-based row delimiters for both the CPU and GPU paths.
    std::vector<index_t> h_rowDelimsLocal(localRows + 1);
    #pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= localRows; ++i) {
        h_rowDelimsLocal[i] = h_rowDelimiters[rowStart + i] - nnzBase;
    }

    std::vector<double> h_out(localRows);
    std::vector<double> h_referenceLocal;

    // For validation, compute the reference solution on the host using the
    // same per-row accumulation order as the original CPU implementation.
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        h_referenceLocal.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimsLocal.data(),
                h_vec.data(), localRows, h_referenceLocal.data());
    }

    // Select one GPU per rank in round-robin fashion.
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA-capable device found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int deviceId = rank % deviceCount;
    checkCuda(cudaSetDevice(deviceId), "cudaSetDevice");

    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelims = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    if (localNnz > 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_val), localNnz * sizeof(double)), "cudaMalloc val");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_cols), localNnz * sizeof(index_t)), "cudaMalloc cols");
    }
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_rowDelims), (localRows + 1) * sizeof(index_t)), "cudaMalloc rowDelims");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_vec), numRows * sizeof(double)), "cudaMalloc vec");
    if (localRows > 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_out), localRows * sizeof(double)), "cudaMalloc out");
    }

    if (localNnz > 0) {
        checkCuda(cudaMemcpy(d_val, h_val.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice), "copy val");
        checkCuda(cudaMemcpy(d_cols, h_cols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice), "copy cols");
    }
    checkCuda(cudaMemcpy(d_rowDelims, h_rowDelimsLocal.data(), (localRows + 1) * sizeof(index_t),
                         cudaMemcpyHostToDevice), "copy rowDelims");
    checkCuda(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice), "copy vec");

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    const int blockSize = 256;
    const int warpsPerBlock = blockSize / 32;
    const int gridSize = static_cast<int>((localRows + warpsPerBlock - 1) / warpsPerBlock);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows > 0) {
            spmvCsrWarpKernel<<<gridSize, blockSize>>>(
                d_val, d_cols, d_rowDelims, d_vec, localRows, d_out);
        }
    }

    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    if (localRows > 0 && iterations > 0) {
        checkCuda(cudaMemcpy(h_out.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost), "copy out");
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsedUs / 1000));

        // Calculate performance metrics
        const double elapsedSec = static_cast<double>(std::max<long long>(elapsedUs, 1)) / 1e6;
        const double gflops = (2.0 * nItems * iterations) / elapsedSec / 1e9;
        const double avgTime = (elapsedSec * 1000.0) / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather the distributed output vector so rank 0 can print/verify it.
    std::vector<double> h_outFull;
    if (rank == 0) {
        h_outFull.resize(numRows);
    }
    MPI_Gatherv(h_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                rank == 0 ? h_outFull.data() : nullptr, rowCounts.data(), rowDispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (printResults && rank == 0) {
        print_results(h_outFull, "OutputVector");
    }

    int status = 0;
    if (validate) {
        std::vector<double> h_referenceFull;
        if (rank == 0) {
            h_referenceFull.resize(numRows);
        }
        MPI_Gatherv(h_referenceLocal.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    rank == 0 ? h_referenceFull.data() : nullptr, rowCounts.data(), rowDispls.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_referenceFull.data(), h_outFull.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
                status = 0;
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (d_val) cudaFree(d_val);
    if (d_cols) cudaFree(d_cols);
    if (d_rowDelims) cudaFree(d_rowDelims);
    if (d_vec) cudaFree(d_vec);
    if (d_out) cudaFree(d_out);

    MPI_Finalize();
    return status;
}
