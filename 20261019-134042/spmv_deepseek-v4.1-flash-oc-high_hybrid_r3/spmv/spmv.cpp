#include <chrono>
#include <algorithm>
#include <climits>
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
    const uint64_t totalEntries = static_cast<uint64_t>(dim) * static_cast<uint64_t>(dim);
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t numEntriesLeft =
                totalEntries - (static_cast<uint64_t>(i) * static_cast<uint64_t>(dim) + j);
            const index_t needToAssign = n - nnzAssigned;
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

__global__ void spmvWarpPerRow(const double* __restrict__ val,
                               const index_t* __restrict__ cols,
                               const index_t* __restrict__ rowDelimiters,
                               const double* __restrict__ vec,
                               const index_t numRows,
                               double* __restrict__ out) {
    const unsigned int warpsPerBlock = blockDim.x >> 5;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const uint64_t row64 = static_cast<uint64_t>(blockIdx.x) * warpsPerBlock + warp;
    if (row64 >= numRows) {
        return;
    }
    const index_t row = static_cast<index_t>(row64);

    const index_t start = rowDelimiters[row];
    const index_t end = rowDelimiters[row + 1];

    double sum = 0.0;
    for (index_t j = start + lane; j < end; j += 32) {
        sum += __ldg(&val[j]) * __ldg(&vec[__ldg(&cols[j])]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0u) {
        out[row] = sum;
    }
}

__global__ void spmvThreadPerRow(const double* __restrict__ val,
                                 const index_t* __restrict__ cols,
                                 const index_t* __restrict__ rowDelimiters,
                                 const double* __restrict__ vec,
                                 const index_t numRows,
                                 double* __restrict__ out) {
    const uint64_t row64 = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row64 >= numRows) {
        return;
    }
    const index_t row = static_cast<index_t>(row64);

    double t = 0.0;
    const index_t start = rowDelimiters[row];
    const index_t end = rowDelimiters[row + 1];
    for (index_t j = start; j < end; ++j) {
        t += __ldg(&val[j]) * __ldg(&vec[__ldg(&cols[j])]);
    }
    out[row] = t;
}

static void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void launchSpMV(const double* d_val, const index_t* d_cols,
                       const index_t* d_rowDelimiters, const double* d_vec,
                       const index_t numRows, double* d_out, const double avgNnzPerRow) {
    if (numRows == 0) {
        return;
    }
    constexpr int blockSize = 256;
    if (avgNnzPerRow >= 8.0) {
        constexpr int warpsPerBlock = blockSize / 32;
        const uint64_t blocks = (static_cast<uint64_t>(numRows) + warpsPerBlock - 1) / warpsPerBlock;
        spmvWarpPerRow<<<static_cast<unsigned int>(blocks), blockSize>>>(
            d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
    } else {
        const uint64_t blocks = (static_cast<uint64_t>(numRows) + blockSize - 1) / blockSize;
        spmvThreadPerRow<<<static_cast<unsigned int>(blocks), blockSize>>>(
            d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
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

    // Parse command line arguments
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

    if (sparsity == 0) {
        if (rank == 0) {
            printf("Sparsity must be greater than zero\n");
        }
        MPI_Finalize();
        return 1;
    }

    const uint64_t totalEntries = static_cast<uint64_t>(numRows) * static_cast<uint64_t>(numRows);
    const uint64_t nItems64 = totalEntries / sparsity;
    if (nItems64 > static_cast<uint64_t>(INT_MAX) || numRows > static_cast<index_t>(INT_MAX)) {
        if (rank == 0) {
            printf("Problem size too large for the supported MPI count range\n");
        }
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nItems64);

    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int nodeSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &nodeSize);

    const int maxThreads = omp_get_max_threads();
    const int coresPerRank = std::max(1, omp_get_num_procs() / std::max(1, nodeSize));
    omp_set_num_threads(std::min({maxThreads, coresPerRank, 32}));

    int deviceCount = 0;
    cudaError_t deviceErr = cudaGetDeviceCount(&deviceCount);
    if (deviceErr != cudaSuccess || deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA device available: %s\n", cudaGetErrorString(deviceErr));
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice");

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) /
                                          (static_cast<double>(numRows) * static_cast<double>(numRows))));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const index_t baseRows = numRows / static_cast<index_t>(numRanks);
    const index_t extraRows = numRows % static_cast<index_t>(numRanks);
    auto rowStartOf = [baseRows, extraRows](int r) -> index_t {
        const index_t rr = static_cast<index_t>(r);
        return rr * baseRows + (rr < extraRows ? rr : extraRows);
    };
    const index_t startRow = rowStartOf(rank);
    const index_t endRow = rowStartOf(rank + 1);
    const index_t localRows = endRow - startRow;

    std::vector<double> h_vec(numRows);
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> fullVal;
    std::vector<index_t> fullCols;

    if (rank == 0) {
        fullVal.resize(nItems);
        fullCols.resize(nItems);
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(fullVal.data(), nItems, maxVal);
        initRandomMatrix(fullCols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t baseNnz = h_rowDelimiters[startRow];
    const index_t localNnz = h_rowDelimiters[endRow] - baseNnz;

    std::vector<int> sendCounts(numRanks, 0);
    std::vector<int> sendDispls(numRanks, 0);
    if (rank == 0) {
        for (int r = 0; r < numRanks; ++r) {
            const index_t rs = rowStartOf(r);
            const index_t re = rowStartOf(r + 1);
            sendCounts[r] = static_cast<int>(h_rowDelimiters[re] - h_rowDelimiters[rs]);
            sendDispls[r] = static_cast<int>(h_rowDelimiters[rs]);
        }
    }

    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);

    MPI_Scatterv(rank == 0 ? fullVal.data() : nullptr, sendCounts.data(), sendDispls.data(),
                 MPI_DOUBLE, localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? fullCols.data() : nullptr, sendCounts.data(), sendDispls.data(),
                 MPI_UINT32_T, localCols.data(), static_cast<int>(localNnz), MPI_UINT32_T, 0,
                 MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<double>().swap(fullVal);
        std::vector<index_t>().swap(fullCols);
    }

    std::vector<index_t> localRowDelimiters(localRows + 1);
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < localRows; ++i) {
        localRowDelimiters[i] = h_rowDelimiters[startRow + i] - baseNnz;
    }
    localRowDelimiters[localRows] = h_rowDelimiters[endRow] - baseNnz;
    std::vector<index_t>().swap(h_rowDelimiters);

    std::vector<double> localRef;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        localRef.resize(localRows);
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(), h_vec.data(),
                localRows, localRef.data());
    }

    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    if (localNnz > 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_val),
                             static_cast<size_t>(localNnz) * sizeof(double)), "cudaMalloc val");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_cols),
                             static_cast<size_t>(localNnz) * sizeof(index_t)), "cudaMalloc cols");
        checkCuda(cudaMemcpy(d_val, localVal.data(), static_cast<size_t>(localNnz) * sizeof(double),
                             cudaMemcpyHostToDevice), "copy val");
        checkCuda(cudaMemcpy(d_cols, localCols.data(), static_cast<size_t>(localNnz) * sizeof(index_t),
                             cudaMemcpyHostToDevice), "copy cols");
    }
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_rowDelimiters),
                         static_cast<size_t>(localRows + 1) * sizeof(index_t)), "cudaMalloc rows");
    checkCuda(cudaMemcpy(d_rowDelimiters, localRowDelimiters.data(),
                         static_cast<size_t>(localRows + 1) * sizeof(index_t),
                         cudaMemcpyHostToDevice), "copy rows");
    if (localRows > 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_vec),
                             static_cast<size_t>(numRows) * sizeof(double)), "cudaMalloc vec");
        checkCuda(cudaMemcpy(d_vec, h_vec.data(), static_cast<size_t>(numRows) * sizeof(double),
                             cudaMemcpyHostToDevice), "copy vec");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_out),
                             static_cast<size_t>(localRows) * sizeof(double)), "cudaMalloc out");
    }

    const double avgNnzPerRow =
        localRows > 0 ? static_cast<double>(localNnz) / static_cast<double>(localRows) : 0.0;

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    if (localRows > 0 && iterations > 0) {
        launchSpMV(d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out, avgNnzPerRow);
        checkCuda(cudaGetLastError(), "warmup launch");
    }
    checkCuda(cudaDeviceSynchronize(), "warmup synchronize");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpMV(d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out, avgNnzPerRow);
    }
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "kernel synchronize");

    auto end = std::chrono::high_resolution_clock::now();
    const double localElapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    double elapsed = localElapsed;
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    std::vector<double> localOut(localRows);
    if (localRows > 0 && iterations > 0) {
        checkCuda(cudaMemcpy(localOut.data(), d_out, static_cast<size_t>(localRows) * sizeof(double),
                             cudaMemcpyDeviceToHost), "copy out");
    }

    std::vector<int> rowCounts(numRanks, 0);
    std::vector<int> rowDispls(numRanks, 0);
    for (int r = 0; r < numRanks; ++r) {
        rowCounts[r] = static_cast<int>(rowStartOf(r + 1) - rowStartOf(r));
        rowDispls[r] = static_cast<int>(rowStartOf(r));
    }

    std::vector<double> h_out;
    if (rank == 0) {
        h_out.resize(numRows);
    }
    MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr, rowCounts.data(), rowDispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) {
            h_reference.resize(numRows);
        }
        MPI_Gatherv(localRef.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    rank == 0 ? h_reference.data() : nullptr, rowCounts.data(), rowDispls.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int returnCode = 0;
    if (rank == 0) {
        const long long elapsedMs =
            static_cast<long long>(std::llround(elapsed * 1000.0));
        printf("Computation time: %lld ms\n", elapsedMs);

        const double metricsTime = elapsed > 0.0 ? elapsed : 1e-9;
        const double gflops = (2.0 * nItems * iterations) / metricsTime / 1e9;
        const double avgTime = metricsTime * 1000.0 / static_cast<double>(iterations > 0 ? iterations : 1);

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
                returnCode = 0;
            } else {
                printf("Validation: FAILED\n");
                returnCode = 1;
            }
        }
    }

    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (d_val) {
        checkCuda(cudaFree(d_val), "cudaFree val");
    }
    if (d_cols) {
        checkCuda(cudaFree(d_cols), "cudaFree cols");
    }
    checkCuda(cudaFree(d_rowDelimiters), "cudaFree rows");
    checkCuda(cudaFree(d_vec), "cudaFree vec");
    if (d_out) {
        checkCuda(cudaFree(d_out), "cudaFree out");
    }

    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return returnCode;
}
