#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int WARPS_PER_BLOCK = 8;

#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)

static void cudaCheck(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line, expression,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

void fill(double* data, index_t n, double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        data[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

void initRandomMatrix(index_t* cols, index_t* rowDelimiters, index_t n, index_t dim) {
    index_t nnzAssigned = 0;
    const double prob = static_cast<double>(n) /
                        (static_cast<double>(dim) * static_cast<double>(dim));

    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t entriesDone = static_cast<uint64_t>(i) * dim + j;
            const uint64_t entriesLeft = static_cast<uint64_t>(dim) * dim - entriesDone;
            const index_t needToAssign = n - nnzAssigned;
            if (entriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            const double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                cols[nnzAssigned++] = j;
            }
        }
    }
    rowDelimiters[dim] = n;
}

void spmvCpu(const double* val, const index_t* cols, const index_t* rows,
             const double* vec, index_t dim, double* out) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(dim); ++i) {
        double sum = 0.0;
        for (index_t j = rows[i]; j < rows[i + 1]; ++j) {
            sum += val[j] * vec[cols[j]];
        }
        out[i] = sum;
    }
}

// A warp cooperatively processes one row.  Consecutive lanes read consecutive
// CSR entries, giving coalesced accesses while supporting arbitrarily long rows.
__global__ void spmvWarpKernel(const double* __restrict__ val,
                               const index_t* __restrict__ cols,
                               const index_t* __restrict__ rows,
                               const double* __restrict__ vec,
                               index_t numRows, double* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const index_t row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= numRows) {
        return;
    }

    double sum = 0.0;
    for (index_t j = rows[row] + lane; j < rows[row + 1]; j += 32) {
        sum += val[j] * __ldg(vec + cols[j]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        out[row] = sum;
    }
}

bool verifyResults(const double* reference, const double* result, index_t size,
                   index_t globalRowBegin, int rank) {
    int bad = -1;
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(size); ++i) {
        const double ref = reference[i];
        const double res = result[i];
        const bool mismatch = std::abs(ref) < 1e-10
                                  ? std::abs(res) > MAX_RELATIVE_ERROR
                                  : std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR;
        if (mismatch) {
#pragma omp critical
            {
                if (bad < 0 || i < bad) {
                    bad = static_cast<int>(i);
                }
            }
        }
    }
    if (bad >= 0) {
        const double ref = reference[bad];
        const double res = result[bad];
        std::fprintf(stderr,
                     "Rank %d validation failed at index %u: reference %.10e, got %.10e\n",
                     rank, globalRowBegin + static_cast<index_t>(bad), ref, res);
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    std::printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
    bool help = false;
    bool argsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            maxVal = std::strtod(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            if (rank == 0) {
                std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            }
            argsValid = false;
        }
    }

    if (help || !argsValid) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argsValid ? 0 : 1;
    }

    const uint64_t denseItems = static_cast<uint64_t>(numRows) * numRows;
    if (numRows == 0 || numRows > static_cast<index_t>(std::numeric_limits<int>::max()) ||
        sparsity == 0 || iterations == 0 ||
        denseItems / sparsity > std::numeric_limits<index_t>::max() ||
        denseItems / sparsity > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "Invalid problem size (dimensions, sparsity, and iterations must be "
                         "positive; NNZ must fit the CSR/MPI index range).\n");
        }
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(denseItems / sparsity);

    // Use the rank within each shared-memory node for deterministic GPU affinity.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA devices found; CUDA execution is required.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));

    if (rank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        std::printf("Matrix size: %u x %u\n", numRows, numRows);
        std::printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        std::printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
                    100.0 * (1.0 - static_cast<double>(nItems) / denseItems));
        std::printf("Iterations: %u\n", iterations);
        std::printf("Max value: %.2f\n", maxVal);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP thread(s)/rank, "
                    "CUDA GPU %d on rank 0\n",
                    nranks, omp_get_max_threads(), device);
        std::printf("Initializing and distributing data structures...\n");
    }

    std::vector<double> vec(numRows);
    std::vector<double> globalVal;
    std::vector<index_t> globalCols;
    std::vector<index_t> globalRows;
    if (rank == 0) {
        globalVal.resize(nItems);
        globalCols.resize(nItems);
        globalRows.resize(numRows + 1);
        fill(vec.data(), numRows, maxVal);
        fill(globalVal.data(), nItems, maxVal);
        initRandomMatrix(globalCols.data(), globalRows.data(), nItems, numRows);
    }
    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Choose contiguous boundaries by nonzero count rather than just row count.
    // This retains output ordering and balances irregular CSR work across GPUs.
    std::vector<int> rowCounts(nranks), rowDispls(nranks);
    if (rank == 0) {
        rowDispls[0] = 0;
        for (int r = 1; r < nranks; ++r) {
            index_t boundary;
            if (nItems == 0) {
                boundary = static_cast<index_t>(static_cast<uint64_t>(numRows) * r / nranks);
            } else {
                const index_t target = static_cast<index_t>(static_cast<uint64_t>(nItems) * r /
                                                            nranks);
                boundary = static_cast<index_t>(
                    std::lower_bound(globalRows.begin(), globalRows.end(), target) -
                    globalRows.begin());
                boundary = std::min(boundary, numRows);
            }
            rowDispls[r] = static_cast<int>(boundary);
        }
        for (int r = 0; r < nranks; ++r) {
            const int end = (r + 1 < nranks) ? rowDispls[r + 1] : static_cast<int>(numRows);
            rowCounts[r] = end - rowDispls[r];
        }
    }
    MPI_Bcast(rowCounts.data(), nranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rowDispls.data(), nranks, MPI_INT, 0, MPI_COMM_WORLD);
    const index_t localRows = static_cast<index_t>(rowCounts[rank]);
    const index_t globalRowBegin = static_cast<index_t>(rowDispls[rank]);

    std::vector<int> nnzCounts(nranks), nnzDispls(nranks);
    if (rank == 0) {
        for (int r = 0; r < nranks; ++r) {
            nnzDispls[r] = static_cast<int>(globalRows[rowDispls[r]]);
            nnzCounts[r] = static_cast<int>(globalRows[rowDispls[r] + rowCounts[r]] -
                                            globalRows[rowDispls[r]]);
        }
    }
    int localNnzInt = 0;
    MPI_Scatter(nnzCounts.data(), 1, MPI_INT, &localNnzInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const index_t localNnz = static_cast<index_t>(localNnzInt);

    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    std::vector<index_t> localRowDelimiters(localRows + 1);
    MPI_Scatterv(globalVal.data(), nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 localVal.data(), localNnzInt, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(globalCols.data(), nnzCounts.data(), nnzDispls.data(), MPI_UINT32_T,
                 localCols.data(), localNnzInt, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    std::vector<int> delimiterCounts(nranks), delimiterDispls(nranks);
    for (int r = 0; r < nranks; ++r) {
        delimiterCounts[r] = rowCounts[r] + 1;
        delimiterDispls[r] = rowDispls[r];
    }
    MPI_Scatterv(globalRows.data(), delimiterCounts.data(), delimiterDispls.data(), MPI_UINT32_T,
                 localRowDelimiters.data(), static_cast<int>(localRows + 1), MPI_UINT32_T, 0,
                 MPI_COMM_WORLD);
    const index_t nnzBase = localRowDelimiters[0];
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i <= static_cast<int64_t>(localRows); ++i) {
        localRowDelimiters[i] -= nnzBase;
    }
    globalVal.clear();
    globalCols.clear();
    globalRows.clear();

    double* dVal = nullptr;
    double* dVec = nullptr;
    double* dOut = nullptr;
    index_t* dCols = nullptr;
    index_t* dRows = nullptr;
    CUDA_CHECK(cudaMalloc(&dVal, std::max<size_t>(1, localNnz) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCols, std::max<size_t>(1, localNnz) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dRows, (static_cast<size_t>(localRows) + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dVec, static_cast<size_t>(numRows) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dOut, std::max<size_t>(1, localRows) * sizeof(double)));
    if (localNnz > 0) {
        CUDA_CHECK(cudaMemcpy(dVal, localVal.data(), localNnz * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dCols, localCols.data(), localNnz * sizeof(index_t),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(dRows, localRowDelimiters.data(),
                          (static_cast<size_t>(localRows) + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVec, vec.data(), static_cast<size_t>(numRows) * sizeof(double),
                          cudaMemcpyHostToDevice));

    const dim3 block(WARPS_PER_BLOCK * 32);
    const dim3 grid((localRows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    if (localRows > 0) {
        // Warm-up excludes lazy CUDA initialization and kernel loading from the benchmark.
        spmvWarpKernel<<<grid, block>>>(dVal, dCols, dRows, dVec, localRows, dOut);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (rank == 0) {
        std::printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (localRows > 0) {
        for (index_t iter = 0; iter < iterations; ++iter) {
            spmvWarpKernel<<<grid, block>>>(dVal, dCols, dRows, dVec, localRows, dOut);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> localOut(localRows);
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(localOut.data(), dOut, localRows * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    int allValid = 1;
    if (validate) {
        std::vector<double> reference(localRows);
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(), vec.data(),
                localRows, reference.data());
        const int localValid = verifyResults(reference.data(), localOut.data(), localRows,
                                             globalRowBegin, rank)
                                   ? 1
                                   : 0;
        MPI_Allreduce(&localValid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }

    std::vector<double> output;
    if (printResults) {
        if (rank == 0) {
            output.resize(numRows);
        }
        MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE, output.data(),
                    rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        const double milliseconds = elapsed * 1000.0;
        const double gflops = 2.0 * static_cast<double>(nItems) * iterations / elapsed / 1e9;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Average time per iteration: %.3f ms\n", milliseconds / iterations);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) {
            print_results(output, "OutputVector");
        }
        if (validate) {
            std::printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
        }
    }

    CUDA_CHECK(cudaFree(dVal));
    CUDA_CHECK(cudaFree(dCols));
    CUDA_CHECK(cudaFree(dRows));
    CUDA_CHECK(cudaFree(dVec));
    CUDA_CHECK(cudaFree(dOut));
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return allValid ? 0 : 1;
}
