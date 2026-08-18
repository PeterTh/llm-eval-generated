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
constexpr int CUDA_BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

static void cudaCheck(cudaError_t status, const char* operation, MPI_Comm comm) {
    if (status != cudaSuccess) {
        int rank = -1;
        MPI_Comm_rank(comm, &rank);
        std::fprintf(stderr, "Rank %d: CUDA error in %s: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(comm, EXIT_FAILURE);
    }
}

void fill(double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n,
                      const index_t dim) {
    index_t nnzAssigned = 0;
    const double prob = static_cast<double>(n) /
                        (static_cast<double>(dim) * static_cast<double>(dim));
    srand(8675309);

    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t linear = static_cast<uint64_t>(i) * dim + j;
            const uint64_t entriesLeft = static_cast<uint64_t>(dim) * dim - linear;
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
             const double* vec, const index_t dim, double* out) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(dim); ++i) {
        double sum = 0.0;
        for (index_t j = rows[i]; j < rows[i + 1]; ++j) {
            sum += val[j] * vec[cols[j]];
        }
        out[i] = sum;
    }
}

// A warp owns a row. Threads access consecutive CSR entries and then reduce in
// registers, avoiding atomics and intermediate storage. This is effective for
// both short and moderately dense rows and is insensitive to row-length skew.
__global__ void spmvCsrWarpKernel(const double* __restrict__ val,
                                  const index_t* __restrict__ cols,
                                  const index_t* __restrict__ rows,
                                  const double* __restrict__ vec,
                                  index_t localRows,
                                  double* __restrict__ out) {
    const int globalThread = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t row = globalThread / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (row >= localRows) {
        return;
    }

    double sum = 0.0;
    for (index_t j = rows[row] + lane; j < rows[row + 1]; j += WARP_SIZE) {
        sum += __ldg(val + j) * __ldg(vec + cols[j]);
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        out[row] = sum;
    }
}

bool verifyResults(const double* reference, const double* result, const index_t size) {
    int64_t firstBad = static_cast<int64_t>(size);
#pragma omp parallel for reduction(min : firstBad) schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(size); ++i) {
        const double ref = reference[i];
        const double res = result[i];
        const bool bad = std::abs(ref) < 1e-10
                             ? std::abs(res) > MAX_RELATIVE_ERROR
                             : std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR;
        if (bad) {
            firstBad = i;
        }
    }
    if (firstBad == static_cast<int64_t>(size)) {
        return true;
    }

    const double ref = reference[firstBad];
    const double res = result[firstBad];
    const double rel = std::abs(ref) < 1e-10 ? std::abs(res) : std::abs((res - ref) / ref);
    std::printf("Validation failed at index %lld: reference %.10e, got %.10e "
                "(error: %.10e)\n", static_cast<long long>(firstBad), ref, res, rel);
    return false;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    std::printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -m <val>     Maximum element value (default: 1.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown or incomplete option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
            break;
        }
    }

    const uint64_t matrixEntries = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nnz64 = sparsity == 0 ? 0 : matrixEntries / sparsity;
    if (parseStatus || numRows == 0 || sparsity == 0 || iterations == 0 ||
        nnz64 > std::numeric_limits<index_t>::max() ||
        numRows > static_cast<index_t>(std::numeric_limits<int>::max())) {
        if (rank == 0 && !parseStatus) {
            std::fprintf(stderr, "Invalid size/options: dimensions, sparsity, and iterations must "
                                 "be positive and the CSR/MPI counts must fit in 32 bits\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    const index_t nItems = static_cast<index_t>(nnz64);

    // Balanced contiguous row ownership. Contiguity minimizes metadata and
    // enables direct Gatherv of the final output.
    std::vector<int> rowCounts(worldSize), rowDispls(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const uint64_t begin = static_cast<uint64_t>(numRows) * r / worldSize;
        const uint64_t end = static_cast<uint64_t>(numRows) * (r + 1) / worldSize;
        rowDispls[r] = static_cast<int>(begin);
        rowCounts[r] = static_cast<int>(end - begin);
    }
    const index_t localRows = static_cast<index_t>(rowCounts[rank]);

    std::vector<double> hVal;
    std::vector<index_t> hCols;
    std::vector<index_t> hRows;
    std::vector<double> hVec(numRows);
    std::vector<double> hOut;
    if (rank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        std::printf("Matrix size: %u x %u\n", numRows, numRows);
        std::printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        std::printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
                    100.0 * (1.0 - static_cast<double>(nItems) / matrixEntries));
        std::printf("Iterations: %u\n", iterations);
        std::printf("Max value: %.2f\n", maxVal);
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", worldSize,
                    omp_get_max_threads());
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing and distributing data structures...\n");

        hVal.resize(nItems);
        hCols.resize(nItems);
        hRows.resize(static_cast<size_t>(numRows) + 1);
        hOut.resize(numRows);
        fill(hVec.data(), numRows, maxVal);
        fill(hVal.data(), nItems, maxVal);
        initRandomMatrix(hCols.data(), hRows.data(), nItems, numRows);
    }

    std::vector<int> nnzCounts(worldSize), nnzDispls(worldSize), rowMetaCounts(worldSize);
    int distributionValid = 1;
    if (rank == 0) {
        for (int r = 0; r < worldSize; ++r) {
            const index_t begin = hRows[rowDispls[r]];
            const index_t end = hRows[rowDispls[r] + rowCounts[r]];
            if (end - begin > static_cast<index_t>(std::numeric_limits<int>::max())) {
                distributionValid = 0;
            }
            nnzDispls[r] = static_cast<int>(begin);
            nnzCounts[r] = static_cast<int>(end - begin);
            rowMetaCounts[r] = rowCounts[r] + 1;
        }
    }
    MPI_Bcast(&distributionValid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!distributionValid) {
        if (rank == 0) std::fprintf(stderr, "A rank's CSR partition exceeds MPI count limits\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int localNnzInt = 0;
    MPI_Scatter(nnzCounts.data(), 1, MPI_INT, &localNnzInt, 1, MPI_INT, 0,
                MPI_COMM_WORLD);
    const index_t localNnz = static_cast<index_t>(localNnzInt);
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    std::vector<index_t> localRowOffsets(static_cast<size_t>(localRows) + 1);

    MPI_Bcast(hVec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? hVal.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                 MPI_DOUBLE, localVal.data(), localNnzInt, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? hCols.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                 MPI_UINT32_T, localCols.data(), localNnzInt, MPI_UINT32_T, 0,
                 MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? hRows.data() : nullptr, rowMetaCounts.data(),
                 rowDispls.data(), MPI_UINT32_T, localRowOffsets.data(),
                 static_cast<int>(localRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    const index_t partitionBase = localRowOffsets[0];
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i <= static_cast<int64_t>(localRows); ++i) {
        localRowOffsets[i] -= partitionBase;
    }

    // Map ranks on the same host round-robin onto its visible accelerators.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", MPI_COMM_WORLD);
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices are visible\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int device = localRank % deviceCount;
    cudaCheck(cudaSetDevice(device), "cudaSetDevice", MPI_COMM_WORLD);
    cudaCheck(cudaFree(nullptr), "CUDA context initialization", MPI_COMM_WORLD);
    MPI_Comm_free(&localComm);

    double *dVal = nullptr, *dVec = nullptr, *dOut = nullptr;
    index_t *dCols = nullptr, *dRows = nullptr;
    cudaCheck(cudaMalloc(&dVal, std::max<size_t>(1, localNnz) * sizeof(double)),
              "cudaMalloc(values)", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&dCols, std::max<size_t>(1, localNnz) * sizeof(index_t)),
              "cudaMalloc(columns)", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&dRows, (static_cast<size_t>(localRows) + 1) * sizeof(index_t)),
              "cudaMalloc(rows)", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&dVec, static_cast<size_t>(numRows) * sizeof(double)),
              "cudaMalloc(vector)", MPI_COMM_WORLD);
    cudaCheck(cudaMalloc(&dOut, std::max<size_t>(1, localRows) * sizeof(double)),
              "cudaMalloc(output)", MPI_COMM_WORLD);
    if (localNnz != 0) {
        cudaCheck(cudaMemcpy(dVal, localVal.data(), localNnz * sizeof(double),
                             cudaMemcpyHostToDevice), "copy values", MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(dCols, localCols.data(), localNnz * sizeof(index_t),
                             cudaMemcpyHostToDevice), "copy columns", MPI_COMM_WORLD);
    }
    cudaCheck(cudaMemcpy(dRows, localRowOffsets.data(), (localRows + 1) * sizeof(index_t),
                         cudaMemcpyHostToDevice), "copy row offsets", MPI_COMM_WORLD);
    cudaCheck(cudaMemcpy(dVec, hVec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice),
              "copy dense vector", MPI_COMM_WORLD);

    // Host staging buffers are no longer needed during the benchmark. Keep the
    // complete root-side matrix only when CPU validation was requested.
    std::vector<double>().swap(localVal);
    std::vector<index_t>().swap(localCols);
    std::vector<index_t>().swap(localRowOffsets);
    if (rank != 0) {
        std::vector<double>().swap(hVec);
    } else if (!validate) {
        std::vector<double>().swap(hVal);
        std::vector<index_t>().swap(hCols);
        std::vector<index_t>().swap(hRows);
    }

    const int warpsPerBlock = CUDA_BLOCK_SIZE / WARP_SIZE;
    const int blocks = (static_cast<int>(localRows) + warpsPerBlock - 1) / warpsPerBlock;
    if (localRows != 0) {
        // Untimed warm-up removes lazy context and instruction-cache costs.
        spmvCsrWarpKernel<<<blocks, CUDA_BLOCK_SIZE>>>(dVal, dCols, dRows, dVec, localRows, dOut);
        cudaCheck(cudaDeviceSynchronize(), "warm-up SpMV", MPI_COMM_WORLD);
    }

    if (rank == 0) std::printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows != 0) {
            spmvCsrWarpKernel<<<blocks, CUDA_BLOCK_SIZE>>>(dVal, dCols, dRows, dVec,
                                                           localRows, dOut);
        }
    }
    cudaCheck(cudaGetLastError(), "SpMV kernel launch", MPI_COMM_WORLD);
    cudaCheck(cudaDeviceSynchronize(), "timed SpMV", MPI_COMM_WORLD);
    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const bool needOutput = validate || printResults;
    std::vector<double> localOut(needOutput ? localRows : 0);
    if (needOutput && localRows != 0) {
        cudaCheck(cudaMemcpy(localOut.data(), dOut, localRows * sizeof(double),
                             cudaMemcpyDeviceToHost), "copy output", MPI_COMM_WORLD);
    }
    if (needOutput) {
        MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    rank == 0 ? hOut.data() : nullptr, rowCounts.data(), rowDispls.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    cudaFree(dOut);
    cudaFree(dVec);
    cudaFree(dRows);
    cudaFree(dCols);
    cudaFree(dVal);

    int exitCode = EXIT_SUCCESS;
    if (rank == 0) {
        const double milliseconds = elapsedSeconds * 1000.0;
        const double gflops = (2.0 * nItems * iterations) / elapsedSeconds / 1e9;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Average time per iteration: %.3f ms\n", milliseconds / iterations);
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(hOut, "OutputVector");
        }
        if (validate) {
            std::printf("Computing reference solution and validating result...\n");
            std::vector<double> reference(numRows);
            spmvCpu(hVal.data(), hCols.data(), hRows.data(), hVec.data(), numRows,
                    reference.data());
            if (verifyResults(reference.data(), hOut.data(), numRows)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = EXIT_FAILURE;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
