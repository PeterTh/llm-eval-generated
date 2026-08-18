#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr int kBlockSize = 256;

void fail(const char* what, int rank) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, what);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

void cudaCheck(cudaError_t status, const char* what, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: %s: %s\n", rank, what,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void cublasCheck(cublasStatus_t status, const char* what, int rank) {
    if (status != CUBLAS_STATUS_SUCCESS) fail(what, rank);
}

void cusolverCheck(cusolverStatus_t status, const char* what, int rank) {
    if (status != CUSOLVER_STATUS_SUCCESS) fail(what, rank);
}

__global__ void addDiagonal(double* matrix, size_t n, size_t firstRow,
                            size_t localRows) {
    const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < localRows) matrix[row * n + firstRow + row] += static_cast<double>(n);
}

struct Partition {
    int rank;
    int ranks;
    size_t n;
    size_t first;
    size_t rows;
    std::vector<size_t> starts;
    std::vector<size_t> sizes;

    Partition(size_t order, int worldRank, int worldSize)
        : rank(worldRank), ranks(worldSize), n(order), starts(worldSize), sizes(worldSize) {
        const size_t base = n / static_cast<size_t>(ranks);
        const size_t extra = n % static_cast<size_t>(ranks);
        size_t cursor = 0;
        for (int r = 0; r < ranks; ++r) {
            sizes[r] = base + (static_cast<size_t>(r) < extra ? 1 : 0);
            starts[r] = cursor;
            cursor += sizes[r];
        }
        first = starts[rank];
        rows = sizes[rank];
    }

    int owner(size_t row) const {
        const size_t base = n / static_cast<size_t>(ranks);
        const size_t extra = n % static_cast<size_t>(ranks);
        const size_t longRows = (base + 1) * extra;
        if (row < longRows) return static_cast<int>(row / (base + 1));
        return static_cast<int>(extra + (row - longRows) / base);
    }
};

// Each rank constructs the same B, then its GPU computes only the rows of B*B^T
// that the rank owns.  This removes an O(n^2) scatter from the timed setup.
void generateLocalMatrix(double* deviceA, const Partition& part,
                         cublasHandle_t blas) {
    std::vector<double> B(part.n * part.n);
    unsigned int seed = 42;
    for (double& value : B) {
        value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

    double* deviceB = nullptr;
    const size_t bBytes = std::max<size_t>(1, part.n * part.n) * sizeof(double);
    cudaCheck(cudaMalloc(&deviceB, bBytes), "cudaMalloc(B) failed", part.rank);
    if (!B.empty()) {
        cudaCheck(cudaMemcpy(deviceB, B.data(), B.size() * sizeof(double),
                             cudaMemcpyHostToDevice),
                  "copying B to the GPU failed", part.rank);
    }

    if (part.rows != 0 && part.n != 0) {
        const double one = 1.0;
        const double zero = 0.0;
        // Row-major C=B_local*B^T is column-major C^T=B*B_local^T.
        cublasCheck(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                static_cast<int>(part.n), static_cast<int>(part.rows),
                                static_cast<int>(part.n), &one, deviceB,
                                static_cast<int>(part.n),
                                deviceB + part.first * part.n,
                                static_cast<int>(part.n), &zero, deviceA,
                                static_cast<int>(part.n)),
                    "cuBLAS matrix generation failed", part.rank);
        addDiagonal<<<static_cast<unsigned>((part.rows + 255) / 256), 256>>>(
            deviceA, part.n, part.first, part.rows);
        cudaCheck(cudaGetLastError(), "diagonal kernel failed", part.rank);
        cudaCheck(cudaDeviceSynchronize(), "matrix generation failed", part.rank);
    }
    cudaCheck(cudaFree(deviceB), "cudaFree(B) failed", part.rank);
}

bool distributedCholesky(double* deviceA, const Partition& part,
                         cublasHandle_t blas, cusolverDnHandle_t solver) {
    if (part.n == 0) return true;
    if (part.n > static_cast<size_t>(std::numeric_limits<int>::max()))
        fail("matrix order exceeds the CUDA library integer range", part.rank);

    const size_t maxBlock = std::min<size_t>(kBlockSize, part.n);
    double* deviceDiagonal = nullptr;
    double* devicePanel = nullptr;
    int* deviceInfo = nullptr;
    double* deviceWork = nullptr;
    double* hostDiagonal = nullptr;
    double* hostSend = nullptr;
    double* hostPanel = nullptr;

    cudaCheck(cudaMalloc(&deviceDiagonal, maxBlock * maxBlock * sizeof(double)),
              "cudaMalloc(diagonal) failed", part.rank);
    cudaCheck(cudaMalloc(&devicePanel, part.n * maxBlock * sizeof(double)),
              "cudaMalloc(panel) failed", part.rank);
    cudaCheck(cudaMalloc(&deviceInfo, sizeof(int)), "cudaMalloc(info) failed", part.rank);
    cudaCheck(cudaHostAlloc(&hostDiagonal, maxBlock * maxBlock * sizeof(double),
                            cudaHostAllocPortable),
              "allocating diagonal staging memory failed", part.rank);
    cudaCheck(cudaHostAlloc(&hostSend, std::max<size_t>(1, part.rows * maxBlock) * sizeof(double),
                            cudaHostAllocPortable),
              "allocating send staging memory failed", part.rank);
    cudaCheck(cudaHostAlloc(&hostPanel, part.n * maxBlock * sizeof(double),
                            cudaHostAllocPortable),
              "allocating panel staging memory failed", part.rank);

    int workspaceSize = 0;
    cusolverCheck(cusolverDnDpotrf_bufferSize(
                      solver, CUBLAS_FILL_MODE_UPPER, static_cast<int>(maxBlock),
                      deviceDiagonal, static_cast<int>(maxBlock), &workspaceSize),
                  "querying Cholesky workspace failed", part.rank);
    cudaCheck(cudaMalloc(&deviceWork,
                         std::max<size_t>(1, static_cast<size_t>(workspaceSize)) * sizeof(double)),
              "cudaMalloc(Cholesky workspace) failed", part.rank);

    std::vector<int> receiveCounts(part.ranks);
    std::vector<int> receiveDisplacements(part.ranks);
    bool success = true;

    for (size_t k = 0; k < part.n;) {
        const int diagonalOwner = part.owner(k);
        // A diagonal block must stay inside one rank's contiguous row extent.
        const size_t ownerEnd = part.starts[diagonalOwner] + part.sizes[diagonalOwner];
        const size_t block = std::min(maxBlock, ownerEnd - k);
        const size_t end = k + block;
        int info = 0;

        if (part.rank == diagonalOwner) {
            const size_t localK = k - part.first;
            // Row-major lower storage is column-major upper storage.
            cusolverCheck(cusolverDnDpotrf(
                              solver, CUBLAS_FILL_MODE_UPPER, static_cast<int>(block),
                              deviceA + localK * part.n + k, static_cast<int>(part.n),
                              deviceWork, workspaceSize, deviceInfo),
                          "GPU diagonal factorization failed", part.rank);
            cudaCheck(cudaMemcpy(&info, deviceInfo, sizeof(int), cudaMemcpyDeviceToHost),
                      "reading Cholesky status failed", part.rank);
            if (info == 0) {
                cudaCheck(cudaMemcpy2D(hostDiagonal, block * sizeof(double),
                                       deviceA + localK * part.n + k,
                                       part.n * sizeof(double), block * sizeof(double), block,
                                       cudaMemcpyDeviceToHost),
                          "copying diagonal block failed", part.rank);
            }
        }
        MPI_Bcast(&info, 1, MPI_INT, diagonalOwner, MPI_COMM_WORLD);
        if (info != 0) {
            if (part.rank == 0)
                std::printf("Error: Matrix is not positive definite at diagonal block %zu\n", k);
            success = false;
            break;
        }
        MPI_Bcast(hostDiagonal, static_cast<int>(block * block), MPI_DOUBLE,
                  diagonalOwner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(deviceDiagonal, hostDiagonal, block * block * sizeof(double),
                             cudaMemcpyHostToDevice),
                  "copying diagonal block to GPU failed", part.rank);

        const size_t firstActive = std::max(end, part.first);
        const size_t activeRows = part.first + part.rows > firstActive
                                      ? part.first + part.rows - firstActive
                                      : 0;
        if (activeRows != 0) {
            const size_t localActive = firstActive - part.first;
            const double one = 1.0;
            // X^T=inv(Lkk)*Aik^T. deviceDiagonal is Lkk^T in column-major form.
            cublasCheck(cublasDtrsm(
                            blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                            static_cast<int>(block), static_cast<int>(activeRows), &one,
                            deviceDiagonal, static_cast<int>(block),
                            deviceA + localActive * part.n + k,
                            static_cast<int>(part.n)),
                        "cuBLAS panel solve failed", part.rank);
        }

        if (end < part.n) {
            if (part.rows != 0) {
                cudaCheck(cudaMemcpy2D(hostSend, block * sizeof(double), deviceA + k,
                                       part.n * sizeof(double), block * sizeof(double), part.rows,
                                       cudaMemcpyDeviceToHost),
                          "packing local panel failed", part.rank);
            }
            for (int r = 0; r < part.ranks; ++r) {
                const size_t count = part.sizes[r] * block;
                const size_t displacement = part.starts[r] * block;
                if (count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                    displacement > static_cast<size_t>(std::numeric_limits<int>::max()))
                    fail("MPI panel count exceeds the integer range", part.rank);
                receiveCounts[r] = static_cast<int>(count);
                receiveDisplacements[r] = static_cast<int>(displacement);
            }
            MPI_Allgatherv(hostSend, static_cast<int>(part.rows * block), MPI_DOUBLE,
                           hostPanel, receiveCounts.data(), receiveDisplacements.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
            cudaCheck(cudaMemcpy(devicePanel, hostPanel, part.n * block * sizeof(double),
                                 cudaMemcpyHostToDevice),
                      "copying global panel to GPU failed", part.rank);

            if (activeRows != 0) {
                const size_t localActive = firstActive - part.first;
                const size_t trailing = part.n - end;
                const double minusOne = -1.0;
                const double one = 1.0;
                // C_local^T -= P_global*P_local^T; a single large GEMM per rank.
                cublasCheck(cublasDgemm(
                                blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                static_cast<int>(trailing), static_cast<int>(activeRows),
                                static_cast<int>(block), &minusOne,
                                devicePanel + end * block, static_cast<int>(block),
                                deviceA + localActive * part.n + k,
                                static_cast<int>(part.n), &one,
                                deviceA + localActive * part.n + end,
                                static_cast<int>(part.n)),
                            "cuBLAS trailing update failed", part.rank);
            }
        }
        k = end;
    }
    cudaCheck(cudaDeviceSynchronize(), "factorization synchronization failed", part.rank);

    cudaFree(deviceWork);
    cudaFree(deviceInfo);
    cudaFree(devicePanel);
    cudaFree(deviceDiagonal);
    cudaFreeHost(hostPanel);
    cudaFreeHost(hostSend);
    cudaFreeHost(hostDiagonal);
    return success;
}

void zeroUpperTriangle(std::vector<double>& localA, const Partition& part) {
#pragma omp parallel for schedule(static)
    for (long long local = 0; local < static_cast<long long>(part.rows); ++local) {
        const size_t global = part.first + static_cast<size_t>(local);
        std::fill(localA.begin() + local * part.n + global + 1,
                  localA.begin() + (local + 1) * part.n, 0.0);
    }
}

bool validateDistributed(const std::vector<double>& L,
                         const std::vector<double>& original,
                         const Partition& part) {
    double maxAbsolute = 0.0;
    double maxRelative = 0.0;
    // Validation is optional; gather L only here so normal benchmark runs stay distributed.
    std::vector<int> counts(part.ranks), displacements(part.ranks);
    for (int r = 0; r < part.ranks; ++r) {
        const size_t count = part.sizes[r] * part.n;
        const size_t disp = part.starts[r] * part.n;
        if (count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            disp > static_cast<size_t>(std::numeric_limits<int>::max()))
            fail("MPI validation count exceeds the integer range", part.rank);
        counts[r] = static_cast<int>(count);
        displacements[r] = static_cast<int>(disp);
    }
    std::vector<double> fullL(part.n * part.n);
    MPI_Allgatherv(L.data(), static_cast<int>(L.size()), MPI_DOUBLE, fullL.data(),
                   counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    maxAbsolute = 0.0;
    maxRelative = 0.0;
#pragma omp parallel for collapse(2) schedule(static) reduction(max : maxAbsolute, maxRelative)
    for (long long local = 0; local < static_cast<long long>(part.rows); ++local) {
        for (long long column = 0; column < static_cast<long long>(part.n); ++column) {
            const size_t globalRow = part.first + static_cast<size_t>(local);
            const size_t j = static_cast<size_t>(column);
            double reconstructed = 0.0;
            for (size_t k = 0; k <= std::min(globalRow, j); ++k)
                reconstructed += L[static_cast<size_t>(local) * part.n + k] *
                                 fullL[j * part.n + k];
            const double expected = original[static_cast<size_t>(local) * part.n + j];
            const double error = std::fabs(reconstructed - expected);
            maxAbsolute = std::max(maxAbsolute, error);
            maxRelative = std::max(maxRelative, error / (std::fabs(expected) + 1e-10));
        }
    }
    double globalAbsolute = 0.0;
    double globalRelative = 0.0;
    MPI_Reduce(&maxAbsolute, &globalAbsolute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&maxRelative, &globalRelative, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (part.rank == 0) {
        std::printf("Max absolute error: %.10e\n", globalAbsolute);
        std::printf("Max relative error: %.10e\n", globalRelative);
        if (globalRelative > 1e-6)
            std::printf("Validation failed: relative error too large\n");
    }
    return globalRelative <= 1e-6;
}

std::vector<double> gatherResult(const std::vector<double>& localA,
                                 const Partition& part) {
    std::vector<int> counts(part.ranks), displacements(part.ranks);
    for (int r = 0; r < part.ranks; ++r) {
        const size_t count = part.sizes[r] * part.n;
        const size_t disp = part.starts[r] * part.n;
        if (count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            disp > static_cast<size_t>(std::numeric_limits<int>::max()))
            fail("MPI result count exceeds the integer range", part.rank);
        counts[r] = static_cast<int>(count);
        displacements[r] = static_cast<int>(disp);
    }
    std::vector<double> result(part.rank == 0 ? part.n * part.n : 0);
    MPI_Gatherv(localA.data(), static_cast<int>(localA.size()), MPI_DOUBLE,
                part.rank == 0 ? result.data() : nullptr, counts.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return result;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI does not provide thread-funneled support", rank);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool help = false;
    bool badArguments = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const long long parsed = std::atoll(argv[++i]);
            if (parsed < 0) badArguments = true;
            else n = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            badArguments = true;
        }
    }
    if (help || badArguments) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return badArguments ? 1 : 0;
    }
    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "Matrix size must be in [1, INT_MAX]\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0)
        fail("the hybrid benchmark requires at least one CUDA GPU per node", rank);
    cudaCheck(cudaSetDevice(localRank % deviceCount), "selecting CUDA device failed", rank);
    MPI_Comm_free(&localComm);

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP thread(s) per rank\n",
                    ranks, omp_get_max_threads());
        std::printf("Generating positive definite matrix...\n");
    }

    Partition part(n, rank, ranks);
    double* deviceA = nullptr;
    cudaCheck(cudaMalloc(&deviceA, std::max<size_t>(1, part.rows * n) * sizeof(double)),
              "cudaMalloc(local matrix) failed", rank);
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    cublasCheck(cublasCreate(&blas), "creating cuBLAS handle failed", rank);
    cusolverCheck(cusolverDnCreate(&solver), "creating cuSOLVER handle failed", rank);
    generateLocalMatrix(deviceA, part, blas);

    std::vector<double> original;
    if (validate) {
        original.resize(part.rows * n);
        if (!original.empty())
            cudaCheck(cudaMemcpy(original.data(), deviceA, original.size() * sizeof(double),
                                 cudaMemcpyDeviceToHost),
                      "saving original matrix failed", rank);
    }

    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    const bool success = distributedCholesky(deviceA, part, blas, solver);
    const auto stop = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(stop - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> localA(part.rows * n);
    if (success && !localA.empty())
        cudaCheck(cudaMemcpy(localA.data(), deviceA, localA.size() * sizeof(double),
                             cudaMemcpyDeviceToHost),
                  "copying result from GPU failed", rank);
    cublasDestroy(blas);
    cusolverDnDestroy(solver);
    cudaFree(deviceA);

    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    zeroUpperTriangle(localA, part);

    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(seconds * 1000.0);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        std::printf("Computation time: %lld ms\n", milliseconds);
        std::printf("Performance: %.3f GFLOPS\n", operations / seconds / 1e9);
    }

    if (printResults) {
        std::vector<double> result = gatherResult(localA, part);
        if (rank == 0) print_results(result, "CholeskyL");
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validateDistributed(localA, original, part);
        int allValid = valid ? 1 : 0;
        MPI_Bcast(&allValid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        valid = allValid != 0;
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    MPI_Finalize();
    return valid ? 0 : 1;
}
