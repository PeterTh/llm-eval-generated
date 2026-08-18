#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// The matrix is distributed by contiguous row blocks.  A blocked Cholesky step
// is consequently:
//   1. POTRF of the diagonal block on its owning GPU,
//   2. broadcast of that block, TRSM of every local block column on the GPUs,
//   3. all-gather of the block column, then a local GPU GEMM update.
// This keeps the O(n^3) work on accelerators and communicates only O(n^2)
// panel data.  The block size is large enough for efficient DGEMM while still
// exposing MPI parallelism across a cluster.
constexpr size_t kBlockSize = 256;
constexpr int kCudaThreads = 256;

[[noreturn]] void abortWithMessage(const char* where, const char* message) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    int rank = -1;
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    std::fprintf(stderr, "Rank %d: %s: %s\n", rank, where, message);
    if (initialized) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::abort();
}

void checkCuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "%s failed at %s:%d (%s)",
                      expression, file, line, cudaGetErrorString(status));
        abortWithMessage("CUDA", message);
    }
}

void checkCublas(cublasStatus_t status, const char* expression, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        char message[512];
        std::snprintf(message, sizeof(message), "%s failed at %s:%d (status %d)",
                      expression, file, line, static_cast<int>(status));
        abortWithMessage("cuBLAS", message);
    }
}

void checkCusolver(cusolverStatus_t status, const char* expression, const char* file, int line) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        char message[512];
        std::snprintf(message, sizeof(message), "%s failed at %s:%d (status %d)",
                      expression, file, line, static_cast<int>(status));
        abortWithMessage("cuSOLVER", message);
    }
}

void checkMpi(int status, const char* expression, const char* file, int line) {
    if (status != MPI_SUCCESS) {
        char mpiError[MPI_MAX_ERROR_STRING]{};
        int length = 0;
        MPI_Error_string(status, mpiError, &length);
        char message[768];
        std::snprintf(message, sizeof(message), "%s failed at %s:%d (%.*s)", expression,
                      file, line, length, mpiError);
        abortWithMessage("MPI", message);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)
#define CUBLAS_CHECK(call) checkCublas((call), #call, __FILE__, __LINE__)
#define CUSOLVER_CHECK(call) checkCusolver((call), #call, __FILE__, __LINE__)
#define MPI_CHECK(call) checkMpi((call), #call, __FILE__, __LINE__)

__global__ void zeroUpperTriangle(double* matrix, size_t localRows, size_t n, size_t globalRowStart) {
    const size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements = localRows * n;
    if (element >= elements) {
        return;
    }
    const size_t localRow = element / n;
    const size_t column = element - localRow * n;
    if (column > globalRowStart + localRow) {
        matrix[element] = 0.0;
    }
}

int mpiCount(size_t count, const char* name) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        abortWithMessage("MPI count", name);
    }
    return static_cast<int>(count);
}

void createRowPartition(size_t n, int ranks, std::vector<size_t>& starts, std::vector<size_t>& rows) {
    starts.resize(ranks);
    rows.resize(ranks);
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t remainder = n % static_cast<size_t>(ranks);
    size_t offset = 0;
    for (int rank = 0; rank < ranks; ++rank) {
        const size_t count = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
        starts[rank] = offset;
        rows[rank] = count;
        offset += count;
    }
}

int ownerOfRow(size_t row, const std::vector<size_t>& starts, const std::vector<size_t>& rows) {
    const auto it = std::upper_bound(starts.begin(), starts.end(), row);
    const int candidate = static_cast<int>(it - starts.begin()) - 1;
    if (candidate >= 0 && row < starts[candidate] + rows[candidate]) {
        return candidate;
    }
    abortWithMessage("row distribution", "could not locate diagonal-block owner");
}

// Generate exactly the same SPD input as the original benchmark.  The random
// stream remains serial for reproducibility; the independent output rows use
// OpenMP and preserve each dot product's original accumulation order.
void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[static_cast<size_t>(i) * n + k] * B[j * n + k];
            }
            A[static_cast<size_t>(i) * n + j] = sum;
        }
    }

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        A[static_cast<size_t>(i) * n + static_cast<size_t>(i)] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& Aorig, size_t n) {
    double maxError = 0.0;
    double relativeError = 0.0;
    const size_t elements = n * n;

#pragma omp parallel for reduction(max : maxError, relativeError) schedule(static)
    for (long long element = 0; element < static_cast<long long>(elements); ++element) {
        const size_t i = static_cast<size_t>(element) / n;
        const size_t j = static_cast<size_t>(element) - i * n;
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) {
            sum += L[i * n + k] * L[j * n + k];
        }
        const double error = std::fabs(sum - Aorig[static_cast<size_t>(element)]);
        maxError = std::max(maxError, error);
        relativeError = std::max(relativeError,
                                 error / (std::fabs(Aorig[static_cast<size_t>(element)]) + 1e-10));
    }

    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relativeError);
    if (relativeError > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

struct GpuContext {
    cublasHandle_t cublas{};
    cusolverDnHandle_t solver{};
    double* matrix{};
    double* diagonal{};
    double* panel{};
    double* workspace{};
    int* info{};
    int workspaceElements{};
};

void destroyGpuContext(GpuContext& gpu) {
    if (gpu.info != nullptr) CUDA_CHECK(cudaFree(gpu.info));
    if (gpu.workspace != nullptr) CUDA_CHECK(cudaFree(gpu.workspace));
    if (gpu.panel != nullptr) CUDA_CHECK(cudaFree(gpu.panel));
    if (gpu.diagonal != nullptr) CUDA_CHECK(cudaFree(gpu.diagonal));
    if (gpu.matrix != nullptr) CUDA_CHECK(cudaFree(gpu.matrix));
    if (gpu.solver != nullptr) CUSOLVER_CHECK(cusolverDnDestroy(gpu.solver));
    if (gpu.cublas != nullptr) CUBLAS_CHECK(cublasDestroy(gpu.cublas));
}

void initialiseGpuContext(GpuContext& gpu, size_t localRows, size_t n, size_t blockSize) {
    const size_t matrixElements = std::max<size_t>(1, localRows * n);
    const size_t diagonalElements = std::max<size_t>(1, blockSize * blockSize);
    const size_t panelElements = std::max<size_t>(1, n * blockSize);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu.matrix), matrixElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu.diagonal), diagonalElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu.panel), panelElements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu.info), sizeof(int)));
    CUBLAS_CHECK(cublasCreate(&gpu.cublas));
    CUSOLVER_CHECK(cusolverDnCreate(&gpu.solver));
    CUBLAS_CHECK(cublasSetMathMode(gpu.cublas, CUBLAS_DEFAULT_MATH));

    // The workspace size is monotonic in the POTRF order, so one allocation
    // for the maximum block serves all (possibly shortened) edge blocks.
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(gpu.solver, CUBLAS_FILL_MODE_UPPER,
                                               mpiCount(blockSize, "POTRF block size"), gpu.matrix,
                                               mpiCount(n, "matrix leading dimension"),
                                               &gpu.workspaceElements));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu.workspace),
                          std::max(1, gpu.workspaceElements) * sizeof(double)));
}

bool distributedCholesky(GpuContext& gpu, size_t n, size_t blockSize, int rank,
                         const std::vector<size_t>& starts, const std::vector<size_t>& rows) {
    const size_t localStart = starts[rank];
    const size_t localRows = rows[rank];
    const size_t localEnd = localStart + localRows;
    const int ranks = static_cast<int>(rows.size());
    std::vector<double> hostDiagonal(blockSize * blockSize);
    std::vector<double> hostLocalPanel(std::max<size_t>(1, localRows * blockSize));
    std::vector<double> hostGlobalPanel(std::max<size_t>(1, n * blockSize));
    std::vector<int> receiveCounts(ranks);
    std::vector<int> displacements(ranks);

    const double one = 1.0;
    const double minusOne = -1.0;
    for (size_t k = 0; k < n;) {
        const int owner = ownerOfRow(k, starts, rows);
        // A panel is kept within its owner.  This avoids a diagonal-block
        // redistribution and only makes an extra (short) panel at a rank cut.
        const size_t panelSize = std::min({blockSize, n - k, starts[owner] + rows[owner] - k});
        const int panelSizeInt = mpiCount(panelSize, "panel size");
        int panelStatus = 0;

        if (rank == owner) {
            const size_t localDiagonalRow = k - localStart;
            double* diagonalInMatrix = gpu.matrix + localDiagonalRow * n + k;
            CUDA_CHECK(cudaMemset(gpu.info, 0, sizeof(int)));
            // Row-major lower-triangular storage is column-major upper storage.
            CUSOLVER_CHECK(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_UPPER, panelSizeInt,
                                            diagonalInMatrix, mpiCount(n, "matrix leading dimension"),
                                            gpu.workspace, gpu.workspaceElements, gpu.info));
            CUDA_CHECK(cudaMemcpy(&panelStatus, gpu.info, sizeof(int), cudaMemcpyDeviceToHost));
            if (panelStatus == 0) {
                CUDA_CHECK(cudaMemcpy2D(hostDiagonal.data(), panelSize * sizeof(double),
                                        diagonalInMatrix, n * sizeof(double), panelSize * sizeof(double),
                                        panelSize, cudaMemcpyDeviceToHost));
            }
        }
        MPI_CHECK(MPI_Bcast(&panelStatus, 1, MPI_INT, owner, MPI_COMM_WORLD));
        if (panelStatus != 0) {
            if (rank == 0) {
                std::printf("Error: Matrix is not positive definite at diagonal block starting at %zu\n", k);
            }
            return false;
        }

        MPI_CHECK(MPI_Bcast(hostDiagonal.data(), mpiCount(panelSize * panelSize, "diagonal panel"),
                            MPI_DOUBLE, owner, MPI_COMM_WORLD));
        CUDA_CHECK(cudaMemcpy(gpu.diagonal, hostDiagonal.data(), panelSize * panelSize * sizeof(double),
                              cudaMemcpyHostToDevice));

        const size_t trailingStart = k + panelSize;
        const size_t solveStart = std::max(trailingStart, localStart);
        const size_t solveRows = solveStart < localEnd ? localEnd - solveStart : 0;
        if (solveRows != 0) {
            // In row-major notation L_ik = A_ik * inv(L_kk^T).  Viewing both
            // matrices in column-major order turns this into a left TRSM.
            CUBLAS_CHECK(cublasDtrsm(gpu.cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                     CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, panelSizeInt,
                                     mpiCount(solveRows, "TRSM row count"), &one, gpu.diagonal,
                                     panelSizeInt, gpu.matrix + (solveStart - localStart) * n + k,
                                     mpiCount(n, "matrix leading dimension")));
        }

        const size_t globalPanelRows = n - trailingStart;
        for (int process = 0; process < ranks; ++process) {
            const size_t begin = std::max(trailingStart, starts[process]);
            const size_t end = starts[process] + rows[process];
            const size_t processRows = begin < end ? end - begin : 0;
            receiveCounts[process] = mpiCount(processRows * panelSize, "panel receive count");
            displacements[process] = mpiCount((begin - trailingStart) * panelSize,
                                              "panel receive displacement");
        }

        if (solveRows != 0) {
            CUDA_CHECK(cudaMemcpy2D(hostLocalPanel.data(), panelSize * sizeof(double),
                                    gpu.matrix + (solveStart - localStart) * n + k, n * sizeof(double),
                                    panelSize * sizeof(double), solveRows, cudaMemcpyDeviceToHost));
        }
        MPI_CHECK(MPI_Allgatherv(solveRows == 0 ? nullptr : hostLocalPanel.data(),
                                 mpiCount(solveRows * panelSize, "panel send count"), MPI_DOUBLE,
                                 hostGlobalPanel.data(), receiveCounts.data(), displacements.data(), MPI_DOUBLE,
                                 MPI_COMM_WORLD));
        if (globalPanelRows != 0) {
            CUDA_CHECK(cudaMemcpy(gpu.panel, hostGlobalPanel.data(),
                                  globalPanelRows * panelSize * sizeof(double), cudaMemcpyHostToDevice));
        }

        if (solveRows != 0 && globalPanelRows != 0) {
            // C^T = P * Q^T, where P is the gathered block column and Q is
            // this rank's local rows.  The transposed row-major view lets
            // cuBLAS DGEMM update the local trailing matrix in place.
            CUBLAS_CHECK(cublasDgemm(gpu.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                     mpiCount(globalPanelRows, "GEMM columns"),
                                     mpiCount(solveRows, "GEMM rows"), panelSizeInt, &minusOne,
                                     gpu.panel, panelSizeInt,
                                     gpu.matrix + (solveStart - localStart) * n + k,
                                     mpiCount(n, "matrix leading dimension"), &one,
                                     gpu.matrix + (solveStart - localStart) * n + trailingStart,
                                     mpiCount(n, "matrix leading dimension")));
        }
        k = trailingStart;
    }

    if (localRows != 0) {
        const size_t elements = localRows * n;
        const size_t blocks = (elements + kCudaThreads - 1) / kCudaThreads;
        zeroUpperTriangle<<<mpiCount(blocks, "CUDA grid size"), kCudaThreads>>>(gpu.matrix, localRows, n,
                                                                                   localStart);
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    if (n == 0 || n > std::numeric_limits<size_t>::max() / n) {
        std::fprintf(stderr, "Matrix size must be positive and fit in memory indexing.\n");
        return 1;
    }

    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) {
        abortWithMessage("MPI", "MPI implementation does not provide MPI_THREAD_FUNNELED");
    }

    int rank = 0;
    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));

    MPI_Comm localComm{};
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        abortWithMessage("CUDA", "no CUDA device is available");
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    std::vector<size_t> rowStarts;
    std::vector<size_t> rowCounts;
    createRowPartition(n, ranks, rowStarts, rowCounts);
    const size_t localRows = rowCounts[rank];
    const size_t blockSize = std::min(kBlockSize, n);

    std::vector<int> matrixCounts(ranks);
    std::vector<int> matrixDisplacements(ranks);
    for (int process = 0; process < ranks; ++process) {
        matrixCounts[process] = mpiCount(rowCounts[process] * n, "matrix scatter count");
        matrixDisplacements[process] = mpiCount(rowStarts[process] * n, "matrix scatter displacement");
    }

    std::vector<double> matrix;
    std::vector<double> original;
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark (MPI + OpenMP + CUDA)\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("MPI ranks: %d\n", ranks);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        std::printf("CUDA block size: %zu\n", blockSize);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
        matrix.resize(n * n);
        generatePositiveDefiniteMatrix(matrix, n);
        if (validate) {
            original = matrix;
        }
    }

    std::vector<double> localMatrix(std::max<size_t>(1, localRows * n));
    MPI_CHECK(MPI_Scatterv(rank == 0 ? matrix.data() : nullptr, matrixCounts.data(),
                           matrixDisplacements.data(), MPI_DOUBLE,
                           localRows == 0 ? nullptr : localMatrix.data(),
                           mpiCount(localRows * n, "local matrix count"), MPI_DOUBLE, 0,
                           MPI_COMM_WORLD));

    GpuContext gpu;
    initialiseGpuContext(gpu, localRows, n, blockSize);
    if (localRows != 0) {
        CUDA_CHECK(cudaMemcpy(gpu.matrix, localMatrix.data(), localRows * n * sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    const bool success = distributedCholesky(gpu, n, blockSize, rank, rowStarts, rowCounts);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsedSeconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (success && localRows != 0) {
        CUDA_CHECK(cudaMemcpy(localMatrix.data(), gpu.matrix, localRows * n * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }
    if (success) {
        if (rank == 0) {
            matrix.assign(n * n, 0.0);
        }
        MPI_CHECK(MPI_Gatherv(localRows == 0 ? nullptr : localMatrix.data(),
                              mpiCount(localRows * n, "local result count"), MPI_DOUBLE,
                              rank == 0 ? matrix.data() : nullptr, matrixCounts.data(),
                              matrixDisplacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
    }

    destroyGpuContext(gpu);
    MPI_CHECK(MPI_Comm_free(&localComm));

    int exitCode = success ? 0 : 1;
    if (rank == 0) {
        if (!success) {
            std::printf("Cholesky decomposition failed\n");
        } else {
            const long long milliseconds = static_cast<long long>(std::llround(elapsedSeconds * 1000.0));
            const double operations = static_cast<double>(n) * n * n / 3.0;
            const double gflops = elapsedSeconds > 0.0 ? operations / elapsedSeconds / 1.0e9 : 0.0;
            std::printf("Computation time: %lld ms\n", milliseconds);
            std::printf("Performance: %.3f GFLOPS\n", gflops);
            if (printResults) {
                print_results(matrix, "CholeskyL");
            }
            if (validate) {
                std::printf("Validating result...\n");
                if (validateCholesky(matrix, original, n)) {
                    std::printf("Validation: PASSED\n");
                } else {
                    std::printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
    }
    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
