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

[[noreturn]] void fatal(const char* operation, const char* detail, int rank) {
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation, detail);
    MPI_Abort(MPI_COMM_WORLD, 2);
    std::abort();
}

void cudaCheck(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) fatal(operation, cudaGetErrorString(status), rank);
}

void cublasCheck(cublasStatus_t status, const char* operation, int rank) {
    if (status != CUBLAS_STATUS_SUCCESS) fatal(operation, "cuBLAS error", rank);
}

void cusolverCheck(cusolverStatus_t status, const char* operation, int rank) {
    if (status != CUSOLVER_STATUS_SUCCESS) fatal(operation, "cuSOLVER error", rank);
}

void mpiCheck(int status, const char* operation, int rank) {
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, message, &length);
    message[length] = '\0';
    fatal(operation, message, rank);
}

// MPI collectives traditionally take an int count.  Chunking keeps large matrices valid.
void broadcastDoubles(double* data, size_t count, int root, int rank) {
    constexpr size_t chunk = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t offset = 0; offset < count; offset += chunk) {
        const int n = static_cast<int>(std::min(chunk, count - offset));
        mpiCheck(MPI_Bcast(data + offset, n, MPI_DOUBLE, root, MPI_COMM_WORLD),
                 "MPI_Bcast", rank);
    }
}

__global__ void addDiagonal(double* matrix, int n, double value) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) matrix[static_cast<size_t>(i) * n + i] += value;
}

struct GpuContext {
    int rank;
    cublasHandle_t blas{};
    cusolverDnHandle_t solver{};
    double* matrix = nullptr;
    double* panel = nullptr;
    double* diagonal = nullptr;
    int* info = nullptr;
    void* workspace = nullptr;
    size_t workspaceBytes = 0;
    double* hostPanel = nullptr;
    double* hostDiagonal = nullptr;

    GpuContext(size_t elements, int n, int rankIn) : rank(rankIn) {
        MPI_Comm localComm;
        mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                     &localComm), "MPI_Comm_split_type", rank);
        int localRank = 0;
        MPI_Comm_rank(localComm, &localRank);
        MPI_Comm_free(&localComm);

        int deviceCount = 0;
        cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
        if (deviceCount == 0) fatal("GPU selection", "no CUDA device is available", rank);
        cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", rank);
        cudaCheck(cudaFree(nullptr), "CUDA context initialization", rank);
        cublasCheck(cublasCreate(&blas), "cublasCreate", rank);
        cusolverCheck(cusolverDnCreate(&solver), "cusolverDnCreate", rank);
        cublasCheck(cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH),
                    "cublasSetMathMode", rank);
        cudaCheck(cudaMalloc(&matrix, elements * sizeof(double)), "cudaMalloc(matrix)", rank);
        cudaCheck(cudaMalloc(&panel, static_cast<size_t>(kBlockSize) * n * sizeof(double)),
                  "cudaMalloc(panel)", rank);
        cudaCheck(cudaMalloc(&diagonal,
                             static_cast<size_t>(kBlockSize) * kBlockSize * sizeof(double)),
                  "cudaMalloc(diagonal)", rank);
        cudaCheck(cudaMalloc(&info, sizeof(int)), "cudaMalloc(info)", rank);
        cudaCheck(cudaHostAlloc(&hostPanel, static_cast<size_t>(kBlockSize) * n * sizeof(double),
                                cudaHostAllocPortable), "cudaHostAlloc(panel)", rank);
        cudaCheck(cudaHostAlloc(&hostDiagonal,
                                static_cast<size_t>(kBlockSize) * kBlockSize * sizeof(double),
                                cudaHostAllocPortable), "cudaHostAlloc(diagonal)", rank);
    }

    void ensureWorkspace(size_t bytes) {
        if (bytes <= workspaceBytes) return;
        if (workspace) cudaCheck(cudaFree(workspace), "cudaFree(workspace)", rank);
        cudaCheck(cudaMalloc(&workspace, bytes), "cudaMalloc(workspace)", rank);
        workspaceBytes = bytes;
    }

    ~GpuContext() {
        if (hostDiagonal) cudaFreeHost(hostDiagonal);
        if (hostPanel) cudaFreeHost(hostPanel);
        if (workspace) cudaFree(workspace);
        if (info) cudaFree(info);
        if (diagonal) cudaFree(diagonal);
        if (panel) cudaFree(panel);
        if (matrix) cudaFree(matrix);
        if (solver) cusolverDnDestroy(solver);
        if (blas) cublasDestroy(blas);
    }
};

// Build A=B*B^T+nI with the expensive matrix product on rank zero's GPU.
void generatePositiveDefiniteMatrix(std::vector<double>& A, int n, GpuContext& gpu, int rank) {
    if (rank != 0) return;
    const size_t elements = static_cast<size_t>(n) * n;
    std::vector<double> B(elements);
    unsigned int seed = 42;
    for (size_t i = 0; i < elements; ++i)
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;

    double* deviceB = nullptr;
    cudaCheck(cudaMalloc(&deviceB, elements * sizeof(double)), "cudaMalloc(B)", rank);
    cudaCheck(cudaMemcpy(deviceB, B.data(), elements * sizeof(double), cudaMemcpyHostToDevice),
              "copy B to GPU", rank);
    const double one = 1.0;
    const double zero = 0.0;
    // Row-major B is column-major B^T, hence (B^T)^T B^T = B B^T.
    cublasCheck(cublasDgemm(gpu.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, deviceB, n,
                            deviceB, n, &zero, gpu.matrix, n),
                "cublasDgemm(matrix generation)", rank);
    addDiagonal<<<(n + 255) / 256, 256>>>(gpu.matrix, n, static_cast<double>(n));
    cudaCheck(cudaGetLastError(), "addDiagonal kernel", rank);
    cudaCheck(cudaMemcpy(A.data(), gpu.matrix, elements * sizeof(double), cudaMemcpyDeviceToHost),
              "copy generated matrix", rank);
    cudaCheck(cudaFree(deviceB), "cudaFree(B)", rank);
}

// One-dimensional block-cyclic, right-looking Cholesky.  The bytes of a symmetric
// row-major A are also a column-major A.  Factoring its upper triangle therefore
// produces the requested row-major lower triangle without a transpose.
bool choleskyDecomposition(std::vector<double>& A, int n, GpuContext& gpu,
                           int rank, int ranks) {
    const size_t elements = static_cast<size_t>(n) * n;
    broadcastDoubles(A.data(), elements, 0, rank);
    if (rank != 0)
        cudaCheck(cudaMemcpy(gpu.matrix, A.data(), elements * sizeof(double),
                             cudaMemcpyHostToDevice), "copy input matrix", rank);

    const double one = 1.0;
    const double minusOne = -1.0;
    const int blockCount = (n + kBlockSize - 1) / kBlockSize;

    for (int block = 0; block < blockCount; ++block) {
        const int k = block * kBlockSize;
        const int kb = std::min(kBlockSize, n - k);
        const int owner = block % ranks;
        int factorInfo = 0;

        if (rank == owner) {
            int workspaceElements = 0;
            cusolverCheck(cusolverDnDpotrf_bufferSize(
                              gpu.solver, CUBLAS_FILL_MODE_UPPER, kb,
                              gpu.matrix + static_cast<size_t>(k) * n + k, n,
                              &workspaceElements),
                          "cusolverDnDpotrf_bufferSize", rank);
            gpu.ensureWorkspace(static_cast<size_t>(workspaceElements) * sizeof(double));
            cusolverCheck(cusolverDnDpotrf(
                              gpu.solver, CUBLAS_FILL_MODE_UPPER, kb,
                              gpu.matrix + static_cast<size_t>(k) * n + k, n,
                              static_cast<double*>(gpu.workspace), workspaceElements, gpu.info),
                          "cusolverDnDpotrf", rank);
            cudaCheck(cudaMemcpy(&factorInfo, gpu.info, sizeof(int), cudaMemcpyDeviceToHost),
                      "copy factorization status", rank);
            cudaCheck(cudaMemcpy2D(gpu.hostDiagonal, kb * sizeof(double),
                                   gpu.matrix + static_cast<size_t>(k) * n + k,
                                   n * sizeof(double), kb * sizeof(double), kb,
                                   cudaMemcpyDeviceToHost), "stage diagonal block", rank);
        }
        mpiCheck(MPI_Bcast(&factorInfo, 1, MPI_INT, owner, MPI_COMM_WORLD),
                 "MPI_Bcast(factor status)", rank);
        if (factorInfo != 0) {
            if (rank == 0)
                std::printf("Error: Matrix is not positive definite at diagonal block %d\n", block);
            return false;
        }
        mpiCheck(MPI_Bcast(gpu.hostDiagonal, kb * kb, MPI_DOUBLE, owner, MPI_COMM_WORLD),
                 "MPI_Bcast(diagonal)", rank);
        if (rank != owner)
            cudaCheck(cudaMemcpy2D(gpu.matrix + static_cast<size_t>(k) * n + k,
                                   n * sizeof(double), gpu.hostDiagonal, kb * sizeof(double),
                                   kb * sizeof(double), kb, cudaMemcpyHostToDevice),
                      "install diagonal block", rank);

        // Each rank computes the panel tiles belonging to its block columns.
        for (int columnBlock = block + 1; columnBlock < blockCount; ++columnBlock) {
            if (columnBlock % ranks != rank) continue;
            const int j = columnBlock * kBlockSize;
            const int jb = std::min(kBlockSize, n - j);
            cublasCheck(cublasDtrsm(gpu.blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, kb, jb, &one,
                                    gpu.matrix + static_cast<size_t>(k) * n + k, n,
                                    gpu.matrix + static_cast<size_t>(j) * n + k, n),
                        "cublasDtrsm(panel)", rank);
        }

        const int remaining = n - (k + kb);
        if (remaining == 0) continue;
        const size_t panelElements = static_cast<size_t>(kb) * remaining;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < panelElements; ++i) gpu.hostPanel[i] = 0.0;

        for (int columnBlock = block + 1; columnBlock < blockCount; ++columnBlock) {
            if (columnBlock % ranks != rank) continue;
            const int j = columnBlock * kBlockSize;
            const int jb = std::min(kBlockSize, n - j);
            cudaCheck(cudaMemcpy2D(gpu.hostPanel + static_cast<size_t>(j - k - kb) * kb,
                                   kb * sizeof(double),
                                   gpu.matrix + static_cast<size_t>(j) * n + k,
                                   n * sizeof(double), kb * sizeof(double), jb,
                                   cudaMemcpyDeviceToHost), "stage local panel", rank);
        }
        mpiCheck(MPI_Allreduce(MPI_IN_PLACE, gpu.hostPanel, static_cast<int>(panelElements),
                               MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
                 "MPI_Allreduce(panel)", rank);
        cudaCheck(cudaMemcpy2D(gpu.matrix + static_cast<size_t>(k + kb) * n + k,
                               n * sizeof(double), gpu.hostPanel, kb * sizeof(double),
                               kb * sizeof(double), remaining, cudaMemcpyHostToDevice),
                  "install global panel", rank);

        // Update only owned block columns.  Diagonal tiles use DSYRK and off-diagonal
        // tiles use DGEMM, retaining all work on the accelerator.
        for (int columnBlock = block + 1; columnBlock < blockCount; ++columnBlock) {
            if (columnBlock % ranks != rank) continue;
            const int j = columnBlock * kBlockSize;
            const int jb = std::min(kBlockSize, n - j);
            for (int rowBlock = block + 1; rowBlock <= columnBlock; ++rowBlock) {
                const int i = rowBlock * kBlockSize;
                const int ib = std::min(kBlockSize, n - i);
                if (rowBlock == columnBlock) {
                    cublasCheck(cublasDsyrk(gpu.blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                            ib, kb, &minusOne,
                                            gpu.matrix + static_cast<size_t>(i) * n + k, n,
                                            &one, gpu.matrix + static_cast<size_t>(i) * n + i, n),
                                "cublasDsyrk(trailing diagonal)", rank);
                } else {
                    cublasCheck(cublasDgemm(gpu.blas, CUBLAS_OP_T, CUBLAS_OP_N, ib, jb, kb,
                                            &minusOne,
                                            gpu.matrix + static_cast<size_t>(i) * n + k, n,
                                            gpu.matrix + static_cast<size_t>(j) * n + k, n,
                                            &one, gpu.matrix + static_cast<size_t>(j) * n + i, n),
                                "cublasDgemm(trailing update)", rank);
                }
            }
        }
        cudaCheck(cudaDeviceSynchronize(), "trailing update synchronization", rank);
    }

    // Gather column-major upper block columns.  In row-major interpretation these
    // are exactly the lower block rows expected by the original program.
    constexpr int gatherTag = 1701;
    for (int block = 0; block < blockCount; ++block) {
        const int start = block * kBlockSize;
        const int width = std::min(kBlockSize, n - start);
        const int owner = block % ranks;
        const size_t count = static_cast<size_t>(n) * width;
        if (rank == owner) {
            cudaCheck(cudaMemcpy(A.data() + static_cast<size_t>(start) * n,
                                 gpu.matrix + static_cast<size_t>(start) * n,
                                 count * sizeof(double), cudaMemcpyDeviceToHost),
                      "gather result from GPU", rank);
            if (owner != 0)
                mpiCheck(MPI_Send(A.data() + static_cast<size_t>(start) * n,
                                  static_cast<int>(count), MPI_DOUBLE, 0, gatherTag,
                                  MPI_COMM_WORLD), "MPI_Send(result block)", rank);
        } else if (rank == 0) {
            mpiCheck(MPI_Recv(A.data() + static_cast<size_t>(start) * n,
                              static_cast<int>(count), MPI_DOUBLE, owner, gatherTag,
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE), "MPI_Recv(result block)", rank);
        }
    }
    if (rank == 0) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) A[static_cast<size_t>(i) * n + j] = 0.0;
    }
    return true;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& original, int n) {
    double maxError = 0.0;
    double relError = 0.0;
#pragma omp parallel for reduction(max : maxError, relError) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            const int end = std::min(i, j);
#pragma omp simd reduction(+ : sum)
            for (int k = 0; k <= end; ++k)
                sum += L[static_cast<size_t>(i) * n + k] * L[static_cast<size_t>(j) * n + k];
            const size_t index = static_cast<size_t>(i) * n + j;
            const double error = std::fabs(sum - original[index]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::fabs(original[index]) + 1e-10));
        }
    }
    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
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

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) fatal("MPI_Init_thread", "MPI_THREAD_FUNNELED unavailable", rank);

    int n = 512;
    bool validate = false;
    bool printResults = false;
    bool help = false;
    bool argumentsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const long value = std::strtol(argv[++i], &end, 10);
            argumentsValid = end && *end == '\0' && value > 0 && value <= std::numeric_limits<int>::max();
            if (argumentsValid) n = static_cast<int>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else argumentsValid = false;
    }
    if (help || !argumentsValid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\n");
        std::printf("Matrix size: %d x %d\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: %d MPI rank(s), up to %d OpenMP thread(s), CUDA GPUs\n",
                    ranks, omp_get_max_threads());
    }

    const size_t elements = static_cast<size_t>(n) * n;
    if (elements > std::numeric_limits<size_t>::max() / sizeof(double))
        fatal("matrix allocation", "matrix size overflow", rank);
    std::vector<double> A(elements);
    std::vector<double> original;
    GpuContext gpu(elements, n, rank);

    if (rank == 0) std::printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n, gpu, rank);
    if (rank == 0 && validate) original = A;

    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(A, n, gpu, rank, ranks);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int exitCode = success ? 0 : 1;
    if (rank == 0 && success) {
        const long long milliseconds = static_cast<long long>(maxElapsed * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        const double gflops = maxElapsed > 0.0 ? operations / maxElapsed / 1e9 : 0.0;
        std::printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(A, "CholeskyL");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateCholesky(A, original, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) exitCode = 1;
        }
    } else if (rank == 0) {
        std::printf("Cholesky decomposition failed\n");
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
