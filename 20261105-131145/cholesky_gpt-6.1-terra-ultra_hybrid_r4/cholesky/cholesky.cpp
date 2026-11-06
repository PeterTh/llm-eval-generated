#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {
constexpr int block_size = 256;
int rank_id = 0;

[[noreturn]] void fail(const char* operation, int code) {
    fprintf(stderr, "Rank %d: %s failed (%d)\n", rank_id, operation, code);
    MPI_Abort(MPI_COMM_WORLD, code ? code : 1);
    std::abort();
}
#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "Rank %d: %s: %s\n", rank_id, #call, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); std::abort(); } } while (0)
#define BLAS_CHECK(call) do { cublasStatus_t s = (call); if (s != CUBLAS_STATUS_SUCCESS) fail(#call, static_cast<int>(s)); } while (0)
#define SOLVER_CHECK(call) do { cusolverStatus_t s = (call); if (s != CUSOLVER_STATUS_SUCCESS) fail(#call, static_cast<int>(s)); } while (0)
#define MPI_CHECK(call) do { int s = (call); if (s != MPI_SUCCESS) fail(#call, s); } while (0)

// MPI counts are int, but a panel or gathered matrix can contain more elements.
void bcast_doubles(double* data, size_t count, int root) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_CHECK(MPI_Bcast(data, chunk, MPI_DOUBLE, root, MPI_COMM_WORLD));
        data += chunk;
        count -= chunk;
    }
}
void transfer_doubles(double* data, size_t count, int peer, bool send) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_CHECK(MPI_Send(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD));
        else MPI_CHECK(MPI_Recv(data, chunk, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        data += chunk;
        count -= chunk;
    }
}

__global__ void add_diagonal(double* matrix, int n, int global_col, int local_col, int width) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < width) matrix[static_cast<size_t>(local_col + j) * n + global_col + j] += n;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Collect column blocks and convert to the benchmark's row-major representation.
std::vector<double> gather_matrix(const double* device, int n, int world, bool lower) {
    std::vector<double> result;
    if (rank_id == 0) result.resize(static_cast<size_t>(n) * n);
    std::vector<double> block(static_cast<size_t>(n) * std::min(n, block_size));
    int blocks = (n + static_cast<long long>(block_size) - 1) / block_size;
    for (int b = 0; b < blocks; ++b) {
        int first = b * block_size;
        int width = std::min(block_size, n - first);
        int owner = b % world;
        size_t count = static_cast<size_t>(n) * width;
        if (rank_id == owner) {
            int local = (b / world) * block_size;
            CUDA_CHECK(cudaMemcpy(block.data(), device + static_cast<size_t>(local) * n,
                                  count * sizeof(double), cudaMemcpyDeviceToHost));
            if (owner != 0) transfer_doubles(block.data(), count, 0, true);
        }
        if (rank_id == 0) {
            if (owner != 0) transfer_doubles(block.data(), count, owner, false);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < width; ++j) {
                    result[static_cast<size_t>(i) * n + first + j] =
                        lower && i < first + j ? 0.0 : block[static_cast<size_t>(j) * n + i];
                }
            }
        }
    }
    return result;
}

bool validateCholesky(std::vector<double>& factor, const std::vector<double>& original,
                      double* matrix, size_t matrix_count, int n, int world,
                      cublasHandle_t blas) {
    if (rank_id != 0) factor.resize(static_cast<size_t>(n) * n);
    bcast_doubles(factor.data(), factor.size(), 0);
    double* factor_device = nullptr;
    if (n) {
        CUDA_CHECK(cudaMalloc(&factor_device, factor.size() * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(factor_device, factor.data(), factor.size() * sizeof(double), cudaMemcpyHostToDevice));
        const double one = 1.0, zero = 0.0;
        int blocks = (n + static_cast<long long>(block_size) - 1) / block_size;
        for (int b = rank_id; b < blocks; b += world) {
            int first = b * block_size, width = std::min(block_size, n - first);
            int local = (b / world) * block_size;
            // The row-major L buffer is column-major L^T.
            BLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, width, n,
                                   &one, factor_device, n,
                                   factor_device + static_cast<size_t>(first) * n, n,
                                   &zero, matrix + static_cast<size_t>(local) * n, n));
        }
    }
    std::vector<double> reconstructed(matrix_count);
    if (matrix_count) CUDA_CHECK(cudaMemcpy(reconstructed.data(), matrix, matrix_count * sizeof(double), cudaMemcpyDeviceToHost));
    if (factor_device) CUDA_CHECK(cudaFree(factor_device));
    double max_error = 0.0, rel_error = 0.0;
    #pragma omp parallel for reduction(max:max_error,rel_error) schedule(static)
    for (size_t i = 0; i < matrix_count; ++i) {
        double error = std::fabs(reconstructed[i] - original[i]);
        max_error = std::max(max_error, error);
        rel_error = std::max(rel_error, error / (std::fabs(original[i]) + 1e-10));
    }
    double local_errors[2] = {max_error, rel_error}, global_errors[2] = {};
    MPI_CHECK(MPI_Allreduce(local_errors, global_errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    if (rank_id == 0) {
        printf("Max absolute error: %.10e\n", global_errors[0]);
        printf("Max relative error: %.10e\n", global_errors[1]);
        if (global_errors[1] > 1e-6) printf("Validation failed: relative error too large\n");
    }
    return global_errors[1] <= 1e-6;
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank_id));
    int world = 1;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI thread support", 1);

    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            long value = strtol(argv[++i], &end, 10);
            if (*end || value < 0 || value > INT_MAX || end == argv[i] ||
                static_cast<unsigned long long>(value) * value > std::numeric_limits<size_t>::max() / sizeof(double)) {
                if (rank_id == 0) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
            n = static_cast<int>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank_id == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank_id == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }

    // Assign devices by node-local rank, so each node can use all its GPUs.
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_id, MPI_INFO_NULL, &local_comm));
    int local_rank = 0, device_count = 0;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) fail("no CUDA devices", 1);
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    BLAS_CHECK(cublasCreate(&blas));
    SOLVER_CHECK(cusolverDnCreate(&solver));

    if (rank_id == 0) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    int blocks = (n + static_cast<long long>(block_size) - 1) / block_size;
    int local_cols = 0;
    for (int b = rank_id; b < blocks; b += world) local_cols += std::min(block_size, n - b * block_size);
    size_t matrix_count = static_cast<size_t>(n) * local_cols;
    double *matrix = nullptr, *random_device = nullptr;
    if (matrix_count) CUDA_CHECK(cudaMalloc(&matrix, matrix_count * sizeof(double)));
    if (n) {
        // Preserve the original single rand_r stream exactly, on every rank.
        std::vector<double> random(static_cast<size_t>(n) * n);
        unsigned int seed = 42;
        for (double& x : random) x = rand_r(&seed);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < random.size(); ++i)
            random[i] = random[i] / static_cast<double>(RAND_MAX) - 0.5;
        CUDA_CHECK(cudaMalloc(&random_device, random.size() * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(random_device, random.data(), random.size() * sizeof(double), cudaMemcpyHostToDevice));
        const double one = 1.0, zero = 0.0;
        for (int b = rank_id; b < blocks; b += world) {
            int first = b * block_size, width = std::min(block_size, n - first);
            int local = (b / world) * block_size;
            // Row-major B is column-major B^T; form the owned columns of B B^T.
            BLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, width, n, &one,
                                   random_device, n, random_device + static_cast<size_t>(first) * n, n,
                                   &zero, matrix + static_cast<size_t>(local) * n, n));
            add_diagonal<<<(width + 255) / 256, 256>>>(matrix, n, first, local, width);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaFree(random_device));
    }
    std::vector<double> original;
    if (validate) {
        original.resize(matrix_count);
        if (matrix_count) CUDA_CHECK(cudaMemcpy(original.data(), matrix, matrix_count * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Each rank owns cyclic column blocks. The owner factors and broadcasts a
    // panel; all ranks update their trailing columns with level-3 BLAS.
    double *panel_device = nullptr, *workspace = nullptr;
    int* info_device = nullptr;
    std::vector<double> panel(static_cast<size_t>(n) * std::min(n, block_size));
    int workspace_capacity = 0;
    if (n) {
        CUDA_CHECK(cudaMalloc(&panel_device, panel.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&info_device, sizeof(int)));
        int workspace_size = 0;
        // Workspace is queried on each owner below because ranks can own no columns.
        if (matrix_count) {
            SOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER,
                         std::min(n, block_size), matrix, n, &workspace_size));
            if (workspace_size) {
                CUDA_CHECK(cudaMalloc(&workspace, static_cast<size_t>(workspace_size) * sizeof(double)));
                workspace_capacity = workspace_size;
            }
        }
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank_id == 0) printf("Computing Cholesky decomposition...\n");
    double start = MPI_Wtime();
    bool success = true;
    for (int b = 0; b < blocks; ++b) {
        int k = b * block_size, width = std::min(block_size, n - k);
        int owner = b % world, rows = n - k, info = 0;
        if (rank_id == owner) {
            int local = (b / world) * block_size;
            double* diagonal = matrix + static_cast<size_t>(local) * n + k;
            int workspace_size = 0;
            SOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, width, diagonal, n, &workspace_size));
            if (workspace_size > workspace_capacity) {
                if (workspace) CUDA_CHECK(cudaFree(workspace));
                CUDA_CHECK(cudaMalloc(&workspace, static_cast<size_t>(workspace_size) * sizeof(double)));
                workspace_capacity = workspace_size;
            }
            SOLVER_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, width, diagonal, n,
                                          workspace, workspace_size, info_device));
            CUDA_CHECK(cudaMemcpy(&info, info_device, sizeof(int), cudaMemcpyDeviceToHost));
            if (!info && rows > width) {
                const double one = 1.0;
                BLAS_CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                       CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, rows - width, width,
                                       &one, diagonal, n, diagonal + width, n));
            }
            if (!info) CUDA_CHECK(cudaMemcpy2D(panel.data(), static_cast<size_t>(rows) * sizeof(double), diagonal,
                                               static_cast<size_t>(n) * sizeof(double), static_cast<size_t>(rows) * sizeof(double),
                                               width, cudaMemcpyDeviceToHost));
        }
        MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, owner, MPI_COMM_WORLD));
        if (info) {
            if (rank_id == 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", k + info - 1);
            success = false; break;
        }
        bcast_doubles(panel.data(), static_cast<size_t>(rows) * width, owner);
        if (k + width < n) {
            CUDA_CHECK(cudaMemcpy(panel_device, panel.data(), static_cast<size_t>(rows) * width * sizeof(double), cudaMemcpyHostToDevice));
            const double minus_one = -1.0, one = 1.0;
            int first_owned = b + 1 + (rank_id - (b + 1) % world + world) % world;
            for (int jblock = first_owned; jblock < blocks; jblock += world) {
                int j = jblock * block_size, jw = std::min(block_size, n - j);
                int local = (jblock / world) * block_size;
                const double* p = panel_device + (j - k);
                BLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, n - j, jw, width,
                                       &minus_one, p, rows, p, rows, &one,
                                       matrix + static_cast<size_t>(local) * n + j, n));
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, max_elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!success) {
        if (rank_id == 0) printf("Cholesky decomposition failed\n");
    } else if (rank_id == 0) {
        long milliseconds = static_cast<long>(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", ops / (milliseconds / 1000.0) / 1e9);
    }
    int valid = success ? 1 : 0;
    if (success && (printResults || validate)) {
        std::vector<double> factor = gather_matrix(matrix, n, world, true);
        if (rank_id == 0 && printResults) print_results(factor, "CholeskyL");
        if (validate) {
            if (rank_id == 0) printf("Validating result...\n");
            valid = validateCholesky(factor, original, matrix, matrix_count, n, world, blas);
            if (rank_id == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    if (workspace) CUDA_CHECK(cudaFree(workspace));
    if (info_device) CUDA_CHECK(cudaFree(info_device));
    if (panel_device) CUDA_CHECK(cudaFree(panel_device));
    if (matrix) CUDA_CHECK(cudaFree(matrix));
    SOLVER_CHECK(cusolverDnDestroy(solver));
    BLAS_CHECK(cublasDestroy(blas));
    MPI_CHECK(MPI_Finalize());
    return valid ? 0 : 1;
}
