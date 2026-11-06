#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {
constexpr int block_size = 256;
int world_rank = 0;

[[noreturn]] void fail(const char* what, int code) {
    fprintf(stderr, "Rank %d: %s failed (%d)\n", world_rank, what, code);
    MPI_Abort(MPI_COMM_WORLD, code ? code : 1);
    std::abort();
}
#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "Rank %d: %s: %s\n", world_rank, #call, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define BLAS(call) do { cublasStatus_t e = (call); if (e != CUBLAS_STATUS_SUCCESS) fail(#call, int(e)); } while (0)
#define SOLVER(call) do { cusolverStatus_t e = (call); if (e != CUSOLVER_STATUS_SUCCESS) fail(#call, int(e)); } while (0)
#define MPI_CHECK(call) do { int e = (call); if (e != MPI_SUCCESS) fail(#call, e); } while (0)

__global__ void add_diagonal(double* a, int n, int global_col, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) a[static_cast<size_t>(j) * n + global_col + j] += n;
}

// MPI counts are int, while a panel can contain more than INT_MAX entries.
void broadcast_doubles(double* data, size_t count, int root) {
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

struct DistributedMatrix {
    int n, rank, ranks, blocks, local_columns;
    double *device = nullptr, *panel = nullptr, *host_panel = nullptr, *workspace = nullptr;
    int *info = nullptr;
    cublasHandle_t blas{};
    cusolverDnHandle_t solver{};
    int workspace_size = 0;

    DistributedMatrix(int size, int r, int p) : n(size), rank(r), ranks(p),
        blocks((size + static_cast<long long>(block_size) - 1) / block_size), local_columns(0) {
        for (int b = rank; b < blocks; b += ranks)
            local_columns += std::min(block_size, n - b * block_size);
        BLAS(cublasCreate(&blas));
        SOLVER(cusolverDnCreate(&solver));
        if (!n) return;
        size_t local_elements = static_cast<size_t>(n) * local_columns;
        if (local_elements) CUDA(cudaMalloc(&device, local_elements * sizeof(double)));
        size_t panel_elements = static_cast<size_t>(n) * std::min(n, block_size);
        CUDA(cudaMalloc(&panel, panel_elements * sizeof(double)));
        CUDA(cudaMallocHost(&host_panel, panel_elements * sizeof(double)));
        CUDA(cudaMalloc(&info, sizeof(int)));
        // The workspace requirement depends only on the tile size and leading dimension.
        if (local_columns) {
            SOLVER(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER,
                std::min(n, block_size), device, n, &workspace_size));
            if (workspace_size) CUDA(cudaMalloc(&workspace, static_cast<size_t>(workspace_size) * sizeof(double)));
        }
    }
    ~DistributedMatrix() {
        if (workspace) cudaFree(workspace);
        if (info) cudaFree(info);
        if (host_panel) cudaFreeHost(host_panel);
        if (panel) cudaFree(panel);
        if (device) cudaFree(device);
        cusolverDnDestroy(solver);
        cublasDestroy(blas);
    }

    void generate() {
        if (!n || !local_columns) return;
        // Generate the original rand_r stream twice. The first pass retains only
        // this rank's rows of B; the second streams all rows in bounded batches.
        // Thus neither host nor device needs a replicated n-by-n matrix of B.
        const int batch_rows = std::min(n, 2048);
        size_t batch_elements = static_cast<size_t>(n) * batch_rows;
        std::vector<double> batch(batch_elements);
        double *local_b = nullptr, *device_batch = nullptr;
        CUDA(cudaMalloc(&local_b, static_cast<size_t>(n) * local_columns * sizeof(double)));
        CUDA(cudaMalloc(&device_batch, batch_elements * sizeof(double)));
        const double one = 1.0, zero = 0.0;
        for (int pass = 0; pass < 2; ++pass) {
            unsigned int seed = 42;
            for (int row = 0; row < n; row += batch_rows) {
                int rows = std::min(batch_rows, n - row);
                size_t count = static_cast<size_t>(rows) * n;
                for (size_t i = 0; i < count; ++i) batch[i] = rand_r(&seed);
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < count; ++i)
                    batch[i] = batch[i] / static_cast<double>(RAND_MAX) - 0.5;
                if (pass == 0) {
                    for (int block = rank; block < blocks; block += ranks) {
                        int col = block * block_size;
                        int width = std::min(block_size, n - col);
                        int first = std::max(row, col);
                        int last = std::min(row + rows, col + width);
                        if (first < last) {
                            int local = (block / ranks) * block_size + first - col;
                            CUDA(cudaMemcpy(local_b + static_cast<size_t>(local) * n,
                                batch.data() + static_cast<size_t>(first - row) * n,
                                static_cast<size_t>(last - first) * n * sizeof(double), cudaMemcpyHostToDevice));
                        }
                    }
                } else {
                    CUDA(cudaMemcpy(device_batch, batch.data(), count * sizeof(double), cudaMemcpyHostToDevice));
                    for (int block = rank; block < blocks; block += ranks) {
                        int col = block * block_size;
                        int width = std::min(block_size, n - col);
                        int local = (block / ranks) * block_size;
                        // Row-major B is column-major B^T: this computes a
                        // row batch of the rank's columns of B * B^T.
                        BLAS(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, width, n,
                            &one, device_batch, n, local_b + static_cast<size_t>(local) * n, n,
                            &zero, device + static_cast<size_t>(local) * n + row, n));
                    }
                }
            }
        }
        for (int block = rank; block < blocks; block += ranks) {
            int col = block * block_size;
            int width = std::min(block_size, n - col);
            int local = (block / ranks) * block_size;
            add_diagonal<<<(width + 255) / 256, 256>>>(device + static_cast<size_t>(local) * n, n, col, width);
            CUDA(cudaGetLastError());
        }
        CUDA(cudaDeviceSynchronize());
        CUDA(cudaFree(device_batch));
        CUDA(cudaFree(local_b));
    }

    bool factor() {
        const double one = 1.0, minus_one = -1.0;
        for (int block = 0; block < blocks; ++block) {
            int k = block * block_size;
            int width = std::min(block_size, n - k);
            int end = k + width;
            int owner = block % ranks;
            int status = 0;
            if (rank == owner) {
                int local = (block / ranks) * block_size;
                double* column = device + static_cast<size_t>(local) * n;
                SOLVER(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, width,
                    column + k, n, workspace, workspace_size, info));
                CUDA(cudaMemcpy(&status, info, sizeof(status), cudaMemcpyDeviceToHost));
                if (!status) {
                    if (end < n) BLAS(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT,
                        CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                        n - end, width, &one, column + k, n, column + end, n));
                    CUDA(cudaMemcpy(host_panel, column, static_cast<size_t>(n) * width * sizeof(double), cudaMemcpyDeviceToHost));
                }
            }
            MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, owner, MPI_COMM_WORLD));
            if (status) {
                if (rank == 0) {
                    if (status > 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", k + status - 1);
                    else fprintf(stderr, "cuSOLVER potrf argument %d is invalid\n", -status);
                }
                return false;
            }
            broadcast_doubles(host_panel, static_cast<size_t>(n) * width, owner);
            CUDA(cudaMemcpy(panel, host_panel, static_cast<size_t>(n) * width * sizeof(double), cudaMemcpyHostToDevice));
            for (int next = block + 1; next < blocks; ++next) {
                if (next % ranks != rank) continue;
                int col = next * block_size;
                int cols = std::min(block_size, n - col);
                int local = (next / ranks) * block_size;
                BLAS(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, n - end, cols, width,
                    &minus_one, panel + end, n, panel + col, n,
                    &one, device + static_cast<size_t>(local) * n + end, n));
            }
        }
        CUDA(cudaDeviceSynchronize());
        return true;
    }

    // Gather to row-major order, optionally discarding the unused upper triangle.
    void gather(std::vector<double>& result, bool triangular) {
        if (rank == 0) result.resize(static_cast<size_t>(n) * n);
        for (int block = 0; block < blocks; ++block) {
            int col = block * block_size;
            int width = std::min(block_size, n - col);
            int owner = block % ranks;
            size_t count = static_cast<size_t>(n) * width;
            if (rank == owner) {
                int local = (block / ranks) * block_size;
                CUDA(cudaMemcpy(host_panel, device + static_cast<size_t>(local) * n,
                    count * sizeof(double), cudaMemcpyDeviceToHost));
                if (owner != 0) transfer_doubles(host_panel, count, 0, true);
            }
            if (rank == 0) {
                if (owner != 0) transfer_doubles(host_panel, count, owner, false);
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < width; ++j)
                        result[static_cast<size_t>(i) * n + col + j] =
                            triangular && i < col + j ? 0.0 : host_panel[static_cast<size_t>(j) * n + i];
            }
        }
    }
};

bool validateCholesky(const std::vector<double>& l, const std::vector<double>& original, int n) {
    double max_error = 0.0, rel_error = 0.0;
    #pragma omp parallel for schedule(static) reduction(max:max_error,rel_error)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            const double* li = l.data() + static_cast<size_t>(i) * n;
            const double* lj = l.data() + static_cast<size_t>(j) * n;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k <= std::min(i, j); ++k) sum += li[k] * lj[k];
            double expected = original[static_cast<size_t>(i) * n + j];
            double error = std::fabs(sum - expected);
            max_error = std::max(max_error, error);
            rel_error = std::max(rel_error, error / (std::fabs(expected) + 1e-10));
        }
    }
    printf("Max absolute error: %.10e\n", max_error);
    printf("Max relative error: %.10e\n", rel_error);
    if (rel_error > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}
void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI thread support", provided);
    int n = 512;
    bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            long value = strtol(argv[++i], &end, 10);
            if (!*argv[i] || *end || value < 0 || value > INT_MAX - block_size ||
                static_cast<unsigned long long>(value) * value > std::numeric_limits<size_t>::max() / sizeof(double)) {
                if (!world_rank) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                MPI_Finalize(); return 1;
            }
            n = static_cast<int>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) print_results_flag = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!world_rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!world_rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local_comm));
    int local_rank = 0, devices = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) fail("No CUDA devices available", 1);
    CUDA(cudaSetDevice(local_rank % devices));
    if (!world_rank) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    int exit_code = 0;
    {
        DistributedMatrix matrix(n, world_rank, ranks);
        matrix.generate();
        std::vector<double> original, result;
        if (validate) matrix.gather(original, false);
        if (!world_rank) printf("Computing Cholesky decomposition...\n");
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        auto start = std::chrono::high_resolution_clock::now();
        bool success = matrix.factor();
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        auto end = std::chrono::high_resolution_clock::now();
        if (!success) {
            if (!world_rank) printf("Cholesky decomposition failed\n");
            exit_code = 1;
        } else {
            if (!world_rank) {
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                double seconds = std::chrono::duration<double>(end - start).count();
                printf("Computation time: %ld ms\n", static_cast<long>(ms));
                printf("Performance: %.3f GFLOPS\n", (static_cast<double>(n) * n * n / 3.0) / seconds / 1e9);
            }
            if (validate || print_results_flag) matrix.gather(result, true);
            if (!world_rank) {
                if (print_results_flag) print_results(result, "CholeskyL");
                if (validate) {
                    printf("Validating result...\n");
                    bool valid = validateCholesky(result, original, n);
                    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                    if (!valid) exit_code = 1;
                }
            }
        }
    }
    MPI_CHECK(MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return exit_code;
}
