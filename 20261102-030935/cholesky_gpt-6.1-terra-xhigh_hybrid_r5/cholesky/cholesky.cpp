#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

static int rank_id;

static void fail(const char* operation, int code) {
    std::fprintf(stderr, "Rank %d: %s failed (error %d)\n", rank_id, operation, code);
    MPI_Abort(MPI_COMM_WORLD, code ? code : 1);
    std::abort();
}
#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) fail(#x, static_cast<int>(e_)); } while (0)
#define BLAS_CHECK(x) do { cublasStatus_t e_ = (x); if (e_ != CUBLAS_STATUS_SUCCESS) fail(#x, static_cast<int>(e_)); } while (0)
#define SOLVER_CHECK(x) do { cusolverStatus_t e_ = (x); if (e_ != CUSOLVER_STATUS_SUCCESS) fail(#x, static_cast<int>(e_)); } while (0)
#define MPI_CHECK(x) do { int e_ = (x); if (e_ != MPI_SUCCESS) fail(#x, e_); } while (0)

// MPI counts are int, so transfer large buffers in chunks.
static void broadcast(double* data, size_t count, int root) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_CHECK(MPI_Bcast(data, chunk, MPI_DOUBLE, root, MPI_COMM_WORLD));
        data += chunk;
        count -= chunk;
    }
}
static void send_buffer(const double* data, size_t count, int dest) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_CHECK(MPI_Send(data, chunk, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD));
        data += chunk; count -= chunk;
    }
}
static void receive_buffer(double* data, size_t count, int src) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_CHECK(MPI_Recv(data, chunk, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        data += chunk; count -= chunk;
    }
}

__global__ void add_diagonal(double* a, int n, int first, int width, int offset) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < width) a[static_cast<size_t>(offset + j) * n + first + j] += n;
}

struct Block { int first, width, offset; };

// Gather column blocks and transpose to the row-major layout of the original program.
static std::vector<double> gather_matrix(double* d_a, int n, int block_size,
                                         int ranks, bool lower_only) {
    std::vector<double> result;
    if (rank_id == 0) result.resize(static_cast<size_t>(n) * n);
    std::vector<double> buffer(static_cast<size_t>(n) * std::min(n, block_size));
    for (int first = 0, block = 0; first < n; first += block_size, ++block) {
        int width = std::min(block_size, n - first);
        int owner = block % ranks;
        int offset = (block / ranks) * block_size;
        size_t count = static_cast<size_t>(n) * width;
        if (rank_id == owner) {
            CUDA_CHECK(cudaMemcpy(buffer.data(), d_a + static_cast<size_t>(offset) * n,
                                  count * sizeof(double), cudaMemcpyDeviceToHost));
            if (owner != 0) send_buffer(buffer.data(), count, 0);
        }
        if (rank_id == 0) {
            if (owner != 0) receive_buffer(buffer.data(), count, owner);
            #pragma omp parallel for schedule(static)
            for (int row = 0; row < n; ++row) {
                for (int j = 0; j < width; ++j) {
                    int col = first + j;
                    result[static_cast<size_t>(row) * n + col] =
                        lower_only && row < col ? 0.0 : buffer[static_cast<size_t>(j) * n + row];
                }
            }
        }
    }
    return result;
}

static bool validateCholesky(const std::vector<double>& l, const std::vector<double>& original, int n) {
    double max_error = 0.0, rel_error = 0.0;
    #pragma omp parallel for reduction(max:max_error,rel_error) schedule(dynamic, 1)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k <= std::min(i, j); ++k)
                sum += l[static_cast<size_t>(i) * n + k] * l[static_cast<size_t>(j) * n + k];
            double a = original[static_cast<size_t>(i) * n + j];
            double error = std::fabs(sum - a);
            max_error = std::max(max_error, error);
            rel_error = std::max(rel_error, error / (std::fabs(a) + 1e-10));
        }
    }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", max_error, rel_error);
    if (rel_error > 1e-6) std::printf("Validation failed: relative error too large\n");
    return rel_error <= 1e-6;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n", name);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank_id));
    int ranks;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI thread support", provided);
    int n = 512;
    bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            long value = std::strtol(argv[++i], &end, 10);
            if (!*argv[i] || *end || value < 0 || value > INT_MAX) {
                if (rank_id == 0) std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                MPI_Finalize(); return 1;
            }
            n = static_cast<int>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) print_results_flag = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank_id == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank_id == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (rank_id == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n",
                    n, n, validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
    }

    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_id, MPI_INFO_NULL, &local_comm));
    int local_rank, devices;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) fail("No CUDA devices available", 1);
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    BLAS_CHECK(cublasCreate(&blas));
    SOLVER_CHECK(cusolverDnCreate(&solver));

    // Each rank owns full-height column blocks in cyclic order.
    constexpr int block_size = 256;
    std::vector<Block> blocks;
    int local_columns = 0;
    for (int first = 0, index = 0; first < n; first += block_size, ++index) {
        int width = std::min(block_size, n - first);
        if (index % ranks == rank_id) {
            blocks.push_back({first, width, local_columns});
            local_columns += width;
        }
    }
    double *d_a = nullptr, *d_b = nullptr, *d_panel = nullptr, *d_work = nullptr;
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, std::max<size_t>(1, static_cast<size_t>(n) * local_columns) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, std::max<size_t>(1, static_cast<size_t>(n) * n) * sizeof(double)));
    {
        // Preserve rand_r's original sequence exactly on every rank.
        std::vector<double> b(static_cast<size_t>(n) * n);
        unsigned int seed = 42;
        for (double& value : b) value = static_cast<double>(rand_r(&seed));
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < b.size(); ++i)
            b[i] = b[i] / static_cast<double>(RAND_MAX) - 0.5;
        if (!b.empty()) CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    const double one = 1.0, zero = 0.0, minus_one = -1.0;
    for (const Block& block : blocks) {
        // Row-major B is column-major B^T: A(:,j) = B * B(j,:)^T.
        BLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, block.width, n,
                               &one, d_b, n, d_b + static_cast<size_t>(block.first) * n, n,
                               &zero, d_a + static_cast<size_t>(block.offset) * n, n));
        add_diagonal<<<(block.width + 255) / 256, 256>>>(d_a, n, block.first, block.width, block.offset);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_b));
    std::vector<double> original;
    if (validate) original = gather_matrix(d_a, n, block_size, ranks, false);

    int panel_width = std::min(n, block_size);
    size_t panel_capacity = static_cast<size_t>(n) * panel_width;
    CUDA_CHECK(cudaMalloc(&d_panel, std::max<size_t>(1, panel_capacity) * sizeof(double)));
    double* h_panel = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_panel, std::max<size_t>(1, panel_capacity) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    int work_size = 0;
    if (n > 0) {
        SOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, panel_width,
                                                d_a, n, &work_size));
    }
    CUDA_CHECK(cudaMalloc(&d_work, std::max(1, work_size) * sizeof(double)));

    if (rank_id == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    bool success = true;
    for (int first = 0, index = 0; first < n; first += block_size, ++index) {
        int width = std::min(block_size, n - first);
        int rows = n - first;
        int owner = index % ranks;
        int info = 0;
        if (rank_id == owner) {
            int offset = (index / ranks) * block_size;
            double* diag = d_a + static_cast<size_t>(offset) * n + first;
            SOLVER_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, width, diag, n,
                                          d_work, work_size, d_info));
            CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
            if (info == 0) {
                if (rows > width) {
                    BLAS_CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                           CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, rows - width, width,
                                           &one, diag, n, diag + width, n));
                }
                if (rows > width)
                    CUDA_CHECK(cudaMemcpy2D(h_panel, static_cast<size_t>(rows) * sizeof(double), diag,
                                            static_cast<size_t>(n) * sizeof(double),
                                            static_cast<size_t>(rows) * sizeof(double), width,
                                            cudaMemcpyDeviceToHost));
            }
        }
        MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, owner, MPI_COMM_WORLD));
        if (info != 0) {
            if (rank_id == 0) {
                if (info > 0) std::printf("Error: Matrix is not positive definite at diagonal element %d\n", first + info - 1);
                else std::printf("Error: Cholesky factorization received invalid argument %d\n", -info);
            }
            success = false; break;
        }
        if (first + width == n) continue;
        broadcast(h_panel, static_cast<size_t>(rows) * width, owner);
        bool has_updates = !blocks.empty() && blocks.back().first > first;
        if (has_updates)
            CUDA_CHECK(cudaMemcpy(d_panel, h_panel, static_cast<size_t>(rows) * width * sizeof(double), cudaMemcpyHostToDevice));
        for (const Block& block : blocks) {
            if (block.first <= first) continue;
            int remaining = n - block.first;
            const double* p = d_panel + (block.first - first);
            BLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, remaining, block.width, width,
                                   &minus_one, p, rows, p, rows, &one,
                                   d_a + static_cast<size_t>(block.offset) * n + block.first, n));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, max_elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank_id == 0 && success) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000));
        std::printf("Performance: %.3f GFLOPS\n", static_cast<double>(n) * n * n / 3.0 / max_elapsed / 1e9);
    }
    if (!success && rank_id == 0) std::printf("Cholesky decomposition failed\n");
    std::vector<double> result;
    if (success && (validate || print_results_flag)) result = gather_matrix(d_a, n, block_size, ranks, true);
    int status = success ? 0 : 1;
    if (rank_id == 0 && success) {
        if (print_results_flag) print_results(result, "CholeskyL");
        if (validate) {
            std::printf("Validating result...\n");
            bool valid = validateCholesky(result, original, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) status = 1;
        }
    }
    MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CUDA_CHECK(cudaFree(d_work)); CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFreeHost(h_panel)); CUDA_CHECK(cudaFree(d_panel)); CUDA_CHECK(cudaFree(d_a));
    SOLVER_CHECK(cusolverDnDestroy(solver)); BLAS_CHECK(cublasDestroy(blas));
    MPI_Finalize();
    return status;
}
