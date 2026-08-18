#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// MPI distributes contiguous row slabs.  Each slab remains resident on one GPU;
// only the newly completed column is staged through host memory for MPI.
#define CUDA_CHECK(call) do {                                                   \
    cudaError_t e_ = (call);                                                    \
    if (e_ != cudaSuccess) {                                                    \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(e_));                                   \
        MPI_Abort(MPI_COMM_WORLD, 2);                                           \
    }                                                                           \
} while (0)

static void decomposition(int n, int size, int rank, int& first, int& rows,
                          std::vector<int>& counts, std::vector<int>& displs) {
    counts.resize(size);
    displs.resize(size);
    const int q = n / size, r = n % size;
    for (int p = 0; p < size; ++p) {
        counts[p] = q + (p < r);
        displs[p] = p * q + std::min(p, r);
    }
    first = displs[rank];
    rows = counts[rank];
}

static int rowOwner(int row, const std::vector<int>& counts,
                    const std::vector<int>& displs) {
    const auto it = std::upper_bound(displs.begin(), displs.end(), row);
    int p = std::max(0, static_cast<int>(it - displs.begin()) - 1);
    while (p + 1 < static_cast<int>(counts.size()) &&
           row >= displs[p] + counts[p]) ++p;
    return p;
}

// Jump ahead in rand_r's 32-bit LCG.  This lets OpenMP threads generate
// independent contiguous chunks while reproducing the original seed-42 stream.
static std::uint32_t advanceRandState(std::uint32_t state, std::uint64_t steps) {
    std::uint32_t acc_mul = 1, acc_add = 0;
    std::uint32_t mul = 1103515245u, add = 12345u;
    while (steps) {
        if (steps & 1u) {
            acc_mul *= mul;
            acc_add = acc_add * mul + add;
        }
        add *= mul + 1u;
        mul *= mul;
        steps >>= 1;
    }
    return acc_mul * state + acc_add;
}

__global__ void formSpd(const double* __restrict__ b, double* __restrict__ a,
                        int n, int first, int rows) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int lr = blockIdx.y * blockDim.y + threadIdx.y;
    if (lr >= rows || j >= n) return;
    const int i = first + lr;
    double sum = 0.0;
    for (int k = 0; k < n; ++k) sum = fma(b[i * n + k], b[j * n + k], sum);
    a[static_cast<size_t>(lr) * n + j] = sum + (i == j ? n : 0);
}

__global__ void divideColumn(double* a, int n, int first, int rows,
                             int k, double diagonal) {
    const int lr = blockIdx.x * blockDim.x + threadIdx.x;
    if (lr < rows && first + lr > k)
        a[static_cast<size_t>(lr) * n + k] /= diagonal;
}

__global__ void extractColumn(const double* a, double* column, int n, int rows,
                              int k) {
    const int lr = blockIdx.x * blockDim.x + threadIdx.x;
    if (lr < rows) column[lr] = a[static_cast<size_t>(lr) * n + k];
}

__global__ void trailingUpdate(double* a, const double* column, int n,
                               int first, int rows, int k) {
    const int j = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    const int lr = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = first + lr;
    if (lr < rows && i > k && j <= i && j < n)
        a[static_cast<size_t>(lr) * n + j] =
            fma(-column[i], column[j], a[static_cast<size_t>(lr) * n + j]);
}

__global__ void zeroUpper(double* a, int n, int first, int rows) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int lr = blockIdx.y * blockDim.y + threadIdx.y;
    if (lr < rows && j < n && j > first + lr)
        a[static_cast<size_t>(lr) * n + j] = 0.0;
}

static bool choleskyDecomposition(double* d_a, int n, int first, int rows,
                                  const std::vector<int>& counts,
                                  const std::vector<int>& displs, int rank) {
    double *d_local_column = nullptr, *d_column = nullptr;
    CUDA_CHECK(cudaMalloc(&d_local_column, std::max(1, rows) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_column, static_cast<size_t>(n) * sizeof(double)));
    double* local_column = nullptr;
    double* column = nullptr;
    CUDA_CHECK(cudaMallocHost(&local_column, std::max(1, rows) * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&column, static_cast<size_t>(n) * sizeof(double)));

    bool ok = true;
    constexpr int threads = 256;
    const dim3 block(16, 16);
    for (int k = 0; k < n; ++k) {
        const int owner = rowOwner(k, counts, displs);
        double diagonal = 0.0;
        if (rank == owner) {
            const size_t index = static_cast<size_t>(k - first) * n + k;
            CUDA_CHECK(cudaMemcpy(&diagonal, d_a + index, sizeof(double),
                                  cudaMemcpyDeviceToHost));
            if (diagonal > 0.0 && std::isfinite(diagonal)) {
                diagonal = std::sqrt(diagonal);
                CUDA_CHECK(cudaMemcpy(d_a + index, &diagonal, sizeof(double),
                                      cudaMemcpyHostToDevice));
            } else {
                ok = false;
            }
        }
        int local_ok = ok ? 1 : 0, global_ok = 0;
        MPI_Bcast(&diagonal, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!global_ok) { ok = false; break; }

        if (rows) {
            divideColumn<<<(rows + threads - 1) / threads, threads>>>(
                d_a, n, first, rows, k, diagonal);
            extractColumn<<<(rows + threads - 1) / threads, threads>>>(
                d_a, d_local_column, n, rows, k);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(local_column, d_local_column,
                                  static_cast<size_t>(rows) * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(local_column, rows, MPI_DOUBLE, column, counts.data(),
                       displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(d_column, column, static_cast<size_t>(n) * sizeof(double),
                              cudaMemcpyHostToDevice));
        if (rows && k + 1 < n) {
            const dim3 grid((n - k - 1 + block.x - 1) / block.x,
                            (rows + block.y - 1) / block.y);
            trailingUpdate<<<grid, block>>>(d_a, d_column, n, first, rows, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    if (ok && rows) {
        const dim3 block2(16, 16), grid2((n + 15) / 16, (rows + 15) / 16);
        zeroUpper<<<grid2, block2>>>(d_a, n, first, rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaFreeHost(column));
    CUDA_CHECK(cudaFreeHost(local_column));
    CUDA_CHECK(cudaFree(d_column));
    CUDA_CHECK(cudaFree(d_local_column));
    return ok;
}

static bool validateCholesky(const std::vector<double>& l,
                             const std::vector<double>& original, int n) {
    double max_error = 0.0, max_relative = 0.0;
#pragma omp parallel for schedule(static) reduction(max:max_error,max_relative)
    for (long long ij = 0; ij < static_cast<long long>(n) * n; ++ij) {
        const int i = static_cast<int>(ij / n), j = static_cast<int>(ij % n);
        double sum = 0.0;
        const int limit = std::min(i, j);
        for (int k = 0; k <= limit; ++k)
            sum = std::fma(l[static_cast<size_t>(i) * n + k],
                           l[static_cast<size_t>(j) * n + k], sum);
        const double error = std::fabs(sum - original[ij]);
        max_error = std::max(max_error, error);
        max_relative = std::max(max_relative,
            error / (std::fabs(original[ij]) + 1e-10));
    }
    std::printf("Max absolute error: %.10e\n", max_error);
    std::printf("Max relative error: %.10e\n", max_relative);
    return max_relative <= 1e-6;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>     Matrix size (default: 512)\n"
                "  -v           Enable validation\n"
                "  -r           Print results for external validation\n"
                "  -h           Show this help message\n", name);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI lacks required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n = 512;
    bool validate = false, print_results_flag = false, help = false, args_ok = true;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) {
            const long value = std::strtol(argv[++i], nullptr, 10);
            if (value <= 0 || value > std::numeric_limits<int>::max()) args_ok = false;
            else n = static_cast<int>(value);
        } else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) print_results_flag = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else args_ok = false;
    }
    if (help || !args_ok) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return args_ok ? 0 : 1;
    }
    if (static_cast<unsigned long long>(n) * n > std::numeric_limits<int>::max()) {
        if (rank == 0) std::fprintf(stderr, "Matrix is too large for MPI counts\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        if (rank == 0) std::fprintf(stderr, "CUDA GPU required\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    CUDA_CHECK(cudaFree(nullptr));

    std::vector<int> counts, displs;
    int first = 0, rows = 0;
    decomposition(n, size, rank, first, rows, counts, displs);
    std::vector<double> b(static_cast<size_t>(n) * n);
#pragma omp parallel
    {
        const std::uint64_t total = static_cast<std::uint64_t>(n) * n;
        const std::uint64_t begin = total * omp_get_thread_num() / omp_get_num_threads();
        const std::uint64_t end = total * (omp_get_thread_num() + 1) / omp_get_num_threads();
        unsigned int seed = advanceRandState(42u, 3u * begin);
        for (std::uint64_t i = begin; i < end; ++i)
            b[i] = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
    }

    double *d_b = nullptr, *d_a = nullptr;
    CUDA_CHECK(cudaMalloc(&d_b, b.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_a, std::max<size_t>(1, static_cast<size_t>(rows) * n) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice));
    b.clear(); b.shrink_to_fit();
    const dim3 block(16, 16), grid((n + 15) / 16, (rows + 15) / 16);
    if (rows) {
        formSpd<<<grid, block>>>(d_b, d_a, n, first, rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaFree(d_b));

    std::vector<double> original_local;
    if (validate) {
        original_local.resize(static_cast<size_t>(rows) * n);
        CUDA_CHECK(cudaMemcpy(original_local.data(), d_a,
                              original_local.size() * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\n", n, n);
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", size, omp_get_max_threads());
        std::printf("Validation: %s\nComputing Cholesky decomposition...\n",
                    validate ? "enabled" : "disabled");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(d_a, n, first, rows,
                                                counts, displs, rank);
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed: matrix is not positive definite\n");
        CUDA_CHECK(cudaFree(d_a));
        MPI_Comm_free(&local_comm); MPI_Finalize(); return 1;
    }

    std::vector<double> local(static_cast<size_t>(rows) * n);
    CUDA_CHECK(cudaMemcpy(local.data(), d_a, local.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    std::vector<int> matrix_counts(size), matrix_displs(size);
    for (int p = 0; p < size; ++p) {
        matrix_counts[p] = counts[p] * n;
        matrix_displs[p] = displs[p] * n;
    }
    std::vector<double> result, original;
    if (validate || print_results_flag) {
        if (rank == 0) result.resize(static_cast<size_t>(n) * n);
        MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                    rank == 0 ? result.data() : nullptr, matrix_counts.data(),
                    matrix_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (validate) {
        if (rank == 0) original.resize(static_cast<size_t>(n) * n);
        MPI_Gatherv(original_local.data(), static_cast<int>(original_local.size()), MPI_DOUBLE,
                    rank == 0 ? original.data() : nullptr, matrix_counts.data(),
                    matrix_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int exit_code = 0;
    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(seconds * 1000.0);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        std::printf("Computation time: %lld ms\nPerformance: %.3f GFLOPS\n",
                    milliseconds, seconds > 0.0 ? ops / seconds / 1e9 : 0.0);
        if (print_results_flag) print_results(result, "CholeskyL");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateCholesky(result, original, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exit_code = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return exit_code;
}
