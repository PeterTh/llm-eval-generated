#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t e_ = (call);                                                     \
    if (e_ != cudaSuccess) {                                                     \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                     cudaGetErrorString(e_));                                    \
        MPI_Abort(MPI_COMM_WORLD, 2);                                            \
    }                                                                            \
} while (0)

__global__ void factor_column(double* a, const double* pivot, size_t n,
                              size_t first_row, size_t local_rows, size_t j,
                              double diagonal) {
    size_t lr = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (lr >= local_rows || first_row + lr <= j) return;
    double* row = a + lr * n;
    double sum = 0.0;
    for (size_t k = 0; k < j; ++k) sum = fma(row[k], pivot[k], sum);
    row[j] = (row[j] - sum) / diagonal;
}

static void partition(size_t n, int ranks, std::vector<size_t>& starts,
                      std::vector<size_t>& rows) {
    starts.resize(ranks); rows.resize(ranks);
    const size_t base = n / static_cast<size_t>(ranks), extra = n % ranks;
    size_t p = 0;
    for (int r = 0; r < ranks; ++r) {
        rows[r] = base + (static_cast<size_t>(r) < extra);
        starts[r] = p; p += rows[r];
    }
}

static int owner_of(size_t row, const std::vector<size_t>& starts,
                    const std::vector<size_t>& rows) {
    for (int r = 0; r < static_cast<int>(rows.size()); ++r)
        if (row >= starts[r] && row < starts[r] + rows[r]) return r;
    return static_cast<int>(rows.size()) - 1;
}

bool distributedCholesky(std::vector<double>& local, size_t n, size_t first,
                         const std::vector<size_t>& starts,
                         const std::vector<size_t>& rows, int rank) {
    double *d_a = nullptr, *d_pivot = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, std::max<size_t>(1, local.size()) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pivot, std::max<size_t>(1, n) * sizeof(double)));
    if (!local.empty())
        CUDA_CHECK(cudaMemcpy(d_a, local.data(), local.size() * sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> pivot(n, 0.0);
    bool ok = true;
    for (size_t j = 0; j < n; ++j) {
        const int owner = owner_of(j, starts, rows);
        double diagonal = 0.0;
        if (rank == owner) {
            const size_t lr = j - first;
            CUDA_CHECK(cudaMemcpy(pivot.data(), d_a + lr * n, (j + 1) * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < j; ++k) sum += pivot[k] * pivot[k];
            const double value = pivot[j] - sum;
            if (value <= 0.0 || !std::isfinite(value)) ok = false;
            else { diagonal = std::sqrt(value); pivot[j] = diagonal; }
        }
        int good = ok ? 1 : 0;
        MPI_Bcast(&good, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!good) { ok = false; break; }
        MPI_Bcast(&diagonal, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(pivot.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(d_pivot, pivot.data(), (j + 1) * sizeof(double), cudaMemcpyHostToDevice));
        if (rank == owner)
            CUDA_CHECK(cudaMemcpy(d_a + (j - first) * n + j, &diagonal,
                                  sizeof(double), cudaMemcpyHostToDevice));
        const int threads = 256;
        const int blocks = static_cast<int>((rows[rank] + threads - 1) / threads);
        if (blocks) factor_column<<<blocks, threads>>>(d_a, d_pivot, n, first, rows[rank], j, diagonal);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    if (ok && !local.empty()) {
        CUDA_CHECK(cudaMemcpy(local.data(), d_a, local.size() * sizeof(double), cudaMemcpyDeviceToHost));
        #pragma omp parallel for schedule(static)
        for (long long lr = 0; lr < static_cast<long long>(rows[rank]); ++lr) {
            size_t global = first + static_cast<size_t>(lr);
            std::fill(local.begin() + lr * n + global + 1, local.begin() + (lr + 1) * n, 0.0);
        }
    }
    cudaFree(d_pivot); cudaFree(d_a);
    return ok;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    #pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < static_cast<long long>(n); ++ii) {
        size_t i = static_cast<size_t>(ii);
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k) sum += B[i*n+k] * B[j*n+k];
            A[i*n+j] = sum;
        }
        A[i*n+i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A, size_t n) {
    double max_abs = 0.0, max_rel = 0.0;
    #pragma omp parallel for schedule(static) reduction(max:max_abs,max_rel)
    for (long long ii = 0; ii < static_cast<long long>(n); ++ii) {
        size_t i = static_cast<size_t>(ii);
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t end = std::min(i, j) + 1;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < end; ++k) sum += L[i*n+k] * L[j*n+k];
            double error = std::fabs(sum - A[i*n+j]);
            max_abs = std::max(max_abs, error);
            max_rel = std::max(max_rel, error / (std::fabs(A[i*n+j]) + 1e-10));
        }
    }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", max_abs, max_rel);
    if (max_rel > 1e-6) std::printf("Validation failed: relative error too large\n");
    return max_rel <= 1e-6;
}

void printUsage(const char* p) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(rank % devices));

    size_t n = 512; bool validate = false, printResults = false; int parse_ok = 1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr; unsigned long long v = std::strtoull(argv[++i], &end, 10);
            if (!end || *end || !v) parse_ok = 0; else n = static_cast<size_t>(v);
        } else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else parse_ok = 0;
    }
    if (!parse_ok || n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (!rank) { std::fprintf(stderr, "Invalid command line or matrix too large for MPI counts\n"); printUsage(argv[0]); }
        MPI_Finalize(); return 1;
    }
    if (!rank) std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",
                           n, n, validate ? "enabled" : "disabled");
    std::vector<size_t> starts, rows; partition(n, ranks, starts, rows);
    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) {
        size_t c = rows[r] * n, d = starts[r] * n;
        if (c > INT_MAX || d > INT_MAX) { if (!rank) std::fprintf(stderr, "Matrix exceeds MPI count range\n"); MPI_Abort(MPI_COMM_WORLD, 3); }
        counts[r] = static_cast<int>(c); displs[r] = static_cast<int>(d);
    }
    std::vector<double> full, original;
    if (!rank) { std::printf("Generating positive definite matrix...\n"); full.resize(n*n); generatePositiveDefiniteMatrix(full,n); if(validate) original=full; }
    std::vector<double> local(rows[rank] * n);
    MPI_Scatterv(rank ? nullptr : full.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 local.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (!rank) { std::vector<double>().swap(full); std::printf("Computing Cholesky decomposition...\n"); }
    MPI_Barrier(MPI_COMM_WORLD); double start = MPI_Wtime();
    bool success = distributedCholesky(local, n, starts[rank], starts, rows, rank);
    double elapsed = MPI_Wtime() - start, max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) full.resize(n*n);
    MPI_Gatherv(local.data(), counts[rank], MPI_DOUBLE, rank ? nullptr : full.data(),
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int rc = success ? 0 : 1;
    if (!rank) {
        if (!success) std::printf("Cholesky decomposition failed\n");
        else {
            long ms = static_cast<long>(max_elapsed * 1000.0);
            std::printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n", ms,
                        (static_cast<double>(n)*n*n/3.0) / max_elapsed / 1e9);
            if (printResults) print_results(full, "CholeskyL");
            if (validate) { std::printf("Validating result...\n"); bool valid=validateCholesky(full,original,n);
                std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); if(!valid) rc=1; }
        }
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Finalize(); return rc;
}
