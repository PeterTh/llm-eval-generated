#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// All ranks retain the factor built so far.  At every block step, ranks split
// the remaining rows, solve their part of the panel on their local GPU, share
// that narrow panel with MPI, and update only their owned trailing rows.
constexpr int kBlock = 64;

#define CUDA_CHECK(call) do { \
    cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
        MPI_Abort(MPI_COMM_WORLD, 2); \
    } \
} while (0)

__global__ void panel_solve(double* a, int n, int k, int b, int first, int last) {
    const int row = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= last) return;
    for (int col = 0; col < b; ++col) {
        double sum = a[row * n + k + col];
        for (int t = 0; t < col; ++t)
            sum -= a[row * n + k + t] * a[(k + col) * n + k + t];
        a[row * n + k + col] = sum / a[(k + col) * n + k + col];
    }
}

__global__ void trailing_update(double* a, int n, int k, int b, int first, int last) {
    const int row = first + blockIdx.y * blockDim.y + threadIdx.y;
    const int col = k + b + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= last || col > row || col >= n) return;
    double sum = 0.0;
    #pragma unroll
    for (int t = 0; t < kBlock; ++t) {
        if (t < b) sum += a[row * n + k + t] * a[col * n + k + t];
    }
    a[row * n + col] -= sum;
}

static void range_for_rank(int first, int n, int rank, int ranks, int& begin, int& end) {
    // Keep row ownership fixed for the full factorization.  A rank's trailing
    // entries are therefore never consumed by a different rank before they
    // have been communicated as a panel or diagonal block.
    begin = std::max(first, (n * rank) / ranks);
    end = std::max(first, (n * (rank + 1)) / ranks);
}

static bool factor_diagonal(double* block, int b) {
    for (int i = 0; i < b; ++i) {
        double diagonal = block[i * b + i];
        #pragma omp simd reduction(-:diagonal)
        for (int t = 0; t < i; ++t) diagonal -= block[i * b + t] * block[i * b + t];
        if (diagonal <= 0.0) return false;
        block[i * b + i] = std::sqrt(diagonal);
        for (int row = i + 1; row < b; ++row) {
            double value = block[row * b + i];
            #pragma omp simd reduction(-:value)
            for (int t = 0; t < i; ++t) value -= block[row * b + t] * block[i * b + t];
            block[row * b + i] = value / block[i * b + i];
        }
        for (int col = i + 1; col < b; ++col) block[i * b + col] = 0.0;
    }
    return true;
}

static void generate_positive_definite(std::vector<double>& a, int n) {
    std::vector<double> b(static_cast<size_t>(n) * n);
    // Preserve the benchmark's deterministic input stream exactly.
    unsigned int seed = 42;
    for (size_t i = 0; i < b.size(); ++i) b[i] = rand_r(&seed) / double(RAND_MAX) - .5;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0.;
            #pragma omp simd reduction(+:sum)
            for (int t = 0; t < n; ++t) sum += b[static_cast<size_t>(i) * n + t] * b[static_cast<size_t>(j) * n + t];
            a[static_cast<size_t>(i) * n + j] = sum + (i == j ? n : 0);
        }
}

static bool validate(const std::vector<double>& l, const std::vector<double>& original, int n) {
    double max_error = 0., max_relative = 0.;
    #pragma omp parallel for reduction(max:max_error,max_relative) schedule(static)
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        double sum = 0.;
        #pragma omp simd reduction(+:sum)
        for (int t = 0; t <= std::min(i, j); ++t) sum += l[static_cast<size_t>(i) * n + t] * l[static_cast<size_t>(j) * n + t];
        const double error = std::fabs(sum - original[static_cast<size_t>(i) * n + j]);
        max_error = std::max(max_error, error);
        max_relative = std::max(max_relative, error / (std::fabs(original[static_cast<size_t>(i) * n + j]) + 1e-10));
    }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", max_error, max_relative);
    return max_relative <= 1e-6;
}

static void usage(const char* name) {
    std::printf("Usage: %s [-n <num>] [-v] [-r] [-h]\n", name);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512; bool check = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) check = true;
        else if (!std::strcmp(argv[i], "-r")) results = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n <= 0) { if (!rank) std::fprintf(stderr, "Matrix size must be positive\n"); MPI_Finalize(); return 1; }

    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int local_rank; MPI_Comm_rank(local, &local_rank);
    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_Comm_free(&local);

    if (!rank) std::printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nMPI ranks: %d, CUDA block: %d\nValidation: %s\n", n, n, ranks, kBlock, check ? "enabled" : "disabled");
    std::vector<double> a(static_cast<size_t>(n) * n), original;
    generate_positive_definite(a, n);
    if (check && !rank) original = a;
    double* d_a = nullptr; CUDA_CHECK(cudaMalloc(&d_a, a.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> diagonal(kBlock * kBlock), panel(static_cast<size_t>(n) * kBlock);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime(); bool ok = true;
    for (int k = 0; k < n; k += kBlock) {
        const int width = std::min(kBlock, n - k), next = k + width;
        // The diagonal block may straddle rank-owned update ranges.  Contribute
        // just each rank's rows, then assemble it before the serial dependency
        // chain inside the small Cholesky block.
        std::fill(diagonal.begin(), diagonal.begin() + width * width, 0.0);
        int owned_begin, owned_end; range_for_rank(k, n, rank, ranks, owned_begin, owned_end);
        const int copy_begin = std::max(k, owned_begin), copy_end = std::min(next, owned_end);
        if (copy_begin < copy_end)
            CUDA_CHECK(cudaMemcpy2D(diagonal.data() + static_cast<size_t>(copy_begin - k) * width,
                                    width * sizeof(double), d_a + static_cast<size_t>(copy_begin) * n + k,
                                    n * sizeof(double), width * sizeof(double), copy_end - copy_begin,
                                    cudaMemcpyDeviceToHost));
        MPI_Allreduce(MPI_IN_PLACE, diagonal.data(), width * width, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (!rank) ok = factor_diagonal(diagonal.data(), width);
        int good = ok ? 1 : 0; MPI_Bcast(&good, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!good) { ok = false; break; }
        MPI_Bcast(diagonal.data(), width * width, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy2D(d_a + static_cast<size_t>(k) * n + k, n * sizeof(double), diagonal.data(), width * sizeof(double), width * sizeof(double), width, cudaMemcpyHostToDevice));

        int first, last; range_for_rank(next, n, rank, ranks, first, last);
        if (first < last) panel_solve<<<(last - first + 255) / 256, 256>>>(d_a, n, k, width, first, last);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        std::fill(panel.begin(), panel.begin() + static_cast<size_t>(n) * width, 0.0);
        if (first < last) CUDA_CHECK(cudaMemcpy2D(panel.data() + static_cast<size_t>(first) * width, width * sizeof(double), d_a + static_cast<size_t>(first) * n + k, n * sizeof(double), width * sizeof(double), last - first, cudaMemcpyDeviceToHost));
        std::vector<int> counts(ranks), offsets(ranks);
        for (int r = 0; r < ranks; ++r) { int rb, re; range_for_rank(next, n, r, ranks, rb, re); counts[r] = (re - rb) * width; offsets[r] = rb * width; }
        // Each local slice is already at its final displacement in panel.
        // MPI_IN_PLACE avoids undefined overlap between the send and receive buffers.
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, panel.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        if (next < n)
            CUDA_CHECK(cudaMemcpy2D(d_a + static_cast<size_t>(next) * n + k, n * sizeof(double),
                                    panel.data() + static_cast<size_t>(next) * width, width * sizeof(double),
                                    width * sizeof(double), n - next, cudaMemcpyHostToDevice));
        if (first < last && next < n) {
            dim3 threads(16, 16), blocks((n - next + 15) / 16, (last - first + 15) / 16);
            trailing_update<<<blocks, threads>>>(d_a, n, k, width, first, last);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    double elapsed = MPI_Wtime() - start, maximum = 0.;
    MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (ok && (check || results) && !rank) CUDA_CHECK(cudaMemcpy(a.data(), d_a, a.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    if (!ok) { if (!rank) std::fprintf(stderr, "Cholesky decomposition failed: matrix is not positive definite\n"); MPI_Finalize(); return 1; }
    int status = 0;
    if (!rank) {
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n", maximum * 1000., (double(n) * n * n / 3.) / maximum / 1e9);
        if (results) print_results(a, "CholeskyL");
        if (check) { status = validate(a, original, n) ? 0 : 1; std::printf("Validation: %s\n", status ? "FAILED" : "PASSED"); }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
