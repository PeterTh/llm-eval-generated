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

#define CUDA_CHECK(call) do { \
    const cudaError_t error__ = (call); \
    if (error__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error__)); \
        MPI_Abort(MPI_COMM_WORLD, 2); \
    } \
} while (0)

// One rank owns a contiguous set of rows.  The panel row is broadcast after
// its diagonal is formed, allowing every GPU to update its owned rows.
__global__ void factor_diagonal_kernel(double* a, int row_offset, int pivot, int n, int* ok) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double* row = a + static_cast<size_t>(pivot - row_offset) * n;
        double sum = 0.0;
        for (int k = 0; k < pivot; ++k) sum += row[k] * row[k];
        const double value = row[pivot] - sum;
        if (value <= 0.0) *ok = 0;
        else row[pivot] = sqrt(value);
    }
}

__global__ void update_column_kernel(double* a, const double* panel, int row_offset,
                                     int rows, int pivot, int n) {
    const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_i = row_offset + local_i;
    if (local_i < rows && global_i > pivot) {
        double* row = a + static_cast<size_t>(local_i) * n;
        double sum = 0.0;
        for (int k = 0; k < pivot; ++k) sum += row[k] * panel[k];
        row[pivot] = (row[pivot] - sum) / panel[pivot];
    }
}

__global__ void zero_upper_kernel(double* a, int row_offset, int rows, int n) {
    const size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(rows) * n;
    if (element < count) {
        const int local_i = element / n;
        const int j = element % n;
        if (j > row_offset + local_i) a[element] = 0.0;
    }
}

static void generatePositiveDefiniteMatrix(std::vector<double>& a, int n) {
    std::vector<double> b(static_cast<size_t>(n) * n);
    unsigned int seed = 42;
    // Keep the benchmark's deterministic input stream while parallelizing the
    // cubic matrix product, which is significant for modest matrix sizes.
    for (double& value : b) value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
#pragma omp simd reduction(+:sum)
            for (int k = 0; k < n; ++k) sum += b[static_cast<size_t>(i) * n + k] * b[static_cast<size_t>(j) * n + k];
            a[static_cast<size_t>(i) * n + j] = sum + (i == j ? n : 0.0);
        }
    }
}

static bool distributedCholesky(std::vector<double>& local_a, int n, int row_offset,
                                 int local_rows, int rank, const std::vector<int>& owners) {
    double *device_a = nullptr, *device_panel = nullptr;
    int *device_ok = nullptr, ok = 1;
    const size_t local_bytes = local_a.size() * sizeof(double);
    // Empty row partitions are possible when a launch uses more ranks than
    // matrix rows; keep a valid allocation but never launch work for them.
    CUDA_CHECK(cudaMalloc(&device_a, std::max(local_bytes, sizeof(double))));
    CUDA_CHECK(cudaMalloc(&device_panel, static_cast<size_t>(n) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&device_ok, sizeof(int)));
    if (local_bytes) CUDA_CHECK(cudaMemcpy(device_a, local_a.data(), local_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_ok, &ok, sizeof(int), cudaMemcpyHostToDevice));
    std::vector<double> panel(n);
    constexpr int threads = 256;

    for (int pivot = 0; pivot < n; ++pivot) {
        const int owner = owners[pivot];
        if (rank == owner) {
            factor_diagonal_kernel<<<1, 1>>>(device_a, row_offset, pivot, n, device_ok);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&ok, device_ok, sizeof(int), cudaMemcpyDeviceToHost));
            if (ok) CUDA_CHECK(cudaMemcpy(panel.data(), device_a + static_cast<size_t>(pivot - row_offset) * n,
                                           static_cast<size_t>(pivot + 1) * sizeof(double), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(&ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!ok) break;
        MPI_Bcast(panel.data(), pivot + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(device_panel, panel.data(), static_cast<size_t>(pivot + 1) * sizeof(double), cudaMemcpyHostToDevice));
        if (local_rows) {
            const int blocks = (local_rows + threads - 1) / threads;
            update_column_kernel<<<blocks, threads>>>(device_a, device_panel, row_offset, local_rows, pivot, n);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    if (ok) {
        const size_t count = static_cast<size_t>(local_rows) * n;
        if (count) {
            zero_upper_kernel<<<(count + threads - 1) / threads, threads>>>(device_a, row_offset, local_rows, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(local_a.data(), device_a, local_bytes, cudaMemcpyDeviceToHost));
        }
    }
    CUDA_CHECK(cudaFree(device_ok)); CUDA_CHECK(cudaFree(device_panel)); CUDA_CHECK(cudaFree(device_a));
    return ok != 0;
}

static bool validateGathered(const std::vector<double>& l, const std::vector<double>& original, int n) {
    double max_relative = 0.0, max_absolute = 0.0;
#pragma omp parallel for reduction(max:max_relative,max_absolute) schedule(static)
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        double sum = 0.0;
#pragma omp simd reduction(+:sum)
        for (int k = 0; k <= std::min(i, j); ++k) sum += l[static_cast<size_t>(i) * n + k] * l[static_cast<size_t>(j) * n + k];
        const double error = std::fabs(sum - original[static_cast<size_t>(i) * n + j]);
        max_absolute = std::max(max_absolute, error);
        max_relative = std::max(max_relative, error / (std::fabs(original[static_cast<size_t>(i) * n + j]) + 1e-10));
    }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", max_absolute, max_relative);
    return max_relative <= 1e-6;
}

static void printUsage(const char* name) { std::printf("Usage: %s [-n <num>] [-v] [-r] [-h]\n", name); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512; bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) results = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (n <= 0 || n > std::numeric_limits<int>::max() / n) { if (!rank) std::fprintf(stderr, "Invalid matrix size\n"); MPI_Finalize(); return 1; }
    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(rank % devices));
    std::vector<int> row_counts(ranks), displacements(ranks), owners(n);
    const int base = n / ranks, extra = n % ranks;
    int offset = 0;
    for (int r = 0; r < ranks; ++r) { row_counts[r] = base + (r < extra); displacements[r] = offset; for (int i = 0; i < row_counts[r]; ++i) owners[offset + i] = r; offset += row_counts[r]; }
    const int local_rows = row_counts[rank], row_offset = displacements[rank];
    std::vector<int> element_counts(ranks), element_displacements(ranks);
    for (int r = 0; r < ranks; ++r) { element_counts[r] = row_counts[r] * n; element_displacements[r] = displacements[r] * n; }
    std::vector<double> full_a, original;
    if (!rank) { full_a.resize(static_cast<size_t>(n) * n); generatePositiveDefiniteMatrix(full_a, n); if (validate) original = full_a; }
    std::vector<double> local_a(static_cast<size_t>(local_rows) * n);
    MPI_Scatterv(rank ? nullptr : full_a.data(), element_counts.data(), element_displacements.data(), MPI_DOUBLE,
                 local_a.data(), element_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (!rank) { std::printf("Cholesky Decomposition Benchmark (MPI + OpenMP + CUDA)\nMatrix size: %d x %d\nMPI ranks: %d\n", n, n, ranks); }
    MPI_Barrier(MPI_COMM_WORLD); const auto start = std::chrono::steady_clock::now();
    const bool success = distributedCholesky(local_a, n, row_offset, local_rows, rank, owners);
    MPI_Barrier(MPI_COMM_WORLD); const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (results || validate) { if (!rank) full_a.resize(static_cast<size_t>(n) * n); MPI_Gatherv(local_a.data(), element_counts[rank], MPI_DOUBLE, rank ? nullptr : full_a.data(), element_counts.data(), element_displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD); }
    int exit_code = success ? 0 : 1;
    if (!rank) {
        if (!success) std::printf("Cholesky decomposition failed\n");
        else { std::printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n", seconds * 1e3, (double(n) * n * n / 3.0) / seconds / 1e9); if (results) print_results(full_a, "CholeskyL"); if (validate) { const bool valid = validateGathered(full_a, original, n); std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); exit_code = valid ? 0 : 1; } }
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Finalize(); return exit_code;
}
