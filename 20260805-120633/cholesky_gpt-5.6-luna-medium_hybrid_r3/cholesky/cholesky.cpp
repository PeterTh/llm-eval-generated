#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t e__ = (call); \
    if (e__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e__)); \
        MPI_Abort(MPI_COMM_WORLD, 2); \
    } \
} while (0)

// One thread computes one local matrix element.  A is distributed by rows,
// while the complete pivot column is replicated by MPI for the rank-local
// trailing update.
__global__ void diagonal_kernel(const double* a, size_t local_row, size_t pivot, size_t n, double* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const double* r = a + local_row * n;
        const double value = r[pivot];
        *out = value > 0.0 ? sqrt(value) : -1.0;
    }
}

__global__ void column_kernel(double* a, size_t first, size_t rows, size_t n,
                              size_t pivot, double pivot_value, double* local_col) {
    const size_t li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li < rows) {
        const size_t i = first + li;
        if (i > pivot) {
            const double* r = a + li * n;
            local_col[li] = r[pivot] / pivot_value;
            a[li * n + pivot] = local_col[li];
        } else if (i == pivot) {
            local_col[li] = pivot_value;
            a[li * n + pivot] = pivot_value;
        } else {
            local_col[li] = 0.0;
        }
    }
}

__global__ void trailing_update_kernel(double* a, size_t first, size_t rows, size_t n,
                                       size_t pivot, const double* column) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = rows * n;
    if (x < total) {
        const size_t li = x / n;
        const size_t j = x % n;
        const size_t i = first + li;
        if (i > pivot && j > pivot) a[x] -= column[i] * column[j];
        if (j > i) a[x] = 0.0;
    }
}

static void generatePositiveDefiniteMatrix(std::vector<double>& a, size_t first,
                                           size_t rows, size_t n) {
    std::vector<double> b(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;

    #pragma omp parallel for schedule(static)
    for (long long li = 0; li < static_cast<long long>(rows); ++li) {
        const size_t i = first + static_cast<size_t>(li);
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += b[i * n + k] * b[j * n + k];
            a[static_cast<size_t>(li) * n + j] = sum;
        }
        a[static_cast<size_t>(li) * n + i] += static_cast<double>(n);
    }
}

static int owner(size_t row, const std::vector<int>& counts, const std::vector<int>& displs) {
    for (int r = 0; r < static_cast<int>(counts.size()); ++r)
        if (row >= static_cast<size_t>(displs[r]) && row < static_cast<size_t>(displs[r] + counts[r])) return r;
    return static_cast<int>(counts.size()) - 1;
}

static bool validateCholesky(const std::vector<double>& local_l, const std::vector<double>& original,
                             size_t first, size_t rows, size_t n, MPI_Comm comm) {
    int world = 1;
    MPI_Comm_size(comm, &world);
    std::vector<int> counts(world), displs(world);
    const int local_elements = static_cast<int>(local_l.size());
    MPI_Allgather(&local_elements, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    displs[0] = 0;
    for (int r = 1; r < world; ++r) displs[r] = displs[r - 1] + counts[r - 1];
    std::vector<double> l(n * n);
    MPI_Allgatherv(local_l.data(), local_elements, MPI_DOUBLE, l.data(), counts.data(), displs.data(), MPI_DOUBLE, comm);

    double max_error = 0.0, max_relative = 0.0;
    #pragma omp parallel for reduction(max:max_error,max_relative) schedule(static)
    for (long long lli = 0; lli < static_cast<long long>(rows); ++lli) {
        const size_t i = first + static_cast<size_t>(lli);
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += l[i * n + k] * l[j * n + k];
            const double error = std::fabs(sum - original[static_cast<size_t>(lli) * n + j]);
            max_error = std::max(max_error, error);
            max_relative = std::max(max_relative, error / (std::fabs(original[static_cast<size_t>(lli) * n + j]) + 1e-10));
        }
    }
    double global_error = 0.0, global_relative = 0.0;
    MPI_Reduce(&max_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&max_relative, &global_relative, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    int rank = 0; MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        printf("Max absolute error: %.10e\nMax relative error: %.10e\n", global_error, global_relative);
        if (global_relative > 1e-6) { printf("Validation failed: relative error too large\n"); return false; }
    }
    int ok = (rank == 0) ? (global_relative <= 1e-6) : 1;
    MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
    return ok != 0;
}

static void printUsage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -n <num> Matrix size (default: 512)\n  -v Validate\n  -r Print results\n  -h Show help\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &world);
    size_t n = 512; bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = static_cast<size_t>(atoll(argv[++i]));
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print_results_flag = true;
        else if (!strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n == 0 || n > static_cast<size_t>(INT_MAX)) { if (rank == 0) printf("Invalid matrix size\n"); MPI_Finalize(); return 1; }

    std::vector<int> counts(world), displs(world);
    for (int r = 0; r < world; ++r) {
        const size_t begin = n * static_cast<size_t>(r) / world;
        const size_t end = n * static_cast<size_t>(r + 1) / world;
        counts[r] = static_cast<int>(end - begin); displs[r] = static_cast<int>(begin);
    }
    const size_t first = static_cast<size_t>(displs[rank]), rows = static_cast<size_t>(counts[rank]);
    std::vector<double> a(rows * n), original;
    if (validate) original.resize(rows * n);
    if (rank == 0) { printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled"); printf("Generating positive definite matrix...\n"); }
    generatePositiveDefiniteMatrix(a, first, rows, n);
    if (validate) original = a;

    int device_count = 0; CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) { if (rank == 0) fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(rank % device_count));
    double* d_a = nullptr; double* d_col = nullptr; double* d_diag = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_col, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_diag, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> local_col(rows), column(n);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    bool success = true;
    const int threads = 256;
    for (size_t k = 0; k < n; ++k) {
        const int pivot_owner = owner(k, counts, displs);
        if (rank == pivot_owner) {
            diagonal_kernel<<<1, 1>>>(d_a, k - first, k, n, d_diag);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaMemcpy(&column[k], d_diag, sizeof(double), cudaMemcpyDeviceToHost));
            if (column[k] <= 0.0) success = false;
        }
        MPI_Bcast(&column[k], 1, MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);
        if (!success) break;
        const size_t blocks = (rows + threads - 1) / threads;
        column_kernel<<<static_cast<unsigned>(blocks), threads>>>(d_a, first, rows, n, k, column[k], d_col);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaMemcpy(local_col.data(), d_col, rows * sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Allgatherv(local_col.data(), counts[rank], MPI_DOUBLE, column.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(d_col, column.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        const size_t total = rows * n;
        trailing_update_kernel<<<static_cast<unsigned>((total + threads - 1) / threads), threads>>>(d_a, first, rows, n, k, d_col);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(a.data(), d_a, a.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_diag)); CUDA_CHECK(cudaFree(d_col)); CUDA_CHECK(cudaFree(d_a));
    MPI_Allreduce(MPI_IN_PLACE, &success, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    const double local_seconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) { if (rank == 0) printf("Cholesky decomposition failed\n"); MPI_Finalize(); return 1; }
    if (rank == 0) {
        printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n", static_cast<long>(seconds * 1000.0), (n * n * n / 3.0) / seconds / 1e9);
    }
    if (print_results_flag) {
        std::vector<double> full;
        if (rank == 0) full.resize(n * n);
        std::vector<int> element_counts(world), element_displs(world);
        for (int r = 0; r < world; ++r) {
            element_counts[r] = counts[r] * static_cast<int>(n);
            element_displs[r] = displs[r] * static_cast<int>(n);
        }
        MPI_Gatherv(a.data(), static_cast<int>(a.size()), MPI_DOUBLE, rank == 0 ? full.data() : nullptr,
                    element_counts.data(), element_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(full, "CholeskyL");
    }
    bool valid = true;
    if (validate) { if (rank == 0) printf("Validating result...\n"); valid = validateCholesky(a, original, first, rows, n, MPI_COMM_WORLD); if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); }
    MPI_Finalize(); return valid ? 0 : 1;
}
