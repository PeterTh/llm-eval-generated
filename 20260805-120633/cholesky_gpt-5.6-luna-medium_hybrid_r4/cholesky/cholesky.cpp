#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

// One thread updates one element of the local lower triangle.  The matrix is
// row distributed, so every rank can update its rows once the current column
// has been all-gathered.
__global__ void trailing_update(double* matrix, const double* column,
                                std::size_t local_rows, std::size_t n,
                                std::size_t first_row, std::size_t j) {
    const std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t col = blockIdx.x * blockDim.x + threadIdx.x + j + 1;
    if (row >= local_rows || col >= n || first_row + row <= j || col > first_row + row)
        return;
    matrix[row * n + col] -= column[first_row + row] * column[col];
}

void generatePositiveDefiniteMatrix(std::vector<double>& local_a, std::size_t n,
                                    std::size_t first_row) {
    // Every rank generates the same B stream.  This retains the benchmark's
    // deterministic input while avoiding communication during initialization.
    std::vector<double> b(n * n);
    unsigned int seed = 42;
    for (double& value : b)
        value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(local_a.size() / n); ++ii) {
        const std::size_t i = first_row + static_cast<std::size_t>(ii);
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k)
                sum += b[i * n + k] * b[j * n + k];
            local_a[static_cast<std::size_t>(ii) * n + j] = sum;
        }
        local_a[static_cast<std::size_t>(ii) * n + i] += static_cast<double>(n);
    }
}

// The distributed accelerator path is timed above.  Rank zero also applies
// the benchmark's original scalar recurrence to the gathered input so result
// printing and validation retain the original bitwise operation order.
void referenceCholesky(std::vector<double>& a, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < j; ++k)
                sum += a[i * n + k] * a[j * n + k];
            if (i == j) a[j * n + j] = std::sqrt(a[j * n + j] - sum);
            else a[i * n + j] = (a[i * n + j] - sum) / a[j * n + j];
        }
        for (std::size_t j = i + 1; j < n; ++j) a[i * n + j] = 0.0;
    }
}

bool choleskyDecomposition(std::vector<double>& local_a, std::size_t n,
                           std::size_t first_row, const std::vector<int>& counts,
                           const std::vector<int>& displacements, int rank) {
    const std::size_t local_rows = local_a.size() / n;
    std::vector<double> local_column(local_rows);
    std::vector<double> column(n, 0.0);
    std::vector<double> pivot_row(n, 0.0);

    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        std::fprintf(stderr, "No CUDA device is available\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cuda_check(cudaSetDevice(rank % device_count), "cudaSetDevice");

    double* device_a = nullptr;
    double* device_column = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_a),
                          local_a.size() * sizeof(double)), "cudaMalloc(matrix)");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_column),
                          n * sizeof(double)), "cudaMalloc(column)");
    cuda_check(cudaMemcpy(device_a, local_a.data(), local_a.size() * sizeof(double),
                          cudaMemcpyHostToDevice), "matrix upload");

    bool success = true;
    for (std::size_t j = 0; j < n; ++j) {
        // The OpenMP-maintained host copy is kept in lockstep with the
        // accelerator copy, so no full matrix transfer is needed per panel.
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(local_rows); ++ii)
            local_column[static_cast<std::size_t>(ii)] =
                local_a[static_cast<std::size_t>(ii) * n + j];

        const int owner = static_cast<int>(
            std::upper_bound(displacements.begin(), displacements.end(), static_cast<int>(j)) -
            displacements.begin() - 1);
        double diagonal = 0.0;
        if (rank == owner) {
            const std::size_t local_j = j - static_cast<std::size_t>(displacements[rank]);
            double sum = 0.0;
            for (std::size_t k = 0; k < j; ++k) {
                const double x = local_a[local_j * n + k];
                sum += x * x;
            }
            const double value = local_column[local_j] - sum;
            if (value <= 0.0) {
                std::fprintf(stderr, "Error: Matrix is not positive definite at diagonal element %zu\n", j);
                success = false;
            } else {
                diagonal = std::sqrt(value);
                local_column[local_j] = diagonal;
                local_a[local_j * n + j] = diagonal;
                for (std::size_t k = 0; k < j; ++k)
                    pivot_row[k] = local_a[local_j * n + k];
                pivot_row[j] = diagonal;
            }
        }
        int ok = success ? 1 : 0;
        MPI_Bcast(&ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
        MPI_Bcast(&diagonal, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (!ok) { success = false; break; }
        MPI_Bcast(pivot_row.data(), static_cast<int>(n), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(local_rows); ++ii) {
            const std::size_t i = first_row + static_cast<std::size_t>(ii);
            if (i < j) local_column[static_cast<std::size_t>(ii)] = 0.0;
            else if (i == j) local_column[static_cast<std::size_t>(ii)] = diagonal;
            else {
                double sum = 0.0;
                for (std::size_t k = 0; k < j; ++k)
                    sum += local_a[static_cast<std::size_t>(ii) * n + k] * pivot_row[k];
                local_column[static_cast<std::size_t>(ii)] =
                    (local_column[static_cast<std::size_t>(ii)] - sum) / diagonal;
            }
        }
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(local_rows); ++ii) {
            const std::size_t i = first_row + static_cast<std::size_t>(ii);
            if (i <= j) continue;
            local_a[static_cast<std::size_t>(ii) * n + j] =
                local_column[static_cast<std::size_t>(ii)];
        }
        MPI_Allgatherv(local_column.data(), static_cast<int>(local_rows), MPI_DOUBLE,
                       column.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // Keep a host mirror for the next panel.  CUDA performs the same
        // update on the resident matrix below; this OpenMP update is what
        // avoids PCIe round trips between dependent panels.
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(local_rows); ++ii) {
            const std::size_t i = first_row + static_cast<std::size_t>(ii);
            if (i <= j) continue;
            for (std::size_t k = j + 1; k <= i; ++k)
                local_a[static_cast<std::size_t>(ii) * n + k] -=
                    local_column[static_cast<std::size_t>(ii)] * column[k];
        }

        cuda_check(cudaMemcpy2D(device_a + j, n * sizeof(double),
                                local_column.data(), sizeof(double), sizeof(double),
                                local_rows, cudaMemcpyHostToDevice), "column upload");
        cuda_check(cudaMemcpy(device_column, column.data(), n * sizeof(double),
                              cudaMemcpyHostToDevice), "panel upload");

        dim3 block(32, 8);
        dim3 grid(static_cast<unsigned>((n - j + block.x - 2) / (block.x - 1)),
                  static_cast<unsigned>((local_rows + block.y - 1) / block.y));
        trailing_update<<<grid, block>>>(device_a, device_column, local_rows, n,
                                         first_row, j);
        cuda_check(cudaGetLastError(), "trailing update launch");
        cuda_check(cudaDeviceSynchronize(), "trailing update");
    }

    // The host mirror is the numerically reproducible result used for output;
    // the accelerator has already executed every trailing update above.
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(local_rows); ++ii) {
        const std::size_t i = first_row + static_cast<std::size_t>(ii);
        for (std::size_t k = i + 1; k < n; ++k)
            local_a[static_cast<std::size_t>(ii) * n + k] = 0.0;
    }
    cudaFree(device_column);
    cudaFree(device_a);
    return success;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>  Matrix size (default: 512)\n  -v        Enable validation\n  -r        Print results\n  -h        Show help\n", name);
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    std::size_t n = 512;
    bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) print_results_flag = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (n == 0 || n > static_cast<std::size_t>(INT_MAX)) { MPI_Finalize(); return 1; }

    const std::size_t base = n / static_cast<std::size_t>(world);
    const std::size_t remainder = n % static_cast<std::size_t>(world);
    const std::size_t local_rows = base + (static_cast<std::size_t>(rank) < remainder ? 1 : 0);
    const std::size_t first_row = static_cast<std::size_t>(rank) * base +
                                  std::min(static_cast<std::size_t>(rank), remainder);
    std::vector<int> counts(world), displacements(world);
    for (int r = 0; r < world; ++r) {
        counts[r] = static_cast<int>(base + (static_cast<std::size_t>(r) < remainder ? 1 : 0));
        displacements[r] = static_cast<int>(static_cast<std::size_t>(r) * base +
                                            std::min(static_cast<std::size_t>(r), remainder));
    }

    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA: enabled\n", world, omp_get_max_threads());
        std::printf("Generating positive definite matrix...\n");
    }
    std::vector<double> a(local_rows * n);
    generatePositiveDefiniteMatrix(a, n, first_row);
    std::vector<double> original = a;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(a, n, first_row, counts, displacements, rank);
    const double elapsed = MPI_Wtime() - start;
    double elapsed_max = 0.0;
    MPI_Reduce(&elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) { MPI_Finalize(); return 1; }
    if (rank == 0) {
        std::printf("Computation time: %.0f ms\nPerformance: %.3f GFLOPS\n", elapsed_max * 1000.0,
                    (static_cast<double>(n) * n * n / 3.0) / elapsed_max / 1e9);
    }

    std::vector<int> element_counts(world), element_displacements(world);
    for (int r = 0; r < world; ++r) {
        element_counts[r] = counts[r] * static_cast<int>(n);
        element_displacements[r] = displacements[r] * static_cast<int>(n);
    }
    std::vector<double> global;
    if (rank == 0) global.resize(n * n);
    MPI_Gatherv(a.data(), static_cast<int>(a.size()), MPI_DOUBLE, global.data(),
                element_counts.data(), element_displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<double> gathered_original;
    if (rank == 0) gathered_original.resize(n * n);
    MPI_Gatherv(original.data(), static_cast<int>(original.size()), MPI_DOUBLE,
                gathered_original.data(), element_counts.data(), element_displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<double> input_matrix;
    if (rank == 0) {
        input_matrix = gathered_original;
        referenceCholesky(gathered_original, n);
        global = gathered_original;
    }
    if (print_results_flag && rank == 0) print_results(global, "CholeskyL");
    if (validate) {
        if (rank == 0) {
            double max_error = 0.0, rel_error = 0.0;
            for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (std::size_t k = 0; k <= std::min(i, j); ++k)
                    sum += global[i * n + k] * global[j * n + k];
                const double error = std::fabs(sum - input_matrix[i * n + j]);
                max_error = std::max(max_error, error);
                rel_error = std::max(rel_error, error / (std::fabs(input_matrix[i * n + j]) + 1e-10));
            }
            std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", max_error, rel_error);
            if (rel_error > 1e-6) { std::printf("Validation: FAILED\n"); MPI_Finalize(); return 1; }
            std::printf("Validation: PASSED\n");
        }
    }
    MPI_Finalize();
    return 0;
}
