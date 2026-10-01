#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// Block cyclic rows distribute both the panels and the level-3 updates.
static constexpr int block = 128;

static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}
static void blasCheck(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %d\n", int(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}
static int globalRow(int local, int rank, int ranks) {
    return (local / block * ranks + rank) * block + local % block;
}
static int localRow(int global, int ranks) {
    return global / (block * ranks) * block + global % block;
}
static int rowCount(int n, int rank, int ranks) {
    int count = 0;
    for (int first = rank * block; first < n; first += ranks * block)
        count += std::min(block, n - first);
    return count;
}
static void gatherMatrix(const std::vector<double>& local, std::vector<double>& full,
                         int n, int rank, int ranks) {
    std::vector<int> counts(ranks), offsets(ranks);
    int total = 0;
    for (int r = 0; r < ranks; ++r) {
        counts[r] = rowCount(n, r, ranks) * n;
        offsets[r] = total;
        total += counts[r];
    }
    std::vector<double> packed;
    if (rank == 0) packed.resize(size_t(n) * n);
    MPI_Gatherv(local.data(), int(local.size()), MPI_DOUBLE,
                packed.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        full.resize(size_t(n) * n);
        for (int r = 0; r < ranks; ++r)
            for (int i = 0; i < rowCount(n, r, ranks); ++i)
                std::copy_n(packed.data() + offsets[r] + size_t(i) * n, n,
                            full.data() + size_t(globalRow(i, r, ranks)) * n);
    }
}
static bool factor(std::vector<double>& local, int n, int rank, int ranks, cublasHandle_t blas) {
    const int rows = rowCount(n, rank, ranks);
    double *deviceA = nullptr, *devicePanel = nullptr;
    cudaCheck(cudaMalloc(&deviceA, std::max<size_t>(1, size_t(rows) * n) * sizeof(double)));
    cudaCheck(cudaMalloc(&devicePanel, size_t(n) * block * sizeof(double)));
    if (rows) cudaCheck(cudaMemcpy(deviceA, local.data(), size_t(rows) * n * sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> localPanel(size_t(rows) * block);
    std::vector<double> panel(size_t(n) * block);
    std::vector<double> diagonal(block * block);
    bool success = true;

    for (int k = 0; k < n; k += block) {
        int width = std::min(block, n - k);
        if (rows) cudaCheck(cudaMemcpy2D(localPanel.data(), size_t(block) * sizeof(double),
                                         deviceA + k, size_t(n) * sizeof(double),
                                         size_t(width) * sizeof(double), rows, cudaMemcpyDeviceToHost));
        int owner = (k / block) % ranks;
        if (rank == owner) {
            int first = localRow(k, ranks);
            for (int i = 0; i < width; ++i) {
                double* row = localPanel.data() + size_t(first + i) * block;
                for (int j = 0; j <= i; ++j) {
                    double value = row[j];
                    for (int p = 0; p < j; ++p)
                        value -= row[p] * diagonal[size_t(j) * block + p];
                    if (i == j) {
                        if (!(value > 0.0)) { success = false; break; }
                        row[j] = std::sqrt(value);
                    } else row[j] = value / diagonal[size_t(j) * block + j];
                    diagonal[size_t(i) * block + j] = row[j];
                }
                if (!success) break;
            }
        }
        int ok = success ? 1 : 0;
        MPI_Bcast(&ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!ok) { success = false; break; }
        MPI_Bcast(diagonal.data(), block * block, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Solve A(i,k) = L(i,k) L(k,k)^T, one independent row per thread.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; ++i) {
            if (globalRow(i, rank, ranks) < k + width) continue;
            double* row = localPanel.data() + size_t(i) * block;
            for (int j = 0; j < width; ++j) {
                double value = row[j];
                for (int p = 0; p < j; ++p)
                    value -= row[p] * diagonal[size_t(j) * block + p];
                row[j] = value / diagonal[size_t(j) * block + j];
            }
        }
        std::fill(panel.begin(), panel.end(), 0.0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; ++i) {
            int global = globalRow(i, rank, ranks);
            if (global < k) continue;
            std::copy_n(localPanel.data() + size_t(i) * block, width,
                        panel.data() + size_t(global) * block);
        }
        MPI_Allreduce(MPI_IN_PLACE, panel.data(), n * block, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (rows) cudaCheck(cudaMemcpy2D(deviceA + k, size_t(n) * sizeof(double),
                                         localPanel.data(), size_t(block) * sizeof(double),
                                         size_t(width) * sizeof(double), rows, cudaMemcpyHostToDevice));
        if (k + width == n) continue;
        cudaCheck(cudaMemcpy(devicePanel, panel.data(), size_t(n) * block * sizeof(double), cudaMemcpyHostToDevice));
        int first = 0;
        while (first < rows && globalRow(first, rank, ranks) < k + width) ++first;
        int remaining = rows - first;
        if (remaining) {
            // Row-major C -= X Y^T is column-major C^T -= Y X^T.
            const double minusOne = -1.0, plusOne = 1.0;
            blasCheck(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                  n - k - width, remaining, width,
                                  &minusOne, devicePanel + size_t(k + width) * block, block,
                                  deviceA + size_t(first) * n + k, n,
                                  &plusOne, deviceA + size_t(first) * n + k + width, n));
        }
    }
    if (success && rows)
        cudaCheck(cudaMemcpy(local.data(), deviceA, size_t(rows) * n * sizeof(double), cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(devicePanel));
    cudaCheck(cudaFree(deviceA));
    if (!success && rank == 0) printf("Error: Matrix is not positive definite\n");
    return success;
}

static void generate(std::vector<double>& local, int n, int rank, int ranks) {
    std::vector<double> B(size_t(n) * n);
    unsigned int seed = 42;
    for (double& value : B) value = (rand_r(&seed) / double(RAND_MAX)) - 0.5;
    int rows = rowCount(n, rank, ranks);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        int global = globalRow(i, rank, ranks);
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int p = 0; p < n; ++p)
                sum += B[size_t(global) * n + p] * B[size_t(j) * n + p];
            local[size_t(i) * n + j] = sum + (global == j ? n : 0);
        }
    }
}
static bool validate(const std::vector<double>& L, const std::vector<double>& original, int n) {
    double absolute = 0.0, relative = 0.0;
    #pragma omp parallel for reduction(max:absolute,relative) schedule(static)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double value = 0;
            for (int p = 0; p <= std::min(i, j); ++p)
                value += L[size_t(i) * n + p] * L[size_t(j) * n + p];
            double error = std::fabs(value - original[size_t(i) * n + j]);
            absolute = std::max(absolute, error);
            relative = std::max(relative, error / (std::fabs(original[size_t(i) * n + j]) + 1e-10));
        }
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n", absolute, relative);
    return relative <= 1e-6;
}
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { if (rank == 0) fprintf(stderr, "CUDA GPU required\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    MPI_Comm shared;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int localRank;
    MPI_Comm_rank(shared, &localRank);
    cudaCheck(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&shared);
    int n = 512;
    bool check = false, print = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) check = true;
        else if (!strcmp(argv[i], "-r")) print = true;
        else if (!strcmp(argv[i], "-h")) {
            if (rank == 0) printf("Usage: %s [options]\n  -n <num> Matrix size (default: 512)\n  -v Validate\n  -r Print results\n  -h Show help\n", argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) fprintf(stderr, "Unknown option or missing value: %s\n", argv[i]);
            MPI_Finalize(); return 1;
        }
    }
    if (n <= 0 || size_t(n) * n > size_t(std::numeric_limits<int>::max()) ||
        size_t(n) * block > size_t(std::numeric_limits<int>::max())) {
        if (rank == 0) fprintf(stderr, "Invalid matrix size\n");
        MPI_Finalize(); return 1;
    }
    if (rank == 0) printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\nGenerating positive definite matrix...\n", n, n, check ? "enabled" : "disabled");
    std::vector<double> local(size_t(rowCount(n, rank, ranks)) * n);
    generate(local, n, rank, ranks);
    std::vector<double> original;
    if (check) gatherMatrix(local, original, n, rank, ranks);
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    cublasHandle_t blas;
    blasCheck(cublasCreate(&blas));
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = factor(local, n, rank, ranks, blas);
    double elapsed = MPI_Wtime() - start;
    blasCheck(cublasDestroy(blas));
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) { if (rank == 0) printf("Cholesky decomposition failed\n"); MPI_Finalize(); return 1; }
    if (rank == 0) {
        printf("Computation time: %ld ms\n", long(duration * 1000));
        printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3.0e9 * duration));
    }
    if (print || check) {
        std::vector<double> full;
        gatherMatrix(local, full, n, rank, ranks);
        if (rank == 0) {
            for (int i = 0; i < n; ++i)
                std::fill(full.begin() + size_t(i) * n + i + 1, full.begin() + size_t(i + 1) * n, 0.0);
            if (print) print_results(full, "CholeskyL");
            if (check) {
                printf("Validating result...\n");
                bool valid = validate(full, original, n);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                success = valid;
            }
        }
        int ok = success ? 1 : 0;
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        success = ok;
    }
    MPI_Finalize();
    return success ? 0 : 1;
}
