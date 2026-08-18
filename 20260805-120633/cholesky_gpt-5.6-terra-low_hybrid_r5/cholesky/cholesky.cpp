#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Every rank has a GPU-resident working matrix.  A rank updates only the rows
// it owns (row % nranks == rank); completed rows are broadcast as panels.
__global__ void scale_panel(double* a, size_t n, size_t k, int rank, int nranks,
                            double diagonal) {
    const size_t i = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && static_cast<int>(i % nranks) == rank)
        a[i * n + k] /= diagonal;
}

__global__ void update_trailing(double* a, size_t n, size_t k, int rank,
                                int nranks) {
    const size_t i = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    const size_t j = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j <= i && static_cast<int>(i % nranks) == rank)
        a[i * n + j] -= a[i * n + k] * a[j * n + k];
}

static void cudaCheck(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", where,
                     cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

static bool hybridCholesky(std::vector<double>& a, size_t n, int rank, int ranks) {
    double* device = nullptr;
    const size_t bytes = n * n * sizeof(double);
    cudaCheck(cudaMalloc(&device, bytes), "cudaMalloc");
    cudaCheck(cudaMemcpy(device, a.data(), bytes, cudaMemcpyHostToDevice), "copy input");
    std::vector<double> column(n, 0.0);

    const dim3 block(16, 16);
    for (size_t k = 0; k < n; ++k) {
        const int owner = static_cast<int>(k % ranks);
        double diagonal = 0.0;
        if (rank == owner) {
            cudaCheck(cudaMemcpy(&diagonal, device + k * n + k, sizeof(double),
                                 cudaMemcpyDeviceToHost), "copy diagonal");
            if (diagonal > 0.0) diagonal = std::sqrt(diagonal);
        }
        MPI_Bcast(&diagonal, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (!(diagonal > 0.0) || !std::isfinite(diagonal)) {
            if (rank == 0)
                std::fprintf(stderr, "Error: Matrix is not positive definite at diagonal element %zu\n", k);
            cudaFree(device);
            return false;
        }
        if (rank == owner)
            cudaCheck(cudaMemcpy(device + k * n + k, &diagonal, sizeof(double),
                                 cudaMemcpyHostToDevice), "store diagonal");

        const size_t remaining = n - k - 1;
        if (remaining) {
            scale_panel<<<(remaining + 255) / 256, 256>>>(device, n, k, rank, ranks, diagonal);
            cudaCheck(cudaGetLastError(), "scale_panel launch");
            cudaCheck(cudaDeviceSynchronize(), "scale panel");
            // Gather the distributed panel column before the rank-local GEMM
            // update: every row update consumes all earlier column entries.
            cudaCheck(cudaMemcpy2D(column.data() + k + 1, sizeof(double),
                                   device + (k + 1) * n + k, n * sizeof(double),
                                   sizeof(double), remaining, cudaMemcpyDeviceToHost), "copy panel column");
            #pragma omp parallel for schedule(static)
            for (long long i = static_cast<long long>(k + 1); i < static_cast<long long>(n); ++i)
                if (static_cast<int>(static_cast<size_t>(i) % ranks) != rank) column[static_cast<size_t>(i)] = 0.0;
            MPI_Allreduce(MPI_IN_PLACE, column.data() + k + 1, static_cast<int>(remaining), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            cudaCheck(cudaMemcpy2D(device + (k + 1) * n + k, n * sizeof(double),
                                   column.data() + k + 1, sizeof(double), sizeof(double), remaining,
                                   cudaMemcpyHostToDevice), "broadcast panel column");
            const dim3 grid((remaining + block.x - 1) / block.x,
                            (remaining + block.y - 1) / block.y);
            update_trailing<<<grid, block>>>(device, n, k, rank, ranks);
            cudaCheck(cudaGetLastError(), "update_trailing launch");
        }
        // The owning rank needs its completed row before it becomes the next panel.
        cudaCheck(cudaDeviceSynchronize(), "factorization iteration");
        if (rank == owner)
            cudaCheck(cudaMemcpy(a.data() + k * n, device + k * n, n * sizeof(double),
                                 cudaMemcpyDeviceToHost), "copy panel");
        MPI_Bcast(a.data() + k * n, static_cast<int>(n), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(device + k * n, a.data() + k * n, n * sizeof(double),
                             cudaMemcpyHostToDevice), "broadcast panel");
    }
    cudaCheck(cudaMemcpy(a.data(), device, bytes, cudaMemcpyDeviceToHost), "copy output");
    cudaCheck(cudaFree(device), "cudaFree");

    // Only owned lower rows are authoritative locally.  Sum them to assemble L.
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        for (size_t j = 0; j < n; ++j)
            if (j > static_cast<size_t>(i) || static_cast<int>(i % ranks) != rank)
                a[static_cast<size_t>(i) * n + j] = 0.0;
    }
    std::vector<double> assembled(n * n);
    MPI_Allreduce(a.data(), assembled.data(), static_cast<int>(n * n), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    a.swap(assembled);
    return true;
}

static void generatePositiveDefiniteMatrix(std::vector<double>& a, size_t n) {
    std::vector<double> b(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) b[i] = rand_r(&seed) / static_cast<double>(RAND_MAX) - .5;
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k) sum += b[static_cast<size_t>(i) * n + k] * b[j * n + k];
            a[static_cast<size_t>(i) * n + j] = sum + (static_cast<size_t>(i) == j ? n : 0.0);
        }
}

static bool validateCholesky(const std::vector<double>& l, const std::vector<double>& original, size_t n) {
    double maxRelative = 0.0, maxAbsolute = 0.0;
    #pragma omp parallel for reduction(max:maxRelative,maxAbsolute) schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k <= std::min(static_cast<size_t>(i), j); ++k)
                sum += l[static_cast<size_t>(i) * n + k] * l[j * n + k];
            const double error = std::abs(sum - original[static_cast<size_t>(i) * n + j]);
            maxAbsolute = std::max(maxAbsolute, error);
            maxRelative = std::max(maxRelative, error / (std::abs(original[static_cast<size_t>(i) * n + j]) + 1e-10));
        }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxAbsolute, maxRelative);
    return maxRelative <= 1e-6;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num> Matrix size (default: 512)\n  -v Validate\n  -r Print results\n  -h Help\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    // Map local MPI ranks round-robin over the accelerators on each node.
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(local, &localRank);
    cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices == 0) {
        if (!rank) std::fprintf(stderr, "No CUDA accelerator is available\n");
        MPI_Comm_free(&local); MPI_Finalize(); return 2;
    }
    cudaCheck(cudaSetDevice(localRank % devices), "cudaSetDevice");
    MPI_Comm_free(&local);
    size_t n = 512; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!n || n > static_cast<size_t>(std::numeric_limits<int>::max()) || n > std::numeric_limits<size_t>::max() / n) {
        if (!rank) std::fprintf(stderr, "Invalid matrix size\n"); MPI_Finalize(); return 1;
    }
    if (!rank) std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nMPI ranks: %d\nValidation: %s\nGenerating positive definite matrix...\n", n, n, ranks, validate ? "enabled" : "disabled");
    std::vector<double> a(n * n), original;
    generatePositiveDefiniteMatrix(a, n);
    if (validate) original = a;
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    const bool success = hybridCholesky(a, n, rank, ranks);
    const double localSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double seconds = 0.0; MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int localOk = success ? 1 : 0, allOk = 0; MPI_Allreduce(&localOk, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!allOk) { MPI_Finalize(); return 1; }
    if (!rank) {
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n", seconds * 1000.0,
                    static_cast<double>(n) * n * n / (3.0 * seconds * 1e9));
        if (printResults) print_results(a, "CholeskyL");
        if (validate) std::printf("Validation: %s\n", validateCholesky(a, original, n) ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return 0;
}
