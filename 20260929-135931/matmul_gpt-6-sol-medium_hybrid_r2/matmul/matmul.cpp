#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
static void blasCheck(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error: %d\n", static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
static void usage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n", name);
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t N = 512;
    bool validate = false, printResults = false, badArgs = false, help = false;
    for (int a = 1; a < argc; ++a) {
        if (std::strcmp(argv[a], "-n") == 0 && a + 1 < argc) {
            char* end;
            const char* arg = argv[++a];
            const unsigned long long value = std::strtoull(arg, &end, 10);
            if (arg[0] == '-' || *end || !value || value > static_cast<unsigned long long>(std::numeric_limits<int>::max())) badArgs = true;
            else N = static_cast<size_t>(value);
        } else if (std::strcmp(argv[a], "-v") == 0) validate = true;
        else if (std::strcmp(argv[a], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[a], "-h") == 0) help = true;
        else badArgs = true;
    }
    if (help || badArgs) {
        if (!rank) usage(argv[0]);
        MPI_Finalize();
        return badArgs ? 1 : 0;
    }
    if (N > std::numeric_limits<size_t>::max() / N ||
        (printResults && N * N > static_cast<size_t>(std::numeric_limits<int>::max()))) {
        if (!rank) std::fprintf(stderr, "Matrix size exceeds supported allocation or MPI count.\n");
        MPI_Finalize();
        return 1;
    }
    if (!rank) {
        std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\nInitializing matrices...\n", validate ? "enabled" : "disabled");
    }
    const size_t first = N * static_cast<size_t>(rank) / ranks;
    const size_t last = N * static_cast<size_t>(rank + 1) / ranks;
    const size_t rows = last - first;
    std::vector<double> A(rows * N), B(rows ? N * N : 0), C(rows * N);
    if (rows) {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j) B[i * N + j] = getPseudoRndValue(N, i, j);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < N; ++j) A[i * N + j] = getPseudoRndValue(N, first + i, j);
    }
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_free(&local);
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) {
        std::fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % devices));
    if (!rank) std::printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (rows) {
        double *dA, *dB, *dC;
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dA), A.size() * sizeof(double)));
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dB), B.size() * sizeof(double)));
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dC), C.size() * sizeof(double)));
        cudaCheck(cudaMemcpy(dA, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice));
        cublasHandle_t handle;
        blasCheck(cublasCreate(&handle));
        const double alpha = 1.0, beta = 0.0;
        // Row-major C = A B corresponds to column-major C^T = B^T A^T.
        blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              static_cast<int>(N), static_cast<int>(rows), static_cast<int>(N),
                              &alpha, dB, static_cast<int>(N), dA, static_cast<int>(N),
                              &beta, dC, static_cast<int>(N)));
        cudaCheck(cudaMemcpy(C.data(), dC, C.size() * sizeof(double), cudaMemcpyDeviceToHost));
        blasCheck(cublasDestroy(handle));
        cudaCheck(cudaFree(dA)); cudaCheck(cudaFree(dB)); cudaCheck(cudaFree(dC));
    }
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000));
        std::printf("Performance: %.3f GFLOPS\n", 2.0 * static_cast<double>(N) * N * N / seconds / 1e9);
    }
    if (printResults) {
        std::vector<double> full;
        std::vector<int> counts, displacements;
        if (!rank) {
            full.resize(N * N);
            counts.resize(ranks); displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                const size_t lo = N * static_cast<size_t>(r) / ranks;
                const size_t hi = N * static_cast<size_t>(r + 1) / ranks;
                counts[r] = static_cast<int>((hi - lo) * N);
                displacements[r] = static_cast<int>(lo * N);
            }
        }
        MPI_Gatherv(C.data(), static_cast<int>(C.size()), MPI_DOUBLE,
                    rank == 0 ? full.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) print_results(full, "MatrixC");
    }
    int localValid = 1;
    if (validate && rows) {
#pragma omp parallel for reduction(&:localValid) schedule(static)
        for (size_t i = 0; i < std::min<size_t>(5, N); ++i) {
            if (i < first || i >= last) continue;
            for (size_t j = 0; j < std::min<size_t>(5, N); ++j) {
                double expected = 0;
                for (size_t k = 0; k < N; ++k) expected += A[(i - first) * N + k] * B[k * N + j];
                const double actual = C[(i - first) * N + j];
                const double error = std::abs((actual - expected) / (expected + 1e-10));
                if (!(error <= 1e-6)) {
#pragma omp critical
                    std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                                i, j, expected, actual, error);
                    localValid = 0;
                }
            }
        }
    }
    int valid = 1;
    if (validate) {
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!rank) std::printf("Validating result...\nValidation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return validate && !valid ? 1 : 0;
}
