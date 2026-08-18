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

namespace {
constexpr int BLOCK_SIZE = 128;

[[noreturn]] void cudaFail(cudaError_t e, const char* where) {
    std::fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e));
    MPI_Abort(MPI_COMM_WORLD, 2);
    std::abort();
}

void cudaCheck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) cudaFail(e, where);
}

// Each thread updates one element of a row owned by this MPI rank.  The
// panel is replicated on every GPU; the matrix itself is distributed by rows.
__global__ void trailingUpdate(double* a, size_t rows, size_t n, size_t firstRow,
                               const double* panel, size_t panelWidth,
                               size_t trail) {
    const size_t j = trail + blockIdx.x * blockDim.x + threadIdx.x;
    const size_t localI = blockIdx.y * blockDim.y + threadIdx.y;
    if (localI >= rows || j >= n) return;
    const size_t i = firstRow + localI;
    if (j > i) return;

    double sum = 0.0;
    for (size_t p = 0; p < panelWidth; ++p)
        sum += panel[i * panelWidth + p] * panel[j * panelWidth + p];
    a[localI * n + j] -= sum;
}

// A rank-contiguous row distribution balances both the panel gather and the
// CUDA update, including when n is not divisible by the number of ranks.
void rowPartition(size_t n, int ranks, int rank, size_t& first, size_t& rows) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    rows = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    first = static_cast<size_t>(rank) * base +
            std::min(static_cast<size_t>(rank), extra);
}

bool choleskyDecomposition(std::vector<double>& localA, size_t n,
                           size_t firstRow, size_t localRows,
                           int rank, int ranks) {
    std::vector<int> panelCounts(ranks), panelDispls(ranks);
    for (int r = 0; r < ranks; ++r) {
        size_t f = 0, rows = 0;
        rowPartition(n, ranks, r, f, rows);
        panelCounts[r] = static_cast<int>(rows * BLOCK_SIZE);
        // displacements are filled per panel below because the panel width changes.
    }

    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices == 0) cudaFail(cudaErrorNoDevice, "cudaGetDeviceCount");
    cudaCheck(cudaSetDevice(rank % devices), "cudaSetDevice");

    double* dA = nullptr;
    double* dPanel = nullptr;
    cudaCheck(cudaMalloc(&dA, localRows * n * sizeof(double)), "cudaMalloc(dA)");
    cudaCheck(cudaMalloc(&dPanel, n * BLOCK_SIZE * sizeof(double)), "cudaMalloc(panel)");
    cudaCheck(cudaMemcpy(dA, localA.data(), localRows * n * sizeof(double),
                         cudaMemcpyHostToDevice), "cudaMemcpy(A)");

    std::vector<double> localPanel(localRows * BLOCK_SIZE);
    std::vector<double> panel(n * BLOCK_SIZE);

    for (size_t kk = 0; kk < n; kk += BLOCK_SIZE) {
        const size_t kb = std::min(static_cast<size_t>(BLOCK_SIZE), n - kk);
        const size_t trail = kk + kb;
        for (int r = 0; r < ranks; ++r) {
            size_t f = 0, rows = 0;
            rowPartition(n, ranks, r, f, rows);
            panelCounts[r] = static_cast<int>(rows * kb);
            panelDispls[r] = static_cast<int>(f * kb);
        }

        // Pack this rank's panel columns and gather them to the panel owner.
        for (size_t i = 0; i < localRows; ++i)
            std::memcpy(localPanel.data() + i * kb,
                        localA.data() + i * n + kk, kb * sizeof(double));
        MPI_Gatherv(localPanel.data(), static_cast<int>(localRows * kb), MPI_DOUBLE,
                    panel.data(), panelCounts.data(), panelDispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Factor the diagonal block and triangularly solve the panel below it.
            for (size_t j = kk; j < trail; ++j) {
                double sum = 0.0;
                for (size_t p = kk; p < j; ++p)
                    sum += panel[j * kb + (p - kk)] * panel[j * kb + (p - kk)];
                const double value = panel[j * kb + (j - kk)] - sum;
                if (value <= 0.0) {
                    std::fprintf(stderr, "Matrix is not positive definite at %zu\n", j);
                    cudaFree(dPanel); cudaFree(dA);
                    return false;
                }
                panel[j * kb + (j - kk)] = std::sqrt(value);

                #pragma omp parallel for schedule(static)
                for (long long ii = static_cast<long long>(j + 1);
                     ii < static_cast<long long>(n); ++ii) {
                    const size_t i = static_cast<size_t>(ii);
                    double dot = 0.0;
                    for (size_t p = kk; p < j; ++p)
                        dot += panel[i * kb + (p - kk)] * panel[j * kb + (p - kk)];
                    panel[i * kb + (j - kk)] =
                        (panel[i * kb + (j - kk)] - dot) /
                        panel[j * kb + (j - kk)];
                }
            }
        }
        MPI_Bcast(panel.data(), static_cast<int>(n * kb), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Install the completed panel locally before launching the GPU update.
        for (size_t i = 0; i < localRows; ++i)
            std::memcpy(localA.data() + i * n + kk,
                        panel.data() + (firstRow + i) * kb, kb * sizeof(double));
        cudaCheck(cudaMemcpy(dPanel, panel.data(), n * kb * sizeof(double),
                             cudaMemcpyHostToDevice), "cudaMemcpy(panel)");

        if (trail < n) {
            const dim3 block(16, 16);
            const dim3 grid(static_cast<unsigned>((n - trail + block.x - 1) / block.x),
                            static_cast<unsigned>((localRows + block.y - 1) / block.y));
            trailingUpdate<<<grid, block>>>(dA, localRows, n, firstRow,
                                            dPanel, kb, trail);
            cudaCheck(cudaGetLastError(), "trailingUpdate launch");
            cudaCheck(cudaDeviceSynchronize(), "trailingUpdate");
            cudaCheck(cudaMemcpy2D(localA.data() + trail, n * sizeof(double),
                                   dA + trail, n * sizeof(double),
                                   (n - trail) * sizeof(double), localRows,
                                   cudaMemcpyDeviceToHost), "cudaMemcpy2D(A)");
        }
    }
    // Preserve the original routine's lower-triangular output contract.
    for (size_t i = 0; i < localRows; ++i)
        for (size_t j = firstRow + i + 1; j < n; ++j)
            localA[i * n + j] = 0.0;
    cudaCheck(cudaFree(dPanel), "cudaFree(panel)");
    cudaCheck(cudaFree(dA), "cudaFree(A)");
    return true;
}
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    #pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < static_cast<long long>(n); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
        A[i * n + i] += static_cast<double>(n);
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A, size_t n) {
    double maxError = 0.0, relError = 0.0;
    #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
    for (long long ii = 0; ii < static_cast<long long>(n * n); ++ii) {
        const size_t index = static_cast<size_t>(ii), i = index / n, j = index % n;
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) sum += L[i * n + k] * L[j * n + k];
        const double error = std::abs(sum - A[index]);
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (std::abs(A[index]) + 1e-10));
    }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
    return relError <= 1e-6;
}

void printUsage(const char* p) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num> Matrix size (default: 512)\n  -v       Enable validation\n  -r       Print results\n  -h       Show help\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 1; }
    }
    size_t firstRow = 0, localRows = 0;
    rowPartition(n, ranks, rank, firstRow, localRows);
    std::vector<double> original, full;
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
        full.resize(n * n); generatePositiveDefiniteMatrix(full, n);
        if (validate) original = full;
    }
    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) { size_t f=0, rows=0; rowPartition(n,ranks,r,f,rows); counts[r]=static_cast<int>(rows*n); displs[r]=static_cast<int>(f*n); }
    std::vector<double> localA(localRows * n);
    MPI_Scatterv(full.data(), counts.data(), displs.data(), MPI_DOUBLE, localA.data(),
                 static_cast<int>(localRows*n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) std::printf("Computing hybrid MPI/OpenMP/CUDA Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD); const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(localA, n, firstRow, localRows, rank, ranks);
    MPI_Barrier(MPI_COMM_WORLD); double elapsed = MPI_Wtime() - start, maximum = 0.0;
    MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) { MPI_Finalize(); return 1; }
    MPI_Gatherv(localA.data(), static_cast<int>(localRows*n), MPI_DOUBLE, full.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long ms = static_cast<long>(maximum * 1000.0);
        std::printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n", ms, (n*n*n/3.0) / std::max(maximum, 1e-12) / 1e9);
        if (printResults) print_results(full, "CholeskyL");
        if (validate) { std::printf("Validating result...\n"); const bool ok = validateCholesky(full, original, n); std::printf("Validation: %s\n", ok ? "PASSED" : "FAILED"); MPI_Finalize(); return ok ? 0 : 1; }
    }
    MPI_Finalize(); return 0;
}
