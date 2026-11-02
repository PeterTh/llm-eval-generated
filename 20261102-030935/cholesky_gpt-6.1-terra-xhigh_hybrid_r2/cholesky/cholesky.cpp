#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// Block columns are assigned cyclically to ranks. Each local column has the
// global leading dimension, so the diagonal tile and its panel are contiguous
// BLAS matrices even on the rank that owns them.
constexpr int block_size = 256;
int rank_id = 0;

[[noreturn]] void fail(const char* what, int code) {
    std::fprintf(stderr, "Rank %d: %s (error %d)\n", rank_id, what, code);
    MPI_Abort(MPI_COMM_WORLD, code ? code : 1);
    std::abort();
}
#define CUDA(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) fail(cudaGetErrorString(e_), static_cast<int>(e_)); } while (0)
#define BLAS(call) do { cublasStatus_t e_ = (call); if (e_ != CUBLAS_STATUS_SUCCESS) fail(#call, static_cast<int>(e_)); } while (0)
#define SOLVER(call) do { cusolverStatus_t e_ = (call); if (e_ != CUSOLVER_STATUS_SUCCESS) fail(#call, static_cast<int>(e_)); } while (0)
#define MPI_CHECK(call) do { int e_ = (call); if (e_ != MPI_SUCCESS) fail(#call, e_); } while (0)

void broadcast_doubles(double* data, size_t count, int root) {
    while (count) {
        int part = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_CHECK(MPI_Bcast(data, part, MPI_DOUBLE, root, MPI_COMM_WORLD));
        data += part;
        count -= part;
    }
}

// MPI counts are int, so transfer large blocks in chunks.
void transfer_doubles(double* data, size_t count, int peer, bool send) {
    while (count) {
        int part = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_CHECK(MPI_Send(data, part, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD));
        else MPI_CHECK(MPI_Recv(data, part, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        data += part;
        count -= part;
    }
}

std::vector<double> gather_matrix(const double* local, int n, int ranks, bool triangular) {
    std::vector<double> matrix;
    std::vector<double> block(static_cast<size_t>(n) * std::min(n, block_size));
    if (rank_id == 0) matrix.resize(static_cast<size_t>(n) * n);
    for (int j = 0, tile = 0; j < n; j += block_size, ++tile) {
        const int width = std::min(block_size, n - j);
        const int owner = tile % ranks;
        const size_t count = static_cast<size_t>(n) * width;
        if (rank_id == owner) {
            const size_t offset = static_cast<size_t>(tile / ranks) * block_size * n;
            CUDA(cudaMemcpy(block.data(), local + offset, count * sizeof(double), cudaMemcpyDeviceToHost));
            if (owner != 0) transfer_doubles(block.data(), count, 0, true);
        }
        if (rank_id == 0) {
            if (owner != 0) transfer_doubles(block.data(), count, owner, false);
            #pragma omp parallel for schedule(static)
            for (int col = 0; col < width; ++col) {
                for (int row = 0; row < n; ++row) {
                    matrix[static_cast<size_t>(row) * n + j + col] =
                        triangular && row < j + col ? 0.0 : block[static_cast<size_t>(col) * n + row];
                }
            }
        }
    }
    return matrix;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    int ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI does not provide MPI_THREAD_FUNNELED", 1);

    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            long value = std::strtol(argv[++i], &end, 10);
            if (*end || end == argv[i] || value < 0 || value > INT_MAX - block_size) {
                if (rank_id == 0) std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
            n = static_cast<int>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank_id == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank_id == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (rank_id == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
    }

    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_id, MPI_INFO_NULL, &local_comm));
    int local_rank = 0, devices = 0;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    CUDA(cudaGetDeviceCount(&devices));
    if (devices == 0) fail("No CUDA device available", 1);
    CUDA(cudaSetDevice(local_rank % devices));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    BLAS(cublasCreate(&blas));
    SOLVER(cusolverDnCreate(&solver));

    const size_t elements = static_cast<size_t>(n) * n;
    int local_columns = 0;
    for (int j = 0, tile = 0; j < n; j += block_size, ++tile)
        if (tile % ranks == rank_id) local_columns += std::min(block_size, n - j);
    double *dB = nullptr, *dBlocal = nullptr, *dA = nullptr, *dpanel = nullptr, *dwork = nullptr, *dones = nullptr;
    int* dinfo = nullptr;
    const size_t local_elements = static_cast<size_t>(n) * local_columns;
    // Stream B in groups of whole block columns. The first pass retains only
    // this rank's columns on its GPU; the second forms all rows of its A.
    // Replaying the same random sequence avoids a replicated n-by-n B.
    const int batch_columns = 4 * block_size;
    const int batch_width = std::min(n, batch_columns);
    std::vector<double> batch(static_cast<size_t>(n) * batch_width);
    if (local_elements) {
        CUDA(cudaMalloc(&dA, local_elements * sizeof(double)));
        CUDA(cudaMalloc(&dBlocal, local_elements * sizeof(double)));
        CUDA(cudaMalloc(&dB, static_cast<size_t>(n) * batch_width * sizeof(double)));
    }
    if (n) {
        CUDA(cudaMalloc(&dpanel, static_cast<size_t>(n) * std::min(n, block_size) * sizeof(double)));
        CUDA(cudaMalloc(&dones, static_cast<size_t>(std::min(n, block_size)) * sizeof(double)));
        std::vector<double> ones(std::min(n, block_size), 1.0);
        CUDA(cudaMemcpy(dones, ones.data(), ones.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA(cudaMalloc(&dinfo, sizeof(int)));
    }
    const double one = 1.0, zero = 0.0, minus_one = -1.0, diagonal = static_cast<double>(n);
    for (int pass = 0; pass < 2; ++pass) {
        unsigned int seed = 42;
        for (int first = 0; first < n; first += batch_columns) {
            const int width = std::min(batch_columns, n - first);
            const size_t count = static_cast<size_t>(n) * width;
            if (rank_id == 0) {
                for (size_t i = 0; i < count; ++i) batch[i] = rand_r(&seed);
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < count; ++i) batch[i] = batch[i] / static_cast<double>(RAND_MAX) - 0.5;
            }
            broadcast_doubles(batch.data(), count, 0);
            if (!local_columns) continue;
            if (pass == 0) {
                for (int offset = 0; offset < width; offset += block_size) {
                    int tile = (first + offset) / block_size;
                    if (tile % ranks != rank_id) continue;
                    int w = std::min(block_size, width - offset);
                    CUDA(cudaMemcpy(dBlocal + static_cast<size_t>(tile / ranks) * block_size * n,
                                    batch.data() + static_cast<size_t>(offset) * n,
                                    static_cast<size_t>(w) * n * sizeof(double), cudaMemcpyHostToDevice));
                }
            } else {
                CUDA(cudaMemcpy(dB, batch.data(), count * sizeof(double), cudaMemcpyHostToDevice));
                // The row-major random B is column-major B^T in BLAS.
                BLAS(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, width, local_columns, n, &one,
                                 dB, n, dBlocal, n, &zero, dA + first, n));
            }
        }
    }
    for (int j = 0, tile = 0; j < n; j += block_size, ++tile) if (tile % ranks == rank_id) {
        int width = std::min(block_size, n - j);
        double* col = dA + static_cast<size_t>(tile / ranks) * block_size * n;
        BLAS(cublasDaxpy(blas, width, &diagonal, dones, 1, col + j, n + 1));
    }
    CUDA(cudaFree(dB));
    CUDA(cudaFree(dBlocal));
    CUDA(cudaFree(dones));
    std::vector<double>().swap(batch);
    std::vector<double> original;
    if (validate) original = gather_matrix(dA, n, ranks, false);

    int lwork = 0;
    if (n && local_columns) {
        SOLVER(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, std::min(n, block_size), dA, n, &lwork));
        CUDA(cudaMalloc(&dwork, static_cast<size_t>(lwork) * sizeof(double)));
    }
    std::vector<double> panel(static_cast<size_t>(n) * std::min(n, block_size));
    if (rank_id == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    bool success = true;
    for (int k = 0, tile = 0; k < n; k += block_size, ++tile) {
        const int width = std::min(block_size, n - k), rows = n - k, owner = tile % ranks;
        int info = 0;
        if (rank_id == owner) {
            double* col = dA + static_cast<size_t>(tile / ranks) * block_size * n + k;
            SOLVER(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, width, col, n, dwork, lwork, dinfo));
            CUDA(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (info == 0) {
                if (rows > width)
                    BLAS(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                     CUBLAS_DIAG_NON_UNIT, rows - width, width, &one, col, n, col + width, n));
                CUDA(cudaMemcpy2D(panel.data(), static_cast<size_t>(rows) * sizeof(double), col,
                                  static_cast<size_t>(n) * sizeof(double), static_cast<size_t>(rows) * sizeof(double),
                                  width, cudaMemcpyDeviceToHost));
            }
        }
        MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, owner, MPI_COMM_WORLD));
        if (info != 0) {
            if (rank_id == 0) std::printf("Error: Matrix is not positive definite at diagonal element %d\n", k + info - 1);
            success = false;
            break;
        }
        broadcast_doubles(panel.data(), static_cast<size_t>(rows) * width, owner);
        if (k + width == n) continue;
        CUDA(cudaMemcpy(dpanel, panel.data(), static_cast<size_t>(rows) * width * sizeof(double), cudaMemcpyHostToDevice));
        for (int j = k + width, t = tile + 1; j < n; j += block_size, ++t) if (t % ranks == rank_id) {
            const int w = std::min(block_size, n - j), remaining = n - j;
            const double* p = dpanel + (j - k);
            double* c = dA + static_cast<size_t>(t / ranks) * block_size * n + j;
            BLAS(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, remaining, w, width, &minus_one,
                             p, rows, p, rows, &one, c, n));
        }
    }
    CUDA(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, max_elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!success) {
        if (rank_id == 0) std::printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    if (rank_id == 0) {
        std::printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000));
        std::printf("Performance: %.3f GFLOPS\n", static_cast<double>(n) * n * n / (3.0 * max_elapsed * 1e9));
    }
    std::vector<double> result;
    if (validate || printResults) result = gather_matrix(dA, n, ranks, true);
    int valid = 1;
    if (rank_id == 0 && printResults) print_results(result, "CholeskyL");
    if (rank_id == 0 && validate) {
        std::printf("Validating result...\n");
        std::vector<double> reconstructed(elements);
        if (n) {
            double *dL = nullptr, *dR = nullptr;
            CUDA(cudaMalloc(&dL, elements * sizeof(double)));
            CUDA(cudaMalloc(&dR, elements * sizeof(double)));
            CUDA(cudaMemcpy(dL, result.data(), elements * sizeof(double), cudaMemcpyHostToDevice));
            BLAS(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, dL, n, dL, n, &zero, dR, n));
            CUDA(cudaMemcpy(reconstructed.data(), dR, elements * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA(cudaFree(dL));
            CUDA(cudaFree(dR));
        }
        double maxError = 0.0, relError = 0.0;
        #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
        for (size_t i = 0; i < elements; ++i) {
            double error = std::fabs(reconstructed[i] - original[i]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::fabs(original[i]) + 1e-10));
        }
        std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
        valid = relError <= 1e-6;
        if (!valid) std::printf("Validation failed: relative error too large\n");
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CUDA(cudaFree(dA));
    CUDA(cudaFree(dpanel));
    CUDA(cudaFree(dwork));
    CUDA(cudaFree(dinfo));
    SOLVER(cusolverDnDestroy(solver));
    BLAS(cublasDestroy(blas));
    MPI_Finalize();
    return valid ? 0 : 1;
}
