#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// Each rank owns a contiguous range of rows of the original row-major matrix.
// On the GPU these are columns of its transpose, stored in column-major order.
// The factorization therefore computes U = L^T using upper-triangular BLAS.
static int rank_id;

#define MPI_CHECK(call) do { int e_ = (call); if (e_ != MPI_SUCCESS) { \
    char msg_[MPI_MAX_ERROR_STRING]; int len_; MPI_Error_string(e_, msg_, &len_); \
    fprintf(stderr, "Rank %d: MPI error at %s:%d: %.*s\n", rank_id, __FILE__, __LINE__, len_, msg_); MPI_Abort(MPI_COMM_WORLD, e_); } } while (0)
#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank_id, __FILE__, __LINE__, cudaGetErrorString(e_)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define LIB_CHECK(call) do { auto e_ = (call); if (e_ != 0) { \
    fprintf(stderr, "Rank %d: CUDA library error %d at %s:%d\n", rank_id, static_cast<int>(e_), __FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

__global__ void add_diagonal(double* a, int n, int first, int count) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < count) a[static_cast<size_t>(j) * n + first + j] += n;
}

__global__ void clear_lower(double* a, int n, int first, int count) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(n) * count;
    if (i < total && static_cast<int>(i % n) > first + static_cast<int>(i / n)) a[i] = 0.0;
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank_id));
    int ranks;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) { if (!rank_id) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            long v = strtol(argv[++i], &end, 10);
            if (*end || v < 0 || v > 46340) {
                if (!rank_id) fprintf(stderr, "Matrix size must be between 0 and 46340\n");
                MPI_Finalize(); return 1;
            }
            n = static_cast<int>(v);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank_id) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank_id) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }

    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_id, MPI_INFO_NULL, &local_comm));
    int local_rank, devices;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank_id); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    LIB_CHECK(cublasCreate(&blas));
    LIB_CHECK(cusolverDnCreate(&solver));

    if (!rank_id) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n"); fflush(stdout);
    }

    // Partition on panel boundaries so every diagonal panel has a single owner.
    constexpr int panel = 256;
    int blocks = (n + panel - 1) / panel;
    std::vector<int> first(ranks + 1), counts(ranks), displs(ranks);
    for (int r = 0; r <= ranks; ++r)
        first[r] = std::min(n, static_cast<int>((static_cast<long long>(blocks) * r / ranks) * panel));
    for (int r = 0; r < ranks; ++r) {
        counts[r] = (first[r + 1] - first[r]) * n;
        displs[r] = first[r] * n;
    }
    const int local_cols = first[rank_id + 1] - first[rank_id];
    const size_t local_size = static_cast<size_t>(n) * local_cols;
    const size_t matrix_size = static_cast<size_t>(n) * n;
    std::vector<double> B(matrix_size);
    if (!rank_id) {
        // Save RNG states at chunk boundaries to reproduce the original serial
        // rand_r sequence while filling independent chunks with OpenMP.
        constexpr size_t chunk = 16384;
        const size_t chunks = (matrix_size + chunk - 1) / chunk;
        std::vector<unsigned int> seeds(chunks);
        unsigned int seed = 42;
        for (size_t c = 0; c < chunks; ++c) {
            seeds[c] = seed;
            if (c + 1 < chunks)
                for (size_t i = 0; i < chunk; ++i) (void)rand_r(&seed);
        }
        #pragma omp parallel for schedule(static)
        for (size_t c = 0; c < chunks; ++c) {
            unsigned int state = seeds[c];
            for (size_t i = c * chunk; i < std::min(matrix_size, (c + 1) * chunk); ++i)
                B[i] = rand_r(&state) / static_cast<double>(RAND_MAX) - 0.5;
        }
    }
    MPI_CHECK(MPI_Bcast(B.data(), static_cast<int>(matrix_size), MPI_DOUBLE, 0, MPI_COMM_WORLD));

    double *d_b = nullptr, *d_a = nullptr, *d_diag = nullptr, *d_panel = nullptr, *d_work = nullptr;
    int *d_info = nullptr, work_size = 0;
    // Allocate at least one element to handle empty matrices and ranks.
    CUDA_CHECK(cudaMalloc(&d_b, std::max<size_t>(1, matrix_size) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_a, std::max<size_t>(1, local_size) * sizeof(double)));
    if (matrix_size) CUDA_CHECK(cudaMemcpy(d_b, B.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double>().swap(B);
    const double one = 1.0, zero = 0.0, minus_one = -1.0;
    if (local_cols) {
        LIB_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, local_cols, n, &one,
                             d_b, n, d_b + static_cast<size_t>(first[rank_id]) * n, n,
                             &zero, d_a, n));
        add_diagonal<<<(local_cols + 255) / 256, 256>>>(d_a, n, first[rank_id], local_cols);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaFree(d_b));
    std::vector<double> original(validate ? local_size : 0);
    if (validate && local_size) CUDA_CHECK(cudaMemcpy(original.data(), d_a, local_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMalloc(&d_diag, panel * panel * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_panel, std::max<size_t>(1, static_cast<size_t>(panel) * n) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    // Pinned staging buffers make panel transfers suitable for large messages.
    double *diag = nullptr, *panel_host = nullptr;
    CUDA_CHECK(cudaMallocHost(&diag, panel * panel * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&panel_host, std::max<size_t>(1, static_cast<size_t>(panel) * n) * sizeof(double)));
    std::vector<int> pc(ranks), pd(ranks);

    if (!rank_id) { printf("Computing Cholesky decomposition...\n"); fflush(stdout); }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();
    bool success = true;
    for (int k = 0; k < n; k += panel) {
        int b = std::min(panel, n - k), next = k + b;
        int owner = static_cast<int>(std::upper_bound(first.begin(), first.end(), k) - first.begin()) - 1;
        int info = 0;
        if (rank_id == owner) {
            double* tile = d_a + static_cast<size_t>(k - first[rank_id]) * n + k;
            int needed = 0;
            LIB_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, b, tile, n, &needed));
            if (needed > work_size) {
                if (d_work) CUDA_CHECK(cudaFree(d_work));
                CUDA_CHECK(cudaMalloc(&d_work, static_cast<size_t>(needed) * sizeof(double)));
                work_size = needed;
            }
            LIB_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, b, tile, n, d_work, work_size, d_info));
            CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
            if (!info) CUDA_CHECK(cudaMemcpy2D(diag, b * sizeof(double), tile, n * sizeof(double), b * sizeof(double), b, cudaMemcpyDeviceToHost));
        }
        MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, owner, MPI_COMM_WORLD));
        if (info) {
            if (!rank_id) printf("Error: Matrix is not positive definite at diagonal element %d\n", k + info - 1);
            success = false; break;
        }
        if (next == n) break;
        MPI_CHECK(MPI_Bcast(diag, b * b, MPI_DOUBLE, owner, MPI_COMM_WORLD));
        CUDA_CHECK(cudaMemcpy(d_diag, diag, static_cast<size_t>(b) * b * sizeof(double), cudaMemcpyHostToDevice));
        int local_start = std::max(next, first[rank_id]);
        int local_tail = std::max(0, first[rank_id + 1] - local_start);
        double* local_panel = local_tail ? d_a + static_cast<size_t>(local_start - first[rank_id]) * n + k : nullptr;
        if (local_tail) {
            LIB_CHECK(cublasDtrsm(blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                  CUBLAS_DIAG_NON_UNIT, b, local_tail, &one, d_diag, b, local_panel, n));
        }
        if (ranks == 1) {
            // The entire solved panel is already on this device.
            LIB_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n - next, n - next, b, &minus_one,
                                  local_panel, n, local_panel, n, &one,
                                  d_a + static_cast<size_t>(next) * n + next, n));
        } else {
            for (int r = 0; r < ranks; ++r) {
                int begin = std::max(next, first[r]);
                pc[r] = b * std::max(0, first[r + 1] - begin);
                pd[r] = b * (std::min(n, begin) - next);
            }
            if (local_tail)
                CUDA_CHECK(cudaMemcpy2D(panel_host + pd[rank_id], b * sizeof(double), local_panel, n * sizeof(double),
                                        b * sizeof(double), local_tail, cudaMemcpyDeviceToHost));
            MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, panel_host, pc.data(), pd.data(), MPI_DOUBLE, MPI_COMM_WORLD));
            if (local_tail) {
                CUDA_CHECK(cudaMemcpy(d_panel, panel_host, static_cast<size_t>(b) * (n - next) * sizeof(double), cudaMemcpyHostToDevice));
                LIB_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n - next, local_tail, b, &minus_one,
                                      d_panel, b, d_panel + static_cast<size_t>(local_start - next) * b, b,
                                      &one, d_a + static_cast<size_t>(local_start - first[rank_id]) * n + next, n));
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    if (!success) { if (!rank_id) printf("Cholesky decomposition failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (!rank_id) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double seconds = std::chrono::duration<double>(end - start).count();
        printf("Computation time: %ld ms\n", static_cast<long>(ms));
        printf("Performance: %.3f GFLOPS\n", static_cast<double>(n) * n * n / 3.0 / seconds / 1e9);
    }

    if (local_size) {
        clear_lower<<<static_cast<unsigned int>((local_size + 255) / 256), 256>>>(d_a, n, first[rank_id], local_cols);
        CUDA_CHECK(cudaGetLastError());
    }
    std::vector<double> local((validate || printResults) ? local_size : 0);
    if (!local.empty()) CUDA_CHECK(cudaMemcpy(local.data(), d_a, local_size * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> full((validate || (printResults && !rank_id)) ? matrix_size : 0);
    if (validate) MPI_CHECK(MPI_Allgatherv(local.data(), static_cast<int>(local_size), MPI_DOUBLE, full.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD));
    else if (printResults) MPI_CHECK(MPI_Gatherv(local.data(), static_cast<int>(local_size), MPI_DOUBLE, full.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
    if (printResults && !rank_id) print_results(full, "CholeskyL");

    bool valid = true;
    if (validate) {
        if (!rank_id) printf("Validating result...\n");
        double max_error = 0.0, rel_error = 0.0;
        if (local_size) {
            double *d_full, *d_reconstructed;
            CUDA_CHECK(cudaMalloc(&d_full, matrix_size * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_reconstructed, local_size * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_full, full.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
            LIB_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, local_cols, n, &one,
                                  d_full, n, d_a, n, &zero, d_reconstructed, n));
            std::vector<double> reconstructed(local_size);
            CUDA_CHECK(cudaMemcpy(reconstructed.data(), d_reconstructed, local_size * sizeof(double), cudaMemcpyDeviceToHost));
            #pragma omp parallel for reduction(max:max_error,rel_error) schedule(static)
            for (size_t i = 0; i < local_size; ++i) {
                double error = std::abs(reconstructed[i] - original[i]);
                max_error = std::max(max_error, error);
                rel_error = std::max(rel_error, error / (std::abs(original[i]) + 1e-10));
            }
            CUDA_CHECK(cudaFree(d_reconstructed)); CUDA_CHECK(cudaFree(d_full));
        }
        double errors[2] = {max_error, rel_error}, global_errors[2] = {};
        MPI_CHECK(MPI_Allreduce(errors, global_errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
        valid = global_errors[1] <= 1e-6;
        if (!rank_id) {
            printf("Max absolute error: %.10e\nMax relative error: %.10e\n", global_errors[0], global_errors[1]);
            if (!valid) printf("Validation failed: relative error too large\n");
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    CUDA_CHECK(cudaFreeHost(panel_host)); CUDA_CHECK(cudaFreeHost(diag));
    if (d_work) CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info)); CUDA_CHECK(cudaFree(d_panel)); CUDA_CHECK(cudaFree(d_diag)); CUDA_CHECK(cudaFree(d_a));
    LIB_CHECK(cusolverDnDestroy(solver)); LIB_CHECK(cublasDestroy(blas));
    MPI_CHECK(MPI_Finalize());
    return valid ? 0 : 1;
}
