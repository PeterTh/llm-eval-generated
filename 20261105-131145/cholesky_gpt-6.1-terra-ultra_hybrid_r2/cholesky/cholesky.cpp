#include <algorithm>
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

// MPI is called only by the main thread. A rank owns a block-cyclic set of
// columns of the upper factor, stored consecutively on its GPU.
static int rank_id;
#define CHECK(call) do { auto status_ = (call); if (status_ != 0) { \
    fprintf(stderr, "Rank %d: %s failed (%d) at %s:%d\n", rank_id, #call, \
            static_cast<int>(status_), __FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, 1); \
} } while (0)

static void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static double* device_alloc(size_t count) {
    double* p = nullptr;
    if (count) CHECK(cudaMalloc(&p, count * sizeof(double)));
    return p;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    int ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (!rank_id) fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            long value = strtol(argv[++i], &end, 10);
            if (*end || value < 0 || value > INT_MAX / 256) {
                if (!rank_id) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                MPI_Finalize(); return 1;
            }
            n = static_cast<int>(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank_id) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!rank_id) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (!rank_id) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_id, MPI_INFO_NULL, &local_comm);
    int local_rank = 0, devices = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA devices available\n", rank_id); MPI_Abort(MPI_COMM_WORLD, 1); }
    CHECK(cudaSetDevice(local_rank % devices));
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    CHECK(cublasCreate(&blas));
    CHECK(cusolverDnCreate(&solver));

    constexpr int block = 256;
    const int tiles = (n + block - 1) / block;
    std::vector<int> offsets(tiles), widths(tiles), local_cols(ranks, 0);
    for (int t = 0; t < tiles; ++t) {
        int owner = t % ranks;
        offsets[t] = local_cols[owner];
        widths[t] = std::min(block, n - t * block);
        local_cols[owner] += widths[t];
    }
    const int mycols = local_cols[rank_id];
    const size_t nn = static_cast<size_t>(n) * n;
    const size_t local_size = static_cast<size_t>(n) * mycols;
    // rand_r is intentionally serial: it reproduces the original random stream.
    std::vector<double> B(nn);
    unsigned int seed = 42;
    for (size_t i = 0; i < nn; ++i) B[i] = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
    double *dB = device_alloc(nn), *dA = device_alloc(local_size);
    if (nn) CHECK(cudaMemcpy(dB, B.data(), nn * sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double>().swap(B);
    const double one = 1.0, zero = 0.0, minus_one = -1.0;
    // Row-major B is column-major B^T. Generate owned columns of B B^T.
    for (int t = rank_id; t < tiles; t += ranks) {
        CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, widths[t], n,
                          &one, dB, n, dB + static_cast<size_t>(t * block) * n, n,
                          &zero, dA + static_cast<size_t>(offsets[t]) * n, n));
    }
    if (dB) CHECK(cudaFree(dB));
    // Add n to the diagonal with a strided BLAS operation.
    std::vector<double> diag_host(block, static_cast<double>(n));
    double* dadd = device_alloc(block);
    CHECK(cudaMemcpy(dadd, diag_host.data(), block * sizeof(double), cudaMemcpyHostToDevice));
    for (int t = rank_id; t < tiles; t += ranks)
        CHECK(cublasDaxpy(blas, widths[t], &one, dadd, 1,
                          dA + static_cast<size_t>(offsets[t]) * n + t * block, n + 1));
    CHECK(cudaFree(dadd));
    std::vector<double> original_local;
    if (validate) {
        original_local.resize(local_size);
        if (local_size) CHECK(cudaMemcpy(original_local.data(), dA, local_size * sizeof(double), cudaMemcpyDeviceToHost));
    }

    double* ddiag = device_alloc(static_cast<size_t>(block) * block);
    double* dpanel = device_alloc(static_cast<size_t>(block) * n);
    int *dinfo = nullptr;
    CHECK(cudaMalloc(&dinfo, sizeof(int)));
    int lwork = 0;
    if (n && mycols) CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER,
                                    std::min(block, n), dA, n, &lwork));
    double* work = device_alloc(lwork);
    std::vector<double> diagonal(static_cast<size_t>(block) * block);
    std::vector<double> send_panel(static_cast<size_t>(block) * mycols);
    std::vector<double> gathered(static_cast<size_t>(block) * n);
    std::vector<double> panel(static_cast<size_t>(block) * n);
    std::vector<int> counts(ranks), displs(ranks), first(ranks);

    if (!rank_id) printf("Computing Cholesky decomposition...\n");
    CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = true;
    for (int t = 0; t < tiles; ++t) {
        int k = t * block, b = widths[t], owner = t % ranks, info = 0;
        if (rank_id == owner) {
            double* akk = dA + static_cast<size_t>(offsets[t]) * n + k;
            CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, b, akk, n, work, lwork, dinfo));
            CHECK(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (!info) CHECK(cudaMemcpy2D(diagonal.data(), b * sizeof(double), akk, n * sizeof(double),
                                          b * sizeof(double), b, cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(&info, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (info) {
            if (!rank_id) printf("Error: Matrix is not positive definite at diagonal element %d\n", k + info - 1);
            success = false; break;
        }
        if (t + 1 == tiles) break;
        MPI_Bcast(diagonal.data(), b * b, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        CHECK(cudaMemcpy(ddiag, diagonal.data(), static_cast<size_t>(b) * b * sizeof(double), cudaMemcpyHostToDevice));
        for (int p = 0; p < ranks; ++p) {
            int next = t + 1 + (p - (t + 1) % ranks + ranks) % ranks;
            first[p] = next < tiles ? offsets[next] : local_cols[p];
            counts[p] = (local_cols[p] - first[p]) * b;
            displs[p] = p ? displs[p - 1] + counts[p - 1] : 0;
        }
        int trailing_local = mycols - first[rank_id];
        if (trailing_local) {
            double* local_panel = dA + static_cast<size_t>(first[rank_id]) * n + k;
            CHECK(cublasDtrsm(blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                              CUBLAS_DIAG_NON_UNIT, b, trailing_local, &one, ddiag, b, local_panel, n));
            CHECK(cudaMemcpy2D(send_panel.data(), b * sizeof(double), local_panel, n * sizeof(double),
                               b * sizeof(double), trailing_local, cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(send_panel.data(), counts[rank_id], MPI_DOUBLE, gathered.data(),
                       counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        // Unpack rank-ordered columns into one globally ordered panel.
        #pragma omp parallel for schedule(static) if(tiles - t > 8)
        for (int u = t + 1; u < tiles; ++u) {
            int p = u % ranks;
            const double* src = gathered.data() + displs[p] + static_cast<size_t>(offsets[u] - first[p]) * b;
            double* dst = panel.data() + static_cast<size_t>(u * block - k - b) * b;
            std::memcpy(dst, src, static_cast<size_t>(widths[u]) * b * sizeof(double));
        }
        int trailing = n - k - b;
        if (trailing_local && trailing) {
            CHECK(cudaMemcpy(dpanel, panel.data(), static_cast<size_t>(b) * trailing * sizeof(double), cudaMemcpyHostToDevice));
            // Only the upper triangle is needed in subsequent steps.
            int next = t + 1 + (rank_id - (t + 1) % ranks + ranks) % ranks;
            for (int u = next; u < tiles; u += ranks) {
                int rows = u * block - k - b, width = widths[u];
                double* col = dA + static_cast<size_t>(offsets[u]) * n;
                if (rows) CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, width, b,
                                             &minus_one, dpanel, b, col + k, n,
                                             &one, col + k + b, n));
                CHECK(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, width, b,
                                  &minus_one, col + k, n, &one, col + u * block, n));
            }
        }
    }
    CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, max_elapsed = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (!rank_id) printf("Cholesky decomposition failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (!rank_id) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000));
        printf("Performance: %.3f GFLOPS\n", (static_cast<double>(n) * n * n / 3.0) / max_elapsed / 1e9);
    }

    // Gather complete columns only when the caller requests output or validation.
    std::vector<double> L, orig, local_result;
    if (printResults || validate) {
        local_result.resize(local_size);
        if (local_size) CHECK(cudaMemcpy(local_result.data(), dA, local_size * sizeof(double), cudaMemcpyDeviceToHost));
        if (!rank_id) {
            L.assign(nn, 0.0);
            if (validate) orig.resize(nn);
        }
        // Transfer one tile at a time so MPI's int counts never limit n*n.
        std::vector<double> tile_buffer(rank_id == 0 ? static_cast<size_t>(n) * std::min(block, n) : 0);
        auto collect = [&](const std::vector<double>& local, bool original) {
            for (int t = 0; t < tiles; ++t) {
                int owner = t % ranks;
                size_t count = static_cast<size_t>(n) * widths[t];
                const double* src = nullptr;
                if (rank_id == owner) {
                    src = local.data() + static_cast<size_t>(offsets[t]) * n;
                    if (owner != 0) MPI_Send(src, static_cast<int>(count), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
                if (!rank_id) {
                    if (owner != 0) {
                        MPI_Recv(tile_buffer.data(), static_cast<int>(count), MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        src = tile_buffer.data();
                    }
                    #pragma omp parallel for schedule(static) if(n > 2048)
                    for (int j = 0; j < widths[t]; ++j) {
                        int row = t * block + j;
                        if (original)
                            std::memcpy(orig.data() + static_cast<size_t>(row) * n,
                                        src + static_cast<size_t>(j) * n, n * sizeof(double));
                        else
                            for (int i = 0; i <= row; ++i)
                                L[static_cast<size_t>(row) * n + i] = src[static_cast<size_t>(j) * n + i];
                    }
                }
            }
        };
        collect(local_result, false);
        if (validate) collect(original_local, true);
    }
    // Release factorization buffers before allocating the validation product.
    if (work) CHECK(cudaFree(work));
    CHECK(cudaFree(dinfo));
    CHECK(cudaFree(ddiag));
    if (dpanel) CHECK(cudaFree(dpanel));
    if (dA) CHECK(cudaFree(dA));
    std::vector<double>().swap(local_result);
    std::vector<double>().swap(original_local);
    int valid = 1;
    if (!rank_id && printResults) print_results(L, "CholeskyL");
    if (!rank_id && validate) {
        printf("Validating result...\n");
        // Row-major L is column-major L^T; compute L L^T on the GPU.
        double* dl = device_alloc(nn);
        double* dc = device_alloc(nn);
        std::vector<double> reconstructed(nn);
        if (nn) {
            CHECK(cudaMemcpy(dl, L.data(), nn * sizeof(double), cudaMemcpyHostToDevice));
            CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, dl, n, dl, n, &zero, dc, n));
            CHECK(cudaMemcpy(reconstructed.data(), dc, nn * sizeof(double), cudaMemcpyDeviceToHost));
        }
        double maxError = 0, relError = 0;
        #pragma omp parallel for reduction(max:maxError,relError) schedule(static) if(nn > 65536)
        for (size_t i = 0; i < nn; ++i) {
            double error = std::fabs(reconstructed[i] - orig[i]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::fabs(orig[i]) + 1e-10));
        }
        printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
        valid = relError <= 1e-6;
        if (!valid) printf("Validation failed: relative error too large\n");
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        if (dl) CHECK(cudaFree(dl));
        if (dc) CHECK(cudaFree(dc));
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CHECK(cusolverDnDestroy(solver));
    CHECK(cublasDestroy(blas));
    MPI_Finalize();
    return valid ? 0 : 1;
}
