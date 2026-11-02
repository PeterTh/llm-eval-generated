#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

static int rank_id;

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank_id, __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define BLAS_CHECK(call) do { cublasStatus_t e = (call); if (e != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "Rank %d: cuBLAS error %d at %s:%d\n", rank_id, int(e), __FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define SOLVER_CHECK(call) do { cusolverStatus_t e = (call); if (e != CUSOLVER_STATUS_SUCCESS) { \
    fprintf(stderr, "Rank %d: cuSOLVER error %d at %s:%d\n", rank_id, int(e), __FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// Local matrices are column-major, with block-cyclic global rows.
__global__ void add_diagonal(double* a, int ld, const int* global_rows, int rows, int n) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) a[r + size_t(global_rows[r]) * ld] += n;
}

// Pack a row interval for MPI. Optionally clear the unused upper triangle.
__global__ void pack_rows(const double* a, double* out, int ld, int local_offset,
                          int rows, int col, int cols, const int* global_rows, bool lower) {
    size_t p = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = size_t(rows) * cols;
    if (p < total) {
        int r = int(p / cols), c = int(p % cols);
        out[p] = lower && col + c > global_rows[local_offset + r] ? 0.0 : a[local_offset + r + size_t(col + c) * ld];
    }
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    int ranks; MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (!rank_id) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr; long value = strtol(argv[++i], &end, 10);
            if (!*argv[i] || *end || value < 0 || value > 46340) {
                if (!rank_id) fprintf(stderr, "Matrix size must be between 0 and 46340\n");
                MPI_Finalize(); return 1;
            }
            n = int(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank_id) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank_id) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    MPI_Comm local; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_id, MPI_INFO_NULL, &local);
    int local_rank; MPI_Comm_rank(local, &local_rank); MPI_Comm_free(&local);
    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank_id); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    cublasHandle_t blas; cusolverDnHandle_t solver;
    BLAS_CHECK(cublasCreate(&blas)); SOLVER_CHECK(cusolverDnCreate(&solver));

    // Distribute whole block rows cyclically to balance every trailing update.
    constexpr int block = 256;
    int blocks = (n + block - 1) / block;
    std::vector<std::vector<int>> global_rows(ranks);
    for (int tile = 0; tile < blocks; ++tile)
        for (int row = tile * block; row < std::min(n, (tile + 1) * block); ++row)
            global_rows[tile % ranks].push_back(row);
    int rows = int(global_rows[rank_id].size());
    std::vector<int> counts(ranks), displs(ranks);
    int total = 0;
    for (int r = 0; r < ranks; ++r) {
        counts[r] = int(global_rows[r].size()) * n;
        displs[r] = total; total += counts[r];
    }
    int* d_global_rows = nullptr;
    CUDA_CHECK(cudaMalloc(&d_global_rows, std::max(rows, 1) * sizeof(int)));
    if (rows) CUDA_CHECK(cudaMemcpy(d_global_rows, global_rows[rank_id].data(), rows * sizeof(int), cudaMemcpyHostToDevice));
    size_t local_size = size_t(rows) * n;
    double *a = nullptr, *diag = nullptr, *panel = nullptr, *packed = nullptr;
    CUDA_CHECK(cudaMalloc(&a, std::max<size_t>(local_size, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&diag, size_t(block) * block * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&panel, std::max<size_t>(size_t(n) * block, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&packed, std::max<size_t>(size_t(rows) * block, 1) * sizeof(double)));
    int *info; CUDA_CHECK(cudaMalloc(&info, sizeof(int)));
    int lwork = 0;
    SOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, block, diag, block, &lwork));
    double* work; CUDA_CHECK(cudaMalloc(&work, std::max(lwork, 1) * sizeof(double)));

    if (!rank_id) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n"); fflush(stdout);
    }
    // Preserve the original rand_r sequence on every rank; the generation GEMM is on the GPU.
    std::vector<double> b(size_t(n) * n);
    unsigned int seed = 42;
    for (double& x : b) x = rand_r(&seed) / double(RAND_MAX) - 0.5;
    double* db = nullptr;
    CUDA_CHECK(cudaMalloc(&db, std::max<size_t>(b.size(), 1) * sizeof(double)));
    if (!b.empty()) CUDA_CHECK(cudaMemcpy(db, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice));
    b.clear(); b.shrink_to_fit();
    const double one = 1.0, zero = 0.0, minus_one = -1.0;
    for (int tile = rank_id; tile < blocks; tile += ranks) {
        int global_start = tile * block, local_start = (tile / ranks) * block;
        int tile_rows = std::min(block, n - global_start);
        BLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, tile_rows, n, n, &one,
                               db + size_t(global_start) * n, n, db, n, &zero, a + local_start, rows));
    }
    if (rows) {
        add_diagonal<<<(rows + 255) / 256, 256>>>(a, rows, d_global_rows, rows, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaFree(db));
    std::vector<double> original;
    if (validate) {
        original.resize(local_size);
        if (local_size) CUDA_CHECK(cudaMemcpy(original.data(), a, local_size * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (!rank_id) { printf("Computing Cholesky decomposition...\n"); fflush(stdout); }
    std::vector<double> host_diag(size_t(block) * block), host_panel(size_t(n) * block);
    std::vector<double> local_panel(size_t(rows) * block), gathered_panel(size_t(n) * block);
    std::vector<int> pc(ranks), pd(ranks), offsets(ranks);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = true;
    for (int k = 0; k < n; k += block) {
        int width = std::min(block, n - k), after = k + width, remain = n - after;
        int owner = (k / block) % ranks;
        int diag_offset = (k / block / ranks) * block;
        int potrf_info = 0;
        if (rank_id == owner) {
            CUDA_CHECK(cudaMemcpy2D(diag, size_t(width) * sizeof(double), a + diag_offset + size_t(k)*rows,
                                    size_t(rows)*sizeof(double), size_t(width)*sizeof(double), width, cudaMemcpyDeviceToDevice));
            SOLVER_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, width, diag, width, work, lwork, info));
            CUDA_CHECK(cudaMemcpy(&potrf_info, info, sizeof(int), cudaMemcpyDeviceToHost));
            if (!potrf_info) {
                CUDA_CHECK(cudaMemcpy2D(a + diag_offset + size_t(k)*rows, size_t(rows)*sizeof(double), diag,
                                        size_t(width)*sizeof(double), size_t(width)*sizeof(double), width, cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(host_diag.data(), diag, size_t(width)*width*sizeof(double), cudaMemcpyDeviceToHost));
            }
        }
        MPI_Bcast(&potrf_info, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (potrf_info) {
            if (!rank_id) {
                if (potrf_info > 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", k + potrf_info - 1);
                else fprintf(stderr, "cuSOLVER potrf failed with argument %d\n", -potrf_info);
            }
            success = false; break;
        }
        if (!remain) break;
        MPI_Bcast(host_diag.data(), width*width, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (rank_id != owner) CUDA_CHECK(cudaMemcpy(diag, host_diag.data(), size_t(width)*width*sizeof(double), cudaMemcpyHostToDevice));
        int displacement = 0;
        for (int r = 0; r < ranks; ++r) {
            offsets[r] = int(std::lower_bound(global_rows[r].begin(), global_rows[r].end(), after) - global_rows[r].begin());
            pc[r] = (int(global_rows[r].size()) - offsets[r]) * width;
            pd[r] = displacement; displacement += pc[r];
        }
        int offset = offsets[rank_id], local_rows = rows - offset;
        if (local_rows) {
            BLAS_CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                   CUBLAS_DIAG_NON_UNIT, local_rows, width, &one, diag, width,
                                   a + offset + size_t(k)*rows, rows));
            size_t count = size_t(local_rows)*width;
            pack_rows<<<(count+255)/256, 256>>>(a, packed, rows, offset, local_rows, k, width, d_global_rows, false);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(local_panel.data(), packed, count*sizeof(double), cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(local_panel.data(), local_rows*width, MPI_DOUBLE, gathered_panel.data(), pc.data(), pd.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        // Restore global row order for the GEMM operand.
        #pragma omp parallel for schedule(static) if(remain > 1024)
        for (int i = 0; i < remain; ++i) {
            int global_row = after + i, tile = global_row / block;
            int r = tile % ranks, local_row = (tile / ranks) * block + global_row % block;
            std::memcpy(host_panel.data() + size_t(i)*width,
                        gathered_panel.data() + pd[r] + size_t(local_row-offsets[r])*width, size_t(width)*sizeof(double));
        }
        if (local_rows) {
            CUDA_CHECK(cudaMemcpy(panel, host_panel.data(), size_t(remain)*width*sizeof(double), cudaMemcpyHostToDevice));
            // Avoid updating the upper triangle in large matrices. Group several
            // local tiles to keep the GEMMs large enough for the GPU.
            int group = remain >= 4096 ? 1024 : local_rows;
            for (int i = 0; i < local_rows; i += group) {
                int group_rows = std::min(group, local_rows - i);
                int columns = remain >= 4096 ? global_rows[rank_id][offset+i+group_rows-1] + 1 - after : remain;
                BLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, group_rows, columns, width, &minus_one,
                                       a + offset + i + size_t(k)*rows, rows, panel, width, &one,
                                       a + offset + i + size_t(after)*rows, rows));
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, max_elapsed = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int result = success ? 0 : 1;
    if (success && !rank_id) {
        long ms = long(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        printf("Performance: %.3f GFLOPS\n", double(n)*n*n / 3.0 / max_elapsed / 1e9);
    }
    if (success && (validate || printResults)) {
        // Pack the distributed L in the original row-major layout.
        double* full_pack = nullptr;
        CUDA_CHECK(cudaMalloc(&full_pack, std::max<size_t>(local_size, 1)*sizeof(double)));
        if (local_size) {
            pack_rows<<<(local_size+255)/256, 256>>>(a, full_pack, rows, 0, rows, 0, n, d_global_rows, true);
            CUDA_CHECK(cudaGetLastError());
        }
        std::vector<double> local_l(local_size), gathered_l(size_t(n)*n), all_l(size_t(n)*n);
        if (local_size) CUDA_CHECK(cudaMemcpy(local_l.data(), full_pack, local_size*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(full_pack));
        MPI_Allgatherv(local_l.data(), int(local_size), MPI_DOUBLE, gathered_l.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        #pragma omp parallel for schedule(static) if(n > 512)
        for (int row = 0; row < n; ++row) {
            int tile = row / block, r = tile % ranks;
            int local_row = (tile / ranks) * block + row % block;
            std::memcpy(all_l.data() + size_t(row)*n,
                        gathered_l.data() + displs[r] + size_t(local_row)*n, size_t(n)*sizeof(double));
        }
        if (printResults && !rank_id) print_results(all_l, "CholeskyL");
        if (validate) {
            if (!rank_id) printf("Validating result...\n");
            double *d_l = nullptr, *reconstructed = nullptr;
            CUDA_CHECK(cudaMalloc(&d_l, std::max<size_t>(all_l.size(), 1)*sizeof(double)));
            CUDA_CHECK(cudaMalloc(&reconstructed, std::max<size_t>(local_size, 1)*sizeof(double)));
            if (!all_l.empty()) CUDA_CHECK(cudaMemcpy(d_l, all_l.data(), all_l.size()*sizeof(double), cudaMemcpyHostToDevice));
            for (int tile = rank_id; tile < blocks; tile += ranks) {
                int global_start = tile * block, local_start = (tile / ranks) * block;
                int tile_rows = std::min(block, n - global_start);
                BLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, tile_rows, n, n, &one,
                                       d_l + size_t(global_start)*n, n, d_l, n, &zero, reconstructed + local_start, rows));
            }
            std::vector<double> reconstructed_host(local_size);
            if (local_size) CUDA_CHECK(cudaMemcpy(reconstructed_host.data(), reconstructed, local_size*sizeof(double), cudaMemcpyDeviceToHost));
            double max_error = 0.0, rel_error = 0.0;
            #pragma omp parallel for reduction(max:max_error,rel_error) if(local_size > 4096)
            for (size_t i = 0; i < local_size; ++i) {
                double error = std::fabs(reconstructed_host[i] - original[i]);
                max_error = std::max(max_error, error);
                rel_error = std::max(rel_error, error / (std::fabs(original[i]) + 1e-10));
            }
            double errors[2] = {max_error, rel_error}, global_errors[2];
            MPI_Allreduce(errors, global_errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (!rank_id) {
                printf("Max absolute error: %.10e\nMax relative error: %.10e\n", global_errors[0], global_errors[1]);
                if (global_errors[1] > 1e-6) printf("Validation failed: relative error too large\n");
                printf("Validation: %s\n", global_errors[1] > 1e-6 ? "FAILED" : "PASSED");
            }
            if (global_errors[1] > 1e-6) result = 1;
            CUDA_CHECK(cudaFree(d_l)); CUDA_CHECK(cudaFree(reconstructed));
        }
    } else if (!success && !rank_id) printf("Cholesky decomposition failed\n");
    CUDA_CHECK(cudaFree(d_global_rows)); CUDA_CHECK(cudaFree(work)); CUDA_CHECK(cudaFree(info)); CUDA_CHECK(cudaFree(packed));
    CUDA_CHECK(cudaFree(panel)); CUDA_CHECK(cudaFree(diag)); CUDA_CHECK(cudaFree(a));
    SOLVER_CHECK(cusolverDnDestroy(solver)); BLAS_CHECK(cublasDestroy(blas));
    MPI_Finalize();
    return result;
}
