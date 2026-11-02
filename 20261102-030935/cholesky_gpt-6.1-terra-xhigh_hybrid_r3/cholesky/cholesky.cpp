#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {
constexpr int block_size = 256;
int rank_id = 0;

[[noreturn]] void fail(const char* what, int code) {
    fprintf(stderr, "Rank %d: %s (error %d)\n", rank_id, what, code);
    MPI_Abort(MPI_COMM_WORLD, code ? code : 1);
    std::abort();
}
#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "Rank %d: %s: %s\n", rank_id, #call, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define BLAS(call) do { cublasStatus_t e = (call); if (e != CUBLAS_STATUS_SUCCESS) fail(#call, int(e)); } while (0)
#define SOLVER(call) do { cusolverStatus_t e = (call); if (e != CUSOLVER_STATUS_SUCCESS) fail(#call, int(e)); } while (0)
#define MPI_CHECK(call) do { int e = (call); if (e != MPI_SUCCESS) fail(#call, e); } while (0)

template<typename T> T* device_alloc(size_t count) {
    T* p = nullptr;
    if (count) CUDA(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)));
    return p;
}

void broadcast_doubles(double* data, size_t count, int root) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        MPI_CHECK(MPI_Bcast(data, chunk, MPI_DOUBLE, root, MPI_COMM_WORLD));
        data += chunk;
        count -= chunk;
    }
}

// rand_r on glibc advances this LCG three times per output. Jumping to each
// row preserves the original serial random stream while allowing parallel fill.
#ifdef __GLIBC__
unsigned int row_seed(uint64_t steps) {
    uint32_t mul = 1103515245u, add = 12345u, acc_mul = 1, acc_add = 0;
    while (steps) {
        if (steps & 1) { acc_add = acc_add * mul + add; acc_mul *= mul; }
        add = add * (mul + 1u);
        mul *= mul;
        steps >>= 1;
    }
    return acc_mul * 42u + acc_add;
}
#endif

void generate_random(double* b, int n) {
#ifndef __GLIBC__
    std::vector<unsigned int> seeds(n);
    unsigned int seed = 42;
    for (int i = 0; i < n; ++i) {
        seeds[i] = seed;
        for (int j = 0; j < n; ++j) (void)rand_r(&seed);
    }
#endif
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
#ifdef __GLIBC__
        unsigned int seed = row_seed(uint64_t(3) * n * i);
#else
        unsigned int seed = seeds[i];
#endif
        for (int j = 0; j < n; ++j)
            b[size_t(i) * n + j] = rand_r(&seed) / double(RAND_MAX) - 0.5;
    }
}

__global__ void add_diagonal(double* a, int n, int first, int width) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < width) a[size_t(j) * n + first + j] += n;
}
// Pack the rows of the current panel belonging to this rank's trailing
// block columns. This lets one large GEMM update all local columns at once.
__global__ void pack_panel(const double* panel, double* packed, int n, int width,
                           int local_first, int cols, int rank, int ranks) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < size_t(cols) * width) {
        int col = idx % cols, k = idx / cols;
        int global = ((local_first + col / block_size) * ranks + rank) * block_size + col % block_size;
        packed[idx] = panel[size_t(k) * n + global];
    }
}
__global__ void clear_upper(double* a, int n, int first, int width) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = size_t(n) * width;
    if (idx < count) {
        int row = idx % n, col = first + idx / n;
        if (row < col) a[idx] = 0.0;
    }
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    int ranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI does not support MPI_THREAD_FUNNELED", 1);

    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            long value = strtol(argv[++i], &end, 10);
            if (*end || value < 0 || value > INT_MAX) {
                if (!rank_id) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                MPI_Finalize(); return 1;
            }
            n = int(value);
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
    if (size_t(n) > std::numeric_limits<size_t>::max() / sizeof(double) / std::max(n, 1)) fail("Matrix size overflow", 1);
    if (!rank_id) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_id, MPI_INFO_NULL, &local_comm));
    int local_rank = 0, device_count = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    CUDA(cudaGetDeviceCount(&device_count));
    if (!device_count) fail("No CUDA device available", 1);
    CUDA(cudaSetDevice(local_rank % device_count));
    MPI_Comm_free(&local_comm);
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    BLAS(cublasCreate(&blas));
    SOLVER(cusolverDnCreate(&solver));
    // Use full FP64 precision for the original double-precision benchmark.
    BLAS(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));

    const int blocks = n / block_size + (n % block_size != 0);
    size_t local_cols = 0;
    for (int b = rank_id; b < blocks; b += ranks) local_cols += std::min(block_size, n - b * block_size);
    double* a = device_alloc<double>(size_t(n) * local_cols);
    double* original = validate ? device_alloc<double>(size_t(n) * local_cols) : nullptr;
    const double one = 1.0, zero = 0.0, minus_one = -1.0;
    if (n && local_cols) {
        std::vector<double> host_b(size_t(n) * n);
        generate_random(host_b.data(), n);
        size_t free_bytes = 0, total_bytes = 0;
        CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
        const size_t full_bytes = size_t(n) * n * sizeof(double);
        if (full_bytes <= free_bytes - free_bytes / 4) {
            double* b = device_alloc<double>(size_t(n) * n);
            CUDA(cudaMemcpy(b, host_b.data(), full_bytes, cudaMemcpyHostToDevice));
            size_t local = 0;
            for (int ib = rank_id; ib < blocks; ib += ranks) {
                int first = ib * block_size, width = std::min(block_size, n - first);
                BLAS(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, width, n, &one, b, n,
                                 b + size_t(first) * n, n, &zero, a + local * n, n));
                add_diagonal<<<(width + 255) / 256, 256>>>(a + local * n, n, first, width);
                CUDA(cudaGetLastError());
                local += width;
            }
            CUDA(cudaFree(b));
        } else {
            // Keep only this rank's rows of B and stream the other rows in
            // chunks. The distributed matrix can then exceed one GPU's memory.
            double* local_b = device_alloc<double>(size_t(n) * local_cols);
            size_t local = 0;
            for (int ib = rank_id; ib < blocks; ib += ranks) {
                int first = ib * block_size, width = std::min(block_size, n - first);
                CUDA(cudaMemcpy(local_b + local * n, host_b.data() + size_t(first) * n,
                                size_t(n) * width * sizeof(double), cudaMemcpyHostToDevice));
                local += width;
            }
            CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
            size_t possible_rows = free_bytes / 2 / (size_t(n) * sizeof(double));
            int tile_rows = int(std::min(size_t(n), std::min(size_t(2048), possible_rows)));
            if (!tile_rows) fail("Insufficient GPU memory for matrix generation", 1);
            double* tile = device_alloc<double>(size_t(n) * tile_rows);
            for (int row = 0; row < n; row += tile_rows) {
                int height = std::min(tile_rows, n - row);
                CUDA(cudaMemcpy(tile, host_b.data() + size_t(row) * n,
                                size_t(n) * height * sizeof(double), cudaMemcpyHostToDevice));
                local = 0;
                for (int ib = rank_id; ib < blocks; ib += ranks) {
                    int width = std::min(block_size, n - ib * block_size);
                    BLAS(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, height, width, n, &one,
                                     tile, n, local_b + local * n, n, &zero, a + local * n + row, n));
                    local += width;
                }
            }
            local = 0;
            for (int ib = rank_id; ib < blocks; ib += ranks) {
                int first = ib * block_size, width = std::min(block_size, n - first);
                add_diagonal<<<(width + 255) / 256, 256>>>(a + local * n, n, first, width);
                CUDA(cudaGetLastError());
                local += width;
            }
            CUDA(cudaFree(tile)); CUDA(cudaFree(local_b));
        }
        if (validate) CUDA(cudaMemcpy(original, a, size_t(n) * local_cols * sizeof(double), cudaMemcpyDeviceToDevice));
    }
    if (!rank_id) printf("Computing Cholesky decomposition...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();
    int* info_d = device_alloc<int>(1);
    int work_size = 0;
    int factor_size = ranks == 1 ? n : std::min(n, block_size);
    if (factor_size) SOLVER(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, factor_size, a, n, &work_size));
    double* work = device_alloc<double>(work_size);
    int info = 0;
    if (ranks == 1 && n) {
        SOLVER(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, n, a, n, work, work_size, info_d));
        CUDA(cudaMemcpy(&info, info_d, sizeof(info), cudaMemcpyDeviceToHost));
    } else if (n) {
        double* panel_d = device_alloc<double>(size_t(n) * block_size);
        double* packed = device_alloc<double>(size_t(n) * block_size);
        double* panel_h = nullptr;
        CUDA(cudaMallocHost(reinterpret_cast<void**>(&panel_h), size_t(n) * block_size * sizeof(double)));
        for (int ib = 0; ib < blocks; ++ib) {
            int first = ib * block_size, width = std::min(block_size, n - first), owner = ib % ranks;
            double* own_panel = rank_id == owner ? a + size_t(ib / ranks) * block_size * n : nullptr;
            if (rank_id == owner) {
                SOLVER(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, width, own_panel + first, n, work, work_size, info_d));
                CUDA(cudaMemcpy(&info, info_d, sizeof(info), cudaMemcpyDeviceToHost));
                if (info > 0) info += first;
            }
            MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, owner, MPI_COMM_WORLD));
            if (info) break;
            if (rank_id == owner) {
                int remaining = n - first - width;
                if (remaining) BLAS(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                               CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, remaining, width,
                                               &one, own_panel + first, n, own_panel + first + width, n));
                CUDA(cudaMemcpy(panel_h, own_panel, size_t(n) * width * sizeof(double), cudaMemcpyDeviceToHost));
            }
            broadcast_doubles(panel_h, size_t(n) * width, owner);
            const double* panel = own_panel;
            if (rank_id != owner) {
                CUDA(cudaMemcpy(panel_d, panel_h, size_t(n) * width * sizeof(double), cudaMemcpyHostToDevice));
                panel = panel_d;
            }
            int local_first = ib / ranks + (rank_id <= owner ? 1 : 0);
            size_t first_col = size_t(local_first) * block_size;
            int trailing_cols = first_col < local_cols ? int(local_cols - first_col) : 0;
            int trailing_rows = n - first - width;
            if (trailing_cols && trailing_rows) {
                size_t count = size_t(trailing_cols) * width;
                pack_panel<<<(count + 255) / 256, 256>>>(panel, packed, n, width,
                                                          local_first, trailing_cols, rank_id, ranks);
                CUDA(cudaGetLastError());
                BLAS(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, trailing_rows, trailing_cols, width,
                                 &minus_one, panel + first + width, n, packed, trailing_cols,
                                 &one, a + first_col * n + first + width, n));
            }
        }
        CUDA(cudaDeviceSynchronize());
        CUDA(cudaFreeHost(panel_h)); CUDA(cudaFree(panel_d)); CUDA(cudaFree(packed));
    }
    CUDA(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    if (info) {
        if (!rank_id) printf("Error: Matrix is not positive definite at diagonal element %d\nCholesky decomposition failed\n", info - 1);
        MPI_Finalize(); return 1;
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (!rank_id) {
        printf("Computation time: %ld ms\n", long(duration.count()));
        double seconds = std::chrono::duration<double>(end - start).count();
        printf("Performance: %.3f GFLOPS\n", double(n) * n * n / 3.0 / seconds / 1e9);
    }
    CUDA(cudaFree(info_d)); CUDA(cudaFree(work));

    // Make the distributed factor explicitly lower triangular, as in the original.
    for (int ib = rank_id; ib < blocks; ib += ranks) {
        int first = ib * block_size, width = std::min(block_size, n - first);
        size_t count = size_t(n) * width;
        clear_upper<<<(count + 255) / 256, 256>>>(a + size_t(ib / ranks) * block_size * n, n, first, width);
        CUDA(cudaGetLastError());
    }

    if (validate || printResults) {
        // Gather factor block columns. Validation needs the factor on each GPU;
        // printing needs only the root's row-major copy.
        double* full_l = validate ? device_alloc<double>(size_t(n) * n) : nullptr;
        std::vector<double> result;
        if (printResults && !rank_id) result.resize(size_t(n) * n);
        std::vector<double> panel(size_t(n) * std::min(n, block_size));
        for (int ib = 0; ib < blocks; ++ib) {
            int first = ib * block_size, width = std::min(block_size, n - first), owner = ib % ranks;
            if (rank_id == owner) CUDA(cudaMemcpy(panel.data(), a + size_t(ib / ranks) * block_size * n,
                                                   size_t(n) * width * sizeof(double), cudaMemcpyDeviceToHost));
            broadcast_doubles(panel.data(), size_t(n) * width, owner);
            if (validate) CUDA(cudaMemcpy(full_l + size_t(first) * n, panel.data(), size_t(n) * width * sizeof(double), cudaMemcpyHostToDevice));
            if (printResults && !rank_id) {
#pragma omp parallel for schedule(static)
                for (int row = 0; row < n; ++row)
                    for (int col = 0; col < width; ++col)
                        result[size_t(row) * n + first + col] = panel[size_t(col) * n + row];
            }
        }
        if (printResults && !rank_id) print_results(result, "CholeskyL");
        if (validate) {
            if (!rank_id) printf("Validating result...\n");
            double max_error = 0.0, rel_error = 0.0;
            double* reconstructed = device_alloc<double>(size_t(n) * std::min(n, block_size));
            std::vector<double> got(size_t(n) * std::min(n, block_size)), reference(got.size());
            for (int ib = rank_id; ib < blocks; ib += ranks) {
                int first = ib * block_size, width = std::min(block_size, n - first);
                BLAS(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, n, width, n, &one,
                                 full_l, n, full_l + first, n, &zero, reconstructed, n));
                size_t count = size_t(n) * width;
                CUDA(cudaMemcpy(got.data(), reconstructed, count * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA(cudaMemcpy(reference.data(), original + size_t(ib / ranks) * block_size * n,
                                count * sizeof(double), cudaMemcpyDeviceToHost));
#pragma omp parallel for reduction(max:max_error,rel_error) schedule(static)
                for (size_t i = 0; i < count; ++i) {
                    double error = fabs(got[i] - reference[i]);
                    max_error = std::max(max_error, error);
                    rel_error = std::max(rel_error, error / (fabs(reference[i]) + 1e-10));
                }
            }
            double local_errors[2] = {max_error, rel_error}, errors[2];
            MPI_CHECK(MPI_Allreduce(local_errors, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
            if (!rank_id) {
                printf("Max absolute error: %.10e\nMax relative error: %.10e\n", errors[0], errors[1]);
                if (errors[1] > 1e-6) printf("Validation failed: relative error too large\n");
                printf("Validation: %s\n", errors[1] > 1e-6 ? "FAILED" : "PASSED");
            }
            CUDA(cudaFree(reconstructed)); CUDA(cudaFree(full_l));
            if (errors[1] > 1e-6) { MPI_Finalize(); return 1; }
        }
    }
    CUDA(cudaFree(a)); CUDA(cudaFree(original));
    BLAS(cublasDestroy(blas)); SOLVER(cusolverDnDestroy(solver));
    MPI_Finalize();
    return 0;
}
