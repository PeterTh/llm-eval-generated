#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr int kBlockSize = 128;
constexpr int kTileSize = 16;

__global__ void gemm_update_kernel(int m, int n, int k, const double* A, const double* B, double* C,
                                   int lda, int ldb, int ldc) {
    const int row = blockIdx.y * kTileSize + threadIdx.y;
    const int col = blockIdx.x * kTileSize + threadIdx.x;
    if (row < m && col < n) {
        double sum = 0.0;
        for (int p = 0; p < k; ++p) {
            sum += A[row * lda + p] * B[col * ldb + p];
        }
        C[row * ldc + col] -= sum;
    }
}

struct CudaWorkspace {
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    cudaStream_t stream = nullptr;
    size_t max_elements = 0;
};

void checkCuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", context, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void initCudaWorkspace(CudaWorkspace& ws, size_t max_block) {
    ws.max_elements = max_block * max_block;
    checkCuda(cudaMalloc(&ws.dA, ws.max_elements * sizeof(double)), "cudaMalloc dA");
    checkCuda(cudaMalloc(&ws.dB, ws.max_elements * sizeof(double)), "cudaMalloc dB");
    checkCuda(cudaMalloc(&ws.dC, ws.max_elements * sizeof(double)), "cudaMalloc dC");
    checkCuda(cudaStreamCreate(&ws.stream), "cudaStreamCreate");
}

void destroyCudaWorkspace(CudaWorkspace& ws) {
    if (ws.stream) {
        cudaStreamDestroy(ws.stream);
    }
    if (ws.dA) {
        cudaFree(ws.dA);
    }
    if (ws.dB) {
        cudaFree(ws.dB);
    }
    if (ws.dC) {
        cudaFree(ws.dC);
    }
}

void gpu_update_block(const double* A, const double* B, double* C, int m, int n, int k,
                      CudaWorkspace& ws) {
    const size_t bytes_a = static_cast<size_t>(m) * k * sizeof(double);
    const size_t bytes_b = static_cast<size_t>(n) * k * sizeof(double);
    const size_t bytes_c = static_cast<size_t>(m) * n * sizeof(double);

    checkCuda(cudaMemcpyAsync(ws.dA, A, bytes_a, cudaMemcpyHostToDevice, ws.stream),
              "cudaMemcpyAsync A");
    checkCuda(cudaMemcpyAsync(ws.dB, B, bytes_b, cudaMemcpyHostToDevice, ws.stream),
              "cudaMemcpyAsync B");
    checkCuda(cudaMemcpyAsync(ws.dC, C, bytes_c, cudaMemcpyHostToDevice, ws.stream),
              "cudaMemcpyAsync C");

    dim3 block(kTileSize, kTileSize);
    dim3 grid((n + kTileSize - 1) / kTileSize, (m + kTileSize - 1) / kTileSize);
    gemm_update_kernel<<<grid, block, 0, ws.stream>>>(m, n, k, ws.dA, ws.dB, ws.dC, k, k, n);
    checkCuda(cudaGetLastError(), "gemm_update_kernel launch");

    checkCuda(cudaMemcpyAsync(C, ws.dC, bytes_c, cudaMemcpyDeviceToHost, ws.stream),
              "cudaMemcpyAsync C back");
    checkCuda(cudaStreamSynchronize(ws.stream), "cudaStreamSynchronize");
}

size_t blockSize(size_t block_index, size_t n) {
    const size_t start = block_index * kBlockSize;
    return std::min(static_cast<size_t>(kBlockSize), n - start);
}

int blockOwner(size_t block_index, size_t n_blocks, int size) {
    const size_t blocks_per_rank = n_blocks / static_cast<size_t>(size);
    const size_t extra = n_blocks % static_cast<size_t>(size);
    if (blocks_per_rank == 0) {
        return static_cast<int>(block_index);
    }
    const size_t cutoff = (blocks_per_rank + 1) * extra;
    if (block_index < cutoff) {
        return static_cast<int>(block_index / (blocks_per_rank + 1));
    }
    return static_cast<int>(extra + (block_index - cutoff) / blocks_per_rank);
}

void localBlockRange(size_t n_blocks, int size, int rank, size_t& block_start, size_t& block_count) {
    const size_t blocks_per_rank = n_blocks / static_cast<size_t>(size);
    const size_t extra = n_blocks % static_cast<size_t>(size);
    if (blocks_per_rank == 0) {
        if (rank < static_cast<int>(n_blocks)) {
            block_start = static_cast<size_t>(rank);
            block_count = 1;
        } else {
            block_start = n_blocks;
            block_count = 0;
        }
        return;
    }
    if (rank < static_cast<int>(extra)) {
        block_count = blocks_per_rank + 1;
        block_start = static_cast<size_t>(rank) * block_count;
    } else {
        block_count = blocks_per_rank;
        block_start = extra * (blocks_per_rank + 1)
                      + static_cast<size_t>(rank - static_cast<int>(extra)) * blocks_per_rank;
    }
}

void packBlock(const double* src, size_t ld_src, size_t rows, size_t cols, double* dst,
               size_t ld_dst) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        std::memcpy(dst + i * ld_dst, src + i * ld_src, cols * sizeof(double));
    }
}

void unpackBlock(double* dst, size_t ld_dst, size_t rows, size_t cols, const double* src,
                 size_t ld_src) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        std::memcpy(dst + i * ld_dst, src + i * ld_src, cols * sizeof(double));
    }
}

double* localBlockPtr(std::vector<double>& A, size_t row_start, size_t local_rows, size_t n,
                      size_t block_i, size_t block_j) {
    const size_t global_row = block_i * kBlockSize;
    if (global_row < row_start || global_row >= row_start + local_rows) {
        return nullptr;
    }
    const size_t local_row = global_row - row_start;
    return A.data() + local_row * n + block_j * kBlockSize;
}

bool choleskyBlock(double* A, size_t lda, size_t n) {
    for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            const double v = A[j * lda + k];
            sum += v * v;
        }
        const double val = A[j * lda + j] - sum;
        if (val <= 0.0) {
            return false;
        }
        A[j * lda + j] = std::sqrt(val);
#pragma omp parallel for schedule(static)
        for (size_t i = j + 1; i < n; ++i) {
            double subsum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                subsum += A[i * lda + k] * A[j * lda + k];
            }
            A[i * lda + j] = (A[i * lda + j] - subsum) / A[j * lda + j];
        }
    }
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * lda + j] = 0.0;
        }
    }
    return true;
}

void trsmBlock(double* Aik, size_t lda, const double* Lkk, size_t ldb, size_t rows, size_t cols) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += Aik[i * lda + k] * Lkk[j * ldb + k];
            }
            Aik[i * lda + j] = (Aik[i * lda + j] - sum) / Lkk[j * ldb + j];
        }
    }
}

bool distributedCholesky(std::vector<double>& A_local, size_t n, size_t row_start,
                          size_t local_rows, int rank, int size, CudaWorkspace& ws) {
    const size_t n_blocks = (n + kBlockSize - 1) / kBlockSize;
    std::vector<double> panel(n_blocks * kBlockSize * kBlockSize, 0.0);
    std::vector<double> diag(kBlockSize * kBlockSize, 0.0);
    std::vector<double> hostA(kBlockSize * kBlockSize);
    std::vector<double> hostB(kBlockSize * kBlockSize);
    std::vector<double> hostC(kBlockSize * kBlockSize);

    for (size_t k_block = 0; k_block < n_blocks; ++k_block) {
        const size_t kb = blockSize(k_block, n);
        const int owner = blockOwner(k_block, n_blocks, size);
        int diag_ok = 1;

        if (rank == owner) {
            double* Lkk = localBlockPtr(A_local, row_start, local_rows, n, k_block, k_block);
            diag_ok = Lkk && choleskyBlock(Lkk, n, kb) ? 1 : 0;
            if (diag_ok == 1) {
                packBlock(Lkk, n, kb, kb, diag.data(), kb);
            }
        }

        MPI_Bcast(&diag_ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (diag_ok == 0) {
            return false;
        }
        MPI_Bcast(diag.data(), static_cast<int>(kb * kb), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        for (size_t i_block = k_block + 1; i_block < n_blocks; ++i_block) {
            const size_t ib = blockSize(i_block, n);
            const int i_owner = blockOwner(i_block, n_blocks, size);
            double* panel_ptr = panel.data() + i_block * kBlockSize * kBlockSize;
            if (rank == i_owner) {
                double* Aik = localBlockPtr(A_local, row_start, local_rows, n, i_block, k_block);
                if (Aik) {
                    trsmBlock(Aik, n, diag.data(), kb, ib, kb);
                    packBlock(Aik, n, ib, kb, panel_ptr, kb);
                }
            }
            MPI_Bcast(panel_ptr, static_cast<int>(ib * kb), MPI_DOUBLE, i_owner, MPI_COMM_WORLD);
        }

        for (size_t i_block = k_block + 1; i_block < n_blocks; ++i_block) {
            if (blockOwner(i_block, n_blocks, size) != rank) {
                continue;
            }
            const size_t ib = blockSize(i_block, n);
            const double* Aik = panel.data() + i_block * kBlockSize * kBlockSize;
            for (size_t j_block = k_block + 1; j_block <= i_block; ++j_block) {
                const size_t jb = blockSize(j_block, n);
                double* Aij = localBlockPtr(A_local, row_start, local_rows, n, i_block, j_block);
                if (!Aij) {
                    continue;
                }

                const double* Ajk = panel.data() + j_block * kBlockSize * kBlockSize;
                std::memcpy(hostA.data(), Aik, ib * kb * sizeof(double));
                std::memcpy(hostB.data(), Ajk, jb * kb * sizeof(double));
                packBlock(Aij, n, ib, jb, hostC.data(), jb);

                gpu_update_block(hostA.data(), hostB.data(), hostC.data(), static_cast<int>(ib),
                                 static_cast<int>(jb), static_cast<int>(kb), ws);

                unpackBlock(Aij, n, ib, jb, hostC.data(), jb);
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (size_t local_i = 0; local_i < local_rows; ++local_i) {
        const size_t global_i = row_start + local_i;
        for (size_t j = global_i + 1; j < n; ++j) {
            A_local[local_i * n + j] = 0.0;
        }
    }

    return true;
}

} // namespace

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);

#pragma omp parallel
    {
        unsigned int seed = 42u + 17u * static_cast<unsigned int>(omp_get_thread_num());
#pragma omp for schedule(static)
        for (size_t i = 0; i < n * n; ++i) {
            B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
        }
    }

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig,
                      const size_t n) {
    std::vector<double> reconstructed(n * n);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

    double maxError = 0.0;
    double relError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        const double error = std::fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        const double rel = error / (std::fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }

    std::printf("Max absolute error: %.10e\n", maxError);
    std::printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = static_cast<size_t>(std::atoi(argv[++i]));
            } else if (std::strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (std::strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (std::strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    uint64_t n_bcast = static_cast<uint64_t>(n);
    int validate_bcast = validate ? 1 : 0;
    int results_bcast = printResults ? 1 : 0;
    MPI_Bcast(&n_bcast, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_bcast, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&results_bcast, 1, MPI_INT, 0, MPI_COMM_WORLD);
    n = static_cast<size_t>(n_bcast);
    validate = validate_bcast != 0;
    printResults = results_bcast != 0;

    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA devices available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(rank % device_count), "cudaSetDevice");

    const size_t n_blocks = (n + kBlockSize - 1) / kBlockSize;
    size_t block_start = 0;
    size_t block_count = 0;
    localBlockRange(n_blocks, size, rank, block_start, block_count);
    const size_t row_start = block_start * kBlockSize;
    const size_t row_end = std::min(n, (block_start + block_count) * kBlockSize);
    const size_t local_rows = row_end > row_start ? row_end - row_start : 0;

    std::vector<double> A_full;
    std::vector<double> A_orig;
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark (MPI + OpenMP + CUDA)\n");
        std::printf("Matrix size: %zu x %zu\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", size);
        std::printf("OpenMP threads: %d\n", omp_get_max_threads());

        A_full.resize(n * n);
        std::printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A_full, n);
        if (validate) {
            A_orig = A_full;
        }
    }

    std::vector<int> counts;
    std::vector<int> displs;
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; ++r) {
            size_t r_block_start = 0;
            size_t r_block_count = 0;
            localBlockRange(n_blocks, size, r, r_block_start, r_block_count);
            const size_t r_row_start = r_block_start * kBlockSize;
            const size_t r_row_end = std::min(n, (r_block_start + r_block_count) * kBlockSize);
            const size_t r_rows = r_row_end > r_row_start ? r_row_end - r_row_start : 0;
            counts[r] = static_cast<int>(r_rows * n);
            displs[r] = static_cast<int>(r_row_start * n);
        }
    }

    std::vector<double> A_local(local_rows * n);
    MPI_Scatterv(rank == 0 ? A_full.data() : nullptr,
                 rank == 0 ? counts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr,
                 MPI_DOUBLE,
                 A_local.empty() ? nullptr : A_local.data(),
                 static_cast<int>(local_rows * n),
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    CudaWorkspace ws;
    initCudaWorkspace(ws, kBlockSize);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();
    const bool success = distributedCholesky(A_local, n, row_start, local_rows, rank, size, ws);
    MPI_Barrier(MPI_COMM_WORLD);
    const double end_time = MPI_Wtime();

    destroyCudaWorkspace(ws);

    if (!success) {
        if (rank == 0) {
            std::printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        A_full.resize(n * n);
    }

    MPI_Gatherv(A_local.empty() ? nullptr : A_local.data(),
                static_cast<int>(local_rows * n),
                MPI_DOUBLE,
                rank == 0 ? A_full.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    const double local_time = end_time - start_time;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", max_time * 1000.0);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        const double gflops = ops / max_time / 1e9;
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(A_full, "CholeskyL");
        }

        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateCholesky(A_full, A_orig, n);
            if (valid) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
