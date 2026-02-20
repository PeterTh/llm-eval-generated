#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Hybrid blocked Cholesky:
// - MPI: distributed-memory parallelism via block-row distribution
// - CUDA: local GPU acceleration (TRSM + trailing update)
// - OpenMP: host-side packing/validation and small CPU diagonal factorizations

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t _e = (call);                                                         \
        if (_e != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

static inline int owner_of_block(const int blockIdx, const int nBlocks, const int nRanks) {
    const int base = nBlocks / nRanks;
    const int rem = nBlocks % nRanks;
    if (base == 0) return blockIdx; // first nBlocks ranks own exactly one block
    const int cut = (base + 1) * rem;
    if (blockIdx < cut) return blockIdx / (base + 1);
    return rem + (blockIdx - cut) / base;
}

static inline void block_range_for_rank(const int rank, const int nBlocks, const int nRanks, int& bStart, int& bCount) {
    const int base = nBlocks / nRanks;
    const int rem = nBlocks % nRanks;
    if (base == 0) {
        bStart = rank;
        bCount = (rank < nBlocks) ? 1 : 0;
        return;
    }
    bCount = base + (rank < rem ? 1 : 0);
    bStart = rank * base + std::min(rank, rem);
}

static bool generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Match original semantics: deterministic rand_r seed=42, A = B*B^T + n*I
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += (double)n;
    }
    return true;
}

static bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
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

#pragma omp parallel
    {
        double maxE = 0.0;
        double maxR = 0.0;
#pragma omp for nowait
        for (size_t i = 0; i < n * n; ++i) {
            const double error = fabs(reconstructed[i] - A_orig[i]);
            maxE = std::max(maxE, error);
            const double rel = error / (fabs(A_orig[i]) + 1e-10);
            maxR = std::max(maxR, rel);
        }
#pragma omp critical
        {
            maxError = std::max(maxError, maxE);
            relError = std::max(relError, maxR);
        }
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

static bool cholesky_factor_inplace(std::vector<double>& Ablk, const int ld, const int n) {
    // Unblocked Cholesky on host for a single diagonal block.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) sum += Ablk[i * ld + k] * Ablk[j * ld + k];

            if (i == j) {
                const double val = Ablk[j * ld + j] - sum;
                if (val <= 0.0) return false;
                Ablk[j * ld + j] = sqrt(val);
            } else {
                Ablk[i * ld + j] = (Ablk[i * ld + j] - sum) / Ablk[j * ld + j];
            }
        }
        for (int j = i + 1; j < n; ++j) Ablk[i * ld + j] = 0.0;
    }
    return true;
}

__global__ void trsm_panel_forward(double* __restrict__ dA, const int n, const int localRows, const int rowOffset,
                                  const int k, const int kb, const double* __restrict__ dLkk) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= (localRows - rowOffset)) return;
    const int li = rowOffset + r;
    double* row = dA + (size_t)li * n + k;

    // Forward solve: Lkk * x^T = a^T (in-place in row[0..kb-1])
    for (int t = 0; t < kb; ++t) {
        double sum = row[t];
        for (int s = 0; s < t; ++s) {
            sum -= dLkk[t * kb + s] * row[s];
        }
        row[t] = sum / dLkk[t * kb + t];
    }
}

__global__ void transpose_panel_tiled(const double* __restrict__ in, double* __restrict__ out,
                                     const int rows, const int cols) {
    __shared__ double tile[16][16 + 1];
    const int x = blockIdx.x * 16 + threadIdx.x; // col
    const int y = blockIdx.y * 16 + threadIdx.y; // row
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    __syncthreads();
    const int tx = blockIdx.y * 16 + threadIdx.x;
    const int ty = blockIdx.x * 16 + threadIdx.y;
    if (tx < rows && ty < cols) out[ty * rows + tx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void trailing_update_gemm(double* __restrict__ dA, const int n, const int localRows, const int rowOffset,
                                    const int k, const int kb, const double* __restrict__ dPanelT, const int ntrail) {
    __shared__ double As[16][16];
    __shared__ double Bs[16][16];

    const int j = blockIdx.x * 16 + threadIdx.x;
    const int r = blockIdx.y * 16 + threadIdx.y;
    const int nRows = localRows - rowOffset;
    const bool active = (j < ntrail) && (r < nRows);

    const int li = rowOffset + r;
    const double* Ap = active ? (dA + (size_t)li * n + k) : nullptr; // kb

    double acc = 0.0;
    // dPanelT is (kb x ntrail): dPanelT[t*ntrail + j]
    for (int t0 = 0; t0 < kb; t0 += 16) {
        const int t = t0 + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (active && t < kb) ? Ap[t] : 0.0;

        const int tt = t0 + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (j < ntrail && tt < kb) ? dPanelT[(size_t)tt * ntrail + j] : 0.0;

        __syncthreads();

        if (active) {
#pragma unroll
            for (int kk = 0; kk < 16; ++kk) {
                acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (active) {
        dA[(size_t)li * n + (k + kb) + j] -= acc;
    }
}

__global__ void zero_upper_triangle(double* __restrict__ dA, const int n, const int localRows, const int localRowStart) {
    const int j = blockIdx.x * 16 + threadIdx.x;
    const int i = blockIdx.y * 16 + threadIdx.y;
    if (i >= localRows || j >= n) return;
    const int gi = localRowStart + i;
    if (j > gi) dA[(size_t)i * n + j] = 0.0;
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, nRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found.\n");
        MPI_Finalize();
        return 1;
    }
    const int dev = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(dev));

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d\n", nRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int nb = 128; // block size
    const int nBlocks = (int)((n + nb - 1) / nb);

    // Determine block-row ownership (contiguous ranges of blocks per rank).
    int myBStart = 0, myBCount = 0;
    block_range_for_rank(rank, nBlocks, nRanks, myBStart, myBCount);
    const int localRowStart = std::min((size_t)myBStart * (size_t)nb, n);
    const int localRowEnd = std::min((size_t)(myBStart + myBCount) * (size_t)nb, n);
    const int localRows = (int)(localRowEnd - localRowStart);

    std::vector<int> allRowStart(nRanks), allRowEnd(nRanks), recvCounts(nRanks), displs(nRanks);
    for (int r = 0; r < nRanks; ++r) {
        int bS = 0, bC = 0;
        block_range_for_rank(r, nBlocks, nRanks, bS, bC);
        const int rs = (int)std::min((size_t)bS * (size_t)nb, n);
        const int re = (int)std::min((size_t)(bS + bC) * (size_t)nb, n);
        allRowStart[r] = rs;
        allRowEnd[r] = re;
        recvCounts[r] = (re - rs) * (int)n;
        displs[r] = rs * (int)n;
    }

    // Rank0 generates full matrix, then scatter rows.
    std::vector<double> A_full;
    std::vector<double> A_orig;
    if (rank == 0) {
        A_full.resize(n * n);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A_full, n);
        if (validate) A_orig = A_full;
    }

    std::vector<double> hA_local((size_t)localRows * n);
    MPI_Scatterv(rank == 0 ? A_full.data() : nullptr, recvCounts.data(), displs.data(), MPI_DOUBLE,
                 hA_local.data(), (int)hA_local.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Keep local matrix resident on GPU.
    double* dA = nullptr;
    if (!hA_local.empty()) {
        CUDA_CHECK(cudaMalloc(&dA, hA_local.size() * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dA, hA_local.data(), hA_local.size() * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Workspace: diagonal block, Lkk on device, panel buffers.
    std::vector<double> hDiag(nb * nb);
    double* dLkk = nullptr;
    CUDA_CHECK(cudaMalloc(&dLkk, nb * nb * sizeof(double)));

    // Panel buffers sized for worst case (n * nb).
    std::vector<double> hPanel((size_t)n * nb);
    std::vector<double> hPanelSend((size_t)localRows * nb);
    double* dPanel = nullptr;
    double* dPanelT = nullptr;
    CUDA_CHECK(cudaMalloc(&dPanel, (size_t)n * nb * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dPanelT, (size_t)nb * n * sizeof(double)));

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (int k = 0; k < (int)n; k += nb) {
        const int kb = std::min(nb, (int)n - k);
        const int blockIdx = k / nb;
        const int owner = owner_of_block(blockIdx, nBlocks, nRanks);

        int ok = 1;
        if (rank == owner) {
            const int localDiagOffset = k - localRowStart;
            if (localDiagOffset < 0 || localDiagOffset + kb > localRows) {
                ok = 0;
            } else {
                // Copy diagonal block to host, factor, copy back.
                CUDA_CHECK(cudaMemcpy2D(hDiag.data(), kb * sizeof(double),
                                        dA + (size_t)localDiagOffset * n + k, n * sizeof(double),
                                        kb * sizeof(double), kb, cudaMemcpyDeviceToHost));

                ok = cholesky_factor_inplace(hDiag, kb, kb) ? 1 : 0;
                if (ok) {
                    CUDA_CHECK(cudaMemcpy2D(dA + (size_t)localDiagOffset * n + k, n * sizeof(double),
                                            hDiag.data(), kb * sizeof(double),
                                            kb * sizeof(double), kb, cudaMemcpyHostToDevice));
                }
            }
        }

        MPI_Bcast(&ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!ok) {
            if (rank == 0) fprintf(stderr, "Error: Matrix is not positive definite at block starting %d\n", k);
            MPI_Finalize();
            return 1;
        }

        // Broadcast Lkk to all ranks (host), then copy to device.
        MPI_Bcast(hDiag.data(), kb * kb, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dLkk, hDiag.data(), kb * kb * sizeof(double), cudaMemcpyHostToDevice));

        // TRSM: compute panel L21 for local rows >= k+kb.
        const int trailStart = k + kb;
        const int rowOffset = std::max(0, trailStart - localRowStart);
        if (dA && rowOffset < localRows) {
            const int nTrailLocalRows = localRows - rowOffset;
            const dim3 trsmBlock(128);
            const dim3 trsmGrid((nTrailLocalRows + trsmBlock.x - 1) / trsmBlock.x);
            trsm_panel_forward<<<trsmGrid, trsmBlock>>>(dA, (int)n, localRows, rowOffset, k, kb, dLkk);
            CUDA_CHECK(cudaGetLastError());
        }

        // Gather full panel (rows trailStart..n-1, cols k..k+kb-1) into hPanel (ntrail x kb).
        const int ntrail = (int)n - trailStart;
        if (ntrail > 0) {
            // Prepare Allgatherv counts/displs for this iteration.
            std::vector<int> pCounts(nRanks, 0), pDispls(nRanks, 0);
            for (int r = 0; r < nRanks; ++r) {
                const int rs = allRowStart[r];
                const int re = allRowEnd[r];
                const int os = std::max(rs, trailStart);
                const int oe = std::max(os, re);
                const int rows = std::max(0, oe - os);
                pCounts[r] = rows * kb;
                pDispls[r] = (os - trailStart) * kb;
            }

            const int os = std::max(localRowStart, trailStart);
            const int oe = std::max(os, localRowEnd);
            const int sendRows = std::max(0, oe - os);
            const int localOff = os - localRowStart;

            if (sendRows > 0) {
                CUDA_CHECK(cudaMemcpy2D(hPanelSend.data(), kb * sizeof(double),
                                        dA + (size_t)localOff * n + k, n * sizeof(double),
                                        kb * sizeof(double), sendRows, cudaMemcpyDeviceToHost));
            }

            MPI_Allgatherv(sendRows > 0 ? hPanelSend.data() : nullptr, sendRows * kb, MPI_DOUBLE,
                           hPanel.data(), pCounts.data(), pDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);

            // Copy panel to device and transpose for coalesced access in update.
            CUDA_CHECK(cudaMemcpy(dPanel, hPanel.data(), (size_t)ntrail * kb * sizeof(double), cudaMemcpyHostToDevice));
            const dim3 tBlock(16, 16);
            const dim3 tGrid((kb + 15) / 16, (ntrail + 15) / 16);
            transpose_panel_tiled<<<tGrid, tBlock>>>(dPanel, dPanelT, ntrail, kb);
            CUDA_CHECK(cudaGetLastError());

            // Update trailing submatrix: A(trail, trail) -= L21 * L21^T
            if (dA && rowOffset < localRows) {
                const int nTrailLocalRows = localRows - rowOffset;
                const dim3 uBlock(16, 16);
                const dim3 uGrid((ntrail + 15) / 16, (nTrailLocalRows + 15) / 16);
                trailing_update_gemm<<<uGrid, uBlock>>>(dA, (int)n, localRows, rowOffset, k, kb, dPanelT, ntrail);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Zero out upper triangle locally.
    if (dA && localRows > 0) {
        const dim3 zBlock(16, 16);
        const dim3 zGrid(((int)n + 15) / 16, (localRows + 15) / 16);
        zero_upper_triangle<<<zGrid, zBlock>>>(dA, (int)n, localRows, localRowStart);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    double localTime = t1 - t0;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy back and gather full result on rank0 for printing/validation.
    if (dA && !hA_local.empty()) {
        CUDA_CHECK(cudaMemcpy(hA_local.data(), dA, hA_local.size() * sizeof(double), cudaMemcpyDeviceToHost));
    }

    if (rank == 0) A_full.assign(n * n, 0.0);
    MPI_Gatherv(hA_local.data(), (int)hA_local.size(), MPI_DOUBLE,
                rank == 0 ? A_full.data() : nullptr, recvCounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int validAll = 1;
    if (rank == 0) {
        const long ms = (long)(maxTime * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double ops = (double)n * (double)n * (double)n / 3.0;
        const double gflops = ops / maxTime / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(A_full, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(A_full, A_orig, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            validAll = valid ? 1 : 0;
        }
    }

    MPI_Bcast(&validAll, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!validAll) {
        CUDA_CHECK(cudaFree(dLkk));
        CUDA_CHECK(cudaFree(dPanel));
        CUDA_CHECK(cudaFree(dPanelT));
        if (dA) CUDA_CHECK(cudaFree(dA));
        MPI_Finalize();
        return 1;
    }

    CUDA_CHECK(cudaFree(dLkk));
    CUDA_CHECK(cudaFree(dPanel));
    CUDA_CHECK(cudaFree(dPanelT));
    if (dA) CUDA_CHECK(cudaFree(dA));

    MPI_Finalize();
    return 0;
}
