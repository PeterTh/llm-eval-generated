#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA blocked Cholesky decomposition.
//
// The matrix is distributed over a 2D process grid using block-cyclic tile
// ownership.  Every tile update is a dense GEMM executed with cuBLAS on the
// GPU owned by the MPI rank, panels are exchanged with MPI collectives and the
// host side (matrix generation / validation) is parallelised with OpenMP.

namespace {

inline void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

inline void cublasCheck(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error (%s): code %d\n", what, static_cast<int>(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Unblocked Cholesky factorisation of one diagonal tile (column-major).
// The accumulation order is kept ascending in k so the result matches the
// sequential reference as closely as possible.
__global__ void potrfKernel(double* A, int ld, int m, int offset, int* info) {
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    for (int j = 0; j < m; ++j) {
        if (tid == 0) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                const double v = A[k * ld + j];
                sum += v * v;
            }
            double val = A[j * ld + j] - sum;
            if (val <= 0.0) {
                if (info != nullptr) {
                    *info = offset + j + 1;
                }
                val = 0.0;
            }
            A[j * ld + j] = sqrt(val);
        }
        __syncthreads();

        const double diag = A[j * ld + j];
        for (int i = j + 1 + tid; i < m; i += nthreads) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += A[k * ld + i] * A[k * ld + j];
            }
            A[j * ld + i] = (A[j * ld + i] - sum) / diag;
        }
        __syncthreads();
    }
}

struct Distribution {
    int rank = 0;
    int size = 1;
    int pr = 1;
    int pc = 1;
    int myrow = 0;
    int mycol = 0;
    int n = 0;
    int nb = 256;
    int N = 0;   // n padded up to a multiple of nb
    int nbr = 0; // number of block rows / columns
    int local_nbr = 0;
    int local_nbc = 0;
    int local_ntiles = 0;
    MPI_Comm comm = MPI_COMM_WORLD;

    int tileIndex(int bi, int bj) const {
        const int lp = (bi - myrow) / pr;
        const int lq = (bj - mycol) / pc;
        return lp * local_nbc + lq;
    }
};

int chooseProcessRows(int size) {
    int best = 1;
    for (int d = 1; d * d <= size; ++d) {
        if (size % d == 0) {
            best = d;
        }
    }
    return best;
}

int chooseBlockSize(int n, int maxGridDim) {
    int nb = 256;
    while (nb > 32 && ((n + nb - 1) / nb) < maxGridDim) {
        nb >>= 1;
    }
    return nb;
}

// Generate the same pseudo random matrix B as the sequential reference.
void generateRandomMatrix(int n, std::vector<double>& B) {
    B.resize(static_cast<size_t>(n) * static_cast<size_t>(n));
    unsigned int seed = 42;
    for (size_t i = 0; i < B.size(); ++i) {
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }
}

// Transpose the column blocks of B that this rank needs for A = B * B^T so the
// inner tile generation loop is contiguous and vectorisable.
void buildTransposedBlocks(const Distribution& dist, const std::vector<double>& B,
                           std::vector<double>& Bt) {
    const int n = dist.n;
    const int nb = dist.nb;
    Bt.assign(static_cast<size_t>(dist.local_nbc) * nb * n, 0.0);

    for (int lq = 0; lq < dist.local_nbc; ++lq) {
        const int bj = dist.mycol + lq * dist.pc;
        double* dst = Bt.data() + static_cast<size_t>(lq) * nb * n;
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < nb; ++j) {
                const int gj = bj * nb + j;
                dst[static_cast<size_t>(k) * nb + j] =
                    (gj < n) ? B[static_cast<size_t>(gj) * n + k] : 0.0;
            }
        }
    }
}

// Compute the lower-triangular tiles owned by this rank.
void generateLocalTiles(const Distribution& dist, const std::vector<double>& B,
                        const std::vector<double>& Bt, std::vector<double>& tiles) {
    const int n = dist.n;
    const int nb = dist.nb;
    const size_t tileElems = static_cast<size_t>(nb) * nb;
    tiles.assign(static_cast<size_t>(dist.local_ntiles) * tileElems, 0.0);

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int lp = 0; lp < dist.local_nbr; ++lp) {
        for (int lq = 0; lq < dist.local_nbc; ++lq) {
            const int bi = dist.myrow + lp * dist.pr;
            const int bj = dist.mycol + lq * dist.pc;
            if (bj > bi) {
                continue; // only the lower triangle is required
            }

            double* tile = tiles.data() + static_cast<size_t>(lp * dist.local_nbc + lq) * tileElems;
            std::vector<double> tmp(tileElems, 0.0);
            const double* bt = Bt.data() + static_cast<size_t>(lq) * nb * n;

            for (int i = 0; i < nb; ++i) {
                const int gi = bi * nb + i;
                double* trow = tmp.data() + static_cast<size_t>(i) * nb;

                if (gi >= n) {
                    // Padding block: use the identity so the extended matrix
                    // remains positive definite.
                    for (int j = 0; j < nb; ++j) {
                        const int gj = bj * nb + j;
                        trow[j] = (gj == gi) ? 1.0 : 0.0;
                    }
                    continue;
                }

                const double* brow = B.data() + static_cast<size_t>(gi) * n;
                for (int k = 0; k < n; ++k) {
                    const double b = brow[k];
                    const double* btk = bt + static_cast<size_t>(k) * nb;
                    for (int j = 0; j < nb; ++j) {
                        trow[j] += b * btk[j];
                    }
                }

                const int jj = gi - bj * nb;
                if (jj >= 0 && jj < nb) {
                    trow[jj] += static_cast<double>(n); // diagonal dominance
                }
            }

            // tmp is row-major (i, j); store the tile column-major.
            for (int i = 0; i < nb; ++i) {
                for (int j = 0; j < nb; ++j) {
                    tile[static_cast<size_t>(j) * nb + i] = tmp[static_cast<size_t>(i) * nb + j];
                }
            }
        }
    }
}

// Factor one diagonal tile with an inner blocked algorithm.  The small
// diagonal sub-blocks are handled by the serial kernel, while the trailing
// updates use cuBLAS so the work is spread over the whole GPU.
void factorTile(cublasHandle_t handle, double* tile, int ld, int offset, int* dInfo) {
    const int inner = 32;
    const double one = 1.0;
    const double minusOne = -1.0;

    for (int jb = 0; jb < ld; jb += inner) {
        const int m = std::min(inner, ld - jb);
        double* diag = tile + static_cast<size_t>(jb) * ld + jb;
        potrfKernel<<<1, m>>>(diag, ld, m, offset + jb, dInfo);
        cudaCheck(cudaGetLastError(), "potrf launch");

        const int rs = jb + m;
        const int rem = ld - rs;
        if (rem <= 0) {
            continue;
        }

        double* panel = tile + static_cast<size_t>(jb) * ld + rs;
        cublasCheck(cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, rem, m, &one,
                                diag, ld, panel, ld),
                    "inner trsm");

        double* trailing = tile + static_cast<size_t>(rs) * ld + rs;
        cublasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, rem, rem, m,
                                &minusOne, panel, ld, panel, ld, &one, trailing, ld),
                    "inner gemm");
    }
}

// Distributed right-looking blocked Cholesky.  The local tiles are already on
// the device (dTiles) and the factor replaces them in place.  failIndex
// receives the global diagonal index if the matrix is not positive definite.
bool distributedCholesky(const Distribution& dist, double* dTiles, double* dPanel,
                         double* hPanel, int* dInfo, cublasHandle_t handle,
                         int& failIndex) {
    const int nb = dist.nb;
    const int N = dist.N;
    const int nbr = dist.nbr;
    const size_t tileElems = static_cast<size_t>(nb) * nb;
    const size_t tileBytes = tileElems * sizeof(double);

    double* hLkk = nullptr;
    double* sendBuf = nullptr;
    double* recvBuf = nullptr;
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&hLkk), tileBytes), "pinned lkk");
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&sendBuf),
                             static_cast<size_t>(std::max(1, dist.local_nbr)) * nb * nb *
                                 sizeof(double)),
              "pinned send");
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&recvBuf),
                             static_cast<size_t>(nbr) * nb * nb * sizeof(double)),
              "pinned recv");
    std::memset(hLkk, 0, tileBytes);

    const double one = 1.0;
    const double minusOne = -1.0;
    bool ok = true;
    failIndex = 0;

    for (int k = 0; k < nbr; ++k) {
        const int Pld = N - k * nb;
        const int owner = (k % dist.pr) * dist.pc + (k % dist.pc);
        int hInfo = 0;

        if (dist.rank == owner) {
            const int idx = dist.tileIndex(k, k);
            double* tile = dTiles + static_cast<size_t>(idx) * tileElems;
            cudaCheck(cudaMemset(dInfo, 0, sizeof(int)), "reset info");
            factorTile(handle, tile, nb, k * nb, dInfo);
            cudaCheck(cudaMemcpy(&hInfo, dInfo, sizeof(int), cudaMemcpyDeviceToHost),
                      "read info");
            cudaCheck(cudaMemcpy(hLkk, tile, tileBytes, cudaMemcpyDeviceToHost),
                      "read lkk");
        }

        MPI_Bcast(&hInfo, 1, MPI_INT, owner, dist.comm);
        if (hInfo != 0) {
            ok = false;
            failIndex = hInfo - 1;
            break;
        }

        MPI_Bcast(hLkk, static_cast<int>(tileElems), MPI_DOUBLE, owner, dist.comm);

        // L_kk into the device panel (block k occupies the first nb rows).
        cudaCheck(cudaMemcpy2D(dPanel, static_cast<size_t>(N) * sizeof(double),
                               hLkk, static_cast<size_t>(nb) * sizeof(double),
                               static_cast<size_t>(nb) * sizeof(double),
                               static_cast<size_t>(nb), cudaMemcpyHostToDevice),
                  "lkk to panel");

        // Phase 1: every rank solves the panel blocks it owns and packs them.
        const int col = k % dist.pc;
        std::vector<int> counts(dist.size, 0);
        std::vector<int> displs(dist.size, 0);
        int totalBlocks = 0;
        for (int r = 0; r < dist.size; ++r) {
            int cnt = 0;
            if (r % dist.pc == col) {
                const int row = r / dist.pc;
                const int lnbr = (row < nbr) ? ((nbr - 1 - row) / dist.pr + 1) : 0;
                if (k < row) {
                    cnt = lnbr;
                } else {
                    cnt = lnbr - ((k - row) / dist.pr + 1);
                }
                if (cnt < 0) {
                    cnt = 0;
                }
            }
            counts[r] = cnt * nb * nb;
            displs[r] = totalBlocks * nb * nb;
            totalBlocks += cnt;
        }
        int myBlocks = 0;
        if (dist.mycol == col) {
            const int lq = (k - dist.mycol) / dist.pc;
            const int p0 = (k < dist.myrow) ? 0 : ((k - dist.myrow) / dist.pr + 1);
            for (int lp = p0; lp < dist.local_nbr; ++lp) {
                const int idx = lp * dist.local_nbc + lq;
                double* tile = dTiles + static_cast<size_t>(idx) * tileElems;

                cublasCheck(cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                        CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, nb, nb, &one,
                                        dPanel, N, tile, nb),
                            "trsm");

                cudaCheck(cudaMemcpy(sendBuf + static_cast<size_t>(myBlocks) * nb * nb,
                                     tile, tileBytes, cudaMemcpyDeviceToHost),
                          "panel to host");
                ++myBlocks;
            }
        }

        MPI_Allgatherv(sendBuf, myBlocks * nb * nb, MPI_DOUBLE,
                       recvBuf, counts.data(), displs.data(), MPI_DOUBLE,
                       dist.comm);

        std::memset(hPanel, 0, static_cast<size_t>(Pld) * nb * sizeof(double));

        for (int r = 0; r < dist.size; ++r) {
            const int cntBlocks = counts[r] / (nb * nb);
            if (cntBlocks == 0) {
                continue;
            }
            const int row = r / dist.pc;
            const int p0 = (k < row) ? 0 : ((k - row) / dist.pr + 1);
            const double* src = recvBuf + displs[r];
            for (int t = 0; t < cntBlocks; ++t) {
                const int bi = row + (p0 + t) * dist.pr;
                double* dst = hPanel + static_cast<size_t>(bi - k) * nb;
                for (int c = 0; c < nb; ++c) {
                    std::memcpy(dst + static_cast<size_t>(c) * Pld,
                                src + static_cast<size_t>(c) * nb,
                                static_cast<size_t>(nb) * sizeof(double));
                }
                src += static_cast<size_t>(nb) * nb;
            }
        }

        cudaCheck(cudaMemcpy2D(dPanel, static_cast<size_t>(N) * sizeof(double),
                               hPanel, static_cast<size_t>(Pld) * sizeof(double),
                               static_cast<size_t>(Pld) * sizeof(double),
                               static_cast<size_t>(nb), cudaMemcpyHostToDevice),
                  "panel to device");
        // Phase 2: update every owned trailing tile.
        for (int lp = 0; lp < dist.local_nbr; ++lp) {
            const int bi = dist.myrow + lp * dist.pr;
            if (bi <= k) {
                continue;
            }
            const double* panelI = dPanel + static_cast<size_t>(bi - k) * nb;
            for (int lq = 0; lq < dist.local_nbc; ++lq) {
                const int bj = dist.mycol + lq * dist.pc;
                if (bj <= k || bj > bi) {
                    continue;
                }
                double* tile = dTiles +
                               static_cast<size_t>(lp * dist.local_nbc + lq) * tileElems;
                const double* panelJ = dPanel + static_cast<size_t>(bj - k) * nb;

                cublasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, nb, nb, nb,
                                        &minusOne, panelI, N, panelJ, N, &one, tile, nb),
                            "gemm");
            }
        }
    }
    cudaFreeHost(hLkk);
    cudaFreeHost(sendBuf);
    cudaFreeHost(recvBuf);
    return ok;
}

// Pack the owned lower-triangular tiles into a dense n x n buffer and reduce
// them on rank 0.  Optionally do the same for the original matrix (used by the
// validation path).
void gatherMatrices(const Distribution& dist, const std::vector<double>& tiles,
                    const std::vector<double>& origTiles, bool needL, bool needA,
                    std::vector<double>& fullL, std::vector<double>& fullA) {
    const int n = dist.n;
    const int nb = dist.nb;
    const size_t n2 = static_cast<size_t>(n) * n;
    std::vector<double> sendL;
    std::vector<double> sendA;
    if (needL) {
        sendL.assign(n2, 0.0);
    }
    if (needA) {
        sendA.assign(n2, 0.0);
    }

    for (int lp = 0; lp < dist.local_nbr; ++lp) {
        const int bi = dist.myrow + lp * dist.pr;
        for (int lq = 0; lq < dist.local_nbc; ++lq) {
            const int bj = dist.mycol + lq * dist.pc;
            if (bj > bi) {
                continue;
            }
            const size_t base = static_cast<size_t>(lp * dist.local_nbc + lq) * nb * nb;
            const double* tile = tiles.data() + base;
            const double* orig = origTiles.empty() ? nullptr : origTiles.data() + base;

            for (int i = 0; i < nb; ++i) {
                const int gi = bi * nb + i;
                if (gi >= n) {
                    continue;
                }
                for (int j = 0; j < nb; ++j) {
                    const int gj = bj * nb + j;
                    if (gj >= n || gj > gi) {
                        continue;
                    }
                    const size_t dst = static_cast<size_t>(gi) * n + gj;
                    if (needL) {
                        sendL[dst] = tile[static_cast<size_t>(j) * nb + i];
                    }
                    if (needA) {
                        sendA[dst] = orig[static_cast<size_t>(j) * nb + i];
                    }
                }
            }
        }
    }

    if (needL) {
        if (dist.rank == 0) {
            fullL.assign(n2, 0.0);
        }
        MPI_Reduce(sendL.data(), dist.rank == 0 ? fullL.data() : nullptr,
                   static_cast<int>(n2), MPI_DOUBLE, MPI_SUM, 0, dist.comm);
    }
    if (needA) {
        if (dist.rank == 0) {
            fullA.assign(n2, 0.0);
        }
        MPI_Reduce(sendA.data(), dist.rank == 0 ? fullA.data() : nullptr,
                   static_cast<int>(n2), MPI_DOUBLE, MPI_SUM, 0, dist.comm);
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig,
                      const size_t n) {
    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for schedule(static) reduction(max : maxError, relError)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        for (long long j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[static_cast<size_t>(i) * n + k] *
                       L[static_cast<size_t>(j) * n + k];
            }
            const double error =
                fabs(sum - A_orig[static_cast<size_t>(i) * n + static_cast<size_t>(j)]);
            maxError = std::max(maxError, error);
            const double rel =
                error / (fabs(A_orig[static_cast<size_t>(i) * n + static_cast<size_t>(j)]) +
                         1e-10);
            relError = std::max(relError, rel);
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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int localRankFromEnvironment() {
    const char* names[] = {"OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK",
                           "SLURM_LOCALID", "PMI_LOCAL_RANK"};
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr && *value != '\0') {
            return std::atoi(value);
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    if (n == 0) {
        if (rank == 0) {
            printf("Matrix size must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Pick one GPU per rank (respecting CUDA_VISIBLE_DEVICES if it is set).
    int deviceCount = 0;
    const int deviceQueryOk =
        ((cudaGetDeviceCount(&deviceCount) == cudaSuccess) && deviceCount > 0) ? 1 : 0;
    int allHaveGpu = 0;
    MPI_Allreduce(&deviceQueryOk, &allHaveGpu, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!allHaveGpu) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA device available; this build requires a GPU\n");
        }
        MPI_Finalize();
        return 1;
    }
    const int localRank = localRankFromEnvironment();
    cudaCheck(cudaSetDevice(localRank % deviceCount), "set device");

    Distribution dist;
    dist.rank = rank;
    dist.size = size;
    dist.pr = chooseProcessRows(size);
    dist.pc = size / dist.pr;
    dist.myrow = rank / dist.pc;
    dist.mycol = rank % dist.pc;
    dist.n = static_cast<int>(n);
    dist.nb = chooseBlockSize(dist.n, std::max(dist.pr, dist.pc));
    dist.N = ((dist.n + dist.nb - 1) / dist.nb) * dist.nb;
    dist.nbr = dist.N / dist.nb;
    dist.local_nbr = (dist.nbr - 1 - dist.myrow) / dist.pr + 1;
    dist.local_nbc = (dist.nbr - 1 - dist.mycol) / dist.pc + 1;
    if (dist.myrow >= dist.nbr || dist.mycol >= dist.nbr) {
        dist.local_nbr = 0;
        dist.local_nbc = 0;
    }
    dist.local_ntiles = dist.local_nbr * dist.local_nbc;

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
        fflush(stdout);
    }

    std::vector<double> B;
    generateRandomMatrix(dist.n, B);

    std::vector<double> Bt;
    buildTransposedBlocks(dist, B, Bt);

    std::vector<double> tiles;
    generateLocalTiles(dist, B, Bt, tiles);
    B.clear();
    B.shrink_to_fit();
    Bt.clear();
    Bt.shrink_to_fit();

    std::vector<double> origTiles;
    if (validate) {
        origTiles = tiles;
    }

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
        fflush(stdout);
    }

    cublasHandle_t handle = nullptr;
    cublasCheck(cublasCreate(&handle), "create handle");

    const size_t tileElems = static_cast<size_t>(dist.nb) * dist.nb;
    const size_t tileBytes = tileElems * sizeof(double);
    double* dTiles = nullptr;
    double* dPanel = nullptr;
    double* hPanel = nullptr;
    int* dInfo = nullptr;

    // Workspace allocation and upload are setup, not part of the measured
    // decomposition (just like the host allocation in the sequential version).
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dTiles),
                         static_cast<size_t>(dist.local_ntiles) * tileBytes),
              "malloc tiles");
    if (dist.local_ntiles > 0) {
        cudaCheck(cudaMemcpy(dTiles, tiles.data(),
                             static_cast<size_t>(dist.local_ntiles) * tileBytes,
                             cudaMemcpyHostToDevice),
                  "upload tiles");
    }
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dPanel),
                         static_cast<size_t>(dist.N) * dist.nb * sizeof(double)),
              "malloc panel");
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&hPanel),
                             static_cast<size_t>(dist.N) * dist.nb * sizeof(double)),
              "malloc host panel");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dInfo), sizeof(int)), "malloc info");
    cudaDeviceSynchronize();
    {
        // Prime cuBLAS kernel selection for every shape used by the solver.
        double* scratch = nullptr;
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&scratch), tileBytes), "warmup tile");
        if (dist.local_ntiles > 0) {
            cudaCheck(cudaMemcpy(scratch, dTiles, tileBytes, cudaMemcpyDeviceToDevice),
                      "warmup copy");
            cudaCheck(cudaMemset(dPanel, 0,
                                 static_cast<size_t>(dist.N) * dist.nb * sizeof(double)),
                      "warmup panel");
            cudaCheck(cudaMemcpy2D(dPanel, static_cast<size_t>(dist.N) * sizeof(double),
                                   scratch, static_cast<size_t>(dist.nb) * sizeof(double),
                                   static_cast<size_t>(dist.nb) * sizeof(double),
                                   static_cast<size_t>(dist.nb),
                                   cudaMemcpyDeviceToDevice),
                      "warmup lkk");
            cudaCheck(cudaMemset(dInfo, 0, sizeof(int)), "warmup info");
            factorTile(handle, scratch, dist.nb, 0, dInfo);

            const double one = 1.0;
            const double minusOne = -1.0;
            cublasCheck(cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, dist.nb, dist.nb,
                                    &one, dPanel, dist.N, scratch, dist.nb),
                        "warmup trsm");
            cublasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, dist.nb, dist.nb,
                                    dist.nb, &minusOne, dPanel, dist.N, dPanel, dist.N,
                                    &one, scratch, dist.nb),
                        "warmup gemm");
        }
        cudaCheck(cudaDeviceSynchronize(), "warmup sync");
        cudaFree(scratch);
        // Only the scratch copy was modified; make sure the device holds the
        // pristine input tiles.
        if (dist.local_ntiles > 0) {
            cudaCheck(cudaMemcpy(dTiles, tiles.data(),
                                 static_cast<size_t>(dist.local_ntiles) * tileBytes,
                                 cudaMemcpyHostToDevice),
                      "restore tiles");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    int failIndex = 0;
    const bool success = distributedCholesky(dist, dTiles, dPanel, hPanel, dInfo,
                                             handle, failIndex);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    cublasDestroy(handle);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double elapsedSeconds =
        std::chrono::duration<double>(end - start).count();

    if (!success) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n",
                   failIndex);
            printf("Cholesky decomposition failed\n");
        }
        cudaFree(dTiles);
        cudaFree(dPanel);
        cudaFreeHost(hPanel);
        cudaFree(dInfo);
        MPI_Finalize();
        return 1;
    }

    if (dist.local_ntiles > 0) {
        cudaCheck(cudaMemcpy(tiles.data(), dTiles,
                             static_cast<size_t>(dist.local_ntiles) * tileBytes,
                             cudaMemcpyDeviceToHost),
                  "download tiles");
    }
    cudaFree(dTiles);
    cudaFree(dPanel);
    cudaFreeHost(hPanel);
    cudaFree(dInfo);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
        const double ops = static_cast<double>(n) * static_cast<double>(n) *
                           static_cast<double>(n) / 3.0;
        const double gflops =
            (elapsedSeconds > 0.0) ? (ops / elapsedSeconds / 1e9) : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
        fflush(stdout);
    }

    std::vector<double> fullL;
    std::vector<double> fullA;
    if (printResults || validate) {
        gatherMatrices(dist, tiles, origTiles, printResults || validate, validate,
                       fullL, fullA);
    }

    int exitCode = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(fullL, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(fullL, fullA, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        fflush(stdout);
    }

    int globalExit = 0;
    MPI_Allreduce(&exitCode, &globalExit, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Finalize();
    return globalExit;
}
