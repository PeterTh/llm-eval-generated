#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition.
//
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// A is stored in row-major order in the reference algorithm; the result L is
// identical in structure (lower triangular, zero upper part).
//
// Parallel decomposition:
//   * MPI:    rows are distributed cyclically over ranks (global row i is owned by
//             rank i % P and stored as local row i / P), which balances the
//             triangular workload. Validation and output gathering use MPI.
//   * CUDA:   each rank keeps its rows on its GPU (column-major). Matrix generation,
//             the dominant work (dot products against already-factored columns) and
//             the panel solve run as CUDA kernels.
//   * OpenMP: host-side work (pseudo-random generation of B, layout conversion of
//             the factor, distributed validation).
//
// Algorithm: blocked left-looking Cholesky. Columns are factored in panels of
// kPanel columns. For every element, the reduction over k is accumulated in the
// same ascending order as the original unblocked algorithm, so results match
// the sequential reference up to floating-point contraction.

namespace {

constexpr size_t kPanel = 64;  // number of columns factored per panel

#define CHECK_CUDA(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err_));                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// Number of global rows owned by `rank` for an n-row matrix distributed over `size` ranks.
size_t localRowCount(size_t n, int rank, int size) {
    const size_t r = static_cast<size_t>(rank);
    const size_t p = static_cast<size_t>(size);
    return n > r ? (n - r + p - 1) / p : 0;
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Pivot panel rows are addressed through rowOff[c]: the offset of panel row c (its
// factored prefix, k < j0, followed by its original panel block) inside the gathered
// pivot buffer. Data is replicated on all ranks.

// accD[a][b] = sum_{k<j0} piv[a][k] * piv[b][k]   for b <= a (panel diagonal block,
// partial sums over the already-factored prefix).
__global__ void kPivotDiag(const double* __restrict__ piv, const int* __restrict__ rowOff,
                           int j0, int w, double* __restrict__ accD) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int a = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= w || b >= w || b > a) return;
    const double* pa = piv + rowOff[a];
    const double* pb = piv + rowOff[b];
    double s = 0.0;
    for (int k = 0; k < j0; ++k) s += pa[k] * pb[k];
    accD[a * w + b] = s;
}

// accE[r][c] = sum_{k<j0} L[r][k] * piv[c][k] for local rows r >= rStart.
// Local matrix is column-major: element (r, k) at Ld[k * nLoc + r].
__global__ void kExternal(const double* __restrict__ Ld, int nLoc, int rStart,
                          const double* __restrict__ piv, const int* __restrict__ rowOff,
                          int j0, int w, double* __restrict__ accE) {
    const int r = rStart + blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= nLoc || c >= w) return;
    const double* pc = piv + rowOff[c];
    double s = 0.0;
    for (int k = 0; k < j0; ++k) s += Ld[static_cast<size_t>(k) * nLoc + r] * pc[k];
    accE[static_cast<size_t>(r) * w + c] = s;
}

// Packs the panel rows owned by this rank into a compact row-major staging buffer:
// stage[t * ldp + k] = L/A[j0 + c][k] for k < ldp (ldp = j1), where c = c0 + t * size
// are the panel rows owned by this rank. Grid: (ceil(ldp / blockDim.x), cntOwned).
__global__ void kPackOwnedPanelRows(const double* __restrict__ Ld, int nLoc, int j0, int ldp,
                                    int rank, int size, int c0,
                                    double* __restrict__ stage) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (k >= ldp) return;
    const int j = j0 + c0 + t * size;  // global row, owned by this rank
    const int lr = (j - rank) / size;
    stage[static_cast<size_t>(t) * ldp + k] = Ld[static_cast<size_t>(k) * nLoc + lr];
}

// Finishes the panel columns for local rows r >= rStart (rows below the panel).
// For each panel column c (in order), the internal sum over panel columns k < c is
// added to the external partial sum, then L[r][j0+c] = (A - sum) / L[j0+c][j0+c].
__global__ void kPanelSolve(double* __restrict__ Ld, int nLoc, int rStart, int j0, int w,
                            const double* __restrict__ accE,
                            const double* __restrict__ Lp,
                            const double* __restrict__ Ldiag) {
    const int r = rStart + blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nLoc) return;
    for (int c = 0; c < w; ++c) {
        double sum = accE[static_cast<size_t>(r) * w + c];
        for (int k = 0; k < c; ++k) {
            sum += Ld[static_cast<size_t>(j0 + k) * nLoc + r] * Lp[c * w + k];
        }
        double& dst = Ld[static_cast<size_t>(j0 + c) * nLoc + r];
        dst = (dst - sum) / Ldiag[c];
    }
}

// Writes the factored panel block (lower triangle incl. diagonal) into the rows
// owned by this rank. Grid: w blocks (panel rows), w threads (panel columns).
__global__ void kWritePanelBlock(double* __restrict__ Ld, int nLoc, int j0, int w,
                                 int rank, int size, const double* __restrict__ Lp) {
    const int cp = blockIdx.x;  // panel row
    const int c = threadIdx.x;  // panel column
    if (cp >= w || c > cp) return;
    const int j = j0 + cp;
    if (j % size != rank) return;
    const int lr = (j - rank) / size;
    Ld[static_cast<size_t>(j0 + c) * nLoc + lr] = Lp[cp * w + c];
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

// Factors the w x w panel diagonal block on the host. Executed identically on all
// ranks (inputs are replicated), so every rank reaches the same decision.
//   piv:    replicated pivot data; panel row c starts at piv[rowOff[c]], and the
//           original panel block of that row is at piv[rowOff[c] + j0 + c'].
//   accD:   external partial sums for the panel block (lower triangle).
// Outputs: Lp (w x w row-major, lower triangle, diagonal holds Ldiag) and Ldiag.
bool factorPanel(const double* piv, const int* rowOff, size_t j0, size_t w, const double* accD,
                 double* Lp, double* Ldiag, int rank) {
    std::fill(Lp, Lp + w * w, 0.0);
    for (size_t c = 0; c < w; ++c) {
        for (size_t r = c; r < w; ++r) {
            double sum = accD[r * w + c];
            for (size_t k = 0; k < c; ++k) sum += Lp[r * w + k] * Lp[c * w + k];
            if (r == c) {
                const double val = piv[rowOff[c] + j0 + c] - sum;
                if (val <= 0.0) {
                    if (rank == 0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                               j0 + c);
                    }
                    return false;
                }
                Ldiag[c] = sqrt(val);
                Lp[c * w + c] = Ldiag[c];
            } else {
                Lp[r * w + c] = (piv[rowOff[r] + j0 + c] - sum) / Ldiag[c];
            }
        }
    }
    return true;
}

// Device and host scratch buffers reused across panels.
struct Workspace {
    double* dPiv = nullptr;    // gathered pivot panel on device (w x ldp, offset-addressed)
    double* dAccD = nullptr;
    double* dAccE = nullptr;
    double* dLp = nullptr;
    double* dLdiag = nullptr;
    int* dRowOff = nullptr;    // offsets of panel rows inside dPiv / hRecv
    double* hSend = nullptr;   // pinned: this rank's owned panel rows (compact)
    double* hRecv = nullptr;   // pinned: gathered panel rows, grouped by owning rank
    int* hRowOff = nullptr;
    double* hAccD = nullptr;
    double* hLp = nullptr;
    double* hLdiag = nullptr;

    Workspace(size_t n, size_t nLoc) {
        const size_t w = std::min(kPanel, std::max<size_t>(n, 1));
        const size_t panelElems = w * std::max<size_t>(n, 1);
        CHECK_CUDA(cudaMalloc(&dPiv, sizeof(double) * panelElems));
        CHECK_CUDA(cudaMalloc(&dAccD, sizeof(double) * w * w));
        CHECK_CUDA(cudaMalloc(&dAccE, sizeof(double) * std::max<size_t>(nLoc * w, 1)));
        CHECK_CUDA(cudaMalloc(&dLp, sizeof(double) * w * w));
        CHECK_CUDA(cudaMalloc(&dLdiag, sizeof(double) * w));
        CHECK_CUDA(cudaMalloc(&dRowOff, sizeof(int) * w));
        CHECK_CUDA(cudaMallocHost(&hSend, sizeof(double) * panelElems));
        CHECK_CUDA(cudaMallocHost(&hRecv, sizeof(double) * panelElems));
        hRowOff = new int[w];
        hAccD = new double[w * w];
        hLp = new double[w * w];
        hLdiag = new double[w];
    }

    ~Workspace() {
        cudaFree(dPiv);
        cudaFree(dAccD);
        cudaFree(dAccE);
        cudaFree(dLp);
        cudaFree(dLdiag);
        cudaFree(dRowOff);
        cudaFreeHost(hSend);
        cudaFreeHost(hRecv);
        delete[] hRowOff;
        delete[] hAccD;
        delete[] hLp;
        delete[] hLdiag;
    }

    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
};

// Distributed in-place Cholesky of the column-major device matrix dL.
// dL holds this rank's rows: element (local row r, column k) at dL[k * nLoc + r].
// Returns false if the matrix is not positive definite (same result on all ranks).
bool choleskyDecomposition(double* dL, size_t n, int rank, int size, size_t nLoc,
                           Workspace& ws) {
    const int nl = static_cast<int>(nLoc);
    const int ni = static_cast<int>(n);
    const size_t dataDouble = sizeof(double);

    for (int j0 = 0; j0 < ni; j0 += static_cast<int>(kPanel)) {
        const int j1 = std::min(ni, j0 + static_cast<int>(kPanel));
        const int w = j1 - j0;
        const int ldp = j0 + w;  // row stride of the pivot buffer

        // 1. Panel rows are owned cyclically: row j0 + c belongs to rank (j0 + c) % size.
        //    Owners pack their rows (factored prefix, columns < j0, and the original
        //    panel block, columns j0..j1-1) into a compact staging buffer.
        //    Panel row c is owned by rank q iff c = c0(q) + t * size.
        const int c0 = ((rank - j0 % size) % size + size) % size;
        const int cntOwned = (c0 < w) ? (w - c0 + size - 1) / size : 0;
        if (cntOwned > 0) {
            dim3 block(256);
            dim3 grid((ldp + 255) / 256, cntOwned);
            kPackOwnedPanelRows<<<grid, block>>>(dL, nl, j0, ldp, rank, size, c0, ws.dPiv);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaMemcpy(ws.hSend, ws.dPiv, dataDouble * static_cast<size_t>(cntOwned) * ldp,
                                  cudaMemcpyDeviceToHost));
        }

        // 2. Replicate the pivot panel on all ranks with an allgather (each row is sent
        //    once, by its owner). Rows stay in gathered order; rowOff maps panel row c
        //    to its position, so no host-side reordering is needed.
        std::vector<int> counts(size), displs(size);
        for (int q = 0; q < size; ++q) {
            const int cq0 = ((q - j0 % size) % size + size) % size;
            const int cq = (cq0 < w) ? (w - cq0 + size - 1) / size : 0;
            counts[q] = cq * ldp;
            displs[q] = (q == 0) ? 0 : displs[q - 1] + counts[q - 1];
            for (int c = cq0, t = 0; c < w; c += size, ++t) {
                ws.hRowOff[c] = displs[q] + t * ldp;
            }
        }
        MPI_Allgatherv(ws.hSend, cntOwned * ldp, MPI_DOUBLE, ws.hRecv, counts.data(),
                       displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        CHECK_CUDA(cudaMemcpy(ws.dPiv, ws.hRecv, dataDouble * w * ldp, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.dRowOff, ws.hRowOff, sizeof(int) * w, cudaMemcpyHostToDevice));

        // 3. External partial sums of the panel diagonal block, then factor it on host.
        {
            dim3 block(16, 16);
            dim3 grid((w + 15) / 16, (w + 15) / 16);
            kPivotDiag<<<grid, block>>>(ws.dPiv, ws.dRowOff, j0, w, ws.dAccD);
            CHECK_CUDA(cudaGetLastError());
        }
        CHECK_CUDA(cudaMemcpy(ws.hAccD, ws.dAccD, dataDouble * w * w, cudaMemcpyDeviceToHost));
        if (!factorPanel(ws.hRecv, ws.hRowOff, static_cast<size_t>(j0), static_cast<size_t>(w),
                         ws.hAccD, ws.hLp, ws.hLdiag, rank)) {
            return false;
        }
        CHECK_CUDA(cudaMemcpy(ws.dLp, ws.hLp, dataDouble * w * w, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.dLdiag, ws.hLdiag, dataDouble * w, cudaMemcpyHostToDevice));

        // 4. Rows below the panel (global index >= j1), owned locally.
        const int rStart = (j1 > rank) ? (j1 - rank + size - 1) / size : 0;
        if (rStart < nl) {
            const int rows = nl - rStart;
            dim3 block(32, 8);
            dim3 grid((rows + 31) / 32, (w + 7) / 8);
            kExternal<<<grid, block>>>(dL, nl, rStart, ws.dPiv, ws.dRowOff, j0, w, ws.dAccE);
            CHECK_CUDA(cudaGetLastError());

            kPanelSolve<<<(rows + 127) / 128, 128>>>(dL, nl, rStart, j0, w, ws.dAccE, ws.dLp,
                                                     ws.dLdiag);
            CHECK_CUDA(cudaGetLastError());
        }

        // 5. Panel rows owned locally: write the factored panel block.
        kWritePanelBlock<<<w, w>>>(dL, nl, j0, w, rank, size, ws.dLp);
        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    return true;
}

// Tiled kernel computing this rank's rows of A = B * B^T + n * I.
// Output is column-major local storage: element (local row r, column j) at L[j * nLoc + r].
// Every element is accumulated over k in ascending order, as in the sequential code.
constexpr int kTile = 32;

__global__ void kGenerate(const double* __restrict__ B, int n, int rank, int size, int nLoc,
                          double* __restrict__ L) {
    __shared__ double As[kTile][kTile + 1];  // As[local row][k]
    __shared__ double Bs[kTile][kTile + 1];  // Bs[column][k]

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int r = blockIdx.x * kTile + tx;  // local row computed by this thread
    const int j = blockIdx.y * kTile + ty;  // column computed by this thread

    const int rowA = blockIdx.x * kTile + ty;  // row loaded by this thread
    const int colB = blockIdx.y * kTile + ty;  // column loaded by this thread
    const bool validA = rowA < nLoc;
    const bool validB = colB < n;
    const size_t iGlobal = static_cast<size_t>(rank) + static_cast<size_t>(validA ? rowA : 0) * size;

    double sum = 0.0;
    for (int k0 = 0; k0 < n; k0 += kTile) {
        const int k = k0 + tx;
        As[ty][tx] = (validA && k < n) ? B[iGlobal * n + k] : 0.0;
        Bs[ty][tx] = (validB && k < n) ? B[static_cast<size_t>(colB) * n + k] : 0.0;
        __syncthreads();

        const int kmax = min(kTile, n - k0);
        for (int t = 0; t < kmax; ++t) {
            sum += As[tx][t] * Bs[ty][t];
        }
        __syncthreads();
    }

    if (r < nLoc && j < n) {
        const int iRow = rank + r * size;
        if (j == iRow) sum += static_cast<double>(n);  // diagonal dominance
        L[static_cast<size_t>(j) * nLoc + r] = sum;
    }
}

// Generates this rank's rows of A = B * B^T + n * I directly into dL (column-major).
// B is pseudo-random and generated sequentially on the host, identically on every rank
// (same sequence as the sequential code), then uploaded for the GPU product.
void generatePositiveDefiniteMatrix(double* dL, size_t n, int rank, int size, size_t nLoc) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    double* dB = nullptr;
    CHECK_CUDA(cudaMalloc(&dB, sizeof(double) * std::max<size_t>(n * n, 1)));
    CHECK_CUDA(cudaMemcpy(dB, B.data(), sizeof(double) * n * n, cudaMemcpyHostToDevice));
    std::vector<double>().swap(B);

    if (nLoc > 0 && n > 0) {
        dim3 block(kTile, kTile);
        dim3 grid(static_cast<unsigned>((nLoc + kTile - 1) / kTile),
                  static_cast<unsigned>((n + kTile - 1) / kTile));
        kGenerate<<<grid, block>>>(dB, static_cast<int>(n), rank, size, static_cast<int>(nLoc), dL);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(dB);
}

// Distributed validation: reconstruct L * L^T for local rows and compare with A.
// Row blocks of L circulate around a ring so each rank sees every block of L once.
// Lrow: local rows of L, row-major. Aorig: local rows of A, column-major (j * nLoc + r).
bool validateCholesky(const std::vector<double>& Lrow, const std::vector<double>& Aorig,
                      size_t n, int rank, int size, size_t nLoc) {
    const size_t maxLoc = localRowCount(n, 0, size);
    std::vector<double> cur(maxLoc * n, 0.0);
    std::vector<double> nxt(maxLoc * n, 0.0);
    std::copy(Lrow.begin(), Lrow.end(), cur.begin());

    double maxError = 0.0;
    double relError = 0.0;

    for (int step = 0; step < size; ++step) {
        const int owner = (rank - step + size) % size;
        const size_t cnt = localRowCount(n, owner, size);

#pragma omp parallel for schedule(dynamic) reduction(max : maxError) reduction(max : relError)
        for (long long rr = 0; rr < static_cast<long long>(nLoc); ++rr) {
            const size_t r = static_cast<size_t>(rr);
            const double* li = &Lrow[r * n];
            for (size_t s = 0; s < cnt; ++s) {
                const size_t j = static_cast<size_t>(owner) + s * static_cast<size_t>(size);
                const double* lj = &cur[s * n];
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) sum += li[k] * lj[k];
                const double orig = Aorig[j * nLoc + r];
                const double error = fabs(sum - orig);
                maxError = std::max(maxError, error);
                const double rel = error / (fabs(orig) + 1e-10);
                relError = std::max(relError, rel);
            }
        }

        if (size > 1 && step + 1 < size) {
            const int src = (rank - step - 1 + size) % size;
            const size_t nextCnt = localRowCount(n, src, size);
            MPI_Sendrecv(cur.data(), static_cast<int>(cnt * n), MPI_DOUBLE, (rank + 1) % size, 0,
                         nxt.data(), static_cast<int>(nextCnt * n), MPI_DOUBLE,
                         (rank - 1 + size) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::swap(cur, nxt);
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, &maxError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &relError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if (relError > 1e-6) {
        if (rank == 0) printf("Validation failed: relative error too large\n");
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

int runBenchmark(int argc, char** argv, int rank, int size) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Bind each rank to a GPU (ranks on the same node are numbered locally)
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "Error: no CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CHECK_CUDA(cudaSetDevice(localRank % deviceCount));

    // Divide hardware threads among the ranks sharing this node
    omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));

    const size_t nLoc = localRowCount(n, rank, size);

    // Generate local rows of A directly in device memory (column-major layout:
    // element (local row r, column k) at k * nLoc + r)
    if (rank == 0) printf("Generating positive definite matrix...\n");
    double* dL = nullptr;
    CHECK_CUDA(cudaMalloc(&dL, sizeof(double) * std::max<size_t>(nLoc * n, 1)));
    generatePositiveDefiniteMatrix(dL, n, rank, size, nLoc);

    std::vector<double> A_orig;
    if (validate) {
        A_orig.resize(nLoc * n);  // Save original local rows for validation
        CHECK_CUDA(cudaMemcpy(A_orig.data(), dL, sizeof(double) * nLoc * n, cudaMemcpyDeviceToHost));
    }

    Workspace ws(n, nLoc);

    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const bool success = choleskyDecomposition(dL, n, rank, size, nLoc, ws);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        cudaFree(dL);
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Copy factor back to host and build row-major local rows with zero upper part
    std::vector<double> Lrow(nLoc * n);
    {
        std::vector<double> Lcm(nLoc * n);
        CHECK_CUDA(cudaMemcpy(Lcm.data(), dL, sizeof(double) * nLoc * n, cudaMemcpyDeviceToHost));
#pragma omp parallel for schedule(static)
        for (long long rr = 0; rr < static_cast<long long>(nLoc); ++rr) {
            const size_t r = static_cast<size_t>(rr);
            const size_t i = static_cast<size_t>(rank) + r * static_cast<size_t>(size);
            for (size_t k = 0; k < n; ++k) {
                Lrow[r * n + k] = (k <= i) ? Lcm[k * nLoc + r] : 0.0;
            }
        }
    }
    cudaFree(dL);

    // Gather full L on rank 0 for external-validation output
    if (printResults) {
        const size_t total = n * n;
        std::vector<int> counts(size), displs(size);
        for (int q = 0; q < size; ++q) {
            counts[q] = static_cast<int>(localRowCount(n, q, size) * n);
            displs[q] = (q == 0) ? 0 : displs[q - 1] + counts[q - 1];
        }
        std::vector<double> gathered(rank == 0 ? total : 0);
        MPI_Gatherv(Lrow.data(), static_cast<int>(nLoc * n), MPI_DOUBLE, gathered.data(),
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<double> L(total);
            for (int q = 0; q < size; ++q) {
                const size_t cnt = localRowCount(n, q, size);
                for (size_t s = 0; s < cnt; ++s) {
                    const size_t i = static_cast<size_t>(q) + s * static_cast<size_t>(size);
                    std::copy_n(gathered.begin() + displs[q] + s * n, n, L.begin() + i * n);
                }
            }
            print_results(L, "CholeskyL");
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(Lrow, A_orig, n, rank, size, nLoc);

        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        return valid ? 0 : 1;
    }

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int ret = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return ret;
}
