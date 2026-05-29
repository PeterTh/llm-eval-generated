#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// ── CUDA kernel: 7-point stencil on interior; global boundary copy ─────────
// Does NOT write halo layer — that's handled by MPI exchange.
// Writes interior cells: stencil for true interior, copy for global boundaries.
__global__ void stencilKernel(const Real* __restrict__ input,
                               Real* __restrict__ output,
                               const int lxh, const int lyh, const int lzh,
                               const int gx0, const int gy0, const int gz0,
                               const int gnx, const int gny, const int gnz) {
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int gz = blockIdx.z * blockDim.z + threadIdx.z;

    // Only process interior (non-halo) cells
    if (gx < 1 || gx >= lxh-1 || gy < 1 || gy >= lyh-1 || gz < 1 || gz >= lzh-1) return;

    const int idx = gz * (lxh * lyh) + gy * lxh + gx;
    const int global_x = gx0 + gx - 1;
    const int global_y = gy0 + gy - 1;
    const int global_z = gz0 + gz - 1;

    if (global_x == 0 || global_x == gnx - 1 ||
        global_y == 0 || global_y == gny - 1 ||
        global_z == 0 || global_z == gnz - 1) {
        // Global boundary: copy from input (never modified)
        output[idx] = input[idx];
    } else {
        // True interior: 7-point stencil
        output[idx] = (input[idx]
                     + input[idx - 1] + input[idx + 1]
                     + input[idx - lxh] + input[idx + lxh]
                     + input[idx - lxh * lyh] + input[idx + lxh * lyh]) / 7.0;
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────
inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ── MPI 3-D domain decomposition ─────────────────────────────────────────────
struct Domain {
    // global grid
    size_t nx, ny, nz;
    // number of MPI ranks in each direction
    int px, py, pz;
    // rank coordinates
    int mx, my, mz;
    // local interior size (no halo)
    size_t lx, ly, lz;
    // local size including halo
    size_t lxh, lyh, lzh;
    // global offset of the interior (first interior cell)
    size_t gx0, gy0, gz0;
    // total local cells (with halo)
    size_t localSize;
};

static Domain decompose(size_t nx, size_t ny, size_t nz, int rank, int nrank) {
    Domain d;
    d.nx = nx; d.ny = ny; d.nz = nz;

    // Choose a 3-D MPI decomposition that is as balanced as possible
    int px = 1, py = 1, pz = 1;
    // Greedy: distribute factors of nrank among dimensions that are largest
    auto factors = [](int n) {
        std::vector<int> f;
        for (int p = 2; p <= n; ++p) {
            while (n % p == 0) { f.push_back(p); n /= p; }
        }
        if (n > 1) f.push_back(n);
        return f;
    };
    // Simple heuristic: try to make px*py*pz == nrank with balanced dims
    // Use a brute-force search over factorizations
    int bestScore = -1;
    for (int a = 1; a <= nrank; ++a) {
        if (nrank % a != 0) continue;
        for (int b = 1; b <= nrank / a; ++b) {
            if ((nrank / a) % b != 0) continue;
            int c = nrank / (a * b);
            // score = product of (dim_size / mpi_dim) variance (lower is better)
            double sx = (double)nx / a, sy = (double)ny / b, sz = (double)nz / c;
            double avg = (sx + sy + sz) / 3.0;
            double var = (sx-avg)*(sx-avg) + (sy-avg)*(sy-avg) + (sz-avg)*(sz-avg);
            int score = (int)(1e9 / (var + 1e-6));
            if (score > bestScore) {
                bestScore = score;
                px = a; py = b; pz = c;
            }
        }
    }
    d.px = px; d.py = py; d.pz = pz;

    // Decompose rank into 3-D coordinates
    int r = rank;
    d.mz = r % pz; r /= pz;
    d.my = r % py; r /= py;
    d.mx = r;

    // Compute local interior sizes (floor division with remainder to last rank)
    auto split_dim = [](size_t dim, int nproc, int myrank) -> size_t {
        size_t base = dim / nproc;
        size_t extra = dim % nproc;
        return base + (myrank < (int)extra ? 1 : 0);
    };
    d.lx = split_dim(nx, px, d.mx);
    d.ly = split_dim(ny, py, d.my);
    d.lz = split_dim(nz, pz, d.mz);

    // Global offset of first interior cell
    auto offset = [](size_t dim, int nproc, int myrank) -> size_t {
        size_t off = 0;
        for (int r = 0; r < myrank; ++r) {
            size_t base = dim / nproc;
            size_t extra = dim % nproc;
            off += base + (r < (int)extra ? 1 : 0);
        }
        return off;
    };
    d.gx0 = offset(nx, px, d.mx);
    d.gy0 = offset(ny, py, d.my);
    d.gz0 = offset(nz, pz, d.mz);

    // Local size including 1-cell halo
    d.lxh = d.lx + 2;
    d.lyh = d.ly + 2;
    d.lzh = d.lz + 2;
    d.localSize = d.lxh * d.lyh * d.lzh;

    return d;
}

// Local index within halo-inclusive buffer
inline size_t lidx(size_t lx, size_t ly, size_t lz, size_t lxh, size_t lyh) {
    return lz * (lxh * lyh) + ly * lxh + lx;
}

// ── Initialize local grid (with halo) ────────────────────────────────────────
void initLocalGrid(Real* grid, const Domain& d) {
    for (size_t lz = 0; lz < d.lz + 2; ++lz) {
        for (size_t ly = 0; ly < d.ly + 2; ++ly) {
            for (size_t lx = 0; lx < d.lx + 2; ++lx) {
                size_t gx = d.gx0 + lx - 1;
                size_t gy = d.gy0 + ly - 1;
                size_t gz = d.gz0 + lz - 1;
                if (gx < d.nx && gy < d.ny && gz < d.nz) {
                    size_t gidx = idx3(gx, gy, gz, d.nx, d.ny);
                    grid[lidx(lx, ly, lz, d.lxh, d.lyh)] = (gidx % 19) * 1.0;
                }
            }
        }
    }
}

// ── Halo exchange (non-blocking MPI, OpenMP for packing/unpacking) ───────────
static void exchangeHalos(Real* grid, const Domain& d, MPI_Comm comm) {
    const size_t lxh = d.lxh, lyh = d.lyh, lzh = d.lzh;
    size_t xySize = lxh * lyh, xzSize = lxh * lzh, yzSize = lyh * lzh;

    Real* sbuf[6] = {}, *rbuf[6] = {};
    size_t bsize[6] = {};
    bsize[0] = yzSize; bsize[1] = yzSize;
    bsize[2] = xzSize; bsize[3] = xzSize;
    bsize[4] = xySize; bsize[5] = xySize;
    for (int i = 0; i < 6; ++i) {
        sbuf[i] = static_cast<Real*>(malloc(bsize[i] * sizeof(Real)));
        rbuf[i] = static_cast<Real*>(malloc(bsize[i] * sizeof(Real)));
    }

    // ── Pack (OpenMP parallel sections) ──────────────────────────────────
    #pragma omp parallel sections
    {
        #pragma omp section
        { for (size_t lz = 0; lz < lzh; ++lz) for (size_t ly = 0; ly < lyh; ++ly) sbuf[0][lz*lyh+ly] = grid[lidx(0,ly,lz,lxh,lyh)]; }
        #pragma omp section
        { for (size_t lz = 0; lz < lzh; ++lz) for (size_t ly = 0; ly < lyh; ++ly) sbuf[1][lz*lyh+ly] = grid[lidx(d.lx+1,ly,lz,lxh,lyh)]; }
        #pragma omp section
        { for (size_t lz = 0; lz < lzh; ++lz) for (size_t lx = 0; lx < lxh; ++lx) sbuf[2][lz*lxh+lx] = grid[lidx(lx,0,lz,lxh,lyh)]; }
        #pragma omp section
        { for (size_t lz = 0; lz < lzh; ++lz) for (size_t lx = 0; lx < lxh; ++lx) sbuf[3][lz*lxh+lx] = grid[lidx(lx,d.ly+1,lz,lxh,lyh)]; }
        #pragma omp section
        { for (size_t ly = 0; ly < lyh; ++ly) for (size_t lx = 0; lx < lxh; ++lx) sbuf[4][ly*lxh+lx] = grid[lidx(lx,ly,0,lxh,lyh)]; }
        #pragma omp section
        { for (size_t ly = 0; ly < lyh; ++ly) for (size_t lx = 0; lx < lxh; ++lx) sbuf[5][ly*lxh+lx] = grid[lidx(lx,ly,d.lz+1,lxh,lyh)]; }
    }

    // ── Neighbor ranks ───────────────────────────────────────────────────
    int rXn = (d.mx > 0)     ? d.mx-1 + d.my*d.px + d.mz*d.px*d.py : MPI_PROC_NULL;
    int rXp = (d.mx < d.px-1)? d.mx+1 + d.my*d.px + d.mz*d.px*d.py : MPI_PROC_NULL;
    int rYn = (d.my > 0)     ? d.mx + (d.my-1)*d.px + d.mz*d.px*d.py : MPI_PROC_NULL;
    int rYp = (d.my < d.py-1)? d.mx + (d.my+1)*d.px + d.mz*d.px*d.py : MPI_PROC_NULL;
    int rZn = (d.mz > 0)     ? d.mx + d.my*d.px + (d.mz-1)*d.px*d.py : MPI_PROC_NULL;
    int rZp = (d.mz < d.pz-1)? d.mx + d.my*d.px + (d.mz+1)*d.px*d.py : MPI_PROC_NULL;

    // Track which faces actually communicated
    int hasRecv[6] = {0};

    // ── Non-blocking MPI ─────────────────────────────────────────────────
    MPI_Request reqs[12];
    int nreq = 0;

    auto post = [&](int face, Real* sb, Real* rb, size_t sz, int sRank, int rRank, int sTag, int rTag) {
        if (sRank != MPI_PROC_NULL)
            MPI_Isend(sb, (int)sz, MPI_DOUBLE, sRank, sTag, comm, &reqs[nreq++]);
        if (rRank != MPI_PROC_NULL) {
            MPI_Irecv(rb, (int)sz, MPI_DOUBLE, rRank, rTag, comm, &reqs[nreq++]);
            hasRecv[face] = 1;
        }
    };

    post(0, sbuf[0], rbuf[0], yzSize, rXn, rXn, 10, 11);
    post(1, sbuf[1], rbuf[1], yzSize, rXp, rXp, 11, 10);
    post(2, sbuf[2], rbuf[2], xzSize, rYn, rYn, 20, 21);
    post(3, sbuf[3], rbuf[3], xzSize, rYp, rYp, 21, 20);
    post(4, sbuf[4], rbuf[4], xySize, rZn, rZn, 30, 31);
    post(5, sbuf[5], rbuf[5], xySize, rZp, rZp, 31, 30);

    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    // ── Unpack only faces that received data (OpenMP parallel sections) ──
    #pragma omp parallel sections
    {
        #pragma omp section
        { if (hasRecv[0]) for (size_t lz = 0; lz < lzh; ++lz) for (size_t ly = 0; ly < lyh; ++ly) grid[lidx(1,ly,lz,lxh,lyh)] = rbuf[0][lz*lyh+ly]; }
        #pragma omp section
        { if (hasRecv[1]) for (size_t lz = 0; lz < lzh; ++lz) for (size_t ly = 0; ly < lyh; ++ly) grid[lidx(d.lx,ly,lz,lxh,lyh)] = rbuf[1][lz*lyh+ly]; }
        #pragma omp section
        { if (hasRecv[2]) for (size_t lz = 0; lz < lzh; ++lz) for (size_t lx = 0; lx < lxh; ++lx) grid[lidx(lx,1,lz,lxh,lyh)] = rbuf[2][lz*lxh+lx]; }
        #pragma omp section
        { if (hasRecv[3]) for (size_t lz = 0; lz < lzh; ++lz) for (size_t lx = 0; lx < lxh; ++lx) grid[lidx(lx,d.ly,lz,lxh,lyh)] = rbuf[3][lz*lxh+lx]; }
        #pragma omp section
        { if (hasRecv[4]) for (size_t ly = 0; ly < lyh; ++ly) for (size_t lx = 0; lx < lxh; ++lx) grid[lidx(lx,ly,1,lxh,lyh)] = rbuf[4][ly*lxh+lx]; }
        #pragma omp section
        { if (hasRecv[5]) for (size_t ly = 0; ly < lyh; ++ly) for (size_t lx = 0; lx < lxh; ++lx) grid[lidx(lx,ly,d.lz,lxh,lyh)] = rbuf[5][ly*lxh+lx]; }
    }

    for (int i = 0; i < 6; ++i) { free(sbuf[i]); free(rbuf[i]); }
}

// ── CUDA stencil iteration (persistent device memory) ────────────────────────
// Device buffers are allocated once and reused across iterations
static Real* g_d_in = nullptr;
static Real* g_d_out = nullptr;
static size_t g_devSize = 0;

static void ensureDeviceMem(size_t need) {
    if (g_d_in && g_devSize >= need) return;
    cudaFree(g_d_in); cudaFree(g_d_out);
    g_d_in = nullptr; g_d_out = nullptr;
    cudaMalloc(&g_d_in, need * sizeof(Real));
    cudaMalloc(&g_d_out, need * sizeof(Real));
    g_devSize = need;
}

static void cleanupDeviceMem() {
    cudaFree(g_d_in); cudaFree(g_d_out);
    g_d_in = nullptr; g_d_out = nullptr; g_devSize = 0;
}

void stencilIterationGPU(Real* input, Real* output, const Domain& d) {
    const size_t lxh = d.lxh, lyh = d.lyh, lzh = d.lzh;
    const size_t localSize = d.localSize;

    // Fallback for small subdomains
    if (d.lx < 2 || d.ly < 2 || d.lz < 2) {
        #pragma omp parallel for collapse(3) schedule(static)
        for (size_t lz = 0; lz < lzh; ++lz)
            for (size_t ly = 0; ly < lyh; ++ly)
                for (size_t lx = 0; lx < lxh; ++lx)
                    output[lidx(lx, ly, lz, lxh, lyh)] = input[lidx(lx, ly, lz, lxh, lyh)];
        return;
    }

    ensureDeviceMem(localSize);

    dim3 blockSize(8, 8, 4);
    dim3 gridSize(
        (lxh + blockSize.x - 1) / blockSize.x,
        (lyh + blockSize.y - 1) / blockSize.y,
        (lzh + blockSize.z - 1) / blockSize.z
    );

    cudaError_t err = cudaMemcpy(g_d_in, input, localSize * sizeof(Real), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "H2D: %s\n", cudaGetErrorString(err)); return; }

    stencilKernel<<<gridSize, blockSize>>>(g_d_in, g_d_out,
        (int)lxh, (int)lyh, (int)lzh,
        (int)d.gx0, (int)d.gy0, (int)d.gz0,
        (int)d.nx, (int)d.ny, (int)d.nz);
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "Kernel: %s\n", cudaGetErrorString(err)); return; }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "Sync: %s\n", cudaGetErrorString(err)); return; }

    err = cudaMemcpy(output, g_d_out, localSize * sizeof(Real), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "D2H: %s\n", cudaGetErrorString(err)); }
}

// ── Gather full grid on rank 0 ───────────────────────────────────────────────
std::vector<Real> gatherGrid(Real* localGrid, const Domain& d, int rank, int nrank, MPI_Comm comm) {
    size_t totalSize = d.nx * d.ny * d.nz;
    std::vector<Real> fullGrid;

    if (rank == 0) {
        fullGrid.resize(totalSize);
    }

    // Gather counts and displacements
    std::vector<int> counts(nrank), displs(nrank);
    for (int r = 0; r < nrank; ++r) {
        Domain dr = decompose(d.nx, d.ny, d.nz, r, nrank);
        counts[r] = (int)dr.localSize;
    }
    displs[0] = 0;
    for (int r = 1; r < nrank; ++r)
        displs[r] = displs[r-1] + counts[r-1];

    size_t gatheredSize = (size_t)displs[nrank-1] + counts[nrank-1];
    std::vector<Real> gathered;
    if (rank == 0) gathered.resize(gatheredSize);

    MPI_Gatherv(localGrid, (int)d.localSize, MPI_DOUBLE,
                gathered.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, comm);

    if (rank == 0) {
        // Scatter interior (non-halo) data into the global grid
        for (int r = 0; r < nrank; ++r) {
            Domain dr = decompose(d.nx, d.ny, d.nz, r, nrank);
            size_t disp = (size_t)displs[r];
            for (size_t lz = 1; lz < dr.lz + 1; ++lz) {
                for (size_t ly = 1; ly < dr.ly + 1; ++ly) {
                    for (size_t lx = 1; lx < dr.lx + 1; ++lx) {
                        size_t gx = dr.gx0 + lx - 1;
                        size_t gy = dr.gy0 + ly - 1;
                        size_t gz = dr.gz0 + lz - 1;
                        size_t gidx = idx3(gx, gy, gz, d.nx, d.ny);
                        size_t lidx_local = lidx(lx, ly, lz, dr.lxh, dr.lyh);
                        fullGrid[gidx] = gathered[disp + lidx_local];
                    }
                }
            }
        }
        // Fill global boundary cells: these are cells where gx==0, gx==nx-1, etc.
        // They live in the halo of the edge ranks. Use the halo value from the
        // rank whose interior is adjacent to that boundary.
        // For each rank, its halo cells that correspond to global boundaries
        // (i.e., gx==0, gx==nx-1, etc.) are valid.
        for (int r = 0; r < nrank; ++r) {
            Domain dr = decompose(d.nx, d.ny, d.nz, r, nrank);
            size_t disp = (size_t)displs[r];
            for (size_t lz = 0; lz < dr.lz + 2; ++lz) {
                for (size_t ly = 0; ly < dr.ly + 2; ++ly) {
                    for (size_t lx = 0; lx < dr.lx + 2; ++lx) {
                        ssize_t gx = (ssize_t)dr.gx0 + (ssize_t)lx - 1;
                        ssize_t gy = (ssize_t)dr.gy0 + (ssize_t)ly - 1;
                        ssize_t gz = (ssize_t)dr.gz0 + (ssize_t)lz - 1;
                        // Only fill global boundary cells that are in this rank's halo
                        bool isGlobalBoundary = (gx == 0 || gx == (ssize_t)d.nx-1 ||
                                                 gy == 0 || gy == (ssize_t)d.ny-1 ||
                                                 gz == 0 || gz == (ssize_t)d.nz-1);
                        bool isHalo = (lx == 0 || lx == dr.lx + 1 ||
                                       ly == 0 || ly == dr.ly + 1 ||
                                       lz == 0 || lz == dr.lz + 1);
                        if (isGlobalBoundary && isHalo && gx >= 0 && gy >= 0 && gz >= 0 &&
                            gx < (ssize_t)d.nx && gy < (ssize_t)d.ny && gz < (ssize_t)d.nz) {
                            size_t gidx = idx3((size_t)gx, (size_t)gy, (size_t)gz, d.nx, d.ny);
                            size_t lidx_local = lidx(lx, ly, lz, dr.lxh, dr.lyh);
                            fullGrid[gidx] = gathered[disp + lidx_local];
                        }
                    }
                }
            }
        }
    }

    return fullGrid;
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    Real minVal = grid[0], maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, nrank = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);

    int cudaDev = 0;
    cudaSetDevice(cudaDev);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d | OpenMP threads: %d\n", nrank, omp_get_max_threads());
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    Domain d = decompose(nx, ny, nz, rank, nrank);

    // Allocate local grids (double buffering, halo-inclusive)
    Real* grid1 = static_cast<Real*>(aligned_alloc(64, d.localSize * sizeof(Real)));
    Real* grid2 = static_cast<Real*>(aligned_alloc(64, d.localSize * sizeof(Real)));
    std::memset(grid1, 0, d.localSize * sizeof(Real));
    std::memset(grid2, 0, d.localSize * sizeof(Real));

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initLocalGrid(grid1, d);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real* input, *output;
        if (iter % 2 == 0) { input = grid1; output = grid2; }
        else               { input = grid2; output = grid1; }

        // CUDA stencil on interior + OpenMP boundary copy
        stencilIterationGPU(input, output, d);

        // Exchange halos via MPI
        exchangeHalos(output, d, MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather final grid on rank 0 for reporting
    const Real* finalLocal = (iterations % 2 == 0) ? grid2 : grid1;
    std::vector<Real> fullGrid = gatherGrid(const_cast<Real*>(finalLocal), d, rank, nrank, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(fullGrid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(fullGrid, nx, ny, nz);
            if (valid) printf("Validation: PASSED\n");
            else       printf("Validation: FAILED\n");
        }
    }

    free(grid1);
    free(grid2);
    MPI_Finalize();
    return 0;
}
