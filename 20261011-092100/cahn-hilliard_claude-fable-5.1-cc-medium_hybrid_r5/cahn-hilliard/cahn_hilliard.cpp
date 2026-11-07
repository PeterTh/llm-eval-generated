// Cahn-Hilliard phase separation benchmark
//
// Hybrid parallelization:
//   * MPI    : 1D slab decomposition of the 3D grid along Z, one rank per GPU.
//              A depth-2 halo of c is exchanged once per time step (per side,
//              independently) and overlapped with the interior stencil work.
//   * CUDA   : the chemical-potential and concentration-update stencils run on
//              the GPU (register-marching 7-point stencil kernels, one compute
//              stream plus one copy stream per halo side).
//   * OpenMP : host-side work (field initialization, validation reductions)
//              is multithreaded.
//
// Semantics (initial field, stencil, clamped boundaries, update, output) are
// identical to the original serial implementation.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Error handling helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t err__ = (call);                                                             \
        if (err__ != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,          \
                    cudaGetErrorString(err__));                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

#define MPI_CHECK(call)                                                                         \
    do {                                                                                        \
        int err__ = (call);                                                                     \
        if (err__ != MPI_SUCCESS) {                                                             \
            fprintf(stderr, "MPI error %s at %s:%d (code %d)\n", #call, __FILE__, __LINE__,     \
                    err__);                                                                     \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// CUDA kernels
//
// Local storage layout on each rank: planes 0 .. nzl+3. Planes 0,1 and
// nzl+2,nzl+3 are depth-2 halo planes received from the neighbouring ranks and
// planes 2 .. nzl+1 hold the rank's own slab (local plane k is stored at k+2).
// The chemical potential is additionally computed on the first halo plane of
// each side (local planes -1 and nzl) so that a single exchange of c per time
// step suffices for both stencil passes.
// ---------------------------------------------------------------------------
namespace {

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;
constexpr int Z_CHUNK = 16;  // planes marched per thread along z
constexpr int HALO = 2;      // halo depth in z (planes per side)

struct StencilParams {
    int nx, ny, nzl;          // local slab dimensions (nzl = own planes)
    int has_prev, has_next;   // 1 if a neighbour rank exists in that z direction
    double idx2, idy2, idz2;  // 1/dx^2, 1/dy^2, 1/dz^2
};

// 7-point Laplacian contribution helper. When UNIT_SPACING is set the grid
// spacing is exactly 1.0, so the (exact) multiplications by 1/h^2 are skipped.
template <bool UNIT_SPACING>
__device__ __forceinline__ double scaleBy(const double v, const double inv_h2) {
    return UNIT_SPACING ? v : v * inv_h2;
}

// Chemical potential: mu = f'(c) - gamma * lap(c) on local planes [z0, z1).
// Each thread owns one (x, y) column and marches along z keeping the three
// z-planes of the centre column in registers; x/y neighbours come from the
// (L1-cached) global loads of the same plane.
template <bool UNIT_SPACING>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                        const StencilParams p, const int z0, const int z1,
                        const double gamma, const double e_AA, const double e_BB,
                        const double e_AB) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= p.nx || y >= p.ny) return;

    const int zbeg = z0 + blockIdx.z * Z_CHUNK;
    const int zend = min(zbeg + Z_CHUNK, z1);
    if (zbeg >= zend) return;

    const int dxp = (x < p.nx - 1) ? 1 : 0;
    const int dxn = (x > 0) ? -1 : 0;
    const int dyp = (y < p.ny - 1) ? p.nx : 0;
    const int dyn = (y > 0) ? -p.nx : 0;
    const size_t plane = static_cast<size_t>(p.nx) * p.ny;
    const size_t off = static_cast<size_t>(y) * p.nx + x;

    // storage plane of first cell and of its lower z neighbour (clamped)
    const int zs0 = zbeg + HALO;
    const double* row = c + static_cast<size_t>(zs0) * plane + off;
    double* out = mu + static_cast<size_t>(zs0) * plane + off;
    double below = (zbeg > 0 || p.has_prev) ? row[-static_cast<ptrdiff_t>(plane)] : row[0];
    double cv = row[0];

    for (int zl = zbeg; zl < zend; ++zl) {
        const bool hasAbove = (zl < p.nzl - 1 || p.has_next);
        const double above = hasAbove ? row[plane] : cv;

        const double cxx = scaleBy<UNIT_SPACING>(row[dxp] + row[dxn] - 2.0 * cv, p.idx2);
        const double cyy = scaleBy<UNIT_SPACING>(row[dyp] + row[dyn] - 2.0 * cv, p.idy2);
        const double czz = scaleBy<UNIT_SPACING>(above + below - 2.0 * cv, p.idz2);

        *out = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
               + 3.0 * cv + cv * cv * cv
               - gamma * (cxx + cyy + czz);

        below = cv;
        cv = above;
        row += plane;
        out += plane;
    }
}

// Concentration update: cnew = cold + dt*D*lap(mu) on local planes [z0, z1)
template <bool UNIT_SPACING>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
             const double* __restrict__ mu, const StencilParams p, const int z0, const int z1,
             const double dtD) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= p.nx || y >= p.ny) return;

    const int zbeg = z0 + blockIdx.z * Z_CHUNK;
    const int zend = min(zbeg + Z_CHUNK, z1);
    if (zbeg >= zend) return;

    const int dxp = (x < p.nx - 1) ? 1 : 0;
    const int dxn = (x > 0) ? -1 : 0;
    const int dyp = (y < p.ny - 1) ? p.nx : 0;
    const int dyn = (y > 0) ? -p.nx : 0;
    const size_t plane = static_cast<size_t>(p.nx) * p.ny;
    const size_t off = static_cast<size_t>(y) * p.nx + x;

    const int zs0 = zbeg + HALO;
    const size_t base = static_cast<size_t>(zs0) * plane + off;
    const double* row = mu + base;
    const double* in = cold + base;
    double* out = cnew + base;
    double below = (zbeg > 0 || p.has_prev) ? row[-static_cast<ptrdiff_t>(plane)] : row[0];
    double mv = row[0];

    for (int zl = zbeg; zl < zend; ++zl) {
        const bool hasAbove = (zl < p.nzl - 1 || p.has_next);
        const double above = hasAbove ? row[plane] : mv;

        const double mxx = scaleBy<UNIT_SPACING>(row[dxp] + row[dxn] - 2.0 * mv, p.idx2);
        const double myy = scaleBy<UNIT_SPACING>(row[dyp] + row[dyn] - 2.0 * mv, p.idy2);
        const double mzz = scaleBy<UNIT_SPACING>(above + below - 2.0 * mv, p.idz2);

        *out = *in + dtD * (mxx + myy + mzz);

        below = mv;
        mv = above;
        row += plane;
        in += plane;
        out += plane;
    }
}

inline dim3 gridFor(const StencilParams& p, const int z0, const int z1) {
    const int nz = std::max(0, z1 - z0);
    return dim3((p.nx + BLOCK_X - 1) / BLOCK_X, (p.ny + BLOCK_Y - 1) / BLOCK_Y,
                (nz + Z_CHUNK - 1) / Z_CHUNK);
}

}  // namespace

// ---------------------------------------------------------------------------
// Distributed simulation state
// ---------------------------------------------------------------------------
struct Domain {
    int rank = 0, size = 1;
    size_t nx = 0, ny = 0, nz = 0;
    int nzl = 0;         // own planes on this rank
    size_t z_off = 0;    // global index of first own plane
    int prev = MPI_PROC_NULL, next = MPI_PROC_NULL;
    std::vector<int> counts, displs;  // planes per rank / first plane (for gather)

    size_t plane() const { return nx * ny; }
    size_t localElems() const { return plane() * nzl; }
    size_t storageElems() const { return plane() * (nzl + 2 * HALO); }
};

static Domain makeDomain(const size_t nx, const size_t ny, const size_t nz) {
    Domain d;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &d.rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &d.size));
    d.nx = nx; d.ny = ny; d.nz = nz;
    d.counts.resize(d.size);
    d.displs.resize(d.size);
    // Every active rank owns at least HALO planes so that the depth-2 halo of a
    // neighbour is always made of planes owned by that neighbour.
    const int active = (nz >= static_cast<size_t>(HALO))
                           ? static_cast<int>(std::min<size_t>(d.size, nz / HALO)) : 1;
    size_t acc = 0;
    for (int r = 0; r < d.size; ++r) {
        size_t cnt = 0;
        if (r < active) cnt = nz / active + (static_cast<size_t>(r) < nz % active ? 1 : 0);
        d.counts[r] = static_cast<int>(cnt);
        d.displs[r] = static_cast<int>(acc);
        acc += cnt;
    }
    d.nzl = d.counts[d.rank];
    d.z_off = d.displs[d.rank];
    // Nearest neighbours that actually own planes (ranks with 0 planes are skipped)
    for (int r = d.rank - 1; r >= 0; --r) if (d.counts[r] > 0) { d.prev = r; break; }
    for (int r = d.rank + 1; r < d.size; ++r) if (d.counts[r] > 0) { d.next = r; break; }
    return d;
}

// Halo exchange of the first / last HALO own planes of a field with the
// neighbouring ranks. The two sides are handled independently so that the
// boundary computation of one side can start while the other side is still in
// flight. D2H/H2D copies are staged through pinned host buffers; a side's
// copies run on its own CUDA stream.
struct HaloExchanger {
    enum Side { LO = 0, HI = 1 };
    const Domain* d = nullptr;
    double* h_send[2] = {nullptr, nullptr};  // own boundary planes -> neighbour
    double* h_recv[2] = {nullptr, nullptr};  // neighbour planes -> halo planes
    MPI_Request recvReq[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Request sendReq[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    void init(const Domain& dom) {
        d = &dom;
        const size_t bytes = HALO * dom.plane() * sizeof(double);
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaHostAlloc(&h_send[i], bytes, cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc(&h_recv[i], bytes, cudaHostAllocDefault));
        }
    }
    void destroy() {
        for (int i = 0; i < 2; ++i) { cudaFreeHost(h_send[i]); cudaFreeHost(h_recv[i]); }
    }

    int neighbour(Side s) const { return s == LO ? d->prev : d->next; }
    bool active(Side s) const { return d->nzl > 0 && neighbour(s) != MPI_PROC_NULL; }
    size_t bytes() const { return HALO * d->plane() * sizeof(double); }
    int count() const { return static_cast<int>(HALO * d->plane()); }
    // storage offset of the own planes sent to / halo planes received from side s
    size_t sendOffset(Side s) const { return s == LO ? HALO * d->plane() : d->plane() * d->nzl; }
    size_t recvOffset(Side s) const { return s == LO ? 0 : d->plane() * (d->nzl + HALO); }

    // Start staging the own boundary planes to the host.
    void start(Side s, const double* dev_field, cudaStream_t stream) {
        if (!active(s)) return;
        CUDA_CHECK(cudaMemcpyAsync(h_send[s], dev_field + sendOffset(s), bytes(),
                                   cudaMemcpyDeviceToHost, stream));
    }
    // Wait for the staging copy, then post the receive and send the planes.
    // The receive is posted only here: the stream synchronisation guarantees
    // that the previous step's H2D copy out of h_recv (ordered earlier on the
    // same stream) has completed, so MPI may overwrite the buffer.
    void send(Side s, cudaStream_t stream) {
        if (!active(s)) return;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_CHECK(MPI_Irecv(h_recv[s], count(), MPI_DOUBLE, neighbour(s), 1 - s, MPI_COMM_WORLD,
                            &recvReq[s]));
        MPI_CHECK(MPI_Isend(h_send[s], count(), MPI_DOUBLE, neighbour(s), s, MPI_COMM_WORLD,
                            &sendReq[s]));
    }
    // Wait for the neighbour's planes and copy them into the halo.
    void receive(Side s, double* dev_field, cudaStream_t stream) {
        if (!active(s)) return;
        MPI_CHECK(MPI_Wait(&recvReq[s], MPI_STATUS_IGNORE));
        CUDA_CHECK(cudaMemcpyAsync(dev_field + recvOffset(s), h_recv[s], bytes(),
                                   cudaMemcpyHostToDevice, stream));
    }
    void finish() {
        MPI_CHECK(MPI_Waitall(2, sendReq, MPI_STATUSES_IGNORE));
    }
};

// Initialize concentration field (own slab only, same formula as the original)
void initializeConcentration(std::vector<double>& c, const Domain& d) {
    const size_t nx = d.nx, ny = d.ny;
    const size_t vol = nx * ny * d.nz;

#pragma omp parallel for collapse(2) schedule(static)
    for (int zl = 0; zl < d.nzl; ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t z = d.z_off + zl;
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, static_cast<size_t>(zl + HALO), nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    bool bad = false;
#pragma omp parallel for reduction(|| : bad) schedule(static)
    for (size_t i = 0; i < c.size(); ++i) {
        bad = bad || std::isnan(c[i]) || std::isinf(c[i]);
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (size_t i = 0; i < c.size(); ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static void selectDevice() {
    MPI_Comm local;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local));
    int local_rank = 0;
    MPI_CHECK(MPI_Comm_rank(local, &local_rank));
    MPI_CHECK(MPI_Comm_free(&local));
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % ndev));
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    int rank = 0, nranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
    const bool isRoot = (rank == 0);

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    selectDevice();

    if (isRoot) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: MPI ranks: %d, OpenMP threads/rank: %d, CUDA device per rank\n",
               nranks, omp_get_max_threads());
    }

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    const size_t gridSize = nx * ny * nz;

    const Domain dom = makeDomain(nx, ny, nz);
    const size_t plane = dom.plane();

    // Allocate arrays (local slab + HALO halo planes on each side)
    std::vector<double> h_c(dom.storageElems(), 0.0);
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    const size_t storageBytes = std::max<size_t>(dom.storageElems(), 1) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_cold, storageBytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, storageBytes));
    CUDA_CHECK(cudaMalloc(&d_mu, storageBytes));
    CUDA_CHECK(cudaMemset(d_cold, 0, storageBytes));
    CUDA_CHECK(cudaMemset(d_cnew, 0, storageBytes));
    CUDA_CHECK(cudaMemset(d_mu, 0, storageBytes));

    // Initialize concentration field
    if (isRoot) printf("Initializing concentration field...\n");
    initializeConcentration(h_c, dom);
    if (dom.nzl > 0)
        CUDA_CHECK(cudaMemcpy(d_cold, h_c.data(), dom.storageElems() * sizeof(double),
                              cudaMemcpyHostToDevice));

    cudaStream_t compStream, commStream[2];
    CUDA_CHECK(cudaStreamCreateWithFlags(&compStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&commStream[0], cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&commStream[1], cudaStreamNonBlocking));
    cudaEvent_t evBoundaryDone, evHaloDone[2];
    CUDA_CHECK(cudaEventCreateWithFlags(&evBoundaryDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evHaloDone[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evHaloDone[1], cudaEventDisableTiming));

    HaloExchanger halo;
    halo.init(dom);

    StencilParams sp;
    sp.nx = static_cast<int>(nx);
    sp.ny = static_cast<int>(ny);
    sp.nzl = dom.nzl;
    sp.has_prev = (dom.prev != MPI_PROC_NULL) ? 1 : 0;
    sp.has_next = (dom.next != MPI_PROC_NULL) ? 1 : 0;
    sp.idx2 = 1.0 / (dx * dx);
    sp.idy2 = 1.0 / (dy * dy);
    sp.idz2 = 1.0 / (dz * dz);
    const double dtD = dt * D;

    // Plane ranges (local z). The interior ranges only touch own planes and are
    // computed while the halo exchange is in flight; the boundary ranges need
    // the halos. mu is also computed on the first halo plane of each side.
    const int nzl = dom.nzl;
    const int hp = sp.has_prev, hn = sp.has_next;
    struct Range { int z0, z1; };
    const Range muInterior{1, std::max(1, nzl - 1)};
    const Range upInterior{2, std::max(2, nzl - 2)};
    // Boundary ranges per side; sideWork[LO] can run once the lower halo has
    // arrived, sideWork[HI] once both have. For small slabs the lower and
    // upper boundary regions overlap and are computed together after both.
    struct SideWork { std::vector<Range> mu, up; };
    SideWork sideWork[2];
    if (nzl > 4) {
        sideWork[0].mu.push_back({-hp, 1});
        sideWork[0].up.push_back({0, 2});
        sideWork[1].mu.push_back({nzl - 1, nzl + hn});
        sideWork[1].up.push_back({nzl - 2, nzl});
    } else if (nzl > 0) {
        if (nzl - 1 <= 1) sideWork[1].mu.push_back({-hp, nzl + hn});
        else { sideWork[1].mu.push_back({-hp, 1}); sideWork[1].mu.push_back({nzl - 1, nzl + hn}); }
        sideWork[1].up.push_back({0, nzl});
    }
    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Grid spacing of exactly 1.0 lets the kernels skip the (exact) scaling
    // multiplications of the Laplacian terms.
    const bool unitSpacing = (sp.idx2 == 1.0 && sp.idy2 == 1.0 && sp.idz2 == 1.0);

    auto launchMu = [&](int z0, int z1) {
        if (z1 <= z0) return;
        const dim3 grid = gridFor(sp, z0, z1);
        if (unitSpacing)
            chemicalPotentialKernel<true><<<grid, block, 0, compStream>>>(
                d_cold, d_mu, sp, z0, z1, gamma, e_AA, e_BB, e_AB);
        else
            chemicalPotentialKernel<false><<<grid, block, 0, compStream>>>(
                d_cold, d_mu, sp, z0, z1, gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());
    };
    auto launchUpdate = [&](int z0, int z1) {
        if (z1 <= z0) return;
        const dim3 grid = gridFor(sp, z0, z1);
        if (unitSpacing)
            updateKernel<true><<<grid, block, 0, compStream>>>(d_cnew, d_cold, d_mu, sp, z0, z1, dtD);
        else
            updateKernel<false><<<grid, block, 0, compStream>>>(d_cnew, d_cold, d_mu, sp, z0, z1, dtD);
        CUDA_CHECK(cudaGetLastError());
    };

    // Run simulation
    if (isRoot) printf("Running Cahn-Hilliard simulation...\n");
    // Warm up the GPU (module load, clocks) with empty-range launches so that
    // one-time start-up costs are not attributed to the timed loop.
    if (nzl > 0) {
        launchMu(0, 1);
        launchUpdate(0, 1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    using Side = HaloExchanger::Side;
    for (int t = 0; t < iterations; ++t) {
        // Start the halo exchange of c once the previous step has completed.
        CUDA_CHECK(cudaEventRecord(evBoundaryDone, compStream));
        for (int sd = 0; sd < 2; ++sd) {
            CUDA_CHECK(cudaStreamWaitEvent(commStream[sd], evBoundaryDone, 0));
            halo.start(static_cast<Side>(sd), d_cold, commStream[sd]);
        }

        // Interior chemical potential and interior update overlap the exchange.
        launchMu(muInterior.z0, muInterior.z1);
        launchUpdate(upInterior.z0, upInterior.z1);

        for (int sd = 0; sd < 2; ++sd) halo.send(static_cast<Side>(sd), commStream[sd]);

        // Boundary planes of each side as soon as its halo has arrived
        // (including mu on the first halo plane of that side).
        for (int sd = 0; sd < 2; ++sd) {
            halo.receive(static_cast<Side>(sd), d_cold, commStream[sd]);
            CUDA_CHECK(cudaEventRecord(evHaloDone[sd], commStream[sd]));
            CUDA_CHECK(cudaStreamWaitEvent(compStream, evHaloDone[sd], 0));
            for (const Range& r : sideWork[sd].mu) launchMu(r.z0, r.z1);
            for (const Range& r : sideWork[sd].up) launchUpdate(r.z0, r.z1);
        }
        halo.finish();

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaStreamSynchronize(compStream));
    CUDA_CHECK(cudaStreamSynchronize(commStream[0]));
    CUDA_CHECK(cudaStreamSynchronize(commStream[1]));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count();
    long global_ms = 0;
    MPI_CHECK(MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD));

    if (isRoot) {
        printf("Computation time: %ld ms\n", global_ms);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (global_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int exitCode = 0;
    if (printResults || validate) {
        // Gather the full field on rank 0 (own planes only, halos stripped)
        if (dom.nzl > 0)
            CUDA_CHECK(cudaMemcpy(h_c.data() + HALO * plane, d_cold + HALO * plane,
                                  dom.localElems() * sizeof(double), cudaMemcpyDeviceToHost));

        std::vector<double> full;
        std::vector<int> rcounts(nranks), rdispls(nranks);
        if (isRoot) full.resize(gridSize);
        // Gatherv counts/displacements in elements; fall back to a
        // plane-granular datatype when they would overflow int.
        bool fitsInt = true;
        for (int r = 0; r < nranks; ++r) {
            const size_t cnt = static_cast<size_t>(dom.counts[r]) * plane;
            const size_t dsp = static_cast<size_t>(dom.displs[r]) * plane;
            if (cnt > static_cast<size_t>(INT32_MAX) || dsp > static_cast<size_t>(INT32_MAX)) fitsInt = false;
            rcounts[r] = static_cast<int>(cnt);
            rdispls[r] = static_cast<int>(dsp);
        }
        if (fitsInt) {
            MPI_CHECK(MPI_Gatherv(h_c.data() + HALO * plane, static_cast<int>(dom.localElems()), MPI_DOUBLE,
                                  isRoot ? full.data() : nullptr, rcounts.data(), rdispls.data(),
                                  MPI_DOUBLE, 0, MPI_COMM_WORLD));
        } else {
            MPI_Datatype planeType;
            MPI_CHECK(MPI_Type_contiguous(static_cast<int>(plane), MPI_DOUBLE, &planeType));
            MPI_CHECK(MPI_Type_commit(&planeType));
            MPI_CHECK(MPI_Gatherv(h_c.data() + HALO * plane, dom.nzl, planeType,
                                  isRoot ? full.data() : nullptr, dom.counts.data(),
                                  dom.displs.data(), planeType, 0, MPI_COMM_WORLD));
            MPI_CHECK(MPI_Type_free(&planeType));
        }

        if (isRoot) {
            // Print results for external validation
            if (printResults) {
                print_results(full, "Concentration");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(full, nx, ny, nz);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
        MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }

    halo.destroy();
    CUDA_CHECK(cudaEventDestroy(evBoundaryDone));
    CUDA_CHECK(cudaEventDestroy(evHaloDone[0]));
    CUDA_CHECK(cudaEventDestroy(evHaloDone[1]));
    CUDA_CHECK(cudaStreamDestroy(compStream));
    CUDA_CHECK(cudaStreamDestroy(commStream[0]));
    CUDA_CHECK(cudaStreamDestroy(commStream[1]));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    MPI_Finalize();
    return exitCode;
}
