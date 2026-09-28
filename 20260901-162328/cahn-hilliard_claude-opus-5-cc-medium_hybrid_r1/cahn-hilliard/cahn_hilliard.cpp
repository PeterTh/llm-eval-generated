// Cahn-Hilliard phase separation benchmark.
//
// Hybrid parallelization: MPI (1D domain decomposition along Z, one rank per GPU)
// + CUDA (all stencil work runs on the device) + OpenMP (host-side field setup,
// halo staging and validation reductions).

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

#if defined(__has_include)
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#endif

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        const cudaError_t err_ = (call);                                                                               \
        if (err_ != cudaSuccess) {                                                                                     \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__);                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                              \
        }                                                                                                              \
    } while (0)

// Thread block / register blocking configuration for the marching stencil kernels.
static constexpr int BX = 32;
static constexpr int BY = 8;
static constexpr int ZCHUNK = 8;

// 3D index calculation (host side, global field)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Local slabs are stored with one ghost plane on each side in Z:
// local plane z (0 <= z < nzloc) lives at device offset (z + 1) * nx * ny.
__device__ __forceinline__ size_t planeOffset(const long long z, const long long nx, const long long ny) {
    return static_cast<size_t>(z + 1) * static_cast<size_t>(nx * ny);
}

// Second-order difference scaling. Dividing by h^2 is exactly equivalent to multiplying by
// 1/h^2 whenever h^2 is a power of two, which lets us avoid the (very slow) FP64 division on
// the device for the usual uniform unit spacing. RECIP selects the exact-reciprocal variant.
template <bool RECIP>
__device__ __forceinline__ double scaleH2(const double v, const double h2) {
    return RECIP ? v * h2 : v / h2;
}

// Chemical potential: mu = 4.5*((c+1)e_AA + (c-1)e_BB - 2c e_AB) + 3c + c^3 - gamma * lap(c)
template <bool RECIP>
__global__ __launch_bounds__(BX* BY) void computeChemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu, const int nx, const int ny, const int z0, const int z1,
    const double dx2, const double dy2, const double dz2, const double gamma, const double e_AA, const double e_BB,
    const double e_AB) {
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const int zstart = z0 + static_cast<int>(blockIdx.z) * ZCHUNK;
    if (zstart >= z1) return;

    // Clamped (zero-gradient) boundaries in X and Y
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int yn = (y > 0) ? y - 1 : 0;

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t row = static_cast<size_t>(y) * nx;
    const size_t self = row + x;

    const double* p = c + planeOffset(zstart, nx, ny);
    double cm = p[self - plane];
    double cv = p[self];

#pragma unroll 1
    for (int k = 0; k < ZCHUNK; ++k) {
        const int z = zstart + k;
        if (z >= z1) break;

        const double* q = c + planeOffset(z, nx, ny);
        const double cp = q[self + plane];

        const double cxx = scaleH2<RECIP>(q[row + xp] + q[row + xn] - 2.0 * cv, dx2);
        const double cyy = scaleH2<RECIP>(
            q[static_cast<size_t>(yp) * nx + x] + q[static_cast<size_t>(yn) * nx + x] - 2.0 * cv, dy2);
        const double czz = scaleH2<RECIP>(cp + cm - 2.0 * cv, dz2);

        mu[planeOffset(z, nx, ny) + self] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) //
                                            + 3.0 * cv + cv * cv * cv                                      //
                                            - gamma * (cxx + cyy + czz);

        cm = cv;
        cv = cp;
    }
}

// Concentration update: cnew = cold + dt * D * lap(mu)
template <bool RECIP>
__global__ __launch_bounds__(BX* BY) void cahnHilliardUpdateKernel(
    double* __restrict__ cnew, const double* __restrict__ cold, const double* __restrict__ mu, const int nx,
    const int ny, const int z0, const int z1, const double D, const double dt, const double dx2, const double dy2,
    const double dz2) {
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const int zstart = z0 + static_cast<int>(blockIdx.z) * ZCHUNK;
    if (zstart >= z1) return;

    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int yn = (y > 0) ? y - 1 : 0;

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t row = static_cast<size_t>(y) * nx;
    const size_t self = row + x;

    const double* p = mu + planeOffset(zstart, nx, ny);
    double mm = p[self - plane];
    double mv = p[self];

#pragma unroll 1
    for (int k = 0; k < ZCHUNK; ++k) {
        const int z = zstart + k;
        if (z >= z1) break;

        const double* q = mu + planeOffset(z, nx, ny);
        const double mp = q[self + plane];

        const double mxx = scaleH2<RECIP>(q[row + xp] + q[row + xn] - 2.0 * mv, dx2);
        const double myy = scaleH2<RECIP>(
            q[static_cast<size_t>(yp) * nx + x] + q[static_cast<size_t>(yn) * nx + x] - 2.0 * mv, dy2);
        const double mzz = scaleH2<RECIP>(mp + mm - 2.0 * mv, dz2);

        const size_t o = planeOffset(z, nx, ny) + self;
        cnew[o] = cold[o] + dt * D * (mxx + myy + mzz);

        mm = mv;
        mv = mp;
    }
}

// Initialize the local slab of the concentration field (values identical to the serial reference)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t zoff, const size_t nzloc) {
    const size_t vol = nx * ny * nz;

#pragma omp parallel for schedule(static)
    for (long long z = 0; z < static_cast<long long>(nzloc); ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, static_cast<size_t>(z), nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = (static_cast<size_t>(z) + zoff) * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
    const long long n = static_cast<long long>(c.size());

    // Check for NaN or Inf
    int bad = 0;
#pragma omp parallel for schedule(static) reduction(| : bad)
    for (long long i = 0; i < n; ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) bad = 1;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
#pragma omp parallel for schedule(static) reduction(min : minVal) reduction(max : maxVal)
    for (long long i = 0; i < n; ++i) {
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

namespace {

// Everything needed to advance one local slab on one GPU.
struct Slab {
    int nx = 0, ny = 0, nzloc = 0;
    size_t plane = 0;     // nx * ny
    size_t bytesPlane = 0;

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;

    // Pinned staging buffers (used when the MPI stack is not CUDA aware)
    double* h_sendLo = nullptr;
    double* h_sendHi = nullptr;
    double* h_recvLo = nullptr;
    double* h_recvHi = nullptr;

    cudaStream_t sCompute = nullptr;
    cudaStream_t sHalo = nullptr;

    // Pre-scaled stencil coefficients: exact reciprocals of h^2 if that is lossless, h^2 otherwise
    bool recip = false;
    double hx = 0.0, hy = 0.0, hz = 0.0;

    MPI_Comm comm = MPI_COMM_NULL;
    int lo = MPI_PROC_NULL; // neighbour towards smaller z
    int hi = MPI_PROC_NULL; // neighbour towards larger z
    bool cudaAware = false;

    double* plane_ptr(double* base, const int z) const { return base + static_cast<size_t>(z + 1) * plane; }
};

// x/h^2 == x*(1/h^2) bit-for-bit exactly when h^2 is a (normal) power of two
bool exactReciprocal(const double h2) {
    int e = 0;
    return std::isnormal(h2) && h2 > 0.0 && std::frexp(h2, &e) == 0.5 && std::isnormal(1.0 / h2);
}

void launchMu(const Slab& s, const double* c, double* mu, const int z0, const int z1, const double gamma,
              const double e_AA, const double e_BB, const double e_AB, cudaStream_t stream) {
    if (z1 <= z0) return;
    const dim3 block(BX, BY, 1);
    const dim3 grid((s.nx + BX - 1) / BX, (s.ny + BY - 1) / BY, (z1 - z0 + ZCHUNK - 1) / ZCHUNK);
    if (s.recip) {
        computeChemicalPotentialKernel<true>
            <<<grid, block, 0, stream>>>(c, mu, s.nx, s.ny, z0, z1, s.hx, s.hy, s.hz, gamma, e_AA, e_BB, e_AB);
    } else {
        computeChemicalPotentialKernel<false>
            <<<grid, block, 0, stream>>>(c, mu, s.nx, s.ny, z0, z1, s.hx, s.hy, s.hz, gamma, e_AA, e_BB, e_AB);
    }
}

void launchUpdate(const Slab& s, double* cnew, const double* cold, const double* mu, const int z0, const int z1,
                  const double D, const double dt, cudaStream_t stream) {
    if (z1 <= z0) return;
    const dim3 block(BX, BY, 1);
    const dim3 grid((s.nx + BX - 1) / BX, (s.ny + BY - 1) / BY, (z1 - z0 + ZCHUNK - 1) / ZCHUNK);
    if (s.recip) {
        cahnHilliardUpdateKernel<true>
            <<<grid, block, 0, stream>>>(cnew, cold, mu, s.nx, s.ny, z0, z1, D, dt, s.hx, s.hy, s.hz);
    } else {
        cahnHilliardUpdateKernel<false>
            <<<grid, block, 0, stream>>>(cnew, cold, mu, s.nx, s.ny, z0, z1, D, dt, s.hx, s.hy, s.hz);
    }
}

// Fill the ghost planes of `field`: MPI halo exchange with the z-neighbours and, at the global
// domain boundaries, a mirror of the outermost plane (which reproduces the clamped stencil).
// `ready` marks the point in the compute stream at which the outermost planes of `field` are
// final; the halo stream waits for it, so it can run concurrently with interior compute.
void exchangeHalos(Slab& s, double* field, cudaEvent_t ready) {
    CUDA_CHECK(cudaStreamWaitEvent(s.sHalo, ready, 0));

    // Mirror at global boundaries (no neighbour in that direction)
    if (s.lo == MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(s.plane_ptr(field, -1), s.plane_ptr(field, 0), s.bytesPlane, cudaMemcpyDeviceToDevice,
                                   s.sHalo));
    }
    if (s.hi == MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(s.plane_ptr(field, s.nzloc), s.plane_ptr(field, s.nzloc - 1), s.bytesPlane,
                                   cudaMemcpyDeviceToDevice, s.sHalo));
    }

    if (s.lo == MPI_PROC_NULL && s.hi == MPI_PROC_NULL) {
        CUDA_CHECK(cudaStreamSynchronize(s.sHalo));
        return;
    }

    const double* sendLo = s.plane_ptr(field, 0);              // to lower neighbour
    const double* sendHi = s.plane_ptr(field, s.nzloc - 1);    // to upper neighbour
    double* recvLo = s.plane_ptr(field, -1);                   // from lower neighbour
    double* recvHi = s.plane_ptr(field, s.nzloc);              // from upper neighbour

    const void* sbufLo = sendLo;
    const void* sbufHi = sendHi;
    void* rbufLo = recvLo;
    void* rbufHi = recvHi;

    if (!s.cudaAware) {
        if (s.lo != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(s.h_sendLo, sendLo, s.bytesPlane, cudaMemcpyDeviceToHost, s.sHalo));
        if (s.hi != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(s.h_sendHi, sendHi, s.bytesPlane, cudaMemcpyDeviceToHost, s.sHalo));
        sbufLo = s.h_sendLo;
        sbufHi = s.h_sendHi;
        rbufLo = s.h_recvLo;
        rbufHi = s.h_recvHi;
    }
    CUDA_CHECK(cudaStreamSynchronize(s.sHalo));

    const int count = static_cast<int>(s.plane);
    MPI_Request reqs[4];
    int nreq = 0;
    if (s.lo != MPI_PROC_NULL) {
        MPI_Irecv(rbufLo, count, MPI_DOUBLE, s.lo, 1, s.comm, &reqs[nreq++]);
        MPI_Isend(const_cast<void*>(sbufLo), count, MPI_DOUBLE, s.lo, 0, s.comm, &reqs[nreq++]);
    }
    if (s.hi != MPI_PROC_NULL) {
        MPI_Irecv(rbufHi, count, MPI_DOUBLE, s.hi, 0, s.comm, &reqs[nreq++]);
        MPI_Isend(const_cast<void*>(sbufHi), count, MPI_DOUBLE, s.hi, 1, s.comm, &reqs[nreq++]);
    }
    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    if (!s.cudaAware) {
        if (s.lo != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(recvLo, s.h_recvLo, s.bytesPlane, cudaMemcpyHostToDevice, s.sHalo));
        if (s.hi != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(recvHi, s.h_recvHi, s.bytesPlane, cudaMemcpyHostToDevice, s.sHalo));
        CUDA_CHECK(cudaStreamSynchronize(s.sHalo));
    }
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

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

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
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

    // --- 1D domain decomposition along Z -------------------------------------------------
    const size_t base = nz / static_cast<size_t>(nranks);
    const size_t rem = nz % static_cast<size_t>(nranks);
    const size_t nzloc = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t zoff = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), rem);

    // Ranks without any planes (more ranks than z-planes) stay out of the halo exchange but
    // still take part in the global collectives.
    const int active = (nzloc > 0) ? 1 : 0;
    MPI_Comm activeComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active, rank, &activeComm);

    // --- Pick a GPU: one per rank, round-robin within the node ---------------------------
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));
    CUDA_CHECK(cudaFree(nullptr)); // establish the context before timing

    Slab slab;
    slab.nx = static_cast<int>(nx);
    slab.ny = static_cast<int>(ny);
    slab.nzloc = static_cast<int>(nzloc);
    slab.plane = nx * ny;
    slab.bytesPlane = slab.plane * sizeof(double);
    slab.comm = activeComm;
    slab.recip = exactReciprocal(dx * dx) && exactReciprocal(dy * dy) && exactReciprocal(dz * dz);
    slab.hx = slab.recip ? 1.0 / (dx * dx) : dx * dx;
    slab.hy = slab.recip ? 1.0 / (dy * dy) : dy * dy;
    slab.hz = slab.recip ? 1.0 / (dz * dz) : dz * dz;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    slab.cudaAware = (MPIX_Query_cuda_support() == 1);
#else
    slab.cudaAware = false;
#endif

    std::vector<double> local(slab.plane * nzloc);
    if (active) {
        int arank = 0, asize = 0;
        MPI_Comm_rank(activeComm, &arank);
        MPI_Comm_size(activeComm, &asize);
        slab.lo = (arank > 0) ? arank - 1 : MPI_PROC_NULL;
        slab.hi = (arank + 1 < asize) ? arank + 1 : MPI_PROC_NULL;

        const size_t bytes = slab.plane * (nzloc + 2) * sizeof(double);
        CUDA_CHECK(cudaMalloc(&slab.d_cold, bytes));
        CUDA_CHECK(cudaMalloc(&slab.d_cnew, bytes));
        CUDA_CHECK(cudaMalloc(&slab.d_mu, bytes));
        if (!slab.cudaAware) {
            CUDA_CHECK(cudaMallocHost(&slab.h_sendLo, slab.bytesPlane));
            CUDA_CHECK(cudaMallocHost(&slab.h_sendHi, slab.bytesPlane));
            CUDA_CHECK(cudaMallocHost(&slab.h_recvLo, slab.bytesPlane));
            CUDA_CHECK(cudaMallocHost(&slab.h_recvHi, slab.bytesPlane));
        }
        CUDA_CHECK(cudaStreamCreate(&slab.sCompute));
        CUDA_CHECK(cudaStreamCreate(&slab.sHalo));

        if (rank == 0) printf("Initializing concentration field...\n");
        initializeConcentration(local, nx, ny, nz, zoff, nzloc);
        CUDA_CHECK(cudaMemcpy(slab.plane_ptr(slab.d_cold, 0), local.data(), local.size() * sizeof(double),
                              cudaMemcpyHostToDevice));
    } else if (rank == 0) {
        printf("Initializing concentration field...\n");
    }

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    if (active) {
        const int nzl = slab.nzloc;
        // Interior planes do not touch the ghost layers and are computed while the halo
        // exchange for the boundary planes is in flight.
        const int i0 = (nzl >= 3) ? 1 : 0;
        const int i1 = (nzl >= 3) ? nzl - 1 : 0;

        cudaEvent_t evMu, evUpdate;
        CUDA_CHECK(cudaEventCreateWithFlags(&evMu, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evUpdate, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(evUpdate, slab.sCompute));

        for (int t = 0; t < iterations; ++t) {
            // Chemical potential: interior planes need no ghosts and overlap the c halo exchange
            launchMu(slab, slab.d_cold, slab.d_mu, i0, i1, gamma, e_AA, e_BB, e_AB, slab.sCompute);
            exchangeHalos(slab, slab.d_cold, evUpdate);
            if (nzl >= 3) {
                launchMu(slab, slab.d_cold, slab.d_mu, 0, 1, gamma, e_AA, e_BB, e_AB, slab.sCompute);
                launchMu(slab, slab.d_cold, slab.d_mu, nzl - 1, nzl, gamma, e_AA, e_BB, e_AB, slab.sCompute);
            } else {
                launchMu(slab, slab.d_cold, slab.d_mu, 0, nzl, gamma, e_AA, e_BB, e_AB, slab.sCompute);
            }
            CUDA_CHECK(cudaEventRecord(evMu, slab.sCompute));

            // Concentration update: same overlap pattern for the mu halo exchange
            launchUpdate(slab, slab.d_cnew, slab.d_cold, slab.d_mu, i0, i1, D, dt, slab.sCompute);
            exchangeHalos(slab, slab.d_mu, evMu);
            if (nzl >= 3) {
                launchUpdate(slab, slab.d_cnew, slab.d_cold, slab.d_mu, 0, 1, D, dt, slab.sCompute);
                launchUpdate(slab, slab.d_cnew, slab.d_cold, slab.d_mu, nzl - 1, nzl, D, dt, slab.sCompute);
            } else {
                launchUpdate(slab, slab.d_cnew, slab.d_cold, slab.d_mu, 0, nzl, D, dt, slab.sCompute);
            }
            CUDA_CHECK(cudaEventRecord(evUpdate, slab.sCompute));

            // Swap buffers
            std::swap(slab.d_cold, slab.d_cnew);
        }
        CUDA_CHECK(cudaStreamSynchronize(slab.sCompute));
        CUDA_CHECK(cudaEventDestroy(evMu));
        CUDA_CHECK(cudaEventDestroy(evUpdate));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance
        const double cellUpdates = (double)gridSize * iterations;
        const double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // --- Gather the full field on rank 0 when results are needed --------------------------
    std::vector<double> full;
    if (printResults || validate) {
        if (active) {
            CUDA_CHECK(cudaMemcpy(local.data(), slab.plane_ptr(slab.d_cold, 0), local.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
        if (rank == 0) full.resize(gridSize);

        std::vector<int> counts(nranks), displs(nranks);
        size_t off = 0;
        for (int r = 0; r < nranks; ++r) {
            const size_t n = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            counts[r] = static_cast<int>(n * slab.plane);
            displs[r] = static_cast<int>(off);
            off += n * slab.plane;
        }
        MPI_Gatherv(local.data(), counts[rank], MPI_DOUBLE, rank == 0 ? full.data() : nullptr, counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int status = 0;
    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(full, "Concentration");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(full, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (active) {
        CUDA_CHECK(cudaFree(slab.d_cold));
        CUDA_CHECK(cudaFree(slab.d_cnew));
        CUDA_CHECK(cudaFree(slab.d_mu));
        if (!slab.cudaAware) {
            CUDA_CHECK(cudaFreeHost(slab.h_sendLo));
            CUDA_CHECK(cudaFreeHost(slab.h_sendHi));
            CUDA_CHECK(cudaFreeHost(slab.h_recvLo));
            CUDA_CHECK(cudaFreeHost(slab.h_recvHi));
        }
        CUDA_CHECK(cudaStreamDestroy(slab.sCompute));
        CUDA_CHECK(cudaStreamDestroy(slab.sHalo));
    }
    MPI_Comm_free(&activeComm);
    MPI_Finalize();
    return status;
}
