// Cahn-Hilliard phase separation benchmark
//
// Hybrid parallelization:
//   * MPI    : 1D slab decomposition of the domain along Z. Each rank owns a
//              contiguous block of Z planes plus 2 ghost planes on each side.
//              One halo exchange per time step (2 planes per side) which is
//              overlapped with the interior chemical-potential computation.
//              Ghost planes are exchanged directly from device memory when the
//              MPI library is CUDA-aware, otherwise staged through pinned host
//              buffers.
//   * CUDA   : all stencil work (chemical potential + concentration update)
//              runs on one GPU per rank. Two streams separate boundary and
//              interior work so communication hides behind computation.
//   * OpenMP : host-side parallelism for initialization, halo staging and
//              validation on the (large) gathered array.
//
// Numerical semantics are identical to the original sequential code: the
// same floating-point expressions are evaluated in the same order and FMA
// contraction is disabled for the device code.

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
#if defined(OPEN_MPI)
#include <mpi-ext.h>  // MPIX_Query_cuda_support (CUDA-aware MPI detection)
#endif

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
            fprintf(stderr, "MPI error %s at %s:%d: code %d\n", #call, __FILE__, __LINE__, err__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

// Number of ghost planes on each side of the local Z slab
constexpr int GHOST = 2;

// Physical parameters (compile-time constants so the device code can fold
// e.g. the division by dx*dx == 1.0 exactly, without altering the arithmetic).
namespace phys {
constexpr double dx = 1.0;
constexpr double dy = 1.0;
constexpr double dz = 1.0;
constexpr double dt = 0.01;
constexpr double e_AA = -(2.0 / 9.0);
constexpr double e_BB = -(2.0 / 9.0);
constexpr double e_AB = (2.0 / 9.0);
constexpr double gamma = 0.5;
constexpr double D = 1.0;
}  // namespace phys

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx,
                                                 const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Each thread owns one (x, y) column and marches along Z over a chunk of
// ZCHUNK planes, keeping the z-1 / z / z+1 values in registers so every value
// is loaded from global memory once for the Z direction (the X/Y neighbours
// are served from L1/L2).
constexpr int ZCHUNK = 8;

// Laplacian of the in-plane neighbours plus the register-held Z neighbours,
// with clamped boundary conditions (identical expression order to the
// original code). `rowz` points to the start of plane z.
__device__ __forceinline__ double laplacianAt(const double* __restrict__ rowz, const size_t nx, const size_t ny,
                                              const size_t x, const size_t y, const double cn, const double cc,
                                              const double cp) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double* row = rowz + y * nx;
    const double cxx = (row[xp] + row[xn] - 2.0 * cc) / (phys::dx * phys::dx);
    const double cyy = (rowz[yp * nx + x] + rowz[yn * nx + x] - 2.0 * cc) / (phys::dy * phys::dy);
    const double czz = (cp + cn - 2.0 * cc) / (phys::dz * phys::dz);

    return cxx + cyy + czz;
}

// Chemical potential for local planes [z0, z1)
__global__ void __launch_bounds__(256) chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                                               const size_t nx, const size_t ny, const size_t nz,
                                                               const long z0, const long z1, const long zoff) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const long zs = z0 + static_cast<long>(blockIdx.z) * ZCHUNK;
    const long ze = (zs + ZCHUNK < z1) ? zs + ZCHUNK : z1;
    if (zs >= ze) return;

    const long plane = static_cast<long>(nx * ny);
    const long col = static_cast<long>(y * nx + x);
    long zg = zs + zoff;

    double cc = c[zs * plane + col];
    double cn = (zg > 0) ? c[(zs - 1) * plane + col] : cc;

    for (long z = zs; z < ze; ++z, ++zg) {
        const double cp = (zg < static_cast<long>(nz) - 1) ? c[(z + 1) * plane + col] : cc;
        const double cv = cc;
        mu[z * plane + col] = 4.5 * ((cv + 1.0) * phys::e_AA + (cv - 1.0) * phys::e_BB - 2.0 * cv * phys::e_AB) +
                              3.0 * cv + cv * cv * cv -
                              phys::gamma * laplacianAt(c + z * plane, nx, ny, x, y, cn, cc, cp);
        cn = cc;
        cc = cp;
    }
}

// Cahn-Hilliard update for local planes [z0, z1)
__global__ void __launch_bounds__(256) cahnHilliardUpdateKernel(double* __restrict__ cnew,
                                                                const double* __restrict__ cold,
                                                                const double* __restrict__ mu, const size_t nx,
                                                                const size_t ny, const size_t nz, const long z0,
                                                                const long z1, const long zoff) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const long zs = z0 + static_cast<long>(blockIdx.z) * ZCHUNK;
    const long ze = (zs + ZCHUNK < z1) ? zs + ZCHUNK : z1;
    if (zs >= ze) return;

    const long plane = static_cast<long>(nx * ny);
    const long col = static_cast<long>(y * nx + x);
    long zg = zs + zoff;

    double mc = mu[zs * plane + col];
    double mn = (zg > 0) ? mu[(zs - 1) * plane + col] : mc;

    for (long z = zs; z < ze; ++z, ++zg) {
        const double mp = (zg < static_cast<long>(nz) - 1) ? mu[(z + 1) * plane + col] : mc;
        const long idx = z * plane + col;
        cnew[idx] = cold[idx] + phys::dt * phys::D * laplacianAt(mu + z * plane, nx, ny, x, y, mn, mc, mp);
        mn = mc;
        mc = mp;
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

// Initialize the local part of the concentration field (planes [zoff, zoff+nzl))
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t zoff, const size_t nzl) {
    const size_t vol = nx * ny * nz;

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t zl = 0; zl < nzl; ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t z = zoff + zl;
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, zl, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    bool bad = false;
#pragma omp parallel for reduction(|| : bad) schedule(static)
    for (size_t i = 0; i < c.size(); ++i) {
        const double val = c[i];
        if (std::isnan(val) || std::isinf(val)) bad = true;
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

// Launch helper covering local planes [z0, z1): 2D thread tiles in X/Y, one block per ZCHUNK planes
static dim3 gridFor(const dim3 block, const size_t nx, const size_t ny, const long z0, const long z1) {
    return dim3(static_cast<unsigned>((nx + block.x - 1) / block.x), static_cast<unsigned>((ny + block.y - 1) / block.y),
                static_cast<unsigned>((static_cast<size_t>(z1 - z0) + ZCHUNK - 1) / ZCHUNK));
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));

    int rank = 0, nranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

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
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", nranks, omp_get_max_threads());
    }

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;

    // ------------------------------------------------------------------
    // GPU selection: one device per rank, chosen by node-local rank
    // ------------------------------------------------------------------
    {
        MPI_Comm nodeComm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
        int localRank = 0;
        MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
        MPI_CHECK(MPI_Comm_free(&nodeComm));
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev <= 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % ndev));
    }

    // CUDA-aware MPI: pass device pointers straight to MPI if supported.
    // Can be forced on/off with CH_CUDA_AWARE_MPI=1/0.
    bool cudaAwareMPI = false;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    cudaAwareMPI = (MPIX_Query_cuda_support() == 1);
#endif
    if (const char* env = getenv("CH_CUDA_AWARE_MPI")) cudaAwareMPI = (atoi(env) != 0);
    {
        int flag = cudaAwareMPI ? 1 : 0;
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &flag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        cudaAwareMPI = (flag != 0);
    }
    if (rank == 0) printf("Halo exchange: %s\n", cudaAwareMPI ? "CUDA-aware MPI (device buffers)" : "host-staged");

    // ------------------------------------------------------------------
    // Domain decomposition along Z. Every active rank owns >= GHOST planes
    // so that ghost planes always come from the direct neighbour.
    // ------------------------------------------------------------------
    int nactive = static_cast<int>(std::min<size_t>(static_cast<size_t>(nranks), std::max<size_t>(1, nz / GHOST)));
    size_t nzl = 0;   // local number of planes
    size_t zoff = 0;  // global index of local plane 0
    if (rank < nactive) {
        const size_t base = nz / nactive;
        const size_t rem = nz % nactive;
        nzl = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
        zoff = static_cast<size_t>(rank) * base + std::min<size_t>(rank, rem);
    }
    const bool active = nzl > 0;
    const int prevRank = (active && rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int nextRank = (active && rank + 1 < nactive) ? rank + 1 : MPI_PROC_NULL;
    const bool hasPrev = prevRank != MPI_PROC_NULL;
    const bool hasNext = nextRank != MPI_PROC_NULL;
    const long nzlL = static_cast<long>(nzl);

    // Plane ranges (local indices):
    //   mu is needed on [muLo, muHi) = local planes extended by one ghost plane
    //   on each side that has a neighbour.
    //   Boundary planes are those whose stencil (directly or via mu) touches
    //   the ghost planes; everything else is interior and is processed while
    //   the halo exchange is in flight:
    //     mu boundary : [muLo, GHOST) and [nzl-GHOST, muHi)   -> need ghost c
    //     mu interior : [GHOST, nzl-GHOST)                    -> local c only
    //     c  boundary : [0, GHOST+1) and [nzl-GHOST-1, nzl)   -> need boundary mu
    //     c  interior : [GHOST+1, nzl-GHOST-1)                -> interior mu only
    const long muLo = hasPrev ? -1 : 0;
    const long muHi = hasNext ? nzlL + 1 : nzlL;
    const long muBndLoEnd = std::min<long>(GHOST, nzlL);
    const long muBndHiBeg = std::max<long>(nzlL - GHOST, muBndLoEnd);
    const long bndLoEnd = std::min<long>(GHOST + 1, nzlL);
    const long bndHiBeg = std::max<long>(nzlL - GHOST - 1, bndLoEnd);

    // ------------------------------------------------------------------
    // Memory allocation
    // ------------------------------------------------------------------
    const size_t localPlanes = nzl + 2 * GHOST;
    const size_t localBytes = localPlanes * plane * sizeof(double);

    double *d_c0 = nullptr, *d_c1 = nullptr, *d_mu = nullptr;
    double *h_sendLo = nullptr, *h_sendHi = nullptr, *h_recvLo = nullptr, *h_recvHi = nullptr;
    std::vector<double> h_local;
    cudaStream_t sBnd, sInt;
    cudaEvent_t evBndUpd, evIntMu;

    if (active) {
        CUDA_CHECK(cudaMalloc(&d_c0, localBytes));
        CUDA_CHECK(cudaMalloc(&d_c1, localBytes));
        CUDA_CHECK(cudaMalloc(&d_mu, localBytes));
        const size_t haloBytes = GHOST * plane * sizeof(double);
        CUDA_CHECK(cudaMallocHost(&h_sendLo, haloBytes));
        CUDA_CHECK(cudaMallocHost(&h_sendHi, haloBytes));
        CUDA_CHECK(cudaMallocHost(&h_recvLo, haloBytes));
        CUDA_CHECK(cudaMallocHost(&h_recvHi, haloBytes));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sBnd, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sInt, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&evBndUpd, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evIntMu, cudaEventDisableTiming));
        h_local.resize(nzl * plane);
    }

    // Pointers to local plane 0 (ghosts at negative offsets)
    double* cold = active ? d_c0 + GHOST * plane : nullptr;
    double* cnew = active ? d_c1 + GHOST * plane : nullptr;
    double* mu = active ? d_mu + GHOST * plane : nullptr;

    // ------------------------------------------------------------------
    // Initialize concentration field
    // ------------------------------------------------------------------
    if (rank == 0) printf("Initializing concentration field...\n");
    if (active) {
        initializeConcentration(h_local, nx, ny, nz, zoff, nzl);
        CUDA_CHECK(cudaMemset(d_c0, 0, localBytes));
        CUDA_CHECK(cudaMemset(d_c1, 0, localBytes));
        CUDA_CHECK(cudaMemset(d_mu, 0, localBytes));
        CUDA_CHECK(cudaMemcpy(cold, h_local.data(), nzl * plane * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        // Nothing is pending on the device yet; record events so that the
        // first iteration's waits are satisfied immediately.
        CUDA_CHECK(cudaEventRecord(evBndUpd, sBnd));
    }

    const dim3 block(32, 8, 1);
    const size_t haloCount = GHOST * plane;

    // ------------------------------------------------------------------
    // Run simulation
    // ------------------------------------------------------------------
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        if (active) {
            // --- 1. Halo exchange of cold (boundary planes), overlapped with interior mu ---
            MPI_Request reqs[4];
            int nreq = 0;
            double* const sendLoDev = cold;
            double* const sendHiDev = cold + (nzlL - GHOST) * static_cast<long>(plane);
            double* const ghostLoDev = cold - GHOST * static_cast<long>(plane);
            double* const ghostHiDev = cold + nzlL * static_cast<long>(plane);
            // Stream sBnd is already ordered after the boundary update that produced the send planes.
            if (!cudaAwareMPI) {
                if (hasPrev) {
                    CUDA_CHECK(cudaMemcpyAsync(h_sendLo, sendLoDev, haloCount * sizeof(double), cudaMemcpyDeviceToHost,
                                               sBnd));
                }
                if (hasNext) {
                    CUDA_CHECK(cudaMemcpyAsync(h_sendHi, sendHiDev, haloCount * sizeof(double), cudaMemcpyDeviceToHost,
                                               sBnd));
                }
            }

            // Interior mu: needs cold on planes [1, nzl-1) -> wait for the boundary update
            // (the interior update is on this stream already). Independent of the exchange.
            CUDA_CHECK(cudaStreamWaitEvent(sInt, evBndUpd, 0));
            if (muBndHiBeg > muBndLoEnd) {
                chemicalPotentialKernel<<<gridFor(block, nx, ny, muBndLoEnd, muBndHiBeg), block, 0, sInt>>>(
                    cold, mu, nx, ny, nz, muBndLoEnd, muBndHiBeg, static_cast<long>(zoff));
            }
            CUDA_CHECK(cudaEventRecord(evIntMu, sInt));

            if (hasPrev || hasNext) {
                // Waits for the send planes to be final (and, in staged mode, to land in
                // host memory) and for the previous step's ghost upload / ghost reads to finish.
                CUDA_CHECK(cudaStreamSynchronize(sBnd));
                double* recvLo = cudaAwareMPI ? ghostLoDev : h_recvLo;
                double* recvHi = cudaAwareMPI ? ghostHiDev : h_recvHi;
                const double* sendLo = cudaAwareMPI ? sendLoDev : h_sendLo;
                const double* sendHi = cudaAwareMPI ? sendHiDev : h_sendHi;
                if (hasPrev) {
                    MPI_CHECK(MPI_Irecv(recvLo, static_cast<int>(haloCount), MPI_DOUBLE, prevRank, 1, MPI_COMM_WORLD,
                                        &reqs[nreq++]));
                }
                if (hasNext) {
                    MPI_CHECK(MPI_Irecv(recvHi, static_cast<int>(haloCount), MPI_DOUBLE, nextRank, 0, MPI_COMM_WORLD,
                                        &reqs[nreq++]));
                }
                if (hasPrev) {
                    MPI_CHECK(MPI_Isend(sendLo, static_cast<int>(haloCount), MPI_DOUBLE, prevRank, 0, MPI_COMM_WORLD,
                                        &reqs[nreq++]));
                }
                if (hasNext) {
                    MPI_CHECK(MPI_Isend(sendHi, static_cast<int>(haloCount), MPI_DOUBLE, nextRank, 1, MPI_COMM_WORLD,
                                        &reqs[nreq++]));
                }
                MPI_CHECK(MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE));
                if (!cudaAwareMPI) {
                    if (hasPrev) {
                        CUDA_CHECK(cudaMemcpyAsync(ghostLoDev, h_recvLo, haloCount * sizeof(double),
                                                   cudaMemcpyHostToDevice, sBnd));
                    }
                    if (hasNext) {
                        CUDA_CHECK(cudaMemcpyAsync(ghostHiDev, h_recvHi, haloCount * sizeof(double),
                                                   cudaMemcpyHostToDevice, sBnd));
                    }
                }
            }

            // --- 2. Boundary mu (needs ghosts + boundary cold, both already on sBnd) ---
            if (muBndLoEnd > muLo) {
                chemicalPotentialKernel<<<gridFor(block, nx, ny, muLo, muBndLoEnd), block, 0, sBnd>>>(
                    cold, mu, nx, ny, nz, muLo, muBndLoEnd, static_cast<long>(zoff));
            }
            if (muHi > muBndHiBeg) {
                chemicalPotentialKernel<<<gridFor(block, nx, ny, muBndHiBeg, muHi), block, 0, sBnd>>>(
                    cold, mu, nx, ny, nz, muBndHiBeg, muHi, static_cast<long>(zoff));
            }

            // --- 3. Boundary update (needs boundary mu + the adjacent interior mu planes) ---
            CUDA_CHECK(cudaStreamWaitEvent(sBnd, evIntMu, 0));
            if (bndLoEnd > 0) {
                cahnHilliardUpdateKernel<<<gridFor(block, nx, ny, 0, bndLoEnd), block, 0, sBnd>>>(
                    cnew, cold, mu, nx, ny, nz, 0, bndLoEnd, static_cast<long>(zoff));
            }
            if (nzlL > bndHiBeg) {
                cahnHilliardUpdateKernel<<<gridFor(block, nx, ny, bndHiBeg, nzlL), block, 0, sBnd>>>(
                    cnew, cold, mu, nx, ny, nz, bndHiBeg, nzlL, static_cast<long>(zoff));
            }
            CUDA_CHECK(cudaEventRecord(evBndUpd, sBnd));

            // --- 4. Interior update (needs interior mu only, same stream) ---
            if (bndHiBeg > bndLoEnd) {
                cahnHilliardUpdateKernel<<<gridFor(block, nx, ny, bndLoEnd, bndHiBeg), block, 0, sInt>>>(
                    cnew, cold, mu, nx, ny, nz, bndLoEnd, bndHiBeg, static_cast<long>(zoff));
            }
        }

        // Swap buffers
        std::swap(cold, cnew);
    }

    if (active) {
        CUDA_CHECK(cudaStreamSynchronize(sBnd));
        CUDA_CHECK(cudaStreamSynchronize(sInt));
        CUDA_CHECK(cudaGetLastError());
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ------------------------------------------------------------------
    // Gather the full field on rank 0 for output / validation
    // ------------------------------------------------------------------
    int ret = 0;
    if (printResults || validate) {
        if (active) {
            CUDA_CHECK(cudaMemcpy(h_local.data(), cold, nzl * plane * sizeof(double), cudaMemcpyDeviceToHost));
        }
        std::vector<double> full;
        std::vector<int> counts, displs;
        if (rank == 0) {
            full.resize(gridSize);
            counts.resize(nranks);
            displs.resize(nranks);
        }
        const int myCount = static_cast<int>(nzl * plane);
        MPI_CHECK(MPI_Gather(&myCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD));
        if (rank == 0) {
            int off = 0;
            for (int r = 0; r < nranks; ++r) {
                displs[r] = off;
                off += counts[r];
            }
        }
        MPI_CHECK(MPI_Gatherv(h_local.data(), myCount, MPI_DOUBLE, full.data(), counts.data(), displs.data(),
                              MPI_DOUBLE, 0, MPI_COMM_WORLD));

        if (rank == 0) {
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
                    ret = 0;
                } else {
                    printf("Validation: FAILED\n");
                    ret = 1;
                }
            }
        }
        MPI_CHECK(MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    if (active) {
        CUDA_CHECK(cudaEventDestroy(evBndUpd));
        CUDA_CHECK(cudaEventDestroy(evIntMu));
        CUDA_CHECK(cudaStreamDestroy(sBnd));
        CUDA_CHECK(cudaStreamDestroy(sInt));
        CUDA_CHECK(cudaFreeHost(h_sendLo));
        CUDA_CHECK(cudaFreeHost(h_sendHi));
        CUDA_CHECK(cudaFreeHost(h_recvLo));
        CUDA_CHECK(cudaFreeHost(h_recvHi));
        CUDA_CHECK(cudaFree(d_c0));
        CUDA_CHECK(cudaFree(d_c1));
        CUDA_CHECK(cudaFree(d_mu));
    }

    MPI_Finalize();
    return ret;
}
