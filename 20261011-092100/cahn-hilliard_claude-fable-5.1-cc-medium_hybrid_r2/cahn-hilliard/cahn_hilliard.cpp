// Cahn-Hilliard phase separation benchmark
//
// Hybrid parallel implementation:
//   * MPI    : 1D slab decomposition of the domain along Z, one rank per GPU,
//              single-plane halo exchange between neighbouring ranks
//              (communication is overlapped with interior computation).
//   * CUDA   : all stencil work (chemical potential + concentration update)
//              runs on the GPU; boundary planes are computed on a dedicated
//              stream so their halos can be exchanged while the interior
//              planes are still being processed.
//   * OpenMP : host-side work (field initialisation, validation reductions,
//              result staging) is multi-threaded.

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

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t err__ = (call);                                                             \
        if (err__ != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__,   \
                    __LINE__);                                                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

#define MPI_CHECK(call)                                                                         \
    do {                                                                                        \
        int err__ = (call);                                                                     \
        if (err__ != MPI_SUCCESS) {                                                             \
            fprintf(stderr, "MPI error %d at %s:%d\n", err__, __FILE__, __LINE__);              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// Device kernels
//
// Local device arrays hold nzl + 2 planes: plane 0 is the lower halo, planes
// 1..nzl are the owned slab, plane nzl + 1 is the upper halo.  The global Z
// coordinate of local plane (lz + 1) is z0 + lz.  Clamped boundary conditions
// are applied in X/Y directly and in Z via the global coordinate, so the halo
// planes are only ever read where a real neighbouring rank exists.
// ---------------------------------------------------------------------------

struct StencilParams {
    size_t nx, ny, nz;  // global grid
    size_t z0;          // global index of first owned plane
    size_t nzl;         // owned planes
    double dx, dy, dz;
    double invDx2, invDy2, invDz2;  // 1/(d*d), used only when bit-exact (see useReciprocal)
};

// Number of consecutive Z planes handled by one thread (register-blocked Z march)
constexpr int kZChunk = 16;

// Divide by h*h.  When UseRecip is set the host has verified that h*h is a power
// of two, so multiplying by the precomputed reciprocal is bit-identical to the
// division of the reference implementation but avoids the slow FP64 divide.
template <bool UseRecip>
__device__ __forceinline__ double divH2(const double v, const double h, const double invH2) {
    if constexpr (UseRecip) {
        return v * invH2;
    } else {
        return v / (h * h);
    }
}

// Compute Laplacian with clamped boundary conditions.  The Z neighbours are
// passed in registers (already clamped), the X/Y neighbours are read from memory.
template <bool UseRecip>
__device__ __forceinline__ double computeLaplacianDev(const double* __restrict__ c, const StencilParams& p,
                                                      const size_t x, const size_t y, const size_t idx,
                                                      const double cc, const double cn, const double cp) {
    const size_t nx = p.nx, ny = p.ny;
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t rowOff = idx - x;  // start of this row

    const double cxx = divH2<UseRecip>(c[rowOff + xp] + c[rowOff + xn] - 2.0 * cc, p.dx, p.invDx2);
    const double cyy = divH2<UseRecip>(c[idx + (yp - y) * nx] + c[idx - (y - yn) * nx] - 2.0 * cc, p.dy, p.invDy2);
    const double czz = divH2<UseRecip>(cp + cn - 2.0 * cc, p.dz, p.invDz2);

    return cxx + cyy + czz;
}

// Generic Z-marching stencil driver: every thread owns one (x, y) column and
// walks kZChunk consecutive owned planes starting at ozBegin + blockIdx.z*kZChunk,
// keeping the three Z neighbours of the input field in registers.
template <typename Op>
__device__ __forceinline__ void marchZ(const double* __restrict__ in, const StencilParams& p,
                                       const size_t ozBegin, const size_t ozCount, Op&& op) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= p.nx || y >= p.ny) return;

    const size_t plane = p.nx * p.ny;
    size_t oz = ozBegin + static_cast<size_t>(blockIdx.z) * kZChunk;
    const size_t ozEnd = (ozBegin + ozCount < oz + kZChunk) ? ozBegin + ozCount : oz + kZChunk;
    if (oz >= ozEnd) return;

    size_t gz = p.z0 + oz;
    size_t idx = idx3(x, y, oz + 1, p.nx, p.ny);  // +1: halo offset

    double cc = in[idx];
    double cn = (gz > 0) ? in[idx - plane] : cc;  // clamped lower neighbour

    for (; oz < ozEnd; ++oz, ++gz, idx += plane) {
        const double cp = (gz < p.nz - 1) ? in[idx + plane] : cc;  // clamped upper neighbour
        op(x, y, idx, cc, cn, cp);
        cn = cc;
        cc = cp;
    }
}

// Compute chemical potential
template <bool UseRecip>
__global__ void __launch_bounds__(256) computeChemicalPotentialKernel(const double* __restrict__ c,
                                                                      double* __restrict__ mu,
                                                                      const StencilParams p,
                                                                      const double gamma, const double e_AA,
                                                                      const double e_BB, const double e_AB,
                                                                      const size_t ozBegin, const size_t ozCount) {
    marchZ(c, p, ozBegin, ozCount,
           [&](const size_t x, const size_t y, const size_t idx, const double cv, const double cn, const double cp) {
               mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv +
                         cv * cv * cv - gamma * computeLaplacianDev<UseRecip>(c, p, x, y, idx, cv, cn, cp);
           });
}

// Cahn-Hilliard update step
template <bool UseRecip>
__global__ void __launch_bounds__(256) cahnHilliardUpdateKernel(double* __restrict__ cnew,
                                                                const double* __restrict__ cold,
                                                                const double* __restrict__ mu,
                                                                const StencilParams p, const double D,
                                                                const double dt, const size_t ozBegin,
                                                                const size_t ozCount) {
    marchZ(mu, p, ozBegin, ozCount,
           [&](const size_t x, const size_t y, const size_t idx, const double mc, const double mn, const double mp) {
               cnew[idx] = cold[idx] + dt * D * computeLaplacianDev<UseRecip>(mu, p, x, y, idx, mc, mn, mp);
           });
}

// True when multiplying by 1/(h*h) is bit-identical to dividing by h*h,
// i.e. h*h is an exact power of two.
static bool reciprocalIsExact(const double h) {
    const double h2 = h * h;
    if (!(h2 > 0.0) || std::isinf(h2)) return false;
    int e = 0;
    const double m = std::frexp(h2, &e);
    return m == 0.5;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

// Initialize concentration field (local slab, global indexing preserved)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t z0, const size_t nzl) {
    const size_t vol = nx * ny * nz;

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t lz = 0; lz < nzl; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t z = z0 + lz;
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Validate the (distributed) result; every rank participates, rank 0 prints.
bool validateResult(const std::vector<double>& c, const int rank) {
    const size_t n = c.size();

    // Check for NaN or Inf
    int localBad = 0;
#pragma omp parallel for reduction(| : localBad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const double val = c[i];
        if (std::isnan(val) || std::isinf(val)) localBad = 1;
    }
    int globalBad = 0;
    MPI_CHECK(MPI_Allreduce(&localBad, &globalBad, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD));
    if (globalBad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = n ? c[0] : HUGE_VAL;
    double maxVal = n ? c[0] : -HUGE_VAL;
#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }
    double gMin = 0.0, gMax = 0.0;
    MPI_CHECK(MPI_Allreduce(&minVal, &gMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&maxVal, &gMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    if (rank == 0) printf("Concentration range: [%.6f, %.6f]\n", gMin, gMax);

    // Values should generally stay within reasonable bounds
    if (gMax > 10.0 || gMin < -10.0) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
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

// Exchange one plane in each direction with the Z neighbours.
// Send buffers hold our first / last owned plane, receive buffers get the halos.
static void exchangeHalos(const double* sendLo, const double* sendHi, double* recvLo, double* recvHi,
                          const int rankLo, const int rankHi, const MPI_Datatype planeType) {
    MPI_Request reqs[4];
    MPI_CHECK(MPI_Irecv(recvLo, 1, planeType, rankLo, 0, MPI_COMM_WORLD, &reqs[0]));
    MPI_CHECK(MPI_Irecv(recvHi, 1, planeType, rankHi, 1, MPI_COMM_WORLD, &reqs[1]));
    MPI_CHECK(MPI_Isend(sendHi, 1, planeType, rankHi, 0, MPI_COMM_WORLD, &reqs[2]));
    MPI_CHECK(MPI_Isend(sendLo, 1, planeType, rankLo, 1, MPI_COMM_WORLD, &reqs[3]));
    MPI_CHECK(MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE));
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

    if (static_cast<size_t>(nranks) > nz) {
        if (rank == 0)
            fprintf(stderr, "Error: number of MPI ranks (%d) exceeds grid size in Z (%zu)\n", nranks, nz);
        MPI_Finalize();
        return 1;
    }

    // --- GPU selection: one device per rank, chosen by node-local rank -------
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_free(&nodeComm));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "Error: no CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    // --- Domain decomposition along Z ---------------------------------------
    const size_t base = nz / nranks;
    const size_t rem = nz % nranks;
    const size_t nzl = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t z0 = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    const int rankLo = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int rankHi = (rank < nranks - 1) ? rank + 1 : MPI_PROC_NULL;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPUs/node: %d\n", nranks, omp_get_max_threads(),
               deviceCount);
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
    const size_t plane = nx * ny;
    const size_t localSize = plane * nzl;
    const size_t localAlloc = plane * (nzl + 2);  // incl. two halo planes

    if (plane > static_cast<size_t>(INT32_MAX)) {
        if (rank == 0) fprintf(stderr, "Error: XY plane too large for MPI datatype\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Datatype planeType;
    MPI_CHECK(MPI_Type_contiguous(static_cast<int>(plane), MPI_DOUBLE, &planeType));
    MPI_CHECK(MPI_Type_commit(&planeType));

    // Allocate arrays (host slab + device slabs with halos)
    std::vector<double> hostField(localSize);

    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, localAlloc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, localAlloc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, localAlloc * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cold, 0, localAlloc * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cnew, 0, localAlloc * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu, 0, localAlloc * sizeof(double)));

    // Pinned host staging buffers for halo exchange
    double *h_sendLo = nullptr, *h_sendHi = nullptr, *h_recvLo = nullptr, *h_recvHi = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_sendLo, plane * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_sendHi, plane * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recvLo, plane * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recvHi, plane * sizeof(double), cudaHostAllocDefault));

    // Boundary work runs on a high-priority stream so its blocks are scheduled
    // ahead of the (much larger) interior kernel and the halo exchange can
    // start as early as possible.
    cudaStream_t sBnd, sInt;
    int prioLeast = 0, prioGreatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLeast, &prioGreatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sBnd, cudaStreamNonBlocking, prioGreatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sInt, cudaStreamNonBlocking, prioLeast));
    cudaEvent_t evInner, evInt;
    CUDA_CHECK(cudaEventCreateWithFlags(&evInner, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evInt, cudaEventDisableTiming));

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(hostField, nx, ny, nz, z0, nzl);
    CUDA_CHECK(cudaMemcpy(d_cold + plane, hostField.data(), localSize * sizeof(double), cudaMemcpyHostToDevice));

    // Initial halo exchange of the concentration field
    {
        const size_t lastOwned = plane * nzl;
        CUDA_CHECK(cudaMemcpy(h_sendLo, d_cold + plane, plane * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sendHi, d_cold + lastOwned, plane * sizeof(double), cudaMemcpyDeviceToHost));
        exchangeHalos(h_sendLo, h_sendHi, h_recvLo, h_recvHi, rankLo, rankHi, planeType);
        if (rankLo != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpy(d_cold, h_recvLo, plane * sizeof(double), cudaMemcpyHostToDevice));
        if (rankHi != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpy(d_cold + plane * (nzl + 1), h_recvHi, plane * sizeof(double),
                                  cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(evInner, sBnd));
    CUDA_CHECK(cudaEventRecord(evInt, sInt));

    // Kernel launch configuration
    const bool useRecip = reciprocalIsExact(dx) && reciprocalIsExact(dy) && reciprocalIsExact(dz);
    const StencilParams params{nx, ny, nz, z0, nzl, dx, dy, dz, 1.0 / (dx * dx), 1.0 / (dy * dy), 1.0 / (dz * dz)};
    const dim3 block(32, 8, 1);
    const unsigned gx = static_cast<unsigned>((nx + block.x - 1) / block.x);
    const unsigned gy = static_cast<unsigned>((ny + block.y - 1) / block.y);
    // Owned planes are split into three stripes per phase:
    //   outer    : planes 0 and nzl-1 (read the halos, produce the planes we send)
    //   inner    : planes 1 and nzl-2 (needed by the next phase's outer planes)
    //   interior : planes [2, nzl-2)  (bulk work, overlaps the halo exchange)
    const bool hasOuterHi = nzl >= 2;
    const bool hasInnerLo = nzl >= 3;
    const bool hasInnerHi = nzl >= 4;
    const size_t intBegin = 2;
    const size_t nIntPlanes = (nzl >= 5) ? nzl - 4 : 0;
    const dim3 gridBnd(gx, gy, 1);
    const dim3 gridInt(gx, gy, static_cast<unsigned>((nIntPlanes + kZChunk - 1) / kZChunk));

    const size_t offLoHalo = 0;
    const size_t offFirst = plane;
    const size_t offLast = plane * nzl;
    const size_t offHiHalo = plane * (nzl + 1);

    // Software pipeline for one phase (a phase computes `field` from the field
    // produced by the previous phase).  launchKernel(grid, stream, ozBegin, ozCount)
    // computes owned planes [ozBegin, ozBegin + ozCount).
    //
    //   sBnd : outer -> D2H send planes -> [host: MPI exchange] -> H2D halos
    //          -> (wait interior of previous phase) -> inner -> evInner
    //   sInt : (wait inner of previous phase) -> interior -> evInt
    //
    // The outer planes depend only on work already queued on sBnd (previous
    // outer/inner planes and halos), so the exchange starts immediately; the
    // interior kernel runs concurrently and hides the communication.
    auto runPhase = [&](auto&& launchKernel, double* field) {
        // Outer planes and staging of the send buffers.
        launchKernel(gridBnd, sBnd, size_t{0}, size_t{1});
        if (hasOuterHi) launchKernel(gridBnd, sBnd, nzl - 1, size_t{1});
        CUDA_CHECK(cudaMemcpyAsync(h_sendLo, field + offFirst, plane * sizeof(double), cudaMemcpyDeviceToHost, sBnd));
        CUDA_CHECK(cudaMemcpyAsync(h_sendHi, field + offLast, plane * sizeof(double), cudaMemcpyDeviceToHost, sBnd));

        // Interior planes: need the previous phase's inner planes (evInner).
        CUDA_CHECK(cudaStreamWaitEvent(sInt, evInner, 0));
        if (nIntPlanes > 0) launchKernel(gridInt, sInt, intBegin, nIntPlanes);
        CUDA_CHECK(cudaEventRecord(evInt, sInt));

        // Halo exchange through pinned host buffers (overlaps the interior kernel).
        CUDA_CHECK(cudaStreamSynchronize(sBnd));
        exchangeHalos(h_sendLo, h_sendHi, h_recvLo, h_recvHi, rankLo, rankHi, planeType);
        if (rankLo != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(field + offLoHalo, h_recvLo, plane * sizeof(double), cudaMemcpyHostToDevice, sBnd));
        if (rankHi != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(field + offHiHalo, h_recvHi, plane * sizeof(double), cudaMemcpyHostToDevice, sBnd));

        // Inner planes: need the previous phase's interior planes (evInt).
        CUDA_CHECK(cudaStreamWaitEvent(sBnd, evInt, 0));
        if (hasInnerLo) launchKernel(gridBnd, sBnd, size_t{1}, size_t{1});
        if (hasInnerHi) launchKernel(gridBnd, sBnd, nzl - 2, size_t{1});
        CUDA_CHECK(cudaEventRecord(evInner, sBnd));
    };

    auto launchMu = [&](const dim3& grid, cudaStream_t s, size_t ozBegin, size_t ozCount) {
        if (useRecip)
            computeChemicalPotentialKernel<true><<<grid, block, 0, s>>>(d_cold, d_mu, params, gamma, e_AA, e_BB, e_AB,
                                                                        ozBegin, ozCount);
        else
            computeChemicalPotentialKernel<false><<<grid, block, 0, s>>>(d_cold, d_mu, params, gamma, e_AA, e_BB,
                                                                         e_AB, ozBegin, ozCount);
        CUDA_CHECK(cudaGetLastError());
    };
    auto launchUpdate = [&](const dim3& grid, cudaStream_t s, size_t ozBegin, size_t ozCount) {
        if (useRecip)
            cahnHilliardUpdateKernel<true><<<grid, block, 0, s>>>(d_cnew, d_cold, d_mu, params, D, dt, ozBegin, ozCount);
        else
            cahnHilliardUpdateKernel<false><<<grid, block, 0, s>>>(d_cnew, d_cold, d_mu, params, D, dt, ozBegin,
                                                                   ozCount);
        CUDA_CHECK(cudaGetLastError());
    };

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        runPhase(launchMu, d_mu);

        // Update concentration
        runPhase(launchUpdate, d_cnew);

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
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

    // Bring the local slab back to the host
    CUDA_CHECK(cudaMemcpy(hostField.data(), d_cold + plane, localSize * sizeof(double), cudaMemcpyDeviceToHost));

    // Print results for external validation (gathered on rank 0 in global order)
    if (printResults) {
        std::vector<double> full;
        std::vector<int> counts(nranks), displs(nranks);
        for (int r = 0; r < nranks; ++r) {
            const size_t rnzl = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            const size_t rz0 = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem);
            counts[r] = static_cast<int>(rnzl);
            displs[r] = static_cast<int>(rz0);
        }
        if (rank == 0) full.resize(gridSize);
        MPI_CHECK(MPI_Gatherv(hostField.data(), static_cast<int>(nzl), planeType, rank == 0 ? full.data() : nullptr,
                              counts.data(), displs.data(), planeType, 0, MPI_COMM_WORLD));
        if (rank == 0) print_results(full, "Concentration");
    }

    int exitCode = 0;

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(hostField, rank);

        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(evInner));
    CUDA_CHECK(cudaEventDestroy(evInt));
    CUDA_CHECK(cudaStreamDestroy(sBnd));
    CUDA_CHECK(cudaStreamDestroy(sInt));
    CUDA_CHECK(cudaFreeHost(h_sendLo));
    CUDA_CHECK(cudaFreeHost(h_sendHi));
    CUDA_CHECK(cudaFreeHost(h_recvLo));
    CUDA_CHECK(cudaFreeHost(h_recvHi));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    MPI_CHECK(MPI_Type_free(&planeType));
    MPI_Finalize();
    return exitCode;
}
