// Cahn-Hilliard phase separation benchmark.
//
// Hybrid parallelization:
//   * MPI   -- 1D domain decomposition along Z, one rank per GPU. A two-plane
//              ghost region lets each rank recompute the chemical potential on
//              its innermost ghost plane, so a time step needs a single halo
//              exchange. The planes the neighbours need are computed first and
//              the exchange (device->host, MPI, host->device) then runs in the
//              shadow of both interior sweeps.
//   * CUDA  -- both stencil sweeps (chemical potential, concentration update)
//              run entirely on the device; each thread walks a short run of Z
//              planes keeping the stencil's Z neighbours in registers.
//   * OpenMP-- host side work (field initialization, validation reductions).

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

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        const cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,       \
                    __LINE__);                                                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                         \
        }                                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Number of Z planes processed by a single thread (rolling registers).
static constexpr int kZTile = 4;

// Depth of the Z ghost region on each side of a rank's slab. Two planes of the
// concentration field let every rank recompute the chemical potential on its
// first ghost plane instead of exchanging mu separately, so a time step needs
// only a single halo exchange.
static constexpr int kGhost = 2;

// Local plane zl (which may be -kGhost .. local_nz + kGhost - 1) lives at
// (zl + kGhost) * nx * ny. Ghost planes hold either the neighbour rank's
// boundary planes or, at a global boundary, a copy of the own boundary plane,
// which reproduces the clamped boundary condition of the serial code exactly.
__device__ __forceinline__ size_t lidx(const int x, const int y, const int zl, const int nx, const int ny) {
    return static_cast<size_t>(zl + kGhost) * (static_cast<size_t>(nx) * ny) + static_cast<size_t>(y) * nx + x;
}

// mu = 4.5*((c+1)e_AA + (c-1)e_BB - 2 c e_AB) + 3 c + c^3 - gamma * lap(c)
__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        const int nx, const int ny, const int z0, const int z1,
                                        const double idx2, const double idy2, const double idz2,
                                        const double gamma, const double e_AA, const double e_BB,
                                        const double e_AB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const int zb = z0 + static_cast<int>(blockIdx.z) * kZTile;
    if (zb >= z1) return;
    const int ze = min(zb + kZTile, z1);

    // Neighbour offsets relative to the current cell (clamped boundaries).
    const int dxp = (x < nx - 1) ? 1 : 0;
    const int dxn = (x > 0) ? -1 : 0;
    const int dyp = ((y < ny - 1) ? 1 : 0) * nx;
    const int dyn = ((y > 0) ? -1 : 0) * nx;

    const size_t plane = static_cast<size_t>(nx) * ny;
    size_t base = lidx(x, y, zb, nx, ny);

    double cm1 = c[base - plane];
    double c0 = c[base];

    for (int z = zb; z < ze; ++z, base += plane) {
        const double cp1 = c[base + plane];
        const double cv = c0;

        const double cxx = (c[base + dxp] + c[base + dxn] - 2.0 * cv) * idx2;
        const double cyy = (c[base + dyp] + c[base + dyn] - 2.0 * cv) * idy2;
        const double czz = (cp1 + cm1 - 2.0 * cv) * idz2;

        mu[base] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                   + 3.0 * cv + cv * cv * cv - gamma * (cxx + cyy + czz);

        cm1 = cv;
        c0 = cp1;
    }
}

// cnew = cold + dt * D * lap(mu)
__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                             const double* __restrict__ mu, const int nx, const int ny,
                             const int z0, const int z1, const double D, const double dt,
                             const double idx2, const double idy2, const double idz2) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const int zb = z0 + static_cast<int>(blockIdx.z) * kZTile;
    if (zb >= z1) return;
    const int ze = min(zb + kZTile, z1);

    // Neighbour offsets relative to the current cell (clamped boundaries).
    const int dxp = (x < nx - 1) ? 1 : 0;
    const int dxn = (x > 0) ? -1 : 0;
    const int dyp = ((y < ny - 1) ? 1 : 0) * nx;
    const int dyn = ((y > 0) ? -1 : 0) * nx;

    const size_t plane = static_cast<size_t>(nx) * ny;
    size_t base = lidx(x, y, zb, nx, ny);

    double mm1 = mu[base - plane];
    double m0 = mu[base];

    for (int z = zb; z < ze; ++z, base += plane) {
        const double mp1 = mu[base + plane];
        const double mv = m0;

        const double cxx = (mu[base + dxp] + mu[base + dxn] - 2.0 * mv) * idx2;
        const double cyy = (mu[base + dyp] + mu[base + dyn] - 2.0 * mv) * idy2;
        const double czz = (mp1 + mm1 - 2.0 * mv) * idz2;

        cnew[base] = cold[base] + dt * D * (cxx + cyy + czz);

        mm1 = mv;
        m0 = mp1;
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

// Initialize the local slab of the concentration field (global semantics kept).
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t nz, const size_t z_offset, const size_t local_nz) {
    const size_t vol = nx * ny * nz;
    const size_t plane = nx * ny;

#pragma omp parallel for schedule(static)
    for (long long zl = 0; zl < static_cast<long long>(local_nz); ++zl) {
        const size_t gz = z_offset + static_cast<size_t>(zl);
        double* dst = c.data() + (static_cast<size_t>(zl) + kGhost) * plane;
        for (size_t i = 0; i < plane; ++i) {
            // Generate pseudo-random value in [-1, 1]
            const size_t linear_id = gz * plane + i;
            const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
            dst[i] = -1.0 + 2.0 * pseudo;
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    bool bad = false;
#pragma omp parallel for schedule(static) reduction(|| : bad)
    for (long long i = 0; i < static_cast<long long>(c.size()); ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) bad = true;
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
    for (long long i = 0; i < static_cast<long long>(c.size()); ++i) {
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

// Everything needed to exchange the Z ghost regions of a device array.
struct HaloContext {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    int up = MPI_PROC_NULL;   // neighbour towards higher z
    int down = MPI_PROC_NULL; // neighbour towards lower z
    int nx = 0, ny = 0, local_nz = 0;
    size_t plane = 0;         // elements per XY plane
    size_t plane_bytes = 0;
    size_t halo_bytes = 0;    // kGhost planes
    double* h_send_lo = nullptr; // pinned staging buffers (kGhost planes each)
    double* h_send_hi = nullptr;
    double* h_recv_lo = nullptr;
    double* h_recv_hi = nullptr;
    cudaStream_t stream_halo = nullptr;
    cudaStream_t stream_bulk = nullptr;
    cudaEvent_t ev_boundary = nullptr; // boundary planes ready for staging
};

inline double* planePtr(double* base, const int zl, const size_t plane) {
    return base + (static_cast<size_t>(zl + kGhost)) * plane;
}

// Mirror the outermost owned plane into the first ghost plane wherever the slab
// touches a global domain boundary; this reproduces the clamped boundary
// condition of the serial code. Enqueued on `stream`.
void mirrorGlobalBoundaries(const HaloContext& ctx, double* d_arr, cudaStream_t stream) {
    if (ctx.down == MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(planePtr(d_arr, -1, ctx.plane), planePtr(d_arr, 0, ctx.plane),
                                   ctx.plane_bytes, cudaMemcpyDeviceToDevice, stream));
    }
    if (ctx.up == MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(planePtr(d_arr, ctx.local_nz, ctx.plane),
                                   planePtr(d_arr, ctx.local_nz - 1, ctx.plane), ctx.plane_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

// Copy the kGhost boundary planes that the neighbours need into pinned host
// memory. The planes of one side are contiguous, so one copy per side suffices.
void stageSendPlanes(const HaloContext& ctx, double* d_arr, cudaStream_t stream) {
    if (ctx.down != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(ctx.h_send_lo, planePtr(d_arr, 0, ctx.plane), ctx.halo_bytes,
                                   cudaMemcpyDeviceToHost, stream));
    }
    if (ctx.up != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(ctx.h_send_hi, planePtr(d_arr, ctx.local_nz - kGhost, ctx.plane),
                                   ctx.halo_bytes, cudaMemcpyDeviceToHost, stream));
    }
}

// Exchange the staged planes with the neighbours and upload the received ghost
// regions back to the device (upload enqueued on `stream`).
void exchangeAndUpload(const HaloContext& ctx, double* d_arr, cudaStream_t stream) {
    const int count = static_cast<int>(ctx.plane) * kGhost;
    MPI_Request reqs[4];
    int nreq = 0;
    if (ctx.down != MPI_PROC_NULL) {
        MPI_Irecv(ctx.h_recv_lo, count, MPI_DOUBLE, ctx.down, 1, ctx.comm, &reqs[nreq++]);
        MPI_Isend(ctx.h_send_lo, count, MPI_DOUBLE, ctx.down, 0, ctx.comm, &reqs[nreq++]);
    }
    if (ctx.up != MPI_PROC_NULL) {
        MPI_Irecv(ctx.h_recv_hi, count, MPI_DOUBLE, ctx.up, 0, ctx.comm, &reqs[nreq++]);
        MPI_Isend(ctx.h_send_hi, count, MPI_DOUBLE, ctx.up, 1, ctx.comm, &reqs[nreq++]);
    }
    if (nreq > 0) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    if (ctx.down != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(planePtr(d_arr, -kGhost, ctx.plane), ctx.h_recv_lo,
                                   ctx.halo_bytes, cudaMemcpyHostToDevice, stream));
    }
    if (ctx.up != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(planePtr(d_arr, ctx.local_nz, ctx.plane), ctx.h_recv_hi,
                                   ctx.halo_bytes, cudaMemcpyHostToDevice, stream));
    }
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (world_rank == 0) {
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

    // The stencils divide by the squared grid spacings; hoisting the reciprocals
    // out of the kernels avoids six FP64 divisions per cell and time step (FP64
    // division is extremely expensive on the target GPUs). For the unit spacings
    // used here this is bit-identical to the division.
    const double idx2 = 1.0 / (dx * dx);
    const double idy2 = 1.0 / (dy * dy);
    const double idz2 = 1.0 / (dz * dz);

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;

    // --- 1D decomposition along Z; surplus ranks stay idle -------------------
    // Every participating rank needs at least kGhost planes so that a neighbour
    // can be served from owned data only.
    const int active_ranks = std::max<int>(
        1, static_cast<int>(std::min<size_t>(nz / kGhost, static_cast<size_t>(world_size))));
    const bool active = world_rank < active_ranks;

    size_t local_nz = 0, z_offset = 0;
    if (active) {
        const size_t base = nz / active_ranks;
        const size_t rem = nz % active_ranks;
        local_nz = base + (static_cast<size_t>(world_rank) < rem ? 1 : 0);
        z_offset = base * world_rank + std::min<size_t>(world_rank, rem);
    }

    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, world_rank, &comm);

    // --- Bind one GPU per rank ---------------------------------------------
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node_comm);
    int node_rank = 0;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_free(&node_comm);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (world_rank == 0) fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(node_rank % device_count));

    std::vector<double> global_c; // assembled on rank 0 for output/validation
    long long elapsed_ms = 0;

    if (active) {
        HaloContext ctx;
        ctx.comm = comm;
        MPI_Comm_rank(comm, &ctx.rank);
        MPI_Comm_size(comm, &ctx.size);
        ctx.down = (ctx.rank > 0) ? ctx.rank - 1 : MPI_PROC_NULL;
        ctx.up = (ctx.rank < ctx.size - 1) ? ctx.rank + 1 : MPI_PROC_NULL;
        ctx.nx = static_cast<int>(nx);
        ctx.ny = static_cast<int>(ny);
        ctx.local_nz = static_cast<int>(local_nz);
        ctx.plane = plane;
        ctx.plane_bytes = plane * sizeof(double);

        ctx.halo_bytes = ctx.plane_bytes * kGhost;

        const size_t padded = (local_nz + 2 * kGhost) * plane;
        const size_t padded_bytes = padded * sizeof(double);

        // Host-side slab (incl. ghost planes) used for initialization.
        std::vector<double> host_c(padded);
        initializeConcentration(host_c, nx, ny, nz, z_offset, local_nz);

        double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
        CUDA_CHECK(cudaMalloc(&d_cold, padded_bytes));
        CUDA_CHECK(cudaMalloc(&d_cnew, padded_bytes));
        CUDA_CHECK(cudaMalloc(&d_mu, padded_bytes));
        CUDA_CHECK(cudaMemcpy(d_cold, host_c.data(), padded_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_cnew, 0, padded_bytes));
        CUDA_CHECK(cudaMemset(d_mu, 0, padded_bytes));

        CUDA_CHECK(cudaHostAlloc(&ctx.h_send_lo, ctx.halo_bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&ctx.h_send_hi, ctx.halo_bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&ctx.h_recv_lo, ctx.halo_bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&ctx.h_recv_hi, ctx.halo_bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaStreamCreate(&ctx.stream_halo));
        CUDA_CHECK(cudaStreamCreate(&ctx.stream_bulk));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.ev_boundary, cudaEventDisableTiming));

        const dim3 block(64, 4, 1);
        const dim3 gridXY((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, 1);
        auto zGrid = [&](const int n) {
            return dim3(gridXY.x, gridXY.y, static_cast<unsigned>((std::max(n, 1) + kZTile - 1) / kZTile));
        };

        const int lnz = ctx.local_nz;
        const bool has_neighbour = (ctx.down != MPI_PROC_NULL) || (ctx.up != MPI_PROC_NULL);

        // mu is evaluated on the owned planes plus, towards a neighbour, on the
        // innermost ghost plane, so that the concentration update needs no
        // second halo exchange.
        const int mu_lo = (ctx.down != MPI_PROC_NULL) ? -1 : 0;
        const int mu_hi = (ctx.up != MPI_PROC_NULL) ? lnz + 1 : lnz;

        // Planes exchanged after the update.
        const int lo_end = kGhost + 1;          // mu planes feeding the low boundary update
        const int hi_start = lnz - (kGhost + 1);
        // Slabs thinner than this are not worth pipelining (and the ranges below
        // would overlap); they take the simple non-overlapped path.
        const bool pipelined = has_neighbour && (lnz >= 3 * kGhost + 2);

        auto muSweep = [&](const int z0, const int z1, cudaStream_t stream) {
            if (z1 <= z0) return;
            chemicalPotentialKernel<<<zGrid(z1 - z0), block, 0, stream>>>(
                d_cold, d_mu, ctx.nx, ctx.ny, z0, z1, idx2, idy2, idz2, gamma, e_AA, e_BB, e_AB);
        };
        auto updateSweep = [&](const int z0, const int z1, cudaStream_t stream) {
            if (z1 <= z0) return;
            updateKernel<<<zGrid(z1 - z0), block, 0, stream>>>(
                d_cnew, d_cold, d_mu, ctx.nx, ctx.ny, z0, z1, D, dt, idx2, idy2, idz2);
        };

        // Establish the initial ghost planes of the concentration field.
        mirrorGlobalBoundaries(ctx, d_cold, ctx.stream_halo);
        stageSendPlanes(ctx, d_cold, ctx.stream_halo);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream_halo));
        exchangeAndUpload(ctx, d_cold, ctx.stream_halo);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream_halo));

        MPI_Barrier(comm);
        const auto start = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < iterations; ++t) {
            if (pipelined) {
                // Produce the planes the neighbours need as early as possible:
                // chemical potential and update are evaluated on the few
                // boundary planes first, so the exchange can then run in the
                // shadow of both interior sweeps.
                muSweep(mu_lo, lo_end, ctx.stream_bulk);
                muSweep(hi_start, mu_hi, ctx.stream_bulk);
                mirrorGlobalBoundaries(ctx, d_mu, ctx.stream_bulk);
                updateSweep(0, kGhost, ctx.stream_bulk);
                updateSweep(lnz - kGhost, lnz, ctx.stream_bulk);
                mirrorGlobalBoundaries(ctx, d_cnew, ctx.stream_bulk);
                CUDA_CHECK(cudaEventRecord(ctx.ev_boundary, ctx.stream_bulk));

                // Staging runs on the copy engine, ordered only after the
                // boundary planes, hence concurrently with the interior sweeps.
                CUDA_CHECK(cudaStreamWaitEvent(ctx.stream_halo, ctx.ev_boundary, 0));
                stageSendPlanes(ctx, d_cnew, ctx.stream_halo);

                muSweep(lo_end, hi_start, ctx.stream_bulk);
                updateSweep(kGhost, lnz - kGhost, ctx.stream_bulk);

                CUDA_CHECK(cudaStreamSynchronize(ctx.stream_halo));
                exchangeAndUpload(ctx, d_cnew, ctx.stream_halo);
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream_halo));
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream_bulk));
            } else {
                muSweep(mu_lo, mu_hi, ctx.stream_bulk);
                mirrorGlobalBoundaries(ctx, d_mu, ctx.stream_bulk);
                updateSweep(0, lnz, ctx.stream_bulk);
                mirrorGlobalBoundaries(ctx, d_cnew, ctx.stream_bulk);
                if (has_neighbour) {
                    stageSendPlanes(ctx, d_cnew, ctx.stream_bulk);
                    CUDA_CHECK(cudaStreamSynchronize(ctx.stream_bulk));
                    exchangeAndUpload(ctx, d_cnew, ctx.stream_halo);
                    CUDA_CHECK(cudaStreamSynchronize(ctx.stream_halo));
                } else {
                    CUDA_CHECK(cudaStreamSynchronize(ctx.stream_bulk));
                }
            }

            std::swap(d_cold, d_cnew);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);
        const auto end = std::chrono::high_resolution_clock::now();
        elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // --- assemble the global field on rank 0 ---------------------------
        if (printResults || validate) {
            CUDA_CHECK(cudaMemcpy(host_c.data(), d_cold, padded_bytes, cudaMemcpyDeviceToHost));

            std::vector<int> counts, displs;
            if (ctx.rank == 0) {
                counts.resize(ctx.size);
                displs.resize(ctx.size);
            }
            const int local_count = static_cast<int>(local_nz * plane);
            MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
            if (ctx.rank == 0) {
                int off = 0;
                for (int r = 0; r < ctx.size; ++r) {
                    displs[r] = off;
                    off += counts[r];
                }
                global_c.resize(gridSize);
            }
            MPI_Gatherv(host_c.data() + kGhost * plane, local_count, MPI_DOUBLE, global_c.data(),
                        counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
        }

        CUDA_CHECK(cudaFree(d_cold));
        CUDA_CHECK(cudaFree(d_cnew));
        CUDA_CHECK(cudaFree(d_mu));
        CUDA_CHECK(cudaFreeHost(ctx.h_send_lo));
        CUDA_CHECK(cudaFreeHost(ctx.h_send_hi));
        CUDA_CHECK(cudaFreeHost(ctx.h_recv_lo));
        CUDA_CHECK(cudaFreeHost(ctx.h_recv_hi));
        CUDA_CHECK(cudaStreamDestroy(ctx.stream_halo));
        CUDA_CHECK(cudaStreamDestroy(ctx.stream_bulk));
        CUDA_CHECK(cudaEventDestroy(ctx.ev_boundary));
        MPI_Comm_free(&comm);
    }

    long long max_ms = 0;
    MPI_Allreduce(&elapsed_ms, &max_ms, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    int exit_code = 0;
    if (world_rank == 0) {
        printf("Initializing concentration field...\n");
        printf("Running Cahn-Hilliard simulation...\n");
        printf("Computation time: %lld ms\n", max_ms);

        // Calculate performance
        const double cellUpdates = (double)gridSize * iterations;
        const double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        // Print results for external validation
        if (printResults) {
            print_results(global_c, "Concentration");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(global_c, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
