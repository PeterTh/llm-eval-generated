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

// Query for CUDA-aware MPI support (Open MPI extension)
#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h>
#endif

// ---------------------------------------------------------------------------
// Hybrid MPI + OpenMP + CUDA implementation.
//
// Parallelization strategy:
//   * MPI  : 1D domain decomposition along Z. Each rank owns a contiguous slab
//            of Z planes and drives one GPU. A halo of 2 planes on each side is
//            exchanged once per time step, which is enough to compute the
//            chemical potential in the first ghost plane locally and therefore
//            to evaluate the second (update) stencil on all owned cells.
//   * CUDA : both stencil sweeps run as barrier-free kernels that march along Z
//            keeping the Z-neighbours in registers and obtaining the
//            X-neighbours through warp shuffles.
//   * OpenMP: host side work (field initialization, validation reductions,
//            halo staging) is threaded.
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        const cudaError_t err_ = (call);                                                                               \
        if (err_ != cudaSuccess) {                                                                                     \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__);                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                              \
        }                                                                                                              \
    } while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Tile geometry for the stencil kernels
#define TX 32
#define TY 8

static_assert(TX == 32, "the x-neighbour warp shuffles require a 32-wide tile");

enum StencilMode { MODE_MU = 0, MODE_UPDATE = 1 };

// Fused tiled stencil kernel.
//   MODE_MU     : out = chemical potential of "in" (== c)
//   MODE_UPDATE : out = aux + dtD * laplacian(in) (in == mu, aux == c)
//
// Local Z indices run over [0, nzloc+4); owned planes are [2, nzloc+2).
// The global Z coordinate of local plane k is gzoff + k, which drives the
// clamped boundary condition at the physical domain boundaries.
template <int MODE>
__global__ __launch_bounds__(TX* TY) void stencilKernel(const double* __restrict__ in, const double* __restrict__ aux,
                                                        double* __restrict__ out, const int nx, const int ny,
                                                        const int kbeg, const int kend, const int kchunk,
                                                        const long gzoff, const long nzg, const double idx2,
                                                        const double idy2, const double idz2, const double gamma,
                                                        const double e_AA, const double e_BB, const double e_AB,
                                                        const double dtD) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * TX + tx;
    const int y = blockIdx.y * TY + ty;

    // Threads outside the domain still evaluate (clamped) coordinates so that
    // they feed the correct halo values to their in-range neighbours.
    const int xc = min(x, nx - 1);
    const int yc = min(y, ny - 1);
    const bool active = (x < nx) && (y < ny);

    const size_t plane = (size_t)nx * (size_t)ny;
    const size_t row = (size_t)yc * (size_t)nx;

    const int k0 = kbeg + (int)blockIdx.z * kchunk;
    const int k1 = min(k0 + kchunk, kend);
    if (k0 >= k1) return;

    const int xm = max(xc - 1, 0);
    const int xp = min(xc + 1, nx - 1);
    const size_t rowm = (size_t)max(yc - 1, 0) * (size_t)nx;
    const size_t rowp = (size_t)min(yc + 1, ny - 1) * (size_t)nx;

    double cm = in[(size_t)(k0 - 1) * plane + row + xc];
    double c0 = in[(size_t)k0 * plane + row + xc];

    for (int k = k0; k < k1; ++k) {
        const size_t base = (size_t)k * plane;
        const double cp = in[base + plane + row + xc];
        // x-neighbours come from the warp (one warp == one row of 32 cells),
        // y-neighbours from L1/L2, z-neighbours from the sweep registers.
        double wl = __shfl_up_sync(0xffffffffu, c0, 1);
        double wr = __shfl_down_sync(0xffffffffu, c0, 1);
        if (tx == 0) wl = in[base + row + xm];
        if (tx == TX - 1) wr = in[base + row + xp];
        const double yl = in[base + rowm + xc];
        const double yr = in[base + rowp + xc];

        const long gz = gzoff + k;
        const double up = (gz < nzg - 1) ? cp : c0;
        const double dn = (gz > 0) ? cm : c0;
        const double two_c0 = 2.0 * c0;
        const double cxx = (wr + wl - two_c0) * idx2;
        const double cyy = (yr + yl - two_c0) * idy2;
        const double czz = (up + dn - two_c0) * idz2;
        const double lap = cxx + cyy + czz;
        if (active) {
            const size_t o = base + (size_t)y * (size_t)nx + (size_t)x;
            if (MODE == MODE_MU) {
                out[o] = 4.5 * ((c0 + 1.0) * e_AA + (c0 - 1.0) * e_BB - two_c0 * e_AB) + 3.0 * c0 + c0 * c0 * c0 -
                         gamma * lap;
            } else {
                out[o] = aux[o] + dtD * lap;
            }
        }
        cm = c0;
        c0 = cp;
    }
}

// Initialize the local slab of the concentration field (host, OpenMP threaded)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t z0, const size_t nzloc) {
    const size_t vol = nx * ny * nz;

#pragma omp parallel for schedule(static)
    for (size_t z = 0; z < nzloc; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = (z + z0) * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    int bad = 0;
    const size_t n = c.size();
#pragma omp parallel for schedule(static) reduction(| : bad)
    for (size_t i = 0; i < n; ++i) {
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
    for (size_t i = 0; i < n; ++i) {
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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (worldRank == 0) {
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

    // ---- MPI decomposition along Z ---------------------------------------
    // Every active rank needs at least 2 owned planes so that a 2-plane halo
    // can be served by its immediate neighbour.
    int nActive = static_cast<int>(std::min<size_t>(static_cast<size_t>(worldSize), std::max<size_t>(nz / 2, 1)));
    const bool activeRank = (worldRank < nActive);

    MPI_Comm simComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, activeRank ? 0 : MPI_UNDEFINED, worldRank, &simComm);

    size_t z0 = 0;
    size_t nzloc = 0;
    std::vector<int> allCounts(worldSize, 0);
    std::vector<size_t> allDispls(worldSize, 0);
    {
        // Block distribution of nz planes over nActive ranks
        size_t off = 0;
        for (int r = 0; r < nActive; ++r) {
            const size_t cnt = nz / nActive + (static_cast<size_t>(r) < nz % nActive ? 1 : 0);
            if (r == worldRank) {
                z0 = off;
                nzloc = cnt;
            }
            allCounts[r] = static_cast<int>(cnt * nx * ny);
            allDispls[r] = off * nx * ny;
            off += cnt;
        }
    }

    // ---- GPU selection: one GPU per rank, round-robin within the node -----
    if (activeRank) {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(simComm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
        int nodeRank = 0;
        MPI_Comm_rank(nodeComm, &nodeRank);
        MPI_Comm_free(&nodeComm);

        int devCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devCount));
        if (devCount == 0) {
            fprintf(stderr, "No CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(nodeRank % devCount));
        CUDA_CHECK(cudaFree(nullptr)); // establish context before timing
    }

    bool cudaAwareMPI = false;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    cudaAwareMPI = (MPIX_Query_cuda_support() == 1);
#endif

    if (worldRank == 0) {
        printf("MPI ranks: %d (active: %d), OpenMP threads/rank: %d, CUDA-aware MPI: %s\n", worldSize, nActive,
               omp_get_max_threads(), cudaAwareMPI ? "yes" : "no");
        printf("Initializing concentration field...\n");
    }

    // ---- Allocation -------------------------------------------------------
    const size_t plane = nx * ny;
    const size_t nzalloc = nzloc + 4;           // 2 ghost planes on each side
    const size_t localBytes = plane * nzalloc * sizeof(double);
    const size_t ownedElems = plane * nzloc;

    double* d_c = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    double *h_sendLo = nullptr, *h_sendHi = nullptr, *h_recvLo = nullptr, *h_recvHi = nullptr;
    cudaStream_t s0 = nullptr, s1 = nullptr;
    cudaEvent_t evMuInt = nullptr;
    cudaEvent_t evMuBnd = nullptr;

    std::vector<double> hostSlab;

    int left = MPI_PROC_NULL;
    int right = MPI_PROC_NULL;

    if (activeRank) {
        int simRank = 0, simSize = 1;
        MPI_Comm_rank(simComm, &simRank);
        MPI_Comm_size(simComm, &simSize);
        left = (simRank > 0) ? simRank - 1 : MPI_PROC_NULL;
        right = (simRank + 1 < simSize) ? simRank + 1 : MPI_PROC_NULL;

        CUDA_CHECK(cudaMalloc(&d_c, localBytes));
        CUDA_CHECK(cudaMalloc(&d_cnew, localBytes));
        CUDA_CHECK(cudaMalloc(&d_mu, localBytes));
        CUDA_CHECK(cudaMemset(d_c, 0, localBytes));
        CUDA_CHECK(cudaMemset(d_cnew, 0, localBytes));
        CUDA_CHECK(cudaMemset(d_mu, 0, localBytes));

        CUDA_CHECK(cudaStreamCreate(&s0));
        CUDA_CHECK(cudaStreamCreate(&s1));
        CUDA_CHECK(cudaEventCreateWithFlags(&evMuInt, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evMuBnd, cudaEventDisableTiming));

        if (!cudaAwareMPI && simSize > 1) {
            CUDA_CHECK(cudaMallocHost(&h_sendLo, 2 * plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&h_sendHi, 2 * plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&h_recvLo, 2 * plane * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&h_recvHi, 2 * plane * sizeof(double)));
        }

        // Initialize on the host (OpenMP) and upload the owned slab
        hostSlab.resize(ownedElems);
        initializeConcentration(hostSlab, nx, ny, nz, z0, nzloc);
        CUDA_CHECK(cudaMemcpy(d_c + 2 * plane, hostSlab.data(), ownedElems * sizeof(double), cudaMemcpyHostToDevice));
    }

    if (worldRank == 0) printf("Running Cahn-Hilliard simulation...\n");

    // ---- Kernel launch configuration -------------------------------------
    const double idx2 = 1.0 / (dx * dx);
    const double idy2 = 1.0 / (dy * dy);
    const double idz2 = 1.0 / (dz * dz);
    const double dtD = dt * D;
    const long gzoff = static_cast<long>(z0) - 2;
    const long nzg = static_cast<long>(nz);

    const int nxi = static_cast<int>(nx);
    const int nyi = static_cast<int>(ny);
    const int nzl = static_cast<int>(nzloc);

    const dim3 block(TX, TY, 1);
    const int gx = (nxi + TX - 1) / TX;
    const int gy = (nyi + TY - 1) / TY;

    // Choose the Z chunk per block so that enough blocks are in flight
    int kchunkMu = 8;
    if (activeRank) {
        int smCount = 0;
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev));
        const int xyBlocks = gx * gy;
        const int wanted = std::max(1, 8 * smCount / std::max(1, xyBlocks));
        kchunkMu = std::max(1, std::min(32, (nzl + 3 + wanted - 1) / wanted));
    }

    // mu is additionally evaluated in the first ghost plane on each side that
    // has a neighbour, so the update stencil can be applied to all owned cells.
    const int muBeg = (left != MPI_PROC_NULL) ? 1 : 2;
    const int muEnd = (right != MPI_PROC_NULL) ? nzl + 3 : nzl + 2;
    // Interior mu range: depends exclusively on owned planes of c.
    const int muIntBeg = 4;
    const int muIntEnd = nzl;
    // Overlap boundary/interior only when there is enough interior work
    const bool splitUpdate = (left != MPI_PROC_NULL || right != MPI_PROC_NULL) && nzl >= 8;

    const size_t haloElems = 2 * plane;
    const size_t haloBytes = haloElems * sizeof(double);
    const int haloCount = static_cast<int>(haloElems);

    // Blocking halo exchange (used for the initial fill and for small slabs).
    auto exchangeHalos = [&](double* buf) {
        if (left == MPI_PROC_NULL && right == MPI_PROC_NULL) return;
        if (cudaAwareMPI) {
            CUDA_CHECK(cudaStreamSynchronize(s0));
            MPI_Sendrecv(buf + 2 * plane, haloCount, MPI_DOUBLE, left, 0,
                         buf + static_cast<size_t>(nzl + 2) * plane, haloCount, MPI_DOUBLE, right, 0, simComm,
                         MPI_STATUS_IGNORE);
            MPI_Sendrecv(buf + static_cast<size_t>(nzl) * plane, haloCount, MPI_DOUBLE, right, 1, buf, haloCount,
                         MPI_DOUBLE, left, 1, simComm, MPI_STATUS_IGNORE);
        } else {
            if (left != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(h_sendLo, buf + 2 * plane, haloBytes, cudaMemcpyDeviceToHost, s0));
            if (right != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(h_sendHi, buf + static_cast<size_t>(nzl) * plane, haloBytes,
                                           cudaMemcpyDeviceToHost, s0));
            CUDA_CHECK(cudaStreamSynchronize(s0));
            MPI_Sendrecv(h_sendLo, haloCount, MPI_DOUBLE, left, 0, h_recvHi, haloCount, MPI_DOUBLE, right, 0, simComm,
                         MPI_STATUS_IGNORE);
            MPI_Sendrecv(h_sendHi, haloCount, MPI_DOUBLE, right, 1, h_recvLo, haloCount, MPI_DOUBLE, left, 1, simComm,
                         MPI_STATUS_IGNORE);
            if (right != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(buf + static_cast<size_t>(nzl + 2) * plane, h_recvHi, haloBytes,
                                           cudaMemcpyHostToDevice, s0));
            if (left != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(buf, h_recvLo, haloBytes, cudaMemcpyHostToDevice, s0));
            CUDA_CHECK(cudaStreamSynchronize(s0));
        }
    };

    // Fill the ghost planes of the initial field before the first sweep
    if (activeRank) exchangeHalos(d_c);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (activeRank) {
        MPI_Request reqs[4];
        bool pending = false;

        for (int t = 0; t < iterations; ++t) {
            if (splitUpdate) {
                // ---- 1) Interior chemical potential: depends only on owned data,
                //         so it runs while the halo exchange of the previous step
                //         is still in flight.
                {
                    const int span = muIntEnd - muIntBeg;
                    const dim3 gi(gx, gy, (span + kchunkMu - 1) / kchunkMu);
                    stencilKernel<MODE_MU><<<gi, block, 0, s1>>>(d_c, nullptr, d_mu, nxi, nyi, muIntBeg, muIntEnd,
                                                                 kchunkMu, gzoff, nzg, idx2, idy2, idz2, gamma, e_AA,
                                                                 e_BB, e_AB, dtD);
                }
                CUDA_CHECK(cudaEventRecord(evMuInt, s1));

                // ---- 2) Land the incoming halos of d_c
                if (pending) {
                    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
                    pending = false;
                    if (right != MPI_PROC_NULL)
                        CUDA_CHECK(cudaMemcpyAsync(d_c + static_cast<size_t>(nzl + 2) * plane, h_recvHi, haloBytes,
                                                   cudaMemcpyHostToDevice, s0));
                    if (left != MPI_PROC_NULL)
                        CUDA_CHECK(cudaMemcpyAsync(d_c, h_recvLo, haloBytes, cudaMemcpyHostToDevice, s0));
                }

                // ---- 3) Chemical potential in the near-boundary planes
                {
                    const dim3 gb(gx, gy, 1);
                    stencilKernel<MODE_MU><<<gb, block, 0, s0>>>(d_c, nullptr, d_mu, nxi, nyi, muBeg, muIntBeg,
                                                                 muIntBeg - muBeg, gzoff, nzg, idx2, idy2, idz2, gamma,
                                                                 e_AA, e_BB, e_AB, dtD);
                    stencilKernel<MODE_MU><<<gb, block, 0, s0>>>(d_c, nullptr, d_mu, nxi, nyi, muIntEnd, muEnd,
                                                                 muEnd - muIntEnd, gzoff, nzg, idx2, idy2, idz2, gamma,
                                                                 e_AA, e_BB, e_AB, dtD);
                }
                CUDA_CHECK(cudaEventRecord(evMuBnd, s0));

                // ---- 4) Boundary update: produces the planes that have to be sent
                CUDA_CHECK(cudaStreamWaitEvent(s0, evMuInt, 0));
                {
                    const dim3 gb(gx, gy, 1);
                    stencilKernel<MODE_UPDATE><<<gb, block, 0, s0>>>(d_mu, d_c, d_cnew, nxi, nyi, 2, 4, 2, gzoff, nzg,
                                                                     idx2, idy2, idz2, gamma, e_AA, e_BB, e_AB, dtD);
                    stencilKernel<MODE_UPDATE><<<gb, block, 0, s0>>>(d_mu, d_c, d_cnew, nxi, nyi, nzl, nzl + 2, 2,
                                                                     gzoff, nzg, idx2, idy2, idz2, gamma, e_AA, e_BB,
                                                                     e_AB, dtD);
                }
                if (left != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(h_sendLo, d_cnew + 2 * plane, haloBytes, cudaMemcpyDeviceToHost, s0));
                if (right != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(h_sendHi, d_cnew + static_cast<size_t>(nzl) * plane, haloBytes,
                                               cudaMemcpyDeviceToHost, s0));

                // ---- 5) Interior update, concurrent with the outgoing exchange
                CUDA_CHECK(cudaStreamWaitEvent(s1, evMuBnd, 0));
                {
                    const int span = nzl - 4;
                    const dim3 gi(gx, gy, (span + kchunkMu - 1) / kchunkMu);
                    stencilKernel<MODE_UPDATE><<<gi, block, 0, s1>>>(d_mu, d_c, d_cnew, nxi, nyi, 4, nzl, kchunkMu,
                                                                     gzoff, nzg, idx2, idy2, idz2, gamma, e_AA, e_BB,
                                                                     e_AB, dtD);
                }

                // ---- 6) Post the non-blocking exchange for the next step
                if (t + 1 < iterations) {
                    CUDA_CHECK(cudaStreamSynchronize(s0));
                    MPI_Irecv(h_recvHi, haloCount, MPI_DOUBLE, right, 0, simComm, &reqs[0]);
                    MPI_Irecv(h_recvLo, haloCount, MPI_DOUBLE, left, 1, simComm, &reqs[1]);
                    MPI_Isend(h_sendLo, haloCount, MPI_DOUBLE, left, 0, simComm, &reqs[2]);
                    MPI_Isend(h_sendHi, haloCount, MPI_DOUBLE, right, 1, simComm, &reqs[3]);
                    pending = true;
                }

                CUDA_CHECK(cudaStreamSynchronize(s1));
                CUDA_CHECK(cudaStreamSynchronize(s0));
            } else {
                {
                    const int span = muEnd - muBeg;
                    const dim3 grid(gx, gy, (span + kchunkMu - 1) / kchunkMu);
                    stencilKernel<MODE_MU><<<grid, block, 0, s0>>>(d_c, nullptr, d_mu, nxi, nyi, muBeg, muEnd, kchunkMu,
                                                                   gzoff, nzg, idx2, idy2, idz2, gamma, e_AA, e_BB,
                                                                   e_AB, dtD);
                }
                {
                    const dim3 grid(gx, gy, (nzl + kchunkMu - 1) / kchunkMu);
                    stencilKernel<MODE_UPDATE><<<grid, block, 0, s0>>>(d_mu, d_c, d_cnew, nxi, nyi, 2, nzl + 2,
                                                                       kchunkMu, gzoff, nzg, idx2, idy2, idz2, gamma,
                                                                       e_AA, e_BB, e_AB, dtD);
                }
                if (t + 1 < iterations) exchangeHalos(d_cnew);
                CUDA_CHECK(cudaStreamSynchronize(s0));
            }

            // Swap buffers
            std::swap(d_c, d_cnew);
        }
        if (pending) MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (worldRank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ---- Gather the full field on rank 0 when it is needed ----------------
    std::vector<double> c;
    if (printResults || validate) {
        if (activeRank) {
            CUDA_CHECK(cudaMemcpy(hostSlab.data(), d_c + 2 * plane, ownedElems * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
        // Explicit point-to-point gather: the global offsets can exceed the
        // range of the "int" displacements an MPI_Gatherv would require.
        if (worldRank == 0) {
            c.resize(gridSize);
            std::copy(hostSlab.begin(), hostSlab.end(), c.begin());
            for (int r = 1; r < nActive; ++r) {
                MPI_Recv(c.data() + allDispls[r], allCounts[r], MPI_DOUBLE, r, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else if (activeRank) {
            MPI_Send(hostSlab.data(), static_cast<int>(ownedElems), MPI_DOUBLE, 0, 42, MPI_COMM_WORLD);
        }
    }

    int exitCode = 0;

    if (worldRank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(c, "Concentration");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(c, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ---- Cleanup ----------------------------------------------------------
    if (activeRank) {
        if (h_sendLo) CUDA_CHECK(cudaFreeHost(h_sendLo));
        if (h_sendHi) CUDA_CHECK(cudaFreeHost(h_sendHi));
        if (h_recvLo) CUDA_CHECK(cudaFreeHost(h_recvLo));
        if (h_recvHi) CUDA_CHECK(cudaFreeHost(h_recvHi));
        CUDA_CHECK(cudaEventDestroy(evMuInt));
        CUDA_CHECK(cudaEventDestroy(evMuBnd));
        CUDA_CHECK(cudaStreamDestroy(s0));
        CUDA_CHECK(cudaStreamDestroy(s1));
        CUDA_CHECK(cudaFree(d_c));
        CUDA_CHECK(cudaFree(d_cnew));
        CUDA_CHECK(cudaFree(d_mu));
        MPI_Comm_free(&simComm);
    }

    MPI_Finalize();
    return exitCode;
}
