// Cahn-Hilliard phase separation benchmark.
//
// Hybrid parallelization:
//   * MPI    - 1D slab decomposition along z across ranks; halo planes are
//              exchanged between neighbouring ranks every time step.
//   * CUDA   - each rank drives one GPU; the chemical potential and update
//              stencils run as CUDA kernels. Halo transfers are overlapped with
//              interior kernels using streams and events.
//   * OpenMP - host-side work: parallel field initialization and parallel
//              min/max/NaN reductions used for validation.
//
// Arithmetic is performed in exactly the same order as the serial reference
// implementation (and FMA contraction is disabled), so the computed field is
// bit-for-bit identical to the original code for any decomposition.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

namespace {

constexpr int TAG_UP = 100;
constexpr int TAG_DOWN = 101;
constexpr unsigned BLOCK_X = 32;
constexpr unsigned BLOCK_Y = 8;
constexpr int MAX_GRID_Z = 65535;

struct Physics {
    double dx, dy, dz, dt, e_AA, e_BB, e_AB, gamma, D;
};

struct Options {
    int nx = 64;
    int ny = 0;
    int nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
};

enum class ParseStatus { Ok, Help, Error };

// Clamped Laplacian of a field stored in padded z-slab layout.
// pc / pm / pp are element offsets of the centre, lower and upper z-planes.
// Clamping at the global domain edges is expressed by pm == pc or pp == pc,
// which reproduces the original (c[z] + c[z] - 2 c[z]) boundary treatment.
__device__ __forceinline__ double laplacianDevice(const double* __restrict__ f, const size_t pc, const size_t pm,
                                                  const size_t pp, const int x, const int y, const int nx,
                                                  const int ny, const double dx, const double dy, const double dz) {
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yn = (y > 0) ? y - 1 : 0;

    const size_t row = static_cast<size_t>(y) * nx;
    const double center = f[pc + row + x];

    const double cxx = (f[pc + row + xp] + f[pc + row + xn] - 2.0 * center) / (dx * dx);
    const double cyy = (f[pc + static_cast<size_t>(yp) * nx + x] + f[pc + static_cast<size_t>(yn) * nx + x] -
                        2.0 * center) / (dy * dy);
    const double czz = (f[pp + row + x] + f[pm + row + x] - 2.0 * center) / (dz * dz);

    return cxx + cyy + czz;
}

// Chemical potential on local z-planes [k0, k0 + kcount) of the slab.
__global__ void chemPotKernel(const double* __restrict__ c, double* __restrict__ mu, const int nx, const int ny,
                              const int nzl, const int k0, const int kcount, const int clampLo, const int clampHi,
                              const double dx, const double dy, const double dz, const double gamma,
                              const double e_AA, const double e_BB, const double e_AB) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= nx || y >= ny) return;

    const size_t plane = static_cast<size_t>(nx) * ny;
    for (int kk = blockIdx.z; kk < kcount; kk += gridDim.z) {
        const int k = k0 + kk;
        const size_t pc = static_cast<size_t>(k + 1) * plane;  // +1 skips the lower ghost plane
        const size_t pm = (k == 0 && clampLo) ? pc : pc - plane;
        const size_t pp = (k == nzl - 1 && clampHi) ? pc : pc + plane;

        const size_t i = pc + static_cast<size_t>(y) * nx + x;
        const double cv = c[i];
        const double lap = laplacianDevice(c, pc, pm, pp, x, y, nx, ny, dx, dy, dz);

        mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                + 3.0 * cv + cv * cv * cv
                - gamma * lap;
    }
}

// Cahn-Hilliard update on local z-planes [k0, k0 + kcount) of the slab.
__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                             const double* __restrict__ mu, const int nx, const int ny, const int nzl,
                             const int k0, const int kcount, const int clampLo, const int clampHi,
                             const double D, const double dt, const double dx, const double dy, const double dz) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= nx || y >= ny) return;

    const size_t plane = static_cast<size_t>(nx) * ny;
    for (int kk = blockIdx.z; kk < kcount; kk += gridDim.z) {
        const int k = k0 + kk;
        const size_t pc = static_cast<size_t>(k + 1) * plane;
        const size_t pm = (k == 0 && clampLo) ? pc : pc - plane;
        const size_t pp = (k == nzl - 1 && clampHi) ? pc : pc + plane;

        const size_t i = pc + static_cast<size_t>(y) * nx + x;
        cnew[i] = cold[i] + dt * D * laplacianDevice(mu, pc, pm, pp, x, y, nx, ny, dx, dy, dz);
    }
}

void launchChemPot(const double* d_c, double* d_mu, const int nx, const int ny, const int nzl, const int k0,
                   const int kcount, const bool clampLo, const bool clampHi, const Physics& p, cudaStream_t s) {
    if (kcount <= 0) return;
    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((nx + BLOCK_X - 1) / BLOCK_X, (ny + BLOCK_Y - 1) / BLOCK_Y,
                    std::min(kcount, MAX_GRID_Z));
    chemPotKernel<<<grid, block, 0, s>>>(d_c, d_mu, nx, ny, nzl, k0, kcount, clampLo ? 1 : 0, clampHi ? 1 : 0,
                                         p.dx, p.dy, p.dz, p.gamma, p.e_AA, p.e_BB, p.e_AB);
    CUDA_CHECK(cudaGetLastError());
}

void launchUpdate(double* d_cnew, const double* d_cold, const double* d_mu, const int nx, const int ny,
                  const int nzl, const int k0, const int kcount, const bool clampLo, const bool clampHi,
                  const Physics& p, cudaStream_t s) {
    if (kcount <= 0) return;
    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((nx + BLOCK_X - 1) / BLOCK_X, (ny + BLOCK_Y - 1) / BLOCK_Y,
                    std::min(kcount, MAX_GRID_Z));
    updateKernel<<<grid, block, 0, s>>>(d_cnew, d_cold, d_mu, nx, ny, nzl, k0, kcount, clampLo ? 1 : 0,
                                        clampHi ? 1 : 0, p.D, p.dt, p.dx, p.dy, p.dz);
    CUDA_CHECK(cudaGetLastError());
}

// Communication state for exchanging ghost planes of a device field.
struct Halo {
    MPI_Comm comm;
    int lo;  // neighbour below (MPI_PROC_NULL at the bottom of the domain)
    int hi;  // neighbour above (MPI_PROC_NULL at the top of the domain)
    int nzl;
    size_t plane;  // elements per z-plane
    cudaStream_t sx;
    double* hSendLo;  // pinned host staging buffers, one plane each
    double* hSendHi;
    double* hRecvLo;
    double* hRecvHi;
};

// Refresh ghost planes of padded device field d_f. `ready` must be recorded
// after the last kernel that wrote the interior planes of d_f.
// `done` is recorded on h.sx once the ghost planes are updated on the device.
void exchangeGhosts(const Halo& h, double* d_f, cudaEvent_t ready, cudaEvent_t done) {
    const size_t bytes = h.plane * sizeof(double);

    CUDA_CHECK(cudaStreamWaitEvent(h.sx, ready, 0));
    CUDA_CHECK(cudaMemcpyAsync(h.hSendLo, d_f + h.plane, bytes, cudaMemcpyDeviceToHost, h.sx));
    CUDA_CHECK(cudaMemcpyAsync(h.hSendHi, d_f + static_cast<size_t>(h.nzl) * h.plane, bytes,
                               cudaMemcpyDeviceToHost, h.sx));
    CUDA_CHECK(cudaStreamSynchronize(h.sx));

    MPI_Sendrecv(h.hSendHi, static_cast<int>(h.plane), MPI_DOUBLE, h.hi, TAG_UP,
                 h.hRecvLo, static_cast<int>(h.plane), MPI_DOUBLE, h.lo, TAG_UP, h.comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(h.hSendLo, static_cast<int>(h.plane), MPI_DOUBLE, h.lo, TAG_DOWN,
                 h.hRecvHi, static_cast<int>(h.plane), MPI_DOUBLE, h.hi, TAG_DOWN, h.comm, MPI_STATUS_IGNORE);

    if (h.lo != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_f, h.hRecvLo, bytes, cudaMemcpyHostToDevice, h.sx));
    }
    if (h.hi != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_f + static_cast<size_t>(h.nzl + 1) * h.plane, h.hRecvHi, bytes,
                                   cudaMemcpyHostToDevice, h.sx));
    }
    CUDA_CHECK(cudaEventRecord(done, h.sx));
}

ParseStatus parseArgs(int argc, char** argv, Options& o, const bool root) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            o.nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            o.ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            o.nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            o.iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            o.validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            o.printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            return ParseStatus::Help;
        } else {
            if (root) printf("Unknown option: %s\n", argv[i]);
            return ParseStatus::Error;
        }
    }
    return ParseStatus::Ok;
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

// Distributed validation with the same checks and messages as the serial version.
bool validateResult(const double* local, const size_t n, MPI_Comm comm, const bool root) {
    int bad = 0;
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -std::numeric_limits<double>::infinity();

#pragma omp parallel for reduction(max : bad) reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        const double v = local[i];
        if (std::isnan(v) || std::isinf(v)) {
            bad = 1;
        }
        minVal = std::min(minVal, v);
        maxVal = std::max(maxVal, v);
    }

    int globalBad = 0;
    double globalMin = 0.0, globalMax = 0.0;
    MPI_Allreduce(&bad, &globalBad, 1, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (globalBad) {
        if (root) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    if (root) printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);

    // Values should generally stay within reasonable bounds
    if (globalMax > 10.0 || globalMin < -10.0) {
        if (root) printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const bool root = (worldRank == 0);

    Options opt;
    const ParseStatus status = parseArgs(argc, argv, opt, root);
    if (status != ParseStatus::Ok) {
        if (root) printUsage(argv[0]);
        MPI_Finalize();
        return (status == ParseStatus::Help) ? 0 : 1;
    }

    if (opt.ny == 0) opt.ny = opt.nx;
    if (opt.nz == 0) opt.nz = opt.nx;
    if (opt.nx <= 0 || opt.ny <= 0 || opt.nz <= 0 || opt.iterations < 0) {
        if (root) {
            printf("Grid dimensions must be positive and iterations must be non-negative\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const size_t nx = static_cast<size_t>(opt.nx);
    const size_t ny = static_cast<size_t>(opt.ny);
    const size_t nz = static_cast<size_t>(opt.nz);
    const int iterations = opt.iterations;
    const bool validate = opt.validate;
    const bool printResults = opt.printResults;

    if (root) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters
    Physics phys{};
    phys.dx = 1.0;
    phys.dy = 1.0;
    phys.dz = 1.0;
    phys.dt = 0.01;
    phys.e_AA = -(2.0 / 9.0);
    phys.e_BB = -(2.0 / 9.0);
    phys.e_AB = (2.0 / 9.0);
    phys.gamma = 0.5;
    phys.D = 1.0;

    // Node-local rank, used to bind each rank to a GPU.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    // Each active rank owns at least one z-plane; surplus ranks sit out.
    const int activeColor = (static_cast<size_t>(worldRank) < nz) ? 0 : MPI_UNDEFINED;
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, activeColor, worldRank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % ndev));

    // 1D decomposition along z; the first `rem` ranks get one extra plane.
    const size_t base = nz / static_cast<size_t>(nranks);
    const size_t rem = nz % static_cast<size_t>(nranks);
    const int nzl = static_cast<int>(base + (static_cast<size_t>(rank) < rem ? 1 : 0));
    const size_t z0 = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);

    const size_t plane = nx * ny;
    const size_t gridSize = nx * ny * nz;
    const size_t padded = static_cast<size_t>(nzl + 2) * plane;  // includes two ghost planes
    const size_t interiorBytes = static_cast<size_t>(nzl) * plane * sizeof(double);

    const bool clampLo = (rank == 0);
    const bool clampHi = (rank == nranks - 1);

    // Host-side initialization (OpenMP parallel over local planes).
    if (root) printf("Initializing concentration field...\n");
    std::vector<double> hInterior(static_cast<size_t>(nzl) * plane);
    {
        const size_t vol = gridSize;
#pragma omp parallel for schedule(static)
        for (long long zl = 0; zl < static_cast<long long>(nzl); ++zl) {
            const size_t z = z0 + static_cast<size_t>(zl);
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    // Generate pseudo-random value in [-1, 1]
                    const size_t linear_id = z * (nx * ny) + y * nx + x;
                    const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                    hInterior[static_cast<size_t>(zl) * plane + y * nx + x] = -1.0 + 2.0 * pseudo;
                }
            }
        }
    }

    // Device state: padded slab with ghost planes at index 0 and nzl+1.
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, padded * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, padded * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, padded * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cold, 0, padded * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cnew, 0, padded * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu, 0, padded * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_cold + plane, hInterior.data(), interiorBytes, cudaMemcpyHostToDevice));

    cudaStream_t sc = nullptr;  // compute stream
    cudaStream_t sx = nullptr;  // transfer stream (device <-> host staging)
    CUDA_CHECK(cudaStreamCreateWithFlags(&sc, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sx, cudaStreamNonBlocking));

    cudaEvent_t evColdReady, evCGhost, evMuReady, evMuGhost;
    CUDA_CHECK(cudaEventCreateWithFlags(&evColdReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evCGhost, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evMuReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evMuGhost, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(evColdReady, sc));  // initial field is ready on sc

    Halo halo{};
    halo.comm = comm;
    halo.lo = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    halo.hi = (rank < nranks - 1) ? rank + 1 : MPI_PROC_NULL;
    halo.nzl = nzl;
    halo.plane = plane;
    halo.sx = sx;
    CUDA_CHECK(cudaMallocHost(&halo.hSendLo, plane * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&halo.hSendHi, plane * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&halo.hRecvLo, plane * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&halo.hRecvHi, plane * sizeof(double)));

    // Run simulation
    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // (1) Interior chemical potential: needs no ghost data, so it overlaps the c-halo exchange.
        if (nzl > 2) {
            launchChemPot(d_cold, d_mu, static_cast<int>(nx), static_cast<int>(ny), nzl, 1, nzl - 2, clampLo,
                          clampHi, phys, sc);
        }

        // (2) Exchange ghost planes of the concentration field.
        exchangeGhosts(halo, d_cold, evColdReady, evCGhost);

        // (3) Chemical potential on the two boundary planes, which need the ghost data.
        CUDA_CHECK(cudaStreamWaitEvent(sc, evCGhost, 0));
        launchChemPot(d_cold, d_mu, static_cast<int>(nx), static_cast<int>(ny), nzl, 0, 1, clampLo, clampHi,
                      phys, sc);
        if (nzl > 1) {
            launchChemPot(d_cold, d_mu, static_cast<int>(nx), static_cast<int>(ny), nzl, nzl - 1, 1, clampLo,
                          clampHi, phys, sc);
        }
        CUDA_CHECK(cudaEventRecord(evMuReady, sc));

        // (4) Interior update: needs only local mu, overlaps the mu-halo exchange.
        if (nzl > 2) {
            launchUpdate(d_cnew, d_cold, d_mu, static_cast<int>(nx), static_cast<int>(ny), nzl, 1, nzl - 2,
                         clampLo, clampHi, phys, sc);
        }

        // (5) Exchange ghost planes of the chemical potential.
        exchangeGhosts(halo, d_mu, evMuReady, evMuGhost);

        // (6) Update of the boundary planes, which need the mu ghost data.
        CUDA_CHECK(cudaStreamWaitEvent(sc, evMuGhost, 0));
        launchUpdate(d_cnew, d_cold, d_mu, static_cast<int>(nx), static_cast<int>(ny), nzl, 0, 1, clampLo,
                     clampHi, phys, sc);
        if (nzl > 1) {
            launchUpdate(d_cnew, d_cold, d_mu, static_cast<int>(nx), static_cast<int>(ny), nzl, nzl - 1, 1,
                         clampLo, clampHi, phys, sc);
        }
        CUDA_CHECK(cudaEventRecord(evColdReady, sc));

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaStreamSynchronize(sc));
    CUDA_CHECK(cudaStreamSynchronize(sx));
    MPI_Barrier(comm);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Bring the final interior field back to the host when it is needed.
    std::vector<double> hFinal;
    if (printResults || validate) {
        hFinal.resize(static_cast<size_t>(nzl) * plane);
        CUDA_CHECK(cudaMemcpy(hFinal.data(), d_cold + plane, interiorBytes, cudaMemcpyDeviceToHost));
    }

    // Print results for external validation
    if (printResults) {
        const size_t localCount = static_cast<size_t>(nzl) * plane;
        if (gridSize > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (root) printf("Result printing supports at most %d elements\n", std::numeric_limits<int>::max());
        } else {
            std::vector<int> counts(nranks), displs(nranks);
            const int localCountInt = static_cast<int>(localCount);
            MPI_Allgather(&localCountInt, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
            int offset = 0;
            for (int r = 0; r < nranks; ++r) {
                displs[r] = offset;
                offset += counts[r];
            }
            std::vector<double> full(root ? gridSize : 0);
            MPI_Gatherv(hFinal.data(), localCountInt, MPI_DOUBLE, full.data(), counts.data(), displs.data(),
                        MPI_DOUBLE, 0, comm);
            if (root) print_results(full, "Concentration");
        }
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (root) printf("Validating result...\n");
        bool valid = validateResult(hFinal.data(), hFinal.size(), comm, root);

        if (root) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        exitCode = valid ? 0 : 1;
    }

    // Release resources
    CUDA_CHECK(cudaEventDestroy(evColdReady));
    CUDA_CHECK(cudaEventDestroy(evCGhost));
    CUDA_CHECK(cudaEventDestroy(evMuReady));
    CUDA_CHECK(cudaEventDestroy(evMuGhost));
    CUDA_CHECK(cudaStreamDestroy(sc));
    CUDA_CHECK(cudaStreamDestroy(sx));
    CUDA_CHECK(cudaFreeHost(halo.hSendLo));
    CUDA_CHECK(cudaFreeHost(halo.hSendHi));
    CUDA_CHECK(cudaFreeHost(halo.hRecvLo));
    CUDA_CHECK(cudaFreeHost(halo.hRecvHi));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    MPI_Comm_free(&comm);
    MPI_Finalize();

    return exitCode;
}
