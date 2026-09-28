// Cahn-Hilliard phase separation benchmark.
//
// Hybrid parallelization:
//   MPI    - 1D domain decomposition along Z, one rank per GPU, single-plane halo
//            exchange (CUDA-aware when available, otherwise staged through pinned
//            host memory) overlapped with the interior compute.
//   CUDA   - all stencil work runs on the device; each thread sweeps a column of z
//            planes keeping the z-neighbours in registers.
//   OpenMP - host-side O(N) work (result validation on the gathered field).

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

#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                                 \
    do {                                                                                                 \
        const cudaError_t err_ = (call);                                                                 \
        if (err_ != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_));      \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                \
        }                                                                                                \
    } while (0)

namespace {

// Thread block shape for the (x,y) tile; each thread sweeps a range of z planes.
constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 4;

// Physical parameters. These are compile-time constants (as in the reference code) so
// that the device code can fold them; FP64 throughput is the limiting resource here.
constexpr double DX = 1.0;
constexpr double DY = 1.0;
constexpr double DZ = 1.0;
constexpr double DT = 0.01;
constexpr double E_AA = -(2.0 / 9.0);
constexpr double E_BB = -(2.0 / 9.0);
constexpr double E_AB = (2.0 / 9.0);
constexpr double GAMMA = 0.5;
constexpr double DIFF = 1.0;

constexpr double INV_DX2 = 1.0 / (DX * DX);
constexpr double INV_DY2 = 1.0 / (DY * DY);
constexpr double INV_DZ2 = 1.0 / (DZ * DZ);

struct Params {
    int nx;
    int ny;
    int nzg;   // global grid size in z
    int nzl;   // local (per-rank) number of z planes
    int z0;    // global z index of the first local plane
};

// Local fields are stored with one ghost plane on each side in z:
// plane 0 = lower ghost, planes 1..nzl = owned, plane nzl+1 = upper ghost.
__device__ __forceinline__ size_t planeOffset(const int z, const int nx, const int ny) {
    return static_cast<size_t>(z) * (static_cast<size_t>(nx) * static_cast<size_t>(ny));
}

// Chemical potential: mu = 4.5*((c+1)e_AA + (c-1)e_BB - 2c e_AB) + 3c + c^3 - gamma * lap(c)
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) chemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu, const Params p, const int zbeg, const int zcount) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= p.nx || y >= p.ny) return;

    const int xp = (x < p.nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yp = (y < p.ny - 1) ? y + 1 : y;
    const int yn = (y > 0) ? y - 1 : 0;

    const size_t plane = static_cast<size_t>(p.nx) * static_cast<size_t>(p.ny);
    const size_t row = static_cast<size_t>(y) * p.nx;
    const size_t base = row + x;

    const double* pc = c + planeOffset(zbeg, p.nx, p.ny);
    double* pmu = mu + planeOffset(zbeg, p.nx, p.ny);

    // Rolling registers in z: the neighbour planes are reused across iterations.
    double ccur = pc[base];
    double cprev = (p.z0 + zbeg > 1) ? pc[base - plane] : ccur;

    // The +z clamp can only apply to the very last plane of the sweep, so it is kept
    // out of the hot loop.
    const bool clampHi = (p.z0 + zbeg + zcount - 2) == (p.nzg - 1);
    const int nfull = clampHi ? zcount - 1 : zcount;

#pragma unroll 4
    for (int i = 0; i < zcount; ++i) {
        const double cnext = (i < nfull) ? pc[base + plane] : ccur;

        const double cxx = (pc[row + xp] + pc[row + xn] - 2.0 * ccur) * INV_DX2;
        const double cyy = (pc[static_cast<size_t>(yp) * p.nx + x] + pc[static_cast<size_t>(yn) * p.nx + x] -
                            2.0 * ccur) *
                           INV_DY2;
        const double czz = (cnext + cprev - 2.0 * ccur) * INV_DZ2;

        pmu[base] = 4.5 * ((ccur + 1.0) * E_AA + (ccur - 1.0) * E_BB - 2.0 * ccur * E_AB) + 3.0 * ccur +
                    ccur * ccur * ccur - GAMMA * (cxx + cyy + czz);

        cprev = ccur;
        ccur = cnext;
        pc += plane;
        pmu += plane;
    }
}

// Concentration update: cnew = cold + dt*D*lap(mu)
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) updateKernel(double* __restrict__ cnew,
                                                                 const double* __restrict__ cold,
                                                                 const double* __restrict__ mu, const Params p,
                                                                 const int zbeg, const int zcount) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= p.nx || y >= p.ny) return;

    const int xp = (x < p.nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yp = (y < p.ny - 1) ? y + 1 : y;
    const int yn = (y > 0) ? y - 1 : 0;

    const size_t plane = static_cast<size_t>(p.nx) * static_cast<size_t>(p.ny);
    const size_t row = static_cast<size_t>(y) * p.nx;
    const size_t base = row + x;

    const double* pm = mu + planeOffset(zbeg, p.nx, p.ny);
    const size_t off0 = planeOffset(zbeg, p.nx, p.ny) + base;

    double mcur = pm[base];
    double mprev = (p.z0 + zbeg > 1) ? pm[base - plane] : mcur;

    const bool clampHi = (p.z0 + zbeg + zcount - 2) == (p.nzg - 1);
    const int nfull = clampHi ? zcount - 1 : zcount;

#pragma unroll 4
    for (int i = 0; i < zcount; ++i) {
        const double mnext = (i < nfull) ? pm[base + plane] : mcur;

        const double mxx = (pm[row + xp] + pm[row + xn] - 2.0 * mcur) * INV_DX2;
        const double myy = (pm[static_cast<size_t>(yp) * p.nx + x] + pm[static_cast<size_t>(yn) * p.nx + x] -
                            2.0 * mcur) *
                           INV_DY2;
        const double mzz = (mnext + mprev - 2.0 * mcur) * INV_DZ2;

        const size_t o = off0 + static_cast<size_t>(i) * plane;
        cnew[o] = cold[o] + (DT * DIFF) * (mxx + myy + mzz);

        mprev = mcur;
        mcur = mnext;
        pm += plane;
    }
}

// Same pseudo-random field as the reference implementation, evaluated from the global index.
__global__ void initKernel(double* __restrict__ c, const int nx, const int ny, const int nzl, const int z0,
                           const unsigned long long vol) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    const int zl = blockIdx.z + 1;
    if (x >= nx || y >= ny || zl > nzl) return;

    const unsigned long long linear_id = static_cast<unsigned long long>(z0 + zl - 1) *
                                             (static_cast<unsigned long long>(nx) * ny) +
                                         static_cast<unsigned long long>(y) * nx + x;
    const double pseudo = static_cast<double>(((linear_id + 1ull) * 1299709ull) % vol) / static_cast<double>(vol);
    c[planeOffset(zl, nx, ny) + static_cast<size_t>(y) * nx + x] = -1.0 + 2.0 * pseudo;
}

int g_rank = 0;
int g_size = 1;
bool g_cudaAwareMPI = false;

// Exchange one ghost plane in each z direction. Called after the field is ready on
// the device; the MPI progress overlaps with interior kernels on the compute stream.
void exchangeHalos(double* d_field, const Params& p, const int down, const int up, MPI_Comm comm,
                   cudaStream_t haloStream, double* h_send, double* h_recv) {
    if (down == MPI_PROC_NULL && up == MPI_PROC_NULL) return;

    const size_t plane = static_cast<size_t>(p.nx) * static_cast<size_t>(p.ny);
    const size_t bytes = plane * sizeof(double);
    double* d_lo = d_field + plane;                                   // first owned plane
    double* d_hi = d_field + static_cast<size_t>(p.nzl) * plane;      // last owned plane
    double* d_ghost_lo = d_field;                                     // lower ghost
    double* d_ghost_hi = d_field + static_cast<size_t>(p.nzl + 1) * plane;

    MPI_Request req[4];
    int nreq = 0;

    if (g_cudaAwareMPI) {
        if (down != MPI_PROC_NULL) {
            MPI_Irecv(d_ghost_lo, static_cast<int>(plane), MPI_DOUBLE, down, 0, comm, &req[nreq++]);
            MPI_Isend(d_lo, static_cast<int>(plane), MPI_DOUBLE, down, 1, comm, &req[nreq++]);
        }
        if (up != MPI_PROC_NULL) {
            MPI_Irecv(d_ghost_hi, static_cast<int>(plane), MPI_DOUBLE, up, 1, comm, &req[nreq++]);
            MPI_Isend(d_hi, static_cast<int>(plane), MPI_DOUBLE, up, 0, comm, &req[nreq++]);
        }
        MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
        return;
    }

    // Staged through pinned host memory: [0] = lower plane, [1] = upper plane.
    if (down != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(h_send, d_lo, bytes, cudaMemcpyDeviceToHost, haloStream));
    if (up != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpyAsync(h_send + plane, d_hi, bytes, cudaMemcpyDeviceToHost, haloStream));
    CUDA_CHECK(cudaStreamSynchronize(haloStream));

    if (down != MPI_PROC_NULL) {
        MPI_Irecv(h_recv, static_cast<int>(plane), MPI_DOUBLE, down, 0, comm, &req[nreq++]);
        MPI_Isend(h_send, static_cast<int>(plane), MPI_DOUBLE, down, 1, comm, &req[nreq++]);
    }
    if (up != MPI_PROC_NULL) {
        MPI_Irecv(h_recv + plane, static_cast<int>(plane), MPI_DOUBLE, up, 1, comm, &req[nreq++]);
        MPI_Isend(h_send + plane, static_cast<int>(plane), MPI_DOUBLE, up, 0, comm, &req[nreq++]);
    }
    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);

    if (down != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpyAsync(d_ghost_lo, h_recv, bytes, cudaMemcpyHostToDevice, haloStream));
    if (up != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpyAsync(d_ghost_hi, h_recv + plane, bytes, cudaMemcpyHostToDevice, haloStream));
    CUDA_CHECK(cudaStreamSynchronize(haloStream));
}

bool validateResult(const std::vector<double>& c) {
    const size_t n = c.size();
    int bad = 0;
    double minVal = c[0];
    double maxVal = c[0];

#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) reduction(| : bad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const double val = c[i];
        if (std::isnan(val) || std::isinf(val)) bad = 1;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
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

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

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
            if (g_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (g_rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Bind each rank to a GPU based on its node-local rank.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (g_rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr));  // establish the context before any MPI/CUDA interaction

#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    g_cudaAwareMPI = MPIX_Query_cuda_support() == 1;
#endif

    const size_t gridSize = nx * ny * nz;

    // 1D decomposition along z.
    const int nzg = static_cast<int>(nz);
    const int base = nzg / g_size;
    const int rem = nzg % g_size;
    const int nzl = base + (g_rank < rem ? 1 : 0);
    const int z0 = g_rank * base + std::min(g_rank, rem);

    // Ranks without any plane stay idle (only possible when there are more ranks than planes).
    MPI_Comm simComm;
    MPI_Comm_split(MPI_COMM_WORLD, nzl > 0 ? 0 : MPI_UNDEFINED, g_rank, &simComm);

    Params p{};
    p.nx = static_cast<int>(nx);
    p.ny = static_cast<int>(ny);
    p.nzg = nzg;
    p.nzl = nzl;
    p.z0 = z0;

    const size_t plane = nx * ny;
    const size_t localCells = plane * static_cast<size_t>(nzl);
    const size_t paddedCells = plane * static_cast<size_t>(nzl + 2);

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    double* h_send = nullptr;
    double* h_recv = nullptr;
    cudaStream_t computeStream = nullptr;
    cudaStream_t haloStream = nullptr;
    cudaEvent_t evCold = nullptr;
    cudaEvent_t evMu = nullptr;

    if (g_rank == 0) printf("Initializing concentration field...\n");

    if (simComm != MPI_COMM_NULL) {
        CUDA_CHECK(cudaMalloc(&d_cold, paddedCells * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cnew, paddedCells * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mu, paddedCells * sizeof(double)));
        CUDA_CHECK(cudaHostAlloc(&h_send, 2 * plane * sizeof(double), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_recv, 2 * plane * sizeof(double), cudaHostAllocDefault));
        CUDA_CHECK(cudaStreamCreate(&computeStream));
        CUDA_CHECK(cudaStreamCreate(&haloStream));
        CUDA_CHECK(cudaEventCreateWithFlags(&evCold, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evMu, cudaEventDisableTiming));

        const dim3 block(BLOCK_X, BLOCK_Y);
        const dim3 initGrid((p.nx + BLOCK_X - 1) / BLOCK_X, (p.ny + BLOCK_Y - 1) / BLOCK_Y, nzl);
        initKernel<<<initGrid, block, 0, computeStream>>>(d_cold, p.nx, p.ny, nzl, z0,
                                                          static_cast<unsigned long long>(gridSize));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
    }

    if (g_rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    if (simComm != MPI_COMM_NULL) {
        int simRank = 0;
        int simSize = 1;
        MPI_Comm_rank(simComm, &simRank);
        MPI_Comm_size(simComm, &simSize);
        const int down = (simRank > 0) ? simRank - 1 : MPI_PROC_NULL;
        const int up = (simRank < simSize - 1) ? simRank + 1 : MPI_PROC_NULL;

        const dim3 block(BLOCK_X, BLOCK_Y);
        const dim3 grid((p.nx + BLOCK_X - 1) / BLOCK_X, (p.ny + BLOCK_Y - 1) / BLOCK_Y);

        // Planes 2..nzl-1 only touch owned data; planes 1 and nzl need the ghosts.
        const int interiorBeg = 2;
        const int interiorCount = std::max(0, nzl - 2);
        const int hiPlane = nzl;  // == 1 when nzl == 1

        CUDA_CHECK(cudaEventRecord(evCold, computeStream));

        for (int t = 0; t < iterations; ++t) {
            if (interiorCount > 0)
                chemicalPotentialKernel<<<grid, block, 0, computeStream>>>(d_cold, d_mu, p, interiorBeg,
                                                                          interiorCount);

            CUDA_CHECK(cudaEventSynchronize(evCold));
            exchangeHalos(d_cold, p, down, up, simComm, haloStream, h_send, h_recv);

            chemicalPotentialKernel<<<grid, block, 0, computeStream>>>(d_cold, d_mu, p, 1, 1);
            if (hiPlane != 1)
                chemicalPotentialKernel<<<grid, block, 0, computeStream>>>(d_cold, d_mu, p, hiPlane, 1);
            CUDA_CHECK(cudaEventRecord(evMu, computeStream));

            if (interiorCount > 0)
                updateKernel<<<grid, block, 0, computeStream>>>(d_cnew, d_cold, d_mu, p, interiorBeg,
                                                                interiorCount);

            CUDA_CHECK(cudaEventSynchronize(evMu));
            exchangeHalos(d_mu, p, down, up, simComm, haloStream, h_send, h_recv);

            updateKernel<<<grid, block, 0, computeStream>>>(d_cnew, d_cold, d_mu, p, 1, 1);
            if (hiPlane != 1) updateKernel<<<grid, block, 0, computeStream>>>(d_cnew, d_cold, d_mu, p, hiPlane, 1);
            CUDA_CHECK(cudaEventRecord(evCold, computeStream));

            std::swap(d_cold, d_cnew);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    const long localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long duration = 0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (g_rank == 0) {
        printf("Computation time: %ld ms\n", duration);
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / (duration / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the distributed field on rank 0 for output/validation.
    int rc = 0;
    if (printResults || validate) {
        std::vector<double> local(localCells);
        if (simComm != MPI_COMM_NULL) {
            CUDA_CHECK(cudaMemcpy(local.data(), d_cold + plane, localCells * sizeof(double), cudaMemcpyDeviceToHost));
        }

        std::vector<double> full;
        std::vector<int> counts, displs;
        if (g_rank == 0) {
            full.resize(gridSize);
            counts.resize(g_size);
            displs.resize(g_size);
        }
        MPI_Gather(&nzl, 1, MPI_INT, g_rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (g_rank == 0) {
            int off = 0;
            for (int i = 0; i < g_size; ++i) {
                displs[i] = off;
                off += counts[i];
            }
        }

        // Use a plane-sized datatype so that element counts stay well within int range.
        MPI_Datatype planeType;
        MPI_Type_contiguous(static_cast<int>(plane), MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);
        MPI_Gatherv(local.data(), nzl, planeType, g_rank == 0 ? full.data() : nullptr,
                    g_rank == 0 ? counts.data() : nullptr, g_rank == 0 ? displs.data() : nullptr, planeType, 0,
                    MPI_COMM_WORLD);
        MPI_Type_free(&planeType);

        if (g_rank == 0) {
            if (printResults) print_results(full, "Concentration");

            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(full);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    rc = 1;
                }
            }
        }
        MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (simComm != MPI_COMM_NULL) {
        CUDA_CHECK(cudaFree(d_cold));
        CUDA_CHECK(cudaFree(d_cnew));
        CUDA_CHECK(cudaFree(d_mu));
        CUDA_CHECK(cudaFreeHost(h_send));
        CUDA_CHECK(cudaFreeHost(h_recv));
        CUDA_CHECK(cudaStreamDestroy(computeStream));
        CUDA_CHECK(cudaStreamDestroy(haloStream));
        CUDA_CHECK(cudaEventDestroy(evCold));
        CUDA_CHECK(cudaEventDestroy(evMu));
        MPI_Comm_free(&simComm);
    }

    MPI_Finalize();
    return rc;
}
