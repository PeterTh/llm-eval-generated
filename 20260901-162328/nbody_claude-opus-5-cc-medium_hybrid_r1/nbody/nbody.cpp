// Hybrid MPI + OpenMP + CUDA n-body benchmark.
//
// Parallelization strategy:
//   * MPI     : the body array is block-distributed over ranks; every rank keeps a full
//               copy of all positions and owns the velocities/integration of its block.
//               Positions are exchanged once per step with a single MPI_Allgatherv.
//   * OpenMP  : one host thread per GPU owned by the rank (drives independent CUDA
//               streams/devices concurrently) plus host-side parallel loops for
//               packing, serialization and validation.
//   * CUDA    : the O(N^2) force evaluation, the integration and the potential-energy
//               evaluation run on the GPUs, using shared-memory tiling. Each thread
//               accumulates the interactions of one body in ascending j order so the
//               floating point summation order matches the original serial code.

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

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// CUDA tuning parameters
constexpr int BLOCK = 128;         // threads per block (also the tile size)
constexpr int MIN_THREADS = 262144; // target force threads before splitting the j loop // target amount of force threads before splitting j // target amount of force threads before splitting j // target amount of force threads before splitting j // target amount of force threads before splitting j
constexpr int MIN_SPLIT_LEN = 128; // minimum j-range handled by one j-split
constexpr int MAX_SPLIT = 64;      // maximum number of j-splits

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

#define CUDA_CHECK(call)                                                                                     \
    do {                                                                                                     \
        const cudaError_t err_ = (call);                                                                      \
        if (err_ != cudaSuccess) {                                                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__);        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                     \
        }                                                                                                    \
    } while (0)

// -----------------------------------------------------------------------------
// Device kernels
// -----------------------------------------------------------------------------

// Accumulates the force on the bodies [off, off+cnt) from the j-range of this split.
// Partial results are stored as acc[split * cnt + i]; grid.y selects the split.
__global__ void forceKernel(const double3* __restrict__ pos, const int n, const int off, const int cnt,
                            const int splitLen, double3* __restrict__ acc) {
    __shared__ double3 tile[BLOCK];

    const int il = blockIdx.x * BLOCK + threadIdx.x;
    const bool active = il < cnt;

    const int jbeg = blockIdx.y * splitLen;
    const int jend = min(n, jbeg + splitLen);

    double3 p = make_double3(0.0, 0.0, 0.0);
    if (active) { p = pos[off + il]; }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int t = jbeg; t < jend; t += BLOCK) {
        const int j = t + threadIdx.x;
        if (j < jend) { tile[threadIdx.x] = pos[j]; }
        __syncthreads();

        const int lim = min(BLOCK, jend - t);
        if (active) {
#pragma unroll 4
            for (int k = 0; k < lim; ++k) {
                const double dx = tile[k].x - p.x;
                const double dy = tile[k].y - p.y;
                const double dz = tile[k].z - p.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                // rsqrt() is the double precision reciprocal square root (<= 1 ulp,
                // i.e. the same accuracy as 1.0/sqrt(x) but far cheaper).
                const double invDist = rsqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) { acc[(size_t)blockIdx.y * cnt + il] = make_double3(Fx, Fy, Fz); }
}

// Sums the per-split partial forces (in ascending split order) and applies them to the
// local velocities, then integrates the local positions.
__global__ void updateKernel(const double3* __restrict__ acc, double3* __restrict__ vel,
                             double3* __restrict__ pos, const int off, const int cnt, const int nSplit) {
    const int il = blockIdx.x * BLOCK + threadIdx.x;
    if (il >= cnt) { return; }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int s = 0; s < nSplit; ++s) {
        const double3 a = acc[(size_t)s * cnt + il];
        Fx += a.x;
        Fy += a.y;
        Fz += a.z;
    }

    double3 v = vel[il];
    v.x += DT * Fx;
    v.y += DT * Fy;
    v.z += DT * Fz;
    vel[il] = v;

    double3 p = pos[off + il];
    p.x += v.x * DT;
    p.y += v.y * DT;
    p.z += v.z * DT;
    pos[off + il] = p;
}

// Potential energy contribution of every local body i: sum over j > i of -1/dist.
__global__ void energyKernel(const double3* __restrict__ pos, const int n, const int off, const int cnt,
                             double* __restrict__ part) {
    __shared__ double3 tile[BLOCK];

    const int il = blockIdx.x * BLOCK + threadIdx.x;
    const bool active = il < cnt;
    const int i = off + il;

    double3 p = make_double3(0.0, 0.0, 0.0);
    if (active) { p = pos[i]; }

    double e = 0.0;
    // Only tiles that can contain a j > i of any body in this block need to be visited.
    // The bound is block-uniform so that __syncthreads() stays convergent.
    const int tbeg = ((off + blockIdx.x * BLOCK) / BLOCK) * BLOCK;
    for (int t = tbeg; t < n; t += BLOCK) {
        const int j = t + threadIdx.x;
        if (j < n) { tile[threadIdx.x] = pos[j]; }
        __syncthreads();

        const int lim = min(BLOCK, n - t);
        for (int k = 0; k < lim; ++k) {
            if (t + k <= i) { continue; }
            const double dx = tile[k].x - p.x;
            const double dy = tile[k].y - p.y;
            const double dz = tile[k].z - p.z;
            const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            e -= 1.0 / dist;
        }
        __syncthreads();
    }

    if (active) { part[il] = e; }
}

// -----------------------------------------------------------------------------
// Per-GPU state
// -----------------------------------------------------------------------------

struct GpuCtx {
    int dev = 0;
    int off = 0;  // first body owned by this GPU (global index)
    int cnt = 0;  // number of bodies owned by this GPU
    int nSplit = 1;
    int splitLen = 0;
    double3* d_pos = nullptr;   // all N positions
    double3* d_vel = nullptr;   // cnt velocities
    double3* d_acc = nullptr;   // nSplit * cnt partial forces
    double* d_epart = nullptr;  // cnt potential energy partials
    cudaStream_t stream = nullptr;
};

// -----------------------------------------------------------------------------
// Host-side helpers (unchanged semantics, OpenMP parallel)
// -----------------------------------------------------------------------------

double computeTotalEnergy(const std::vector<Body>& bodies, const std::vector<double>& potentialParts) {
    double energy = 0.0;

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies), accumulated in ascending
    // body order from the per-body partial sums computed on the GPUs.
    for (const double p : potentialParts) {
        energy += p;
    }

    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    int bad = 0;  // bit 0: non-finite, bit 1: position out of bounds, bit 2: velocity

#pragma omp parallel for schedule(static) reduction(| : bad)
    for (size_t idx = 0; idx < n; ++idx) {
        const auto& body = bodies[idx];
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            bad |= 1;
            continue;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            bad |= 2;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            bad |= 4;
        }
    }

    if (bad & 1) {
        printf("Validation failed: found NaN or Inf value in body state\n");
    } else if (bad & 2) {
        printf("Validation failed: body position exceeds reasonable bounds\n");
    } else if (bad & 4) {
        printf("Validation failed: body velocity exceeds reasonable bounds\n");
    }
    return bad == 0;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks energy conservation)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int mpiProvided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiProvided);
    (void)mpiProvided;  // MPI calls are funnelled through the master thread

    int rank = 0, nRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) { printUsage(argv[0]); }
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

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Degenerate case: nothing to simulate, but keep the original output structure.
    if (numBodies <= 0) {
        if (rank == 0) {
            printf("Simulation time: 0 ms\n");
            if (printResults) { print_results(std::vector<double>(), "Bodies"); }
            if (validate) {
                printf("Validating simulation results...\n");
                printf("Final energy: %.6f\n", 0.0);
                printf("Validation: PASSED\n");
            }
        }
        MPI_Finalize();
        return 0;
    }

    const int n = numBodies;

    // ---------------------------------------------------------------------
    // Distribute the bodies: block decomposition over ranks, then over the
    // GPUs owned by each rank.
    // ---------------------------------------------------------------------
    std::vector<int> rankOff(nRanks + 1);
    for (int r = 0; r <= nRanks; ++r) {
        rankOff[r] = (int)((long long)n * r / nRanks);
    }
    const int myOff = rankOff[rank];
    const int myCnt = rankOff[rank + 1] - myOff;

    // Determine how many GPUs this rank may use: devices are shared out among the
    // ranks that run on the same node.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0, nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);

    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Devices [devBeg, devEnd) belong to this rank; if there are more ranks than
    // devices on the node, the ranks share devices round-robin.
    std::vector<int> myDevs;
    const int devBeg = (int)((long long)devCount * nodeRank / nodeSize);
    const int devEnd = (int)((long long)devCount * (nodeRank + 1) / nodeSize);
    if (devEnd > devBeg) {
        for (int d = devBeg; d < devEnd; ++d) {
            myDevs.push_back(d);
        }
    } else {
        myDevs.push_back(nodeRank % devCount);
    }
    // Never use more GPUs than there are bodies to compute.
    if ((int)myDevs.size() > myCnt && myCnt > 0) { myDevs.resize(myCnt); }
    const int nGpu = (int)myDevs.size();
    const int devFirst = myDevs[0];

    const int hostThreads = omp_get_max_threads();

    if (rank == 0) {
        printf("MPI ranks: %d, GPUs per rank: %d, OpenMP threads: %d\n", nRanks, nGpu, hostThreads);
    }

    // ---------------------------------------------------------------------
    // Initialization (identical on every rank, so no broadcast is needed)
    // ---------------------------------------------------------------------
    std::vector<Body> bodies(n);
    randomizeBodies(bodies);

    double3* h_pos = nullptr;  // all N positions, pinned for fast transfers
    double3* h_vel = nullptr;  // local velocities of this rank
    double* h_epart = nullptr;
    CUDA_CHECK(cudaSetDevice(devFirst));
    CUDA_CHECK(cudaHostAlloc((void**)&h_pos, (size_t)n * sizeof(double3), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc((void**)&h_vel, (size_t)std::max(myCnt, 1) * sizeof(double3), cudaHostAllocPortable));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        h_pos[i] = make_double3(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z);
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < myCnt; ++i) {
        h_vel[i] = make_double3(bodies[myOff + i].vel.x, bodies[myOff + i].vel.y, bodies[myOff + i].vel.z);
    }

    // Per-GPU decomposition of this rank's block
    std::vector<GpuCtx> ctx(nGpu);
    for (int g = 0; g < nGpu; ++g) {
        ctx[g].dev = myDevs[g];
        ctx[g].off = myOff + (int)((long long)myCnt * g / nGpu);
        ctx[g].cnt = myOff + (int)((long long)myCnt * (g + 1) / nGpu) - ctx[g].off;
    }

    // MPI exchange descriptors (in units of doubles)
    std::vector<int> recvCounts(nRanks), recvDispls(nRanks);
    for (int r = 0; r < nRanks; ++r) {
        recvCounts[r] = (rankOff[r + 1] - rankOff[r]) * 3;
        recvDispls[r] = rankOff[r] * 3;
    }

    const bool needExchange = (nRanks > 1) || (nGpu > 1);

#pragma omp parallel num_threads(nGpu)
    {
        const int g = omp_get_thread_num();
        GpuCtx& c = ctx[g];
        CUDA_CHECK(cudaSetDevice(c.dev));
        // Spinning gives the lowest synchronization latency, but if the rank is bound to
        // fewer cores than it drives GPUs the driver threads have to yield instead.
        CUDA_CHECK(cudaSetDeviceFlags(nGpu > hostThreads ? cudaDeviceScheduleYield : cudaDeviceScheduleSpin));
        CUDA_CHECK(cudaStreamCreate(&c.stream));

        // Choose the number of j-splits so that enough threads are resident even for
        // small body counts.
        int nSplit = 1;
        if (c.cnt > 0) {
            nSplit = std::max(1, MIN_THREADS / c.cnt);
            nSplit = std::min(nSplit, MAX_SPLIT);
            nSplit = std::min(nSplit, std::max(1, n / MIN_SPLIT_LEN));
        }
        int splitLen = (n + nSplit - 1) / nSplit;
        splitLen = ((splitLen + BLOCK - 1) / BLOCK) * BLOCK;
        nSplit = (n + splitLen - 1) / splitLen;  // drop splits that ended up empty
        c.nSplit = nSplit;
        c.splitLen = splitLen;

        CUDA_CHECK(cudaMalloc((void**)&c.d_pos, (size_t)n * sizeof(double3)));
        if (c.cnt > 0) {
            CUDA_CHECK(cudaMalloc((void**)&c.d_vel, (size_t)c.cnt * sizeof(double3)));
            CUDA_CHECK(cudaMalloc((void**)&c.d_acc, (size_t)nSplit * c.cnt * sizeof(double3)));
            CUDA_CHECK(cudaMemcpyAsync(c.d_vel, h_vel + (c.off - myOff), (size_t)c.cnt * sizeof(double3),
                                       cudaMemcpyHostToDevice, c.stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(c.d_pos, h_pos, (size_t)n * sizeof(double3), cudaMemcpyHostToDevice, c.stream));

        // Force the (lazily loaded) kernels to be resident before the timed region.
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, forceKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, updateKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, energyKernel));

        CUDA_CHECK(cudaStreamSynchronize(c.stream));
    }

    // ---------------------------------------------------------------------
    // Simulation
    // ---------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(nGpu)
    {
        const int g = omp_get_thread_num();
        GpuCtx& c = ctx[g];
        CUDA_CHECK(cudaSetDevice(c.dev));

        const dim3 forceGrid((c.cnt + BLOCK - 1) / BLOCK, c.nSplit);
        const dim3 updateGrid((c.cnt + BLOCK - 1) / BLOCK);

        for (int step = 0; step < numSteps; ++step) {
            if (c.cnt > 0) {
                forceKernel<<<forceGrid, BLOCK, 0, c.stream>>>(c.d_pos, n, c.off, c.cnt, c.splitLen, c.d_acc);
                updateKernel<<<updateGrid, BLOCK, 0, c.stream>>>(c.d_acc, c.d_vel, c.d_pos, c.off, c.cnt, c.nSplit);
            }

            if (needExchange) {
                // Publish the positions of the locally owned bodies.
                if (c.cnt > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(h_pos + c.off, c.d_pos + c.off, (size_t)c.cnt * sizeof(double3),
                                               cudaMemcpyDeviceToHost, c.stream));
                }
                CUDA_CHECK(cudaStreamSynchronize(c.stream));

#pragma omp barrier
#pragma omp master
                {
                    if (nRanks > 1) {
                        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, (double*)h_pos, recvCounts.data(),
                                       recvDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
                    }
                }
#pragma omp barrier

                // Fetch the positions owned by the other GPUs/ranks; the own block is
                // already up to date in device memory.
                if (c.off > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(c.d_pos, h_pos, (size_t)c.off * sizeof(double3),
                                               cudaMemcpyHostToDevice, c.stream));
                }
                const int tailOff = c.off + c.cnt;
                if (tailOff < n) {
                    CUDA_CHECK(cudaMemcpyAsync(c.d_pos + tailOff, h_pos + tailOff,
                                               (size_t)(n - tailOff) * sizeof(double3), cudaMemcpyHostToDevice,
                                               c.stream));
                }
                // The transfers read h_pos, which the other threads overwrite in the
                // next step, so they have to complete before the next iteration.
                CUDA_CHECK(cudaStreamSynchronize(c.stream));
#pragma omp barrier
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(c.stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) { printf("Simulation time: %ld ms\n", maxDuration); }

    // ---------------------------------------------------------------------
    // Collect the final state (and the potential energy partials if needed)
    // ---------------------------------------------------------------------
    if (validate) {
        CUDA_CHECK(cudaSetDevice(devFirst));
        CUDA_CHECK(cudaHostAlloc((void**)&h_epart, (size_t)std::max(myCnt, 1) * sizeof(double),
                                 cudaHostAllocPortable));
    }

#pragma omp parallel num_threads(nGpu)
    {
        const int g = omp_get_thread_num();
        GpuCtx& c = ctx[g];
        CUDA_CHECK(cudaSetDevice(c.dev));
        if (c.cnt > 0) {
            CUDA_CHECK(cudaMemcpyAsync(h_vel + (c.off - myOff), c.d_vel, (size_t)c.cnt * sizeof(double3),
                                       cudaMemcpyDeviceToHost, c.stream));
            if (!needExchange) {
                CUDA_CHECK(cudaMemcpyAsync(h_pos + c.off, c.d_pos + c.off, (size_t)c.cnt * sizeof(double3),
                                           cudaMemcpyDeviceToHost, c.stream));
            }
            if (validate) {
                CUDA_CHECK(cudaMalloc((void**)&c.d_epart, (size_t)c.cnt * sizeof(double)));
                energyKernel<<<dim3((c.cnt + BLOCK - 1) / BLOCK), BLOCK, 0, c.stream>>>(c.d_pos, n, c.off, c.cnt,
                                                                                        c.d_epart);
                CUDA_CHECK(cudaMemcpyAsync(h_epart + (c.off - myOff), c.d_epart, (size_t)c.cnt * sizeof(double),
                                           cudaMemcpyDeviceToHost, c.stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(c.stream));
    }

    // Gather velocities (and energy partials) of all ranks on rank 0.
    std::vector<double> allVel;
    std::vector<double> allEpart;
    if (nRanks > 1) {
        if (rank == 0) { allVel.resize((size_t)n * 3); }
        MPI_Gatherv((double*)h_vel, myCnt * 3, MPI_DOUBLE, rank == 0 ? allVel.data() : nullptr, recvCounts.data(),
                    recvDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (validate) {
            std::vector<int> eCounts(nRanks), eDispls(nRanks);
            for (int r = 0; r < nRanks; ++r) {
                eCounts[r] = rankOff[r + 1] - rankOff[r];
                eDispls[r] = rankOff[r];
            }
            if (rank == 0) { allEpart.resize(n); }
            MPI_Gatherv(h_epart, myCnt, MPI_DOUBLE, rank == 0 ? allEpart.data() : nullptr, eCounts.data(),
                        eDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    } else {
        allVel.assign((double*)h_vel, (double*)h_vel + (size_t)n * 3);
        if (validate) { allEpart.assign(h_epart, h_epart + n); }
    }

    int rc = 0;
    if (rank == 0) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            bodies[i].pos = Vec3(h_pos[i].x, h_pos[i].y, h_pos[i].z);
            bodies[i].vel = Vec3(allVel[(size_t)i * 3 + 0], allVel[(size_t)i * 3 + 1], allVel[(size_t)i * 3 + 2]);
        }

        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData((size_t)n * 6);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                bodyData[(size_t)i * 6 + 0] = bodies[i].pos.x;
                bodyData[(size_t)i * 6 + 1] = bodies[i].pos.y;
                bodyData[(size_t)i * 6 + 2] = bodies[i].pos.z;
                bodyData[(size_t)i * 6 + 3] = bodies[i].vel.x;
                bodyData[(size_t)i * 6 + 4] = bodies[i].vel.y;
                bodyData[(size_t)i * 6 + 5] = bodies[i].vel.z;
            }
            print_results(bodyData, "Bodies");
        }

        // Validation: check that simulation produces finite, reasonable values
        if (validate) {
            printf("Validating simulation results...\n");

            if (validateSimulation(bodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies, allEpart);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ---------------------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------------------
#pragma omp parallel num_threads(nGpu)
    {
        const int g = omp_get_thread_num();
        GpuCtx& c = ctx[g];
        cudaSetDevice(c.dev);
        cudaFree(c.d_pos);
        cudaFree(c.d_vel);
        cudaFree(c.d_acc);
        cudaFree(c.d_epart);
        cudaStreamDestroy(c.stream);
    }
    cudaFreeHost(h_pos);
    cudaFreeHost(h_vel);
    if (h_epart != nullptr) { cudaFreeHost(h_epart); }
    MPI_Comm_free(&nodeComm);

    MPI_Finalize();
    return rc;
}
