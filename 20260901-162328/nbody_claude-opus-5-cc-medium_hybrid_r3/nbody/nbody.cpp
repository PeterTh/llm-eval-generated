// Hybrid MPI + OpenMP + CUDA n-body simulation.
//
// Parallelisation strategy:
//   * MPI    - the global body array is split into one contiguous slice per
//              rank.  Every rank keeps a full copy of all positions (needed for
//              the all-pairs force evaluation) and integrates only its own
//              slice.  Updated positions are published with one Allgatherv per
//              step, velocities are exchanged once at the very end.
//   * CUDA   - every rank drives its share of the node's GPUs.  A tiled
//              shared-memory kernel evaluates the forces for a chunk of the
//              rank's slice.  The j-range is additionally split into segments
//              so that even small chunks fill the whole device; the segment
//              contributions are combined in ascending order by a second pass.
//   * OpenMP - the remaining part of the rank's slice is evaluated on the host
//              cores (vectorised inner loop) concurrently with the GPU kernels,
//              and drives the energy/validation reductions.
//
// The host/GPU split is re-balanced after every step from the measured
// throughput of each worker, so that host cores and accelerators finish at the
// same time regardless of the machine the benchmark runs on.
//
// This file is compiled twice: once by nvcc (via a .cu symlink created by
// CMake) which keeps only the device code below, and once by the host C++
// compiler which keeps everything else.  Splitting the translation units this
// way is what allows the host force loop to be properly auto-vectorised.

#include <cmath>

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// Threads per CUDA block for the force kernel.
constexpr int NBODY_BLOCK = 256;

#ifdef __CUDACC__
// ===========================================================================
// Device translation unit
// ===========================================================================
#include <cuda_runtime.h>

// Force accumulation for bodies [start, start + cnt) of the global array.
// blockIdx.y selects the segment of the j-range this block accumulates; the
// per-segment sums are stored in `partial` and combined in ascending segment
// order by the reduction kernel below, which keeps the result deterministic
// and independent of the scheduling order.
template <int BS>
__global__ __launch_bounds__(BS) void forceKernel(const double* __restrict__ px, const double* __restrict__ py,
                                                  const double* __restrict__ pz, double* __restrict__ partial,
                                                  const int n, const int start, const int cnt,
                                                  const int segLen) {
    __shared__ double sx[BS], sy[BS], sz[BS];

    const int t = threadIdx.x;
    const int idx = blockIdx.x * BS + t;
    const int seg = blockIdx.y;
    const bool active = idx < cnt;

    const double xi = active ? px[start + idx] : 0.0;
    const double yi = active ? py[start + idx] : 0.0;
    const double zi = active ? pz[start + idx] : 0.0;

    const int jBeg = seg * segLen;
    const int jEnd = min(n, jBeg + segLen);

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = jBeg; tile < jEnd; tile += BS) {
        const int j = tile + t;
        sx[t] = j < jEnd ? px[j] : 0.0;
        sy[t] = j < jEnd ? py[j] : 0.0;
        sz[t] = j < jEnd ? pz[j] : 0.0;
        __syncthreads();

        const int lim = min(BS, jEnd - tile);
        #pragma unroll 4
        for (int k = 0; k < lim; ++k) {
            const double dx = sx[k] - xi;
            const double dy = sy[k] - yi;
            const double dz = sz[k] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    if (active) {
        const size_t o = (size_t)seg * cnt + idx;
        const size_t stride = (size_t)gridDim.y * cnt;
        partial[o] = Fx;
        partial[o + stride] = Fy;
        partial[o + 2 * stride] = Fz;
    }
}

// Combine the segment contributions, advance velocities and produce the new
// positions of the chunk.
__global__ void reduceKernel(const double* __restrict__ partial, const double* __restrict__ px,
                             const double* __restrict__ py, const double* __restrict__ pz,
                             double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
                             double* __restrict__ opx, double* __restrict__ opy, double* __restrict__ opz,
                             const int start, const int cnt, const int segments) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cnt) return;

    const size_t stride = (size_t)segments * cnt;
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int s = 0; s < segments; ++s) {
        const size_t o = (size_t)s * cnt + idx;
        Fx += partial[o];
        Fy += partial[o + stride];
        Fz += partial[o + 2 * stride];
    }

    const double nvx = vx[idx] + DT * Fx;
    const double nvy = vy[idx] + DT * Fy;
    const double nvz = vz[idx] + DT * Fz;
    vx[idx] = nvx;
    vy[idx] = nvy;
    vz[idx] = nvz;
    opx[idx] = px[start + idx] + nvx * DT;
    opy[idx] = py[start + idx] + nvy * DT;
    opz[idx] = pz[start + idx] + nvz * DT;
}

extern "C" void nbodyLaunchForces(const double* px, const double* py, const double* pz, double* vx, double* vy,
                                  double* vz, double* opx, double* opy, double* opz, double* partial,
                                  const int n, const int start, const int cnt, const int segments,
                                  cudaStream_t stream) {
    if (cnt <= 0) return;

    const int segLen = ((n + segments - 1) / segments + NBODY_BLOCK - 1) / NBODY_BLOCK * NBODY_BLOCK;
    const dim3 grid((cnt + NBODY_BLOCK - 1) / NBODY_BLOCK, segments);
    forceKernel<NBODY_BLOCK><<<grid, NBODY_BLOCK, 0, stream>>>(px, py, pz, partial, n, start, cnt, segLen);

    const int rblocks = (cnt + NBODY_BLOCK - 1) / NBODY_BLOCK;
    reduceKernel<<<rblocks, NBODY_BLOCK, 0, stream>>>(partial, px, py, pz, vx, vy, vz, opx, opy, opz, start,
                                                      cnt, segments);
}

#else
// ===========================================================================
// Host translation unit
// ===========================================================================

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <sched.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

extern "C" void nbodyLaunchForces(const double* px, const double* py, const double* pz, double* vx, double* vy,
                                  double* vz, double* opx, double* opy, double* opz, double* partial, int n,
                                  int start, int cnt, int segments, cudaStream_t stream);

// Initial throughput guesses (body-body interactions per second) used for the
// very first step; they are replaced by measured values afterwards.
constexpr double INITIAL_GPU_RATE = 8.0e9;
constexpr double INITIAL_CPU_RATE_PER_THREAD = 0.45e9;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call)                                                                              \
    do {                                                                                              \
        const cudaError_t err_ = (call);                                                              \
        if (err_ != cudaSuccess) {                                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                             \
        }                                                                                             \
    } while (0)

// ---------------------------------------------------------------------------
// Structure-of-arrays body state.  Positions are replicated on every rank,
// velocities are only valid inside the rank's own slice until the final
// exchange.
// ---------------------------------------------------------------------------
struct State {
    double *px = nullptr, *py = nullptr, *pz = nullptr;     // n entries, replicated
    double *vx = nullptr, *vy = nullptr, *vz = nullptr;     // n entries, local slice valid
    double *npx = nullptr, *npy = nullptr, *npz = nullptr;  // new positions, local slice only
    int n = 0;
    int lo = 0, hi = 0;  // this rank's slice [lo, hi)
};

// Per-GPU context owned by this rank.
struct GpuCtx {
    int dev = 0;
    int maxThreads = 0;  // resident threads, used to size the j-segmentation
    cudaStream_t stream = nullptr;
    cudaEvent_t evStart = nullptr, evStop = nullptr;
    double *px = nullptr, *py = nullptr, *pz = nullptr;     // full position copy
    double *vx = nullptr, *vy = nullptr, *vz = nullptr;     // chunk velocities
    double *opx = nullptr, *opy = nullptr, *opz = nullptr;  // chunk output positions
    double* partial = nullptr;                              // segment partial forces
    int start = 0, cnt = 0;
};

// ---------------------------------------------------------------------------
// Host force evaluation for [first, last): OpenMP over the bodies, vectorised
// inner loop.  New positions go to a separate buffer so that the positions stay
// untouched while forces are evaluated.
// ---------------------------------------------------------------------------
static void computeForcesHost(const State& s, const int first, const int last) {
    const double* __restrict__ px = s.px;
    const double* __restrict__ py = s.py;
    const double* __restrict__ pz = s.pz;
    const int n = s.n;

    #pragma omp parallel for schedule(static)
    for (int i = first; i < last; ++i) {
        const double xi = px[i], yi = py[i], zi = pz[i];
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        #pragma omp simd reduction(+ : Fx, Fy, Fz)
        for (int j = 0; j < n; ++j) {
            const double dx = px[j] - xi;
            const double dy = py[j] - yi;
            const double dz = pz[j] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        const double nvx = s.vx[i] + DT * Fx;
        const double nvy = s.vy[i] + DT * Fy;
        const double nvz = s.vz[i] + DT * Fz;
        s.vx[i] = nvx;
        s.vy[i] = nvy;
        s.vz[i] = nvz;
        s.npx[i - s.lo] = xi + nvx * DT;
        s.npy[i - s.lo] = yi + nvy * DT;
        s.npz[i - s.lo] = zi + nvz * DT;
    }
}

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

// Total energy, distributed over MPI ranks (cyclic i-distribution for load
// balance) and OpenMP threads.  Every rank holds the full state at this point.
double computeTotalEnergy(const State& s, const int rank, const int nRanks, MPI_Comm comm) {
    const double* __restrict__ px = s.px;
    const double* __restrict__ py = s.py;
    const double* __restrict__ pz = s.pz;
    const int n = s.n;

    double energy = 0.0;

    // Kinetic energy (assuming unit mass)
    #pragma omp parallel for schedule(static) reduction(+ : energy)
    for (int i = rank; i < n; i += nRanks) {
        energy += 0.5 * (s.vx[i] * s.vx[i] + s.vy[i] * s.vy[i] + s.vz[i] * s.vz[i]);
    }

    // Potential energy (assuming unit mass for all bodies)
    #pragma omp parallel for schedule(static) reduction(+ : energy)
    for (int i = rank; i < n; i += nRanks) {
        const double xi = px[i], yi = py[i], zi = pz[i];
        double local = 0.0;
        #pragma omp simd reduction(+ : local)
        for (int j = i + 1; j < n; ++j) {
            const double dx = px[j] - xi;
            const double dy = py[j] - yi;
            const double dz = pz[j] - zi;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            local += 1.0 / dist;
        }
        energy -= local;
    }

    double total = 0.0;
    MPI_Allreduce(&energy, &total, 1, MPI_DOUBLE, MPI_SUM, comm);
    return total;
}

// Validate that simulation produces finite, reasonable values.
// Returns 0 on success, otherwise the code of the violated check.
int validateSimulation(const State& s, const int rank, const int nRanks, MPI_Comm comm) {
    const int n = s.n;
    int code = 0;

    #pragma omp parallel for schedule(static) reduction(max : code)
    for (int i = rank; i < n; i += nRanks) {
        // Check for NaN or Inf values
        if (!std::isfinite(s.px[i]) || !std::isfinite(s.py[i]) || !std::isfinite(s.pz[i]) ||
            !std::isfinite(s.vx[i]) || !std::isfinite(s.vy[i]) || !std::isfinite(s.vz[i])) {
            code = std::max(code, 1);
            continue;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(s.px[i]) > maxPos || std::abs(s.py[i]) > maxPos || std::abs(s.pz[i]) > maxPos) {
            code = std::max(code, 2);
        } else if (std::abs(s.vx[i]) > maxVel || std::abs(s.vy[i]) > maxVel || std::abs(s.vz[i]) > maxVel) {
            code = std::max(code, 3);
        }
    }

    int global = 0;
    MPI_Allreduce(&code, &global, 1, MPI_INT, MPI_MAX, comm);
    return global;
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

// ---------------------------------------------------------------------------
// CPU affinity handling.  A hybrid run wants one OpenMP thread per physical
// core, and ranks that share a socket must not end up on the same cores.  We
// therefore enumerate the physical cores ourselves and give every rank a
// disjoint slice of them.
// ---------------------------------------------------------------------------

// Parse a Linux CPU list such as "0-1" or "0,128".
static void parseCpuList(const char* s, std::vector<int>& out) {
    while (*s) {
        char* endp = nullptr;
        const long a = strtol(s, &endp, 10);
        if (endp == s) break;
        long b = a;
        if (*endp == '-') b = strtol(endp + 1, &endp, 10);
        for (long c = a; c <= b; ++c) out.push_back((int)c);
        while (*endp == ',' || *endp == ' ' || *endp == '\n') ++endp;
        s = endp;
    }
}

// All physical cores of the node (each entry lists its hardware threads),
// optionally restricted to the CPUs of `mask`.
static std::vector<std::vector<int>> enumerateCores(const cpu_set_t* mask) {
    const int nCpus = (int)sysconf(_SC_NPROCESSORS_CONF);
    std::vector<std::vector<int>> cores;
    std::vector<int> seen;

    for (int cpu = 0; cpu < nCpus && cpu < CPU_SETSIZE; ++cpu) {
        if (mask && !CPU_ISSET(cpu, mask)) continue;

        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        std::vector<int> siblings;
        FILE* f = fopen(path, "r");
        if (f) {
            char buf[256] = {0};
            if (fgets(buf, sizeof(buf), f)) parseCpuList(buf, siblings);
            fclose(f);
        }
        if (siblings.empty()) siblings.push_back(cpu);

        int leader = siblings[0];
        for (int s : siblings) leader = std::min(leader, s);
        if (std::find(seen.begin(), seen.end(), leader) != seen.end()) continue;
        seen.push_back(leader);

        std::vector<int> usable;
        for (int s : siblings) {
            if (s < CPU_SETSIZE && (!mask || CPU_ISSET(s, mask))) usable.push_back(s);
        }
        if (usable.empty()) usable.push_back(cpu);
        cores.push_back(usable);
    }

    std::sort(cores.begin(), cores.end(),
              [](const std::vector<int>& a, const std::vector<int>& b) { return a[0] < b[0]; });
    return cores;
}

// FNV-1a over the process' CPU affinity mask - used to detect how many ranks of
// this node share the same set of cores.
static uint64_t maskHash(const cpu_set_t& set) {
    uint64_t h = 0xcbf29ce484222325ull;
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&set);
    for (size_t i = 0; i < sizeof(set); ++i) h = (h ^ bytes[i]) * 0x100000001b3ull;
    return h;
}

// Bind this rank to its own share of the physical cores and return how many
// cores it got (i.e. the number of OpenMP threads to use).
static int partitionCores(MPI_Comm nodeComm, const int localRank, const int localSize) {
    cpu_set_t mine;
    CPU_ZERO(&mine);
    const bool haveMask = (sched_getaffinity(0, sizeof(mine), &mine) == 0) && CPU_COUNT(&mine) > 0;

    // Ranks with an identical mask have to share the cores of that mask.
    const uint64_t myHash = haveMask ? maskHash(mine) : 0;
    std::vector<uint64_t> hashes(localSize);
    MPI_Allgather(&myHash, 1, MPI_UINT64_T, hashes.data(), 1, MPI_UINT64_T, nodeComm);
    int sharers = 0, myIndex = 0;
    for (int r = 0; r < localSize; ++r) {
        if (hashes[r] != myHash) continue;
        if (r < localRank) ++myIndex;
        ++sharers;
    }
    if (sharers < 1) sharers = 1;

    const std::vector<std::vector<int>> nodeCores = enumerateCores(nullptr);
    std::vector<std::vector<int>> pool = haveMask ? enumerateCores(&mine) : nodeCores;
    int parts = sharers, index = myIndex;

    // If the launcher bound us so tightly that cores would stay idle, ignore
    // its binding and hand out the node's cores ourselves.
    if (pool.size() / (size_t)sharers < nodeCores.size() / (size_t)localSize) {
        pool = nodeCores;
        parts = localSize;
        index = localRank;
    }
    if (pool.empty()) return std::max(1, omp_get_max_threads());

    const size_t first = pool.size() * (size_t)index / (size_t)parts;
    const size_t last = pool.size() * (size_t)(index + 1) / (size_t)parts;
    const size_t nCores = (last > first) ? last - first : 1;

    cpu_set_t target;
    CPU_ZERO(&target);
    for (size_t c = first; c < first + nCores && c < pool.size(); ++c) {
        for (int cpu : pool[c]) CPU_SET(cpu, &target);
    }
    if (CPU_COUNT(&target) > 0) sched_setaffinity(0, sizeof(target), &target);

    return (int)nCores;
}

// Split [0, total) into chunks proportional to `weights`; all but the last
// (host) chunk are rounded down to a multiple of `gran`.
static void weightedSplit(const int total, const std::vector<double>& weights, std::vector<int>& counts,
                          const int gran) {
    const int parts = (int)weights.size();
    counts.assign(parts, 0);
    if (total <= 0 || parts == 0) return;

    double sum = 0.0;
    for (double w : weights) sum += w > 0.0 ? w : 0.0;
    if (sum <= 0.0) {
        counts[parts - 1] = total;
        return;
    }

    int assigned = 0;
    for (int k = 0; k < parts - 1; ++k) {
        int c = (int)(total * (weights[k] > 0.0 ? weights[k] : 0.0) / sum);
        c = (c / gran) * gran;  // keep GPU chunks block aligned
        if (c > total - assigned) c = total - assigned;
        counts[k] = c;
        assigned += c;
    }
    counts[parts - 1] = total - assigned;  // remainder goes to the host
}

// Number of j-segments needed to fill a device with `cnt` bodies.
static int segmentsFor(const int cnt, const int maxThreads, const int n) {
    if (cnt <= 0) return 1;
    int s = (maxThreads + cnt - 1) / cnt;
    const int maxSeg = std::max(1, (n + NBODY_BLOCK - 1) / NBODY_BLOCK);
    s = std::min(s, maxSeg);
    return std::max(1, std::min(s, 64));
}

int main(int argc, char** argv) {
    // Sensible OpenMP defaults; anything the user already set wins.
    setenv("OMP_PROC_BIND", "spread", 0);
    setenv("OMP_PLACES", "cores", 0);

    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, nRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    const bool isRoot = (rank == 0);

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

    if (isRoot) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    if (numBodies <= 0) {
        if (isRoot) {
            if (printResults) print_results(std::vector<double>(), "Bodies");
            if (validate) {
                printf("Validating simulation results...\n");
                printf("Final energy: %.6f\n", 0.0);
                printf("Validation: PASSED\n");
            }
        }
        MPI_Finalize();
        return 0;
    }

    // ---------------------------------------------------------------- topology
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);

    // Give every rank a disjoint set of physical cores and one thread per core.
    const int myCores = partitionCores(nodeComm, localRank, localSize);
    if (!getenv("OMP_NUM_THREADS")) omp_set_num_threads(myCores);
    const int numThreads = omp_get_max_threads();

    // ---------------------------------------------------------------- devices
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) deviceCount = 0;
    if (deviceCount == 0) {
        if (isRoot) fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<int> myDevices;
    if (localSize >= deviceCount) {
        myDevices.push_back(localRank % deviceCount);
    } else {
        const int first = (int)((long)localRank * deviceCount / localSize);
        const int last = (int)((long)(localRank + 1) * deviceCount / localSize);
        for (int d = first; d < last; ++d) myDevices.push_back(d);
    }
    const int nGpu = (int)myDevices.size();

    // ------------------------------------------------------------- rank slice
    std::vector<int> counts(nRanks), displs(nRanks);
    for (int r = 0; r < nRanks; ++r) {
        const int b = (int)((long)r * numBodies / nRanks);
        const int e = (int)((long)(r + 1) * numBodies / nRanks);
        displs[r] = b;
        counts[r] = e - b;
    }

    State st;
    st.n = numBodies;
    st.lo = displs[rank];
    st.hi = displs[rank] + counts[rank];
    const int localN = counts[rank];

    // Pinned host memory for fast (and asynchronous) transfers.
    CUDA_CHECK(cudaSetDevice(myDevices[0]));
    const size_t fullBytes = sizeof(double) * (size_t)numBodies;
    const size_t sliceBytes = sizeof(double) * (size_t)std::max(1, localN);
    CUDA_CHECK(cudaHostAlloc(&st.px, fullBytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&st.py, fullBytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&st.pz, fullBytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&st.vx, fullBytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&st.vy, fullBytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&st.vz, fullBytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&st.npx, sliceBytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&st.npy, sliceBytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&st.npz, sliceBytes, cudaHostAllocPortable));

    // Initialize bodies (identical sequence on every rank)
    {
        std::vector<Body> bodies(numBodies);
        randomizeBodies(bodies);
        for (int i = 0; i < numBodies; ++i) {
            st.px[i] = bodies[i].pos.x;
            st.py[i] = bodies[i].pos.y;
            st.pz[i] = bodies[i].pos.z;
            st.vx[i] = bodies[i].vel.x;
            st.vy[i] = bodies[i].vel.y;
            st.vz[i] = bodies[i].vel.z;
        }
    }

    // ------------------------------------------------------------ GPU buffers
    std::vector<GpuCtx> gpus(nGpu);
    for (int g = 0; g < nGpu; ++g) {
        GpuCtx& c = gpus[g];
        c.dev = myDevices[g];
        CUDA_CHECK(cudaSetDevice(c.dev));

        int sms = 1, threadsPerSm = 1024;
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, c.dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&threadsPerSm, cudaDevAttrMaxThreadsPerMultiProcessor, c.dev));
        c.maxThreads = sms * threadsPerSm;

        CUDA_CHECK(cudaStreamCreate(&c.stream));
        CUDA_CHECK(cudaEventCreate(&c.evStart));
        CUDA_CHECK(cudaEventCreate(&c.evStop));
        CUDA_CHECK(cudaMalloc(&c.px, fullBytes));
        CUDA_CHECK(cudaMalloc(&c.py, fullBytes));
        CUDA_CHECK(cudaMalloc(&c.pz, fullBytes));
        CUDA_CHECK(cudaMalloc(&c.vx, sliceBytes));
        CUDA_CHECK(cudaMalloc(&c.vy, sliceBytes));
        CUDA_CHECK(cudaMalloc(&c.vz, sliceBytes));
        CUDA_CHECK(cudaMalloc(&c.opx, sliceBytes));
        CUDA_CHECK(cudaMalloc(&c.opy, sliceBytes));
        CUDA_CHECK(cudaMalloc(&c.opz, sliceBytes));
        // segments * cnt is bounded by maxThreads + cnt for any chunk size
        const size_t partialSlots = 3 * ((size_t)c.maxThreads + (size_t)localN + NBODY_BLOCK);
        CUDA_CHECK(cudaMalloc(&c.partial, sizeof(double) * partialSlots));
    }

    // Worker weights: one per GPU plus the host at the end.
    std::vector<double> rates(nGpu + 1, INITIAL_GPU_RATE);
    rates[nGpu] = INITIAL_CPU_RATE_PER_THREAD * numThreads;
    std::vector<int> chunk(nGpu + 1, 0);
    std::vector<double> secs(nGpu + 1, 0.0);

    // Warm up the OpenMP thread pool and the CUDA modules so that the first
    // step measures steady-state throughput (its timings drive the balancing).
    #pragma omp parallel
    { }
    for (int g = 0; g < nGpu; ++g) {
        GpuCtx& c = gpus[g];
        const int warmN = std::min(numBodies, NBODY_BLOCK);
        const int warm = std::min(localN, warmN);
        if (warm <= 0) continue;
        CUDA_CHECK(cudaSetDevice(c.dev));
        nbodyLaunchForces(c.px, c.py, c.pz, c.vx, c.vy, c.vz, c.opx, c.opy, c.opz, c.partial, warmN, 0, warm, 1,
                          c.stream);
        CUDA_CHECK(cudaStreamSynchronize(c.stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ------------------------------------------------------------- simulation
    const auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        weightedSplit(localN, rates, chunk, NBODY_BLOCK);

        int offset = st.lo;
        for (int g = 0; g < nGpu; ++g) {
            gpus[g].start = offset;
            gpus[g].cnt = chunk[g];
            offset += chunk[g];
        }
        const int cpuFirst = offset;
        const int cpuLast = st.hi;

        // Launch the GPU part asynchronously, then run the host part.
        for (int g = 0; g < nGpu; ++g) {
            GpuCtx& c = gpus[g];
            if (c.cnt <= 0) continue;
            CUDA_CHECK(cudaSetDevice(c.dev));
            CUDA_CHECK(cudaEventRecord(c.evStart, c.stream));
            CUDA_CHECK(cudaMemcpyAsync(c.px, st.px, fullBytes, cudaMemcpyHostToDevice, c.stream));
            CUDA_CHECK(cudaMemcpyAsync(c.py, st.py, fullBytes, cudaMemcpyHostToDevice, c.stream));
            CUDA_CHECK(cudaMemcpyAsync(c.pz, st.pz, fullBytes, cudaMemcpyHostToDevice, c.stream));
            const size_t cb = sizeof(double) * (size_t)c.cnt;
            CUDA_CHECK(cudaMemcpyAsync(c.vx, st.vx + c.start, cb, cudaMemcpyHostToDevice, c.stream));
            CUDA_CHECK(cudaMemcpyAsync(c.vy, st.vy + c.start, cb, cudaMemcpyHostToDevice, c.stream));
            CUDA_CHECK(cudaMemcpyAsync(c.vz, st.vz + c.start, cb, cudaMemcpyHostToDevice, c.stream));

            nbodyLaunchForces(c.px, c.py, c.pz, c.vx, c.vy, c.vz, c.opx, c.opy, c.opz, c.partial, numBodies,
                              c.start, c.cnt, segmentsFor(c.cnt, c.maxThreads, numBodies), c.stream);

            CUDA_CHECK(cudaMemcpyAsync(st.vx + c.start, c.vx, cb, cudaMemcpyDeviceToHost, c.stream));
            CUDA_CHECK(cudaMemcpyAsync(st.vy + c.start, c.vy, cb, cudaMemcpyDeviceToHost, c.stream));
            CUDA_CHECK(cudaMemcpyAsync(st.vz + c.start, c.vz, cb, cudaMemcpyDeviceToHost, c.stream));
            const int off = c.start - st.lo;
            CUDA_CHECK(cudaMemcpyAsync(st.npx + off, c.opx, cb, cudaMemcpyDeviceToHost, c.stream));
            CUDA_CHECK(cudaMemcpyAsync(st.npy + off, c.opy, cb, cudaMemcpyDeviceToHost, c.stream));
            CUDA_CHECK(cudaMemcpyAsync(st.npz + off, c.opz, cb, cudaMemcpyDeviceToHost, c.stream));
            CUDA_CHECK(cudaEventRecord(c.evStop, c.stream));
        }

        const auto cpuStart = std::chrono::high_resolution_clock::now();
        computeForcesHost(st, cpuFirst, cpuLast);
        secs[nGpu] = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - cpuStart).count();

        for (int g = 0; g < nGpu; ++g) {
            GpuCtx& c = gpus[g];
            if (c.cnt <= 0) {
                secs[g] = 0.0;
                continue;
            }
            CUDA_CHECK(cudaSetDevice(c.dev));
            CUDA_CHECK(cudaStreamSynchronize(c.stream));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, c.evStart, c.evStop));
            secs[g] = ms * 1e-3;
        }

        // Re-estimate the throughput of every worker for the next step.
        for (int k = 0; k <= nGpu; ++k) {
            if (chunk[k] > 0 && secs[k] > 1e-9) {
                const double r = (double)chunk[k] * (double)numBodies / secs[k];
                rates[k] = (step == 0) ? r : 0.5 * rates[k] + 0.5 * r;
            }
        }

        // Publish the new positions to all ranks.
        MPI_Allgatherv(st.npx, localN, MPI_DOUBLE, st.px, counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(st.npy, localN, MPI_DOUBLE, st.py, counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(st.npz, localN, MPI_DOUBLE, st.pz, counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    double maxElapsed = elapsed;
    MPI_Allreduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (isRoot) printf("Simulation time: %ld ms\n", (long)maxElapsed);

    // Velocities are only needed for the output - exchange them once.
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, st.vx, counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, st.vy, counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, st.vz, counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    // Print results for external validation
    if (printResults && isRoot) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData((size_t)numBodies * 6);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            bodyData[(size_t)i * 6 + 0] = st.px[i];
            bodyData[(size_t)i * 6 + 1] = st.py[i];
            bodyData[(size_t)i * 6 + 2] = st.pz[i];
            bodyData[(size_t)i * 6 + 3] = st.vx[i];
            bodyData[(size_t)i * 6 + 4] = st.vy[i];
            bodyData[(size_t)i * 6 + 5] = st.vz[i];
        }
        print_results(bodyData, "Bodies");
    }

    int exitCode = 0;

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (isRoot) printf("Validating simulation results...\n");

        const int code = validateSimulation(st, rank, nRanks, MPI_COMM_WORLD);
        if (code == 0) {
            // Report final energy for reference
            const double finalEnergy = computeTotalEnergy(st, rank, nRanks, MPI_COMM_WORLD);
            if (isRoot) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (isRoot) {
                if (code == 1)
                    printf("Validation failed: found NaN or Inf value in body state\n");
                else if (code == 2)
                    printf("Validation failed: body position exceeds reasonable bounds\n");
                else
                    printf("Validation failed: body velocity exceeds reasonable bounds\n");
                printf("Validation: FAILED\n");
            }
            exitCode = 1;
        }
    }

    for (int g = 0; g < nGpu; ++g) {
        GpuCtx& c = gpus[g];
        cudaSetDevice(c.dev);
        cudaFree(c.px);
        cudaFree(c.py);
        cudaFree(c.pz);
        cudaFree(c.vx);
        cudaFree(c.vy);
        cudaFree(c.vz);
        cudaFree(c.opx);
        cudaFree(c.opy);
        cudaFree(c.opz);
        cudaFree(c.partial);
        cudaEventDestroy(c.evStart);
        cudaEventDestroy(c.evStop);
        cudaStreamDestroy(c.stream);
    }
    cudaFreeHost(st.px);
    cudaFreeHost(st.py);
    cudaFreeHost(st.pz);
    cudaFreeHost(st.vx);
    cudaFreeHost(st.vy);
    cudaFreeHost(st.vz);
    cudaFreeHost(st.npx);
    cudaFreeHost(st.npy);
    cudaFreeHost(st.npz);

    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return exitCode;
}

#endif  // __CUDACC__
