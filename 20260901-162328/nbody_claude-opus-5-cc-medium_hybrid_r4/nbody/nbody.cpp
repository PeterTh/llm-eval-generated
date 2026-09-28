#include <chrono>
#include <cmath>
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

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// ---------------------------------------------------------------------------
// Parallelization strategy
//
//   MPI    : the body index range is distributed over ranks. Every rank holds a
//            full copy of all positions and updates only its own slice of the
//            bodies. Positions are re-synchronized with one Allgatherv per step.
//   CUDA   : every rank drives one or more GPUs (devices are partitioned over
//            the ranks of a node) which compute the forces for a part of the
//            rank's slice.
//   OpenMP : the remaining part of the rank's slice is computed by the CPU
//            cores concurrently with the GPU work; OpenMP also drives the
//            integration, the packing/unpacking of communication buffers and
//            the energy computation.
//
// The CPU/GPU work split is auto-tuned at run time from measured throughputs.
//
// Both the CPU and the GPU force kernel accumulate the interactions of one body
// strictly in ascending j order (the CPU kernel vectorizes across *i*, not over
// the reduction axis), and all fused multiply-adds are spelled out explicitly
// (with FP contraction disabled in the build), so the results are bit-identical
// to the sequential reference implementation on both the CPU and the GPU.
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                                                     \
    do {                                                                                                     \
        const cudaError_t err_ = (call);                                                                     \
        if (err_ != cudaSuccess) {                                                                           \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__);                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                    \
        }                                                                                                    \
    } while (0)

constexpr int TILE = 32;  // threads (and bodies) per block

// One thread integrates the force on one body over all other bodies, in
// ascending j order, reading the source positions through a shared memory tile.
__global__ __launch_bounds__(TILE) void forceKernel(const double* __restrict__ px, const double* __restrict__ py,
                                                    const double* __restrict__ pz, double* __restrict__ vx,
                                                    double* __restrict__ vy, double* __restrict__ vz, const int n,
                                                    const int start, const int count) {
    __shared__ double sx[TILE];
    __shared__ double sy[TILE];
    __shared__ double sz[TILE];

    const int li = blockIdx.x * TILE + threadIdx.x;
    const int i = start + (li < count ? li : 0);

    const double xi = px[i];
    const double yi = py[i];
    const double zi = pz[i];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int t = 0; t < n; t += TILE) {
        const int idx = t + threadIdx.x;
        if (idx < n) {
            sx[threadIdx.x] = px[idx];
            sy[threadIdx.x] = py[idx];
            sz[threadIdx.x] = pz[idx];
        }
        __syncthreads();

        const int m = min(TILE, n - t);
#pragma unroll 4
        for (int k = 0; k < m; ++k) {
            const double dx = sx[k] - xi;
            const double dy = sy[k] - yi;
            const double dz = sz[k] - zi;
            const double distSqr = fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx = fma(dx, invDist3, Fx);
            Fy = fma(dy, invDist3, Fy);
            Fz = fma(dz, invDist3, Fz);
        }
        __syncthreads();
    }

    if (li < count) {
        vx[li] += DT * Fx;
        vy[li] += DT * Fy;
        vz[li] += DT * Fz;
    }
}

struct GpuCtx {
    int dev = 0;
    cudaStream_t stream = nullptr;
    double *px = nullptr, *py = nullptr, *pz = nullptr;
    double *vx = nullptr, *vy = nullptr, *vz = nullptr;
    cudaEvent_t evStart = nullptr, evStop = nullptr;
    int start = 0, count = 0;
};

// Force computation on the CPU cores. Vectorization happens across the i-axis
// (VL bodies at a time) so that every accumulator sees the j-values in the
// original sequential order.
constexpr int VL = 8;

static void computeForcesCPU(const double* __restrict__ px, const double* __restrict__ py,
                             const double* __restrict__ pz, double* __restrict__ vx, double* __restrict__ vy,
                             double* __restrict__ vz, const int n, const int start, const int count) {
    if (count <= 0) return;
    const int end = start + count;

#pragma omp parallel for schedule(static)
    for (int ib = start; ib < end; ib += VL) {
        double xi[VL], yi[VL], zi[VL];
        double Fx[VL], Fy[VL], Fz[VL];

        const int m = (end - ib) < VL ? (end - ib) : VL;
        for (int k = 0; k < VL; ++k) {
            const int idx = (k < m) ? (ib + k) : ib;
            xi[k] = px[idx];
            yi[k] = py[idx];
            zi[k] = pz[idx];
            Fx[k] = 0.0;
            Fy[k] = 0.0;
            Fz[k] = 0.0;
        }

        for (int j = 0; j < n; ++j) {
            const double xj = px[j];
            const double yj = py[j];
            const double zj = pz[j];
#pragma omp simd simdlen(VL)
            for (int k = 0; k < VL; ++k) {
                const double dx = xj - xi[k];
                const double dy = yj - yi[k];
                const double dz = zj - zi[k];
                const double distSqr = __builtin_fma(dz, dz, __builtin_fma(dx, dx, dy * dy)) + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx[k] = __builtin_fma(dx, invDist3, Fx[k]);
                Fy[k] = __builtin_fma(dy, invDist3, Fy[k]);
                Fz[k] = __builtin_fma(dz, invDist3, Fz[k]);
            }
        }

        for (int k = 0; k < m; ++k) {
            vx[ib + k] += DT * Fx[k];
            vy[ib + k] += DT * Fy[k];
            vz[ib + k] += DT * Fz[k];
        }
    }
}

void randomizeBodies(double* px, double* py, double* pz, double* vx, double* vy, double* vz, const int n,
                     unsigned int seed = 42) {
    for (int i = 0; i < n; ++i) {
        px[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        py[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vx[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vy[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

void integrateBodies(double* px, double* py, double* pz, const double* vx, const double* vy, const double* vz,
                     const int start, const int count) {
#pragma omp parallel for schedule(static)
    for (int i = start; i < start + count; ++i) {
        px[i] = __builtin_fma(vx[i], DT, px[i]);
        py[i] = __builtin_fma(vy[i], DT, py[i]);
        pz[i] = __builtin_fma(vz[i], DT, pz[i]);
    }
}

// Total energy (unit masses). The potential part is evaluated as one partial
// sum per body (ascending j order) which are then accumulated in ascending i
// order, matching the reference evaluation order.
double computeTotalEnergy(const double* px, const double* py, const double* pz, const double* vx, const double* vy,
                          const double* vz, const int n, const int rank, const int nranks) {
    // Balance the triangular loop over the ranks.
    std::vector<int> bounds(nranks + 1);
    bounds[0] = 0;
    bounds[nranks] = n;
    for (int r = 1; r < nranks; ++r) {
        const double frac = static_cast<double>(r) / static_cast<double>(nranks);
        int b = static_cast<int>(n * (1.0 - std::sqrt(1.0 - frac)) + 0.5);
        if (b < bounds[r - 1]) b = bounds[r - 1];
        if (b > n) b = n;
        bounds[r] = b;
    }

    const int lstart = bounds[rank];
    const int lcount = bounds[rank + 1] - bounds[rank];

    std::vector<double> partial(n, 0.0);

#pragma omp parallel for schedule(dynamic, 16)
    for (int i = lstart; i < lstart + lcount; ++i) {
        double s = 0.0;
        const double xi = px[i], yi = py[i], zi = pz[i];
        for (int j = i + 1; j < n; ++j) {
            const double dx = px[j] - xi;
            const double dy = py[j] - yi;
            const double dz = pz[j] - zi;
            const double dist = std::sqrt(__builtin_fma(dz, dz, __builtin_fma(dx, dx, dy * dy)) + SOFTENING);
            s -= 1.0 / dist;
        }
        partial[i] = s;
    }

    if (nranks > 1) {
        std::vector<int> counts(nranks), displs(nranks);
        for (int r = 0; r < nranks; ++r) {
            displs[r] = bounds[r];
            counts[r] = bounds[r + 1] - bounds[r];
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, partial.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }

    double energy = 0.0;
    // Kinetic energy (assuming unit mass)
    for (int i = 0; i < n; ++i) {
        energy += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    }
    // Potential energy (assuming unit mass for all bodies)
    for (int i = 0; i < n; ++i) {
        energy += partial[i];
    }

    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const double* px, const double* py, const double* pz, const double* vx, const double* vy,
                        const double* vz, const int n) {
    for (int i = 0; i < n; ++i) {
        // Check for NaN or Inf values
        if (!std::isfinite(px[i]) || !std::isfinite(py[i]) || !std::isfinite(pz[i]) || !std::isfinite(vx[i]) ||
            !std::isfinite(vy[i]) || !std::isfinite(vz[i])) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(px[i]) > maxPos || std::abs(py[i]) > maxPos || std::abs(pz[i]) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(vx[i]) > maxVel || std::abs(vy[i]) > maxVel || std::abs(vz[i]) > maxVel) {
            printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Rank layout on this node (used for device assignment and thread counts).
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);

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

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    if (numBodies <= 0) {
        MPI_Finalize();
        return 0;
    }

    const int n = numBodies;

    // Block distribution of the bodies over the ranks.
    std::vector<int> rStart(nranks + 1);
    for (int r = 0; r <= nranks; ++r) {
        rStart[r] = static_cast<int>((static_cast<long long>(n) * r) / nranks);
    }
    const int lstart = rStart[rank];
    const int lcount = rStart[rank + 1] - rStart[rank];

    // Host state (pinned for fast transfers).
    double *px, *py, *pz, *vx, *vy, *vz, *cbuf;
    CUDA_CHECK(cudaMallocHost(&px, sizeof(double) * n));
    CUDA_CHECK(cudaMallocHost(&py, sizeof(double) * n));
    CUDA_CHECK(cudaMallocHost(&pz, sizeof(double) * n));
    CUDA_CHECK(cudaMallocHost(&vx, sizeof(double) * n));
    CUDA_CHECK(cudaMallocHost(&vy, sizeof(double) * n));
    CUDA_CHECK(cudaMallocHost(&vz, sizeof(double) * n));
    CUDA_CHECK(cudaMallocHost(&cbuf, sizeof(double) * 3 * n));

    // Identical (deterministic) initialization on every rank.
    randomizeBodies(px, py, pz, vx, vy, vz, n);

    // Assign the node's GPUs to the ranks of this node.
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        if (rank == 0) printf("No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Every device of the node is owned by exactly one rank; if there are more
    // ranks than devices, the ranks share them round robin.
    std::vector<int> myDevices;
    for (int d = localRank; d < devCount; d += localSize) myDevices.push_back(d);
    if (myDevices.empty()) myDevices.push_back(localRank % devCount);

    std::vector<GpuCtx> gpus;
    for (const int d : myDevices) {
        GpuCtx g;
        g.dev = d;
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamCreate(&g.stream));
        CUDA_CHECK(cudaMalloc(&g.px, sizeof(double) * n));
        CUDA_CHECK(cudaMalloc(&g.py, sizeof(double) * n));
        CUDA_CHECK(cudaMalloc(&g.pz, sizeof(double) * n));
        CUDA_CHECK(cudaMalloc(&g.vx, sizeof(double) * (lcount + 1)));
        CUDA_CHECK(cudaMalloc(&g.vy, sizeof(double) * (lcount + 1)));
        CUDA_CHECK(cudaMalloc(&g.vz, sizeof(double) * (lcount + 1)));
        CUDA_CHECK(cudaEventCreate(&g.evStart));
        CUDA_CHECK(cudaEventCreate(&g.evStop));
        gpus.push_back(g);
    }
    const int numGpus = static_cast<int>(gpus.size());

    // Pick the OpenMP team size: the share of the node's cores that belongs to
    // this rank, minus a few cores that are left to the CUDA driver threads
    // (having every logical CPU spinning in an OpenMP barrier stalls the
    // asynchronous GPU work and costs far more than the lost cores).
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        const long online = sysconf(_SC_NPROCESSORS_ONLN);
        cpu_set_t mask;
        CPU_ZERO(&mask);
        int inMask = static_cast<int>(online);
        if (sched_getaffinity(0, sizeof(mask), &mask) == 0) inMask = CPU_COUNT(&mask);

        // MPI launchers bind a rank to a single core by default, which would
        // leave most of the node idle for a hybrid run: claim this rank's share
        // of the node instead (a cgroup restriction still takes precedence).
        if (inMask < online / localSize) {
            cpu_set_t wide;
            CPU_ZERO(&wide);
            for (long c = 0; c < online; ++c) {
                if (c % localSize == localRank) CPU_SET(c, &wide);
            }
            if (sched_setaffinity(0, sizeof(wide), &wide) == 0) inMask = CPU_COUNT(&wide);
        }

        // Ranks may share an (overlapping) binding, so never claim more than an
        // equal share of the node.
        const int avail = std::min<int>(inMask, static_cast<int>(online / localSize));
        int threads = avail - (numGpus + 1);
        if (threads < 1) threads = 1;
        omp_set_num_threads(threads);
    }

    // Communication descriptors (3 doubles per body, interleaved).
    std::vector<int> ccounts(nranks), cdispls(nranks);
    for (int r = 0; r < nranks; ++r) {
        cdispls[r] = 3 * rStart[r];
        ccounts[r] = 3 * (rStart[r + 1] - rStart[r]);
    }

    // Issues the force computation for [gstart, gstart+gtotal) on the GPUs of
    // this rank (split evenly) into their streams; returns immediately.
    auto gpuLaunch = [&](const int gstart, const int gtotal, double* hvx, double* hvy, double* hvz) {
        for (int gi = 0; gi < numGpus; ++gi) {
            GpuCtx& g = gpus[gi];
            const int base = static_cast<int>((static_cast<long long>(gtotal) * gi) / numGpus);
            const int next = static_cast<int>((static_cast<long long>(gtotal) * (gi + 1)) / numGpus);
            g.start = gstart + base;
            g.count = next - base;
            if (g.count <= 0) continue;
            CUDA_CHECK(cudaSetDevice(g.dev));
            CUDA_CHECK(cudaEventRecord(g.evStart, g.stream));
            CUDA_CHECK(cudaMemcpyAsync(g.px, px, sizeof(double) * n, cudaMemcpyHostToDevice, g.stream));
            CUDA_CHECK(cudaMemcpyAsync(g.py, py, sizeof(double) * n, cudaMemcpyHostToDevice, g.stream));
            CUDA_CHECK(cudaMemcpyAsync(g.pz, pz, sizeof(double) * n, cudaMemcpyHostToDevice, g.stream));
            CUDA_CHECK(
                cudaMemcpyAsync(g.vx, hvx + g.start, sizeof(double) * g.count, cudaMemcpyHostToDevice, g.stream));
            CUDA_CHECK(
                cudaMemcpyAsync(g.vy, hvy + g.start, sizeof(double) * g.count, cudaMemcpyHostToDevice, g.stream));
            CUDA_CHECK(
                cudaMemcpyAsync(g.vz, hvz + g.start, sizeof(double) * g.count, cudaMemcpyHostToDevice, g.stream));
            const int blocks = (g.count + TILE - 1) / TILE;
            forceKernel<<<blocks, TILE, 0, g.stream>>>(g.px, g.py, g.pz, g.vx, g.vy, g.vz, n, g.start, g.count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(
                cudaMemcpyAsync(hvx + g.start, g.vx, sizeof(double) * g.count, cudaMemcpyDeviceToHost, g.stream));
            CUDA_CHECK(
                cudaMemcpyAsync(hvy + g.start, g.vy, sizeof(double) * g.count, cudaMemcpyDeviceToHost, g.stream));
            CUDA_CHECK(
                cudaMemcpyAsync(hvz + g.start, g.vz, sizeof(double) * g.count, cudaMemcpyDeviceToHost, g.stream));
            CUDA_CHECK(cudaEventRecord(g.evStop, g.stream));
        }
    };

    // Waits for the outstanding GPU work and returns the time the slowest
    // device spent on it (seconds, measured on the device itself).
    auto gpuWait = [&]() {
        double tmax = 0.0;
        for (int gi = 0; gi < numGpus; ++gi) {
            GpuCtx& g = gpus[gi];
            if (g.count <= 0) continue;
            CUDA_CHECK(cudaSetDevice(g.dev));
            CUDA_CHECK(cudaStreamSynchronize(g.stream));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, g.evStart, g.evStop));
            if (ms * 1e-3 > tmax) tmax = ms * 1e-3;
        }
        return tmax;
    };

    // Use one team size for every parallel region of a step: alternating team
    // sizes makes the OpenMP runtime tear down and re-create threads. Small
    // problems get a smaller team, otherwise the barriers dominate.
    const int maxThreads = omp_get_max_threads();
    const long long work = static_cast<long long>(lcount) * n;
    const int byWork = static_cast<int>(work / 131072);
    const int byBlocks = (lcount + VL - 1) / VL;
    omp_set_num_threads(std::max(1, std::min(maxThreads, std::min(byWork, byBlocks))));

    // --- calibration -------------------------------------------------------
    // The GPU force kernel is latency bound for small body counts and only
    // reaches its peak rate once the devices are saturated, so its cost is
    // modelled as gpuFixed + gpuSlope * bodies. Both coefficients (and the CPU
    // rate) are calibrated here and refined from the measurements of every step.
    double cpuRate = 1.0;                     // bodies per second
    double gpuFixed = 0.0, gpuSlope = 1e-12;  // seconds, seconds per body
    {
        std::vector<double> svx(n, 0.0), svy(n, 0.0), svz(n, 0.0);

        // Two probe sizes: the difference of the two timings cancels the fixed
        // cost of a parallel region, which would otherwise dominate the small
        // probe and make the CPU look much slower than it is.
        const int c1 = std::min(lcount, 2048);
        const int c2 = std::min(lcount, 8192);
        if (c1 > 0) {
            computeForcesCPU(px, py, pz, svx.data(), svy.data(), svz.data(), n, lstart, c1);  // thread pool
            const double t0 = omp_get_wtime();
            computeForcesCPU(px, py, pz, svx.data(), svy.data(), svz.data(), n, lstart, c1);
            const double t1 = omp_get_wtime() - t0;
            computeForcesCPU(px, py, pz, svx.data(), svy.data(), svz.data(), n, lstart, c2);
            const double t2 = omp_get_wtime() - t0 - t1;
            if (c2 > c1 && t2 > t1) {
                cpuRate = (c2 - c1) / (t2 - t1);
            } else if (t1 > 0.0) {
                cpuRate = c1 / t1;
            }
        }

        // One block per device measures the latency floor of the kernel (the
        // devices are idle for most of it). The first launch is discarded: it
        // still contains context creation and the clock ramp up.
        const int warmProbe = std::min(lcount, TILE * numGpus);
        if (warmProbe > 0) {
            gpuLaunch(lstart, warmProbe, svx.data(), svy.data(), svz.data());
            gpuWait();
            gpuLaunch(lstart, warmProbe, svx.data(), svy.data(), svz.data());
            gpuFixed = gpuWait();
            // Prior for the marginal cost: a device saturates at a few thousand
            // bodies, which is refined from the measurements of every step.
            gpuSlope = std::max(gpuFixed / (7500.0 * numGpus), 1e-12);
        }
    }

    // Work split minimizing max(CPU time, GPU time) for the model above; the
    // new value is damped against the current one to avoid oscillations.
    auto solveSplit = [&](const int previous, const double damp) {
        if (numGpus == 0) return 0;
        const double g = (lcount / cpuRate - gpuFixed) / (gpuSlope + 1.0 / cpuRate);
        int target = (g <= 0.0) ? 0 : static_cast<int>(damp * previous + (1.0 - damp) * g);
        target = (target / TILE) * TILE;
        if (target > lcount) target = lcount;
        if (target < TILE) target = 0;
        return target;
    };

    int gpuTotal = solveSplit(0, 0.0);
    double prevG = 0.0, prevT = 0.0;  // previous observation used for the fit

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        const int cpuCount = lcount - gpuTotal;

        // --- launch GPU part (asynchronous) ---
        if (gpuTotal > 0) gpuLaunch(lstart + cpuCount, gpuTotal, vx, vy, vz);

        // --- CPU part, concurrent with the GPUs ---
        const double tc0 = omp_get_wtime();
        computeForcesCPU(px, py, pz, vx, vy, vz, n, lstart, cpuCount);
        const double tcpu = omp_get_wtime() - tc0;

        const double tgpu = (gpuTotal > 0) ? gpuWait() : 0.0;

        // --- refine the model and rebalance for the next step ---
        if (step + 1 < numSteps) {
            if (cpuCount > 0 && tcpu > 0.0) {
                cpuRate = 0.5 * (cpuRate + cpuCount / tcpu);
            }
            if (gpuTotal > 0 && tgpu > 0.0) {
                if (std::abs(gpuTotal - prevG) > 0.02 * lcount) {
                    // Two sufficiently distinct observations: fit both coefficients.
                    const double slope = (tgpu - prevT) / (gpuTotal - prevG);
                    if (slope > 0.0) {
                        gpuSlope = slope;
                        gpuFixed = std::max(tgpu - slope * gpuTotal, 0.0);
                    } else {
                        // Cost did not grow with the work: the devices are purely
                        // latency bound in this range.
                        gpuSlope = 1e-12;
                        gpuFixed = std::max(tgpu, prevT);
                    }
                } else if (tgpu > gpuFixed) {
                    gpuSlope = 0.5 * (gpuSlope + (tgpu - gpuFixed) / gpuTotal);
                }
                prevG = gpuTotal;
                prevT = tgpu;
            }
            gpuTotal = solveSplit(gpuTotal, 0.5);
        }

        // --- integrate and exchange positions ---
        integrateBodies(px, py, pz, vx, vy, vz, lstart, lcount);

        if (nranks > 1) {
#pragma omp parallel for schedule(static)
            for (int i = lstart; i < lstart + lcount; ++i) {
                cbuf[3 * i + 0] = px[i];
                cbuf[3 * i + 1] = py[i];
                cbuf[3 * i + 2] = pz[i];
            }
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, cbuf, ccounts.data(), cdispls.data(), MPI_DOUBLE,
                           MPI_COMM_WORLD);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                if (i >= lstart && i < lstart + lcount) continue;
                px[i] = cbuf[3 * i + 0];
                py[i] = cbuf[3 * i + 1];
                pz[i] = cbuf[3 * i + 2];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", maxDuration);
    }

    // Collect the velocities (positions are already replicated).
    if (nranks > 1) {
#pragma omp parallel for schedule(static)
        for (int i = lstart; i < lstart + lcount; ++i) {
            cbuf[3 * i + 0] = vx[i];
            cbuf[3 * i + 1] = vy[i];
            cbuf[3 * i + 2] = vz[i];
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, cbuf, ccounts.data(), cdispls.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            vx[i] = cbuf[3 * i + 0];
            vy[i] = cbuf[3 * i + 1];
            vz[i] = cbuf[3 * i + 2];
        }
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(n) * 6);
        for (int i = 0; i < n; ++i) {
            bodyData.push_back(px[i]);
            bodyData.push_back(py[i]);
            bodyData.push_back(pz[i]);
            bodyData.push_back(vx[i]);
            bodyData.push_back(vy[i]);
            bodyData.push_back(vz[i]);
        }
        print_results(bodyData, "Bodies");
    }

    omp_set_num_threads(maxThreads);

    int exitCode = 0;

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        // The state is replicated, so one rank checks it and tells the others.
        int ok = (rank == 0) ? (validateSimulation(px, py, pz, vx, vy, vz, n) ? 1 : 0) : 0;
        if (nranks > 1) MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (ok) {
            // Report final energy for reference
            const double finalEnergy = computeTotalEnergy(px, py, pz, vx, vy, vz, n, rank, nranks);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    for (auto& g : gpus) {
        CUDA_CHECK(cudaSetDevice(g.dev));
        cudaFree(g.px);
        cudaFree(g.py);
        cudaFree(g.pz);
        cudaFree(g.vx);
        cudaFree(g.vy);
        cudaFree(g.vz);
        cudaEventDestroy(g.evStart);
        cudaEventDestroy(g.evStop);
        cudaStreamDestroy(g.stream);
    }
    cudaFreeHost(px);
    cudaFreeHost(py);
    cudaFreeHost(pz);
    cudaFreeHost(vx);
    cudaFreeHost(vy);
    cudaFreeHost(vz);
    cudaFreeHost(cbuf);

    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return exitCode;
}
