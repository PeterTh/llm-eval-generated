// N-body benchmark: hybrid MPI + OpenMP + CUDA implementation.
//
// Decomposition:
//   * MPI    : bodies are partitioned into contiguous blocks, one block per rank.
//              Every rank keeps a full copy of all positions (needed for the
//              all-pairs force evaluation) which is refreshed with an
//              MPI_Allgatherv after every step.  Each rank binds to one GPU
//              (chosen by node-local rank).
//   * CUDA   : the O(N^2) force evaluation and the integration of the local
//              block are done on the GPU.  Positions of all bodies are streamed
//              through shared memory in tiles; the j-range may additionally be
//              split across blocks (fixed-order partial sums) to fill the GPU
//              for small local block sizes.
//   * OpenMP : host-side O(N^2) energy validation (local rows, cyclic
//              distribution across ranks for load balance) and host-side
//              packing of the result data.
//
// Semantics match the original serial code: the force on body i at step t
// is the sum over all j (including j == i, which contributes 0) using the
// positions at the start of the step, velocities are updated first and the
// positions are then advanced with the updated velocities.

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

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// ---------------------------------------------------------------------------
// Error handling helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,         \
                    cudaGetErrorString(err__));                                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
        }                                                                                      \
    } while (0)

#define MPI_CHECK(call)                                                                        \
    do {                                                                                       \
        const int err__ = (call);                                                              \
        if (err__ != MPI_SUCCESS) {                                                            \
            fprintf(stderr, "MPI error %d in %s at %s:%d\n", err__, #call, __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
        }                                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Initialization (deterministic, identical on every rank)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------
constexpr int BLOCK = 256;

__device__ __forceinline__ void accumulate(const double4& pi, const double4& pj, double& Fx, double& Fy,
                                           double& Fz) {
    const double dx = pj.x - pi.x;
    const double dy = pj.y - pi.y;
    const double dz = pj.z - pi.z;
    const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
    const double invDist = 1.0 / sqrt(distSqr);
    const double invDist3 = invDist * invDist * invDist;
    Fx += dx * invDist3;
    Fy += dy * invDist3;
    Fz += dz * invDist3;
}

// Computes the force on local bodies [0, nLocal) (global index offset + li)
// over the j-range [blockIdx.y * jChunk, min(n, (blockIdx.y + 1) * jChunk)).
// If FUSED, there is only one j-split and the kernel directly integrates;
// otherwise it writes partial sums to partial[blockIdx.y * nLocal + li].
template <bool FUSED>
__global__ void __launch_bounds__(BLOCK) forceKernel(const double4* __restrict__ posAll, const int n,
                                                     const int offset, const int nLocal, const int jChunk,
                                                     double4* __restrict__ vel, double4* __restrict__ posNew,
                                                     double4* __restrict__ partial) {
    __shared__ double4 sh[BLOCK];

    const int li = blockIdx.x * BLOCK + threadIdx.x;
    const int jBegin = blockIdx.y * jChunk;
    const int jEnd = min(n, jBegin + jChunk);

    const double4 pi = (li < nLocal) ? posAll[offset + li] : make_double4(0.0, 0.0, 0.0, 0.0);
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = jBegin; tile < jEnd; tile += BLOCK) {
        const int j = tile + threadIdx.x;
        sh[threadIdx.x] = (j < jEnd) ? posAll[j] : make_double4(0.0, 0.0, 0.0, 0.0);
        __syncthreads();

        const int cnt = min(BLOCK, jEnd - tile);
        if (cnt == BLOCK) {
#pragma unroll 8
            for (int k = 0; k < BLOCK; ++k) accumulate(pi, sh[k], Fx, Fy, Fz);
        } else {
            for (int k = 0; k < cnt; ++k) accumulate(pi, sh[k], Fx, Fy, Fz);
        }
        __syncthreads();
    }

    if (li < nLocal) {
        if constexpr (FUSED) {
            double4 v = vel[li];
            v.x += DT * Fx;
            v.y += DT * Fy;
            v.z += DT * Fz;
            vel[li] = v;
            posNew[li] = make_double4(pi.x + v.x * DT, pi.y + v.y * DT, pi.z + v.z * DT, 0.0);
        } else {
            partial[blockIdx.y * nLocal + li] = make_double4(Fx, Fy, Fz, 0.0);
        }
    }
}

// Reduces the j-split partial sums in a fixed order and integrates.
__global__ void integrateKernel(const double4* __restrict__ posAll, const int offset, const int nLocal,
                                const int splits, const double4* __restrict__ partial,
                                double4* __restrict__ vel, double4* __restrict__ posNew) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= nLocal) return;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int s = 0; s < splits; ++s) {
        const double4 p = partial[s * nLocal + li];
        Fx += p.x;
        Fy += p.y;
        Fz += p.z;
    }
    double4 v = vel[li];
    v.x += DT * Fx;
    v.y += DT * Fy;
    v.z += DT * Fz;
    vel[li] = v;
    const double4 pi = posAll[offset + li];
    posNew[li] = make_double4(pi.x + v.x * DT, pi.y + v.y * DT, pi.z + v.z * DT, 0.0);
}

// ---------------------------------------------------------------------------
// Distributed simulation state
// ---------------------------------------------------------------------------
struct Simulation {
    int rank = 0, size = 1;
    int n = 0;                       // total bodies
    int nLocal = 0, offset = 0;      // local block
    std::vector<int> counts, displs; // in units of doubles (4 per body) for Allgatherv

    double4* d_posAll = nullptr;  // all positions (n)
    double4* d_vel = nullptr;     // local velocities (nLocal)
    double4* d_posNew = nullptr;  // local new positions (nLocal)
    double4* d_partial = nullptr; // j-split partial forces (splits * nLocal)
    double4* h_posAll = nullptr;  // pinned host copy of all positions (n)
    double4* h_vel = nullptr;     // pinned host copy of local velocities (nLocal)

    int splits = 1;
    int jChunk = 0;
    dim3 forceGrid;
    cudaStream_t stream = nullptr;
};

static void selectDevice(int& localRank, int& deviceId) {
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm));
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_free(&nodeComm));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    deviceId = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(deviceId));
}

static void setupSimulation(Simulation& sim, const std::vector<Body>& bodies) {
    sim.n = static_cast<int>(bodies.size());
    sim.counts.resize(sim.size);
    sim.displs.resize(sim.size);
    const int base = sim.n / sim.size, rem = sim.n % sim.size;
    int acc = 0;
    for (int r = 0; r < sim.size; ++r) {
        const int c = base + (r < rem ? 1 : 0);
        sim.counts[r] = 4 * c;
        sim.displs[r] = 4 * acc;
        acc += c;
    }
    sim.nLocal = sim.counts[sim.rank] / 4;
    sim.offset = sim.displs[sim.rank] / 4;

    CUDA_CHECK(cudaStreamCreate(&sim.stream));

    const size_t nAlloc = std::max(sim.n, 1);
    const size_t nLocalAlloc = std::max(sim.nLocal, 1);
    CUDA_CHECK(cudaMallocHost(&sim.h_posAll, nAlloc * sizeof(double4)));
    CUDA_CHECK(cudaMallocHost(&sim.h_vel, nLocalAlloc * sizeof(double4)));
    CUDA_CHECK(cudaMalloc(&sim.d_posAll, nAlloc * sizeof(double4)));
    CUDA_CHECK(cudaMalloc(&sim.d_vel, nLocalAlloc * sizeof(double4)));
    CUDA_CHECK(cudaMalloc(&sim.d_posNew, nLocalAlloc * sizeof(double4)));

    // Pack initial state (host-parallel).
#pragma omp parallel for schedule(static)
    for (int i = 0; i < sim.n; ++i) {
        sim.h_posAll[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < sim.nLocal; ++i) {
        const Body& b = bodies[sim.offset + i];
        sim.h_vel[i] = make_double4(b.vel.x, b.vel.y, b.vel.z, 0.0);
    }
    CUDA_CHECK(cudaMemcpy(sim.d_posAll, sim.h_posAll, sim.n * sizeof(double4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.d_vel, sim.h_vel, sim.nLocal * sizeof(double4), cudaMemcpyHostToDevice));

    // Choose a j-split factor so that the GPU is reasonably filled even when
    // the local block is small (e.g. many ranks or a small N).
    int device = 0, numSMs = 1;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
    const int blocksX = (sim.nLocal + BLOCK - 1) / BLOCK;
    const int targetBlocks = numSMs * 8;
    const int maxSplits = std::max(1, (sim.n + BLOCK - 1) / BLOCK);
    int splits = 1;
    if (blocksX > 0) splits = std::clamp(targetBlocks / blocksX, 1, std::min(maxSplits, 64));
    // jChunk must be a multiple of BLOCK so that tiles never straddle splits.
    int jChunk = (sim.n + splits - 1) / splits;
    jChunk = ((jChunk + BLOCK - 1) / BLOCK) * BLOCK;
    jChunk = std::max(jChunk, BLOCK);
    splits = std::max(1, (sim.n + jChunk - 1) / jChunk);
    sim.splits = splits;
    sim.jChunk = jChunk;
    sim.forceGrid = dim3(std::max(blocksX, 1), splits, 1);
    if (splits > 1) {
        CUDA_CHECK(cudaMalloc(&sim.d_partial, static_cast<size_t>(splits) * nLocalAlloc * sizeof(double4)));
    }

    // Force the kernels' modules to be loaded now so that lazy module loading
    // does not land inside the timed region.
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, forceKernel<true>));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, forceKernel<false>));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, integrateKernel));
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void destroySimulation(Simulation& sim) {
    if (sim.d_partial) cudaFree(sim.d_partial);
    cudaFree(sim.d_posNew);
    cudaFree(sim.d_vel);
    cudaFree(sim.d_posAll);
    cudaFreeHost(sim.h_vel);
    cudaFreeHost(sim.h_posAll);
    cudaStreamDestroy(sim.stream);
}

// One full time step: force + integrate on the GPU, then refresh the global
// position array on every rank.
static void stepSimulation(Simulation& sim) {
    if (sim.nLocal > 0) {
        if (sim.splits == 1) {
            forceKernel<true><<<sim.forceGrid, BLOCK, 0, sim.stream>>>(sim.d_posAll, sim.n, sim.offset, sim.nLocal,
                                                                       sim.jChunk, sim.d_vel, sim.d_posNew,
                                                                       nullptr);
        } else {
            forceKernel<false><<<sim.forceGrid, BLOCK, 0, sim.stream>>>(sim.d_posAll, sim.n, sim.offset,
                                                                        sim.nLocal, sim.jChunk, nullptr, nullptr,
                                                                        sim.d_partial);
            const int blocks = (sim.nLocal + BLOCK - 1) / BLOCK;
            integrateKernel<<<blocks, BLOCK, 0, sim.stream>>>(sim.d_posAll, sim.offset, sim.nLocal, sim.splits,
                                                              sim.d_partial, sim.d_vel, sim.d_posNew);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    if (sim.size == 1) {
        // Single rank: positions never need to leave the device.
        CUDA_CHECK(cudaMemcpyAsync(sim.d_posAll + sim.offset, sim.d_posNew, sim.nLocal * sizeof(double4),
                                   cudaMemcpyDeviceToDevice, sim.stream));
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(sim.h_posAll + sim.offset, sim.d_posNew, sim.nLocal * sizeof(double4),
                               cudaMemcpyDeviceToHost, sim.stream));
    CUDA_CHECK(cudaStreamSynchronize(sim.stream));
    MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, sim.h_posAll, sim.counts.data(),
                             sim.displs.data(), MPI_DOUBLE, MPI_COMM_WORLD));
    CUDA_CHECK(cudaMemcpyAsync(sim.d_posAll, sim.h_posAll, sim.n * sizeof(double4), cudaMemcpyHostToDevice,
                               sim.stream));
}

// Bring the final state back to the host. After this call every rank holds all
// positions in h_posAll and its local velocities in h_vel.
static void syncToHost(Simulation& sim) {
    if (sim.size == 1) {
        CUDA_CHECK(cudaMemcpyAsync(sim.h_posAll, sim.d_posAll, sim.n * sizeof(double4), cudaMemcpyDeviceToHost,
                                   sim.stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(sim.h_vel, sim.d_vel, sim.nLocal * sizeof(double4), cudaMemcpyDeviceToHost,
                               sim.stream));
    CUDA_CHECK(cudaStreamSynchronize(sim.stream));
}

// Gathers the complete body state onto rank 0 (other ranks return an empty vector).
static std::vector<Body> gatherBodies(const Simulation& sim) {
    std::vector<double4> velAll;
    if (sim.rank == 0) velAll.resize(std::max(sim.n, 1));
    MPI_CHECK(MPI_Gatherv(sim.h_vel, sim.counts[sim.rank], MPI_DOUBLE, velAll.data(), sim.counts.data(),
                          sim.displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
    std::vector<Body> bodies;
    if (sim.rank == 0) {
        bodies.resize(sim.n);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < sim.n; ++i) {
            bodies[i].pos = Vec3(sim.h_posAll[i].x, sim.h_posAll[i].y, sim.h_posAll[i].z);
            bodies[i].vel = Vec3(velAll[i].x, velAll[i].y, velAll[i].z);
        }
    }
    return bodies;
}

// Total energy (unit masses), distributed over ranks and OpenMP threads.
// Rows i of the pair sum are assigned cyclically to ranks for load balance.
double computeTotalEnergy(const Simulation& sim) {
    double energy = 0.0;
    const int n = sim.n;
    const double4* pos = sim.h_posAll;

    // Kinetic energy of the local bodies
#pragma omp parallel for reduction(+ : energy) schedule(static)
    for (int i = 0; i < sim.nLocal; ++i) {
        const double4 v = sim.h_vel[i];
        energy += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }

    // Potential energy for pairs (i, j>i) with i in this rank's cyclic rows
#pragma omp parallel for reduction(+ : energy) schedule(dynamic, 8)
    for (int i = sim.rank; i < n; i += sim.size) {
        const double4 pi = pos[i];
        double local = 0.0;
        for (int j = i + 1; j < n; ++j) {
            const double dx = pos[j].x - pi.x;
            const double dy = pos[j].y - pi.y;
            const double dz = pos[j].z - pi.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            local -= 1.0 / dist;
        }
        energy += local;
    }

    double total = 0.0;
    MPI_CHECK(MPI_Reduce(&energy, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
    return total;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
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
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));

    Simulation sim;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &sim.rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &sim.size));
    const bool isRoot = (sim.rank == 0);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on all ranks)
    int earlyExit = -1;
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
            earlyExit = 0;
            break;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            earlyExit = 1;
            break;
        }
    }
    if (earlyExit >= 0) {
        MPI_Finalize();
        return earlyExit;
    }
    if (numBodies < 0) numBodies = 0;

    if (isRoot) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        fflush(stdout);
    }

    int localRank = 0, deviceId = 0;
    selectDevice(localRank, deviceId);

    // Initialize bodies (deterministic sequence, replicated on every rank)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    setupSimulation(sim, bodies);
    bodies.clear();
    bodies.shrink_to_fit();

    // Run simulation
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        stepSimulation(sim);
    }
    CUDA_CHECK(cudaStreamSynchronize(sim.stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (isRoot) printf("Simulation time: %ld ms\n", static_cast<long>(duration.count()));

    syncToHost(sim);
    const std::vector<Body> finalBodies = gatherBodies(sim);

    // Print results for external validation
    if (printResults && isRoot) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const Body& body = finalBodies[i];
            double* out = bodyData.data() + static_cast<size_t>(i) * 6;
            out[0] = body.pos.x;
            out[1] = body.pos.y;
            out[2] = body.pos.z;
            out[3] = body.vel.x;
            out[4] = body.vel.y;
            out[5] = body.vel.z;
        }
        print_results(bodyData, "Bodies");
    }

    int exitCode = 0;
    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (isRoot) printf("Validating simulation results...\n");

        int ok = isRoot ? (validateSimulation(finalBodies) ? 1 : 0) : 0;
        MPI_CHECK(MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD));
        if (ok) {
            // Report final energy for reference
            const double finalEnergy = computeTotalEnergy(sim);
            if (isRoot) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (isRoot) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    destroySimulation(sim);
    MPI_Finalize();
    return exitCode;
}
