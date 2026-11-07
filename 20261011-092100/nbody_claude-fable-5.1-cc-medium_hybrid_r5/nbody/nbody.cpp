// Hybrid MPI + OpenMP + CUDA N-body benchmark.
//
// Decomposition:
//   * MPI     : every rank owns a contiguous block of bodies. Each step the rank
//               computes forces / integrates its own bodies against the full
//               position set and the updated positions are exchanged with
//               MPI_Allgatherv (host staged in pinned memory).
//   * OpenMP  : one host thread per GPU attached to the rank drives its CUDA
//               stream concurrently (ranks with several visible GPUs split
//               their block across them). OpenMP is also used for the host
//               side O(N) passes (validation, result packing).
//   * CUDA    : the O(N^2) force evaluation and the energy evaluation run on
//               the GPU with shared-memory tiling of the position set. The j
//               range is additionally split into K chunks so that small local
//               blocks still produce enough threads to fill the device; the
//               chunk partial sums are reduced in a fixed order.
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
constexpr int BLOCK = 256;  // threads per block == bodies per shared-memory tile

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t err__ = (call);                                                             \
        if (err__ != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,          \
                    cudaGetErrorString(err__));                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

// ----------------------------------------------------------------------------
// CUDA kernels
// ----------------------------------------------------------------------------

// Partial force on local body i (global index start+i) from the j-chunk blockIdx.y.
// pos: full position array (x,y,z interleaved), partial: SoA [K][3][nLocal].
__global__ void forcesKernel(const double* __restrict__ pos, const int n, const int start, const int nLocal,
                             const int chunkLen, double* __restrict__ partial) {
    __shared__ double sh[3 * BLOCK];

    const int i = blockIdx.x * BLOCK + threadIdx.x;
    const int c = blockIdx.y;
    const int jBegin = c * chunkLen;
    const int jEnd = min(n, jBegin + chunkLen);

    double px = 0.0, py = 0.0, pz = 0.0;
    if (i < nLocal) {
        const double* p = pos + 3 * (size_t)(start + i);
        px = p[0];
        py = p[1];
        pz = p[2];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int tile = jBegin; tile < jEnd; tile += BLOCK) {
        const int cnt = min(BLOCK, jEnd - tile);
        for (int t = threadIdx.x; t < 3 * cnt; t += BLOCK) sh[t] = pos[3 * (size_t)tile + t];
        __syncthreads();
        if (i < nLocal) {
#pragma unroll 8
            for (int j = 0; j < cnt; ++j) {
                const double dx = sh[3 * j + 0] - px;
                const double dy = sh[3 * j + 1] - py;
                const double dz = sh[3 * j + 2] - pz;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (i < nLocal) {
        double* out = partial + (size_t)c * 3 * nLocal;
        out[0 * (size_t)nLocal + i] = Fx;
        out[1 * (size_t)nLocal + i] = Fy;
        out[2 * (size_t)nLocal + i] = Fz;
    }
}

// Reduce the K chunk partials in fixed order, kick the velocity, drift the position.
__global__ void updateKernel(const double* __restrict__ partial, const int K, const int start, const int nLocal,
                             double* __restrict__ vel, double* __restrict__ pos) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= nLocal) return;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int c = 0; c < K; ++c) {
        const double* in = partial + (size_t)c * 3 * nLocal;
        Fx += in[0 * (size_t)nLocal + i];
        Fy += in[1 * (size_t)nLocal + i];
        Fz += in[2 * (size_t)nLocal + i];
    }

    double* v = vel + 3 * (size_t)i;
    const double vx = v[0] + DT * Fx;
    const double vy = v[1] + DT * Fy;
    const double vz = v[2] + DT * Fz;
    v[0] = vx;
    v[1] = vy;
    v[2] = vz;

    double* p = pos + 3 * (size_t)(start + i);
    p[0] += vx * DT;
    p[1] += vy * DT;
    p[2] += vz * DT;
}

// Total energy contribution of the local bodies (unit mass):
// kinetic 0.5*v^2 plus half of the pair potential -1/dist over all j != i.
__global__ void energyKernel(const double* __restrict__ pos, const double* __restrict__ vel, const int n,
                             const int start, const int nLocal, double* __restrict__ blockSums) {
    __shared__ double sh[3 * BLOCK];
    __shared__ double red[BLOCK];

    const int i = blockIdx.x * BLOCK + threadIdx.x;
    const int gi = start + i;

    double px = 0.0, py = 0.0, pz = 0.0, e = 0.0;
    if (i < nLocal) {
        const double* p = pos + 3 * (size_t)gi;
        px = p[0];
        py = p[1];
        pz = p[2];
        const double* v = vel + 3 * (size_t)i;
        e = 0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }

    double pe = 0.0;
    for (int tile = 0; tile < n; tile += BLOCK) {
        const int cnt = min(BLOCK, n - tile);
        for (int t = threadIdx.x; t < 3 * cnt; t += BLOCK) sh[t] = pos[3 * (size_t)tile + t];
        __syncthreads();
        if (i < nLocal) {
            for (int j = 0; j < cnt; ++j) {
                if (tile + j == gi) continue;
                const double dx = sh[3 * j + 0] - px;
                const double dy = sh[3 * j + 1] - py;
                const double dz = sh[3 * j + 2] - pz;
                pe -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            }
        }
        __syncthreads();
    }
    e += 0.5 * pe;

    red[threadIdx.x] = e;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) blockSums[blockIdx.x] = red[0];
}

// ----------------------------------------------------------------------------
// Host side
// ----------------------------------------------------------------------------

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

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    const double maxPos = 1e6;
    const double maxVel = 1e6;
    int badFinite = 0, badPos = 0, badVel = 0;
    const long n = (long)bodies.size();

#pragma omp parallel for reduction(| : badFinite, badPos, badVel) schedule(static)
    for (long k = 0; k < n; ++k) {
        const Body& body = bodies[k];
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            badFinite = 1;
        } else {
            if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos)
                badPos = 1;
            if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel)
                badVel = 1;
        }
    }

    if (badFinite) {
        printf("Validation failed: found NaN or Inf value in body state\n");
        return false;
    }
    if (badPos) {
        printf("Validation failed: body position exceeds reasonable bounds\n");
        return false;
    }
    if (badVel) {
        printf("Validation failed: body velocity exceeds reasonable bounds\n");
        return false;
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

// Per-GPU state owned by one OpenMP thread of this rank.
struct GpuSlice {
    int dev = 0;
    int start = 0;   // first global body index
    int nLocal = 0;  // number of bodies on this GPU
    int K = 1;       // number of j-chunks
    int chunkLen = 0;
    cudaStream_t stream = nullptr;
    double* d_pos = nullptr;      // full positions, 3*n
    double* d_vel = nullptr;      // local velocities, 3*nLocal
    double* d_partial = nullptr;  // 3*K*nLocal
};

static inline int divUp(const int a, const int b) { return (a + b - 1) / b; }

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const bool isRoot = (worldRank == 0);

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

    // ---- GPU assignment: ranks sharing a node share its GPUs ----------------
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0, nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_free(&nodeComm);

    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount <= 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> myDevices;
    if (nodeSize >= devCount) {
        myDevices.push_back(nodeRank % devCount);
    } else {
        for (int d = 0; d < devCount; ++d)
            if (d % nodeSize == nodeRank) myDevices.push_back(d);
    }
    const int nGpu = (int)myDevices.size();

    // ---- Body decomposition across ranks ------------------------------------
    const int n = numBodies;
    std::vector<int> rankCounts(worldSize), rankDispls(worldSize);  // in doubles (3 per body)
    for (int r = 0; r < worldSize; ++r) {
        const int b = (int)((long)n * r / worldSize);
        const int e = (int)((long)n * (r + 1) / worldSize);
        rankDispls[r] = 3 * b;
        rankCounts[r] = 3 * (e - b);
    }
    const int rankStart = rankDispls[worldRank] / 3;
    const int rankCount = rankCounts[worldRank] / 3;

    std::vector<GpuSlice> gpus(nGpu);
    for (int g = 0; g < nGpu; ++g) {
        gpus[g].dev = myDevices[g];
        gpus[g].start = rankStart + (int)((long)rankCount * g / nGpu);
        gpus[g].nLocal = rankStart + (int)((long)rankCount * (g + 1) / nGpu) - gpus[g].start;
    }

    // ---- Initialize bodies (identical deterministic sequence on every rank) --
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Pinned host staging buffers: full positions (exchanged each step) and velocities.
    double* h_pos = nullptr;
    double* h_vel = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pos, std::max<size_t>(1, 3 * (size_t)n) * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_vel, std::max<size_t>(1, 3 * (size_t)n) * sizeof(double)));
#pragma omp parallel for schedule(static)
    for (long k = 0; k < (long)n; ++k) {
        h_pos[3 * k + 0] = bodies[k].pos.x;
        h_pos[3 * k + 1] = bodies[k].pos.y;
        h_pos[3 * k + 2] = bodies[k].pos.z;
        h_vel[3 * k + 0] = bodies[k].vel.x;
        h_vel[3 * k + 1] = bodies[k].vel.y;
        h_vel[3 * k + 2] = bodies[k].vel.z;
    }

    const bool needExchange = (worldSize > 1) || (nGpu > 1);
    double localEnergy = 0.0;
    int validationOk = 1;
    std::chrono::high_resolution_clock::time_point start, end;

    // One OpenMP thread per GPU drives that GPU for the whole run; MPI calls are
    // funneled through the master thread.
#pragma omp parallel num_threads(nGpu) reduction(+ : localEnergy)
    {
        const int g = omp_get_thread_num();
        GpuSlice& S = gpus[g];
        CUDA_CHECK(cudaSetDevice(S.dev));
        CUDA_CHECK(cudaStreamCreateWithFlags(&S.stream, cudaStreamNonBlocking));

        // Choose the j-split so that nLocal*K threads can fill the device.
        int numSMs = 1;
        CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, S.dev));
        if (S.nLocal > 0 && n > 0) {
            const long targetThreads = (long)numSMs * 2048;
            int K = (int)std::min<long>(128, std::max<long>(1, divUp((int)std::min<long>(targetThreads, 1L << 30), S.nLocal)));
            S.chunkLen = divUp(divUp(n, K), BLOCK) * BLOCK;
            S.K = divUp(n, S.chunkLen);
        }

        const size_t posBytes = std::max<size_t>(1, 3 * (size_t)n) * sizeof(double);
        const size_t velBytes = std::max<size_t>(1, 3 * (size_t)S.nLocal) * sizeof(double);
        CUDA_CHECK(cudaMalloc(&S.d_pos, posBytes));
        CUDA_CHECK(cudaMalloc(&S.d_vel, velBytes));
        CUDA_CHECK(cudaMalloc(&S.d_partial, std::max<size_t>(1, 3 * (size_t)S.K * S.nLocal) * sizeof(double)));
        CUDA_CHECK(cudaMemcpyAsync(S.d_pos, h_pos, 3 * (size_t)n * sizeof(double), cudaMemcpyHostToDevice, S.stream));
        CUDA_CHECK(cudaMemcpyAsync(S.d_vel, h_vel + 3 * (size_t)S.start, 3 * (size_t)S.nLocal * sizeof(double),
                                   cudaMemcpyHostToDevice, S.stream));
        CUDA_CHECK(cudaStreamSynchronize(S.stream));

        // Force the (lazily loaded) kernel modules onto this device before timing starts.
        {
            cudaFuncAttributes attr;
            CUDA_CHECK(cudaFuncGetAttributes(&attr, forcesKernel));
            CUDA_CHECK(cudaFuncGetAttributes(&attr, updateKernel));
            CUDA_CHECK(cudaFuncGetAttributes(&attr, energyKernel));
        }

        const dim3 forceGrid(std::max(1, divUp(S.nLocal, BLOCK)), std::max(1, S.K));
        const int updateGrid = std::max(1, divUp(S.nLocal, BLOCK));

        // Byte ranges of the full position array before / after this GPU's slice
        // (everything that must be refreshed from the host after the exchange).
        const size_t ownOff = 3 * (size_t)S.start;
        const size_t ownLen = 3 * (size_t)S.nLocal;
        const size_t tailOff = ownOff + ownLen;
        const size_t tailLen = 3 * (size_t)n - tailOff;

#pragma omp barrier
#pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
            start = std::chrono::high_resolution_clock::now();
        }
#pragma omp barrier

        for (int step = 0; step < numSteps; ++step) {
            if (S.nLocal > 0) {
                forcesKernel<<<forceGrid, BLOCK, 0, S.stream>>>(S.d_pos, n, S.start, S.nLocal, S.chunkLen, S.d_partial);
                updateKernel<<<updateGrid, BLOCK, 0, S.stream>>>(S.d_partial, S.K, S.start, S.nLocal, S.d_vel, S.d_pos);
            }
            if (needExchange) {
                if (S.nLocal > 0)
                    CUDA_CHECK(cudaMemcpyAsync(h_pos + ownOff, S.d_pos + ownOff, ownLen * sizeof(double),
                                               cudaMemcpyDeviceToHost, S.stream));
                CUDA_CHECK(cudaStreamSynchronize(S.stream));
#pragma omp barrier
#pragma omp master
                {
                    if (worldSize > 1)
                        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_pos, rankCounts.data(), rankDispls.data(),
                                       MPI_DOUBLE, MPI_COMM_WORLD);
                }
#pragma omp barrier
                if (ownOff > 0)
                    CUDA_CHECK(cudaMemcpyAsync(S.d_pos, h_pos, ownOff * sizeof(double), cudaMemcpyHostToDevice, S.stream));
                if (tailLen > 0)
                    CUDA_CHECK(cudaMemcpyAsync(S.d_pos + tailOff, h_pos + tailOff, tailLen * sizeof(double),
                                               cudaMemcpyHostToDevice, S.stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(S.stream));
        CUDA_CHECK(cudaGetLastError());

#pragma omp barrier
#pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
            end = std::chrono::high_resolution_clock::now();
        }

        // Bring the final local state back to the host.
        if (S.nLocal > 0) {
            CUDA_CHECK(cudaMemcpyAsync(h_pos + ownOff, S.d_pos + ownOff, ownLen * sizeof(double), cudaMemcpyDeviceToHost,
                                       S.stream));
            CUDA_CHECK(cudaMemcpyAsync(h_vel + ownOff, S.d_vel, ownLen * sizeof(double), cudaMemcpyDeviceToHost, S.stream));
            CUDA_CHECK(cudaStreamSynchronize(S.stream));
        }
    }

    // ---- Collect the final state on rank 0 ----------------------------------
    if (worldSize > 1) {
        if (isRoot) {
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_pos, rankCounts.data(), rankDispls.data(), MPI_DOUBLE, 0,
                        MPI_COMM_WORLD);
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_vel, rankCounts.data(), rankDispls.data(), MPI_DOUBLE, 0,
                        MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(h_pos + rankDispls[worldRank], rankCounts[worldRank], MPI_DOUBLE, nullptr, nullptr, nullptr,
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(h_vel + rankDispls[worldRank], rankCounts[worldRank], MPI_DOUBLE, nullptr, nullptr, nullptr,
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    long localTimeMs = (long)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long maxTimeMs = 0;
    MPI_Reduce(&localTimeMs, &maxTimeMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (isRoot) {
        printf("Simulation time: %ld ms\n", maxTimeMs);

#pragma omp parallel for schedule(static)
        for (long k = 0; k < (long)n; ++k) {
            bodies[k].pos.x = h_pos[3 * k + 0];
            bodies[k].pos.y = h_pos[3 * k + 1];
            bodies[k].pos.z = h_pos[3 * k + 2];
            bodies[k].vel.x = h_vel[3 * k + 0];
            bodies[k].vel.y = h_vel[3 * k + 1];
            bodies[k].vel.z = h_vel[3 * k + 2];
        }

        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData(6 * (size_t)n);
#pragma omp parallel for schedule(static)
            for (long k = 0; k < (long)n; ++k) {
                bodyData[6 * k + 0] = bodies[k].pos.x;
                bodyData[6 * k + 1] = bodies[k].pos.y;
                bodyData[6 * k + 2] = bodies[k].pos.z;
                bodyData[6 * k + 3] = bodies[k].vel.x;
                bodyData[6 * k + 4] = bodies[k].vel.y;
                bodyData[6 * k + 5] = bodies[k].vel.z;
            }
            print_results(bodyData, "Bodies");
        }

        // Validation: check that simulation produces finite, reasonable values
        if (validate) {
            printf("Validating simulation results...\n");
            validationOk = validateSimulation(bodies) ? 1 : 0;
        }
    }
    MPI_Bcast(&validationOk, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rc = 0;
    if (validate) {
        if (validationOk) {
            // Energy of the local bodies, evaluated on the GPUs, then reduced across ranks.
#pragma omp parallel num_threads(nGpu) reduction(+ : localEnergy)
            {
                GpuSlice& S = gpus[omp_get_thread_num()];
                CUDA_CHECK(cudaSetDevice(S.dev));
                if (S.nLocal > 0) {
                    const int blocks = divUp(S.nLocal, BLOCK);
                    double* d_sums = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_sums, blocks * sizeof(double)));
                    energyKernel<<<blocks, BLOCK, 0, S.stream>>>(S.d_pos, S.d_vel, n, S.start, S.nLocal, d_sums);
                    std::vector<double> sums(blocks);
                    CUDA_CHECK(cudaMemcpyAsync(sums.data(), d_sums, blocks * sizeof(double), cudaMemcpyDeviceToHost,
                                               S.stream));
                    CUDA_CHECK(cudaStreamSynchronize(S.stream));
                    CUDA_CHECK(cudaGetLastError());
                    for (int b = 0; b < blocks; ++b) localEnergy += sums[b];
                    CUDA_CHECK(cudaFree(d_sums));
                }
            }
            double finalEnergy = 0.0;
            MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (isRoot) {
                // Report final energy for reference
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (isRoot) printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    // ---- Cleanup -------------------------------------------------------------
    for (GpuSlice& S : gpus) {
        CUDA_CHECK(cudaSetDevice(S.dev));
        CUDA_CHECK(cudaFree(S.d_partial));
        CUDA_CHECK(cudaFree(S.d_vel));
        CUDA_CHECK(cudaFree(S.d_pos));
        CUDA_CHECK(cudaStreamDestroy(S.stream));
    }
    CUDA_CHECK(cudaFreeHost(h_vel));
    CUDA_CHECK(cudaFreeHost(h_pos));
    MPI_Finalize();
    return rc;
}
