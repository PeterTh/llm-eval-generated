// Hybrid MPI + OpenMP + CUDA N-body benchmark.
//
// Decomposition:
//   * MPI     : bodies are split into contiguous slices, one per rank. Every
//               rank keeps a full copy of all positions; after each step the
//               updated positions of the local slice are exchanged with
//               MPI_Allgatherv.
//   * CUDA    : the O(n^2) force evaluation (and the O(n^2) energy check) run
//               on the GPUs. The kernel keeps the original j = 0..n-1
//               accumulation order per body, so results match the serial code.
//   * OpenMP  : one host thread per GPU owned by a rank (ranks with several
//               GPUs drive them concurrently), plus host-side data packing,
//               reductions and validation.
//
// This file is compiled as CUDA (see CMakeLists.txt).

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
#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t err__ = (call);                                                           \
        if (err__ != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,        \
                    cudaGetErrorString(err__));                                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                     \
        }                                                                                     \
    } while (0)

#define MPI_CHECK(call)                                                                       \
    do {                                                                                      \
        int err__ = (call);                                                                   \
        if (err__ != MPI_SUCCESS) {                                                           \
            fprintf(stderr, "MPI error %s at %s:%d (code %d)\n", #call, __FILE__, __LINE__,   \
                    err__);                                                                   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                     \
        }                                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------
constexpr int TILE = 256;  // threads per block == bodies per shared-memory tile

// Computes the force on bodies [first, first + count) against all n bodies,
// updates their velocities and writes their integrated positions to newPos.
// The j loop runs sequentially over 0..n-1 (tile by tile) so that the floating
// point accumulation order is identical to the serial reference.
__global__ void __launch_bounds__(TILE)
forceIntegrateKernel(const double4* __restrict__ pos, double4* __restrict__ newPos,
                     double4* __restrict__ vel, const int n, const int first, const int count) {
    __shared__ double4 tilePos[TILE];

    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < count;
    const int i = first + local;

    double px = 0.0, py = 0.0, pz = 0.0;
    if (active) {
        const double4 p = pos[i];
        px = p.x; py = p.y; pz = p.z;
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    const int numTiles = (n + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        const int base = t * TILE;
        const int jLoad = base + threadIdx.x;
        if (jLoad < n) {
            tilePos[threadIdx.x] = pos[jLoad];
        }
        __syncthreads();

        const int tileCount = min(TILE, n - base);
        if (active) {
            if (tileCount == TILE) {
#pragma unroll 8
                for (int j = 0; j < TILE; ++j) {
                    const double4 q = tilePos[j];
                    const double dx = q.x - px;
                    const double dy = q.y - py;
                    const double dz = q.z - pz;
                    const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    const double invDist = 1.0 / sqrt(distSqr);
                    const double invDist3 = invDist * invDist * invDist;
                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
            } else {
                for (int j = 0; j < tileCount; ++j) {
                    const double4 q = tilePos[j];
                    const double dx = q.x - px;
                    const double dy = q.y - py;
                    const double dz = q.z - pz;
                    const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    const double invDist = 1.0 / sqrt(distSqr);
                    const double invDist3 = invDist * invDist * invDist;
                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        double4 v = vel[local];
        v.x += DT * Fx;
        v.y += DT * Fy;
        v.z += DT * Fz;
        vel[local] = v;

        double4 p;
        p.x = px + v.x * DT;
        p.y = py + v.y * DT;
        p.z = pz + v.z * DT;
        p.w = 0.0;
        newPos[local] = p;
    }
}

// Per-block partial energies: kinetic energy of local bodies plus the
// potential energy of pairs (i, j) with j > i for i in [first, first + count).
__global__ void __launch_bounds__(TILE)
energyKernel(const double4* __restrict__ pos, const double4* __restrict__ vel, const int n,
             const int first, const int count, double* __restrict__ partial) {
    __shared__ double4 tilePos[TILE];
    __shared__ double red[TILE];

    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < count;
    const int i = first + local;

    double energy = 0.0;
    double px = 0.0, py = 0.0, pz = 0.0;
    if (active) {
        const double4 p = pos[i];
        px = p.x; py = p.y; pz = p.z;
        const double4 v = vel[local];
        energy = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }

    // Only tiles that may contain j > i for some i of this block are needed.
    const int blockFirstI = first + blockIdx.x * blockDim.x;
    const int startTile = (blockFirstI + 1) / TILE;
    const int numTiles = (n + TILE - 1) / TILE;
    for (int t = startTile; t < numTiles; ++t) {
        const int base = t * TILE;
        const int jLoad = base + threadIdx.x;
        if (jLoad < n) {
            tilePos[threadIdx.x] = pos[jLoad];
        }
        __syncthreads();

        const int tileCount = min(TILE, n - base);
        if (active) {
            for (int j = 0; j < tileCount; ++j) {
                if (base + j > i) {
                    const double4 q = tilePos[j];
                    const double dx = q.x - px;
                    const double dy = q.y - py;
                    const double dz = q.z - pz;
                    const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                    energy -= 1.0 / dist;
                }
            }
        }
        __syncthreads();
    }

    red[threadIdx.x] = energy;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = red[0];
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Sequential by construction: rand_r is a serial recurrence.
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// State for one GPU owned by this rank.
struct GpuSlice {
    int device = 0;
    int first = 0;   // global index of first body handled by this GPU
    int count = 0;   // number of bodies handled by this GPU
    cudaStream_t stream = nullptr;
    double4* d_pos = nullptr;      // all n positions
    double4* d_newPos = nullptr;   // count integrated positions
    double4* d_vel = nullptr;      // count velocities
    double* d_partial = nullptr;   // energy partial sums (one per block)
    int blocks = 0;
};

static int divUp(const int a, const int b) { return (a + b - 1) / b; }

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    // 0 = ok, 1 = NaN/Inf, 2 = position bound, 3 = velocity bound.
    // The first failing body (lowest index) determines the message, as in the
    // serial version.
    const long n = static_cast<long>(bodies.size());
    long firstBad = n;
    int reason = 0;
#pragma omp parallel
    {
        long myBad = n;
        int myReason = 0;
#pragma omp for schedule(static) nowait
        for (long k = 0; k < n; ++k) {
            if (myBad != n) continue;
            const auto& body = bodies[k];
            const double maxPos = 1e6;
            const double maxVel = 1e6;
            int r = 0;
            if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
                !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
                r = 1;
            } else if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos ||
                       std::abs(body.pos.z) > maxPos) {
                r = 2;
            } else if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel ||
                       std::abs(body.vel.z) > maxVel) {
                r = 3;
            }
            if (r != 0) {
                myBad = k;
                myReason = r;
            }
        }
#pragma omp critical
        {
            if (myBad < firstBad) {
                firstBad = myBad;
                reason = myReason;
            }
        }
    }
    switch (reason) {
        case 1: printf("Validation failed: found NaN or Inf value in body state\n"); return false;
        case 2: printf("Validation failed: body position exceeds reasonable bounds\n"); return false;
        case 3: printf("Validation failed: body velocity exceeds reasonable bounds\n"); return false;
        default: return true;
    }
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
    int rank = 0, size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    bool exitEarly = false;

    // Parse command line arguments (all ranks see the same argv)
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
            exitEarly = true;
            exitCode = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitEarly = true;
            exitCode = 1;
            break;
        }
    }
    if (exitEarly) {
        MPI_Finalize();
        return exitCode;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ------------------------------------------------------------------
    // GPU assignment: ranks sharing a node split the node's GPUs evenly.
    // ------------------------------------------------------------------
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int nodeRank = 0, nodeSize = 1;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &nodeRank));
    MPI_CHECK(MPI_Comm_size(nodeComm, &nodeSize));
    MPI_CHECK(MPI_Comm_free(&nodeComm));

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices <= 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<int> myDevices;
    if (nodeSize >= numDevices) {
        myDevices.push_back(nodeRank % numDevices);
    } else {
        // Distribute devices [0, numDevices) over nodeSize ranks as evenly as possible.
        const int base = numDevices / nodeSize;
        const int rem = numDevices % nodeSize;
        const int begin = nodeRank * base + std::min(nodeRank, rem);
        const int cnt = base + (nodeRank < rem ? 1 : 0);
        for (int d = begin; d < begin + cnt; ++d) myDevices.push_back(d);
    }
    const int numGpus = static_cast<int>(myDevices.size());

    // ------------------------------------------------------------------
    // Body decomposition across ranks (contiguous, balanced slices).
    // ------------------------------------------------------------------
    std::vector<int> rankCounts(size), rankDispls(size);  // in bodies
    for (int r = 0; r < size; ++r) {
        const int base = numBodies / size;
        const int rem = numBodies % size;
        rankCounts[r] = base + (r < rem ? 1 : 0);
        rankDispls[r] = r * base + std::min(r, rem);
    }
    const int myFirst = rankDispls[rank];
    const int myCount = rankCounts[rank];

    // Allgatherv counts/displacements in doubles (4 per body: double4).
    std::vector<int> gCounts(size), gDispls(size);
    for (int r = 0; r < size; ++r) {
        gCounts[r] = rankCounts[r] * 4;
        gDispls[r] = rankDispls[r] * 4;
    }

    // Initialize bodies (identical sequence on every rank, same as serial).
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Host position buffer for all bodies (pinned for fast H2D/D2H).
    double4* h_pos = nullptr;
    double4* h_vel = nullptr;  // local velocities (for final gather)
    CUDA_CHECK(cudaHostAlloc(&h_pos, sizeof(double4) * std::max(numBodies, 1), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_vel, sizeof(double4) * std::max(myCount, 1), cudaHostAllocPortable));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        h_pos[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < myCount; ++i) {
        const Body& b = bodies[myFirst + i];
        h_vel[i] = make_double4(b.vel.x, b.vel.y, b.vel.z, 0.0);
    }

    // ------------------------------------------------------------------
    // Per-GPU setup (split local slice over local GPUs).
    // ------------------------------------------------------------------
    std::vector<GpuSlice> gpus(numGpus);
    for (int g = 0; g < numGpus; ++g) {
        const int base = myCount / numGpus;
        const int rem = myCount % numGpus;
        gpus[g].device = myDevices[g];
        gpus[g].count = base + (g < rem ? 1 : 0);
        gpus[g].first = myFirst + g * base + std::min(g, rem);
        gpus[g].blocks = divUp(gpus[g].count, TILE);
    }

#pragma omp parallel for num_threads(numGpus) schedule(static)
    for (int g = 0; g < numGpus; ++g) {
        GpuSlice& s = gpus[g];
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&s.d_pos, sizeof(double4) * std::max(numBodies, 1)));
        CUDA_CHECK(cudaMalloc(&s.d_newPos, sizeof(double4) * std::max(s.count, 1)));
        CUDA_CHECK(cudaMalloc(&s.d_vel, sizeof(double4) * std::max(s.count, 1)));
        CUDA_CHECK(cudaMalloc(&s.d_partial, sizeof(double) * std::max(s.blocks, 1)));
        CUDA_CHECK(cudaMemcpyAsync(s.d_pos, h_pos, sizeof(double4) * numBodies, cudaMemcpyHostToDevice, s.stream));
        if (s.count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(s.d_vel, h_vel + (s.first - myFirst), sizeof(double4) * s.count,
                                       cudaMemcpyHostToDevice, s.stream));
        }
        // Warm-up launch (no active bodies) so module loading and context
        // initialisation happen outside the timed region.
        forceIntegrateKernel<<<1, TILE, 0, s.stream>>>(s.d_pos, s.d_newPos, s.d_vel, numBodies, s.first, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
    }

    // ------------------------------------------------------------------
    // Run simulation
    // ------------------------------------------------------------------
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Forces + integration of the local slice on each local GPU; new
        // positions are copied back into the local part of h_pos.
#pragma omp parallel for num_threads(numGpus) schedule(static)
        for (int g = 0; g < numGpus; ++g) {
            GpuSlice& s = gpus[g];
            if (s.count == 0) continue;
            CUDA_CHECK(cudaSetDevice(s.device));
            forceIntegrateKernel<<<s.blocks, TILE, 0, s.stream>>>(s.d_pos, s.d_newPos, s.d_vel, numBodies,
                                                                  s.first, s.count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(h_pos + s.first, s.d_newPos, sizeof(double4) * s.count,
                                       cudaMemcpyDeviceToHost, s.stream));
            CUDA_CHECK(cudaStreamSynchronize(s.stream));
        }

        // Exchange updated positions between ranks.
        if (size > 1) {
            MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_pos, gCounts.data(), gDispls.data(),
                                     MPI_DOUBLE, MPI_COMM_WORLD));
        }

        // Upload the full position array to every local GPU. With a single
        // GPU per rank and a single rank only the local slice is already
        // in place on the device, so copy only what changed elsewhere.
#pragma omp parallel for num_threads(numGpus) schedule(static)
        for (int g = 0; g < numGpus; ++g) {
            GpuSlice& s = gpus[g];
            CUDA_CHECK(cudaSetDevice(s.device));
            if (numGpus == 1 && size == 1) {
                if (s.count > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(s.d_pos + s.first, s.d_newPos, sizeof(double4) * s.count,
                                               cudaMemcpyDeviceToDevice, s.stream));
                }
            } else {
                CUDA_CHECK(cudaMemcpyAsync(s.d_pos, h_pos, sizeof(double4) * numBodies, cudaMemcpyHostToDevice,
                                           s.stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(s.stream));
        }
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long elapsed_ms = duration.count();
    long max_elapsed_ms = 0;
    MPI_CHECK(MPI_Reduce(&elapsed_ms, &max_elapsed_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", max_elapsed_ms);
    }

    // ------------------------------------------------------------------
    // Gather final state on rank 0 (positions are already replicated).
    // ------------------------------------------------------------------
#pragma omp parallel for num_threads(numGpus) schedule(static)
    for (int g = 0; g < numGpus; ++g) {
        GpuSlice& s = gpus[g];
        if (s.count == 0) continue;
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaMemcpyAsync(h_vel + (s.first - myFirst), s.d_vel, sizeof(double4) * s.count,
                                   cudaMemcpyDeviceToHost, s.stream));
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
    }

    double4* h_velAll = nullptr;
    if (rank == 0) {
        CUDA_CHECK(cudaHostAlloc(&h_velAll, sizeof(double4) * std::max(numBodies, 1), cudaHostAllocPortable));
    }
    if (size > 1) {
        MPI_CHECK(MPI_Gatherv(h_vel, myCount * 4, MPI_DOUBLE, h_velAll, gCounts.data(), gDispls.data(), MPI_DOUBLE,
                              0, MPI_COMM_WORLD));
    } else {
        memcpy(h_velAll, h_vel, sizeof(double4) * numBodies);
    }

    if (rank == 0) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos = Vec3(h_pos[i].x, h_pos[i].y, h_pos[i].z);
            bodies[i].vel = Vec3(h_velAll[i].x, h_velAll[i].y, h_velAll[i].z);
        }
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const Body& body = bodies[i];
            double* d = bodyData.data() + static_cast<size_t>(i) * 6;
            d[0] = body.pos.x;
            d[1] = body.pos.y;
            d[2] = body.pos.z;
            d[3] = body.vel.x;
            d[4] = body.vel.y;
            d[5] = body.vel.z;
        }
        print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        int ok = 1;
        if (rank == 0) ok = validateSimulation(bodies) ? 1 : 0;
        MPI_CHECK(MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD));

        if (ok) {
            // Total energy: local GPUs compute their rows, MPI reduces.
            double localEnergy = 0.0;
#pragma omp parallel for num_threads(numGpus) schedule(static) reduction(+ : localEnergy)
            for (int g = 0; g < numGpus; ++g) {
                GpuSlice& s = gpus[g];
                if (s.count == 0) continue;
                CUDA_CHECK(cudaSetDevice(s.device));
                energyKernel<<<s.blocks, TILE, 0, s.stream>>>(s.d_pos, s.d_vel, numBodies, s.first, s.count,
                                                              s.d_partial);
                CUDA_CHECK(cudaGetLastError());
                std::vector<double> partial(s.blocks);
                CUDA_CHECK(cudaMemcpyAsync(partial.data(), s.d_partial, sizeof(double) * s.blocks,
                                           cudaMemcpyDeviceToHost, s.stream));
                CUDA_CHECK(cudaStreamSynchronize(s.stream));
                localEnergy += kahan_sum(partial);
            }
            double finalEnergy = 0.0;
            MPI_CHECK(MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            exitCode = 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    // Cleanup
    for (int g = 0; g < numGpus; ++g) {
        GpuSlice& s = gpus[g];
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaFree(s.d_pos));
        CUDA_CHECK(cudaFree(s.d_newPos));
        CUDA_CHECK(cudaFree(s.d_vel));
        CUDA_CHECK(cudaFree(s.d_partial));
        CUDA_CHECK(cudaStreamDestroy(s.stream));
    }
    CUDA_CHECK(cudaFreeHost(h_pos));
    CUDA_CHECK(cudaFreeHost(h_vel));
    if (h_velAll) CUDA_CHECK(cudaFreeHost(h_velAll));

    MPI_Finalize();
    return exitCode;
}
