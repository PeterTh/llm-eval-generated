// N-body benchmark: hybrid MPI + OpenMP + CUDA implementation.
//
// Decomposition:
//   * MPI  : bodies are partitioned into contiguous slices, one per rank. Each
//            step every rank computes forces/velocities/positions for its own
//            slice against ALL bodies, then the new positions are exchanged with
//            a single MPI_Allgatherv.
//   * CUDA : the O(N^2) force evaluation and the integration run on the GPU in a
//            shared-memory tiled kernel. Every rank drives all GPUs that are
//            assigned to it (GPUs on a node are divided round-robin among the
//            ranks on that node), so a single rank on a 4-GPU node uses 4 GPUs.
//   * OpenMP: one host thread per GPU drives the devices concurrently; the host
//            side O(N^2) energy check, data packing and layout conversion are
//            OpenMP parallel loops.
//
// The force summation keeps the sequential j = 0..N-1 order of the original
// code per body so results are numerically equivalent to the serial version.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// ----------------------------------------------------------------------------
// Error handling helpers
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t err__ = (call);                                                             \
        if (err__ != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,          \
                    cudaGetErrorString(err__));                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

#define MPI_CHECK(call)                                                                         \
    do {                                                                                        \
        int err__ = (call);                                                                     \
        if (err__ != MPI_SUCCESS) {                                                             \
            fprintf(stderr, "MPI error %d in %s at %s:%d\n", err__, #call, __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

// ----------------------------------------------------------------------------
// CUDA kernels
// ----------------------------------------------------------------------------
constexpr int TILE = 256;  // bodies per shared-memory tile

// One thread per owned body. Positions of all N bodies are streamed through
// shared memory in tiles; the j loop visits bodies in increasing index order
// exactly like the serial reference. Velocity update and position integration
// are fused: the new position is written to a separate buffer so that posAll
// (read by every thread) is never modified during the kernel.
//
// BLOCK is chosen at launch time (32..256) so that even small per-GPU slices
// spread over all SMs; the per-body arithmetic is identical for any BLOCK.
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK) nbodyStepKernel(const double4* __restrict__ posAll,
                                                         double4* __restrict__ vel,
                                                         double4* __restrict__ posNew,
                                                         const int n, const int begin,
                                                         const int count) {
    __shared__ double4 tile[TILE];

    const int li = blockIdx.x * BLOCK + threadIdx.x;
    const bool active = li < count;

    double4 p = make_double4(0.0, 0.0, 0.0, 0.0);
    if (active) p = posAll[begin + li];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += TILE) {
        __syncthreads();
        for (int t = threadIdx.x; t < TILE; t += BLOCK) {
            const int jj = base + t;
            tile[t] = (jj < n) ? posAll[jj] : make_double4(0.0, 0.0, 0.0, 0.0);
        }
        __syncthreads();

        const int limit = min(TILE, n - base);
        if (active) {
#pragma unroll 8
            for (int j = 0; j < limit; ++j) {
                const double4 q = tile[j];
                const double dx = q.x - p.x;
                const double dy = q.y - p.y;
                const double dz = q.z - p.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
    }

    if (active) {
        double4 v = vel[li];
        v.x += DT * Fx;
        v.y += DT * Fy;
        v.z += DT * Fz;
        vel[li] = v;
        p.x += v.x * DT;
        p.y += v.y * DT;
        p.z += v.z * DT;
        posNew[li] = p;
    }
}

// Launch helper: use the largest block size for which the slice still
// occupies every SM of the device (FP64 throughput per SM is the limiter).
static void launchStep(const double4* posAll, double4* vel, double4* posNew, const int n, const int begin,
                       const int count, const int numSMs, cudaStream_t stream) {
    auto blocksFor = [&](int b) { return std::max(1, (count + b - 1) / b); };
    if (blocksFor(256) >= numSMs) {
        nbodyStepKernel<256><<<blocksFor(256), 256, 0, stream>>>(posAll, vel, posNew, n, begin, count);
    } else if (blocksFor(128) >= numSMs) {
        nbodyStepKernel<128><<<blocksFor(128), 128, 0, stream>>>(posAll, vel, posNew, n, begin, count);
    } else if (blocksFor(64) >= numSMs) {
        nbodyStepKernel<64><<<blocksFor(64), 64, 0, stream>>>(posAll, vel, posNew, n, begin, count);
    } else {
        nbodyStepKernel<32><<<blocksFor(32), 32, 0, stream>>>(posAll, vel, posNew, n, begin, count);
    }
}

// ----------------------------------------------------------------------------
// Host code
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

// Per-GPU state owned by one MPI rank.
struct DeviceSlice {
    int device = 0;
    int begin = 0;   // global index of first body handled by this GPU
    int count = 0;   // number of bodies handled by this GPU
    double4* d_posAll = nullptr;
    double4* d_vel = nullptr;
    double4* d_posNew = nullptr;
    int numSMs = 1;
    cudaStream_t stream = nullptr;
};

// Distributed total energy: kinetic energy from the locally owned velocities,
// potential energy over all pairs (i, j>i) with i in the local slice. Each rank
// uses OpenMP for its share; the result is reduced to rank 0.
double computeTotalEnergyDistributed(const std::vector<double4>& posAll, const std::vector<double4>& velLocal,
                                     const int begin, const int count) {
    const int n = static_cast<int>(posAll.size());
    const double4* pos = posAll.data();
    double energy = 0.0;

    // Kinetic energy (assuming unit mass)
#pragma omp parallel for reduction(+ : energy) schedule(static)
    for (int i = 0; i < count; ++i) {
        const double4 v = velLocal[i];
        energy += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }

    // Potential energy (assuming unit mass for all bodies)
#pragma omp parallel for reduction(+ : energy) schedule(dynamic, 16)
    for (int li = 0; li < count; ++li) {
        const int i = begin + li;
        const double4 pi = pos[i];
        double e = 0.0;
        for (int j = i + 1; j < n; ++j) {
            const double dx = pos[j].x - pi.x;
            const double dy = pos[j].y - pi.y;
            const double dz = pos[j].z - pi.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            e -= 1.0 / dist;
        }
        energy += e;
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

    int rank = 0, size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;

    // Parse command line arguments (identical on every rank)
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

    // ---------------------------------------------------------------------
    // GPU assignment: GPUs on each node are shared round-robin among the
    // ranks running on that node. A rank may own several GPUs.
    // ---------------------------------------------------------------------
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int nodeRank = 0, nodeSize = 1;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &nodeRank));
    MPI_CHECK(MPI_Comm_size(nodeComm, &nodeSize));
    MPI_CHECK(MPI_Comm_free(&nodeComm));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> myDevices;
    for (int d = nodeRank; d < deviceCount; d += nodeSize) myDevices.push_back(d);
    if (myDevices.empty()) myDevices.push_back(nodeRank % deviceCount);  // more ranks than GPUs: share
    const int numDevices = static_cast<int>(myDevices.size());

    // ---------------------------------------------------------------------
    // Global work decomposition: contiguous slice per rank, sub-slices per GPU.
    // ---------------------------------------------------------------------
    std::vector<int> rankCounts(size), rankDispls(size);
    for (int r = 0; r < size; ++r) {
        rankCounts[r] = numBodies / size + (r < numBodies % size ? 1 : 0);
        rankDispls[r] = (r == 0) ? 0 : rankDispls[r - 1] + rankCounts[r - 1];
    }
    const int localBegin = rankDispls[rank];
    const int localCount = rankCounts[rank];

    std::vector<DeviceSlice> slices(numDevices);
    {
        int off = localBegin;
        for (int d = 0; d < numDevices; ++d) {
            slices[d].device = myDevices[d];
            slices[d].begin = off;
            slices[d].count = localCount / numDevices + (d < localCount % numDevices ? 1 : 0);
            off += slices[d].count;
        }
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---------------------------------------------------------------------
    // Initialize bodies. The RNG stream is sequential, so every rank
    // regenerates the full (identical) initial state and keeps its slice.
    // ---------------------------------------------------------------------
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Host buffers: pinned for fast and asynchronous host<->device transfers.
    double4* h_posAll = nullptr;
    double4* h_posLocal = nullptr;
    double4* h_velLocal = nullptr;
    CUDA_CHECK(cudaSetDevice(myDevices[0]));
    CUDA_CHECK(cudaHostAlloc(&h_posAll, sizeof(double4) * std::max(numBodies, 1), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_posLocal, sizeof(double4) * std::max(localCount, 1), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_velLocal, sizeof(double4) * std::max(localCount, 1), cudaHostAllocPortable));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        h_posAll[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < localCount; ++i) {
        const Body& b = bodies[localBegin + i];
        h_velLocal[i] = make_double4(b.vel.x, b.vel.y, b.vel.z, 0.0);
    }

    // Allgatherv bookkeeping in units of doubles (4 per body).
    std::vector<int> gCounts(size), gDispls(size);
    for (int r = 0; r < size; ++r) {
        gCounts[r] = rankCounts[r] * 4;
        gDispls[r] = rankDispls[r] * 4;
    }

    // Device setup: one OpenMP thread per GPU.
#pragma omp parallel num_threads(numDevices)
    {
        DeviceSlice& s = slices[omp_get_thread_num()];
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaDeviceGetAttribute(&s.numSMs, cudaDevAttrMultiProcessorCount, s.device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&s.d_posAll, sizeof(double4) * std::max(numBodies, 1)));
        CUDA_CHECK(cudaMalloc(&s.d_vel, sizeof(double4) * std::max(s.count, 1)));
        CUDA_CHECK(cudaMalloc(&s.d_posNew, sizeof(double4) * std::max(s.count, 1)));
        CUDA_CHECK(cudaMemcpyAsync(s.d_posAll, h_posAll, sizeof(double4) * numBodies, cudaMemcpyHostToDevice,
                                   s.stream));
        if (s.count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(s.d_vel, h_velLocal + (s.begin - localBegin), sizeof(double4) * s.count,
                                       cudaMemcpyHostToDevice, s.stream));
        }
        // Warm-up: a zero-work launch forces module loading / context setup
        // on every device so that it is not attributed to the timed region.
        launchStep(s.d_posAll, s.d_vel, s.d_posNew, 0, 0, 0, s.numSMs, s.stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
    }

    // ---------------------------------------------------------------------
    // Run simulation
    // ---------------------------------------------------------------------
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Each GPU: forces + integration for its sub-slice, then copy the new
        // positions of the sub-slice back into the rank's local host buffer.
#pragma omp parallel num_threads(numDevices)
        {
            DeviceSlice& s = slices[omp_get_thread_num()];
            CUDA_CHECK(cudaSetDevice(s.device));
            if (s.count > 0) {
                launchStep(s.d_posAll, s.d_vel, s.d_posNew, numBodies, s.begin, s.count, s.numSMs, s.stream);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(h_posLocal + (s.begin - localBegin), s.d_posNew,
                                           sizeof(double4) * s.count, cudaMemcpyDeviceToHost, s.stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(s.stream));
        }

        // Exchange the updated positions among all ranks.
        MPI_CHECK(MPI_Allgatherv(h_posLocal, localCount * 4, MPI_DOUBLE, h_posAll, gCounts.data(),
                                 gDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD));

        // Refresh the full position array on every GPU (positions of the
        // bodies owned by this GPU are included so no separate update needed).
#pragma omp parallel num_threads(numDevices)
        {
            DeviceSlice& s = slices[omp_get_thread_num()];
            CUDA_CHECK(cudaSetDevice(s.device));
            CUDA_CHECK(cudaMemcpyAsync(s.d_posAll, h_posAll, sizeof(double4) * numBodies, cudaMemcpyHostToDevice,
                                       s.stream));
            CUDA_CHECK(cudaStreamSynchronize(s.stream));
        }
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long local_duration_ms = duration.count();
    long max_duration_ms = 0;
    MPI_CHECK(MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) printf("Simulation time: %ld ms\n", max_duration_ms);

    // ---------------------------------------------------------------------
    // Collect final state: velocities back to host, gather everything on rank 0.
    // ---------------------------------------------------------------------
#pragma omp parallel num_threads(numDevices)
    {
        DeviceSlice& s = slices[omp_get_thread_num()];
        CUDA_CHECK(cudaSetDevice(s.device));
        if (s.count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(h_velLocal + (s.begin - localBegin), s.d_vel, sizeof(double4) * s.count,
                                       cudaMemcpyDeviceToHost, s.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
    }

    std::vector<double4> velAll(rank == 0 ? numBodies : 0);
    MPI_CHECK(MPI_Gatherv(h_velLocal, localCount * 4, MPI_DOUBLE, rank == 0 ? velAll.data() : nullptr,
                          gCounts.data(), gDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));

    if (rank == 0) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos = Vec3(h_posAll[i].x, h_posAll[i].y, h_posAll[i].z);
            bodies[i].vel = Vec3(velAll[i].x, velAll[i].y, velAll[i].z);
        }
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const Body& body = bodies[i];
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

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        int ok = 1;
        if (rank == 0) {
            printf("Validating simulation results...\n");
            ok = validateSimulation(bodies) ? 1 : 0;
        }
        MPI_CHECK(MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD));

        if (ok) {
            // Report final energy for reference (distributed computation)
            std::vector<double4> posAll(h_posAll, h_posAll + numBodies);
            std::vector<double4> velLocal(h_velLocal, h_velLocal + localCount);
            double finalEnergy = computeTotalEnergyDistributed(posAll, velLocal, localBegin, localCount);
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
#pragma omp parallel num_threads(numDevices)
    {
        DeviceSlice& s = slices[omp_get_thread_num()];
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaFree(s.d_posAll));
        CUDA_CHECK(cudaFree(s.d_vel));
        CUDA_CHECK(cudaFree(s.d_posNew));
        CUDA_CHECK(cudaStreamDestroy(s.stream));
    }
    CUDA_CHECK(cudaFreeHost(h_posAll));
    CUDA_CHECK(cudaFreeHost(h_posLocal));
    CUDA_CHECK(cudaFreeHost(h_velLocal));

    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
