// N-body benchmark: hybrid MPI + OpenMP + CUDA implementation.
//
// Decomposition:
//   * MPI    : bodies are partitioned into contiguous slices, one per rank.
//              Every rank owns the velocities of its slice and keeps a full
//              replicated copy of all positions (refreshed each step with an
//              MPI_Allgatherv of the local position slices).
//   * CUDA   : each rank drives one GPU (selected by node-local rank). The
//              all-pairs force computation for the local slice against all
//              bodies and the local integration run on the device.
//   * OpenMP : host-side work (result packing, validation, energy) is
//              multi-threaded within each rank.
//
// Numerics: the inner j-loop runs in the same order as the serial code and
// uses the same FMA contraction pattern GCC emits for the original source,
// with automatic contraction disabled (-fmad=false), so the simulation
// produces results equivalent to the original program.

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

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t err__ = (call);                                                             \
        if (err__ != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,          \
                    cudaGetErrorString(err__));                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Force computation for the local slice [offset, offset + nLocal) of bodies
// against all n bodies. One thread per local body; source positions are
// staged through shared memory in tiles, iterated in ascending j order so the
// accumulation order matches the serial reference exactly.
__global__ void __launch_bounds__(256)
computeForcesKernel(const double3* __restrict__ pos, double3* __restrict__ vel,
                    const int n, const int offset, const int nLocal) {
    extern __shared__ double3 tile[];

    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = li < nLocal;

    double3 p = make_double3(0.0, 0.0, 0.0);
    if (active) p = pos[offset + li];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += blockDim.x) {
        const int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = pos[j];
        __syncthreads();

        const int cnt = min((int)blockDim.x, n - base);
#pragma unroll 8
        for (int k = 0; k < cnt; ++k) {
            const double3 q = tile[k];
            const double dx = q.x - p.x;
            const double dy = q.y - p.y;
            const double dz = q.z - p.z;
            // Same contraction pattern as the serial build:
            // distSqr = fma(dx,dx, dy*dy); distSqr = fma(dz,dz,distSqr); + SOFTENING
            double distSqr = fma(dx, dx, dy * dy);
            distSqr = fma(dz, dz, distSqr);
            distSqr = distSqr + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = (invDist * invDist) * invDist;
            Fx = fma(dx, invDist3, Fx);
            Fy = fma(dy, invDist3, Fy);
            Fz = fma(dz, invDist3, Fz);
        }
        __syncthreads();
    }

    if (active) {
        double3 v = vel[li];
        v.x = v.x + DT * Fx;
        v.y = v.y + DT * Fy;
        v.z = v.z + DT * Fz;
        vel[li] = v;
    }
}

__global__ void integrateBodiesKernel(double3* __restrict__ pos, const double3* __restrict__ vel,
                                      const int offset, const int nLocal) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li < nLocal) {
        double3 p = pos[offset + li];
        const double3 v = vel[li];
        p.x = fma(v.x, DT, p.x);
        p.y = fma(v.y, DT, p.y);
        p.z = fma(v.z, DT, p.z);
        pos[offset + li] = p;
    }
}

// ---------------------------------------------------------------------------
// Host code
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

// Total energy of the full system. Work is distributed cyclically over MPI
// ranks (balances the triangular potential loop) and over OpenMP threads
// within each rank, then reduced across ranks.
double computeTotalEnergy(const std::vector<Body>& bodies, const int rank, const int nranks) {
    const long n = (long)bodies.size();
    double local = 0.0;

#pragma omp parallel for schedule(static) reduction(+ : local)
    for (long i = rank; i < n; i += nranks) {
        const Body& body = bodies[i];
        local += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }

#pragma omp parallel for schedule(dynamic, 16) reduction(+ : local)
    for (long i = rank; i < n; i += nranks) {
        const double px = bodies[i].pos.x, py = bodies[i].pos.y, pz = bodies[i].pos.z;
        double acc = 0.0;
        for (long j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - px;
            const double dy = bodies[j].pos.y - py;
            const double dz = bodies[j].pos.z - pz;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            acc -= 1.0 / dist;
        }
        local += acc;
    }

    double energy = 0.0;
    MPI_Allreduce(&local, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return energy;
}

// Validate that simulation produces finite, reasonable values.
// Runs on the full (replicated) state; the result is identical on all ranks.
// Only rank 0 prints diagnostics.
bool validateSimulation(const std::vector<Body>& bodies, const int rank) {
    const long n = (long)bodies.size();
    const double maxPos = 1e6;
    const double maxVel = 1e6;
    int badFinite = 0, badPos = 0, badVel = 0;

#pragma omp parallel for schedule(static) reduction(+ : badFinite, badPos, badVel)
    for (long i = 0; i < n; ++i) {
        const Body& body = bodies[i];
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            ++badFinite;
        }
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            ++badPos;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            ++badVel;
        }
    }

    if (badFinite) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value in body state\n");
        return false;
    }
    if (badPos) {
        if (rank == 0) printf("Validation failed: body position exceeds reasonable bounds\n");
        return false;
    }
    if (badVel) {
        if (rank == 0) printf("Validation failed: body velocity exceeds reasonable bounds\n");
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

// Pick a block size that yields enough blocks to occupy the device for the
// local problem size, while keeping tiles large for big problems.
static int chooseBlockSize(const int nLocal, const int numSMs) {
    int block = 256;
    while (block > 64 && (nLocal + block - 1) / block < 2 * numSMs) block /= 2;
    return block;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    int parseStatus = -1;  // -1: continue, 0: exit 0, 1: exit 1
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
            parseStatus = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
            break;
        }
    }
    if (parseStatus >= 0) {
        MPI_Finalize();
        return parseStatus;
    }
    if (numBodies < 0) numBodies = 0;

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---- GPU selection: one device per node-local rank (round-robin) ----
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    int numSMs = 1;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));

    // ---- Domain decomposition: contiguous slices ----
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        counts[r] = numBodies / nranks + (r < numBodies % nranks ? 1 : 0);
        displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
    }
    const int nLocal = counts[rank];
    const int offset = displs[rank];
    // Byte-level counts for MPI_Allgatherv of double3 positions (3 doubles each)
    std::vector<int> counts3(nranks), displs3(nranks);
    for (int r = 0; r < nranks; ++r) {
        counts3[r] = counts[r] * 3;
        displs3[r] = displs[r] * 3;
    }

    // Initialize bodies (deterministic, identical sequence on all ranks)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Host staging buffers (structure of arrays: positions replicated,
    // velocities local)
    std::vector<double3> hPos(numBodies);
    std::vector<double3> hVelLocal(nLocal);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        hPos[i] = make_double3(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z);
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nLocal; ++i) {
        const Body& b = bodies[offset + i];
        hVelLocal[i] = make_double3(b.vel.x, b.vel.y, b.vel.z);
    }

    // Device buffers
    double3* dPos = nullptr;
    double3* dVel = nullptr;
    CUDA_CHECK(cudaMalloc(&dPos, sizeof(double3) * (numBodies > 0 ? numBodies : 1)));
    CUDA_CHECK(cudaMalloc(&dVel, sizeof(double3) * (nLocal > 0 ? nLocal : 1)));
    if (numBodies > 0) CUDA_CHECK(cudaMemcpy(dPos, hPos.data(), sizeof(double3) * numBodies, cudaMemcpyHostToDevice));
    if (nLocal > 0) CUDA_CHECK(cudaMemcpy(dVel, hVelLocal.data(), sizeof(double3) * nLocal, cudaMemcpyHostToDevice));

    // Pinned host buffer for the per-step position exchange
    double3* hPosPinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&hPosPinned, sizeof(double3) * (numBodies > 0 ? numBodies : 1)));

    const int blockSize = chooseBlockSize(nLocal, numSMs);
    const int gridSize = (nLocal + blockSize - 1) / blockSize;
    const size_t shmem = sizeof(double3) * blockSize;
    const int intBlock = 256;
    const int intGrid = (nLocal + intBlock - 1) / intBlock;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Run simulation
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        if (nLocal > 0) {
            computeForcesKernel<<<gridSize, blockSize, shmem, stream>>>(dPos, dVel, numBodies, offset, nLocal);
            integrateBodiesKernel<<<intGrid, intBlock, 0, stream>>>(dPos, dVel, offset, nLocal);
        }
        if (nranks > 1) {
            // Exchange updated positions: download local slice, allgather, upload all.
            if (nLocal > 0) {
                CUDA_CHECK(cudaMemcpyAsync(hPosPinned + offset, dPos + offset, sizeof(double3) * nLocal,
                                           cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hPosPinned, counts3.data(), displs3.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
            if (numBodies > 0) {
                CUDA_CHECK(cudaMemcpyAsync(dPos, hPosPinned, sizeof(double3) * numBodies, cudaMemcpyHostToDevice,
                                           stream));
            }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long local_ms = duration.count();
    long global_ms = 0;
    MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", global_ms);

    // ---- Gather final state (positions and velocities) on all ranks ----
    if (numBodies > 0) CUDA_CHECK(cudaMemcpy(hPos.data(), dPos, sizeof(double3) * numBodies, cudaMemcpyDeviceToHost));
    if (nLocal > 0) CUDA_CHECK(cudaMemcpy(hVelLocal.data(), dVel, sizeof(double3) * nLocal, cudaMemcpyDeviceToHost));
    if (nranks > 1) {
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hPos.data(), counts3.data(), displs3.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }
    std::vector<double3> hVel(numBodies);
    if (nranks > 1) {
        MPI_Allgatherv(hVelLocal.data(), nLocal * 3, MPI_DOUBLE, hVel.data(), counts3.data(), displs3.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
    } else {
        hVel = hVelLocal;
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos = Vec3(hPos[i].x, hPos[i].y, hPos[i].z);
        bodies[i].vel = Vec3(hVel[i].x, hVel[i].y, hVel[i].z);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(hPosPinned));
    CUDA_CHECK(cudaFree(dPos));
    CUDA_CHECK(cudaFree(dVel));

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData((size_t)numBodies * 6);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const Body& body = bodies[i];
            double* d = bodyData.data() + (size_t)i * 6;
            d[0] = body.pos.x;
            d[1] = body.pos.y;
            d[2] = body.pos.z;
            d[3] = body.vel.x;
            d[4] = body.vel.y;
            d[5] = body.vel.z;
        }
        print_results(bodyData, "Bodies");
    }

    int exitCode = 0;
    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        if (validateSimulation(bodies, rank)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies, rank, nranks);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Finalize();
    return exitCode;
}
