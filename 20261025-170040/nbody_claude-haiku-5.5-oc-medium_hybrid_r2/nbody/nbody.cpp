// Hybrid MPI + OpenMP + CUDA N-body benchmark.
//
// Decomposition:
//   - MPI:    bodies are block-partitioned across ranks. Each rank owns the
//             velocities and positions of its block and, after every step,
//             the position blocks are exchanged with MPI_Allgatherv.
//   - CUDA:   each rank drives one GPU (local rank mod visible device count).
//             Force evaluation (shared-memory tiled O(N^2/P) per rank) and the
//             position integration run on the device.
//   - OpenMP: parallelizes the O(N^2) potential-energy evaluation used by the
//             validation path on the root rank.
//
// Semantics: for every body i the sum over j is accumulated in the same order
// as the original serial code, and the floating-point expressions are written
// with explicit fma() where the original (GCC, -march=native) contracted them,
// so results are independent of the number of ranks/GPUs and match the
// original serial implementation.

#include <mpi.h>
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
constexpr int BLOCK = 256;  // threads per block for the element-wise integration kernel

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};
static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be tightly packed for MPI transfers");

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err_));                                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

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

// Force accumulation for the local bodies [0, nLocal) (global indices offset..).
//
// Work decomposition inside a block: BI bodies x TJ source bodies per tile.
//   1. All threads evaluate the pair terms (dx, dy, dz, 1/|r|^3) of the tile in
//      parallel and stage them in shared memory.
//   2. One thread per body accumulates its force by walking the tile's j in
//      increasing order, carrying the accumulator across tiles.
// Every body therefore sums its terms in exactly the order of the original
// serial loop (j = 0..n-1), while the expensive sqrt/division work is spread
// over the whole block, which keeps the GPU occupied even when a rank owns
// only a few thousand bodies.
constexpr int BI = 32;       // bodies per block
constexpr int TJ = 32;       // source bodies per tile
constexpr int THREADS = 256; // threads per block (>= BI)

__global__ void __launch_bounds__(THREADS)
computeForcesKernel(const Vec3* __restrict__ pos, Vec3* __restrict__ vel,
                    const int n, const int offset, const int nLocal) {
    __shared__ double siX[BI], siY[BI], siZ[BI];    // positions of this block's bodies
    __shared__ double sjX[TJ], sjY[TJ], sjZ[TJ];    // positions of the current tile
    __shared__ double pDx[BI * TJ], pDy[BI * TJ], pDz[BI * TJ], pInv3[BI * TJ];

    const int tid = threadIdx.x;
    const int ib = blockIdx.x * BI;               // first local body of this block
    const bool accumThread = (tid < BI) && (ib + tid < nLocal);

    if (tid < BI) {
        const int li = ib + tid;
        const bool valid = li < nLocal;
        siX[tid] = valid ? pos[offset + li].x : 0.0;
        siY[tid] = valid ? pos[offset + li].y : 0.0;
        siZ[tid] = valid ? pos[offset + li].z : 0.0;
    }
    __syncthreads();

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;  // accumulators (used by threads tid < BI)

    for (int base = 0; base < n; base += TJ) {
        const int cnt = (n - base < TJ) ? (n - base) : TJ;

        if (tid < cnt) {
            sjX[tid] = pos[base + tid].x;
            sjY[tid] = pos[base + tid].y;
            sjZ[tid] = pos[base + tid].z;
        }
        __syncthreads();

        // Phase 1: pair terms. Layout p = bj * BI + bi keeps accumulation reads conflict-free.
        for (int p = tid; p < cnt * BI; p += THREADS) {
            const int bj = p / BI;
            const int bi = p - bj * BI;
            if (ib + bi < nLocal) {
                const double dx = sjX[bj] - siX[bi];
                const double dy = sjY[bj] - siY[bi];
                const double dz = sjZ[bj] - siZ[bi];
                const double distSqr = fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                pDx[p] = dx;
                pDy[p] = dy;
                pDz[p] = dz;
                pInv3[p] = invDist * invDist * invDist;
            }
        }
        __syncthreads();

        // Phase 2: ordered accumulation over the tile (ascending j)
        if (accumThread) {
            for (int bj = 0; bj < cnt; ++bj) {
                const int p = bj * BI + tid;
                Fx = fma(pDx[p], pInv3[p], Fx);
                Fy = fma(pDy[p], pInv3[p], Fy);
                Fz = fma(pDz[p], pInv3[p], Fz);
            }
        }
        __syncthreads();
    }

    if (accumThread) {
        const int li = ib + tid;
        vel[li].x += DT * Fx;
        vel[li].y += DT * Fy;
        vel[li].z += DT * Fz;
    }
}

// Position update for the local bodies. Must run after all force kernels have
// finished (it is a separate launch, so stream ordering guarantees this).
__global__ void integrateKernel(Vec3* __restrict__ pos, const Vec3* __restrict__ vel,
                                const int offset, const int nLocal) {
    const int li = blockIdx.x * BLOCK + threadIdx.x;
    if (li < nLocal) {
        pos[offset + li].x = fma(vel[li].x, DT, pos[offset + li].x);
        pos[offset + li].y = fma(vel[li].y, DT, pos[offset + li].y);
        pos[offset + li].z = fma(vel[li].z, DT, pos[offset + li].z);
    }
}

// Potential energy pair terms are computed in parallel (OpenMP) but accumulated
// serially in the original order, so the result is bitwise identical.
double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        const double s = fma(body.vel.z, body.vel.z,
                             fma(body.vel.x, body.vel.x, body.vel.y * body.vel.y));
        energy = fma(0.5, s, energy);
    }

    // Potential energy (assuming unit mass for all bodies)
    std::vector<double> term(n);
    for (size_t i = 0; i < n; ++i) {
        const double xi = bodies[i].pos.x, yi = bodies[i].pos.y, zi = bodies[i].pos.z;
#pragma omp parallel for schedule(static)
        for (long long jj = static_cast<long long>(i) + 1; jj < static_cast<long long>(n); ++jj) {
            const size_t j = static_cast<size_t>(jj);
            const double dx = bodies[j].pos.x - xi;
            const double dy = bodies[j].pos.y - yi;
            const double dz = bodies[j].pos.z - zi;
            const double distSqr = fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
            term[j] = 1.0 / std::sqrt(distSqr);
        }
        for (size_t j = i + 1; j < n; ++j) {
            energy -= term[j];
        }
    }

    return energy;
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
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank, so all ranks agree on exits)
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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (numBodies < 0) {
        if (root) printf("Invalid number of bodies: %d\n", numBodies);
        MPI_Finalize();
        return 1;
    }

    if (root) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int N = numBodies;

    // Bind this rank to a GPU: local rank (ranks sharing the node) mod device count.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // Block partition of bodies across ranks
    std::vector<int> counts(size), displs(size);
    {
        const int base = N / size, rem = N % size;
        int off = 0;
        for (int r = 0; r < size; ++r) {
            counts[r] = base + (r < rem ? 1 : 0);
            displs[r] = off;
            off += counts[r];
        }
    }
    std::vector<int> countsD(size), displsD(size);  // in doubles (3 per body)
    for (int r = 0; r < size; ++r) {
        countsD[r] = 3 * counts[r];
        displsD[r] = 3 * displs[r];
    }
    const int nLocal = counts[rank];
    const int offset = displs[rank];

    // Initialize bodies (deterministic; every rank generates the identical state)
    std::vector<Body> bodies(N);
    randomizeBodies(bodies);

    const size_t allBytes = static_cast<size_t>(N > 0 ? N : 1) * sizeof(Vec3);
    const size_t localBytes = static_cast<size_t>(nLocal > 0 ? nLocal : 1) * sizeof(Vec3);

    // Host staging buffer for the full position array (pinned for fast DMA)
    Vec3* hPos = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hPos), allBytes));
    for (int i = 0; i < N; ++i) hPos[i] = bodies[i].pos;

    Vec3 *dPos = nullptr, *dVel = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dPos), allBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dVel), localBytes));
    if (N > 0) {
        CUDA_CHECK(cudaMemcpy(dPos, hPos, static_cast<size_t>(N) * sizeof(Vec3), cudaMemcpyHostToDevice));
    }
    if (nLocal > 0) {
        std::vector<Vec3> hVelLocal(nLocal);
        for (int i = 0; i < nLocal; ++i) hVelLocal[i] = bodies[offset + i].vel;
        CUDA_CHECK(cudaMemcpy(dVel, hVelLocal.data(), static_cast<size_t>(nLocal) * sizeof(Vec3),
                              cudaMemcpyHostToDevice));
    }

    const int gridIntegrate = (nLocal + BLOCK - 1) / BLOCK;
    const int gridForces = (nLocal + BI - 1) / BI;

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        if (nLocal > 0) {
            computeForcesKernel<<<gridForces, THREADS>>>(dPos, dVel, N, offset, nLocal);
            integrateKernel<<<gridIntegrate, BLOCK>>>(dPos, dVel, offset, nLocal);
            CUDA_CHECK(cudaGetLastError());

            // Local positions to host staging buffer
            CUDA_CHECK(cudaMemcpy(hPos + offset, dPos + offset,
                                  static_cast<size_t>(nLocal) * sizeof(Vec3), cudaMemcpyDeviceToHost));
        }

        // Exchange position blocks so every rank has the full, updated state
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hPos, countsD.data(), displsD.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Refresh remote position blocks on this rank's device (own block is already current)
        if (offset > 0) {
            CUDA_CHECK(cudaMemcpy(dPos, hPos, static_cast<size_t>(offset) * sizeof(Vec3),
                                  cudaMemcpyHostToDevice));
        }
        const int tail = N - offset - nLocal;
        if (tail > 0) {
            CUDA_CHECK(cudaMemcpy(dPos + offset + nLocal, hPos + offset + nLocal,
                                  static_cast<size_t>(tail) * sizeof(Vec3), cudaMemcpyHostToDevice));
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long long elapsedMs = duration.count(), maxMs = 0;
    MPI_Reduce(&elapsedMs, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (root) printf("Simulation time: %lld ms\n", maxMs);

    // Gather the full final state on the root when results or validation are requested
    if (printResults || validate) {
        std::vector<Vec3> hVelLocal(nLocal > 0 ? nLocal : 0);
        if (nLocal > 0) {
            CUDA_CHECK(cudaMemcpy(hVelLocal.data(), dVel, static_cast<size_t>(nLocal) * sizeof(Vec3),
                                  cudaMemcpyDeviceToHost));
        }
        std::vector<Vec3> velAll(root ? static_cast<size_t>(N) : 0);
        MPI_Gatherv(hVelLocal.data(), 3 * nLocal, MPI_DOUBLE,
                    root ? reinterpret_cast<double*>(velAll.data()) : nullptr,
                    countsD.data(), displsD.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (root) {
            for (int i = 0; i < N; ++i) {
                bodies[i].pos = hPos[i];
                bodies[i].vel = velAll[i];
            }
        }
    }

    int validationOk = 1;

    if (root && printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(N) * 6);
        for (const auto& body : bodies) {
            bodyData.push_back(body.pos.x);
            bodyData.push_back(body.pos.y);
            bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x);
            bodyData.push_back(body.vel.y);
            bodyData.push_back(body.vel.z);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    if (validate && root) {
        printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            validationOk = 0;
        }
    }
    if (validate) {
        MPI_Bcast(&validationOk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    CUDA_CHECK(cudaFree(dPos));
    CUDA_CHECK(cudaFree(dVel));
    CUDA_CHECK(cudaFreeHost(hPos));
    MPI_Finalize();

    return validationOk ? 0 : 1;
}
