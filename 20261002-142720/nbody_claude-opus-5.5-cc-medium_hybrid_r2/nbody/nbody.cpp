#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

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
// Hybrid MPI + OpenMP + CUDA parallelization
//
// * MPI:    bodies are block-partitioned across ranks; after each step the
//           updated positions are exchanged with MPI_Allgatherv.
// * OpenMP: inside a rank, one host thread drives each GPU assigned to it
//           (GPUs of a node are split among the node-local ranks); OpenMP is
//           also used for host-side O(N^2) energy evaluation.
// * CUDA:   force evaluation (shared-memory tiled, all-pairs, FP64) and
//           integration of the bodies owned by each GPU.
//
// Forces only depend on positions, which are not modified during
// computeForces, so every body's update is independent (same semantics as
// the original sequential code).
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

constexpr int BLOCK = 128;  // bodies per block == positions per shared-memory tile

// Partial forces of bodies [0, nLocal) (global index offset + i) against the
// j-range belonging to split blockIdx.y. Output layout: F[split][comp][nLocal].
__global__ void __launch_bounds__(BLOCK)
forceKernel(const double4* __restrict__ pos, int n, int offset, int nLocal,
            int tilesPerSplit, double* __restrict__ partial) {
    __shared__ double4 tile[BLOCK];
    const int li = blockIdx.x * BLOCK + threadIdx.x;
    const bool active = li < nLocal;
    const double4 pi = active ? pos[offset + li] : make_double4(0.0, 0.0, 0.0, 0.0);

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    const int jBegin = blockIdx.y * tilesPerSplit * BLOCK;
    const int jEnd = min(n, jBegin + tilesPerSplit * BLOCK);

    for (int j0 = jBegin; j0 < jEnd; j0 += BLOCK) {
        const int jj = j0 + threadIdx.x;
        if (jj < jEnd) tile[threadIdx.x] = pos[jj];
        __syncthreads();
        const int cnt = min(BLOCK, jEnd - j0);
        if (cnt == BLOCK) {
#pragma unroll 8
            for (int k = 0; k < BLOCK; ++k) {
                const double4 pj = tile[k];
                const double dx = pj.x - pi.x;
                const double dy = pj.y - pi.y;
                const double dz = pj.z - pi.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = rsqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        } else {
            for (int k = 0; k < cnt; ++k) {
                const double4 pj = tile[k];
                const double dx = pj.x - pi.x;
                const double dy = pj.y - pi.y;
                const double dz = pj.z - pi.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = rsqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        double* out = partial + (size_t)blockIdx.y * 3 * nLocal;
        out[li] = Fx;
        out[nLocal + li] = Fy;
        out[2 * (size_t)nLocal + li] = Fz;
    }
}

// Reduce partial forces (in split order), kick velocities, drift positions.
__global__ void integrateKernel(double4* __restrict__ pos, double* __restrict__ vel, int offset,
                                int nLocal, int nSplit, const double* __restrict__ partial) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= nLocal) return;
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int s = 0; s < nSplit; ++s) {
        const double* p = partial + (size_t)s * 3 * nLocal;
        Fx += p[li];
        Fy += p[nLocal + li];
        Fz += p[2 * (size_t)nLocal + li];
    }
    double* v = vel + 3 * (size_t)li;
    const double vx = v[0] + DT * Fx;
    const double vy = v[1] + DT * Fy;
    const double vz = v[2] + DT * Fz;
    v[0] = vx;
    v[1] = vy;
    v[2] = vz;
    double4 p = pos[offset + li];
    p.x += vx * DT;
    p.y += vy * DT;
    p.z += vz * DT;
    pos[offset + li] = p;
}

struct DeviceCtx {
    int device = 0;
    int offset = 0;  // global index of first owned body
    int count = 0;   // number of owned bodies
    int nSplit = 1;
    int tilesPerSplit = 1;
    cudaStream_t stream = nullptr;
    double4* dPos = nullptr;      // all n positions
    double* dVel = nullptr;       // owned velocities (AoS xyz)
    double* dPartial = nullptr;   // nSplit * 3 * count partial forces
};

static void blockRange(long long n, int parts, int idx, int& begin, int& cnt) {
    const long long base = n / parts, rem = n % parts;
    const long long b = idx * base + std::min<long long>(idx, rem);
    begin = (int)b;
    cnt = (int)(base + (idx < rem ? 1 : 0));
}

static void setupDevice(DeviceCtx& c, int n, const double4* hPos, const std::vector<Body>& bodies) {
    CUDA_CHECK(cudaSetDevice(c.device));
    CUDA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, c.device));

    // Split the j-range so the GPU is saturated even for few owned bodies.
    const int nTiles = std::max(1, (n + BLOCK - 1) / BLOCK);
    const int bx = (c.count + BLOCK - 1) / BLOCK;
    const int targetBlocks = prop.multiProcessorCount * 8;
    int split = 1;
    if (bx > 0 && bx < targetBlocks) split = (targetBlocks + bx - 1) / bx;
    split = std::max(1, std::min(split, nTiles));
    c.tilesPerSplit = (nTiles + split - 1) / split;
    c.nSplit = std::max(1, (nTiles + c.tilesPerSplit - 1) / c.tilesPerSplit);

    CUDA_CHECK(cudaMalloc(&c.dPos, sizeof(double4) * std::max(n, 1)));
    CUDA_CHECK(cudaMalloc(&c.dVel, sizeof(double) * 3 * std::max(c.count, 1)));
    CUDA_CHECK(cudaMalloc(&c.dPartial, sizeof(double) * 3 * (size_t)c.nSplit * std::max(c.count, 1)));
    CUDA_CHECK(cudaMemcpy(c.dPos, hPos, sizeof(double4) * n, cudaMemcpyHostToDevice));
    std::vector<double> v(3 * (size_t)c.count);
    for (int i = 0; i < c.count; ++i) {
        const Body& b = bodies[c.offset + i];
        v[3 * i] = b.vel.x;
        v[3 * i + 1] = b.vel.y;
        v[3 * i + 2] = b.vel.z;
    }
    if (c.count > 0)
        CUDA_CHECK(cudaMemcpy(c.dVel, v.data(), sizeof(double) * v.size(), cudaMemcpyHostToDevice));
}

static void stepDevice(DeviceCtx& c, int n) {
    if (c.count == 0) return;
    const dim3 grid((c.count + BLOCK - 1) / BLOCK, c.nSplit);
    forceKernel<<<grid, BLOCK, 0, c.stream>>>(c.dPos, n, c.offset, c.count, c.tilesPerSplit,
                                               c.dPartial);
    integrateKernel<<<(c.count + 255) / 256, 256, 0, c.stream>>>(c.dPos, c.dVel, c.offset, c.count,
                                                                 c.nSplit, c.dPartial);
    CUDA_CHECK(cudaGetLastError());
}

// Energy is computed collectively: rows of the pairwise potential sum are
// distributed cyclically over MPI ranks and dynamically over OpenMP threads.
// Every rank must hold the full body state; the result is valid on all ranks.
double computeTotalEnergy(const std::vector<Body>& bodies, int rank, int nRanks) {
    double energy = 0.0;
    const long long n = (long long)bodies.size();

    // Kinetic energy (assuming unit mass)
    if (rank == 0) {
#pragma omp parallel for reduction(+ : energy) schedule(static)
        for (long long i = 0; i < n; ++i) {
            const Body& body = bodies[i];
            energy += 0.5 * (body.vel.x * body.vel.x +
                            body.vel.y * body.vel.y +
                            body.vel.z * body.vel.z);
        }
    }

    // Potential energy (assuming unit mass for all bodies)
    double potential = 0.0;
#pragma omp parallel for reduction(+ : potential) schedule(dynamic, 16)
    for (long long i = rank; i < n; i += nRanks) {
        const Vec3 pi = bodies[i].pos;
        double rowSum = 0.0;
        for (long long j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - pi.x;
            const double dy = bodies[j].pos.y - pi.y;
            const double dz = bodies[j].pos.z - pi.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            rowSum += 1.0 / dist;
        }
        potential += rowSum;
    }
    energy -= potential;

    double total = 0.0;
    MPI_Allreduce(&energy, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    const bool root = rank == 0;

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
    
    if (root) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    const int n = std::max(numBodies, 0);

    // Initialize bodies (deterministic, identical on every rank)
    std::vector<Body> bodies(n);
    randomizeBodies(bodies);

    // GPU assignment: split the node's GPUs among the node-local ranks; a rank
    // owning several GPUs drives each one from its own OpenMP thread.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);
    int nDevTotal = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nDevTotal));
    if (nDevTotal < 1) {
        fprintf(stderr, "No CUDA device found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> myDevices;
    if (localSize <= nDevTotal) {
        for (int d = localRank; d < nDevTotal; d += localSize) myDevices.push_back(d);
    } else {
        myDevices.push_back(localRank % nDevTotal);
    }
    const int nDev = (int)myDevices.size();

    // Block decomposition: ranks own contiguous ranges, split further per GPU.
    std::vector<int> rankCounts(nRanks), rankDispls(nRanks);
    for (int r = 0; r < nRanks; ++r) blockRange(n, nRanks, r, rankDispls[r], rankCounts[r]);
    const int myBegin = rankDispls[rank], myCount = rankCounts[rank];

    std::vector<DeviceCtx> ctx(nDev);
    for (int d = 0; d < nDev; ++d) {
        int b, c;
        blockRange(myCount, nDev, d, b, c);
        ctx[d].device = myDevices[d];
        ctx[d].offset = myBegin + b;
        ctx[d].count = c;
    }

    double4* hPos = nullptr;
    CUDA_CHECK(cudaMallocHost(&hPos, sizeof(double4) * std::max(n, 1)));
    for (int i = 0; i < n; ++i)
        hPos[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);

#pragma omp parallel for num_threads(nDev) schedule(static, 1)
    for (int d = 0; d < nDev; ++d) setupDevice(ctx[d], n, hPos, bodies);

    MPI_Datatype posType;
    MPI_Type_contiguous(4, MPI_DOUBLE, &posType);
    MPI_Type_commit(&posType);
    const bool exchange = nRanks > 1 || nDev > 1;

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(nDev)
    {
        DeviceCtx& c = ctx[omp_get_thread_num()];
        CUDA_CHECK(cudaSetDevice(c.device));
        for (int step = 0; step < numSteps; ++step) {
            stepDevice(c, n);
            if (exchange) {
                if (c.count > 0)
                    CUDA_CHECK(cudaMemcpyAsync(hPos + c.offset, c.dPos + c.offset,
                                               sizeof(double4) * c.count, cudaMemcpyDeviceToHost,
                                               c.stream));
                CUDA_CHECK(cudaStreamSynchronize(c.stream));
#pragma omp barrier
#pragma omp master
                MPI_Allgatherv(MPI_IN_PLACE, 0, posType, hPos, rankCounts.data(), rankDispls.data(),
                               posType, MPI_COMM_WORLD);
#pragma omp barrier
                if (step + 1 < numSteps) {
                    // Upload every position not produced by this GPU.
                    if (c.offset > 0)
                        CUDA_CHECK(cudaMemcpyAsync(c.dPos, hPos, sizeof(double4) * c.offset,
                                                   cudaMemcpyHostToDevice, c.stream));
                    const int tail = c.offset + c.count;
                    if (tail < n)
                        CUDA_CHECK(cudaMemcpyAsync(c.dPos + tail, hPos + tail,
                                                   sizeof(double4) * (n - tail),
                                                   cudaMemcpyHostToDevice, c.stream));
                }
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
    if (root) printf("Simulation time: %ld ms\n", maxDuration);

    // Collect final state: positions are already replicated in hPos when
    // exchanging; velocities are gathered on rank 0 (and everywhere for -v).
    std::vector<double> hVel(3 * (size_t)n);
    for (int d = 0; d < nDev; ++d) {
        DeviceCtx& c = ctx[d];
        CUDA_CHECK(cudaSetDevice(c.device));
        if (!exchange && n > 0)
            CUDA_CHECK(cudaMemcpy(hPos, c.dPos, sizeof(double4) * n, cudaMemcpyDeviceToHost));
        if (c.count > 0)
            CUDA_CHECK(cudaMemcpy(hVel.data() + 3 * (size_t)c.offset, c.dVel,
                                  sizeof(double) * 3 * c.count, cudaMemcpyDeviceToHost));
    }
    if (nRanks > 1) {
        MPI_Datatype velType;
        MPI_Type_contiguous(3, MPI_DOUBLE, &velType);
        MPI_Type_commit(&velType);
        if (validate) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, velType, hVel.data(), rankCounts.data(),
                           rankDispls.data(), velType, MPI_COMM_WORLD);
        } else if (root) {
            MPI_Gatherv(MPI_IN_PLACE, 0, velType, hVel.data(), rankCounts.data(),
                        rankDispls.data(), velType, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(hVel.data() + 3 * (size_t)myBegin, myCount, velType, nullptr, nullptr,
                        nullptr, velType, 0, MPI_COMM_WORLD);
        }
        MPI_Type_free(&velType);
    }
    for (int i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(hPos[i].x, hPos[i].y, hPos[i].z);
        bodies[i].vel = Vec3(hVel[3 * (size_t)i], hVel[3 * (size_t)i + 1], hVel[3 * (size_t)i + 2]);
    }

    for (auto& c : ctx) {
        CUDA_CHECK(cudaSetDevice(c.device));
        CUDA_CHECK(cudaFree(c.dPos));
        CUDA_CHECK(cudaFree(c.dVel));
        CUDA_CHECK(cudaFree(c.dPartial));
        CUDA_CHECK(cudaStreamDestroy(c.stream));
    }
    CUDA_CHECK(cudaFreeHost(hPos));
    MPI_Type_free(&posType);

    // Print results for external validation
    if (printResults && root) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
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
    int exitCode = 0;
    if (validate) {
        if (root) printf("Validating simulation results...\n");

        int ok = root ? (validateSimulation(bodies) ? 1 : 0) : 0;
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (ok) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies, rank, nRanks);
            if (root) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (root) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Finalize();
    return exitCode;
}
