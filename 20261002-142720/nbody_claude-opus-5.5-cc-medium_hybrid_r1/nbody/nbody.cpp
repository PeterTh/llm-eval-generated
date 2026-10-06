// Hybrid MPI + OpenMP + CUDA N-body benchmark.
//
// Decomposition:
//  * MPI: bodies are block-distributed across ranks. Every rank owns a
//    contiguous slice of bodies (velocities stay resident on its GPU) and keeps
//    a replicated copy of all positions, refreshed with MPI_Allgatherv once
//    per step.
//  * CUDA: each rank drives one GPU (node-local rank -> device). A single fused
//    kernel computes the all-pairs forces for the owned bodies, then updates
//    their velocities and positions (kick + drift), exactly like
//    computeForces() followed by integrateBodies() in the serial code.
//  * OpenMP: host-side work (initial state packing, validation and the O(N^2)
//    energy computation, which is additionally distributed across ranks).
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

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
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

// ---------------------------------------------------------------------------
// CUDA kernel
// ---------------------------------------------------------------------------
constexpr int BLOCK_SIZE = 256;   // threads per block
constexpr int TILE = 256;         // bodies per shared-memory tile

// Each body i (local index) is handled by LANES consecutive threads of a warp;
// lane k accumulates interactions j = k, k+LANES, ... and the partial sums are
// combined with warp shuffles. LANES > 1 is used when the local body count is
// too small to fill the GPU.
template <int LANES>
__global__ void __launch_bounds__(BLOCK_SIZE)
stepKernel(const double* __restrict__ posAll,   // 3*n, all positions (old)
           double* __restrict__ posOut,         // 3*nLocal, new positions of owned bodies
           double* __restrict__ vel,            // 3*nLocal, velocities of owned bodies
           int n, int offset, int nLocal) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE];

    const int lane = threadIdx.x % LANES;
    const int li = (blockIdx.x * BLOCK_SIZE + threadIdx.x) / LANES;
    const bool active = li < nLocal;

    double px = 0.0, py = 0.0, pz = 0.0;
    if (active) {
        const int gi = offset + li;
        px = posAll[3 * gi + 0];
        py = posAll[3 * gi + 1];
        pz = posAll[3 * gi + 2];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += TILE) {
        const int cnt = min(TILE, n - base);
        __syncthreads();
        for (int t = threadIdx.x; t < cnt; t += BLOCK_SIZE) {
            const int j = base + t;
            sx[t] = posAll[3 * j + 0];
            sy[t] = posAll[3 * j + 1];
            sz[t] = posAll[3 * j + 2];
        }
        __syncthreads();
        if (active) {
            if (cnt == TILE) {
#pragma unroll 8
                for (int t = lane; t < TILE; t += LANES) {
                    const double dx = sx[t] - px;
                    const double dy = sy[t] - py;
                    const double dz = sz[t] - pz;
                    const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    const double invDist = rsqrt(distSqr);
                    const double invDist3 = invDist * invDist * invDist;
                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
            } else {
                for (int t = lane; t < cnt; t += LANES) {
                    const double dx = sx[t] - px;
                    const double dy = sy[t] - py;
                    const double dz = sz[t] - pz;
                    const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    const double invDist = rsqrt(distSqr);
                    const double invDist3 = invDist * invDist * invDist;
                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
            }
        }
    }

    if constexpr (LANES > 1) {
#pragma unroll
        for (int off = LANES / 2; off > 0; off >>= 1) {
            Fx += __shfl_down_sync(0xffffffffu, Fx, off, LANES);
            Fy += __shfl_down_sync(0xffffffffu, Fy, off, LANES);
            Fz += __shfl_down_sync(0xffffffffu, Fz, off, LANES);
        }
    }

    if (active && lane == 0) {
        double vx = vel[3 * li + 0] + DT * Fx;
        double vy = vel[3 * li + 1] + DT * Fy;
        double vz = vel[3 * li + 2] + DT * Fz;
        vel[3 * li + 0] = vx;
        vel[3 * li + 1] = vy;
        vel[3 * li + 2] = vz;
        posOut[3 * li + 0] = px + vx * DT;
        posOut[3 * li + 1] = py + vy * DT;
        posOut[3 * li + 2] = pz + vz * DT;
    }
}

template <int LANES>
static void launchStep(const double* posAll, double* posOut, double* vel, int n, int offset,
                       int nLocal, cudaStream_t stream) {
    const long long threads = (long long)nLocal * LANES;
    const int blocks = (int)((threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
    stepKernel<LANES><<<blocks, BLOCK_SIZE, 0, stream>>>(posAll, posOut, vel, n, offset, nLocal);
}

static void launchStepDispatch(int lanes, const double* posAll, double* posOut, double* vel,
                               int n, int offset, int nLocal, cudaStream_t stream) {
    switch (lanes) {
        case 32: launchStep<32>(posAll, posOut, vel, n, offset, nLocal, stream); break;
        case 16: launchStep<16>(posAll, posOut, vel, n, offset, nLocal, stream); break;
        case 8: launchStep<8>(posAll, posOut, vel, n, offset, nLocal, stream); break;
        case 4: launchStep<4>(posAll, posOut, vel, n, offset, nLocal, stream); break;
        case 2: launchStep<2>(posAll, posOut, vel, n, offset, nLocal, stream); break;
        default: launchStep<1>(posAll, posOut, vel, n, offset, nLocal, stream); break;
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers (OpenMP, energy additionally split across MPI ranks)
// ---------------------------------------------------------------------------
double computeTotalEnergy(const std::vector<Body>& bodies, int rank, int nranks) {
    const long long n = (long long)bodies.size();
    double energy = 0.0;

    // Kinetic energy (assuming unit mass), computed on rank 0 only
    if (rank == 0) {
#pragma omp parallel for reduction(+ : energy) schedule(static)
        for (long long i = 0; i < n; ++i) {
            const Body& body = bodies[i];
            energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
        }
    }

    // Potential energy (assuming unit mass); rows i distributed cyclically
    // across ranks for load balance of the triangular loop.
    double pot = 0.0;
#pragma omp parallel for reduction(+ : pot) schedule(dynamic, 16)
    for (long long i = rank; i < n; i += nranks) {
        double local = 0.0;
        for (long long j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            local -= 1.0 / dist;
        }
        pot += local;
    }
    energy += pot;

    double total = 0.0;
    MPI_Allreduce(&energy, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return total;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    const long long n = (long long)bodies.size();
    // 0 = ok, 1 = NaN/Inf, 2 = position bound, 3 = velocity bound; first failing body wins
    long long firstBad = n;
#pragma omp parallel for reduction(min : firstBad) schedule(static)
    for (long long i = 0; i < n; ++i) {
        const Body& body = bodies[i];
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        const bool bad =
            !std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z) ||
            std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos ||
            std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel;
        if (bad && i < firstBad) firstBad = i;
    }
    if (firstBad == n) return true;

    const Body& body = bodies[firstBad];
    if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
        !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
        printf("Validation failed: found NaN or Inf value in body state\n");
    } else if (std::abs(body.pos.x) > 1e6 || std::abs(body.pos.y) > 1e6 || std::abs(body.pos.z) > 1e6) {
        printf("Validation failed: body position exceeds reasonable bounds\n");
    } else {
        printf("Validation failed: body velocity exceeds reasonable bounds\n");
    }
    return false;
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
    if (numBodies < 0) numBodies = 0;

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        fflush(stdout);
    }

    // ---- GPU selection: node-local rank -> device -------------------------
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount <= 0) {
        fprintf(stderr, "No CUDA device available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % devCount;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // ---- Domain decomposition ---------------------------------------------
    std::vector<int> counts(nranks), displs(nranks), counts3(nranks), displs3(nranks);
    {
        const int base = numBodies / nranks, rem = numBodies % nranks;
        int off = 0;
        for (int r = 0; r < nranks; ++r) {
            counts[r] = base + (r < rem ? 1 : 0);
            displs[r] = off;
            counts3[r] = 3 * counts[r];
            displs3[r] = 3 * off;
            off += counts[r];
        }
    }
    const int nLocal = counts[rank];
    const int offset = displs[rank];
    const size_t n = (size_t)numBodies;

    // Initialize bodies (deterministic; every rank generates the same state)
    std::vector<Body> bodies(n);
    randomizeBodies(bodies);

    // Pinned host buffers: replicated positions and owned velocities (packed xyz)
    double* hPos = nullptr;
    double* hVel = nullptr;
    CUDA_CHECK(cudaMallocHost(&hPos, 3 * n * sizeof(double) + 8));
    CUDA_CHECK(cudaMallocHost(&hVel, 3 * (size_t)nLocal * sizeof(double) + 8));
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n; ++i) {
        hPos[3 * i + 0] = bodies[i].pos.x;
        hPos[3 * i + 1] = bodies[i].pos.y;
        hPos[3 * i + 2] = bodies[i].pos.z;
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nLocal; ++i) {
        hVel[3 * i + 0] = bodies[offset + i].vel.x;
        hVel[3 * i + 1] = bodies[offset + i].vel.y;
        hVel[3 * i + 2] = bodies[offset + i].vel.z;
    }

    // Device buffers. With one rank we ping-pong between two full position
    // arrays; with several ranks the kernel writes the owned slice into a
    // separate buffer that is exchanged through MPI.
    double *dPosA = nullptr, *dPosB = nullptr, *dVel = nullptr;
    CUDA_CHECK(cudaMalloc(&dPosA, 3 * n * sizeof(double) + 8));
    CUDA_CHECK(cudaMalloc(&dPosB, 3 * n * sizeof(double) + 8));
    CUDA_CHECK(cudaMalloc(&dVel, 3 * (size_t)nLocal * sizeof(double) + 8));
    CUDA_CHECK(cudaMemcpy(dPosA, hPos, 3 * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVel, hVel, 3 * (size_t)nLocal * sizeof(double), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Split each body's j-loop over several lanes if there are too few owned
    // bodies to saturate the GPU.
    int lanes = 1;
    {
        const long long target = (long long)prop.multiProcessorCount * 1024;
        while (lanes < 32 && (long long)nLocal * lanes < target && (long long)lanes * 32 <= (long long)n) lanes *= 2;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ---- Run simulation ----------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    double* dCur = dPosA;
    double* dNext = dPosB;
    for (int step = 0; step < numSteps && n > 0; ++step) {
        if (nLocal > 0) {
            launchStepDispatch(lanes, dCur, dNext + 3 * (size_t)offset, dVel, numBodies, offset, nLocal,
                               stream);
        }
        if (nranks > 1) {
            // Exchange the freshly updated owned slices of positions.
            if (nLocal > 0) {
                CUDA_CHECK(cudaMemcpyAsync(hPos + 3 * (size_t)offset, dNext + 3 * (size_t)offset,
                                           3 * (size_t)nLocal * sizeof(double), cudaMemcpyDeviceToHost,
                                           stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hPos, counts3.data(), displs3.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpyAsync(dNext, hPos, 3 * n * sizeof(double), cudaMemcpyHostToDevice, stream));
        }
        std::swap(dCur, dNext);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) printf("Simulation time: %ld ms\n", duration.count());

    // ---- Collect final state ----------------------------------------------
    // Positions: already replicated on host for multi-rank runs.
    if (nranks == 1 || numSteps <= 0) {
        CUDA_CHECK(cudaMemcpy(hPos, dCur, 3 * n * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(hVel, dVel, 3 * (size_t)nLocal * sizeof(double), cudaMemcpyDeviceToHost));

    const bool needState = printResults || validate;
    std::vector<double> velAll;
    if (needState) {
        velAll.resize(3 * n + 1);
        if (nranks > 1) {
            MPI_Allgatherv(hVel, 3 * nLocal, MPI_DOUBLE, velAll.data(), counts3.data(), displs3.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
        } else {
            std::memcpy(velAll.data(), hVel, 3 * n * sizeof(double));
        }
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)n; ++i) {
            bodies[i].pos = Vec3(hPos[3 * i + 0], hPos[3 * i + 1], hPos[3 * i + 2]);
            bodies[i].vel = Vec3(velAll[3 * i + 0], velAll[3 * i + 1], velAll[3 * i + 2]);
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(dPosA));
    CUDA_CHECK(cudaFree(dPosB));
    CUDA_CHECK(cudaFree(dVel));
    CUDA_CHECK(cudaFreeHost(hPos));
    CUDA_CHECK(cudaFreeHost(hVel));

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(n * 6);
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

    int ret = 0;
    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        int ok = 0;
        if (rank == 0) {
            printf("Validating simulation results...\n");
            ok = validateSimulation(bodies) ? 1 : 0;
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (ok) {
            // Report final energy for reference
            const double finalEnergy = computeTotalEnergy(bodies, rank, nranks);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            ret = 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            ret = 1;
        }
    }

    fflush(stdout);
    MPI_Finalize();
    return ret;
}
