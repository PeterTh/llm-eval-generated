#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int TILE = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// GPU implementation of the force loop.  One thread handles one body and
// walks the source bodies in the original order, so every per-body sum is
// accumulated exactly like the reference implementation.  Explicit fma()
// calls reproduce the contraction chosen by the host compiler for the
// reference build; --fmad=false prevents any additional contraction.
__global__ void computeForcesKernel(const double* __restrict__ pos,
                                    double* __restrict__ vel,
                                    const int n,
                                    const int localStart,
                                    const int localN) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = t < localN;
    const int i = localStart + t;

    double ix = 0.0, iy = 0.0, iz = 0.0;
    if (active) {
        const size_t b = 3 * static_cast<size_t>(i);
        ix = pos[b];
        iy = pos[b + 1];
        iz = pos[b + 2];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    __shared__ double tile[3 * TILE];
    const int lane = threadIdx.x;

    for (int base = 0; base < n; base += TILE) {
        const int gj = base + lane;
        if (gj < n) {
            const size_t sb = 3 * static_cast<size_t>(gj);
            tile[3 * lane] = pos[sb];
            tile[3 * lane + 1] = pos[sb + 1];
            tile[3 * lane + 2] = pos[sb + 2];
        } else {
            tile[3 * lane] = 0.0;
            tile[3 * lane + 1] = 0.0;
            tile[3 * lane + 2] = 0.0;
        }
        __syncthreads();

        int cnt = n - base;
        if (cnt > TILE) cnt = TILE;

        if (active) {
            for (int j = 0; j < cnt; ++j) {
                const double dx = tile[3 * j] - ix;
                const double dy = tile[3 * j + 1] - iy;
                const double dz = tile[3 * j + 2] - iz;
                const double dy2 = dy * dy;
                const double dxy = fma(dx, dx, dy2);
                const double distSqr = fma(dz, dz, dxy) + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = (invDist * invDist) * invDist;

                Fx = fma(dx, invDist3, Fx);
                Fy = fma(dy, invDist3, Fy);
                Fz = fma(dz, invDist3, Fz);
            }
        }
        __syncthreads();
    }

    if (active) {
        const size_t lv = 3 * static_cast<size_t>(t);
        vel[lv] = vel[lv] + DT * Fx;
        vel[lv + 1] = vel[lv + 1] + DT * Fy;
        vel[lv + 2] = vel[lv + 2] + DT * Fz;
    }
}

// OpenMP implementation used for the bodies that are not handled by the GPU
// (and as a complete fallback when no CUDA device is available).
void computeForcesCPURange(const std::vector<double>& allpos,
                           const std::vector<double>& localpos,
                           std::vector<double>& localvel,
                           const int n,
                           const int begin,
                           const int end) {
#pragma omp parallel for schedule(static)
    for (int t = begin; t < end; ++t) {
        const double ix = localpos[3 * t];
        const double iy = localpos[3 * t + 1];
        const double iz = localpos[3 * t + 2];

        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        for (int j = 0; j < n; ++j) {
            const double dx = allpos[3 * j] - ix;
            const double dy = allpos[3 * j + 1] - iy;
            const double dz = allpos[3 * j + 2] - iz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        localvel[3 * t] += DT * Fx;
        localvel[3 * t + 1] += DT * Fy;
        localvel[3 * t + 2] += DT * Fz;
    }
}

void integrateBodies(std::vector<double>& localpos,
                     const std::vector<double>& localvel,
                     const int localN) {
#pragma omp parallel for schedule(static)
    for (int t = 0; t < localN; ++t) {
        localpos[3 * t] += localvel[3 * t] * DT;
        localpos[3 * t + 1] += localvel[3 * t + 1] * DT;
        localpos[3 * t + 2] += localvel[3 * t + 2] * DT;
    }
}

void randomizeBodies(std::vector<double>& localpos,
                     std::vector<double>& localvel,
                     const int numBodies,
                     const int start,
                     const int localN,
                     unsigned int seed = 42) {
    for (int b = 0; b < numBodies; ++b) {
        const double px = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double py = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double pz = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double vx = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double vy = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double vz = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;

        if (b >= start && b < start + localN) {
            const int t = b - start;
            localpos[3 * t] = px;
            localpos[3 * t + 1] = py;
            localpos[3 * t + 2] = pz;
            localvel[3 * t] = vx;
            localvel[3 * t + 1] = vy;
            localvel[3 * t + 2] = vz;
        }
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
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

struct GpuState {
    double* d_pos = nullptr;
    double* d_vel = nullptr;
    bool active = false;
};

bool setupGpu(GpuState& gpu,
              const std::vector<double>& localvel,
              const int n,
              const int localN,
              const int rank) {
    if (n <= 0) {
        return false;
    }

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
        cudaGetLastError();
        return false;
    }

    int localRank = 0;
    MPI_Comm nodeComm = MPI_COMM_NULL;
    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm) == MPI_SUCCESS) {
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);
    }
    const int device = localRank % deviceCount;
    if (cudaSetDevice(device) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    const size_t posBytes = 3 * sizeof(double) * static_cast<size_t>(n);
    const size_t velBytes = 3 * sizeof(double) * static_cast<size_t>(localN);

    if (cudaMalloc(&gpu.d_pos, posBytes) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (localN > 0) {
        if (cudaMalloc(&gpu.d_vel, velBytes) != cudaSuccess) {
            cudaGetLastError();
            cudaFree(gpu.d_pos);
            gpu.d_pos = nullptr;
            return false;
        }
        if (cudaMemcpy(gpu.d_vel, localvel.data(), velBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaGetLastError();
            cudaFree(gpu.d_vel);
            cudaFree(gpu.d_pos);
            gpu.d_vel = nullptr;
            gpu.d_pos = nullptr;
            return false;
        }
    }

    // Warm up the context and load the kernel before the timed region.
    if (localN > 0) {
        cudaMemset(gpu.d_pos, 0, posBytes);
        const int grid = (localN + TILE - 1) / TILE;
        computeForcesKernel<<<grid, TILE>>>(gpu.d_pos, gpu.d_vel, n, 0, localN);
        cudaDeviceSynchronize();
        cudaMemcpy(gpu.d_vel, localvel.data(), velBytes, cudaMemcpyHostToDevice);
    }

    gpu.active = true;
    return true;
}

// Upload the positions and launch the asynchronous force kernel.  The kernel
// updates the first `count` local velocities in device memory.
void gpuLaunchForces(GpuState& gpu,
                     const std::vector<double>& allpos,
                     const int n,
                     const int localStart,
                     const int count) {
    const size_t posBytes = 3 * sizeof(double) * static_cast<size_t>(n);
    cudaMemcpy(gpu.d_pos, allpos.data(), posBytes, cudaMemcpyHostToDevice);

    if (count > 0) {
        const int grid = (count + TILE - 1) / TILE;
        computeForcesKernel<<<grid, TILE>>>(gpu.d_pos, gpu.d_vel, n, localStart, count);
    }
}

void gpuDownloadForces(GpuState& gpu,
                       std::vector<double>& localvel,
                       const int count) {
    if (count > 0) {
        const size_t velBytes = 3 * sizeof(double) * static_cast<size_t>(count);
        cudaMemcpy(localvel.data(), gpu.d_vel, velBytes, cudaMemcpyDeviceToHost);
    }
}

// Pick how many of the local bodies the GPU should handle.  Both backends
// compute independent bodies, so any split preserves the result exactly.  A
// short calibration before the timed region measures complete step times for
// several candidate splits and keeps the fastest one for this machine.
int chooseGpuCount(GpuState& gpu,
                   const std::vector<double>& allpos,
                   const std::vector<double>& localpos,
                   const std::vector<double>& localvel,
                   const int n,
                   const int localStart,
                   const int localN) {
    if (!gpu.active || localN <= 0) {
        return 0;
    }

    using clock = std::chrono::high_resolution_clock;
    std::vector<double> tmpvel(localvel.size(), 0.0);

    // Warm up the OpenMP thread team and the GPU copy path.
    computeForcesCPURange(allpos, localpos, tmpvel, n, 0, std::min(localN, 256));
    gpuLaunchForces(gpu, allpos, n, localStart, std::min(localN, 256));
    gpuDownloadForces(gpu, tmpvel, std::min(localN, 256));

    double fractions[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
    int numFractions = 5;
    if (localN > 100000) {
        fractions[1] = 0.5;
        fractions[2] = 1.0;
        numFractions = 3;
    }

    double bestTime = 1e300;
    int bestCount = 0;
    const int reps = (localN > 20000) ? 1 : 2;

    for (int f = 0; f < numFractions; ++f) {
        int count = static_cast<int>(std::llround(fractions[f] * static_cast<double>(localN)));
        if (count > localN) count = localN;
        double best = 1e300;
        for (int rep = 0; rep < reps; ++rep) {
            const auto t0 = clock::now();
            if (count > 0) {
                gpuLaunchForces(gpu, allpos, n, localStart, count);
                if (count < localN) {
                    computeForcesCPURange(allpos, localpos, tmpvel, n, count, localN);
                }
                gpuDownloadForces(gpu, tmpvel, count);
            } else {
                computeForcesCPURange(allpos, localpos, tmpvel, n, 0, localN);
            }
            const auto t1 = clock::now();
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        if (best < bestTime) {
            bestTime = best;
            bestCount = count;
        }
    }

    // Restore the device velocities modified during calibration.
    const size_t velBytes = 3 * sizeof(double) * static_cast<size_t>(localN);
    cudaMemcpy(gpu.d_vel, localvel.data(), velBytes, cudaMemcpyHostToDevice);

    return bestCount;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
    if (numSteps < 0) numSteps = 0;

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Contiguous block decomposition of the bodies over MPI ranks.
    const int base = numBodies / size;
    const int rem = numBodies % size;
    const int localN = base + (rank < rem ? 1 : 0);
    const int localStart = rank * base + std::min(rank, rem);

    std::vector<int> counts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        const int cnt = base + (r < rem ? 1 : 0);
        const int off = r * base + std::min(r, rem);
        counts[r] = 3 * cnt;
        displs[r] = 3 * off;
    }

    // Local state kept in structure-of-arrays form for coalesced access.
    std::vector<double> localpos(3 * static_cast<size_t>(localN) + (localN == 0 ? 1 : 0));
    std::vector<double> localvel(3 * static_cast<size_t>(localN) + (localN == 0 ? 1 : 0));
    std::vector<double> allpos(3 * static_cast<size_t>(numBodies) + (numBodies == 0 ? 1 : 0));

    randomizeBodies(localpos, localvel, numBodies, localStart, localN);

    GpuState gpu;
    const bool useGpu = setupGpu(gpu, localvel, numBodies, localN, rank);

    // Populate the replicated position array before calibrating the split.
    MPI_Allgatherv(localpos.data(), 3 * localN, MPI_DOUBLE,
                   allpos.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    const int gpuCount = useGpu
        ? chooseGpuCount(gpu, allpos, localpos, localvel, numBodies, localStart, localN)
        : 0;

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        MPI_Allgatherv(localpos.data(), 3 * localN, MPI_DOUBLE,
                       allpos.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        if (useGpu && gpuCount > 0) {
            // Launch the GPU part first, then let OpenMP handle the rest of
            // the local bodies while the kernel is running.
            gpuLaunchForces(gpu, allpos, numBodies, localStart, gpuCount);
            if (gpuCount < localN) {
                computeForcesCPURange(allpos, localpos, localvel, numBodies, gpuCount, localN);
            }
            gpuDownloadForces(gpu, localvel, gpuCount);
        } else {
            computeForcesCPURange(allpos, localpos, localvel, numBodies, 0, localN);
        }

        integrateBodies(localpos, localvel, localN);
    }

    if (useGpu) {
        cudaDeviceSynchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    const long localMs = static_cast<long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    long maxMs = localMs;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", maxMs);
    }

    // Assemble the complete state on rank 0 for output and validation.
    std::vector<double> fullpos;
    std::vector<double> fullvel;
    if (rank == 0) {
        fullpos.resize(3 * static_cast<size_t>(numBodies));
        fullvel.resize(3 * static_cast<size_t>(numBodies));
    }

    if (printResults || validate) {
        MPI_Gatherv(localpos.data(), 3 * localN, MPI_DOUBLE,
                    rank == 0 ? fullpos.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        MPI_Gatherv(localvel.data(), 3 * localN, MPI_DOUBLE,
                    rank == 0 ? fullvel.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        std::vector<Body> bodies;
        if (printResults || validate) {
            bodies.resize(static_cast<size_t>(numBodies));
            for (int b = 0; b < numBodies; ++b) {
                bodies[b].pos.x = fullpos[3 * b];
                bodies[b].pos.y = fullpos[3 * b + 1];
                bodies[b].pos.z = fullpos[3 * b + 2];
                bodies[b].vel.x = fullvel[3 * b];
                bodies[b].vel.y = fullvel[3 * b + 1];
                bodies[b].vel.z = fullvel[3 * b + 2];
            }
        }

        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.reserve(static_cast<size_t>(numBodies) * 6);
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
        if (validate) {
            printf("Validating simulation results...\n");

            if (validateSimulation(bodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    if (validate) {
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (useGpu) {
        if (gpu.d_vel) cudaFree(gpu.d_vel);
        if (gpu.d_pos) cudaFree(gpu.d_pos);
    }

    MPI_Finalize();
    return exitCode;
}
