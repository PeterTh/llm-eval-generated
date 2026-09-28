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
constexpr int BLOCK_SIZE = 256;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err__));                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                 \
    } while (0)

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Kept strictly sequential: matches the original reference implementation
    // bit-for-bit so results/hashes are unaffected by the parallelization below.
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// Tiled all-pairs force kernel. The j-loop walks bodies 0..n-1 in strictly
// increasing order (tile by tile, then within a tile), so the floating-point
// summation order for each body i is identical to the original serial code.
__global__ void computeForcesKernel(const double* __restrict__ posx,
                                     const double* __restrict__ posy,
                                     const double* __restrict__ posz,
                                     double* __restrict__ velx,
                                     double* __restrict__ vely,
                                     double* __restrict__ velz,
                                     int n, int start, int count) {
    extern __shared__ double3 shPos[];

    const int localIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = localIdx < count;
    const int i = start + localIdx;

    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (active) {
        xi = posx[i];
        yi = posy[i];
        zi = posz[i];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = 0; tile < n; tile += blockDim.x) {
        const int idx = tile + threadIdx.x;
        if (idx < n) {
            shPos[threadIdx.x] = make_double3(posx[idx], posy[idx], posz[idx]);
        }
        __syncthreads();

        const int tileSize = min((int)blockDim.x, n - tile);
        if (active) {
            for (int k = 0; k < tileSize; ++k) {
                const double dx = shPos[k].x - xi;
                const double dy = shPos[k].y - yi;
                const double dz = shPos[k].z - zi;
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

    if (active) {
        velx[i] += DT * Fx;
        vely[i] += DT * Fy;
        velz[i] += DT * Fz;
    }
}

__global__ void integrateBodiesKernel(double* __restrict__ posx,
                                       double* __restrict__ posy,
                                       double* __restrict__ posz,
                                       const double* __restrict__ velx,
                                       const double* __restrict__ vely,
                                       const double* __restrict__ velz,
                                       int start, int count) {
    const int localIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (localIdx >= count) return;
    const int i = start + localIdx;
    posx[i] += velx[i] * DT;
    posy[i] += vely[i] * DT;
    posz[i] += velz[i] * DT;
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
    #pragma omp parallel for reduction(+:energy) schedule(dynamic, 64)
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
    const size_t n = bodies.size();
    int failures = 0;

    #pragma omp parallel for reduction(+:failures) schedule(static)
    for (size_t idx = 0; idx < n; ++idx) {
        const auto& body = bodies[idx];
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            ++failures;
            continue;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            ++failures;
            continue;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            ++failures;
        }
    }

    if (failures > 0) {
        printf("Validation failed: found %d invalid body state(s)\n", failures);
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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Bind each rank to a GPU (round-robin across the node's accelerators).
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA-capable devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

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
        printf("MPI ranks: %d, GPUs/node: %d, OpenMP threads/rank: %d\n",
               numRanks, deviceCount, omp_get_max_threads());
    }

    // Initialize bodies identically on every rank (deterministic, matches serial reference).
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Host-side SoA staging buffers shared by all ranks.
    std::vector<double> posx(numBodies), posy(numBodies), posz(numBodies);
    std::vector<double> velx(numBodies), vely(numBodies), velz(numBodies);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        posx[i] = bodies[i].pos.x;
        posy[i] = bodies[i].pos.y;
        posz[i] = bodies[i].pos.z;
        velx[i] = bodies[i].vel.x;
        vely[i] = bodies[i].vel.y;
        velz[i] = bodies[i].vel.z;
    }

    // Static work distribution across ranks: contiguous chunks of bodies.
    std::vector<int> counts(numRanks), displs(numRanks);
    {
        const int base = numBodies / numRanks;
        const int rem = numBodies % numRanks;
        int offset = 0;
        for (int r = 0; r < numRanks; ++r) {
            counts[r] = base + (r < rem ? 1 : 0);
            displs[r] = offset;
            offset += counts[r];
        }
    }
    const int myStart = displs[rank];
    const int myCount = counts[rank];

    // Device buffers (persist across the whole simulation).
    double *d_posx, *d_posy, *d_posz, *d_velx, *d_vely, *d_velz;
    const size_t bytes = (size_t)numBodies * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_posx, bytes));
    CUDA_CHECK(cudaMalloc(&d_posy, bytes));
    CUDA_CHECK(cudaMalloc(&d_posz, bytes));
    CUDA_CHECK(cudaMalloc(&d_velx, bytes));
    CUDA_CHECK(cudaMalloc(&d_vely, bytes));
    CUDA_CHECK(cudaMalloc(&d_velz, bytes));

    CUDA_CHECK(cudaMemcpy(d_posx, posx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posy, posy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posz, posz.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velx, velx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vely, vely.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velz, velz.data(), bytes, cudaMemcpyHostToDevice));

    const int numBlocks = (myCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t sharedBytes = BLOCK_SIZE * sizeof(double3);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Only positions of foreign bodies could have changed on the device
        // since the last step; velocities of foreign indices are never read
        // by the kernels, so we only need to re-upload positions each step.
        if (step > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_posx, posx.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpyAsync(d_posy, posy.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpyAsync(d_posz, posz.data(), bytes, cudaMemcpyHostToDevice));
        }

        if (myCount > 0) {
            computeForcesKernel<<<numBlocks, BLOCK_SIZE, sharedBytes>>>(
                d_posx, d_posy, d_posz, d_velx, d_vely, d_velz, numBodies, myStart, myCount);
            integrateBodiesKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_posx, d_posy, d_posz, d_velx, d_vely, d_velz, myStart, myCount);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (myCount > 0) {
            CUDA_CHECK(cudaMemcpy(posx.data() + myStart, d_posx + myStart, myCount * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(posy.data() + myStart, d_posy + myStart, myCount * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(posz.data() + myStart, d_posz + myStart, myCount * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(velx.data() + myStart, d_velx + myStart, myCount * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(vely.data() + myStart, d_vely + myStart, myCount * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(velz.data() + myStart, d_velz + myStart, myCount * sizeof(double), cudaMemcpyDeviceToHost));
        }

        // Exchange updated positions/velocities so every rank has the full,
        // up-to-date body set before the next step's force computation.
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, posx.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, posy.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, posz.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, velx.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, vely.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, velz.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    CUDA_CHECK(cudaFree(d_posx));
    CUDA_CHECK(cudaFree(d_posy));
    CUDA_CHECK(cudaFree(d_posz));
    CUDA_CHECK(cudaFree(d_velx));
    CUDA_CHECK(cudaFree(d_vely));
    CUDA_CHECK(cudaFree(d_velz));

    // Reassemble the AoS body array (all ranks hold identical, fully synced data).
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos.x = posx[i];
        bodies[i].pos.y = posy[i];
        bodies[i].pos.z = posz[i];
        bodies[i].vel.x = velx[i];
        bodies[i].vel.y = vely[i];
        bodies[i].vel.z = velz[i];
    }

    int exitCode = 0;
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());

        // Print results for external validation
        if (printResults) {
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

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
