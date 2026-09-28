// Hybrid MPI + OpenMP + CUDA n-body benchmark.
//
// Decomposition:
//  - MPI: bodies are block-distributed across ranks; each rank owns one GPU
//    (round-robin over the node's devices) and updates only its own slice.
//    Positions are exchanged with MPI_Allgatherv after each step.
//  - CUDA: the O(n^2) force computation and the integration of the local
//    slice run on the GPU (one thread per body, shared-memory tiling).
//  - OpenMP: host-side O(n^2) energy validation and other CPU loops.

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

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err_ = (call);                                                           \
        if (err_ != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,  \
                    __LINE__);                                                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                    \
        }                                                                                    \
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
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

constexpr int BLOCK_SIZE = 256;

// One thread per locally-owned body; all n positions are streamed through
// shared memory in tiles. The j-loop visits bodies in the same 0..n-1 order
// as the original serial code, preserving the per-body summation order.
__global__ void forceKernel(const double3* __restrict__ pos, double3* __restrict__ vel,
                            const int n, const int offset, const int localN) {
    __shared__ double3 tile[BLOCK_SIZE];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = idx < localN;

    double px = 0.0, py = 0.0, pz = 0.0;
    if (active) {
        const double3 p = pos[offset + idx];
        px = p.x;
        py = p.y;
        pz = p.z;
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int t = 0; t < n; t += BLOCK_SIZE) {
        const int j = t + threadIdx.x;
        if (j < n) {
            tile[threadIdx.x] = pos[j];
        }
        __syncthreads();

        const int m = min(BLOCK_SIZE, n - t);
        if (active) {
            for (int k = 0; k < m; ++k) {
                const double dx = tile[k].x - px;
                const double dy = tile[k].y - py;
                const double dz = tile[k].z - pz;
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
        vel[idx].x += DT * Fx;
        vel[idx].y += DT * Fy;
        vel[idx].z += DT * Fz;
    }
}

__global__ void integrateKernel(double3* __restrict__ pos, const double3* __restrict__ vel,
                                const int offset, const int localN) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < localN) {
        pos[offset + idx].x += vel[idx].x * DT;
        pos[offset + idx].y += vel[idx].y * DT;
        pos[offset + idx].z += vel[idx].z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const long n = (long)bodies.size();

    // Kinetic energy (assuming unit mass)
#pragma omp parallel for reduction(+ : energy)
    for (long i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : energy)
    for (long i = 0; i < n; ++i) {
        for (long j = i + 1; j < n; ++j) {
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
    bool ok = true;
#pragma omp parallel for reduction(&& : ok)
    for (long i = 0; i < (long)bodies.size(); ++i) {
        const auto& body = bodies[i];
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            ok = false;
            continue;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            ok = false;
            continue;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            ok = false;
        }
    }
    if (!ok) {
        printf("Validation failed: body state contains NaN/Inf or exceeds reasonable bounds\n");
    }
    return ok;
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

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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

    // Bind each rank to a GPU (round-robin over the devices visible on the node)
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % numDevices));

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs per node: %d, OpenMP threads: %d\n",
               numRanks, numDevices, omp_get_max_threads());
    }

    // Initialize bodies (identical deterministic sequence on every rank)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Block distribution of bodies across ranks
    std::vector<int> counts(numRanks), displs(numRanks);
    {
        const int base = numBodies / numRanks;
        const int rem = numBodies % numRanks;
        int off = 0;
        for (int r = 0; r < numRanks; ++r) {
            counts[r] = base + (r < rem ? 1 : 0);
            displs[r] = off;
            off += counts[r];
        }
    }
    const int localN = counts[rank];
    const int offset = displs[rank];

    // Element counts/offsets in units of doubles (3 per body) for Allgatherv
    std::vector<int> counts3(numRanks), displs3(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        counts3[r] = 3 * counts[r];
        displs3[r] = 3 * displs[r];
    }

    // Split AoS bodies into position/velocity arrays (pinned for fast DMA)
    double3 *hPos = nullptr, *hVel = nullptr;
    CUDA_CHECK(cudaMallocHost(&hPos, (size_t)numBodies * sizeof(double3)));
    CUDA_CHECK(cudaMallocHost(&hVel, (size_t)std::max(localN, 1) * sizeof(double3)));
#pragma omp parallel for
    for (int i = 0; i < numBodies; ++i) {
        hPos[i] = make_double3(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z);
    }
#pragma omp parallel for
    for (int i = 0; i < localN; ++i) {
        hVel[i] = make_double3(bodies[offset + i].vel.x, bodies[offset + i].vel.y,
                               bodies[offset + i].vel.z);
    }

    double3 *dPos = nullptr, *dVel = nullptr;
    CUDA_CHECK(cudaMalloc(&dPos, (size_t)numBodies * sizeof(double3)));
    CUDA_CHECK(cudaMalloc(&dVel, (size_t)std::max(localN, 1) * sizeof(double3)));
    CUDA_CHECK(cudaMemcpy(dPos, hPos, (size_t)numBodies * sizeof(double3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVel, hVel, (size_t)localN * sizeof(double3), cudaMemcpyHostToDevice));

    const int numBlocks = (std::max(localN, 1) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    for (int step = 0; step < numSteps; ++step) {
        if (localN > 0) {
            forceKernel<<<numBlocks, BLOCK_SIZE>>>(dPos, dVel, numBodies, offset, localN);
            integrateKernel<<<numBlocks, BLOCK_SIZE>>>(dPos, dVel, offset, localN);
            // Fetch only the locally updated positions for the exchange
            CUDA_CHECK(cudaMemcpy(hPos + offset, dPos + offset, (size_t)localN * sizeof(double3),
                                  cudaMemcpyDeviceToHost));
        }

        if (numRanks > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hPos, counts3.data(),
                           displs3.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            // Push the refreshed global positions back to the device
            CUDA_CHECK(cudaMemcpy(dPos, hPos, (size_t)numBodies * sizeof(double3),
                                  cudaMemcpyHostToDevice));
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Gather final velocities on rank 0 and reassemble the full body state
    CUDA_CHECK(cudaMemcpy(hVel, dVel, (size_t)localN * sizeof(double3), cudaMemcpyDeviceToHost));

    std::vector<double3> allVel(numBodies);
    memcpy(allVel.data() + offset, hVel, (size_t)localN * sizeof(double3));
    if (numRanks > 1) {
        MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : (const void*)(allVel.data() + offset),
                    counts3[rank], MPI_DOUBLE, allVel.data(), counts3.data(), displs3.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int exitCode = 0;

    if (rank == 0) {
#pragma omp parallel for
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos = Vec3(hPos[i].x, hPos[i].y, hPos[i].z);
            bodies[i].vel = Vec3(allVel[i].x, allVel[i].y, allVel[i].z);
        }

        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.reserve((size_t)numBodies * 6);
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

    CUDA_CHECK(cudaFree(dPos));
    CUDA_CHECK(cudaFree(dVel));
    CUDA_CHECK(cudaFreeHost(hPos));
    CUDA_CHECK(cudaFreeHost(hVel));

    MPI_Finalize();
    return exitCode;
}
