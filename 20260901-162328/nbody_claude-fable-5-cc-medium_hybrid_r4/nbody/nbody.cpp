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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
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

// One simulation step for the local slice of bodies: accumulate forces from all
// n bodies (tiled through shared memory), update velocity, and write the new
// position to a separate buffer so the position array stays read-only within
// the kernel.
__global__ void bodyForceIntegrateKernel(const double* __restrict__ pos, double* __restrict__ vel,
                                         double* __restrict__ newPos, int n, int offset, int nLocal) {
    __shared__ double tile[BLOCK_SIZE * 3];

    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = li < nLocal;

    double px = 0.0, py = 0.0, pz = 0.0;
    if (active) {
        const int gi = offset + li;
        px = pos[3 * gi + 0];
        py = pos[3 * gi + 1];
        pz = pos[3 * gi + 2];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tileStart = 0; tileStart < n; tileStart += BLOCK_SIZE) {
        const int j = tileStart + threadIdx.x;
        if (j < n) {
            tile[3 * threadIdx.x + 0] = pos[3 * j + 0];
            tile[3 * threadIdx.x + 1] = pos[3 * j + 1];
            tile[3 * threadIdx.x + 2] = pos[3 * j + 2];
        }
        __syncthreads();

        if (active) {
            const int tileCount = min(BLOCK_SIZE, n - tileStart);
            #pragma unroll 8
            for (int t = 0; t < tileCount; ++t) {
                const double dx = tile[3 * t + 0] - px;
                const double dy = tile[3 * t + 1] - py;
                const double dz = tile[3 * t + 2] - pz;
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
        const double vx = vel[3 * li + 0] + DT * Fx;
        const double vy = vel[3 * li + 1] + DT * Fy;
        const double vz = vel[3 * li + 2] + DT * Fz;
        vel[3 * li + 0] = vx;
        vel[3 * li + 1] = vy;
        vel[3 * li + 2] = vz;
        newPos[3 * li + 0] = px + vx * DT;
        newPos[3 * li + 1] = py + vy * DT;
        newPos[3 * li + 2] = pz + vz * DT;
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
    const long n = (long)bodies.size();
    int nanCount = 0, posCount = 0, velCount = 0;

    #pragma omp parallel for reduction(+ : nanCount, posCount, velCount)
    for (long i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            ++nanCount;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            ++posCount;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            ++velCount;
        }
    }

    if (nanCount > 0) {
        printf("Validation failed: found NaN or Inf value in body state\n");
        return false;
    }
    if (posCount > 0) {
        printf("Validation failed: body position exceeds reasonable bounds\n");
        return false;
    }
    if (velCount > 0) {
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, nRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

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
        printf("MPI ranks: %d, OpenMP threads: %d\n", nRanks, omp_get_max_threads());
    }

    // Bind each rank to a GPU (round-robin over the node's devices)
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % numDevices));

    // Initialize bodies: every rank computes the identical deterministic
    // initial state, so no broadcast is needed.
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Contiguous slice of bodies owned by each rank
    std::vector<int> counts(nRanks), displs(nRanks);
    for (int r = 0; r < nRanks; ++r) {
        const int base = numBodies / nRanks;
        const int rem = numBodies % nRanks;
        counts[r] = 3 * (base + (r < rem ? 1 : 0));
        displs[r] = r == 0 ? 0 : displs[r - 1] + counts[r - 1];
    }
    const int nLocal = counts[rank] / 3;
    const int offset = displs[rank] / 3;

    // Host staging buffers (pinned for fast transfers): full interleaved
    // positions and the local slice of velocities.
    double* h_pos = nullptr;
    double* h_localPos = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pos, sizeof(double) * 3 * numBodies));
    CUDA_CHECK(cudaMallocHost(&h_localPos, sizeof(double) * 3 * (nLocal > 0 ? nLocal : 1)));
    std::vector<double> h_localVel(3 * (nLocal > 0 ? nLocal : 1));

    #pragma omp parallel for
    for (int i = 0; i < numBodies; ++i) {
        h_pos[3 * i + 0] = bodies[i].pos.x;
        h_pos[3 * i + 1] = bodies[i].pos.y;
        h_pos[3 * i + 2] = bodies[i].pos.z;
    }
    #pragma omp parallel for
    for (int i = 0; i < nLocal; ++i) {
        h_localVel[3 * i + 0] = bodies[offset + i].vel.x;
        h_localVel[3 * i + 1] = bodies[offset + i].vel.y;
        h_localVel[3 * i + 2] = bodies[offset + i].vel.z;
    }

    double *d_pos = nullptr, *d_vel = nullptr, *d_newPos = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pos, sizeof(double) * 3 * numBodies));
    CUDA_CHECK(cudaMalloc(&d_vel, sizeof(double) * 3 * (nLocal > 0 ? nLocal : 1)));
    CUDA_CHECK(cudaMalloc(&d_newPos, sizeof(double) * 3 * (nLocal > 0 ? nLocal : 1)));
    CUDA_CHECK(cudaMemcpy(d_pos, h_pos, sizeof(double) * 3 * numBodies, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, h_localVel.data(), sizeof(double) * 3 * nLocal, cudaMemcpyHostToDevice));

    const int numBlocks = (nLocal + BLOCK_SIZE - 1) / BLOCK_SIZE;

    MPI_Barrier(MPI_COMM_WORLD);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        if (nLocal > 0) {
            bodyForceIntegrateKernel<<<numBlocks, BLOCK_SIZE>>>(d_pos, d_vel, d_newPos,
                                                                numBodies, offset, nLocal);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_localPos, d_newPos, sizeof(double) * 3 * nLocal,
                                  cudaMemcpyDeviceToHost));
        }

        // Exchange updated positions so every rank sees the full system
        MPI_Allgatherv(h_localPos, counts[rank], MPI_DOUBLE, h_pos, counts.data(),
                       displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(d_pos, h_pos, sizeof(double) * 3 * numBodies,
                              cudaMemcpyHostToDevice));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Simulation time: %ld ms\n", maxDuration);

    // Collect final velocities on rank 0 (positions are already replicated)
    if (nLocal > 0) {
        CUDA_CHECK(cudaMemcpy(h_localVel.data(), d_vel, sizeof(double) * 3 * nLocal,
                              cudaMemcpyDeviceToHost));
    }
    std::vector<double> allVel(rank == 0 ? (size_t)3 * numBodies : 1);
    MPI_Gatherv(h_localVel.data(), counts[rank], MPI_DOUBLE, allVel.data(), counts.data(),
                displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int rc = 0;
    if (rank == 0) {
        #pragma omp parallel for
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos = Vec3(h_pos[3 * i + 0], h_pos[3 * i + 1], h_pos[3 * i + 2]);
            bodies[i].vel = Vec3(allVel[3 * i + 0], allVel[3 * i + 1], allVel[3 * i + 2]);
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
                rc = 1;
            }
        }
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_vel));
    CUDA_CHECK(cudaFree(d_newPos));
    CUDA_CHECK(cudaFreeHost(h_pos));
    CUDA_CHECK(cudaFreeHost(h_localPos));

    MPI_Finalize();
    return rc;
}
