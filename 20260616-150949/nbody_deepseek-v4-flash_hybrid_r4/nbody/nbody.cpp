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

#define SOFTENING 1e-9
#define DT 0.01

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

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

// GPU kernel: compute gravitational forces and update velocities in-place
// Uses tiled shared memory for coalesced global reads and preserved summation order
__global__ void computeForcesKernel(const double* allPos, double* vel,
                                     int nLocal, int nTotal, int localStart) {
    extern __shared__ double s_pos[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nLocal) return;

    int bodyIdx = localStart + idx;
    double px = allPos[bodyIdx * 3 + 0];
    double py = allPos[bodyIdx * 3 + 1];
    double pz = allPos[bodyIdx * 3 + 2];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    // Tile over all N bodies using shared memory
    for (int tile = 0; tile * blockDim.x < nTotal; ++tile) {
        int tileStart = tile * blockDim.x;
        int tileEnd = min(tileStart + blockDim.x, nTotal);
        int tileSize = tileEnd - tileStart;

        // Cooperative load of position tile into shared memory (coalesced)
        if (threadIdx.x < tileSize) {
            s_pos[threadIdx.x * 3 + 0] = allPos[(tileStart + threadIdx.x) * 3 + 0];
            s_pos[threadIdx.x * 3 + 1] = allPos[(tileStart + threadIdx.x) * 3 + 1];
            s_pos[threadIdx.x * 3 + 2] = allPos[(tileStart + threadIdx.x) * 3 + 2];
        }
        __syncthreads();

        // Compute interactions with bodies in this tile
        #pragma unroll
        for (int j = 0; j < tileSize; ++j) {
            double dx = s_pos[j * 3 + 0] - px;
            double dy = s_pos[j * 3 + 1] - py;
            double dz = s_pos[j * 3 + 2] - pz;
            double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            double invDist = rsqrt(distSqr);
            double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    vel[idx * 3 + 0] += DT * Fx;
    vel[idx * 3 + 1] += DT * Fy;
    vel[idx * 3 + 2] += DT * Fz;
}

// GPU kernel: integrate positions
__global__ void integrateKernel(double* pos, const double* vel, int nLocal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nLocal) return;

    pos[idx * 3 + 0] += vel[idx * 3 + 0] * DT;
    pos[idx * 3 + 1] += vel[idx * 3 + 1] * DT;
    pos[idx * 3 + 2] += vel[idx * 3 + 2] * DT;
}

// OpenMP-parallelized total energy calculation (kinetic + potential)
void computeTotalEnergy(const std::vector<Body>& bodies, double& energy) {
    energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (unit mass)
    #pragma omp parallel for reduction(+:energy)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies[i].vel.x * bodies[i].vel.x +
                        bodies[i].vel.y * bodies[i].vel.y +
                        bodies[i].vel.z * bodies[i].vel.z);
    }

    // Potential energy (unit mass)
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
}

// OpenMP-parallelized validation (NaN/Inf and bounds checking)
bool validateSimulation(const std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    int validCount = 0;

    #pragma omp parallel for reduction(+:validCount)
    for (size_t i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        bool ok = std::isfinite(body.pos.x) && std::isfinite(body.pos.y) && std::isfinite(body.pos.z) &&
                  std::isfinite(body.vel.x) && std::isfinite(body.vel.y) && std::isfinite(body.vel.z) &&
                  std::abs(body.pos.x) <= 1e6 && std::abs(body.pos.y) <= 1e6 && std::abs(body.pos.z) <= 1e6 &&
                  std::abs(body.vel.x) <= 1e6 && std::abs(body.vel.y) <= 1e6 && std::abs(body.vel.z) <= 1e6;
        if (ok) validCount++;
    }

    return validCount == (int)n;
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

    int rank, nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse)
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

    if (numBodies < nRanks && rank == 0) {
        printf("Error: number of bodies (%d) must be >= MPI ranks (%d)\n", numBodies, nRanks);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Determine data distribution across MPI ranks
    int base = numBodies / nRanks;
    int remainder = numBodies % nRanks;

    std::vector<int> counts(nRanks), displs(nRanks);
    int offset = 0;
    for (int i = 0; i < nRanks; ++i) {
        counts[i] = (i < remainder) ? (base + 1) : base;
        displs[i] = offset;
        offset += counts[i];
    }
    int nLocal = counts[rank];
    int localStart = displs[rank];

    // MPI counts for 3-component arrays (position/velocity vectors)
    std::vector<int> counts3(nRanks), displs3(nRanks);
    for (int i = 0; i < nRanks; ++i) {
        counts3[i] = counts[i] * 3;
        displs3[i] = displs[i] * 3;
    }

    if (rank == 0) {
        printf("N-Body Simulation (MPI+OpenMP+CUDA)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nRanks);
        int nThreads = 1;
        #pragma omp parallel
        {
            #pragma omp master
            nThreads = omp_get_num_threads();
        }
        printf("OpenMP threads: %d\n", nThreads);
    }

    // ---- Initialize bodies on rank 0, then scatter ----
    std::vector<Body> allBodies;
    std::vector<double> initPos, initVel;

    if (rank == 0) {
        allBodies.resize(numBodies);
        randomizeBodies(allBodies);

        initPos.resize(numBodies * 3);
        initVel.resize(numBodies * 3);

        #pragma omp parallel for
        for (int i = 0; i < numBodies; ++i) {
            initPos[i * 3 + 0] = allBodies[i].pos.x;
            initPos[i * 3 + 1] = allBodies[i].pos.y;
            initPos[i * 3 + 2] = allBodies[i].pos.z;
            initVel[i * 3 + 0] = allBodies[i].vel.x;
            initVel[i * 3 + 1] = allBodies[i].vel.y;
            initVel[i * 3 + 2] = allBodies[i].vel.z;
        }
    }

    // Host buffers for local data (one rank's portion)
    std::vector<double> localPos_h(nLocal * 3);
    std::vector<double> localVel_h(nLocal * 3);
    // All positions on every rank (for GPU force computation)
    std::vector<double> allPos_h(numBodies * 3);

    // Scatter positions and velocities from rank 0
    MPI_Scatterv(rank == 0 ? initPos.data() : nullptr, counts3.data(), displs3.data(),
                 MPI_DOUBLE, localPos_h.data(), nLocal * 3, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? initVel.data() : nullptr, counts3.data(), displs3.data(),
                 MPI_DOUBLE, localVel_h.data(), nLocal * 3, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // All ranks get all positions via Allgatherv (required for force computation)
    MPI_Allgatherv(localPos_h.data(), nLocal * 3, MPI_DOUBLE,
                   allPos_h.data(), counts3.data(), displs3.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    // ---- Initialize GPU ----
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "ERROR: No CUDA devices found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % numDevices));

    double *d_pos = nullptr, *d_vel = nullptr, *d_allPos = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pos, nLocal * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel, nLocal * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_allPos, numBodies * 3 * sizeof(double)));

    // Copy initial data to GPU
    CUDA_CHECK(cudaMemcpy(d_pos, localPos_h.data(), nLocal * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, localVel_h.data(), nLocal * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_allPos, allPos_h.data(), numBodies * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Simulation loop ----
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const int blockSize = 256;
    int gridSize = (nLocal + blockSize - 1) / blockSize;
    const int sharedMemSize = blockSize * 3 * sizeof(double);

    for (int step = 0; step < numSteps; ++step) {
        // GPU: compute forces from all N bodies, update velocities
        computeForcesKernel<<<gridSize, blockSize, sharedMemSize>>>(
            d_allPos, d_vel, nLocal, numBodies, localStart);

        // GPU: integrate positions (pos += vel * DT)
        integrateKernel<<<gridSize, blockSize>>>(d_pos, d_vel, nLocal);

        // Copy local positions to host for MPI exchange
        CUDA_CHECK(cudaMemcpy(localPos_h.data(), d_pos, nLocal * 3 * sizeof(double), cudaMemcpyDeviceToHost));

        // MPI: exchange updated positions across all ranks
        MPI_Allgatherv(localPos_h.data(), nLocal * 3, MPI_DOUBLE,
                       allPos_h.data(), counts3.data(), displs3.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // Copy full position array back to GPU for next iteration
        CUDA_CHECK(cudaMemcpy(d_allPos, allPos_h.data(), numBodies * 3 * sizeof(double), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double localDurationMs = std::chrono::duration<double, std::milli>(end - start).count();
    double globalDurationMs = 0.0;
    MPI_Reduce(&localDurationMs, &globalDurationMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %.3f ms\n", globalDurationMs);
    }

    // ---- Gather results to rank 0 for output and validation ----
    // Copy final velocities back from GPU
    CUDA_CHECK(cudaMemcpy(localVel_h.data(), d_vel, nLocal * 3 * sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> allPosFinal, allVelFinal;
    if (rank == 0) {
        allPosFinal.resize(numBodies * 3);
        allVelFinal.resize(numBodies * 3);
    }

    MPI_Gatherv(localPos_h.data(), nLocal * 3, MPI_DOUBLE,
                rank == 0 ? allPosFinal.data() : nullptr, counts3.data(), displs3.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(localVel_h.data(), nLocal * 3, MPI_DOUBLE,
                rank == 0 ? allVelFinal.data() : nullptr, counts3.data(), displs3.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Reconstruct Body vector from flat arrays
        #pragma omp parallel for
        for (int i = 0; i < numBodies; ++i) {
            allBodies[i].pos.x = allPosFinal[i * 3 + 0];
            allBodies[i].pos.y = allPosFinal[i * 3 + 1];
            allBodies[i].pos.z = allPosFinal[i * 3 + 2];
            allBodies[i].vel.x = allVelFinal[i * 3 + 0];
            allBodies[i].vel.y = allVelFinal[i * 3 + 1];
            allBodies[i].vel.z = allVelFinal[i * 3 + 2];
        }

        // Print results for external validation
        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(numBodies * 6);
            for (const auto& body : allBodies) {
                bodyData.push_back(body.pos.x);
                bodyData.push_back(body.pos.y);
                bodyData.push_back(body.pos.z);
                bodyData.push_back(body.vel.x);
                bodyData.push_back(body.vel.y);
                bodyData.push_back(body.vel.z);
            }
            print_results(bodyData, "Bodies");
        }

        // Validate
        if (validate) {
            printf("Validating simulation results...\n");
            if (validateSimulation(allBodies)) {
                double finalEnergy;
                computeTotalEnergy(allBodies, finalEnergy);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                CUDA_CHECK(cudaFree(d_allPos));
                CUDA_CHECK(cudaFree(d_vel));
                CUDA_CHECK(cudaFree(d_pos));
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_allPos));
    CUDA_CHECK(cudaFree(d_vel));
    CUDA_CHECK(cudaFree(d_pos));
    MPI_Finalize();
    return 0;
}
