#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel: compute forces on local bodies and integrate
// Each thread handles one local body, iterating over all N bodies
__global__ void computeForcesAndIntegrateKernel(
    const double* __restrict__ allPosX,
    const double* __restrict__ allPosY,
    const double* __restrict__ allPosZ,
    double* __restrict__ localVelX,
    double* __restrict__ localVelY,
    double* __restrict__ localVelZ,
    double* __restrict__ localPosX,
    double* __restrict__ localPosY,
    double* __restrict__ localPosZ,
    const int nLocal,
    const int nTotal,
    const double dt,
    const double softening)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nLocal) return;

    const double px = localPosX[i];
    const double py = localPosY[i];
    const double pz = localPosZ[i];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int j = 0; j < nTotal; ++j) {
        const double dx = allPosX[j] - px;
        const double dy = allPosY[j] - py;
        const double dz = allPosZ[j] - pz;
        const double distSqr = dx * dx + dy * dy + dz * dz + softening;
        const double invDist = rsqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    // Update velocity
    double vx = localVelX[i] + dt * Fx;
    double vy = localVelY[i] + dt * Fy;
    double vz = localVelZ[i] + dt * Fz;

    localVelX[i] = vx;
    localVelY[i] = vy;
    localVelZ[i] = vz;

    // Update position
    localPosX[i] = px + vx * dt;
    localPosY[i] = py + vy * dt;
    localPosZ[i] = pz + vz * dt;
}

// CUDA kernel for computing kinetic energy (local portion)
__global__ void computeKineticEnergyKernel(
    const double* __restrict__ velX,
    const double* __restrict__ velY,
    const double* __restrict__ velZ,
    double* __restrict__ partialEnergy,
    const int nLocal)
{
    extern __shared__ double sdata[];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    double energy = 0.0;
    if (i < nLocal) {
        double vx = velX[i], vy = velY[i], vz = velZ[i];
        energy = 0.5 * (vx * vx + vy * vy + vz * vz);
    }
    sdata[tid] = energy;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partialEnergy[blockIdx.x] = sdata[0];
}

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

// CPU fallback for computeTotalEnergy with OpenMP
double computeTotalEnergyCPU(const std::vector<Body>& bodies, int rank, int size) {
    const size_t n = bodies.size();
    double localKinetic = 0.0;

    #pragma omp parallel for reduction(+:localKinetic) schedule(static)
    for (size_t idx = 0; idx < n; ++idx) {
        localKinetic += 0.5 * (bodies[idx].vel.x * bodies[idx].vel.x +
                                bodies[idx].vel.y * bodies[idx].vel.y +
                                bodies[idx].vel.z * bodies[idx].vel.z);
    }

    // Potential energy: need all bodies - compute locally possible pairs
    // For simplicity, this is only called at end for validation on rank 0
    double localPotential = 0.0;
    #pragma omp parallel for reduction(+:localPotential) schedule(dynamic, 64)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            localPotential -= 1.0 / dist;
        }
    }

    return localKinetic + localPotential;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < bodies.size(); ++i) {
        const auto& body = bodies[i];
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            #pragma omp critical
            {
                printf("Validation failed: found NaN or Inf value in body state\n");
            }
            // Can't easily break from omp parallel for, use flag
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            #pragma omp critical
            {
                printf("Validation failed: body position exceeds reasonable bounds\n");
            }
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            #pragma omp critical
            {
                printf("Validation failed: body velocity exceeds reasonable bounds\n");
            }
        }
    }

    // Re-check serially for return value
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            return false;
        }
        if (std::abs(body.pos.x) > 1e6 || std::abs(body.pos.y) > 1e6 || std::abs(body.pos.z) > 1e6) return false;
        if (std::abs(body.vel.x) > 1e6 || std::abs(body.vel.y) > 1e6 || std::abs(body.vel.z) > 1e6) return false;
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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize CUDA on each rank
    int numGPUs = 0;
    cudaGetDeviceCount(&numGPUs);
    if (numGPUs == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Assign GPU round-robin based on local rank
    int localRank = rank % numGPUs;
    CUDA_CHECK(cudaSetDevice(localRank));

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse same args)
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
            printUsage(argv[0]);
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
        printf("MPI ranks: %d, GPUs: %d\n", size, numGPUs);
    }

    // Compute local body distribution
    int nLocal = numBodies / size;
    int remainder = numBodies % size;
    int localStart = rank * nLocal + std::min(rank, remainder);
    if (rank < remainder) nLocal++;

    // Rank 0 randomizes all bodies, then scatters
    std::vector<Body> localBodies(nLocal);

    if (size == 1) {
        // Single rank: randomize directly
        randomizeBodies(localBodies);
    } else {
        // Rank 0 generates all bodies
        std::vector<Body> allBodies;
        if (rank == 0) {
            allBodies.resize(numBodies);
            randomizeBodies(allBodies);
        }

        // Scatter counts and displacements
        std::vector<int> sendCounts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            int cnt = numBodies / size;
            if (r < remainder) cnt++;
            sendCounts[r] = cnt * sizeof(Body);
            int start = r * (numBodies / size) + std::min(r, remainder);
            displs[r] = start * sizeof(Body);
        }

        MPI_Scatterv(
            rank == 0 ? allBodies.data() : nullptr,
            sendCounts.data(), displs.data(), MPI_BYTE,
            localBodies.data(), nLocal * sizeof(Body), MPI_BYTE,
            0, MPI_COMM_WORLD
        );
    }

    // Allocate GPU memory for local bodies (SoA layout)
    double *d_localPosX, *d_localPosY, *d_localPosZ;
    double *d_localVelX, *d_localVelY, *d_localVelZ;

    CUDA_CHECK(cudaMalloc(&d_localPosX, nLocal * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_localPosY, nLocal * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_localPosZ, nLocal * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_localVelX, nLocal * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_localVelY, nLocal * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_localVelZ, nLocal * sizeof(double)));

    // Allocate GPU memory for all positions (for force computation)
    double *d_allPosX, *d_allPosY, *d_allPosZ;
    CUDA_CHECK(cudaMalloc(&d_allPosX, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_allPosY, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_allPosZ, numBodies * sizeof(double)));

    // Copy local bodies to GPU
    {
        std::vector<double> posX(nLocal), posY(nLocal), posZ(nLocal);
        std::vector<double> velX(nLocal), velY(nLocal), velZ(nLocal);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nLocal; ++i) {
            posX[i] = localBodies[i].pos.x;
            posY[i] = localBodies[i].pos.y;
            posZ[i] = localBodies[i].pos.z;
            velX[i] = localBodies[i].vel.x;
            velY[i] = localBodies[i].vel.y;
            velZ[i] = localBodies[i].vel.z;
        }
        CUDA_CHECK(cudaMemcpy(d_localPosX, posX.data(), nLocal * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_localPosY, posY.data(), nLocal * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_localPosZ, posZ.data(), nLocal * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_localVelX, velX.data(), nLocal * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_localVelY, velY.data(), nLocal * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_localVelZ, velZ.data(), nLocal * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Host buffers for MPI_Allgather
    std::vector<double> localPosHostX(nLocal), localPosHostY(nLocal), localPosHostZ(nLocal);
    std::vector<double> allPosHostX(numBodies), allPosHostY(numBodies), allPosHostZ(numBodies);

    // MPI distribution info for Allgather
    std::vector<int> recvCounts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        int cnt = numBodies / size;
        if (r < remainder) cnt++;
        recvCounts[r] = cnt;
        int start = r * (numBodies / size) + std::min(r, remainder);
        displs[r] = start;
    }

    // CUDA kernel launch config
    const int blockSize = 256;
    const int numBlocks = (nLocal + blockSize - 1) / blockSize;

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Copy local positions from GPU to host
        CUDA_CHECK(cudaMemcpy(localPosHostX.data(), d_localPosX, nLocal * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localPosHostY.data(), d_localPosY, nLocal * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localPosHostZ.data(), d_localPosZ, nLocal * sizeof(double), cudaMemcpyDeviceToHost));

        // MPI_Allgather positions
        MPI_Allgatherv(
            localPosHostX.data(), nLocal, MPI_DOUBLE,
            allPosHostX.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
            MPI_COMM_WORLD
        );
        MPI_Allgatherv(
            localPosHostY.data(), nLocal, MPI_DOUBLE,
            allPosHostY.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
            MPI_COMM_WORLD
        );
        MPI_Allgatherv(
            localPosHostZ.data(), nLocal, MPI_DOUBLE,
            allPosHostZ.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
            MPI_COMM_WORLD
        );

        // Copy all positions to GPU
        CUDA_CHECK(cudaMemcpy(d_allPosX, allPosHostX.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_allPosY, allPosHostY.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_allPosZ, allPosHostZ.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));

        // Launch CUDA kernel: compute forces + integrate
        computeForcesAndIntegrateKernel<<<numBlocks, blockSize>>>(
            d_allPosX, d_allPosY, d_allPosZ,
            d_localVelX, d_localVelY, d_localVelZ,
            d_localPosX, d_localPosY, d_localPosZ,
            nLocal, numBodies, DT, SOFTENING
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long globalDuration = 0;
    MPI_Reduce(&localDuration, &globalDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", globalDuration);
    }

    // Copy results back from GPU to host
    {
        std::vector<double> velX(nLocal), velY(nLocal), velZ(nLocal);
        std::vector<double> posX(nLocal), posY(nLocal), posZ(nLocal);
        CUDA_CHECK(cudaMemcpy(posX.data(), d_localPosX, nLocal * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(posY.data(), d_localPosY, nLocal * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(posZ.data(), d_localPosZ, nLocal * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velX.data(), d_localVelX, nLocal * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velY.data(), d_localVelY, nLocal * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(velZ.data(), d_localVelZ, nLocal * sizeof(double), cudaMemcpyDeviceToHost));

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nLocal; ++i) {
            localBodies[i].pos.x = posX[i];
            localBodies[i].pos.y = posY[i];
            localBodies[i].pos.z = posZ[i];
            localBodies[i].vel.x = velX[i];
            localBodies[i].vel.y = velY[i];
            localBodies[i].vel.z = velZ[i];
        }
    }

    // Gather all bodies to rank 0 for output
    std::vector<Body> allBodies;
    if (rank == 0) allBodies.resize(numBodies);

    {
        std::vector<int> byteCounts(size), byteDispls(size);
        for (int r = 0; r < size; ++r) {
            int cnt = numBodies / size;
            if (r < remainder) cnt++;
            byteCounts[r] = cnt * sizeof(Body);
            int st = r * (numBodies / size) + std::min(r, remainder);
            byteDispls[r] = st * sizeof(Body);
        }

        MPI_Gatherv(
            localBodies.data(), nLocal * sizeof(Body), MPI_BYTE,
            rank == 0 ? allBodies.data() : nullptr,
            byteCounts.data(), byteDispls.data(), MPI_BYTE,
            0, MPI_COMM_WORLD
        );
    }

    // Print results for external validation
    if (printResults && rank == 0) {
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

    // Validation
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");

        if (validateSimulation(allBodies)) {
            double finalEnergy = computeTotalEnergyCPU(allBodies, rank, size);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // Cleanup GPU memory
    CUDA_CHECK(cudaFree(d_localPosX));
    CUDA_CHECK(cudaFree(d_localPosY));
    CUDA_CHECK(cudaFree(d_localPosZ));
    CUDA_CHECK(cudaFree(d_localVelX));
    CUDA_CHECK(cudaFree(d_localVelY));
    CUDA_CHECK(cudaFree(d_localVelZ));
    CUDA_CHECK(cudaFree(d_allPosX));
    CUDA_CHECK(cudaFree(d_allPosY));
    CUDA_CHECK(cudaFree(d_allPosZ));

    MPI_Finalize();
    return 0;
}
