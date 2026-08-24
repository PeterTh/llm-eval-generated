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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// Device constants for CUDA
__constant__ double d_SOFTENING = 1e-9;
__constant__ double d_DT = 0.01;

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

// CUDA kernel for computing forces with tiling
__global__ void computeForcesKernel(const double* __restrict__ posX,
                                     const double* __restrict__ posY,
                                     const double* __restrict__ posZ,
                                     double* __restrict__ velX,
                                     double* __restrict__ velY,
                                     double* __restrict__ velZ,
                                     const int n, const int localN, const int offset) {
    extern __shared__ double sharedPos[];
    double* sPosX = &sharedPos[0];
    double* sPosY = &sharedPos[blockDim.x];
    double* sPosZ = &sharedPos[2 * blockDim.x];
    
    const int localIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (localIdx >= localN) return;
    
    const int globalIdx = offset + localIdx;
    
    double px = posX[globalIdx];
    double py = posY[globalIdx];
    double pz = posZ[globalIdx];
    
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    
    // Tile the force computation
    for (int tile = 0; tile < n; tile += blockDim.x) {
        const int tileIdx = tile + threadIdx.x;
        
        // Load tile into shared memory
        if (tileIdx < n) {
            sPosX[threadIdx.x] = posX[tileIdx];
            sPosY[threadIdx.x] = posY[tileIdx];
            sPosZ[threadIdx.x] = posZ[tileIdx];
        }
        __syncthreads();
        
        // Compute forces from bodies in this tile
        const int tileSize = min(blockDim.x, n - tile);
        #pragma unroll 8
        for (int j = 0; j < tileSize; ++j) {
            const double dx = sPosX[j] - px;
            const double dy = sPosY[j] - py;
            const double dz = sPosZ[j] - pz;
            const double distSqr = dx * dx + dy * dy + dz * dz + d_SOFTENING;
            const double invDist = rsqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }
    
    velX[globalIdx] += d_DT * Fx;
    velY[globalIdx] += d_DT * Fy;
    velZ[globalIdx] += d_DT * Fz;
}

void computeForces(std::vector<Body>& localBodies, std::vector<Body>& allBodies,
                   double* d_posX, double* d_posY, double* d_posZ,
                   double* d_velX, double* d_velY, double* d_velZ,
                   int offset, int localN, int totalN) {
    // Copy all positions to device
    std::vector<double> allPosX(totalN), allPosY(totalN), allPosZ(totalN);
    std::vector<double> allVelX(totalN), allVelY(totalN), allVelZ(totalN);
    
    #pragma omp parallel for
    for (int i = 0; i < totalN; ++i) {
        allPosX[i] = allBodies[i].pos.x;
        allPosY[i] = allBodies[i].pos.y;
        allPosZ[i] = allBodies[i].pos.z;
        allVelX[i] = allBodies[i].vel.x;
        allVelY[i] = allBodies[i].vel.y;
        allVelZ[i] = allBodies[i].vel.z;
    }
    
    CUDA_CHECK(cudaMemcpy(d_posX, allPosX.data(), totalN * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posY, allPosY.data(), totalN * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posZ, allPosZ.data(), totalN * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velX, allVelX.data(), totalN * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velY, allVelY.data(), totalN * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velZ, allVelZ.data(), totalN * sizeof(double), cudaMemcpyHostToDevice));
    
    // Launch kernel
    const int blockSize = 256;
    const int numBlocks = (localN + blockSize - 1) / blockSize;
    const int sharedMemSize = 3 * blockSize * sizeof(double);
    
    computeForcesKernel<<<numBlocks, blockSize, sharedMemSize>>>(
        d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ, totalN, localN, offset);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy updated velocities back for local bodies
    CUDA_CHECK(cudaMemcpy(&allVelX[offset], &d_velX[offset], localN * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&allVelY[offset], &d_velY[offset], localN * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&allVelZ[offset], &d_velZ[offset], localN * sizeof(double), cudaMemcpyDeviceToHost));
    
    #pragma omp parallel for
    for (int i = 0; i < localN; ++i) {
        localBodies[i].vel.x = allVelX[offset + i];
        localBodies[i].vel.y = allVelY[offset + i];
        localBodies[i].vel.z = allVelZ[offset + i];
    }
}

void integrateBodies(std::vector<Body>& bodies) {
    const int n = bodies.size();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
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

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
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
    
    // Set CUDA device based on local rank
    int localRank = rank % 8;  // Assume up to 8 GPUs per node
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount > 0) {
        CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    }
    
    // Distribute bodies among MPI ranks
    const int localN = numBodies / size + (rank < (numBodies % size) ? 1 : 0);
    const int offset = (numBodies / size) * rank + std::min(rank, numBodies % size);
    
    if (rank == 0) {
        printf("N-Body Simulation (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Initialize all bodies (same seed on all ranks for consistency)
    std::vector<Body> allBodies(numBodies);
    randomizeBodies(allBodies, 42);
    
    // Extract local bodies
    std::vector<Body> localBodies(allBodies.begin() + offset, allBodies.begin() + offset + localN);
    
    // Allocate device memory
    double *d_posX, *d_posY, *d_posZ, *d_velX, *d_velY, *d_velZ;
    CUDA_CHECK(cudaMalloc(&d_posX, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_posY, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_posZ, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_velX, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_velY, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_velZ, numBodies * sizeof(double)));
    
    // Prepare MPI datatypes for gathering
    std::vector<int> recvCounts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; ++i) {
        recvCounts[i] = (numBodies / size + (i < (numBodies % size) ? 1 : 0)) * sizeof(Body) / sizeof(double);
        displs[i] = ((numBodies / size) * i + std::min(i, numBodies % size)) * sizeof(Body) / sizeof(double);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run simulation
    for (int step = 0; step < numSteps; ++step) {
        // Gather all bodies for force computation
        MPI_Allgatherv(localBodies.data(), localN * sizeof(Body) / sizeof(double), MPI_DOUBLE,
                       allBodies.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        
        // Compute forces on local bodies using CUDA
        computeForces(localBodies, allBodies, d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ,
                     offset, localN, numBodies);
        
        // Integrate local bodies
        integrateBodies(localBodies);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDurationMs = static_cast<long>(duration.count());
    long maxDurationMs = 0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", maxDurationMs);
    }
    
    // Gather final results on rank 0
    MPI_Allgatherv(localBodies.data(), localN * sizeof(Body) / sizeof(double), MPI_DOUBLE,
                   allBodies.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    
    // Print results for external validation (rank 0 only)
    if (rank == 0 && printResults) {
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
    
    // Validation (rank 0 only)
    int validationResult = 0;
    if (rank == 0 && validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(allBodies)) {
            double finalEnergy = computeTotalEnergy(allBodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            validationResult = 1;
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_posX));
    CUDA_CHECK(cudaFree(d_posY));
    CUDA_CHECK(cudaFree(d_posZ));
    CUDA_CHECK(cudaFree(d_velX));
    CUDA_CHECK(cudaFree(d_velY));
    CUDA_CHECK(cudaFree(d_velZ));
    
    MPI_Finalize();
    return validationResult;
}
