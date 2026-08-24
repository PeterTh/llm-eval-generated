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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// CUDA Kernel for force calculation
__global__ void computeForcesKernel(Body* bodies, const Vec3* allPositions, int numBodies, int numAllBodies, double dt, double softening) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    Vec3 myPos = bodies[i].pos;

    for (int j = 0; j < numAllBodies; ++j) {
        const double dx = allPositions[j].x - myPos.x;
        const double dy = allPositions[j].y - myPos.y;
        const double dz = allPositions[j].z - myPos.z;
        const double distSqr = dx * dx + dy * dy + dz * dz + softening;
        const double invDist = 1.0 / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    bodies[i].vel.x += dt * Fx;
    bodies[i].vel.y += dt * Fy;
    bodies[i].vel.z += dt * Fz;
}

// CUDA Kernel for integration
__global__ void integrateBodiesKernel(Body* bodies, int numBodies, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    bodies[i].pos.x += bodies[i].vel.x * dt;
    bodies[i].pos.y += bodies[i].vel.y * dt;
    bodies[i].pos.z += bodies[i].vel.z * dt;
}

__global__ void extractPositionsKernel(Body* bodies, Vec3* positions, int numBodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numBodies) {
        positions[i] = bodies[i].pos;
    }
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Generate all bodies on all ranks to ensure consistency with original serial code
    for (size_t i = 0; i < bodies.size(); ++i) {
        bodies[i].pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// Original computeForces for validation (CPU)
void computeForces(std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (size_t j = 0; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        
        bodies[i].vel.x += DT * Fx;
        bodies[i].vel.y += DT * Fy;
        bodies[i].vel.z += DT * Fz;
    }
}

// Original integrateBodies for validation (CPU)
void integrateBodies(std::vector<Body>& bodies) {
    #pragma omp parallel for
    for (size_t i = 0; i < bodies.size(); ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy
    #pragma omp parallel for reduction(+:energy)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies[i].vel.x * bodies[i].vel.x + 
                        bodies[i].vel.y * bodies[i].vel.y + 
                        bodies[i].vel.z * bodies[i].vel.z);
    }
    
    // Potential energy
    #pragma omp parallel for reduction(+:energy) schedule(dynamic)
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("N-Body Simulation (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI Ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Assign Bodies to MPI Ranks
    int bodiesPerRank = numBodies / size;
    int startIdx = rank * bodiesPerRank;
    int endIdx = startIdx + bodiesPerRank;
    if (rank == size - 1) {
        endIdx = numBodies; // Handle remainder
    }
    int myNumBodies = endIdx - startIdx;

    // Initialize all bodies on all ranks (simplest for consistent initial state)
    // In production, we would scatter, but generation is fast enough.
    std::vector<Body> allBodies(numBodies);
    randomizeBodies(allBodies);

    // Prepare GPU data
    Body* d_myBodies;
    Vec3* d_allPositions;
    Vec3* d_myPositions; // Buffer for my positions on GPU
    Vec3* h_allPositions; // Pinned memory for transfer
    Vec3* h_myPositions; // Pinned memory for sending

    CUDA_CHECK(cudaMalloc(&d_myBodies, myNumBodies * sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&d_allPositions, numBodies * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_myPositions, myNumBodies * sizeof(Vec3)));
    CUDA_CHECK(cudaMallocHost(&h_allPositions, numBodies * sizeof(Vec3)));
    CUDA_CHECK(cudaMallocHost(&h_myPositions, myNumBodies * sizeof(Vec3)));

    // Copy initial state to GPU (only my bodies)
    CUDA_CHECK(cudaMemcpy(d_myBodies, &allBodies[startIdx], myNumBodies * sizeof(Body), cudaMemcpyHostToDevice));

    // Fill initial positions buffer
    #pragma omp parallel for
    for (int i = 0; i < numBodies; ++i) {
        h_allPositions[i] = allBodies[i].pos;
    }
    
    // We already have all positions on all ranks initially, so copy to GPU
    CUDA_CHECK(cudaMemcpy(d_allPositions, h_allPositions, numBodies * sizeof(Vec3), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        // 1. Compute forces on GPU
        int blockSize = 256;
        int gridSize = (myNumBodies + blockSize - 1) / blockSize;
        computeForcesKernel<<<gridSize, blockSize>>>(d_myBodies, d_allPositions, myNumBodies, numBodies, DT, SOFTENING);
        CUDA_CHECK(cudaGetLastError());

        // 2. Integrate on GPU
        integrateBodiesKernel<<<gridSize, blockSize>>>(d_myBodies, myNumBodies, DT);
        CUDA_CHECK(cudaGetLastError());
        
        // 3. Extract positions on GPU
        extractPositionsKernel<<<gridSize, blockSize>>>(d_myBodies, d_myPositions, myNumBodies);
        CUDA_CHECK(cudaGetLastError());

        // 4. Copy my positions to host pinned memory
        CUDA_CHECK(cudaMemcpy(h_myPositions, d_myPositions, myNumBodies * sizeof(Vec3), cudaMemcpyDeviceToHost));
        
        // Ensure data is ready for MPI
        CUDA_CHECK(cudaDeviceSynchronize()); 

        // 5. MPI Allgatherv
        std::vector<int> recvCountsBytes(size);
        std::vector<int> displsBytes(size);
        int currentDisp = 0;
        for (int r = 0; r < size; ++r) {
            int count = (numBodies / size) + (r == size - 1 ? numBodies % size : 0);
            recvCountsBytes[r] = count * sizeof(Vec3);
            displsBytes[r] = currentDisp;
            currentDisp += recvCountsBytes[r];
        }

        MPI_Allgatherv(h_myPositions, myNumBodies * sizeof(Vec3), MPI_BYTE,
                       h_allPositions, recvCountsBytes.data(), displsBytes.data(), MPI_BYTE,
                       MPI_COMM_WORLD);

        // Copy new allPositions to GPU for next step
        CUDA_CHECK(cudaMemcpy(d_allPositions, h_allPositions, numBodies * sizeof(Vec3), cudaMemcpyHostToDevice));
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = duration.count();
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", max_duration_ms);
    }
    
    // Gather final state for validation/output
    // Each rank has its updated bodies in myBodiesUpdated (from last step's loop or copy now)
    std::vector<Body> myBodiesFinal(myNumBodies);
    CUDA_CHECK(cudaMemcpy(myBodiesFinal.data(), d_myBodies, myNumBodies * sizeof(Body), cudaMemcpyDeviceToHost));

    // We need to gather the full Body structs now
    std::vector<Body> finalBodies(numBodies);
    
    // Gatherv
    std::vector<int> recvCountsBodies(size);
    std::vector<int> displsBodies(size);
    int currentDisp = 0;
    for (int r = 0; r < size; ++r) {
        int count = (numBodies / size) + (r == size - 1 ? numBodies % size : 0);
        recvCountsBodies[r] = count * sizeof(Body);
        displsBodies[r] = currentDisp;
        currentDisp += recvCountsBodies[r];
    }
    
    MPI_Allgatherv(myBodiesFinal.data(), myNumBodies * sizeof(Body), MPI_BYTE,
                   finalBodies.data(), recvCountsBodies.data(), displsBodies.data(), MPI_BYTE,
                   MPI_COMM_WORLD);

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (const auto& body : finalBodies) {
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
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(finalBodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(finalBodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            // Don't exit with 1 here to avoid aborting MPI if other ranks are waiting? 
            // Only rank 0 does this.
        }
    }
    
    CUDA_CHECK(cudaFree(d_myBodies));
    CUDA_CHECK(cudaFree(d_allPositions));
    CUDA_CHECK(cudaFree(d_myPositions));
    CUDA_CHECK(cudaFreeHost(h_allPositions));
    CUDA_CHECK(cudaFreeHost(h_myPositions));

    MPI_Finalize();
    return 0;
}
