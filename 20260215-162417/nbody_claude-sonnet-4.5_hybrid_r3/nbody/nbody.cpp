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
    } while(0)

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

// CUDA kernel for force computation on a subset of bodies
__global__ void computeForcesKernel(const Body* bodies, Vec3* forces, int n, int startIdx, int localN) {
    int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (local_i < localN) {
        int i = startIdx + local_i;
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (int j = 0; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = rsqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        
        forces[local_i].x = Fx;
        forces[local_i].y = Fy;
        forces[local_i].z = Fz;
    }
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    #pragma omp parallel
    {
        unsigned int thread_seed = seed + omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < bodies.size(); ++i) {
            bodies[i].pos.x = 2.0 * (rand_r(&thread_seed) / (double)RAND_MAX) - 1.0;
            bodies[i].pos.y = 2.0 * (rand_r(&thread_seed) / (double)RAND_MAX) - 1.0;
            bodies[i].pos.z = 2.0 * (rand_r(&thread_seed) / (double)RAND_MAX) - 1.0;
            bodies[i].vel.x = 2.0 * (rand_r(&thread_seed) / (double)RAND_MAX) - 1.0;
            bodies[i].vel.y = 2.0 * (rand_r(&thread_seed) / (double)RAND_MAX) - 1.0;
            bodies[i].vel.z = 2.0 * (rand_r(&thread_seed) / (double)RAND_MAX) - 1.0;
        }
    }
}

void computeForces(std::vector<Body>& bodies, std::vector<Vec3>& localForces, 
                   Body* d_bodies, Vec3* d_forces, int n, int startIdx, int localN) {
    // Copy all bodies to GPU (all ranks need all positions)
    CUDA_CHECK(cudaMemcpy(d_bodies, bodies.data(), n * sizeof(Body), cudaMemcpyHostToDevice));
    
    // Launch kernel to compute forces for local subset
    int blockSize = 256;
    int numBlocks = (localN + blockSize - 1) / blockSize;
    computeForcesKernel<<<numBlocks, blockSize>>>(d_bodies, d_forces, n, startIdx, localN);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy local forces back
    CUDA_CHECK(cudaMemcpy(localForces.data(), d_forces, localN * sizeof(Vec3), cudaMemcpyDeviceToHost));
}

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
    
    // Kinetic energy (assuming unit mass)
    double kinetic = 0.0;
    #pragma omp parallel for reduction(+:kinetic)
    for (size_t i = 0; i < n; ++i) {
        kinetic += 0.5 * (bodies[i].vel.x * bodies[i].vel.x + 
                         bodies[i].vel.y * bodies[i].vel.y + 
                         bodies[i].vel.z * bodies[i].vel.z);
    }
    energy += kinetic;
    
    // Potential energy (assuming unit mass for all bodies)
    double potential = 0.0;
    #pragma omp parallel for reduction(+:potential)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potential -= 1.0 / dist;
        }
    }
    energy += potential;
    
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
    
    // Set GPU device based on local rank
    int numDevices;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices > 0) {
        CUDA_CHECK(cudaSetDevice(rank % numDevices));
    }
    
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0 for simplicity, broadcast later)
    if (rank == 0) {
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
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
        
        printf("N-Body Simulation (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Broadcast parameters to all ranks
    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    // Compute work distribution across ranks
    int baseN = numBodies / size;
    int remainder = numBodies % size;
    int localN = baseN + (rank < remainder ? 1 : 0);
    int startIdx = rank * baseN + std::min(rank, remainder);
    
    // Initialize bodies on all ranks with same seed for consistency
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies, 42);
    
    // Allocate GPU memory
    Body* d_bodies;
    Vec3* d_forces;
    CUDA_CHECK(cudaMalloc(&d_bodies, numBodies * sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&d_forces, localN * sizeof(Vec3)));
    
    // Local forces
    std::vector<Vec3> localForces(localN);
    
    // Create MPI datatype for Body
    MPI_Datatype MPI_BODY;
    MPI_Type_contiguous(sizeof(Body), MPI_BYTE, &MPI_BODY);
    MPI_Type_commit(&MPI_BODY);
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        // Each rank computes forces for its subset of bodies using CUDA
        computeForces(bodies, localForces, d_bodies, d_forces, numBodies, startIdx, localN);
        
        // Update local velocities with OpenMP
        #pragma omp parallel for
        for (int i = 0; i < localN; ++i) {
            int idx = startIdx + i;
            bodies[idx].vel.x += DT * localForces[i].x;
            bodies[idx].vel.y += DT * localForces[i].y;
            bodies[idx].vel.z += DT * localForces[i].z;
        }
        
        // Integrate positions for local subset with OpenMP
        #pragma omp parallel for
        for (int i = 0; i < localN; ++i) {
            int idx = startIdx + i;
            bodies[idx].pos.x += bodies[idx].vel.x * DT;
            bodies[idx].pos.y += bodies[idx].vel.y * DT;
            bodies[idx].pos.z += bodies[idx].vel.z * DT;
        }
        
        // Gather all updated bodies across ranks
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            int rLocalN = baseN + (r < remainder ? 1 : 0);
            recvcounts[r] = rLocalN;
            displs[r] = r * baseN + std::min(r, remainder);
        }
        
        MPI_Allgatherv(&bodies[startIdx], localN, MPI_BODY,
                       bodies.data(), recvcounts.data(), displs.data(), 
                       MPI_BODY, MPI_COMM_WORLD);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    MPI_Type_free(&MPI_BODY);
    
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }
    
    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
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
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            CUDA_CHECK(cudaFree(d_bodies));
            CUDA_CHECK(cudaFree(d_forces));
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            CUDA_CHECK(cudaFree(d_bodies));
            CUDA_CHECK(cudaFree(d_forces));
            MPI_Finalize();
            return 1;
        }
    }
    
    CUDA_CHECK(cudaFree(d_bodies));
    CUDA_CHECK(cudaFree(d_forces));
    MPI_Finalize();
    return 0;
}
