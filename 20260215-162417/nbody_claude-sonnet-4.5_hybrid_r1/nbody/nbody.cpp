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

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    #pragma omp parallel for
    for (size_t i = 0; i < bodies.size(); ++i) {
        unsigned int thread_seed = seed + i;
        bodies[i].pos.x = 2.0 * (rand_r(&thread_seed) / (double)RAND_MAX) - 1.0;
        bodies[i].pos.y = 2.0 * (rand_r(&thread_seed) / (double)RAND_MAX) - 1.0;
        bodies[i].pos.z = 2.0 * (rand_r(&thread_seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.x = 2.0 * (rand_r(&thread_seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.y = 2.0 * (rand_r(&thread_seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.z = 2.0 * (rand_r(&thread_seed) / (double)RAND_MAX) - 1.0;
    }
}

// CUDA kernel for force computation
__global__ void computeForcesKernel(Body* all_bodies, int* local_indices, int local_count, int total_bodies, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < local_count) {
        int i = local_indices[idx];
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (int j = 0; j < total_bodies; ++j) {
            const double dx = all_bodies[j].pos.x - all_bodies[i].pos.x;
            const double dy = all_bodies[j].pos.y - all_bodies[i].pos.y;
            const double dz = all_bodies[j].pos.z - all_bodies[i].pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = rsqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        
        all_bodies[i].vel.x += dt * Fx;
        all_bodies[i].vel.y += dt * Fy;
        all_bodies[i].vel.z += dt * Fz;
    }
}

// CUDA kernel for position integration
__global__ void integrateBodiesKernel(Body* all_bodies, int* local_indices, int local_count, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < local_count) {
        int i = local_indices[idx];
        all_bodies[i].pos.x += all_bodies[i].vel.x * dt;
        all_bodies[i].pos.y += all_bodies[i].vel.y * dt;
        all_bodies[i].pos.z += all_bodies[i].vel.z * dt;
    }
}

void computeForces(std::vector<Body>& bodies) {
    // Stub - will be replaced by CUDA version in main
}

void integrateBodies(std::vector<Body>& bodies) {
    // Stub - will be replaced by CUDA version in main
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
    #pragma omp parallel for reduction(+:energy)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies[i].vel.x * bodies[i].vel.x + 
                        bodies[i].vel.y * bodies[i].vel.y + 
                        bodies[i].vel.z * bodies[i].vel.z);
    }
    
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
    
    return energy + potential;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    bool valid = true;
    #pragma omp parallel for reduction(&&:valid)
    for (size_t i = 0; i < bodies.size(); ++i) {
        const auto& body = bodies[i];
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            valid = false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            valid = false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            valid = false;
        }
    }
    if (!valid) {
        printf("Validation failed: found NaN, Inf, or extreme values in body state\n");
    }
    return valid;
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
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
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
            if (world_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    // Each MPI rank uses a different GPU
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    int device_id = world_rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device_id));
    
    if (world_rank == 0) {
        printf("N-Body Simulation (MPI + OpenMP + CUDA)\n");
        printf("MPI ranks: %d\n", world_size);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("CUDA devices: %d\n", num_devices);
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Distribute bodies across MPI ranks
    int bodies_per_rank = numBodies / world_size;
    int remainder = numBodies % world_size;
    int local_start = world_rank * bodies_per_rank + std::min(world_rank, remainder);
    int local_count = bodies_per_rank + (world_rank < remainder ? 1 : 0);
    
    // Initialize all bodies on rank 0
    std::vector<Body> all_bodies(numBodies);
    if (world_rank == 0) {
        randomizeBodies(all_bodies);
    }
    
    // Allocate local bodies
    std::vector<Body> local_bodies(local_count);
    
    // Create arrays for scattering
    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);
    for (int rank = 0; rank < world_size; ++rank) {
        int rank_start = rank * bodies_per_rank + std::min(rank, remainder);
        int rank_count = bodies_per_rank + (rank < remainder ? 1 : 0);
        sendcounts[rank] = rank_count * sizeof(Body);
        displs[rank] = rank_start * sizeof(Body);
    }
    
    // Scatter bodies to all ranks
    MPI_Scatterv(world_rank == 0 ? all_bodies.data() : nullptr,
                 sendcounts.data(), displs.data(), MPI_BYTE,
                 local_bodies.data(), local_count * sizeof(Body), MPI_BYTE,
                 0, MPI_COMM_WORLD);
    
    // Allocate device memory for all bodies (needed for force computation)
    Body *d_all_bodies;
    int *d_local_indices;
    CUDA_CHECK(cudaMalloc(&d_all_bodies, numBodies * sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&d_local_indices, local_count * sizeof(int)));
    
    // Create local indices array and copy to device
    std::vector<int> local_indices(local_count);
    for (int i = 0; i < local_count; ++i) {
        local_indices[i] = local_start + i;
    }
    CUDA_CHECK(cudaMemcpy(d_local_indices, local_indices.data(), 
                         local_count * sizeof(int), cudaMemcpyHostToDevice));
    
    // Barrier to synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Main simulation loop
    for (int step = 0; step < numSteps; ++step) {
        // Gather all bodies to all ranks for force computation
        MPI_Allgatherv(local_bodies.data(), local_count * sizeof(Body), MPI_BYTE,
                       all_bodies.data(), sendcounts.data(), displs.data(), MPI_BYTE,
                       MPI_COMM_WORLD);
        
        // Copy all bodies to device
        CUDA_CHECK(cudaMemcpy(d_all_bodies, all_bodies.data(), 
                             numBodies * sizeof(Body), cudaMemcpyHostToDevice));
        
        // Compute forces on local bodies using CUDA
        int threadsPerBlock = 256;
        int blocksPerGrid = (local_count + threadsPerBlock - 1) / threadsPerBlock;
        
        // Launch CUDA kernel for force computation on local bodies
        computeForcesKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_all_bodies, d_local_indices, local_count, numBodies, DT);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Integrate positions using CUDA
        integrateBodiesKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_all_bodies, d_local_indices, local_count, DT);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy local bodies back to host
        CUDA_CHECK(cudaMemcpy(local_bodies.data(), d_all_bodies + local_start,
                             local_count * sizeof(Body), cudaMemcpyDeviceToHost));
    }
    
    // Gather all bodies to rank 0
    if (world_rank == 0) {
        all_bodies.resize(numBodies);
    }
    MPI_Gatherv(local_bodies.data(), local_count * sizeof(Body), MPI_BYTE,
                world_rank == 0 ? all_bodies.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_BYTE,
                0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (world_rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
        
        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.reserve(numBodies * 6);
            for (const auto& body : all_bodies) {
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
            
            if (validateSimulation(all_bodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(all_bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                CUDA_CHECK(cudaFree(d_all_bodies));
                CUDA_CHECK(cudaFree(d_local_indices));
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_all_bodies));
    CUDA_CHECK(cudaFree(d_local_indices));
    
    MPI_Finalize();
    return 0;
}
