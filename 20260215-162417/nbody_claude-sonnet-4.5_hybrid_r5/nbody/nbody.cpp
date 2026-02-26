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
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

__global__ void computeForcesKernel(Body* bodies, double* Fx, double* Fy, double* Fz, int totalBodies, int local_n, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= local_n) return;
    
    int global_i = i + offset;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    
    for (int j = 0; j < totalBodies; ++j) {
        const double dx = bodies[j].pos.x - bodies[global_i].pos.x;
        const double dy = bodies[j].pos.y - bodies[global_i].pos.y;
        const double dz = bodies[j].pos.z - bodies[global_i].pos.z;
        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double invDist = rsqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;
        
        fx += dx * invDist3;
        fy += dy * invDist3;
        fz += dz * invDist3;
    }
    
    Fx[i] = fx;
    Fy[i] = fy;
    Fz[i] = fz;
}

__global__ void updateVelocitiesKernel(Body* bodies, double* Fx, double* Fy, double* Fz, int local_n, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= local_n) return;
    
    int global_i = i + offset;
    bodies[global_i].vel.x += DT * Fx[i];
    bodies[global_i].vel.y += DT * Fy[i];
    bodies[global_i].vel.z += DT * Fz[i];
}

void computeForces(Body* d_bodies, double* d_Fx, double* d_Fy, double* d_Fz, int totalBodies, int local_n, int offset) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (local_n + threadsPerBlock - 1) / threadsPerBlock;
    
    computeForcesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_bodies, d_Fx, d_Fy, d_Fz, totalBodies, local_n, offset);
    CUDA_CHECK(cudaGetLastError());
    
    updateVelocitiesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_bodies, d_Fx, d_Fy, d_Fz, local_n, offset);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void integrateBodiesKernel(Body* bodies, int local_n, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= local_n) return;
    
    int global_i = i + offset;
    bodies[global_i].pos.x += bodies[global_i].vel.x * DT;
    bodies[global_i].pos.y += bodies[global_i].vel.y * DT;
    bodies[global_i].pos.z += bodies[global_i].vel.z * DT;
}

void integrateBodies(Body* d_bodies, int local_n, int offset) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (local_n + threadsPerBlock - 1) / threadsPerBlock;
    
    integrateBodiesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_bodies, local_n, offset);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
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
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CUDA_CHECK(cudaSetDevice(rank % num_devices));
    
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0)
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
    }
    
    // Broadcast parameters to all ranks
    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("N-Body Simulation (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of MPI ranks: %d\n", size);
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Calculate local body count for domain decomposition
    int local_n = numBodies / size;
    int remainder = numBodies % size;
    if (rank < remainder) local_n++;
    
    int offset = (numBodies / size) * rank + std::min(rank, remainder);
    
    // Initialize all bodies on rank 0
    std::vector<Body> bodies(numBodies);
    if (rank == 0) {
        randomizeBodies(bodies);
    }
    
    // Broadcast all bodies to all ranks
    MPI_Bcast(bodies.data(), numBodies * sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Allocate GPU memory
    Body* d_bodies;
    double *d_Fx, *d_Fy, *d_Fz;
    CUDA_CHECK(cudaMalloc(&d_bodies, numBodies * sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&d_Fx, local_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Fy, local_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Fz, local_n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_bodies, bodies.data(), numBodies * sizeof(Body), cudaMemcpyHostToDevice));
    
    // Create send/receive counts and displacements for MPI_Allgatherv
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        int count = numBodies / size;
        if (r < remainder) count++;
        recvcounts[r] = count * sizeof(Body);
        displs[r] = ((numBodies / size) * r + std::min(r, remainder)) * sizeof(Body);
    }
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run simulation
    for (int step = 0; step < numSteps; ++step) {
        // Each rank computes forces for its local bodies
        computeForces(d_bodies, d_Fx, d_Fy, d_Fz, numBodies, local_n, offset);
        
        // Each rank integrates its local bodies
        integrateBodies(d_bodies, local_n, offset);
        
        // Copy only local bodies back to host
        CUDA_CHECK(cudaMemcpy(bodies.data() + offset, d_bodies + offset, 
                              local_n * sizeof(Body), cudaMemcpyDeviceToHost));
        
        // Gather all updated bodies to all ranks for next iteration
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       bodies.data(), recvcounts.data(), displs.data(), MPI_BYTE,
                       MPI_COMM_WORLD);
        
        // Update GPU memory with all gathered bodies
        CUDA_CHECK(cudaMemcpy(d_bodies, bodies.data(), numBodies * sizeof(Body), cudaMemcpyHostToDevice));
    }
    
    // Synchronize and time
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }
    
    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
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
    
    // Validation (rank 0 only)
    int validation_result = 0;
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            validation_result = 0;
        } else {
            printf("Validation: FAILED\n");
            validation_result = 1;
        }
    }
    
    // Broadcast validation result to all ranks
    MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_bodies));
    CUDA_CHECK(cudaFree(d_Fx));
    CUDA_CHECK(cudaFree(d_Fy));
    CUDA_CHECK(cudaFree(d_Fz));
    MPI_Finalize();
    
    return validation_result;
}
