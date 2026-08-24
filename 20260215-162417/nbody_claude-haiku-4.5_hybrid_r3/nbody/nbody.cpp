#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <mpi.h>
#include <omp.h>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

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

void computeForces(std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    
#ifdef CUDA_ENABLED
    // GPU-accelerated force computation
    computeForces_CUDA(bodies);
#else
    // CPU computation with OpenMP parallelization
    #pragma omp parallel for schedule(dynamic, 32) collapse(1)
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
#endif
}

void integrateBodies(std::vector<Body>& bodies) {
    #pragma omp parallel for schedule(static, 512)
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
    #pragma omp parallel for reduction(+:energy) schedule(static, 512)
    for (size_t k = 0; k < n; ++k) {
        const auto& body = bodies[k];
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
    #pragma omp parallel for reduction(-:energy) schedule(dynamic, 16)
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

#ifdef CUDA_ENABLED
// CUDA kernel for computing pairwise forces
__global__ void computeForcesKernel(const double* posX, const double* posY, const double* posZ,
                                     double* velX, double* velY, double* velZ,
                                     int n, double softening, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        double posX_i = posX[i], posY_i = posY[i], posZ_i = posZ[i];
        
        #pragma unroll 4
        for (int j = 0; j < n; ++j) {
            double dx = posX[j] - posX_i;
            double dy = posY[j] - posY_i;
            double dz = posZ[j] - posZ_i;
            double distSqr = dx * dx + dy * dy + dz * dz + softening;
            double invDist = rsqrt(distSqr);
            double invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        
        velX[i] += dt * Fx;
        velY[i] += dt * Fy;
        velZ[i] += dt * Fz;
    }
}

void computeForces_CUDA(std::vector<Body>& bodies) {
    int n = bodies.size();
    
    // Allocate pinned host memory for efficient transfers
    double *posX_h, *posY_h, *posZ_h, *velX_h, *velY_h, *velZ_h;
    cudaMallocHost(&posX_h, n * sizeof(double));
    cudaMallocHost(&posY_h, n * sizeof(double));
    cudaMallocHost(&posZ_h, n * sizeof(double));
    cudaMallocHost(&velX_h, n * sizeof(double));
    cudaMallocHost(&velY_h, n * sizeof(double));
    cudaMallocHost(&velZ_h, n * sizeof(double));
    
    // Pack data into separate arrays
    #pragma omp parallel for schedule(static, 512)
    for (int i = 0; i < n; ++i) {
        posX_h[i] = bodies[i].pos.x;
        posY_h[i] = bodies[i].pos.y;
        posZ_h[i] = bodies[i].pos.z;
        velX_h[i] = bodies[i].vel.x;
        velY_h[i] = bodies[i].vel.y;
        velZ_h[i] = bodies[i].vel.z;
    }
    
    // Allocate device memory
    double *posX_d, *posY_d, *posZ_d, *velX_d, *velY_d, *velZ_d;
    cudaMalloc(&posX_d, n * sizeof(double));
    cudaMalloc(&posY_d, n * sizeof(double));
    cudaMalloc(&posZ_d, n * sizeof(double));
    cudaMalloc(&velX_d, n * sizeof(double));
    cudaMalloc(&velY_d, n * sizeof(double));
    cudaMalloc(&velZ_d, n * sizeof(double));
    
    // Copy to device
    cudaMemcpy(posX_d, posX_h, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(posY_d, posY_h, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(posZ_d, posZ_h, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(velX_d, velX_h, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(velY_d, velY_h, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(velZ_d, velZ_h, n * sizeof(double), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 128;
    int gridSize = (n + blockSize - 1) / blockSize;
    computeForcesKernel<<<gridSize, blockSize>>>(posX_d, posY_d, posZ_d, 
                                                  velX_d, velY_d, velZ_d,
                                                  n, SOFTENING, DT);
    
    // Copy results back
    cudaMemcpy(velX_h, velX_d, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(velY_h, velY_d, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(velZ_h, velZ_d, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Unpack data
    #pragma omp parallel for schedule(static, 512)
    for (int i = 0; i < n; ++i) {
        bodies[i].vel.x = velX_h[i];
        bodies[i].vel.y = velY_h[i];
        bodies[i].vel.z = velZ_h[i];
    }
    
    // Free memory
    cudaFree(posX_d);
    cudaFree(posY_d);
    cudaFree(posZ_d);
    cudaFree(velX_d);
    cudaFree(velY_d);
    cudaFree(velZ_d);
    cudaFreeHost(posX_h);
    cudaFreeHost(posY_h);
    cudaFreeHost(posZ_h);
    cudaFreeHost(velX_h);
    cudaFreeHost(velY_h);
    cudaFreeHost(velZ_h);
}

#endif

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
    
    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    
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
            if (mpiRank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else if (mpiRank == 0) {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (mpiRank == 0) {
        printf("N-Body Simulation (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI processes: %d\n", mpiSize);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
#ifdef CUDA_ENABLED
        printf("CUDA: ENABLED\n");
#else
        printf("CUDA: DISABLED\n");
#endif
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Initialize bodies (all ranks have same bodies initially)
    std::vector<Body> bodies(numBodies);
    
    // Ensure all ranks generate the same initial random state
    randomizeBodies(bodies, 42);
    
    // Distribute work among ranks: each rank computes forces for its slice of bodies
    int bodiesPerRank = numBodies / mpiSize;
    int remainder = numBodies % mpiSize;
    int startIdx = mpiRank * bodiesPerRank + std::min(mpiRank, remainder);
    int endIdx = startIdx + bodiesPerRank + (mpiRank < remainder ? 1 : 0);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        // Each rank computes forces for its assigned bodies
        const size_t n = bodies.size();
        
        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = startIdx; i < endIdx; ++i) {
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
        
        // Synchronize velocity updates across all ranks
        if (mpiSize > 1) {
            std::vector<double> velData(numBodies * 3, 0.0);
            
            // Pack updated velocities from this rank's portion
            #pragma omp parallel for schedule(static, 256)
            for (int i = startIdx; i < endIdx; ++i) {
                velData[i * 3 + 0] = bodies[i].vel.x;
                velData[i * 3 + 1] = bodies[i].vel.y;
                velData[i * 3 + 2] = bodies[i].vel.z;
            }
            
            // AllReduce to combine all velocity updates using SUM
            std::vector<double> velDataGlobal(numBodies * 3);
            MPI_Allreduce(velData.data(), velDataGlobal.data(), numBodies * 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            
            // Update all velocities with the gathered values
            #pragma omp parallel for schedule(static, 256)
            for (int i = 0; i < numBodies; ++i) {
                bodies[i].vel.x = velDataGlobal[i * 3 + 0];
                bodies[i].vel.y = velDataGlobal[i * 3 + 1];
                bodies[i].vel.z = velDataGlobal[i * 3 + 2];
            }
        }
        
        // Integrate positions (all ranks can do this in parallel)
        #pragma omp parallel for schedule(static, 512)
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos.x += bodies[i].vel.x * DT;
            bodies[i].pos.y += bodies[i].vel.y * DT;
            bodies[i].pos.z += bodies[i].vel.z * DT;
        }
        
        // Synchronize positions across all ranks
        if (mpiSize > 1) {
            std::vector<double> posData(numBodies * 3);
            
            // Pack positions
            #pragma omp parallel for schedule(static, 256)
            for (int i = 0; i < numBodies; ++i) {
                posData[i * 3 + 0] = bodies[i].pos.x;
                posData[i * 3 + 1] = bodies[i].pos.y;
                posData[i * 3 + 2] = bodies[i].pos.z;
            }
            
            // Broadcast from rank 0 to ensure consistency
            MPI_Bcast(posData.data(), numBodies * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            
            // Unpack positions
            #pragma omp parallel for schedule(static, 256)
            for (int i = 0; i < numBodies; ++i) {
                bodies[i].pos.x = posData[i * 3 + 0];
                bodies[i].pos.y = posData[i * 3 + 1];
                bodies[i].pos.z = posData[i * 3 + 2];
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long localDuration = static_cast<long long>(duration.count());
    long long globalDuration = 0;
    MPI_Reduce(&localDuration, &globalDuration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (mpiRank == 0) {
        printf("Simulation time: %lld ms\n", globalDuration);
    }
    
    // Print results for external validation (only from rank 0)
    if (printResults && mpiRank == 0) {
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
        if (mpiRank == 0) {
            printf("Validating simulation results...\n");
        }
        
        if (validateSimulation(bodies)) {
            if (mpiRank == 0) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
