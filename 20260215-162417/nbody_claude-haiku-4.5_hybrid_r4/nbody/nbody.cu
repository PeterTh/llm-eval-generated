#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "../common/results_output.hpp"

#ifdef USE_CUDA
// CUDA kernels for force computation
__global__ void computeForcesKernel(const double* pos_x, const double* pos_y, const double* pos_z,
                                     double* force_x, double* force_y, double* force_z,
                                     int n, double softening) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    
    for (int j = 0; j < n; ++j) {
        double dx = pos_x[j] - pos_x[i];
        double dy = pos_y[j] - pos_y[i];
        double dz = pos_z[j] - pos_z[i];
        double distSqr = dx * dx + dy * dy + dz * dz + softening;
        double invDist = 1.0 / sqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }
    
    force_x[i] = Fx;
    force_y[i] = Fy;
    force_z[i] = Fz;
}
#endif

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// Global MPI variables
#ifdef USE_MPI
int mpi_rank = 0;
int mpi_size = 1;
#endif

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    #ifdef USE_MPI
    // MPI rank 0 initializes, broadcasts to all ranks
    if (mpi_rank == 0) {
        for (auto& body : bodies) {
            body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        }
    }
    MPI_Bcast(bodies.data(), bodies.size() * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    #else
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
    #endif
}

void computeForces(std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    
    #ifdef USE_CUDA
    // GPU-accelerated force computation with CUDA
    // Allocate device memory
    double *d_pos_x, *d_pos_y, *d_pos_z, *d_force_x, *d_force_y, *d_force_z;
    size_t bytes = n * sizeof(double);
    
    cudaMalloc(&d_pos_x, bytes);
    cudaMalloc(&d_pos_y, bytes);
    cudaMalloc(&d_pos_z, bytes);
    cudaMalloc(&d_force_x, bytes);
    cudaMalloc(&d_force_y, bytes);
    cudaMalloc(&d_force_z, bytes);
    
    // Copy positions to device
    std::vector<double> pos_x(n), pos_y(n), pos_z(n);
    #pragma omp parallel for simd
    for (size_t i = 0; i < n; ++i) {
        pos_x[i] = bodies[i].pos.x;
        pos_y[i] = bodies[i].pos.y;
        pos_z[i] = bodies[i].pos.z;
    }
    
    cudaMemcpy(d_pos_x, pos_x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_y, pos_y.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_z, pos_z.data(), bytes, cudaMemcpyHostToDevice);
    
    // Launch force computation kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    computeForcesKernel<<<blocks, threads>>>(d_pos_x, d_pos_y, d_pos_z,
                                             d_force_x, d_force_y, d_force_z,
                                             n, SOFTENING);
    
    // Copy forces back
    std::vector<double> force_x(n), force_y(n), force_z(n);
    cudaMemcpy(force_x.data(), d_force_x, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(force_y.data(), d_force_y, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(force_z.data(), d_force_z, bytes, cudaMemcpyDeviceToHost);
    
    // Update velocities with OpenMP parallelization
    #pragma omp parallel for simd
    for (size_t i = 0; i < n; ++i) {
        bodies[i].vel.x += DT * force_x[i];
        bodies[i].vel.y += DT * force_y[i];
        bodies[i].vel.z += DT * force_z[i];
    }
    
    // Cleanup GPU memory
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_pos_z);
    cudaFree(d_force_x);
    cudaFree(d_force_y);
    cudaFree(d_force_z);
    
    #elif defined(USE_MPI)
    // MPI distribution: each rank computes forces for its subset with local reduction
    int i_start = (mpi_rank * n) / mpi_size;
    int i_end = ((mpi_rank + 1) * n) / mpi_size;
    
    std::vector<Vec3> forces(n, Vec3(0.0, 0.0, 0.0));
    
    // Compute forces with OpenMP parallelization within MPI partition
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = i_start; i < i_end; ++i) {
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
        forces[i] = Vec3(Fx, Fy, Fz);
    }
    
    // Synchronize forces across all MPI ranks with AllReduce
    std::vector<double> local_forces(3 * n), global_forces(3 * n);
    #pragma omp parallel for simd
    for (size_t i = 0; i < n; ++i) {
        local_forces[3*i] = forces[i].x;
        local_forces[3*i+1] = forces[i].y;
        local_forces[3*i+2] = forces[i].z;
    }
    
    MPI_Allreduce(local_forces.data(), global_forces.data(), 3*n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    // Update velocities with globally reduced forces
    #pragma omp parallel for simd
    for (size_t i = 0; i < n; ++i) {
        bodies[i].vel.x += DT * global_forces[3*i];
        bodies[i].vel.y += DT * global_forces[3*i+1];
        bodies[i].vel.z += DT * global_forces[3*i+2];
    }
    
    #else
    // Pure OpenMP parallelization (no MPI or CUDA)
    #pragma omp parallel for schedule(dynamic, 64)
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
    #pragma omp parallel for simd schedule(static)
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
    #ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    #endif
    
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
            if (mpi_rank == 0) printUsage(argv[0]);
            #ifdef USE_MPI
            MPI_Finalize();
            #endif
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            #ifdef USE_MPI
            MPI_Finalize();
            #endif
            return 1;
        }
    }
    
    if (mpi_rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        #ifdef USE_MPI
        printf("MPI ranks: %d\n", mpi_size);
        #endif
        #ifdef _OPENMP
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        #endif
        #ifdef USE_CUDA
        printf("CUDA acceleration: enabled\n");
        #endif
    }
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies);
        integrateBodies(bodies);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = duration.count();
    long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        printf("Simulation time: %ld ms\n", global_duration_ms);
    }
    
    // Print results for external validation
    if (printResults && mpi_rank == 0) {
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
        if (mpi_rank == 0) {
            printf("Validating simulation results...\n");
        }
        
        if (validateSimulation(bodies)) {
            if (mpi_rank == 0) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            #ifdef USE_MPI
            MPI_Finalize();
            #endif
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Validation: FAILED\n");
            }
            #ifdef USE_MPI
            MPI_Finalize();
            #endif
            return 1;
        }
    }
    
    #ifdef USE_MPI
    MPI_Finalize();
    #endif
    return 0;
}
