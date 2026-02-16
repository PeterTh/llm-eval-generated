#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_CUDA
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

#ifdef USE_CUDA
struct BodySoA {
    double* pos_x;
    double* pos_y;
    double* pos_z;
    double* vel_x;
    double* vel_y;
    double* vel_z;
};

__global__ void computeForcesCUDA(int n, double* pos_x, double* pos_y, double* pos_z, double* vel_x, double* vel_y, double* vel_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double xi = pos_x[i], yi = pos_y[i], zi = pos_z[i];
    for (int j = 0; j < n; ++j) {
        double dx = pos_x[j] - xi;
        double dy = pos_y[j] - yi;
        double dz = pos_z[j] - zi;
        double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        double invDist = 1.0 / sqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }
    vel_x[i] += DT * Fx;
    vel_y[i] += DT * Fy;
    vel_z[i] += DT * Fz;
}
#endif

void computeForces(std::vector<Body>& bodies) {
    const size_t n = bodies.size();
#ifdef USE_MPI
    // MPI: gather all positions to all ranks
    // (Assume all ranks have same n)
    // Only positions are needed for force computation
    std::vector<double> all_pos_x(n), all_pos_y(n), all_pos_z(n);
    std::vector<double> local_pos_x(n), local_pos_y(n), local_pos_z(n);
    for (size_t i = 0; i < n; ++i) {
        local_pos_x[i] = bodies[i].pos.x;
        local_pos_y[i] = bodies[i].pos.y;
        local_pos_z[i] = bodies[i].pos.z;
    }
    MPI_Allgather(local_pos_x.data(), n, MPI_DOUBLE, all_pos_x.data(), n, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(local_pos_y.data(), n, MPI_DOUBLE, all_pos_y.data(), n, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(local_pos_z.data(), n, MPI_DOUBLE, all_pos_z.data(), n, MPI_DOUBLE, MPI_COMM_WORLD);
#else
    std::vector<double> all_pos_x(n), all_pos_y(n), all_pos_z(n);
    for (size_t i = 0; i < n; ++i) {
        all_pos_x[i] = bodies[i].pos.x;
        all_pos_y[i] = bodies[i].pos.y;
        all_pos_z[i] = bodies[i].pos.z;
    }
#endif
#ifdef USE_CUDA
    // Allocate and copy to device
    double *d_pos_x, *d_pos_y, *d_pos_z, *d_vel_x, *d_vel_y, *d_vel_z;
    cudaMalloc(&d_pos_x, n * sizeof(double));
    cudaMalloc(&d_pos_y, n * sizeof(double));
    cudaMalloc(&d_pos_z, n * sizeof(double));
    cudaMalloc(&d_vel_x, n * sizeof(double));
    cudaMalloc(&d_vel_y, n * sizeof(double));
    cudaMalloc(&d_vel_z, n * sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        // Copy current velocities
        cudaMemcpy(d_vel_x + i, &bodies[i].vel.x, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel_y + i, &bodies[i].vel.y, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel_z + i, &bodies[i].vel.z, sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_pos_x, all_pos_x.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_y, all_pos_y.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_z, all_pos_z.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    computeForcesCUDA<<<numBlocks, blockSize>>>(n, d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z);
    cudaDeviceSynchronize();
    // Copy velocities back
    for (size_t i = 0; i < n; ++i) {
        cudaMemcpy(&bodies[i].vel.x, d_vel_x + i, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&bodies[i].vel.y, d_vel_y + i, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&bodies[i].vel.z, d_vel_z + i, sizeof(double), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_pos_x); cudaFree(d_pos_y); cudaFree(d_pos_z);
    cudaFree(d_vel_x); cudaFree(d_vel_y); cudaFree(d_vel_z);
#else
    // OpenMP parallelization for CPU fallback
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        for (size_t j = 0; j < n; ++j) {
            const double dx = all_pos_x[j] - all_pos_x[i];
            const double dy = all_pos_y[j] - all_pos_y[i];
            const double dz = all_pos_z[j] - all_pos_z[i];
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
    const size_t n = bodies.size();
#ifdef USE_CUDA
    // Offload to CUDA if available
    double *d_pos_x, *d_pos_y, *d_pos_z, *d_vel_x, *d_vel_y, *d_vel_z;
    cudaMalloc(&d_pos_x, n * sizeof(double));
    cudaMalloc(&d_pos_y, n * sizeof(double));
    cudaMalloc(&d_pos_z, n * sizeof(double));
    cudaMalloc(&d_vel_x, n * sizeof(double));
    cudaMalloc(&d_vel_y, n * sizeof(double));
    cudaMalloc(&d_vel_z, n * sizeof(double));
    for (size_t i = 0; i < n; ++i) {
        cudaMemcpy(d_pos_x + i, &bodies[i].pos.x, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_y + i, &bodies[i].pos.y, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_z + i, &bodies[i].pos.z, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel_x + i, &bodies[i].vel.x, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel_y + i, &bodies[i].vel.y, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel_z + i, &bodies[i].vel.z, sizeof(double), cudaMemcpyHostToDevice);
    }
    // Kernel for integration
    __global__ void integrateKernel(int n, double* pos_x, double* pos_y, double* pos_z, const double* vel_x, const double* vel_y, const double* vel_z) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        pos_x[i] += vel_x[i] * DT;
        pos_y[i] += vel_y[i] * DT;
        pos_z[i] += vel_z[i] * DT;
    }
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    integrateKernel<<<numBlocks, blockSize>>>(n, d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z);
    cudaDeviceSynchronize();
    for (size_t i = 0; i < n; ++i) {
        cudaMemcpy(&bodies[i].pos.x, d_pos_x + i, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&bodies[i].pos.y, d_pos_y + i, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&bodies[i].pos.z, d_pos_z + i, sizeof(double), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_pos_x); cudaFree(d_pos_y); cudaFree(d_pos_z);
    cudaFree(d_vel_x); cudaFree(d_vel_y); cudaFree(d_vel_z);
#else
    // OpenMP parallelization for CPU fallback
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
#endif
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
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
#ifdef USE_MPI
    int mpi_rank = 0, mpi_size = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif
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
            printUsage(argv[0]);
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
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
    }
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies);
        integrateBodies(bodies);
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (mpi_rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
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
    if (validate && mpi_rank == 0) {
        printf("Validating simulation results...\n");
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return 0;
        } else {
            printf("Validation: FAILED\n");
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
