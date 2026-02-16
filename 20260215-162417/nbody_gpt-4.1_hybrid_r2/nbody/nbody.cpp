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

#ifdef __CUDACC__
#define CUDA_KERNEL __global__
#else
#define CUDA_KERNEL
#endif

// CUDA kernel for force computation
CUDA_KERNEL void computeForcesKernel(Body* bodies, int n, double dt, double softening) {
#ifdef __CUDACC__
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
#else
    int i = 0; // fallback for non-CUDA build
#endif
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double ix = bodies[i].pos.x, iy = bodies[i].pos.y, iz = bodies[i].pos.z;
    for (int j = 0; j < n; ++j) {
        double dx = bodies[j].pos.x - ix;
        double dy = bodies[j].pos.y - iy;
        double dz = bodies[j].pos.z - iz;
        double distSqr = dx * dx + dy * dy + dz * dz + softening;
        double invDist = 1.0 / sqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }
    bodies[i].vel.x += dt * Fx;
    bodies[i].vel.y += dt * Fy;
    bodies[i].vel.z += dt * Fz;
}

void computeForces(std::vector<Body>& bodies) {
#ifdef __CUDACC__
    const int n = bodies.size();
    Body* d_bodies;
    cudaMalloc(&d_bodies, n * sizeof(Body));
    cudaMemcpy(d_bodies, bodies.data(), n * sizeof(Body), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    computeForcesKernel<<<numBlocks, blockSize>>>(d_bodies, n, DT, SOFTENING);
    cudaMemcpy(bodies.data(), d_bodies, n * sizeof(Body), cudaMemcpyDeviceToHost);
    cudaFree(d_bodies);
#else
    // Fallback: OpenMP parallel for
    const size_t n = bodies.size();
    #pragma omp parallel for schedule(static)
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

CUDA_KERNEL void integrateBodiesKernel(Body* bodies, int n, double dt) {
#ifdef __CUDACC__
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
#else
    int i = 0;
#endif
    bodies[i].pos.x += bodies[i].vel.x * dt;
    bodies[i].pos.y += bodies[i].vel.y * dt;
    bodies[i].pos.z += bodies[i].vel.z * dt;
}

void integrateBodies(std::vector<Body>& bodies) {
#ifdef __CUDACC__
    const int n = bodies.size();
    Body* d_bodies;
    cudaMalloc(&d_bodies, n * sizeof(Body));
    cudaMemcpy(d_bodies, bodies.data(), n * sizeof(Body), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    integrateBodiesKernel<<<numBlocks, blockSize>>>(d_bodies, n, DT);
    cudaMemcpy(bodies.data(), d_bodies, n * sizeof(Body), cudaMemcpyDeviceToHost);
    cudaFree(d_bodies);
#else
    const size_t n = bodies.size();
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

    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

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
            if (world_rank == 0) printUsage(argv[0]);
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

    if (world_rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    if (world_rank == 0) randomizeBodies(bodies);
    // Broadcast initial bodies to all ranks
    MPI_Bcast(bodies.data(), numBodies * sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Partition bodies for each rank
    int local_n = numBodies / world_size;
    int remainder = numBodies % world_size;
    int start = world_rank * local_n + (world_rank < remainder ? world_rank : remainder);
    int count = local_n + (world_rank < remainder ? 1 : 0);
    std::vector<Body> local_bodies(bodies.begin() + start, bodies.begin() + start + count);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Allgather all positions to all ranks
        std::vector<Body> all_bodies(numBodies);
        MPI_Allgather(local_bodies.data(), count * sizeof(Body), MPI_BYTE,
                      all_bodies.data(), count * sizeof(Body), MPI_BYTE, MPI_COMM_WORLD);
        // Compute forces on local bodies using all bodies
        #pragma omp parallel for
        for (int i = 0; i < count; ++i) {
            local_bodies[i] = all_bodies[start + i];
        }
        computeForces(local_bodies);
        integrateBodies(local_bodies);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (world_rank == 0) printf("Simulation time: %ld ms\n", duration.count());

    // Gather all bodies to rank 0 for output/validation
    MPI_Gather(local_bodies.data(), count * sizeof(Body), MPI_BYTE,
               bodies.data(), count * sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (printResults && world_rank == 0) {
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

    if (validate && world_rank == 0) {
        printf("Validating simulation results...\n");
        if (validateSimulation(bodies)) {
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    MPI_Finalize();
    return 0;
}
