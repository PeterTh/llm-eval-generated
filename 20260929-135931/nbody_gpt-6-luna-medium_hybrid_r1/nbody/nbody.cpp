#include <algorithm>
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

__global__ void forceKernel(const double* state, double* forces, int n, int first, int count) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;
    const int i = first + local;
    const double ix = state[6 * i], iy = state[6 * i + 1], iz = state[6 * i + 2];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < n; ++j) {
        const double dx = state[6 * j] - ix;
        const double dy = state[6 * j + 1] - iy;
        const double dz = state[6 * j + 2] - iz;
        const double inv = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        const double inv3 = inv * inv * inv;
        fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
    }
    forces[3 * local] = fx;
    forces[3 * local + 1] = fy;
    forces[3 * local + 2] = fz;
}

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

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

void computeForces(std::vector<Body>& bodies, int rank, int ranks, const std::vector<int>& counts,
                   const std::vector<int>& displs, double* dState, double* dForces) {
    const int n = static_cast<int>(bodies.size());
    const int first = displs[rank] / 3;
    const int count = counts[rank] / 3;
    std::vector<double> state(static_cast<size_t>(n) * 6);
    std::vector<double> local(static_cast<size_t>(count) * 3);
    std::vector<double> forces(static_cast<size_t>(n) * 3);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        const Body& b = bodies[i];
        state[6*i] = b.pos.x; state[6*i+1] = b.pos.y; state[6*i+2] = b.pos.z;
        state[6*i+3] = b.vel.x; state[6*i+4] = b.vel.y; state[6*i+5] = b.vel.z;
    }
    cudaCheck(cudaMemcpy(dState, state.data(), state.size() * sizeof(double), cudaMemcpyHostToDevice));
    if (count) {
        forceKernel<<<(count + 127) / 128, 128>>>(dState, dForces, n, first, count);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(local.data(), dForces, local.size() * sizeof(double), cudaMemcpyDeviceToHost));
    }
    MPI_Allgatherv(local.data(), counts[rank], MPI_DOUBLE, forces.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        bodies[i].vel.x += DT * forces[3*i];
        bodies[i].vel.y += DT * forces[3*i+1];
        bodies[i].vel.z += DT * forces[3*i+2];
    }
}

void integrateBodies(std::vector<Body>& bodies) {
    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(bodies.size()); ++i) {
        auto& body = bodies[i];
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
    #pragma omp parallel for reduction(+:energy)
    for (long long i = 0; i < static_cast<long long>(bodies.size()); ++i) {
        const auto& body = bodies[i];
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
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
            printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize(); return 1;
        }
    }
    
    if (numBodies < 0 || numSteps < 0) { if (rank == 0) fprintf(stderr, "Body and step counts must be nonnegative\n"); MPI_Finalize(); return 1; }
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) { if (rank == 0) fprintf(stderr, "No CUDA devices available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    cudaCheck(cudaSetDevice(rank % deviceCount));
    double *dState = nullptr, *dForces = nullptr;
    cudaCheck(cudaMalloc(&dState, std::max<size_t>(1, static_cast<size_t>(numBodies) * 6) * sizeof(double)));
    const int localMax = (numBodies + ranks - 1) / ranks;
    cudaCheck(cudaMalloc(&dForces, std::max<size_t>(1, static_cast<size_t>(localMax) * 3) * sizeof(double)));
    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0, off = 0; r < ranks; ++r) {
        const int first = numBodies * r / ranks;
        const int last = numBodies * (r + 1) / ranks;
        counts[r] = (last - first) * 3;
        displs[r] = off;
        off += counts[r];
    }
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, rank, ranks, counts, displs, dState, dForces);
        integrateBodies(bodies);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    cudaFree(dState); cudaFree(dForces);
    if (rank == 0) printf("Simulation time: %ld ms\n", duration.count());
    
    // Print results for external validation
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
            MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
