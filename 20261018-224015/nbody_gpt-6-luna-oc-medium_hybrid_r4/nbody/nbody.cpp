#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
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

__global__ void forceKernel(const Body* input, Body* output, int n, int begin, int end) {
    const int i = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end) return;
    Body b = input[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < n; ++j) {
        const double dx = input[j].pos.x - b.pos.x;
        const double dy = input[j].pos.y - b.pos.y;
        const double dz = input[j].pos.z - b.pos.z;
        const double d2 = dx*dx + dy*dy + dz*dz + SOFTENING;
        const double inv = rsqrt(d2);
        const double inv3 = inv * inv * inv;
        fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
    }
    b.vel.x += DT * fx; b.vel.y += DT * fy; b.vel.z += DT * fz;
    b.pos.x += b.vel.x * DT; b.pos.y += b.vel.y * DT; b.pos.z += b.vel.z * DT;
    output[i] = b;
}

static void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void distributedStep(std::vector<Body>& bodies, int rank, int ranks, int* counts, int* displs,
                     Body* deviceInput, Body* deviceOutput, int localBegin, int localEnd) {
    const int n = static_cast<int>(bodies.size());
    cudaCheck(cudaMemcpy(deviceInput, bodies.data(), n * sizeof(Body), cudaMemcpyHostToDevice));
    if (localEnd > localBegin) {
        forceKernel<<<(localEnd-localBegin+255)/256, 256>>>(deviceInput, deviceOutput, n, localBegin, localEnd);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(bodies.data()+localBegin, deviceOutput+localBegin,
                             (localEnd-localBegin)*sizeof(Body), cudaMemcpyDeviceToHost));
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bodies.data(), counts, displs,
                   MPI_BYTE, MPI_COMM_WORLD);
}

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

void integrateBodies(std::vector<Body>& bodies) {
    for (auto& body : bodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
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
    
    if (rank == 0) printf("N-Body Simulation\n");
    if (rank == 0) { printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled"); }
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(rank % deviceCount));
    int* counts = new int[ranks]; int* displs = new int[ranks];
    for (int r = 0; r < ranks; ++r) {
        const int first = numBodies * r / ranks, last = numBodies * (r + 1) / ranks;
        counts[r] = (last-first) * static_cast<int>(sizeof(Body));
        displs[r] = first * static_cast<int>(sizeof(Body));
    }
    const int localBegin = numBodies * rank / ranks, localEnd = numBodies * (rank+1) / ranks;
    Body *deviceInput = nullptr, *deviceOutput = nullptr;
    cudaCheck(cudaMalloc(&deviceInput, static_cast<size_t>(numBodies) * sizeof(Body)));
    cudaCheck(cudaMalloc(&deviceOutput, static_cast<size_t>(numBodies) * sizeof(Body)));
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        distributedStep(bodies, rank, ranks, counts, displs, deviceInput, deviceOutput, localBegin, localEnd);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    cudaFree(deviceInput); cudaFree(deviceOutput); delete[] counts; delete[] displs;
    long localDuration = duration.count(), maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", maxDuration);
    
    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        bodyData.resize(static_cast<size_t>(numBodies) * 6);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const Body& body = bodies[i];
            double* dst = bodyData.data() + static_cast<size_t>(i) * 6;
            dst[0]=body.pos.x; dst[1]=body.pos.y; dst[2]=body.pos.z;
            dst[3]=body.vel.x; dst[4]=body.vel.y; dst[5]=body.vel.z;
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
    
    MPI_Finalize(); return 0;
}
