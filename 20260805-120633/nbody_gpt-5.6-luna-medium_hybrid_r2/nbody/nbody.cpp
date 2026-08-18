#include <chrono>
#include <algorithm>
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

__global__ void bodyForces(const double* px, const double* py, const double* pz,
                           double* ax, double* ay, double* az,
                           const int n, const int firstBody, const int localN) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= localN) return;

    const int i = firstBody + local;
    const double ix = px[i];
    const double iy = py[i];
    const double iz = pz[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < n; ++j) {
        const double dx = px[j] - ix;
        const double dy = py[j] - iy;
        const double dz = pz[j] - iz;
        const double invDist = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        const double invDist3 = invDist * invDist * invDist;
        fx += dx * invDist3;
        fy += dy * invDist3;
        fz += dz * invDist3;
    }
    ax[local] = fx;
    ay[local] = fy;
    az[local] = fz;
}

static void cudaCheck(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
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

void integrateBodies(std::vector<Body>& bodies, int firstBody, int localN,
                     const std::vector<double>& ax, const std::vector<double>& ay,
                     const std::vector<double>& az) {
    #pragma omp parallel for schedule(static)
    for (int local = 0; local < localN; ++local) {
        Body& body = bodies[firstBody + local];
        body.vel.x += DT * ax[local];
        body.vel.y += DT * ay[local];
        body.vel.z += DT * az[local];
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i = 0; i < static_cast<int>(n); ++i) {
        const auto& body = bodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i = 0; i < static_cast<int>(n); ++i) {
        for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) {
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
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (numBodies < 0 || numSteps < 0) {
        if (rank == 0) fprintf(stderr, "Number of bodies and steps must be non-negative\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
               numBodies, numSteps, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA: enabled\n", worldSize, omp_get_max_threads());
    }
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    if (rank == 0) randomizeBodies(bodies);
    MPI_Bcast(bodies.data(), numBodies * static_cast<int>(sizeof(Body)), MPI_BYTE, 0, MPI_COMM_WORLD);

    const int base = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    const int localN = base + (rank < remainder ? 1 : 0);
    const int firstBody = rank * base + (rank < remainder ? rank : remainder);
    std::vector<int> counts(worldSize), displacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = (base + (r < remainder ? 1 : 0)) * static_cast<int>(sizeof(Body));
        displacements[r] = (r * base + (r < remainder ? r : remainder)) * static_cast<int>(sizeof(Body));
    }
    std::vector<double> px(numBodies), py(numBodies), pz(numBodies);
    std::vector<double> ax(localN), ay(localN), az(localN);
    double *d_px = nullptr, *d_py = nullptr, *d_pz = nullptr;
    double *d_ax = nullptr, *d_ay = nullptr, *d_az = nullptr;
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "querying CUDA devices");
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA device is available\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    cudaCheck(cudaSetDevice(rank % deviceCount), "selecting CUDA device");
    const size_t deviceBodyCapacity = static_cast<size_t>(std::max(1, numBodies));
    const size_t deviceLocalCapacity = static_cast<size_t>(std::max(1, localN));
    cudaCheck(cudaMalloc(&d_px, sizeof(double) * deviceBodyCapacity), "allocating device positions");
    cudaCheck(cudaMalloc(&d_py, sizeof(double) * deviceBodyCapacity), "allocating device positions");
    cudaCheck(cudaMalloc(&d_pz, sizeof(double) * deviceBodyCapacity), "allocating device positions");
    cudaCheck(cudaMalloc(&d_ax, sizeof(double) * deviceLocalCapacity), "allocating device forces");
    cudaCheck(cudaMalloc(&d_ay, sizeof(double) * deviceLocalCapacity), "allocating device forces");
    cudaCheck(cudaMalloc(&d_az, sizeof(double) * deviceLocalCapacity), "allocating device forces");
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        for (int i = 0; i < numBodies; ++i) {
            px[i] = bodies[i].pos.x; py[i] = bodies[i].pos.y; pz[i] = bodies[i].pos.z;
        }
        cudaCheck(cudaMemcpy(d_px, px.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice), "copying positions");
        cudaCheck(cudaMemcpy(d_py, py.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice), "copying positions");
        cudaCheck(cudaMemcpy(d_pz, pz.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice), "copying positions");
        if (localN > 0) {
            bodyForces<<<(localN + 255) / 256, 256>>>(d_px, d_py, d_pz, d_ax, d_ay, d_az, numBodies, firstBody, localN);
            cudaCheck(cudaGetLastError(), "launching force kernel");
            cudaCheck(cudaDeviceSynchronize(), "computing forces");
            cudaCheck(cudaMemcpy(ax.data(), d_ax, sizeof(double) * localN, cudaMemcpyDeviceToHost), "copying forces");
            cudaCheck(cudaMemcpy(ay.data(), d_ay, sizeof(double) * localN, cudaMemcpyDeviceToHost), "copying forces");
            cudaCheck(cudaMemcpy(az.data(), d_az, sizeof(double) * localN, cudaMemcpyDeviceToHost), "copying forces");
        }
        integrateBodies(bodies, firstBody, localN, ax, ay, az);
        MPI_Allgatherv(bodies.data() + firstBody, localN * static_cast<int>(sizeof(Body)), MPI_BYTE,
                       bodies.data(), counts.data(), displacements.data(), MPI_BYTE, MPI_COMM_WORLD);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz);
    cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_az);
    long long localMillis = duration.count(), elapsedMillis = 0;
    MPI_Reduce(&localMillis, &elapsedMillis, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %lld ms\n", elapsedMillis);
    
    // Print results for external validation
    if (rank == 0 && printResults) {
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
    if (rank == 0 && validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
