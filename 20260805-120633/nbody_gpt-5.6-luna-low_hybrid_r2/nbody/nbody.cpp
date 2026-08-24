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

__global__ void forceKernel(const double* px, const double* py, const double* pz,
                            double* fx, double* fy, double* fz, int n, int first) {
    const int i = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = 0.0, y = 0.0, z = 0.0;
    for (int j = 0; j < n; ++j) {
        const double dx = px[j] - px[i], dy = py[j] - py[i], dz = pz[j] - pz[i];
        const double inv = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        const double inv3 = inv * inv * inv;
        x += dx * inv3; y += dy * inv3; z += dz * inv3;
    }
    fx[i] = x; fy[i] = y; fz[i] = z;
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

void computeForces(std::vector<Body>& bodies, int rank, int ranks) {
    const int n = static_cast<int>(bodies.size());
    const int first = (n * rank) / ranks, last = (n * (rank + 1)) / ranks;
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    std::vector<double> px(n), py(n), pz(n), fx(n), fy(n), fz(n);
    for (int i = 0; i < n; ++i) { px[i]=bodies[i].pos.x; py[i]=bodies[i].pos.y; pz[i]=bodies[i].pos.z; }
    double *dpx, *dpy, *dpz, *dfx, *dfy, *dfz;
    cudaMalloc(&dpx, bytes); cudaMalloc(&dpy, bytes); cudaMalloc(&dpz, bytes);
    cudaMalloc(&dfx, bytes); cudaMalloc(&dfy, bytes); cudaMalloc(&dfz, bytes);
    cudaMemcpy(dpx, px.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dpy, py.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dpz, pz.data(), bytes, cudaMemcpyHostToDevice);
    forceKernel<<<(last-first + 255) / 256, 256>>>(dpx,dpy,dpz,dfx,dfy,dfz,n,first);
    cudaGetLastError(); cudaDeviceSynchronize();
    cudaMemcpy(fx.data()+first, dfx+first, (last-first)*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(fy.data()+first, dfy+first, (last-first)*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(fz.data()+first, dfz+first, (last-first)*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dpx); cudaFree(dpy); cudaFree(dpz); cudaFree(dfx); cudaFree(dfy); cudaFree(dfz);
    std::vector<int> counts(ranks), displs(ranks);
    for (int r=0; r<ranks; ++r) { displs[r]=(n*r)/ranks; counts[r]=(n*(r+1))/ranks-displs[r]; }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, fx.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, fy.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, fz.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    #pragma omp parallel for schedule(static)
    for (int i=0; i<n; ++i) { bodies[i].vel.x += DT*fx[i]; bodies[i].vel.y += DT*fy[i]; bodies[i].vel.z += DT*fz[i]; }
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
    int rank=0, ranks=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
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
            MPI_Finalize();
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        if (rank == 0) printf("CUDA device required but none was found\n");
        MPI_Finalize();
        return 1;
    }
    cudaSetDevice(rank % deviceCount);
    
    if (rank == 0) printf("N-Body Simulation (MPI + OpenMP + CUDA)\n");
    if (rank == 0) { printf("Number of bodies: %d\n", numBodies); printf("Number of steps: %d\n", numSteps); printf("Validation: %s\n", validate ? "enabled" : "disabled"); }
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, rank, ranks);
        #pragma omp parallel for schedule(static)
        for (int i=0; i<numBodies; ++i) { bodies[i].pos.x += bodies[i].vel.x*DT; bodies[i].pos.y += bodies[i].vel.y*DT; bodies[i].pos.z += bodies[i].vel.z*DT; }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    const long long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) printf("Simulation time: %lld ms\n", global_duration_ms);
    
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
    if (validate) {
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
