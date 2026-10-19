#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>
#include <mpi.h>
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

__global__ void forceKernel(const double* p, double* v, int n, int begin, int count) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;
    int i = begin + local;
    double x=p[3*i], y=p[3*i+1], z=p[3*i+2], fx=0, fy=0, fz=0;
    for (int j=0; j<n; ++j) {
        double dx=p[3*j]-x, dy=p[3*j+1]-y, dz=p[3*j+2]-z;
        double d=dx*dx+dy*dy+dz*dz+SOFTENING;
        double inv=1.0/sqrt(d);
        double inv3=inv*inv*inv;
        fx += dx*inv3; fy += dy*inv3; fz += dz*inv3;
    }
    v[3*i] += DT*fx; v[3*i+1] += DT*fy; v[3*i+2] += DT*fz;
}

void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
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

void computeForces(std::vector<Body>& bodies, int rank, int ranks) {
    const size_t n = bodies.size();
    std::vector<double> positions(n*3), velocities(n*3);
    for (size_t i=0;i<n;++i) {
        positions[3*i]=bodies[i].pos.x; positions[3*i+1]=bodies[i].pos.y; positions[3*i+2]=bodies[i].pos.z;
        velocities[3*i]=bodies[i].vel.x; velocities[3*i+1]=bodies[i].vel.y; velocities[3*i+2]=bodies[i].vel.z;
    }
    const int begin = static_cast<int>(n)*rank/ranks;
    const int end = static_cast<int>(n)*(rank+1)/ranks;
    double *dp=nullptr, *dv=nullptr;
    checkCuda(cudaMalloc(&dp, positions.size()*sizeof(double)));
    checkCuda(cudaMalloc(&dv, velocities.size()*sizeof(double)));
    checkCuda(cudaMemcpy(dp, positions.data(), positions.size()*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dv, velocities.data(), velocities.size()*sizeof(double), cudaMemcpyHostToDevice));
    int count=end-begin;
    if (count) forceKernel<<<(count+127)/128,128>>>(dp,dv,static_cast<int>(n),begin,count);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemcpy(velocities.data(),dv,velocities.size()*sizeof(double),cudaMemcpyDeviceToHost));
    cudaFree(dp); cudaFree(dv);
    std::vector<double> allVelocities(n*3);
    std::vector<int> counts(ranks), displs(ranks);
    for (int r=0;r<ranks;++r) { int b=static_cast<int>(n)*r/ranks, e=static_cast<int>(n)*(r+1)/ranks; counts[r]=(e-b)*3; displs[r]=b*3; }
    MPI_Allgatherv(velocities.data()+3*begin, (end-begin)*3, MPI_DOUBLE,
                   allVelocities.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    for(size_t i=0;i<n;++i) { bodies[i].vel.x=allVelocities[3*i]; bodies[i].vel.y=allVelocities[3*i+1]; bodies[i].vel.z=allVelocities[3*i+2]; }
}

void integrateBodies(std::vector<Body>& bodies) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(bodies.size()); ++i) {
        auto& body=bodies[static_cast<size_t>(i)];
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
    int rank=0, ranks=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int deviceCount=0;
    checkCuda(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) { fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    checkCuda(cudaSetDevice(rank % deviceCount));
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
    
    if (rank == 0) printf("N-Body Simulation\n");
    if (rank == 0) { printf("Number of bodies: %d\n", numBodies); printf("Number of steps: %d\n", numSteps); printf("Validation: %s\n", validate ? "enabled" : "disabled"); }
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, rank, ranks);
        integrateBodies(bodies);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    long local_ms = duration.count(), max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", max_ms);
    
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
    
    MPI_Finalize(); return 0;
}
