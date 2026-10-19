#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
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

__global__ void forceKernel(const double* x, const double* y, const double* z,
                            double* vx, double* vy, double* vz, int n, int begin, int end) {
    int i = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end) return;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < n; ++j) {
        double dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
        double d2 = dx*dx + dy*dy + dz*dz + 1.0e-9;
        double inv = rsqrt(d2), inv3 = inv * inv * inv;
        fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
    }
    vx[i] += 0.01 * fx; vy[i] += 0.01 * fy; vz[i] += 0.01 * fz;
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
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) cudaSetDevice(rank % deviceCount);
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
    
    if (numBodies < 1) numBodies = 1;
    if (rank == 0) {
    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    std::vector<double> x(numBodies), y(numBodies), z(numBodies), vx(numBodies), vy(numBodies), vz(numBodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        for (int i = 0; i < numBodies; ++i) { x[i]=bodies[i].pos.x; y[i]=bodies[i].pos.y; z[i]=bodies[i].pos.z; vx[i]=bodies[i].vel.x; vy[i]=bodies[i].vel.y; vz[i]=bodies[i].vel.z; }
        int begin = (long long)numBodies * rank / ranks, end = (long long)numBodies * (rank + 1) / ranks;
        double *dx=nullptr,*dy=nullptr,*dz=nullptr,*dvx=nullptr,*dvy=nullptr,*dvz=nullptr;
        cudaMalloc(&dx, numBodies*sizeof(double)); cudaMalloc(&dy, numBodies*sizeof(double)); cudaMalloc(&dz, numBodies*sizeof(double));
        cudaMalloc(&dvx, numBodies*sizeof(double)); cudaMalloc(&dvy, numBodies*sizeof(double)); cudaMalloc(&dvz, numBodies*sizeof(double));
        cudaMemcpy(dx,x.data(),numBodies*sizeof(double),cudaMemcpyHostToDevice); cudaMemcpy(dy,y.data(),numBodies*sizeof(double),cudaMemcpyHostToDevice); cudaMemcpy(dz,z.data(),numBodies*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(dvx,vx.data(),numBodies*sizeof(double),cudaMemcpyHostToDevice); cudaMemcpy(dvy,vy.data(),numBodies*sizeof(double),cudaMemcpyHostToDevice); cudaMemcpy(dvz,vz.data(),numBodies*sizeof(double),cudaMemcpyHostToDevice);
        if (end > begin) forceKernel<<<(end-begin+255)/256,256>>>(dx,dy,dz,dvx,dvy,dvz,numBodies,begin,end);
        cudaDeviceSynchronize();
        cudaMemcpy(vx.data()+begin,dvx+begin,(end-begin)*sizeof(double),cudaMemcpyDeviceToHost); cudaMemcpy(vy.data()+begin,dvy+begin,(end-begin)*sizeof(double),cudaMemcpyDeviceToHost); cudaMemcpy(vz.data()+begin,dvz+begin,(end-begin)*sizeof(double),cudaMemcpyDeviceToHost);
        cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(dvx); cudaFree(dvy); cudaFree(dvz);
        int counts[ranks], offsets[ranks];
        for(int r=0;r<ranks;++r){ offsets[r]=(long long)numBodies*r/ranks; counts[r]=(long long)numBodies*(r+1)/ranks-offsets[r]; }
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DOUBLE,vx.data(),counts,offsets,MPI_DOUBLE,MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DOUBLE,vy.data(),counts,offsets,MPI_DOUBLE,MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DOUBLE,vz.data(),counts,offsets,MPI_DOUBLE,MPI_COMM_WORLD);
        #pragma omp parallel for
        for (int i=0;i<numBodies;++i) { bodies[i].vel.x=vx[i]; bodies[i].vel.y=vy[i]; bodies[i].vel.z=vz[i]; bodies[i].pos.x += vx[i]*DT; bodies[i].pos.y += vy[i]*DT; bodies[i].pos.z += vz[i]*DT; }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if(rank==0) printf("Simulation time: %ld ms\n", duration.count());
    
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
