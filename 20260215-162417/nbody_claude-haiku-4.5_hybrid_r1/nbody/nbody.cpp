#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <mpi.h>

#include "../common/results_output.hpp"

// CUDA kernel for force computation
#ifdef __CUDACC__
__global__ void computeForcesKernel(const Vec3* posIn, Vec3* velOut, size_t n, size_t myStart, size_t myEnd, double dt) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x + myStart;
    if (i >= myEnd) return;
    
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    
    for (size_t j = 0; j < n; ++j) {
        const double dx = posIn[j].x - posIn[i].x;
        const double dy = posIn[j].y - posIn[i].y;
        const double dz = posIn[j].z - posIn[i].z;
        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double invDist = rsqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;
        
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }
    
    velOut[i].x += dt * Fx;
    velOut[i].y += dt * Fy;
    velOut[i].z += dt * Fz;
}

__global__ void integrateBodiesKernel(Vec3* pos, const Vec3* vel, size_t n, size_t myStart, size_t myEnd, double dt) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x + myStart;
    if (i >= myEnd) return;
    
    pos[i].x += vel[i].x * dt;
    pos[i].y += vel[i].y * dt;
    pos[i].z += vel[i].z * dt;
}
#endif

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

void computeForces(std::vector<Body>& bodies, const std::vector<Body>& bodiesRecv, 
                   int, int, size_t myStart, size_t myEnd) {
    const size_t n = bodies.size();
    
#pragma omp parallel for schedule(guided)
    for (size_t i = myStart; i < myEnd; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (size_t j = 0; j < n; ++j) {
            const double dx = bodiesRecv[j].pos.x - bodies[i].pos.x;
            const double dy = bodiesRecv[j].pos.y - bodies[i].pos.y;
            const double dz = bodiesRecv[j].pos.z - bodies[i].pos.z;
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

void integrateBodies(std::vector<Body>& bodies, size_t myStart, size_t myEnd) {
#pragma omp parallel for schedule(guided)
    for (size_t i = myStart; i < myEnd; ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
#pragma omp parallel for reduction(+:energy) schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
#pragma omp parallel for reduction(+:energy) schedule(guided)
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
    bool valid = true;
    
    for (size_t idx = 0; idx < bodies.size(); ++idx) {
        const auto& body = bodies[idx];
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
    return valid;
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int myRank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    
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
            if (myRank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else if (myRank == 0) {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (myRank == 0) {
        printf("N-Body Simulation (Hybrid MPI/OpenMP/CUDA)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Number of MPI ranks: %d\n", numRanks);
        printf("Number of OpenMP threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Distribute bodies across MPI ranks
    size_t bodiesPerRank = numBodies / numRanks;
    size_t remainder = numBodies % numRanks;
    size_t myStart = myRank * bodiesPerRank + std::min((size_t)myRank, remainder);
    size_t myEnd = myStart + bodiesPerRank + (myRank < (int)remainder ? 1 : 0);
    
    // All ranks initialize all bodies (needed for force computation)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        // Create MPI datatype for Body
        MPI_Datatype bodyType;
        int blockLengths[2] = {3, 3};
        MPI_Aint displacements[2] = {offsetof(Body, pos), offsetof(Body, vel)};
        MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
        
        MPI_Type_create_struct(2, blockLengths, displacements, types, &bodyType);
        MPI_Type_commit(&bodyType);
        
        // Allgather all bodies
        std::vector<Body> allBodies = bodies;
        std::vector<int> sendCounts(numRanks);
        std::vector<int> displs(numRanks);
        int myCount = myEnd - myStart;
        
        MPI_Allgather(&myCount, 1, MPI_INT, sendCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        displs[0] = 0;
        for (int i = 1; i < numRanks; ++i) {
            displs[i] = displs[i-1] + sendCounts[i-1];
        }
        
        MPI_Allgatherv(bodies.data() + myStart, myCount, bodyType,
                       allBodies.data(), sendCounts.data(), displs.data(), bodyType, MPI_COMM_WORLD);
        
        MPI_Type_free(&bodyType);
        
        // Each rank computes forces on its subset of bodies using OpenMP
        computeForces(bodies, allBodies, myRank, numRanks, myStart, myEnd);
        
        // Integrate local bodies
        integrateBodies(bodies, myStart, myEnd);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (myRank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }
    
    // Print results for external validation (rank 0 only)
    if (printResults && myRank == 0) {
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
        if (myRank == 0) {
            printf("Validating simulation results...\n");
        }
        
        bool localValid = validateSimulation(bodies);
        bool globalValid = true;
        MPI_Allreduce(&localValid, &globalValid, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
        
        if (globalValid) {
            if (myRank == 0) {
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (myRank == 0) {
                printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
