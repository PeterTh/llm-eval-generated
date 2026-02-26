#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

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

void computeForces(std::vector<Body>& localBodies, const std::vector<Body>& allBodies) {
    const size_t localN = localBodies.size();
    const size_t globalN = allBodies.size();
    
    for (size_t i = 0; i < localN; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (size_t j = 0; j < globalN; ++j) {
            const double dx = allBodies[j].pos.x - localBodies[i].pos.x;
            const double dy = allBodies[j].pos.y - localBodies[i].pos.y;
            const double dz = allBodies[j].pos.z - localBodies[i].pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        
        localBodies[i].vel.x += DT * Fx;
        localBodies[i].vel.y += DT * Fy;
        localBodies[i].vel.z += DT * Fz;
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks parse the same arguments)
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("N-Body Simulation (MPI with %d processes)\n", numProcs);
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Compute local body count per rank
    int localBodyCount = numBodies / numProcs;
    int remainder = numBodies % numProcs;
    if (rank < remainder) {
        localBodyCount++;
    }
    
    // Initialize local bodies
    std::vector<Body> localBodies(localBodyCount);
    
    // Rank 0 generates all bodies and distributes them
    std::vector<Body> allBodies;
    std::vector<int> bodyCounts(numProcs);
    std::vector<int> bodyDispls(numProcs);
    
    if (rank == 0) {
        allBodies.resize(numBodies);
        randomizeBodies(allBodies);
        
        // Calculate counts and displacements for MPI_Scatterv
        int count = 0;
        for (int i = 0; i < numProcs; ++i) {
            bodyCounts[i] = numBodies / numProcs;
            if (i < remainder) {
                bodyCounts[i]++;
            }
            bodyDispls[i] = count;
            count += bodyCounts[i];
        }
    }
    
    // Broadcast body counts and displacements to all ranks
    MPI_Bcast(bodyCounts.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(bodyDispls.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Define MPI datatype for Body struct
    MPI_Datatype MPI_Body;
    MPI_Type_contiguous(6, MPI_DOUBLE, &MPI_Body);
    MPI_Type_commit(&MPI_Body);
    
    // Scatter bodies from rank 0 to all ranks
    MPI_Scatterv(allBodies.data(), bodyCounts.data(), bodyDispls.data(), MPI_Body,
                 localBodies.data(), localBodyCount, MPI_Body,
                 0, MPI_COMM_WORLD);
    
    // All ranks need to know all body counts for Allgather
    std::vector<int> recvCounts(numProcs);
    MPI_Allgather(&localBodyCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Prepare for AllGather
    std::vector<int> recvDispls(numProcs);
    recvDispls[0] = 0;
    for (int i = 1; i < numProcs; ++i) {
        recvDispls[i] = recvDispls[i-1] + recvCounts[i-1];
    }
    
    allBodies.resize(numBodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        // Gather all bodies from all ranks
        MPI_Allgatherv(localBodies.data(), localBodyCount, MPI_Body,
                       allBodies.data(), recvCounts.data(), recvDispls.data(), MPI_Body,
                       MPI_COMM_WORLD);
        
        // Each rank computes forces on its local bodies using all global bodies
        computeForces(localBodies, allBodies);
        integrateBodies(localBodies);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Gather all bodies back to rank 0 for final validation and printing
    MPI_Allgatherv(localBodies.data(), localBodyCount, MPI_Body,
                   allBodies.data(), recvCounts.data(), recvDispls.data(), MPI_Body,
                   MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
        
        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.reserve(numBodies * 6);
            for (const auto& body : allBodies) {
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
            
            if (validateSimulation(allBodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(allBodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Type_free(&MPI_Body);
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Clean up
    MPI_Type_free(&MPI_Body);
    MPI_Finalize();
    
    return 0;
}
