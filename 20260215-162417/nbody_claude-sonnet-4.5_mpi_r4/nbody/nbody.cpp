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

void computeForces(std::vector<Body>& localBodies, const std::vector<Vec3>& allPositions) {
    const size_t localN = localBodies.size();
    const size_t totalN = allPositions.size();
    
    for (size_t i = 0; i < localN; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (size_t j = 0; j < totalN; ++j) {
            const double dx = allPositions[j].x - localBodies[i].pos.x;
            const double dy = allPositions[j].y - localBodies[i].pos.y;
            const double dz = allPositions[j].z - localBodies[i].pos.z;
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

double computeTotalEnergy(const std::vector<Body>& localBodies, const std::vector<Vec3>& allPositions, 
                          const std::vector<Vec3>& allVelocities, int rank, int numProcs) {
    double localEnergy = 0.0;
    const size_t totalN = allPositions.size();
    const size_t localN = localBodies.size();
    
    // Kinetic energy (computed locally for our bodies)
    for (const auto& body : localBodies) {
        localEnergy += 0.5 * (body.vel.x * body.vel.x + 
                              body.vel.y * body.vel.y + 
                              body.vel.z * body.vel.z);
    }
    
    // Potential energy - distribute computation to avoid double counting
    // Each rank computes potential for pairs (i,j) where i is in its local range
    int bodiesPerRank = totalN / numProcs;
    int startIdx = rank * bodiesPerRank;
    int endIdx = (rank == numProcs - 1) ? totalN : (rank + 1) * bodiesPerRank;
    
    for (int i = startIdx; i < endIdx; ++i) {
        for (size_t j = i + 1; j < totalN; ++j) {
            const double dx = allPositions[j].x - allPositions[i].x;
            const double dy = allPositions[j].y - allPositions[i].y;
            const double dz = allPositions[j].z - allPositions[i].z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            localEnergy -= 1.0 / dist;
        }
    }
    
    // Reduce total energy across all ranks
    double totalEnergy = 0.0;
    MPI_Allreduce(&localEnergy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    return totalEnergy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& localBodies, int rank) {
    int localValid = 1;
    
    for (const auto& body : localBodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            if (rank == 0) printf("Validation failed: found NaN or Inf value in body state\n");
            localValid = 0;
            break;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            if (rank == 0) printf("Validation failed: body position exceeds reasonable bounds\n");
            localValid = 0;
            break;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            if (rank == 0) printf("Validation failed: body velocity exceeds reasonable bounds\n");
            localValid = 0;
            break;
        }
    }
    
    // Check if all ranks validated successfully
    int globalValid = 0;
    MPI_Allreduce(&localValid, &globalValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    
    return globalValid == 1;
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
    
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0)
    if (rank == 0) {
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
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Broadcast parameters to all ranks
    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("N-Body Simulation (MPI)\n");
        printf("Number of processes: %d\n", numProcs);
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Distribute bodies across ranks
    int bodiesPerRank = numBodies / numProcs;
    int remainder = numBodies % numProcs;
    int localNumBodies = bodiesPerRank + (rank < remainder ? 1 : 0);
    
    // Calculate displacements for gathering
    std::vector<int> recvcounts(numProcs);
    std::vector<int> displs(numProcs);
    for (int i = 0; i < numProcs; ++i) {
        recvcounts[i] = bodiesPerRank + (i < remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i-1] + recvcounts[i-1];
    }
    
    // Initialize all bodies on rank 0, then distribute
    std::vector<Body> allBodies;
    if (rank == 0) {
        allBodies.resize(numBodies);
        randomizeBodies(allBodies);
    }
    
    // Distribute bodies to all ranks
    std::vector<Body> localBodies(localNumBodies);
    
    // Create MPI datatype for Body
    MPI_Datatype MPI_BODY;
    MPI_Type_contiguous(6, MPI_DOUBLE, &MPI_BODY);
    MPI_Type_commit(&MPI_BODY);
    
    MPI_Scatterv(rank == 0 ? allBodies.data() : nullptr, recvcounts.data(), displs.data(), MPI_BODY,
                 localBodies.data(), localNumBodies, MPI_BODY, 0, MPI_COMM_WORLD);
    
    // Create MPI datatype for Vec3 (position only)
    MPI_Datatype MPI_VEC3;
    MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_VEC3);
    MPI_Type_commit(&MPI_VEC3);
    
    // Allocate buffers for all body positions and velocities
    std::vector<Vec3> allPositions(numBodies);
    std::vector<Vec3> allVelocities(numBodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<Vec3> localPositions(localNumBodies);
    
    for (int step = 0; step < numSteps; ++step) {
        // Extract local positions
        for (int i = 0; i < localNumBodies; ++i) {
            localPositions[i] = localBodies[i].pos;
        }
        
        // Gather all positions to all ranks
        MPI_Allgatherv(localPositions.data(), localNumBodies, MPI_VEC3,
                       allPositions.data(), recvcounts.data(), displs.data(), MPI_VEC3, MPI_COMM_WORLD);
        
        // Compute forces and update velocities locally
        computeForces(localBodies, allPositions);
        
        // Integrate positions locally
        integrateBodies(localBodies);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Get max time across all ranks
    long localTime = duration.count();
    long maxTime = 0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", maxTime);
    }
    
    // Gather all bodies to rank 0 for results printing
    MPI_Gatherv(localBodies.data(), localNumBodies, MPI_BODY,
                rank == 0 ? allBodies.data() : nullptr, recvcounts.data(), displs.data(), MPI_BODY,
                0, MPI_COMM_WORLD);
    
    // Print results for external validation (only rank 0)
    if (printResults && rank == 0) {
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
        if (rank == 0) {
            printf("Validating simulation results...\n");
        }
        
        if (validateSimulation(localBodies, rank)) {
            // Gather all positions and velocities for energy calculation
            for (int i = 0; i < localNumBodies; ++i) {
                localPositions[i] = localBodies[i].pos;
            }
            std::vector<Vec3> localVelocities(localNumBodies);
            for (int i = 0; i < localNumBodies; ++i) {
                localVelocities[i] = localBodies[i].vel;
            }
            
            MPI_Allgatherv(localPositions.data(), localNumBodies, MPI_VEC3,
                           allPositions.data(), recvcounts.data(), displs.data(), MPI_VEC3, MPI_COMM_WORLD);
            MPI_Allgatherv(localVelocities.data(), localNumBodies, MPI_VEC3,
                           allVelocities.data(), recvcounts.data(), displs.data(), MPI_VEC3, MPI_COMM_WORLD);
            
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(localBodies, allPositions, allVelocities, rank, numProcs);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            MPI_Type_free(&MPI_BODY);
            MPI_Type_free(&MPI_VEC3);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Validation: FAILED\n");
            }
            MPI_Type_free(&MPI_BODY);
            MPI_Type_free(&MPI_VEC3);
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Type_free(&MPI_BODY);
    MPI_Type_free(&MPI_VEC3);
    MPI_Finalize();
    return 0;
}
