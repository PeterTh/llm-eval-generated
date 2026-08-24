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
    const size_t totalN = allBodies.size();
    
    for (size_t i = 0; i < localN; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (size_t j = 0; j < totalN; ++j) {
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
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
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
        printf("N-Body Simulation (MPI)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Calculate local size for this rank
    int localSize = numBodies / size;
    int remainder = numBodies % size;
    
    // Distribute remainder among first ranks
    int localStart = rank * localSize + std::min(rank, remainder);
    if (rank < remainder) {
        localSize++;
    }
    
    // Prepare arrays for Allgatherv
    std::vector<int> recvCounts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        recvCounts[r] = (numBodies / size) * 6; // 6 doubles per body (pos + vel)
        if (r < remainder) {
            recvCounts[r] += 6;
        }
        displs[r] = r * (numBodies / size) * 6 + std::min(r, remainder) * 6;
    }
    
    // Initialize all bodies on rank 0
    std::vector<Body> allBodies(numBodies);
    if (rank == 0) {
        randomizeBodies(allBodies);
    }
    
    // Broadcast all bodies to all ranks (needed for initial state)
    MPI_Bcast(allBodies.data(), numBodies * sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Extract local bodies
    std::vector<Body> localBodies(localSize);
    for (int i = 0; i < localSize; ++i) {
        localBodies[i] = allBodies[localStart + i];
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run simulation
    std::vector<double> sendBuf(localSize * 6);
    std::vector<double> recvBuf(numBodies * 6);
    
    for (int step = 0; step < numSteps; ++step) {
        // Pack local bodies for communication (positions only needed for force computation)
        for (int i = 0; i < localSize; ++i) {
            sendBuf[i * 6 + 0] = localBodies[i].pos.x;
            sendBuf[i * 6 + 1] = localBodies[i].pos.y;
            sendBuf[i * 6 + 2] = localBodies[i].pos.z;
            sendBuf[i * 6 + 3] = localBodies[i].vel.x;
            sendBuf[i * 6 + 4] = localBodies[i].vel.y;
            sendBuf[i * 6 + 5] = localBodies[i].vel.z;
        }
        
        // Gather all body states
        MPI_Allgatherv(sendBuf.data(), localSize * 6, MPI_DOUBLE,
                       recvBuf.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        
        // Unpack all bodies
        for (int i = 0; i < numBodies; ++i) {
            allBodies[i].pos.x = recvBuf[i * 6 + 0];
            allBodies[i].pos.y = recvBuf[i * 6 + 1];
            allBodies[i].pos.z = recvBuf[i * 6 + 2];
            allBodies[i].vel.x = recvBuf[i * 6 + 3];
            allBodies[i].vel.y = recvBuf[i * 6 + 4];
            allBodies[i].vel.z = recvBuf[i * 6 + 5];
        }
        
        // Compute forces for local bodies
        computeForces(localBodies, allBodies);
        
        // Integrate local bodies
        integrateBodies(localBodies);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long localDuration = static_cast<long long>(duration.count());
    long long globalDuration = 0;
    MPI_Reduce(&localDuration, &globalDuration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Simulation time: %lld ms\n", globalDuration);
    }
    
    // Gather all bodies to rank 0 for final output
    for (int i = 0; i < localSize; ++i) {
        sendBuf[i * 6 + 0] = localBodies[i].pos.x;
        sendBuf[i * 6 + 1] = localBodies[i].pos.y;
        sendBuf[i * 6 + 2] = localBodies[i].pos.z;
        sendBuf[i * 6 + 3] = localBodies[i].vel.x;
        sendBuf[i * 6 + 4] = localBodies[i].vel.y;
        sendBuf[i * 6 + 5] = localBodies[i].vel.z;
    }
    
    MPI_Gatherv(sendBuf.data(), localSize * 6, MPI_DOUBLE,
                recvBuf.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        // Unpack all bodies
        for (int i = 0; i < numBodies; ++i) {
            allBodies[i].pos.x = recvBuf[i * 6 + 0];
            allBodies[i].pos.y = recvBuf[i * 6 + 1];
            allBodies[i].pos.z = recvBuf[i * 6 + 2];
            allBodies[i].vel.x = recvBuf[i * 6 + 3];
            allBodies[i].vel.y = recvBuf[i * 6 + 4];
            allBodies[i].vel.z = recvBuf[i * 6 + 5];
        }
        
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
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
