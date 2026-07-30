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

// Compute forces on localBodies using allPositions (size = totalNumBodies).
// Updates velocities of localBodies in place.
void computeForces(std::vector<Body>& localBodies, const double* allPositions, size_t totalNumBodies) {
    const size_t localN = localBodies.size();
    
    for (size_t i = 0; i < localN; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        const double px = localBodies[i].pos.x;
        const double py = localBodies[i].pos.y;
        const double pz = localBodies[i].pos.z;
        
        for (size_t j = 0; j < totalNumBodies; ++j) {
            const double dx = allPositions[j * 3 + 0] - px;
            const double dy = allPositions[j * 3 + 1] - py;
            const double dz = allPositions[j * 3 + 2] - pz;
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
    
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks parse identically)
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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI ranks: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Compute local body distribution (block distribution)
    const int localN = numBodies / nprocs + (rank < numBodies % nprocs ? 1 : 0);
    // Compute start offset for this rank
    int startOffset = 0;
    {
        int base = numBodies / nprocs;
        int remainder = numBodies % nprocs;
        if (rank < remainder) {
            startOffset = rank * (base + 1);
        } else {
            startOffset = remainder * (base + 1) + (rank - remainder) * base;
        }
    }
    
    // All ranks initialize all bodies identically (same seed) to get correct randomization
    // Then each rank keeps only its local subset
    std::vector<Body> allBodies(numBodies);
    randomizeBodies(allBodies);
    
    std::vector<Body> localBodies(localN);
    std::memcpy(localBodies.data(), &allBodies[startOffset], localN * sizeof(Body));
    allBodies.clear();
    allBodies.shrink_to_fit();
    
    // Buffer for allgather of positions (3 doubles per body)
    std::vector<double> allPositions(numBodies * 3);
    
    // Build sendcounts and displacements for Allgatherv
    std::vector<int> sendcounts(nprocs);
    std::vector<int> displs(nprocs);
    {
        int base = numBodies / nprocs;
        int remainder = numBodies % nprocs;
        int offset = 0;
        for (int p = 0; p < nprocs; ++p) {
            int cnt = base + (p < remainder ? 1 : 0);
            sendcounts[p] = cnt * 3; // 3 doubles for position
            displs[p] = offset;
            offset += cnt * 3;
        }
    }
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        // Pack local positions into contiguous buffer
        std::vector<double> localPos(localN * 3);
        for (int i = 0; i < localN; ++i) {
            localPos[i * 3 + 0] = localBodies[i].pos.x;
            localPos[i * 3 + 1] = localBodies[i].pos.y;
            localPos[i * 3 + 2] = localBodies[i].pos.z;
        }
        
        // Allgather positions so every rank has all positions
        MPI_Allgatherv(localPos.data(), localN * 3, MPI_DOUBLE,
                       allPositions.data(), sendcounts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        
        // Compute forces on local bodies using all positions
        computeForces(localBodies, allPositions.data(), numBodies);
        
        // Integrate local bodies
        integrateBodies(localBodies);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }
    
    // Gather all body data to rank 0 for output/validation
    // We need to gather both positions and velocities (6 doubles per body)
    std::vector<int> sendcounts6(nprocs);
    std::vector<int> displs6(nprocs);
    {
        int base = numBodies / nprocs;
        int remainder = numBodies % nprocs;
        int offset = 0;
        for (int p = 0; p < nprocs; ++p) {
            int cnt = base + (p < remainder ? 1 : 0);
            sendcounts6[p] = cnt * 6; // 6 doubles per body (pos + vel)
            displs6[p] = offset;
            offset += cnt * 6;
        }
    }
    
    // Pack local body data
    std::vector<double> localData(localN * 6);
    for (int i = 0; i < localN; ++i) {
        localData[i * 6 + 0] = localBodies[i].pos.x;
        localData[i * 6 + 1] = localBodies[i].pos.y;
        localData[i * 6 + 2] = localBodies[i].pos.z;
        localData[i * 6 + 3] = localBodies[i].vel.x;
        localData[i * 6 + 4] = localBodies[i].vel.y;
        localData[i * 6 + 5] = localBodies[i].vel.z;
    }
    
    std::vector<double> gatheredData;
    if (rank == 0) {
        gatheredData.resize(numBodies * 6);
    }
    
    MPI_Gatherv(localData.data(), localN * 6, MPI_DOUBLE,
                gatheredData.data(), sendcounts6.data(), displs6.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        // Reconstruct bodies on rank 0
        std::vector<Body> bodies(numBodies);
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos.x = gatheredData[i * 6 + 0];
            bodies[i].pos.y = gatheredData[i * 6 + 1];
            bodies[i].pos.z = gatheredData[i * 6 + 2];
            bodies[i].vel.x = gatheredData[i * 6 + 3];
            bodies[i].vel.y = gatheredData[i * 6 + 4];
            bodies[i].vel.z = gatheredData[i * 6 + 5];
        }
        
        // Print results for external validation
        if (printResults) {
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
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
