#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <type_traits>
#include <vector>

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

static_assert(std::is_standard_layout_v<Body> && sizeof(Body) == 6 * sizeof(double),
              "Body must be a contiguous six-double representation");

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

void computeForces(std::vector<Body>& bodies, const std::vector<Vec3>& positions) {
    const size_t n = positions.size();

    for (Body& body : bodies) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (size_t j = 0; j < n; ++j) {
            const double dx = positions[j].x - body.pos.x;
            const double dy = positions[j].y - body.pos.y;
            const double dz = positions[j].z - body.pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        
        body.vel.x += DT * Fx;
        body.vel.y += DT * Fy;
        body.vel.z += DT * Fz;
    }
}

void integrateBodies(std::vector<Body>& bodies) {
    for (auto& body : bodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies, const std::vector<Vec3>& positions,
                          const int firstBody) {
    double energy = 0.0;
    const size_t n = positions.size();
    
    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
    for (size_t localI = 0; localI < bodies.size(); ++localI) {
        const size_t i = static_cast<size_t>(firstBody) + localI;
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = positions[j].x - bodies[localI].pos.x;
            const double dy = positions[j].y - bodies[localI].pos.y;
            const double dz = positions[j].z - bodies[localI].pos.z;
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
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
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
            if (rank == 0) printUsage(argv[0]);
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
    
    if (numBodies < 0 || numSteps < 0) {
        if (rank == 0) printf("Number of bodies and steps must be non-negative\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // A contiguous block decomposition gives every rank an almost equal amount
    // of O(N^2) work.  Root generates the original deterministic sequence and
    // distributes it, preserving the serial initialization exactly.
    std::vector<int> counts(worldSize), displacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = numBodies / worldSize + (r < numBodies % worldSize ? 1 : 0);
        displacements[r] = r == 0 ? 0 : displacements[r - 1] + counts[r - 1];
    }
    const int localCount = counts[rank];
    const int firstBody = displacements[rank];
    std::vector<Body> bodies(localCount);
    std::vector<Body> initialBodies;
    if (rank == 0) {
        initialBodies.resize(numBodies);
        randomizeBodies(initialBodies);
    }

    std::vector<int> stateCounts(worldSize), stateDisplacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        stateCounts[r] = counts[r] * 6;
        stateDisplacements[r] = displacements[r] * 6;
    }
    MPI_Scatterv(rank == 0 ? initialBodies.data() : nullptr, stateCounts.data(),
                 stateDisplacements.data(), MPI_DOUBLE, bodies.data(), localCount * 6,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> positionCounts(worldSize), positionDisplacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        positionCounts[r] = counts[r] * 3;
        positionDisplacements[r] = displacements[r] * 3;
    }
    std::vector<Vec3> positions(numBodies);
    MPI_Datatype contiguousPositionType, positionType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &contiguousPositionType);
    MPI_Type_create_resized(contiguousPositionType, 0, sizeof(Body), &positionType);
    MPI_Type_free(&contiguousPositionType);
    MPI_Type_commit(&positionType);
    auto synchronizePositions = [&]() {
        MPI_Allgatherv(bodies.data(), localCount, positionType, positions.data(),
                       positionCounts.data(), positionDisplacements.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    };
    synchronizePositions();
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, positions);
        integrateBodies(bodies);
        synchronizePositions();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    long long localMilliseconds = duration.count(), maxMilliseconds = 0;
    MPI_Reduce(&localMilliseconds, &maxMilliseconds, 1, MPI_LONG_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %lld ms\n", maxMilliseconds);
    
    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        if (rank == 0) bodyData.resize(static_cast<size_t>(numBodies) * 6);
        MPI_Gatherv(bodies.data(), localCount * 6, MPI_DOUBLE,
                    rank == 0 ? bodyData.data() : nullptr, stateCounts.data(),
                    stateDisplacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(bodyData, "Bodies");
    }
    
    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        
        const int localValid = validateSimulation(bodies) ? 1 : 0;
        int globallyValid = 0;
        MPI_Allreduce(&localValid, &globallyValid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (globallyValid) {
            // Report final energy for reference
            const double localEnergy = computeTotalEnergy(bodies, positions, firstBody);
            double finalEnergy = 0.0;
            MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                printf("Validation failed: found invalid body state\n");
                printf("Validation: FAILED\n");
            }
            MPI_Type_free(&positionType);
            MPI_Finalize();
            return 1;
        }
    }
    MPI_Type_free(&positionType);
    MPI_Finalize();
    return 0;
}
