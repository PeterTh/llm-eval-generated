#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

static_assert(sizeof(Vec3) == 3 * sizeof(double));
static_assert(sizeof(Body) == 6 * sizeof(double));

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

// Compute forces for the local, contiguous range of target bodies.  The
// positions array contains all bodies in their original global order, so the
// inner-loop traversal and floating-point operation order remain unchanged.
void computeForces(std::vector<Body>& bodies, const std::vector<double>& positions) {
    const size_t n = positions.size() / 3;
    const size_t localCount = bodies.size();
    
    for (size_t i = 0; i < localCount; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        const double xi = bodies[i].pos.x;
        const double yi = bodies[i].pos.y;
        const double zi = bodies[i].pos.z;
        
        for (size_t j = 0; j < n; ++j) {
            const double dx = positions[3 * j] - xi;
            const double dy = positions[3 * j + 1] - yi;
            const double dz = positions[3 * j + 2] - zi;
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

void packPositions(const std::vector<Body>& bodies, std::vector<double>& positions) {
    for (size_t i = 0; i < bodies.size(); ++i) {
        positions[3 * i] = bodies[i].pos.x;
        positions[3 * i + 1] = bodies[i].pos.y;
        positions[3 * i + 2] = bodies[i].pos.z;
    }
}

void gatherBodies(const std::vector<Body>& localBodies,
                  std::vector<Body>& bodies,
                  const std::vector<int>& bodyCounts,
                  const std::vector<int>& bodyDisplacements,
                  int rank) {
    const int localCount = static_cast<int>(localBodies.size());
    const int localDoubleCount = localCount * 6;
    MPI_Gatherv(localBodies.empty() ? nullptr : static_cast<const void*>(localBodies.data()),
                localDoubleCount,
                MPI_DOUBLE,
                rank == 0 && !bodies.empty() ? static_cast<void*>(bodies.data()) : nullptr,
                rank == 0 ? bodyCounts.data() : nullptr,
                rank == 0 ? bodyDisplacements.data() : nullptr,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);
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

    int rank = 0;
    int worldSize = 1;
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

    if (numBodies < 0 || numSteps < 0 || numBodies > std::numeric_limits<int>::max() / 6) {
        if (rank == 0) {
            printf("Number of bodies and steps must be non-negative, and the body count is too large.\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Split the global body array into contiguous ranges.  Keeping this
    // layout fixed makes the final gather retain the original body order.
    std::vector<int> bodyCounts(worldSize);
    std::vector<int> bodyDisplacements(worldSize);
    const int bodiesPerRank = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    for (int process = 0; process < worldSize; ++process) {
        bodyCounts[process] = bodiesPerRank + (process < remainder ? 1 : 0);
        bodyDisplacements[process] = process == 0
            ? 0
            : bodyDisplacements[process - 1] + bodyCounts[process - 1];
    }

    std::vector<int> positionCounts(worldSize);
    std::vector<int> positionDisplacements(worldSize);
    for (int process = 0; process < worldSize; ++process) {
        positionCounts[process] = bodyCounts[process] * 3;
        positionDisplacements[process] = bodyDisplacements[process] * 3;
    }
    
    std::vector<Body> bodies;
    if (rank == 0) {
        bodies.resize(numBodies);
        randomizeBodies(bodies);
    }

    const int localCount = bodyCounts[rank];
    std::vector<Body> localBodies(localCount);
    std::vector<int> bodyDoubleCounts(worldSize);
    std::vector<int> bodyDoubleDisplacements(worldSize);
    for (int process = 0; process < worldSize; ++process) {
        bodyDoubleCounts[process] = bodyCounts[process] * 6;
        bodyDoubleDisplacements[process] = bodyDisplacements[process] * 6;
    }
    MPI_Scatterv(rank == 0 && !bodies.empty() ? static_cast<const void*>(bodies.data()) : nullptr,
                 bodyDoubleCounts.data(),
                 bodyDoubleDisplacements.data(),
                 MPI_DOUBLE,
                 localBodies.empty() ? nullptr : static_cast<void*>(localBodies.data()),
                 localCount * 6,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    std::vector<double> localPositions(3 * localCount);
    std::vector<double> globalPositions(3 * numBodies);
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (int step = 0; step < numSteps; ++step) {
        packPositions(localBodies, localPositions);
        MPI_Allgatherv(localPositions.empty() ? nullptr : localPositions.data(),
                       localCount * 3,
                       MPI_DOUBLE,
                       globalPositions.empty() ? nullptr : globalPositions.data(),
                       positionCounts.data(),
                       positionDisplacements.data(),
                       MPI_DOUBLE,
                       MPI_COMM_WORLD);
        computeForces(localBodies, globalPositions);
        integrateBodies(localBodies);
    }
    
    const double localDuration = MPI_Wtime() - start;
    double simulationDuration = 0.0;
    MPI_Reduce(&localDuration, &simulationDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Restore the complete global state on rank 0 for the original output
    // and validation paths.  This is outside the timed timestep loop.
    if (rank == 0) {
        bodies.resize(numBodies);
    }
    gatherBodies(localBodies, bodies, bodyDoubleCounts, bodyDoubleDisplacements, rank);
    
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", static_cast<long>(simulationDuration * 1000.0));
    }
    
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
    int exitCode = 0;
    if (rank == 0 && validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }
    
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
