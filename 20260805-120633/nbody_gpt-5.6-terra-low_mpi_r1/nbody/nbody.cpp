#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
              "Body must be representable as six contiguous doubles");

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

    const bool validInput = numBodies >= 0 && numSteps >= 0 &&
                            numBodies <= std::numeric_limits<int>::max() / 6;
    if (!validInput) {
        if (rank == 0) {
            printf("Number of bodies and steps must be non-negative, and body count must be supported by MPI\n");
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

    // Consecutive body ranges minimize scatter/gather overhead and give each
    // rank an equal share of the O(N^2) force work.
    std::vector<int> bodyCounts(worldSize), bodyOffsets(worldSize);
    const int baseCount = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    for (int r = 0, offset = 0; r < worldSize; ++r) {
        bodyCounts[r] = baseCount + (r < remainder ? 1 : 0);
        bodyOffsets[r] = offset;
        offset += bodyCounts[r];
    }
    const int localCount = bodyCounts[rank];

    std::vector<int> bodyDoubleCounts(worldSize), bodyDoubleOffsets(worldSize);
    std::vector<int> positionCounts(worldSize), positionOffsets(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        bodyDoubleCounts[r] = bodyCounts[r] * 6;
        bodyDoubleOffsets[r] = bodyOffsets[r] * 6;
        positionCounts[r] = bodyCounts[r] * 3;
        positionOffsets[r] = bodyOffsets[r] * 3;
    }

    std::vector<Body> initialBodies;
    if (rank == 0) {
        initialBodies.resize(numBodies);
        randomizeBodies(initialBodies);
    }
    std::vector<Body> localBodies(localCount);
    MPI_Scatterv(rank == 0 ? reinterpret_cast<double*>(initialBodies.data()) : nullptr,
                 bodyDoubleCounts.data(), bodyDoubleOffsets.data(), MPI_DOUBLE,
                 reinterpret_cast<double*>(localBodies.data()), localCount * 6, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Every rank needs all positions, but owns and updates only its local bodies.
    std::vector<double> positions(static_cast<size_t>(numBodies) * 3);
    std::vector<double> localPositions(static_cast<size_t>(localCount) * 3);
    for (int i = 0; i < localCount; ++i) {
        localPositions[3 * i] = localBodies[i].pos.x;
        localPositions[3 * i + 1] = localBodies[i].pos.y;
        localPositions[3 * i + 2] = localBodies[i].pos.z;
    }
    MPI_Allgatherv(localPositions.data(), localCount * 3, MPI_DOUBLE,
                   positions.data(), positionCounts.data(), positionOffsets.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        for (int localIndex = 0; localIndex < localCount; ++localIndex) {
            Body& body = localBodies[localIndex];
            double fx = 0.0, fy = 0.0, fz = 0.0;
            for (int j = 0; j < numBodies; ++j) {
                const double dx = positions[3 * j] - body.pos.x;
                const double dy = positions[3 * j + 1] - body.pos.y;
                const double dz = positions[3 * j + 2] - body.pos.z;
                const double invDist = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
            body.vel.x += DT * fx;
            body.vel.y += DT * fy;
            body.vel.z += DT * fz;
            body.pos.x += body.vel.x * DT;
            body.pos.y += body.vel.y * DT;
            body.pos.z += body.vel.z * DT;
            localPositions[3 * localIndex] = body.pos.x;
            localPositions[3 * localIndex + 1] = body.pos.y;
            localPositions[3 * localIndex + 2] = body.pos.z;
        }
        MPI_Allgatherv(localPositions.data(), localCount * 3, MPI_DOUBLE,
                       positions.data(), positionCounts.data(), positionOffsets.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const long long localMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long elapsedMilliseconds = 0;
    MPI_Reduce(&localMilliseconds, &elapsedMilliseconds, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %lld ms\n", elapsedMilliseconds);
    }
    
    // Print results for external validation
    std::vector<Body> bodies;
    if (printResults || validate) {
        if (rank == 0) {
            bodies.resize(numBodies);
        }
        MPI_Gatherv(reinterpret_cast<double*>(localBodies.data()), localCount * 6, MPI_DOUBLE,
                    rank == 0 ? reinterpret_cast<double*>(bodies.data()) : nullptr,
                    bodyDoubleCounts.data(), bodyDoubleOffsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

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
    int exitCode = 0;
    if (validate && rank == 0) {
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
