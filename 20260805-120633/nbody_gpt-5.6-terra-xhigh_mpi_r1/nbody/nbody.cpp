#include <cmath>
#include <cstddef>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
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

static_assert(std::is_standard_layout_v<Vec3>);
static_assert(std::is_standard_layout_v<Body>);
static_assert(sizeof(Vec3) == 3 * sizeof(double));
static_assert(offsetof(Body, pos) == 0);
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

void computeForces(const std::vector<Vec3>& localPositions, std::vector<Vec3>& localVelocities,
                   const std::vector<Vec3>& allPositions) {
    const size_t numLocalBodies = localPositions.size();
    const size_t numBodies = allPositions.size();
    const Vec3* const positions = allPositions.data();

    for (size_t i = 0; i < numLocalBodies; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        const Vec3 position = localPositions[i];
        for (size_t j = 0; j < numBodies; ++j) {
            const double dx = positions[j].x - position.x;
            const double dy = positions[j].y - position.y;
            const double dz = positions[j].z - position.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        localVelocities[i].x += DT * Fx;
        localVelocities[i].y += DT * Fy;
        localVelocities[i].z += DT * Fz;
    }
}

void integrateBodies(std::vector<Vec3>& positions, const std::vector<Vec3>& velocities) {
    const size_t numBodies = positions.size();
    for (size_t i = 0; i < numBodies; ++i) {
        positions[i].x += velocities[i].x * DT;
        positions[i].y += velocities[i].y * DT;
        positions[i].z += velocities[i].z * DT;
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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

    if (numBodies < 0 || numSteps < 0 || numBodies > INT_MAX / 3) {
        if (rank == 0) {
            printf("Number of bodies and simulation steps must be non-negative, and the body count must fit MPI counts\n");
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

    // Assign each rank a contiguous global range.  This keeps the gather order
    // identical to the original single-process source-body ordering.
    std::vector<int> bodyCounts(numRanks);
    std::vector<int> bodyDisplacements(numRanks);
    std::vector<int> positionCounts(numRanks);
    std::vector<int> positionDisplacements(numRanks);
    const int baseCount = numBodies / numRanks;
    const int remainder = numBodies % numRanks;
    int displacement = 0;
    for (int process = 0; process < numRanks; ++process) {
        const int count = baseCount + (process < remainder ? 1 : 0);
        bodyCounts[process] = count;
        bodyDisplacements[process] = displacement;
        positionCounts[process] = 3 * count;
        positionDisplacements[process] = 3 * displacement;
        displacement += count;
    }

    MPI_Datatype bodyType;
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);

    // Initialization remains on rank zero so that the fixed seed produces the
    // same global initial state as the original program.
    std::vector<Body> initialBodies;
    if (rank == 0) {
        initialBodies.resize(static_cast<size_t>(numBodies));
        randomizeBodies(initialBodies);
    }

    const int numLocalBodies = bodyCounts[rank];
    std::vector<Body> localInitialBodies(static_cast<size_t>(numLocalBodies));
    MPI_Scatterv(rank == 0 ? initialBodies.data() : nullptr,
                 bodyCounts.data(), bodyDisplacements.data(), bodyType,
                 localInitialBodies.data(), numLocalBodies, bodyType, 0, MPI_COMM_WORLD);

    // Keep the repeatedly communicated and updated state in contiguous arrays.
    // This lets MPI send positions directly rather than packing a strided field
    // from an array of Body objects on every timestep.
    std::vector<Vec3> localPositions(static_cast<size_t>(numLocalBodies));
    std::vector<Vec3> localVelocities(static_cast<size_t>(numLocalBodies));
    for (int i = 0; i < numLocalBodies; ++i) {
        localPositions[static_cast<size_t>(i)] = localInitialBodies[static_cast<size_t>(i)].pos;
        localVelocities[static_cast<size_t>(i)] = localInitialBodies[static_cast<size_t>(i)].vel;
    }
    std::vector<Body>().swap(localInitialBodies);
    std::vector<Body>().swap(initialBodies);

    std::vector<Vec3> allPositions(static_cast<size_t>(numBodies));
    auto gatherPositions = [&]() {
        MPI_Allgatherv(localPositions.data(), 3 * numLocalBodies, MPI_DOUBLE,
                       allPositions.data(), positionCounts.data(), positionDisplacements.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    };

    // Run simulation.  Position replication is O(N) per timestep, while the
    // direct force work is distributed as O(N^2 / P).
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (numSteps > 0) {
        gatherPositions();
    }
    for (int step = 0; step < numSteps; ++step) {
        computeForces(localPositions, localVelocities, allPositions);
        integrateBodies(localPositions, localVelocities);
        if (step + 1 < numSteps) {
            gatherPositions();
        }
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", static_cast<long>(duration * 1000.0));
    }

    std::vector<Body> finalBodies;
    if ((printResults || validate) && rank == 0) {
        finalBodies.resize(static_cast<size_t>(numBodies));
    }
    if (printResults || validate) {
        std::vector<Body> localFinalBodies(static_cast<size_t>(numLocalBodies));
        for (int i = 0; i < numLocalBodies; ++i) {
            localFinalBodies[static_cast<size_t>(i)].pos = localPositions[static_cast<size_t>(i)];
            localFinalBodies[static_cast<size_t>(i)].vel = localVelocities[static_cast<size_t>(i)];
        }
        MPI_Gatherv(localFinalBodies.data(), numLocalBodies, bodyType,
                    rank == 0 ? finalBodies.data() : nullptr,
                    bodyCounts.data(), bodyDisplacements.data(), bodyType, 0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
        for (const auto& body : finalBodies) {
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

        if (validateSimulation(finalBodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(finalBodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Type_free(&bodyType);
    MPI_Finalize();
    return exitCode;
}
