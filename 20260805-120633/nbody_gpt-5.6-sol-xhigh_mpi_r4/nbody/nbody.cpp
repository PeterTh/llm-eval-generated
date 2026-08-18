#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept
        : x(x), y(y), z(z) {}
};

static_assert(sizeof(Vec3) == 3 * sizeof(double));

// Rank zero follows exactly the same random-number call order as the serial
// implementation. Positions are replicated; velocities are scattered to the
// rank that owns each corresponding body.
void initializeBodies(std::vector<Vec3>& positions, std::vector<Vec3>& velocities,
                      unsigned int seed = 42) {
    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i].x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        positions[i].y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        positions[i].z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        velocities[i].x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        velocities[i].y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        velocities[i].z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

// Compute the forces for this rank's targets. Every target still visits source
// bodies in global index order, preserving the serial floating-point operation
// order. Integration is a separate pass so all forces use one position snapshot.
void advanceBodies(std::vector<Vec3>& positions, std::vector<Vec3>& localVelocities,
                   const int firstBody) {
    const int numBodies = static_cast<int>(positions.size());
    const int localBodies = static_cast<int>(localVelocities.size());
    const Vec3* const positionData = positions.data();

    for (int localIndex = 0; localIndex < localBodies; ++localIndex) {
        const int bodyIndex = firstBody + localIndex;
        const double px = positionData[bodyIndex].x;
        const double py = positionData[bodyIndex].y;
        const double pz = positionData[bodyIndex].z;
        double Fx = 0.0;
        double Fy = 0.0;
        double Fz = 0.0;

        for (int j = 0; j < numBodies; ++j) {
            const double dx = positionData[j].x - px;
            const double dy = positionData[j].y - py;
            const double dz = positionData[j].z - pz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        localVelocities[localIndex].x += DT * Fx;
        localVelocities[localIndex].y += DT * Fy;
        localVelocities[localIndex].z += DT * Fz;
    }

    for (int localIndex = 0; localIndex < localBodies; ++localIndex) {
        Vec3& position = positions[firstBody + localIndex];
        const Vec3& velocity = localVelocities[localIndex];
        position.x += velocity.x * DT;
        position.y += velocity.y * DT;
        position.z += velocity.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Vec3>& positions,
                          const std::vector<Vec3>& velocities) {
    double energy = 0.0;
    const size_t n = positions.size();

    for (const Vec3& velocity : velocities) {
        energy += 0.5 * (velocity.x * velocity.x + velocity.y * velocity.y +
                         velocity.z * velocity.z);
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = positions[j].x - positions[i].x;
            const double dy = positions[j].y - positions[i].y;
            const double dz = positions[j].z - positions[i].z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

bool validateSimulation(const std::vector<Vec3>& positions,
                        const std::vector<Vec3>& velocities) {
    for (size_t i = 0; i < positions.size(); ++i) {
        const Vec3& position = positions[i];
        const Vec3& velocity = velocities[i];
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z) || !std::isfinite(velocity.x) ||
            !std::isfinite(velocity.y) || !std::isfinite(velocity.z)) {
            std::printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        constexpr double maxPos = 1e6;
        constexpr double maxVel = 1e6;
        if (std::abs(position.x) > maxPos || std::abs(position.y) > maxPos ||
            std::abs(position.z) > maxPos) {
            std::printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(velocity.x) > maxVel || std::abs(velocity.y) > maxVel ||
            std::abs(velocity.z) > maxVel) {
            std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks energy conservation)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
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
    int argumentStatus = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            argumentStatus = 1;
            break;
        }
    }

    if (numBodies < 0) {
        if (rank == 0) {
            std::printf("Number of bodies must be non-negative\n");
        }
        argumentStatus = 1;
    }
    if (argumentStatus != 0) {
        MPI_Finalize();
        return argumentStatus;
    }

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> bodyCounts(worldSize);
    std::vector<int> bodyDisplacements(worldSize);
    const int bodiesPerRank = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    int displacement = 0;
    for (int process = 0; process < worldSize; ++process) {
        bodyCounts[process] = bodiesPerRank + (process < remainder ? 1 : 0);
        bodyDisplacements[process] = displacement;
        displacement += bodyCounts[process];
    }

    const int localBodyCount = bodyCounts[rank];
    const int firstBody = bodyDisplacements[rank];
    const bool evenlyDistributed = remainder == 0;
    std::vector<Vec3> positions(static_cast<size_t>(numBodies));
    std::vector<Vec3> initialVelocities;
    if (rank == 0) {
        initialVelocities.resize(static_cast<size_t>(numBodies));
        initializeBodies(positions, initialVelocities);
    }
    std::vector<Vec3> localVelocities(static_cast<size_t>(localBodyCount));

    MPI_Datatype vec3Type;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);

    MPI_Bcast(positions.data(), numBodies, vec3Type, 0, MPI_COMM_WORLD);
    if (evenlyDistributed) {
        MPI_Scatter(rank == 0 ? initialVelocities.data() : nullptr, localBodyCount, vec3Type,
                    localVelocities.data(), localBodyCount, vec3Type, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(rank == 0 ? initialVelocities.data() : nullptr, bodyCounts.data(),
                     bodyDisplacements.data(), vec3Type, localVelocities.data(), localBodyCount,
                     vec3Type, 0, MPI_COMM_WORLD);
    }
    initialVelocities.clear();
    initialVelocities.shrink_to_fit();

    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        advanceBodies(positions, localVelocities, firstBody);
        if (evenlyDistributed) {
            MPI_Allgather(MPI_IN_PLACE, 0, vec3Type, positions.data(), localBodyCount, vec3Type,
                          MPI_COMM_WORLD);
        } else {
            MPI_Allgatherv(MPI_IN_PLACE, 0, vec3Type, positions.data(), bodyCounts.data(),
                           bodyDisplacements.data(), vec3Type, MPI_COMM_WORLD);
        }
    }

    const double localElapsed = MPI_Wtime() - startTime;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMs = static_cast<long>(elapsed * 1000.0);
        std::printf("Simulation time: %ld ms\n", durationMs);
    }

    std::vector<Vec3> velocities;
    if (printResults || validate) {
        if (rank == 0) {
            velocities.resize(static_cast<size_t>(numBodies));
        }
        if (evenlyDistributed) {
            MPI_Gather(localVelocities.data(), localBodyCount, vec3Type,
                       rank == 0 ? velocities.data() : nullptr, localBodyCount, vec3Type, 0,
                       MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(localVelocities.data(), localBodyCount, vec3Type,
                        rank == 0 ? velocities.data() : nullptr, bodyCounts.data(),
                        bodyDisplacements.data(), vec3Type, 0, MPI_COMM_WORLD);
        }
    }

    int exitStatus = 0;
    if (rank == 0 && printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
        for (int i = 0; i < numBodies; ++i) {
            bodyData.push_back(positions[i].x);
            bodyData.push_back(positions[i].y);
            bodyData.push_back(positions[i].z);
            bodyData.push_back(velocities[i].x);
            bodyData.push_back(velocities[i].y);
            bodyData.push_back(velocities[i].z);
        }
        print_results(bodyData, "Bodies");
    }

    if (rank == 0 && validate) {
        std::printf("Validating simulation results...\n");
        if (validateSimulation(positions, velocities)) {
            const double finalEnergy = computeTotalEnergy(positions, velocities);
            std::printf("Final energy: %.6f\n", finalEnergy);
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            exitStatus = 1;
        }
    }

    MPI_Bcast(&exitStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Type_free(&vec3Type);
    MPI_Finalize();
    return exitStatus;
}
