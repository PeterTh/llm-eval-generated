#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

static_assert(std::is_standard_layout_v<Vec3> && sizeof(Vec3) == 3 * sizeof(double));
static_assert(std::is_standard_layout_v<Body> && sizeof(Body) == 6 * sizeof(double));
static_assert(offsetof(Body, pos) == 0 && offsetof(Body, vel) == sizeof(Vec3));

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

// Stage each rank's current positions in its own receive-buffer segment, then
// use an in-place collective so the network payload is tightly packed Vec3s.
void exchangePositions(const std::vector<Body>& localBodies,
                       std::vector<Vec3>& allPositions,
                       const std::vector<int>& bodyCounts,
                       const std::vector<int>& bodyDisplacements,
                       int localStart,
                       MPI_Datatype vec3Type) {
    if (!localBodies.empty()) {
        Vec3* const localPositions = allPositions.data() + localStart;
        for (size_t body = 0; body < localBodies.size(); ++body) {
            localPositions[body] = localBodies[body].pos;
        }
    }

    void* receiveBuffer = allPositions.empty() ? nullptr : allPositions.data();

    MPI_Allgatherv(MPI_IN_PLACE,
                   0,
                   MPI_DATATYPE_NULL,
                   receiveBuffer,
                   bodyCounts.data(),
                   bodyDisplacements.data(),
                   vec3Type,
                   MPI_COMM_WORLD);
}

void computeForces(std::vector<Body>& localBodies, const std::vector<Vec3>& allPositions) {
    const Vec3* const positions = allPositions.data();
    const size_t numBodies = allPositions.size();

    for (Body& body : localBodies) {
        const double x = body.pos.x;
        const double y = body.pos.y;
        const double z = body.pos.z;
        double Fx = 0.0;
        double Fy = 0.0;
        double Fz = 0.0;

        // All ranks traverse the globally gathered positions in the original
        // index order, preserving the per-body force calculation semantics.
        for (size_t j = 0; j < numBodies; ++j) {
            const double dx = positions[j].x - x;
            const double dy = positions[j].y - y;
            const double dz = positions[j].z - z;
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

void integrateBodies(std::vector<Body>& localBodies) {
    for (Body& body : localBodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

double computeLocalEnergy(const std::vector<Body>& localBodies,
                          const std::vector<Vec3>& allPositions,
                          int globalStart) {
    double energy = 0.0;

    for (const Body& body : localBodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                         body.vel.y * body.vel.y +
                         body.vel.z * body.vel.z);
    }

    // Each pair is owned by the rank that owns its lower global index.
    for (size_t localIndex = 0; localIndex < localBodies.size(); ++localIndex) {
        const Body& body = localBodies[localIndex];
        const size_t globalIndex = static_cast<size_t>(globalStart) + localIndex;
        for (size_t j = globalIndex + 1; j < allPositions.size(); ++j) {
            const double dx = allPositions[j].x - body.pos.x;
            const double dy = allPositions[j].y - body.pos.y;
            const double dz = allPositions[j].z - body.pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

// Returns zero when valid, otherwise a code that identifies the failed check.
int validationFailure(const std::vector<Body>& localBodies) {
    for (const Body& body : localBodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            return 1;
        }

        constexpr double maxPosition = 1e6;
        constexpr double maxVelocity = 1e6;
        if (std::abs(body.pos.x) > maxPosition || std::abs(body.pos.y) > maxPosition ||
            std::abs(body.pos.z) > maxPosition) {
            return 2;
        }
        if (std::abs(body.vel.x) > maxVelocity || std::abs(body.vel.y) > maxVelocity ||
            std::abs(body.vel.z) > maxVelocity) {
            return 3;
        }
    }
    return 0;
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

    // Settings are parsed once, then broadcast so every rank follows exactly
    // the same collective-control path.
    int settings[5] = {1024, 10, 0, 0, 0};
    // settings: body count, step count, validate, print results, exit status.
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                settings[0] = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                settings[1] = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                settings[2] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                settings[3] = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                settings[4] = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                settings[4] = 2;
                break;
            }
        }

        if (settings[4] == 0 &&
            (settings[0] < 0 || settings[1] < 0 || settings[0] > std::numeric_limits<int>::max() / 6)) {
            printf("Number of bodies and steps must be non-negative; body count must fit MPI counts\n");
            settings[4] = 2;
        }
    }

    MPI_Bcast(settings, 5, MPI_INT, 0, MPI_COMM_WORLD);
    if (settings[4] != 0) {
        MPI_Finalize();
        return settings[4] == 1 ? 0 : 1;
    }

    const int numBodies = settings[0];
    const int numSteps = settings[1];
    const bool validate = settings[2] != 0;
    const bool printResults = settings[3] != 0;

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> bodyCounts(numRanks);
    std::vector<int> bodyDisplacements(numRanks);
    const int bodiesPerRank = numBodies / numRanks;
    const int extraBodies = numBodies % numRanks;
    int bodyOffset = 0;
    for (int process = 0; process < numRanks; ++process) {
        bodyCounts[process] = bodiesPerRank + (process < extraBodies ? 1 : 0);
        bodyDisplacements[process] = bodyOffset;
        bodyOffset += bodyCounts[process];
    }

    std::vector<int> doubleCounts(numRanks);
    std::vector<int> doubleDisplacements(numRanks);
    for (int process = 0; process < numRanks; ++process) {
        doubleCounts[process] = bodyCounts[process] * 6;
        doubleDisplacements[process] = bodyDisplacements[process] * 6;
    }

    const int localBodyCount = bodyCounts[rank];
    const int localStart = bodyDisplacements[rank];
    std::vector<Body> localBodies(localBodyCount);

    // Rank zero initializes with the exact serial random sequence, then hands
    // each contiguous range to its owning rank.
    {
        std::vector<Body> initialBodies;
        if (rank == 0) {
            initialBodies.resize(numBodies);
            randomizeBodies(initialBodies);
        }
        const void* initialSendBuffer = rank == 0 && !initialBodies.empty() ? initialBodies.data() : nullptr;
        void* localReceiveBuffer = localBodies.empty() ? nullptr : localBodies.data();
        MPI_Scatterv(initialSendBuffer,
                     doubleCounts.data(),
                     doubleDisplacements.data(),
                     MPI_DOUBLE,
                     localReceiveBuffer,
                     localBodyCount * 6,
                     MPI_DOUBLE,
                     0,
                     MPI_COMM_WORLD);
    }

    MPI_Datatype vec3Type;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);

    std::vector<Vec3> allPositions(numBodies);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        exchangePositions(localBodies,
                          allPositions,
                          bodyCounts,
                          bodyDisplacements,
                          localStart,
                          vec3Type);
        computeForces(localBodies, allPositions);
        integrateBodies(localBodies);
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", static_cast<long>(duration * 1000.0));
    }

    if (printResults) {
        std::vector<double> bodyData;
        if (rank == 0) {
            bodyData.resize(static_cast<size_t>(numBodies) * 6);
        }

        void* resultReceiveBuffer = rank == 0 && !bodyData.empty() ? bodyData.data() : nullptr;
        MPI_Gatherv(localBodies.empty() ? nullptr : localBodies.data(),
                    localBodyCount * 6,
                    MPI_DOUBLE,
                    resultReceiveBuffer,
                    doubleCounts.data(),
                    doubleDisplacements.data(),
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(bodyData, "Bodies");
        }
    }

    int exitCode = 0;
    if (validate) {
        // The final integration changes positions, so synchronize them once
        // more before assigning potential-energy pairs to their owner ranks.
        exchangePositions(localBodies,
                          allPositions,
                          bodyCounts,
                          bodyDisplacements,
                          localStart,
                          vec3Type);

        const int localFailure = validationFailure(localBodies);
        int failure = 0;
        MPI_Allreduce(&localFailure, &failure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (failure == 0) {
            const double localEnergy = computeLocalEnergy(localBodies, allPositions, localStart);
            double finalEnergy = 0.0;
            MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                printf("Validating simulation results...\n");
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                printf("Validating simulation results...\n");
                if (failure == 1) {
                    printf("Validation failed: found NaN or Inf value in body state\n");
                } else if (failure == 2) {
                    printf("Validation failed: body position exceeds reasonable bounds\n");
                } else {
                    printf("Validation failed: body velocity exceeds reasonable bounds\n");
                }
                printf("Validation: FAILED\n");
            }
            exitCode = 1;
        }
    }

    MPI_Type_free(&vec3Type);
    MPI_Finalize();
    return exitCode;
}
