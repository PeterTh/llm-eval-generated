#include <algorithm>
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
constexpr int STATE_COMPONENTS = 6;
constexpr int POSITION_COMPONENTS = 3;

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
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

void packBodies(const std::vector<Body>& bodies, std::vector<double>& packed) {
    packed.resize(bodies.size() * STATE_COMPONENTS);
    for (size_t i = 0; i < bodies.size(); ++i) {
        const Body& body = bodies[i];
        double* const destination = packed.data() + i * STATE_COMPONENTS;
        destination[0] = body.pos.x;
        destination[1] = body.pos.y;
        destination[2] = body.pos.z;
        destination[3] = body.vel.x;
        destination[4] = body.vel.y;
        destination[5] = body.vel.z;
    }
}

void unpackBodies(const std::vector<double>& packed, std::vector<Body>& bodies) {
    for (size_t i = 0; i < bodies.size(); ++i) {
        const double* const source = packed.data() + i * STATE_COMPONENTS;
        Body& body = bodies[i];
        body.pos.x = source[0];
        body.pos.y = source[1];
        body.pos.z = source[2];
        body.vel.x = source[3];
        body.vel.y = source[4];
        body.vel.z = source[5];
    }
}

void packPositions(const std::vector<Body>& bodies, std::vector<double>& packed) {
    packed.resize(bodies.size() * POSITION_COMPONENTS);
    for (size_t i = 0; i < bodies.size(); ++i) {
        const Body& body = bodies[i];
        double* const destination = packed.data() + i * POSITION_COMPONENTS;
        destination[0] = body.pos.x;
        destination[1] = body.pos.y;
        destination[2] = body.pos.z;
    }
}

void exchangePositions(const std::vector<Body>& localBodies,
                       std::vector<double>& localPositions,
                       std::vector<double>& globalPositions,
                       const std::vector<int>& positionCounts,
                       const std::vector<int>& positionDisplacements,
                       MPI_Comm communicator) {
    packPositions(localBodies, localPositions);
    MPI_Allgatherv(localPositions.empty() ? nullptr : localPositions.data(),
                   static_cast<int>(localPositions.size()),
                   MPI_DOUBLE,
                   globalPositions.empty() ? nullptr : globalPositions.data(),
                   positionCounts.data(),
                   positionDisplacements.data(),
                   MPI_DOUBLE,
                   communicator);
}

void computeForces(std::vector<Body>& localBodies,
                   const std::vector<double>& globalPositions) {
    const int numBodies = static_cast<int>(globalPositions.size() / POSITION_COMPONENTS);
    const double* const positions = globalPositions.data();

    for (Body& body : localBodies) {
        const double x = body.pos.x;
        const double y = body.pos.y;
        const double z = body.pos.z;
        double Fx = 0.0;
        double Fy = 0.0;
        double Fz = 0.0;

        // The global position array is ordered by body index, exactly matching
        // the interaction order used by the serial implementation.
        for (int j = 0; j < numBodies; ++j) {
            const double* const other = positions + j * POSITION_COMPONENTS;
            const double dx = other[0] - x;
            const double dy = other[1] - y;
            const double dz = other[2] - z;
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
    for (Body& body : bodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& localBodies,
                          const std::vector<double>& globalPositions,
                          int globalFirstBody,
                          MPI_Comm communicator) {
    double localEnergy = 0.0;

    for (const Body& body : localBodies) {
        localEnergy += 0.5 * (body.vel.x * body.vel.x +
                              body.vel.y * body.vel.y +
                              body.vel.z * body.vel.z);
    }

    const int numBodies = static_cast<int>(globalPositions.size() / POSITION_COMPONENTS);
    const double* const positions = globalPositions.data();
    for (int localIndex = 0; localIndex < static_cast<int>(localBodies.size()); ++localIndex) {
        const int globalIndex = globalFirstBody + localIndex;
        const double* const bodyPosition = positions + globalIndex * POSITION_COMPONENTS;
        for (int j = globalIndex + 1; j < numBodies; ++j) {
            const double* const otherPosition = positions + j * POSITION_COMPONENTS;
            const double dx = otherPosition[0] - bodyPosition[0];
            const double dy = otherPosition[1] - bodyPosition[1];
            const double dz = otherPosition[2] - bodyPosition[2];
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            localEnergy -= 1.0 / dist;
        }
    }

    double totalEnergy = 0.0;
    MPI_Reduce(&localEnergy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, communicator);
    return totalEnergy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const Body& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            return false;
        }

        constexpr double maxPosition = 1e6;
        constexpr double maxVelocity = 1e6;
        if (std::abs(body.pos.x) > maxPosition || std::abs(body.pos.y) > maxPosition ||
            std::abs(body.pos.z) > maxPosition || std::abs(body.vel.x) > maxVelocity ||
            std::abs(body.vel.y) > maxVelocity || std::abs(body.vel.z) > maxVelocity) {
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
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

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

    // MPI collective counts are integers.  The original state representation
    // needs six doubles per body, so reject inputs MPI cannot represent safely.
    if (numBodies < 0 || numSteps < 0 || numBodies > std::numeric_limits<int>::max() / STATE_COMPONENTS) {
        if (rank == 0) {
            printf("Number of bodies and steps must be non-negative and representable by MPI\n");
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

    std::vector<int> bodyCounts(ranks);
    std::vector<int> bodyDisplacements(ranks);
    const int bodiesPerRank = numBodies / ranks;
    const int remainder = numBodies % ranks;
    int nextBody = 0;
    for (int process = 0; process < ranks; ++process) {
        bodyCounts[process] = bodiesPerRank + (process < remainder ? 1 : 0);
        bodyDisplacements[process] = nextBody;
        nextBody += bodyCounts[process];
    }

    std::vector<int> stateCounts(ranks);
    std::vector<int> stateDisplacements(ranks);
    std::vector<int> positionCounts(ranks);
    std::vector<int> positionDisplacements(ranks);
    for (int process = 0; process < ranks; ++process) {
        stateCounts[process] = bodyCounts[process] * STATE_COMPONENTS;
        stateDisplacements[process] = bodyDisplacements[process] * STATE_COMPONENTS;
        positionCounts[process] = bodyCounts[process] * POSITION_COMPONENTS;
        positionDisplacements[process] = bodyDisplacements[process] * POSITION_COMPONENTS;
    }

    const int localBodyCount = bodyCounts[rank];
    const int globalFirstBody = bodyDisplacements[rank];
    std::vector<Body> localBodies(static_cast<size_t>(localBodyCount));
    std::vector<double> localState(static_cast<size_t>(localBodyCount) * STATE_COMPONENTS);
    std::vector<double> initialState;
    if (rank == 0) {
        std::vector<Body> initialBodies(static_cast<size_t>(numBodies));
        randomizeBodies(initialBodies);
        packBodies(initialBodies, initialState);
    }

    MPI_Scatterv(rank == 0 && !initialState.empty() ? initialState.data() : nullptr,
                 stateCounts.data(),
                 stateDisplacements.data(),
                 MPI_DOUBLE,
                 localState.empty() ? nullptr : localState.data(),
                 static_cast<int>(localState.size()),
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);
    unpackBodies(localState, localBodies);
    std::vector<double>().swap(initialState);
    std::vector<double>().swap(localState);

    std::vector<double> localPositions;
    std::vector<double> globalPositions(static_cast<size_t>(numBodies) * POSITION_COMPONENTS);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        exchangePositions(localBodies, localPositions, globalPositions, positionCounts, positionDisplacements, MPI_COMM_WORLD);
        computeForces(localBodies, globalPositions);
        integrateBodies(localBodies);
    }
    const double localDuration = MPI_Wtime() - start;
    double simulationDuration = 0.0;
    MPI_Reduce(&localDuration, &simulationDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", static_cast<long>(simulationDuration * 1000.0));
    }

    if (printResults) {
        packBodies(localBodies, localState);
        std::vector<double> globalState;
        if (rank == 0) {
            globalState.resize(static_cast<size_t>(numBodies) * STATE_COMPONENTS);
        }
        MPI_Gatherv(localState.empty() ? nullptr : localState.data(),
                    static_cast<int>(localState.size()),
                    MPI_DOUBLE,
                    rank == 0 && !globalState.empty() ? globalState.data() : nullptr,
                    stateCounts.data(),
                    stateDisplacements.data(),
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(globalState, "Bodies");
        }
    }

    int localValid = validateSimulation(localBodies) ? 1 : 0;
    int simulationValid = 0;
    if (validate) {
        MPI_Allreduce(&localValid, &simulationValid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (simulationValid) {
            exchangePositions(localBodies, localPositions, globalPositions, positionCounts, positionDisplacements, MPI_COMM_WORLD);
            const double finalEnergy = computeTotalEnergy(localBodies, globalPositions, globalFirstBody, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Validating simulation results...\n");
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else if (rank == 0) {
            printf("Validating simulation results...\n");
            printf("Validation failed: found invalid or unreasonable body state\n");
            printf("Validation: FAILED\n");
        }
    }

    MPI_Finalize();
    return validate && !simulationValid ? 1 : 0;
}
