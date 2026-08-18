#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static_assert(std::is_standard_layout_v<Vec3> && sizeof(Vec3) == 3 * sizeof(double));
static_assert(std::is_standard_layout_v<Body> && sizeof(Body) == 6 * sizeof(double));

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

// Each rank updates a contiguous global range.  Positions are replicated, but
// velocities remain distributed, so the O(N^2) force work is divided by rank.
// Keeping the j loop in increasing global-index order preserves the original
// floating-point accumulation order for every body.
void computeForces(std::vector<Body>& localBodies, const std::vector<Vec3>& positions) {
    const int numBodies = static_cast<int>(positions.size());

    for (Body& body : localBodies) {
        double forceX = 0.0;
        double forceY = 0.0;
        double forceZ = 0.0;
        const Vec3 position = body.pos;

        for (int j = 0; j < numBodies; ++j) {
            const Vec3& otherPosition = positions[j];
            const double dx = otherPosition.x - position.x;
            const double dy = otherPosition.y - position.y;
            const double dz = otherPosition.z - position.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            forceX += dx * invDist3;
            forceY += dy * invDist3;
            forceZ += dz * invDist3;
        }

        body.vel.x += DT * forceX;
        body.vel.y += DT * forceY;
        body.vel.z += DT * forceZ;
    }
}

void integrateBodies(std::vector<Body>& localBodies) {
    for (Body& body : localBodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

enum ValidationFailure {
    VALID = 0,
    NONFINITE_VALUE = 1,
    POSITION_OUT_OF_BOUNDS = 2,
    VELOCITY_OUT_OF_BOUNDS = 3,
};

ValidationFailure validateSimulation(const std::vector<Body>& localBodies) {
    for (const Body& body : localBodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            return NONFINITE_VALUE;
        }

        constexpr double maxPosition = 1e6;
        constexpr double maxVelocity = 1e6;
        if (std::abs(body.pos.x) > maxPosition || std::abs(body.pos.y) > maxPosition ||
            std::abs(body.pos.z) > maxPosition) {
            return POSITION_OUT_OF_BOUNDS;
        }
        if (std::abs(body.vel.x) > maxVelocity || std::abs(body.vel.y) > maxVelocity ||
            std::abs(body.vel.z) > maxVelocity) {
            return VELOCITY_OUT_OF_BOUNDS;
        }
    }
    return VALID;
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;

    for (const Body& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }

    const int numBodies = static_cast<int>(bodies.size());
    for (int i = 0; i < numBodies; ++i) {
        for (int j = i + 1; j < numBodies; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double distance = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / distance;
        }
    }

    return energy;
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;

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
            exitCode = 1;
        }
    }

    if (numBodies < 0) {
        if (rank == 0) {
            std::printf("Number of bodies must be non-negative\n");
        }
        exitCode = 1;
    }

    if (exitCode != 0) {
        MPI_Finalize();
        return exitCode;
    }

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> bodiesPerRank(numRanks);
    std::vector<int> bodyOffsets(numRanks);
    const int bodiesPerRankBase = numBodies / numRanks;
    const int remainder = numBodies % numRanks;
    int offset = 0;
    for (int process = 0; process < numRanks; ++process) {
        bodiesPerRank[process] = bodiesPerRankBase + (process < remainder ? 1 : 0);
        bodyOffsets[process] = offset;
        offset += bodiesPerRank[process];
    }

    MPI_Datatype vec3Type;
    MPI_Datatype bodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);

    std::vector<Body> initialBodies;
    if (rank == 0) {
        initialBodies.resize(numBodies);
        randomizeBodies(initialBodies);
    }

    const int localBodyCount = bodiesPerRank[rank];
    std::vector<Body> localBodies(localBodyCount);
    MPI_Scatterv(rank == 0 ? initialBodies.data() : nullptr,
                 bodiesPerRank.data(),
                 bodyOffsets.data(),
                 bodyType,
                 localBodies.data(),
                 localBodyCount,
                 bodyType,
                 0,
                 MPI_COMM_WORLD);

    std::vector<Vec3> localPositions(localBodyCount);
    std::vector<Vec3> positions(numBodies);

    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        for (int localIndex = 0; localIndex < localBodyCount; ++localIndex) {
            localPositions[localIndex] = localBodies[localIndex].pos;
        }

        MPI_Allgatherv(localPositions.data(),
                       localBodyCount,
                       vec3Type,
                       positions.data(),
                       bodiesPerRank.data(),
                       bodyOffsets.data(),
                       vec3Type,
                       MPI_COMM_WORLD);
        computeForces(localBodies, positions);
        integrateBodies(localBodies);
    }

    const double localDuration = MPI_Wtime() - startTime;
    double maxDuration = 0.0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Simulation time: %ld ms\n", static_cast<long>(maxDuration * 1000.0));
    }

    std::vector<Body> gatheredBodies;
    if (printResults || validate) {
        if (rank == 0) {
            gatheredBodies.resize(numBodies);
        }
        MPI_Gatherv(localBodies.data(),
                    localBodyCount,
                    bodyType,
                    rank == 0 ? gatheredBodies.data() : nullptr,
                    bodiesPerRank.data(),
                    bodyOffsets.data(),
                    bodyType,
                    0,
                    MPI_COMM_WORLD);

    }

    if (printResults && rank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
        for (const Body& body : gatheredBodies) {
            bodyData.push_back(body.pos.x);
            bodyData.push_back(body.pos.y);
            bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x);
            bodyData.push_back(body.vel.y);
            bodyData.push_back(body.vel.z);
        }
        print_results(bodyData, "Bodies");
    }

    if (validate) {
        if (rank == 0) {
            std::printf("Validating simulation results...\n");
            const ValidationFailure failure = validateSimulation(gatheredBodies);
            if (failure == VALID) {
                // Validation is deliberately outside the timed region.  Use
                // the original serial summation order so its reference value
                // is independent of MPI rank count.
                const double finalEnergy = computeTotalEnergy(gatheredBodies);
                std::printf("Final energy: %.6f\n", finalEnergy);
                std::printf("Validation: PASSED\n");
            } else {
                switch (failure) {
                    case NONFINITE_VALUE:
                        std::printf("Validation failed: found NaN or Inf value in body state\n");
                        break;
                    case POSITION_OUT_OF_BOUNDS:
                        std::printf("Validation failed: body position exceeds reasonable bounds\n");
                        break;
                    case VELOCITY_OUT_OF_BOUNDS:
                        std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
                        break;
                    default:
                        break;
                }
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Type_free(&bodyType);
    MPI_Type_free(&vec3Type);
    MPI_Finalize();
    return exitCode;
}
