#include <cmath>
#include <limits>
#include <mpi.h>
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

static_assert(std::is_standard_layout_v<Vec3> && std::is_trivially_copyable_v<Vec3>);
static_assert(sizeof(Vec3) == 3 * sizeof(double));

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

// Each rank owns a contiguous set of bodies.  The global position vector is a
// snapshot taken before this step, preserving the original force/update order
// while allowing all owned bodies to be advanced independently.
void advanceBodies(std::vector<Vec3>& localPositions,
                   std::vector<Vec3>& localVelocities,
                   const std::vector<Vec3>& globalPositions) {
    const size_t localCount = localPositions.size();
    const size_t globalCount = globalPositions.size();

    for (size_t i = 0; i < localCount; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        const Vec3 position = localPositions[i];

        for (size_t j = 0; j < globalCount; ++j) {
            const double dx = globalPositions[j].x - position.x;
            const double dy = globalPositions[j].y - position.y;
            const double dz = globalPositions[j].z - position.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        Vec3& velocity = localVelocities[i];
        velocity.x += DT * Fx;
        velocity.y += DT * Fy;
        velocity.z += DT * Fz;

        localPositions[i].x += velocity.x * DT;
        localPositions[i].y += velocity.y * DT;
        localPositions[i].z += velocity.z * DT;
    }
}

double computeLocalEnergy(const std::vector<Vec3>& localPositions,
                          const std::vector<Vec3>& localVelocities,
                          const std::vector<Vec3>& globalPositions,
                          const int globalOffset) {
    double energy = 0.0;

    // Kinetic energy (assuming unit mass)
    for (const Vec3& velocity : localVelocities) {
        energy += 0.5 * (velocity.x * velocity.x +
                         velocity.y * velocity.y +
                         velocity.z * velocity.z);
    }

    // Every pair belongs to exactly one rank: the owner of the lower index.
    for (size_t localIndex = 0; localIndex < localPositions.size(); ++localIndex) {
        const size_t globalIndex = static_cast<size_t>(globalOffset) + localIndex;
        for (size_t j = globalIndex + 1; j < globalPositions.size(); ++j) {
            const double dx = globalPositions[j].x - localPositions[localIndex].x;
            const double dy = globalPositions[j].y - localPositions[localIndex].y;
            const double dz = globalPositions[j].z - localPositions[localIndex].z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    
    return energy;
}

enum class ValidationFailure : int {
    None = 0,
    NonFinite = 1,
    PositionOutOfBounds = 2,
    VelocityOutOfBounds = 3,
};

// Validate each owned body and report the first local failure, if any.
ValidationFailure validateLocalSimulation(const std::vector<Vec3>& localPositions,
                                          const std::vector<Vec3>& localVelocities,
                                          int& failingLocalIndex) {
    for (size_t i = 0; i < localPositions.size(); ++i) {
        const Vec3& position = localPositions[i];
        const Vec3& velocity = localVelocities[i];
        // Check for NaN or Inf values
        if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z) ||
            !std::isfinite(velocity.x) || !std::isfinite(velocity.y) || !std::isfinite(velocity.z)) {
            failingLocalIndex = static_cast<int>(i);
            return ValidationFailure::NonFinite;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(position.x) > maxPos || std::abs(position.y) > maxPos || std::abs(position.z) > maxPos) {
            failingLocalIndex = static_cast<int>(i);
            return ValidationFailure::PositionOutOfBounds;
        }
        if (std::abs(velocity.x) > maxVel || std::abs(velocity.y) > maxVel || std::abs(velocity.z) > maxVel) {
            failingLocalIndex = static_cast<int>(i);
            return ValidationFailure::VelocityOutOfBounds;
        }
    }
    failingLocalIndex = -1;
    return ValidationFailure::None;
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
    int rankCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Only rank 0 reports argument errors and status; the parsed configuration
    // is then broadcast so every rank follows the identical control path.
    int argumentStatus = 0; // 0 = proceed, 1 = error, 2 = help requested
    if (rank == 0) {
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
                argumentStatus = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                argumentStatus = 1;
                break;
            }
        }

        if (argumentStatus == 0 &&
            (numBodies < 0 || numSteps < 0 || numBodies > std::numeric_limits<int>::max() / 6)) {
            printf("Number of bodies and steps must be non-negative, and the body count is too large for MPI.\n");
            argumentStatus = 1;
        }
    }

    int configuration[5] = {argumentStatus, numBodies, numSteps, static_cast<int>(validate), static_cast<int>(printResults)};
    MPI_Bcast(configuration, 5, MPI_INT, 0, MPI_COMM_WORLD);
    argumentStatus = configuration[0];
    numBodies = configuration[1];
    numSteps = configuration[2];
    validate = configuration[3] != 0;
    printResults = configuration[4] != 0;

    if (argumentStatus != 0) {
        MPI_Finalize();
        return argumentStatus == 2 ? 0 : 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> bodyCounts(rankCount);
    std::vector<int> bodyDisplacements(rankCount);
    const int bodiesPerRank = numBodies / rankCount;
    const int remainder = numBodies % rankCount;
    int displacement = 0;
    for (int process = 0; process < rankCount; ++process) {
        bodyCounts[process] = bodiesPerRank + (process < remainder ? 1 : 0);
        bodyDisplacements[process] = displacement;
        displacement += bodyCounts[process];
    }

    std::vector<int> stateCounts(rankCount);
    std::vector<int> stateDisplacements(rankCount);
    for (int process = 0; process < rankCount; ++process) {
        stateCounts[process] = 6 * bodyCounts[process];
        stateDisplacements[process] = 6 * bodyDisplacements[process];
    }

    const int localBodyCount = bodyCounts[rank];
    std::vector<double> initialState;
    if (rank == 0) {
        std::vector<Body> initialBodies(static_cast<size_t>(numBodies));
        randomizeBodies(initialBodies);
        initialState.resize(static_cast<size_t>(numBodies) * 6);
        for (int i = 0; i < numBodies; ++i) {
            const Body& body = initialBodies[static_cast<size_t>(i)];
            const size_t offset = static_cast<size_t>(i) * 6;
            initialState[offset] = body.pos.x;
            initialState[offset + 1] = body.pos.y;
            initialState[offset + 2] = body.pos.z;
            initialState[offset + 3] = body.vel.x;
            initialState[offset + 4] = body.vel.y;
            initialState[offset + 5] = body.vel.z;
        }
    }

    std::vector<double> localInitialState(static_cast<size_t>(localBodyCount) * 6);
    double dummy = 0.0;
    MPI_Scatterv(rank == 0 && !initialState.empty() ? initialState.data() : &dummy,
                 stateCounts.data(), stateDisplacements.data(), MPI_DOUBLE,
                 localInitialState.empty() ? &dummy : localInitialState.data(),
                 stateCounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<Vec3> localPositions(static_cast<size_t>(localBodyCount));
    std::vector<Vec3> localVelocities(static_cast<size_t>(localBodyCount));
    for (int i = 0; i < localBodyCount; ++i) {
        const size_t offset = static_cast<size_t>(i) * 6;
        localPositions[i] = Vec3(localInitialState[offset], localInitialState[offset + 1], localInitialState[offset + 2]);
        localVelocities[i] = Vec3(localInitialState[offset + 3], localInitialState[offset + 4], localInitialState[offset + 5]);
    }

    // Describe Vec3 directly to MPI so positions need no packing in the
    // timestep hot path.
    MPI_Datatype vec3Type;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);
    std::vector<Vec3> globalPositions(static_cast<size_t>(numBodies));
    const void* localPositionBuffer = localPositions.empty()
                                          ? static_cast<const void*>(&dummy)
                                          : static_cast<const void*>(localPositions.data());
    void* globalPositionBuffer = globalPositions.empty()
                                      ? static_cast<void*>(&dummy)
                                      : static_cast<void*>(globalPositions.data());

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        MPI_Allgatherv(localPositionBuffer, localBodyCount, vec3Type, globalPositionBuffer,
                       bodyCounts.data(), bodyDisplacements.data(), vec3Type, MPI_COMM_WORLD);
        advanceBodies(localPositions, localVelocities, globalPositions);
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", static_cast<long>(duration * 1000.0));
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> localState(static_cast<size_t>(localBodyCount) * 6);
        for (int i = 0; i < localBodyCount; ++i) {
            const size_t offset = static_cast<size_t>(i) * 6;
            localState[offset] = localPositions[i].x;
            localState[offset + 1] = localPositions[i].y;
            localState[offset + 2] = localPositions[i].z;
            localState[offset + 3] = localVelocities[i].x;
            localState[offset + 4] = localVelocities[i].y;
            localState[offset + 5] = localVelocities[i].z;
        }

        std::vector<double> bodyData;
        if (rank == 0) {
            bodyData.resize(static_cast<size_t>(numBodies) * 6);
        }
        MPI_Gatherv(localState.empty() ? &dummy : localState.data(), stateCounts[rank], MPI_DOUBLE,
                    rank == 0 && !bodyData.empty() ? bodyData.data() : &dummy,
                    stateCounts.data(), stateDisplacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(bodyData, "Bodies");
        }
    }

    // Validation: check that simulation produces finite, reasonable values
    int result = 0;
    if (validate) {
        int failingLocalIndex = -1;
        const ValidationFailure localFailure = validateLocalSimulation(localPositions, localVelocities,
                                                                         failingLocalIndex);
        const int localValid = localFailure == ValidationFailure::None ? 1 : 0;
        int globallyValid = 0;
        MPI_Allreduce(&localValid, &globallyValid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validating simulation results...\n");
        }

        if (globallyValid) {
            // The last timestep gathered the pre-integration positions, so
            // obtain one final snapshot for the distributed potential sum.
            MPI_Allgatherv(localPositionBuffer, localBodyCount, vec3Type, globalPositionBuffer,
                           bodyCounts.data(), bodyDisplacements.data(), vec3Type, MPI_COMM_WORLD);
            const double localEnergy = computeLocalEnergy(localPositions, localVelocities,
                                                          globalPositions, bodyDisplacements[rank]);
            double finalEnergy = 0.0;
            MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            const int localFailureIndex = failingLocalIndex < 0
                                              ? std::numeric_limits<int>::max()
                                              : bodyDisplacements[rank] + failingLocalIndex;
            int firstFailureIndex = 0;
            MPI_Allreduce(&localFailureIndex, &firstFailureIndex, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            const int localFirstFailure = localFailureIndex == firstFailureIndex
                                              ? static_cast<int>(localFailure)
                                              : static_cast<int>(ValidationFailure::None);
            int firstFailure = 0;
            MPI_Allreduce(&localFirstFailure, &firstFailure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

            if (rank == 0) {
                switch (static_cast<ValidationFailure>(firstFailure)) {
                    case ValidationFailure::NonFinite:
                        printf("Validation failed: found NaN or Inf value in body state\n");
                        break;
                    case ValidationFailure::PositionOutOfBounds:
                        printf("Validation failed: body position exceeds reasonable bounds\n");
                        break;
                    case ValidationFailure::VelocityOutOfBounds:
                        printf("Validation failed: body velocity exceeds reasonable bounds\n");
                        break;
                    case ValidationFailure::None:
                        break;
                }
                printf("Validation: FAILED\n");
                result = 1;
            }
        }
    }

    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Type_free(&vec3Type);
    MPI_Finalize();
    return result;
}
