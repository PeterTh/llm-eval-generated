#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int ROOT_RANK = 0;

struct Distribution {
    int localOffset = 0;
    int localCount = 0;
    std::vector<int> counts;
    std::vector<int> displacements;
};

Distribution makeDistribution(const int numBodies, const int rank, const int numRanks) {
    Distribution distribution;
    distribution.counts.resize(numRanks);
    distribution.displacements.resize(numRanks);

    // Keep process boundaries aligned with computeForces' SIMD target blocks.
    // This gives every global body the same arithmetic path regardless of the
    // number of ranks while retaining an imbalance of at most one block.
    constexpr int targetBlockSize = 4;
    const int numBlocks = numBodies / targetBlockSize;
    const int baseBlocks = numBlocks / numRanks;
    const int remainderBlocks = numBlocks % numRanks;
    const int tailBodies = numBodies % targetBlockSize;
    int offset = 0;
    for (int process = 0; process < numRanks; ++process) {
        const int bodyCount =
            targetBlockSize * (baseBlocks + (process < remainderBlocks ? 1 : 0)) +
            (process == numRanks - 1 ? tailBodies : 0);
        distribution.counts[process] = 3 * bodyCount;
        distribution.displacements[process] = 3 * offset;
        if (process == rank) {
            distribution.localOffset = offset;
            distribution.localCount = bodyCount;
        }
        offset += bodyCount;
    }

    return distribution;
}

void initializeBodies(std::vector<double>& positions,
                      std::vector<double>& velocities,
                      unsigned int seed = 42) {
    const std::size_t numBodies = positions.size() / 3;
    for (std::size_t body = 0; body < numBodies; ++body) {
        const std::size_t index = 3 * body;
        positions[index] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        positions[index + 1] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        positions[index + 2] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        velocities[index] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        velocities[index + 1] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        velocities[index + 2] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noinline, optimize("no-tree-loop-vectorize")))
#elif defined(__clang__)
__attribute__((noinline))
#endif
void computeCleanupForces(const double* __restrict positions,
                          double* __restrict localVelocities,
                          const int numBodies,
                          const int localOffset,
                          const int firstLocalBody,
                          const int localCount) {
    for (int localBody = firstLocalBody; localBody < localCount; ++localBody) {
        const std::size_t globalIndex =
            3ULL * static_cast<std::size_t>(localOffset + localBody);
        const double targetX = positions[globalIndex];
        const double targetY = positions[globalIndex + 1];
        const double targetZ = positions[globalIndex + 2];
        double forceX = 0.0;
        double forceY = 0.0;
        double forceZ = 0.0;

#if defined(__clang__)
#pragma clang loop vectorize(disable)
#endif
        for (int sourceBody = 0; sourceBody < numBodies; ++sourceBody) {
            const std::size_t sourceIndex = 3ULL * static_cast<std::size_t>(sourceBody);
            const double dx = positions[sourceIndex] - targetX;
            const double dy = positions[sourceIndex + 1] - targetY;
            const double dz = positions[sourceIndex + 2] - targetZ;
            const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double inverseDistance = 1.0 / std::sqrt(distanceSquared);
            const double inverseDistanceCubed =
                inverseDistance * inverseDistance * inverseDistance;
            forceX += dx * inverseDistanceCubed;
            forceY += dy * inverseDistanceCubed;
            forceZ += dz * inverseDistanceCubed;
        }

        const std::size_t localIndex = 3ULL * static_cast<std::size_t>(localBody);
        localVelocities[localIndex] += DT * forceX;
        localVelocities[localIndex + 1] += DT * forceY;
        localVelocities[localIndex + 2] += DT * forceZ;
    }
}

// Compute four independent target bodies together. This retains the original
// j-order for every body while exposing enough independent work for SIMD and
// reusing each source position across several force calculations.
void computeForces(const double* __restrict positions,
                   double* __restrict localVelocities,
                   const int numBodies,
                   const int localOffset,
                   const int localCount) {
    constexpr int targetBlockSize = 4;
    int localBody = 0;

    for (; localBody + targetBlockSize <= localCount; localBody += targetBlockSize) {
        double targetX[targetBlockSize];
        double targetY[targetBlockSize];
        double targetZ[targetBlockSize];
        double forceX[targetBlockSize] = {};
        double forceY[targetBlockSize] = {};
        double forceZ[targetBlockSize] = {};

        for (int lane = 0; lane < targetBlockSize; ++lane) {
            const std::size_t globalIndex =
                3ULL * static_cast<std::size_t>(localOffset + localBody + lane);
            targetX[lane] = positions[globalIndex];
            targetY[lane] = positions[globalIndex + 1];
            targetZ[lane] = positions[globalIndex + 2];
        }

        for (int sourceBody = 0; sourceBody < numBodies; ++sourceBody) {
            const std::size_t sourceIndex = 3ULL * static_cast<std::size_t>(sourceBody);
            const double sourceX = positions[sourceIndex];
            const double sourceY = positions[sourceIndex + 1];
            const double sourceZ = positions[sourceIndex + 2];

            for (int lane = 0; lane < targetBlockSize; ++lane) {
                const double dx = sourceX - targetX[lane];
                const double dy = sourceY - targetY[lane];
                const double dz = sourceZ - targetZ[lane];
                const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inverseDistance = 1.0 / std::sqrt(distanceSquared);
                const double inverseDistanceCubed =
                    inverseDistance * inverseDistance * inverseDistance;
                forceX[lane] += dx * inverseDistanceCubed;
                forceY[lane] += dy * inverseDistanceCubed;
                forceZ[lane] += dz * inverseDistanceCubed;
            }
        }

        for (int lane = 0; lane < targetBlockSize; ++lane) {
            const std::size_t localIndex =
                3ULL * static_cast<std::size_t>(localBody + lane);
            localVelocities[localIndex] += DT * forceX[lane];
            localVelocities[localIndex + 1] += DT * forceY[lane];
            localVelocities[localIndex + 2] += DT * forceZ[lane];
        }
    }

    // There are at most three cleanup targets globally. Preserve the original
    // source-order reduction for them instead of reassociating the sum.
    if (localBody < localCount) {
        computeCleanupForces(positions,
                             localVelocities,
                             numBodies,
                             localOffset,
                             localBody,
                             localCount);
    }
}

void integrateLocalBodies(double* positions,
                          const double* localVelocities,
                          const int localOffset,
                          const int localCount) {
    for (int localBody = 0; localBody < localCount; ++localBody) {
        const std::size_t globalIndex =
            3ULL * static_cast<std::size_t>(localOffset + localBody);
        const std::size_t localIndex = 3ULL * static_cast<std::size_t>(localBody);
        positions[globalIndex] += localVelocities[localIndex] * DT;
        positions[globalIndex + 1] += localVelocities[localIndex + 1] * DT;
        positions[globalIndex + 2] += localVelocities[localIndex + 2] * DT;
    }
}

double computeTotalEnergy(const std::vector<double>& positions,
                          const std::vector<double>& velocities) {
    double energy = 0.0;
    const std::size_t numBodies = positions.size() / 3;

    for (std::size_t body = 0; body < numBodies; ++body) {
        const std::size_t index = 3 * body;
        energy += 0.5 * (velocities[index] * velocities[index] +
                        velocities[index + 1] * velocities[index + 1] +
                        velocities[index + 2] * velocities[index + 2]);
    }

    for (std::size_t firstBody = 0; firstBody < numBodies; ++firstBody) {
        const std::size_t firstIndex = 3 * firstBody;
        for (std::size_t secondBody = firstBody + 1; secondBody < numBodies; ++secondBody) {
            const std::size_t secondIndex = 3 * secondBody;
            const double dx = positions[secondIndex] - positions[firstIndex];
            const double dy = positions[secondIndex + 1] - positions[firstIndex + 1];
            const double dz = positions[secondIndex + 2] - positions[firstIndex + 2];
            const double distance = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / distance;
        }
    }

    return energy;
}

bool validateSimulation(const std::vector<double>& positions,
                        const std::vector<double>& velocities) {
    const double maxPosition = 1e6;
    const double maxVelocity = 1e6;
    const std::size_t numBodies = positions.size() / 3;

    for (std::size_t body = 0; body < numBodies; ++body) {
        const std::size_t index = 3 * body;
        for (int component = 0; component < 3; ++component) {
            const double position = positions[index + component];
            const double velocity = velocities[index + component];
            if (!std::isfinite(position) || !std::isfinite(velocity)) {
                std::printf("Validation failed: found NaN or Inf value in body state\n");
                return false;
            }
            if (std::abs(position) > maxPosition) {
                std::printf("Validation failed: body position exceeds reasonable bounds\n");
                return false;
            }
            if (std::abs(velocity) > maxVelocity) {
                std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks energy conservation)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::fprintf(stderr, "Failed to initialize MPI\n");
        return 1;
    }

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int options[5] = {1024, 10, 0, 0, 0};
    // options: number of bodies, steps, validation, result printing, exit code.
    if (rank == ROOT_RANK) {
        for (int argument = 1; argument < argc; ++argument) {
            if (std::strcmp(argv[argument], "-n") == 0 && argument + 1 < argc) {
                options[0] = std::atoi(argv[++argument]);
            } else if (std::strcmp(argv[argument], "-s") == 0 && argument + 1 < argc) {
                options[1] = std::atoi(argv[++argument]);
            } else if (std::strcmp(argv[argument], "-v") == 0) {
                options[2] = 1;
            } else if (std::strcmp(argv[argument], "-r") == 0) {
                options[3] = 1;
            } else if (std::strcmp(argv[argument], "-h") == 0) {
                printUsage(argv[0]);
                options[4] = -1;
                break;
            } else {
                std::printf("Unknown option: %s\n", argv[argument]);
                printUsage(argv[0]);
                options[4] = 1;
                break;
            }
        }

        if (options[4] == 0 &&
            (options[0] < 0 || options[1] < 0 ||
             options[0] > std::numeric_limits<int>::max() / 3)) {
            std::fprintf(stderr,
                         "Number of bodies and steps must be non-negative, and the body count "
                         "must fit MPI collective counts\n");
            options[4] = 1;
        }
    }

    MPI_Bcast(options, 5, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
    if (options[4] != 0) {
        const int exitCode = options[4] > 0 ? options[4] : 0;
        MPI_Finalize();
        return exitCode;
    }

    const int numBodies = options[0];
    const int numSteps = options[1];
    const bool validate = options[2] != 0;
    const bool printResults = options[3] != 0;
    const Distribution distribution = makeDistribution(numBodies, rank, numRanks);

    if (rank == ROOT_RANK) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> positions(3ULL * static_cast<std::size_t>(numBodies));
    std::vector<double> localVelocities(
        3ULL * static_cast<std::size_t>(distribution.localCount));
    std::vector<double> initialVelocities;
    if (rank == ROOT_RANK) {
        initialVelocities.resize(3ULL * static_cast<std::size_t>(numBodies));
        initializeBodies(positions, initialVelocities);
    }

    MPI_Bcast(positions.data(), 3 * numBodies, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);
    MPI_Scatterv(initialVelocities.data(),
                 distribution.counts.data(),
                 distribution.displacements.data(),
                 MPI_DOUBLE,
                 localVelocities.data(),
                 3 * distribution.localCount,
                 MPI_DOUBLE,
                 ROOT_RANK,
                 MPI_COMM_WORLD);
    initialVelocities.clear();
    initialVelocities.shrink_to_fit();

    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(positions.data(),
                      localVelocities.data(),
                      numBodies,
                      distribution.localOffset,
                      distribution.localCount);
        integrateLocalBodies(positions.data(),
                             localVelocities.data(),
                             distribution.localOffset,
                             distribution.localCount);

        // Every rank owns an already-updated, contiguous segment of positions.
        // MPI_IN_PLACE avoids a packing/copy pass before the collective.
        MPI_Allgatherv(MPI_IN_PLACE,
                       0,
                       MPI_DOUBLE,
                       positions.data(),
                       distribution.counts.data(),
                       distribution.displacements.data(),
                       MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }

    const double localElapsed = MPI_Wtime() - startTime;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, ROOT_RANK, MPI_COMM_WORLD);

    if (rank == ROOT_RANK) {
        const long long durationMilliseconds = static_cast<long long>(elapsed * 1000.0);
        std::printf("Simulation time: %lld ms\n", durationMilliseconds);
    }

    std::vector<double> velocities;
    if (printResults || validate) {
        if (rank == ROOT_RANK) {
            velocities.resize(3ULL * static_cast<std::size_t>(numBodies));
        }
        MPI_Gatherv(localVelocities.data(),
                    3 * distribution.localCount,
                    MPI_DOUBLE,
                    velocities.data(),
                    distribution.counts.data(),
                    distribution.displacements.data(),
                    MPI_DOUBLE,
                    ROOT_RANK,
                    MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == ROOT_RANK && printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(6ULL * static_cast<std::size_t>(numBodies));
        for (int body = 0; body < numBodies; ++body) {
            const std::size_t index = 3ULL * static_cast<std::size_t>(body);
            bodyData.push_back(positions[index]);
            bodyData.push_back(positions[index + 1]);
            bodyData.push_back(positions[index + 2]);
            bodyData.push_back(velocities[index]);
            bodyData.push_back(velocities[index + 1]);
            bodyData.push_back(velocities[index + 2]);
        }
        print_results(bodyData, "Bodies");
    }

    if (rank == ROOT_RANK && validate) {
        std::printf("Validating simulation results...\n");
        if (validateSimulation(positions, velocities)) {
            const double finalEnergy = computeTotalEnergy(positions, velocities);
            std::printf("Final energy: %.6f\n", finalEnergy);
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
