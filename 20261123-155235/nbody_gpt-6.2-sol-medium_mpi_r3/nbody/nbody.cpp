#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr size_t TARGET_WIDTH = 4;

struct Vec3 {
    double x, y, z;
};

static_assert(sizeof(Vec3) == 3 * sizeof(double));

void randomizeBodies(std::span<Vec3> positions, std::span<Vec3> velocities) {
    unsigned int seed = 42;
    const auto coordinate = [&seed] {
        return 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    };
    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i] = {coordinate(), coordinate(), coordinate()};
        velocities[i] = {coordinate(), coordinate(), coordinate()};
    }
}

void computeKicks(std::span<const Vec3> positions, const Vec3* targets, Vec3* kicks) {
    double x[TARGET_WIDTH], y[TARGET_WIDTH], z[TARGET_WIDTH];
    double fx[TARGET_WIDTH]{}, fy[TARGET_WIDTH]{}, fz[TARGET_WIDTH]{};
    for (size_t lane = 0; lane < TARGET_WIDTH; ++lane) {
        x[lane] = targets[lane].x;
        y[lane] = targets[lane].y;
        z[lane] = targets[lane].z;
    }

    for (const Vec3& source : positions) {
        // Keep the target loop intact for the compiler's loop vectorizer.
#pragma GCC unroll 1
        for (size_t lane = 0; lane < TARGET_WIDTH; ++lane) {
            const double dx = source.x - x[lane];
            const double dy = source.y - y[lane];
            const double dz = source.z - z[lane];
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            fx[lane] += dx * invDist3;
            fy[lane] += dy * invDist3;
            fz[lane] += dz * invDist3;
        }
    }

    for (size_t lane = 0; lane < TARGET_WIDTH; ++lane) {
        kicks[lane] = {DT * fx[lane], DT * fy[lane], DT * fz[lane]};
    }
}

void advanceBodies(std::span<const Vec3> positions, std::span<Vec3> localPositions,
                   std::span<Vec3> velocities, std::span<Vec3> kicks) {
    // SIMD over targets keeps each force sum in the original source order.
    size_t i = 0;
    for (; i + TARGET_WIDTH <= kicks.size(); i += TARGET_WIDTH) {
        computeKicks(positions, localPositions.data() + i, kicks.data() + i);
    }
    if (i < kicks.size()) {
        // Use the same arithmetic for the tail on every partition.
        Vec3 targets[TARGET_WIDTH]{}, tailKicks[TARGET_WIDTH];
        const size_t tail = kicks.size() - i;
        std::copy_n(localPositions.data() + i, tail, targets);
        computeKicks(positions, targets, tailKicks);
        std::copy_n(tailKicks, tail, kicks.data() + i);
    }

    for (size_t i = 0; i < localPositions.size(); ++i) {
        Vec3& velocity = velocities[i];
        velocity.x += kicks[i].x;
        velocity.y += kicks[i].y;
        velocity.z += kicks[i].z;
        localPositions[i].x += velocity.x * DT;
        localPositions[i].y += velocity.y * DT;
        localPositions[i].z += velocity.z * DT;
    }
}

double computeLocalEnergy(std::span<const Vec3> positions, std::span<const Vec3> velocities,
                          size_t rank, size_t ranks) {
    double energy = 0.0;
    for (const Vec3& vel : velocities) {
        energy += 0.5 * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
    }

    // Round-robin assignment balances the triangular set of potential-energy pairs.
    for (size_t i = rank; i < positions.size(); i += ranks) {
        for (size_t j = i + 1; j < positions.size(); ++j) {
            const double dx = positions[j].x - positions[i].x;
            const double dy = positions[j].y - positions[i].y;
            const double dz = positions[j].z - positions[i].z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    return energy;
}

enum ValidationError { valid, nonFinite, positionOutOfBounds, velocityOutOfBounds };

ValidationError validateSimulation(std::span<const Vec3> positions, std::span<const Vec3> velocities) {
    constexpr double maxPos = 1e6;
    constexpr double maxVel = 1e6;
    for (size_t i = 0; i < positions.size(); ++i) {
        const Vec3& pos = positions[i];
        const Vec3& vel = velocities[i];
        if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
            !std::isfinite(vel.x) || !std::isfinite(vel.y) || !std::isfinite(vel.z)) {
            return nonFinite;
        }
        if (std::abs(pos.x) > maxPos || std::abs(pos.y) > maxPos || std::abs(pos.z) > maxPos) {
            return positionOutOfBounds;
        }
        if (std::abs(vel.x) > maxVel || std::abs(vel.y) > maxVel || std::abs(vel.z) > maxVel) {
            return velocityOutOfBounds;
        }
    }
    return valid;
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

int simulate(int numBodies, int numSteps, bool validate, bool printResults,
             int rank, int ranks, MPI_Comm comm) {
    std::vector<int> counts(ranks), offsets(ranks);
    const int base = numBodies / ranks;
    const int remainder = numBodies % ranks;
    for (int r = 0; r < ranks; ++r) {
        counts[r] = base + (r < remainder);
        offsets[r] = r * base + std::min(r, remainder);
    }

    MPI_Datatype vectorType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vectorType);
    MPI_Type_commit(&vectorType);

    // Each rank owns its velocities and keeps a snapshot of the global positions.
    std::vector<Vec3> positions(numBodies);
    auto localPositions = std::span(positions).subspan(offsets[rank], counts[rank]);
    std::vector<Vec3> velocities(counts[rank]), kicks(counts[rank]);
    {
        std::vector<Vec3> initialVelocities(rank == 0 ? numBodies : 0);
        if (rank == 0) {
            randomizeBodies(positions, initialVelocities);
        }
        MPI_Scatterv(initialVelocities.data(), counts.data(), offsets.data(), vectorType,
                     velocities.data(), counts[rank], vectorType, 0, comm);
        MPI_Bcast(positions.data(), numBodies, vectorType, 0, comm);
    }

    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        advanceBodies(positions, localPositions, velocities, kicks);

        if (step + 1 < numSteps || validate) {
            // Integration wrote directly into this rank's slot in the receive buffer.
            if (remainder == 0) {
                MPI_Allgather(MPI_IN_PLACE, 0, vectorType, positions.data(), base, vectorType, comm);
            } else {
                MPI_Allgatherv(MPI_IN_PLACE, 0, vectorType, positions.data(), counts.data(),
                               offsets.data(), vectorType, comm);
            }
        }
    }

    const double elapsed = MPI_Wtime() - start;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Simulation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
    }

    if (printResults) {
        std::vector<double> localData;
        localData.reserve(6 * localPositions.size());
        for (size_t i = 0; i < localPositions.size(); ++i) {
            const Vec3& pos = localPositions[i];
            const Vec3& vel = velocities[i];
            localData.insert(localData.end(), {pos.x, pos.y, pos.z, vel.x, vel.y, vel.z});
        }
        std::vector<double> bodyData(rank == 0 ? 6 * static_cast<size_t>(numBodies) : 0);
        MPI_Datatype bodyType;
        MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
        MPI_Type_commit(&bodyType);
        MPI_Gatherv(localData.data(), counts[rank], bodyType, bodyData.data(),
                    counts.data(), offsets.data(), bodyType, 0, comm);
        MPI_Type_free(&bodyType);
        if (rank == 0) {
            print_results(bodyData, "Bodies");
        }
    }
    MPI_Type_free(&vectorType);

    if (!validate) {
        return 0;
    }
    if (rank == 0) {
        printf("Validating simulation results...\n");
    }

    // Contiguous ownership lets the lowest failing rank report the first invalid body.
    const auto error = validateSimulation(localPositions, velocities);
    const std::array<int, 2> localStatus{error == valid ? ranks : rank, error};
    std::array<int, 2> status;
    MPI_Allreduce(localStatus.data(), status.data(), 1, MPI_2INT, MPI_MINLOC, comm);
    if (status[1] != valid) {
        if (rank == 0) {
            const char* messages[] = {
                "",
                "found NaN or Inf value in body state",
                "body position exceeds reasonable bounds",
                "body velocity exceeds reasonable bounds"
            };
            printf("Validation failed: %s\n", messages[status[1]]);
            printf("Validation: FAILED\n");
        }
        return 1;
    }

    const double localEnergy = computeLocalEnergy(positions, velocities, rank, ranks);
    double finalEnergy;
    MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    if (rank == 0) {
        printf("Final energy: %.6f\n", finalEnergy);
        printf("Validation: PASSED\n");
    }
    return 0;
}

int run(int argc, char** argv) {
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (numBodies < 0) {
        if (rank == 0) {
            fprintf(stderr, "Number of bodies must be nonnegative\n");
        }
        return 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int ranks = std::min(worldSize, std::max(1, numBodies));
    MPI_Comm comm = MPI_COMM_WORLD;
    if (ranks < worldSize) {
        MPI_Comm_split(MPI_COMM_WORLD, rank < ranks ? 0 : MPI_UNDEFINED, rank, &comm);
    }
    if (comm == MPI_COMM_NULL) {
        return 0;
    }

    const int result = simulate(numBodies, numSteps, validate, printResults, rank, ranks, comm);
    if (comm != MPI_COMM_WORLD) {
        MPI_Comm_free(&comm);
    }
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int result = run(argc, argv);
    MPI_Finalize();
    return result;
}
