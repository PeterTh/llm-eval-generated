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

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

static_assert(sizeof(Vec3) == 3 * sizeof(double));

// Fix the contraction points so scalar and vector targets round identically.
inline double multiplyAdd(double a, double b, double c) {
#if defined(FP_FAST_FMA) || defined(__FMA__) || defined(__aarch64__)
    return std::fma(a, b, c);
#else
    return a * b + c;
#endif
}

void randomizeBodies(std::vector<Vec3>& positions, std::vector<Vec3>& velocities) {
    unsigned int seed = 42;
    const auto randomComponent = [&seed]() {
        return 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    };
    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i] = {randomComponent(), randomComponent(), randomComponent()};
        velocities[i] = {randomComponent(), randomComponent(), randomComponent()};
    }
}

// Vectorize across target bodies while visiting the sources in their original
// order for each target. All targets read the same snapshot of positions.
template <size_t Width>
void computeForces(const std::vector<Vec3>& positions, size_t first, Vec3* velocities) {
    double x[Width], y[Width], z[Width];
    double fx[Width]{}, fy[Width]{}, fz[Width]{};

    for (size_t i = 0; i < Width; ++i) {
        x[i] = positions[first + i].x;
        y[i] = positions[first + i].y;
        z[i] = positions[first + i].z;
    }
    for (const auto& source : positions) {
        for (size_t i = 0; i < Width; ++i) {
            const double dx = source.x - x[i];
            const double dy = source.y - y[i];
            const double dz = source.z - z[i];
            const double xy = multiplyAdd(dx, dx, dy * dy);
            const double distSqr = multiplyAdd(dz, dz, xy) + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            fx[i] = multiplyAdd(dx, invDist3, fx[i]);
            fy[i] = multiplyAdd(dy, invDist3, fy[i]);
            fz[i] = multiplyAdd(dz, invDist3, fz[i]);
        }
    }
    for (size_t i = 0; i < Width; ++i) {
        velocities[i].x += DT * fx[i];
        velocities[i].y += DT * fy[i];
        velocities[i].z += DT * fz[i];
    }
}

void advanceBodies(std::vector<Vec3>& positions, size_t first, std::vector<Vec3>& velocities) {
    constexpr size_t width = 4;
    size_t i = 0;
    for (; i + width <= velocities.size(); i += width) {
        computeForces<width>(positions, first + i, velocities.data() + i);
    }
    for (; i < velocities.size(); ++i) {
        computeForces<1>(positions, first + i, velocities.data() + i);
    }
    for (i = 0; i < velocities.size(); ++i) {
        auto& pos = positions[first + i];
        const auto& vel = velocities[i];
        pos.x = multiplyAdd(vel.x, DT, pos.x);
        pos.y = multiplyAdd(vel.y, DT, pos.y);
        pos.z = multiplyAdd(vel.z, DT, pos.z);
    }
}

double computeLocalEnergy(const std::vector<Vec3>& positions, const std::vector<Vec3>& velocities,
                          int rank, int ranks) {
    double energy = 0.0;
    for (const auto& vel : velocities) {
        energy += 0.5 * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
    }

    const size_t n = positions.size();
    const auto potentialRow = [&](size_t i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = positions[j].x - positions[i].x;
            const double dy = positions[j].y - positions[i].y;
            const double dz = positions[j].z - positions[i].z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    };
    // Pair the longest rows of the interaction triangle with the shortest.
    for (size_t i = rank; i < (n + 1) / 2; i += ranks) {
        potentialRow(i);
        const size_t opposite = n - 1 - i;
        if (opposite != i) {
            potentialRow(opposite);
        }
    }
    return energy;
}

enum ValidationError { Valid, NonFinite, PositionOutOfBounds, VelocityOutOfBounds };

ValidationError validateBody(const Vec3& pos, const Vec3& vel) {
    if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
        !std::isfinite(vel.x) || !std::isfinite(vel.y) || !std::isfinite(vel.z)) {
        return NonFinite;
    }
    constexpr double bound = 1e6;
    if (std::abs(pos.x) > bound || std::abs(pos.y) > bound || std::abs(pos.z) > bound) {
        return PositionOutOfBounds;
    }
    if (std::abs(vel.x) > bound || std::abs(vel.y) > bound || std::abs(vel.z) > bound) {
        return VelocityOutOfBounds;
    }
    return Valid;
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

int runSimulation(int argc, char** argv, int rank, int ranks) {
    const bool root = rank == 0;
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
            if (root) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (numBodies < 0) {
        if (root) {
            printf("Number of bodies must be nonnegative\n");
        }
        return 1;
    }

    if (root) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int perRank = numBodies / ranks;
    const int remainder = numBodies % ranks;
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = perRank + (r < remainder);
        offsets[r] = r * perRank + std::min(r, remainder);
    }
    const int localCount = counts[rank];
    const int first = offsets[rank];

    MPI_Datatype mpiVec3;
    MPI_Type_contiguous(3, MPI_DOUBLE, &mpiVec3);
    MPI_Type_commit(&mpiVec3);

    std::vector<Vec3> positions(numBodies);
    std::vector<Vec3> velocities(localCount);
    {
        std::vector<Vec3> initialVelocities;
        if (root) {
            initialVelocities.resize(numBodies);
            randomizeBodies(positions, initialVelocities);
        }
        MPI_Scatterv(initialVelocities.data(), counts.data(), offsets.data(), mpiVec3,
                     velocities.data(), localCount, mpiVec3, 0, MPI_COMM_WORLD);
        if (numSteps > 0 || validate) {
            MPI_Bcast(positions.data(), numBodies, mpiVec3, 0, MPI_COMM_WORLD);
        }
    }

    const auto exchangePositions = [&] {
        if (remainder == 0) {
            MPI_Allgather(MPI_IN_PLACE, 0, mpiVec3, positions.data(),
                          perRank, mpiVec3, MPI_COMM_WORLD);
        } else {
            MPI_Allgatherv(MPI_IN_PLACE, 0, mpiVec3, positions.data(),
                           counts.data(), offsets.data(), mpiVec3, MPI_COMM_WORLD);
        }
    };
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        advanceBodies(positions, first, velocities);
        if (step + 1 < numSteps) {
            exchangePositions();
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (root) {
        printf("Simulation time: %lld ms\n", static_cast<long long>(maxElapsed * 1000));
    }

    // Only energy calculation needs the final positions on every rank.
    if (validate && numSteps > 0) {
        exchangePositions();
    }
    if (printResults) {
        if (numSteps > 0 && !validate) {
            const Vec3* localPositions = localCount ? positions.data() + first : nullptr;
            MPI_Gatherv(root ? MPI_IN_PLACE : localPositions, localCount, mpiVec3,
                        positions.data(), counts.data(), offsets.data(), mpiVec3, 0, MPI_COMM_WORLD);
        }
        std::vector<Vec3> allVelocities(root ? numBodies : 0);
        MPI_Gatherv(velocities.data(), localCount, mpiVec3, allVelocities.data(),
                    counts.data(), offsets.data(), mpiVec3, 0, MPI_COMM_WORLD);
        if (root) {
            std::vector<double> bodyData;
            bodyData.reserve(positions.size() * 6);
            for (size_t i = 0; i < positions.size(); ++i) {
                const auto& pos = positions[i];
                const auto& vel = allVelocities[i];
                bodyData.insert(bodyData.end(), {pos.x, pos.y, pos.z, vel.x, vel.y, vel.z});
            }
            print_results(bodyData, "Bodies");
        }
    }
    MPI_Type_free(&mpiVec3);

    if (validate) {
        if (root) {
            printf("Validating simulation results...\n");
        }

        // Resolve failures in global body order, as in the serial scan.
        int localFailure[2] = {std::numeric_limits<int>::max(), Valid};
        for (int i = 0; i < localCount; ++i) {
            const auto error = validateBody(positions[first + i], velocities[i]);
            if (error != Valid) {
                localFailure[0] = first + i;
                localFailure[1] = error;
                break;
            }
        }
        int failure[2];
        MPI_Allreduce(localFailure, failure, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
        if (failure[1] != Valid) {
            if (root) {
                const char* messages[] = {"", "found NaN or Inf value in body state",
                                          "body position exceeds reasonable bounds",
                                          "body velocity exceeds reasonable bounds"};
                printf("Validation failed: %s\n", messages[failure[1]]);
                printf("Validation: FAILED\n");
            }
            return 1;
        }

        const double localEnergy = computeLocalEnergy(positions, velocities, rank, ranks);
        double energy = 0.0;
        MPI_Reduce(&localEnergy, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (root) {
            printf("Final energy: %.6f\n", energy);
            printf("Validation: PASSED\n");
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int result = runSimulation(argc, argv, rank, ranks);
    MPI_Finalize();
    return result;
}
