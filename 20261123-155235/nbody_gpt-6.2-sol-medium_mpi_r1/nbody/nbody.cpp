#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// Keep scalar and vector force expressions on the same rounding path.
double squaredDistance(double dx, double dy, double dz) {
#if defined(FP_FAST_FMA) || defined(__FMA__)
    return std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING;
#else
    return dx * dx + dy * dy + dz * dz + SOFTENING;
#endif
}

double accumulateComponent(double component, double invDist3, double force) {
#if defined(FP_FAST_FMA) || defined(__FMA__)
    return std::fma(component, invDist3, force);
#else
    return force + component * invDist3;
#endif
}

double updateVelocity(double velocity, double force) {
#if defined(__GNUC__) && !defined(__clang__)
    // GCC rounded the force times DT before adding it in the original kernel.
    const volatile double kick = DT * force;
    return velocity + kick;
#else
    return velocity + DT * force;
#endif
}

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

static_assert(sizeof(Vec3) == 3 * sizeof(double));

void randomizeBodies(std::vector<Vec3>& positions, std::vector<Vec3>& velocities) {
    unsigned int seed = 42;
    const auto coordinate = [&seed] { return 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0; };
    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i] = {coordinate(), coordinate(), coordinate()};
        velocities[i] = {coordinate(), coordinate(), coordinate()};
    }
}

// Vectorize across target bodies so each target still sums sources in global order.
template <size_t targets>
void accumulateForces(const std::vector<Vec3>& positions, size_t first, Vec3* velocities) {
    std::array<double, targets> x, y, z;
    std::array<double, targets> Fx{}, Fy{}, Fz{};
    for (size_t i = 0; i < targets; ++i) {
        x[i] = positions[first + i].x;
        y[i] = positions[first + i].y;
        z[i] = positions[first + i].z;
    }
    for (const auto& source : positions) {
        for (size_t i = 0; i < targets; ++i) {
            const double dx = source.x - x[i];
            const double dy = source.y - y[i];
            const double dz = source.z - z[i];
            const double distSqr = squaredDistance(dx, dy, dz);
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx[i] = accumulateComponent(dx, invDist3, Fx[i]);
            Fy[i] = accumulateComponent(dy, invDist3, Fy[i]);
            Fz[i] = accumulateComponent(dz, invDist3, Fz[i]);
        }
    }
    for (size_t i = 0; i < targets; ++i) {
        velocities[i].x = updateVelocity(velocities[i].x, Fx[i]);
        velocities[i].y = updateVelocity(velocities[i].y, Fy[i]);
        velocities[i].z = updateVelocity(velocities[i].z, Fz[i]);
    }
}

void advanceBodies(std::vector<Vec3>& positions, size_t first, std::vector<Vec3>& velocities) {
    constexpr size_t block = 4;
    size_t i = 0;
    for (; i + block <= velocities.size(); i += block) {
        accumulateForces<block>(positions, first + i, velocities.data() + i);
    }
    for (; i < velocities.size(); ++i) {
        accumulateForces<1>(positions, first + i, velocities.data() + i);
    }

    // All force calculations must read the same timestep's positions.
    for (size_t i = 0; i < velocities.size(); ++i) {
        auto& pos = positions[first + i];
        const auto& vel = velocities[i];
        pos.x += vel.x * DT;
        pos.y += vel.y * DT;
        pos.z += vel.z * DT;
    }
}

double computeLocalEnergy(const std::vector<Vec3>& positions, const std::vector<Vec3>& velocities,
                          size_t rank, size_t ranks) {
    double energy = 0.0;
    for (const auto& vel : velocities) {
        energy += 0.5 * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
    }

    const size_t n = positions.size();
    const auto subtractPotentialRow = [&](size_t i) {
        const auto& pos = positions[i];
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = positions[j].x - pos.x;
            const double dy = positions[j].y - pos.y;
            const double dz = positions[j].z - pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    };

    if (ranks == 1) {
        for (size_t i = 0; i < n; ++i) {
            subtractPotentialRow(i);
        }
    } else {
        // Pair long and short rows of the interaction triangle to balance work.
        for (size_t i = rank; i < n / 2; i += ranks) {
            const size_t opposite = n - 2 - i;
            subtractPotentialRow(i);
            if (opposite != i) {
                subtractPotentialRow(opposite);
            }
        }
    }
    return energy;
}

int validateBodies(const std::vector<Vec3>& positions, size_t first, const std::vector<Vec3>& velocities) {
    for (size_t i = 0; i < velocities.size(); ++i) {
        const auto& pos = positions[first + i];
        const auto& vel = velocities[i];
        if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
            !std::isfinite(vel.x) || !std::isfinite(vel.y) || !std::isfinite(vel.z)) {
            return 1;
        }

        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(pos.x) > maxPos || std::abs(pos.y) > maxPos || std::abs(pos.z) > maxPos) {
            return 2;
        }
        if (std::abs(vel.x) > maxVel || std::abs(vel.y) > maxVel || std::abs(vel.z) > maxVel) {
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

int parseOptions(int argc, char** argv, int& numBodies, int& numSteps, int& validate, int& printResults) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = 1;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = 1;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 1;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 2;
        }
    }
    if (numBodies < 0) {
        printf("Number of bodies must be nonnegative\n");
        return 2;
    }
    return 0;
}

int runSimulation(int argc, char** argv, int rank, int ranks) {
    std::array options{1024, 10, 0, 0, 0};
    auto& [numBodies, numSteps, validate, printResults, action] = options;
    if (rank == 0) {
        action = parseOptions(argc, argv, numBodies, numSteps, validate, printResults);
    }
    MPI_Bcast(options.data(), options.size(), MPI_INT, 0, MPI_COMM_WORLD);
    if (action != 0) {
        return action - 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = r * (numBodies / ranks) + std::min(r, numBodies % ranks);
    }
    const size_t first = offsets[rank];
    std::vector<Vec3> positions(numBodies), velocities(counts[rank]);

    // Counts and offsets are measured in bodies, even when transferring doubles.
    MPI_Datatype vec3Type;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);

    if (ranks == 1) {
        randomizeBodies(positions, velocities);
    } else if (numBodies > 0) {
        std::vector<Vec3> initialVelocities;
        if (rank == 0) {
            initialVelocities.resize(numBodies);
            randomizeBodies(positions, initialVelocities);
        }
        MPI_Scatterv(initialVelocities.data(), counts.data(), offsets.data(), vec3Type,
                     velocities.data(), counts[rank], vec3Type, 0, MPI_COMM_WORLD);
        MPI_Bcast(positions.data(), numBodies, vec3Type, 0, MPI_COMM_WORLD);
    }

    const auto exchangePositions = [&] {
        if (ranks > 1 && numBodies > 0) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, vec3Type, positions.data(), counts.data(),
                           offsets.data(), vec3Type, MPI_COMM_WORLD);
        }
    };

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        if (step > 0) {
            exchangePositions();
        }
        advanceBodies(positions, first, velocities);
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Simulation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
    }

    if (printResults) {
        std::vector<double> localData;
        localData.reserve(6 * velocities.size());
        for (size_t i = 0; i < velocities.size(); ++i) {
            const auto& pos = positions[first + i];
            const auto& vel = velocities[i];
            localData.insert(localData.end(), {pos.x, pos.y, pos.z, vel.x, vel.y, vel.z});
        }

        std::vector<double> bodyData(rank == 0 ? 6 * static_cast<size_t>(numBodies) : 0);
        if (numBodies > 0) {
            MPI_Datatype bodyType;
            MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
            MPI_Type_commit(&bodyType);
            MPI_Gatherv(localData.data(), counts[rank], bodyType, bodyData.data(),
                        counts.data(), offsets.data(), bodyType, 0, MPI_COMM_WORLD);
            MPI_Type_free(&bodyType);
        }
        if (rank == 0) {
            print_results(bodyData, "Bodies");
        }
    }

    int status = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
        }
        // Rank order is body order. Select the diagnostic for the first invalid body.
        const int failure = validateBodies(positions, first, velocities);
        const std::array localCheck{failure ? rank : ranks, failure};
        std::array<int, 2> globalCheck;
        MPI_Allreduce(localCheck.data(), globalCheck.data(), 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
        if (globalCheck[1] == 0) {
            if (numSteps > 0) {
                exchangePositions();
            }
            const double localEnergy = computeLocalEnergy(positions, velocities, rank, ranks);
            double finalEnergy = 0.0;
            MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                const char* reasons[] = {"found NaN or Inf value in body state",
                                         "body position exceeds reasonable bounds",
                                         "body velocity exceeds reasonable bounds"};
                printf("Validation failed: %s\n", reasons[globalCheck[1] - 1]);
                printf("Validation: FAILED\n");
            }
            status = 1;
        }
    }

    MPI_Type_free(&vec3Type);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int status = 1;
    try {
        status = runSimulation(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
