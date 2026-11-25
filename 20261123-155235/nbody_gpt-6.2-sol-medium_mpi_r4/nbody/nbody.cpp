#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

namespace {

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
};
static_assert(sizeof(Vec3) == 3 * sizeof(double));

struct Partition {
    std::vector<int> counts;
    std::vector<int> offsets;

    Partition(int numBodies, int ranks) : counts(ranks), offsets(ranks) {
        int offset = 0;
        for (int rank = 0; rank < ranks; ++rank) {
            counts[rank] = numBodies / ranks + (rank < numBodies % ranks);
            offsets[rank] = offset;
            offset += counts[rank];
        }
    }
};

void randomizeBodies(std::vector<Vec3>& positions, std::vector<Vec3>& velocities) {
    unsigned int seed = 42;
    const auto coordinate = [&seed]() {
        return 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    };
    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i] = {coordinate(), coordinate(), coordinate()};
        velocities[i] = {coordinate(), coordinate(), coordinate()};
    }
}

inline double multiplyAdd(double a, double b, double c) {
#if defined(FP_FAST_FMA) || defined(__FMA__) || defined(__aarch64__)
    return std::fma(a, b, c);
#else
    return a * b + c;
#endif
}

inline double squaredDistance(double dx, double dy, double dz) {
    return multiplyAdd(dz, dz, multiplyAdd(dx, dx, dy * dy)) + SOFTENING;
}

inline double updateVelocity(double velocity, double force) {
    // Match the contraction of the original optimized velocity update.
#ifdef __clang__
    return multiplyAdd(DT, force, velocity);
#else
    return velocity + DT * force;
#endif
}

// Vectorize across independent targets while visiting sources in index order.
template <size_t Width>
void updateVelocities(const std::vector<Vec3>& positions, size_t first, Vec3* velocities) {
    std::array<double, Width> x, y, z;
    std::array<double, Width> fx{}, fy{}, fz{};
    for (size_t lane = 0; lane < Width; ++lane) {
        const Vec3& target = positions[first + lane];
        x[lane] = target.x;
        y[lane] = target.y;
        z[lane] = target.z;
    }
    for (const Vec3& source : positions) {
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
        for (size_t lane = 0; lane < Width; ++lane) {
            const double dx = source.x - x[lane];
            const double dy = source.y - y[lane];
            const double dz = source.z - z[lane];
            const double invDist = 1.0 / std::sqrt(squaredDistance(dx, dy, dz));
            const double invDist3 = invDist * invDist * invDist;
            fx[lane] = multiplyAdd(dx, invDist3, fx[lane]);
            fy[lane] = multiplyAdd(dy, invDist3, fy[lane]);
            fz[lane] = multiplyAdd(dz, invDist3, fz[lane]);
        }
    }
    for (size_t lane = 0; lane < Width; ++lane) {
        velocities[lane].x = updateVelocity(velocities[lane].x, fx[lane]);
        velocities[lane].y = updateVelocity(velocities[lane].y, fy[lane]);
        velocities[lane].z = updateVelocity(velocities[lane].z, fz[lane]);
    }
}

void advanceBodies(std::vector<Vec3>& positions, std::vector<Vec3>& velocities, size_t first) {
    constexpr size_t width = 4;
    size_t i = 0;
    for (; i + width <= velocities.size(); i += width) {
        updateVelocities<width>(positions, first + i, velocities.data() + i);
    }
    if (velocities.size() - i >= 2) {
        updateVelocities<2>(positions, first + i, velocities.data() + i);
        i += 2;
    }
    if (i < velocities.size()) {
        updateVelocities<1>(positions, first + i, velocities.data() + i);
    }
    for (i = 0; i < velocities.size(); ++i) {
        Vec3& pos = positions[first + i];
        const Vec3& vel = velocities[i];
        pos.x = multiplyAdd(vel.x, DT, pos.x);
        pos.y = multiplyAdd(vel.y, DT, pos.y);
        pos.z = multiplyAdd(vel.z, DT, pos.z);
    }
}

void exchangePositions(std::vector<Vec3>& positions, const Partition& partition, MPI_Datatype vecType) {
    if (partition.counts.size() == 1 || positions.empty()) {
        return;
    }
    // Each rank has already updated its portion of the receive buffer.
    if (partition.counts.front() == partition.counts.back()) {
        MPI_Allgather(MPI_IN_PLACE, 0, vecType, positions.data(), partition.counts.front(),
                      vecType, MPI_COMM_WORLD);
    } else {
        MPI_Allgatherv(MPI_IN_PLACE, 0, vecType, positions.data(), partition.counts.data(),
                       partition.offsets.data(), vecType, MPI_COMM_WORLD);
    }
}

void reportBodies(const std::vector<Vec3>& positions, const std::vector<Vec3>& velocities,
                  const Partition& partition, int rank) {
    constexpr size_t bodyWidth = 6;
    const size_t first = partition.offsets[rank];
    std::vector<double> data(bodyWidth * (rank == 0 ? positions.size() : velocities.size()));
    for (size_t i = 0; i < velocities.size(); ++i) {
        const Vec3& pos = positions[first + i];
        const Vec3& vel = velocities[i];
        double* body = data.data() + bodyWidth * i;
        body[0] = pos.x;
        body[1] = pos.y;
        body[2] = pos.z;
        body[3] = vel.x;
        body[4] = vel.y;
        body[5] = vel.z;
    }

    MPI_Datatype bodyType;
    MPI_Type_contiguous(bodyWidth, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);
    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : data.data(), partition.counts[rank], bodyType,
                rank == 0 ? data.data() : nullptr, partition.counts.data(),
                partition.offsets.data(), bodyType, 0, MPI_COMM_WORLD);
    MPI_Type_free(&bodyType);
    if (rank == 0) {
        print_results(data, "Bodies");
    }
}

enum ValidationError { Valid, NonFinite, PositionOutOfBounds, VelocityOutOfBounds };

ValidationError validateBody(const Vec3& pos, const Vec3& vel) {
    if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
        !std::isfinite(vel.x) || !std::isfinite(vel.y) || !std::isfinite(vel.z)) {
        return NonFinite;
    }
    constexpr double maxPos = 1e6;
    constexpr double maxVel = 1e6;
    if (std::abs(pos.x) > maxPos || std::abs(pos.y) > maxPos || std::abs(pos.z) > maxPos) {
        return PositionOutOfBounds;
    }
    if (std::abs(vel.x) > maxVel || std::abs(vel.y) > maxVel || std::abs(vel.z) > maxVel) {
        return VelocityOutOfBounds;
    }
    return Valid;
}

ValidationError validateSimulation(const std::vector<Vec3>& positions,
                                   const std::vector<Vec3>& velocities, int first) {
    int local[2] = {static_cast<int>(positions.size()), Valid};
    for (size_t i = 0; i < velocities.size(); ++i) {
        const auto error = validateBody(positions[first + i], velocities[i]);
        if (error != Valid) {
            local[0] = first + static_cast<int>(i);
            local[1] = error;
            break;
        }
    }

    int global[2];
    MPI_Allreduce(local, global, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
    return static_cast<ValidationError>(global[1]);
}

void reportValidationError(ValidationError error) {
    switch (error) {
        case NonFinite:
            printf("Validation failed: found NaN or Inf value in body state\n");
            break;
        case PositionOutOfBounds:
            printf("Validation failed: body position exceeds reasonable bounds\n");
            break;
        case VelocityOutOfBounds:
            printf("Validation failed: body velocity exceeds reasonable bounds\n");
            break;
        case Valid:
            break;
    }
    printf("Validation: FAILED\n");
}

double computeLocalEnergy(const std::vector<Vec3>& positions,
                          const std::vector<Vec3>& velocities, int rank, int ranks) {
    double energy = 0.0;
    for (const Vec3& vel : velocities) {
        const double speedSquared = multiplyAdd(vel.z, vel.z, multiplyAdd(vel.x, vel.x, vel.y * vel.y));
        energy = multiplyAdd(0.5, speedSquared, energy);
    }

    // Assign equal-size ranges of the upper triangle in row-major order.
    const int64_t n = positions.size();
    const auto precedingPairs = [n](int64_t row) { return row * (2 * n - row - 1) / 2; };
    const int64_t total = precedingPairs(n);
    const int64_t perRank = total / ranks;
    const int64_t extra = total % ranks;
    const int64_t begin = perRank * rank + std::min<int64_t>(rank, extra);
    int64_t remaining = perRank + (rank < extra);
    if (remaining == 0) {
        return energy;
    }

    int64_t low = 0;
    int64_t high = n - 1;
    while (high - low > 1) {
        const int64_t mid = low + (high - low) / 2;
        if (precedingPairs(mid) <= begin) {
            low = mid;
        } else {
            high = mid;
        }
    }

    int64_t j = low + 1 + begin - precedingPairs(low);
    for (int64_t i = low; remaining > 0; ++i) {
        const Vec3 target = positions[i];
        const int64_t count = std::min(remaining, n - j);
        const int64_t end = j + count;
        for (; j < end; ++j) {
            const Vec3& source = positions[j];
            const double dx = source.x - target.x;
            const double dy = source.y - target.y;
            const double dz = source.z - target.z;
            energy -= 1.0 / std::sqrt(squaredDistance(dx, dy, dz));
        }
        remaining -= count;
        j = i + 2;
    }
    return energy;
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

// Return a status on all ranks, including when rank zero handles a command-line error.
int runSimulation(int argc, char** argv, int rank, int ranks) {
    enum Option { Bodies, Steps, Validate, Results, Stop, OptionCount };
    int options[OptionCount] = {1024, 10, 0, 0, 0};
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                options[Bodies] = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                options[Steps] = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                options[Validate] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                options[Results] = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                options[Stop] = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                options[Stop] = 2;
                break;
            }
        }
        if (!options[Stop] && options[Bodies] < 0) {
            fprintf(stderr, "Number of bodies must be nonnegative\n");
            options[Stop] = 2;
        }
    }
    MPI_Bcast(options, OptionCount, MPI_INT, 0, MPI_COMM_WORLD);
    if (options[Stop]) {
        return options[Stop] - 1;
    }

    const int numBodies = options[Bodies];
    const int numSteps = options[Steps];
    const bool validate = options[Validate];
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const Partition partition(numBodies, ranks);
    std::vector<Vec3> positions(numBodies);
    std::vector<Vec3> velocities(partition.counts[rank]);
    MPI_Datatype vecType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vecType);
    MPI_Type_commit(&vecType);

    {
        // Generate the original seed stream once, then distribute the initial state.
        std::vector<Vec3> initialVelocities(rank == 0 ? numBodies : 0);
        if (rank == 0) {
            randomizeBodies(positions, initialVelocities);
        }
        MPI_Scatterv(initialVelocities.data(), partition.counts.data(), partition.offsets.data(),
                     vecType, velocities.data(), partition.counts[rank], vecType, 0, MPI_COMM_WORLD);
        MPI_Bcast(positions.data(), numBodies, vecType, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        advanceBodies(positions, velocities, partition.offsets[rank]);
        if (step + 1 < numSteps) {
            exchangePositions(positions, partition, vecType);
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Simulation time: %lld ms\n", static_cast<long long>(maxElapsed * 1000.0));
    }

    if (options[Results]) {
        reportBodies(positions, velocities, partition, rank);
    }

    int status = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
        }
        const auto error = validateSimulation(positions, velocities, partition.offsets[rank]);
        if (error == Valid) {
            if (numSteps > 0) {
                exchangePositions(positions, partition, vecType);
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
                reportValidationError(error);
            }
            status = 1;
        }
    }

    MPI_Type_free(&vecType);
    return status;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        const int status = runSimulation(argc, argv, rank, ranks);
        MPI_Finalize();
        return status;
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
}
