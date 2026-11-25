#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if __has_include(<experimental/simd>)
#include <experimental/simd>
#define NBODY_HAS_SIMD 1
#endif

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

template<typename T>
struct Components {
    T x, y, z;
};

using Vec3 = Components<double>;
static_assert(sizeof(Vec3) == 3 * sizeof(double));

struct BodyPartition {
    std::vector<int> counts;
    std::vector<int> offsets;
    bool uniform;

    BodyPartition(int numBodies, int ranks)
        : counts(ranks), offsets(ranks), uniform(numBodies % ranks == 0) {
        const int perRank = numBodies / ranks;
        const int remainder = numBodies % ranks;
        int offset = 0;
        for (int rank = 0; rank < ranks; ++rank) {
            counts[rank] = perRank + (rank < remainder);
            offsets[rank] = offset;
            offset += counts[rank];
        }
    }
};

class MpiDoubleRecord {
public:
    explicit MpiDoubleRecord(int fields) {
        MPI_Type_contiguous(fields, MPI_DOUBLE, &type);
        MPI_Type_commit(&type);
    }

    ~MpiDoubleRecord() { MPI_Type_free(&type); }

    MpiDoubleRecord(const MpiDoubleRecord&) = delete;
    MpiDoubleRecord& operator=(const MpiDoubleRecord&) = delete;

    MPI_Datatype type;
};

void initializeBodies(std::vector<Vec3>& positions, std::vector<Vec3>& velocities,
                      const BodyPartition& partition, int rank, MPI_Datatype vecType) {
    std::vector<Vec3> initialVelocities;
    if (rank == 0) {
        // Preserve the serial random stream and its position/velocity interleaving.
        unsigned int seed = 42;
        const auto randomComponent = [&seed]() {
            return 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        };
        initialVelocities.resize(positions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            positions[i] = {randomComponent(), randomComponent(), randomComponent()};
            initialVelocities[i] = {randomComponent(), randomComponent(), randomComponent()};
        }
    }

    MPI_Scatterv(initialVelocities.data(), partition.counts.data(), partition.offsets.data(),
                 vecType, velocities.data(), partition.counts[rank], vecType, 0, MPI_COMM_WORLD);
    MPI_Bcast(positions.data(), static_cast<int>(positions.size()), vecType, 0, MPI_COMM_WORLD);
}

template<typename T>
Components<T> sumForces(const std::vector<Vec3>& positions, const Components<T>& target) {
    using std::fma;
    using std::sqrt;
    Components<T> force{T(0.0), T(0.0), T(0.0)};
    for (const Vec3& source : positions) {
        const T dx = source.x - target.x;
        const T dy = source.y - target.y;
        const T dz = source.z - target.z;
        // Match the serial kernel's contraction order when native FMA is available.
#if defined(FP_FAST_FMA) || defined(__FMA__)
        const T distSqr = fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
#else
        const T distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
#endif
        const T invDist = 1.0 / sqrt(distSqr);
        const T invDist3 = invDist * invDist * invDist;

#if defined(FP_FAST_FMA) || defined(__FMA__)
        force.x = fma(dx, invDist3, force.x);
        force.y = fma(dy, invDist3, force.y);
        force.z = fma(dz, invDist3, force.z);
#else
        force.x += dx * invDist3;
        force.y += dy * invDist3;
        force.z += dz * invDist3;
#endif
    }
    return force;
}

void updateVelocity(Vec3& velocity, const Vec3& force) {
    // Round each velocity increment before adding it, as in the serial kernel.
    const volatile double dvx = DT * force.x;
    const volatile double dvy = DT * force.y;
    const volatile double dvz = DT * force.z;
    velocity.x += dvx;
    velocity.y += dvy;
    velocity.z += dvz;
}

void computeForces(const std::vector<Vec3>& positions, std::vector<Vec3>& velocities,
                   size_t offset) {
#ifdef NBODY_HAS_SIMD
    // Each SIMD lane sums forces for one target in the original source order.
    using Simd = std::experimental::native_simd<double>;
    constexpr size_t width = Simd::size();
    for (size_t first = 0; first < velocities.size(); first += width) {
        const size_t count = std::min(width, velocities.size() - first);
        const auto target = [&](size_t lane) -> const Vec3& {
            return positions[offset + first + std::min(lane, count - 1)];
        };
        const Components<Simd> targets{
            Simd([&](auto lane) { return target(lane).x; }),
            Simd([&](auto lane) { return target(lane).y; }),
            Simd([&](auto lane) { return target(lane).z; })
        };
        const auto force = sumForces(positions, targets);
        for (size_t lane = 0; lane < count; ++lane) {
            updateVelocity(velocities[first + lane], {force.x[lane], force.y[lane], force.z[lane]});
        }
    }
#else
    for (size_t i = 0; i < velocities.size(); ++i) {
        updateVelocity(velocities[i], sumForces(positions, positions[offset + i]));
    }
#endif
}

void integrateBodies(std::vector<Vec3>& positions, const std::vector<Vec3>& velocities,
                     size_t offset) {
    for (size_t i = 0; i < velocities.size(); ++i) {
        Vec3& pos = positions[offset + i];
        const Vec3& vel = velocities[i];
        pos.x += vel.x * DT;
        pos.y += vel.y * DT;
        pos.z += vel.z * DT;
    }
}

void exchangePositions(std::vector<Vec3>& positions, const BodyPartition& partition,
                       MPI_Datatype vecType) {
    if (partition.counts.size() == 1 || positions.empty()) {
        return;
    }
    if (partition.uniform) {
        MPI_Allgather(MPI_IN_PLACE, 0, vecType, positions.data(), partition.counts[0],
                      vecType, MPI_COMM_WORLD);
    } else {
        MPI_Allgatherv(MPI_IN_PLACE, 0, vecType, positions.data(), partition.counts.data(),
                       partition.offsets.data(), vecType, MPI_COMM_WORLD);
    }
}

void printBodyResults(const std::vector<Vec3>& positions, const std::vector<Vec3>& velocities,
                      const BodyPartition& partition, int rank) {
    constexpr size_t fields = 6;
    const size_t offset = partition.offsets[rank];
    std::vector<double> data(fields * (rank == 0 ? positions.size() : velocities.size()));
    for (size_t i = 0; i < velocities.size(); ++i) {
        const Vec3& pos = positions[offset + i];
        const Vec3& vel = velocities[i];
        double* body = data.data() + fields * i;
        body[0] = pos.x;
        body[1] = pos.y;
        body[2] = pos.z;
        body[3] = vel.x;
        body[4] = vel.y;
        body[5] = vel.z;
    }

    const MpiDoubleRecord bodyType(fields);
    if (rank == 0) {
        MPI_Gatherv(MPI_IN_PLACE, 0, bodyType.type, data.data(), partition.counts.data(),
                    partition.offsets.data(), bodyType.type, 0, MPI_COMM_WORLD);
        print_results(data, "Bodies");
    } else {
        MPI_Gatherv(data.data(), partition.counts[rank], bodyType.type, nullptr, nullptr,
                    nullptr, bodyType.type, 0, MPI_COMM_WORLD);
    }
}

double computeLocalEnergy(const std::vector<Vec3>& positions, const std::vector<Vec3>& velocities,
                          int rank, int ranks) {
    double energy = 0.0;
    for (const Vec3& vel : velocities) {
        energy += 0.5 * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
    }

    // Distribute the triangular pair loop cyclically to balance the work.
    for (size_t i = rank; i < positions.size(); i += ranks) {
        const Vec3& target = positions[i];
        for (size_t j = i + 1; j < positions.size(); ++j) {
            const double dx = positions[j].x - target.x;
            const double dy = positions[j].y - target.y;
            const double dz = positions[j].z - target.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    return energy;
}

bool validateSimulation(const std::vector<Vec3>& positions, const std::vector<Vec3>& velocities,
                        size_t offset, int rank, int ranks) {
    constexpr const char* errors[] = {
        "found NaN or Inf value in body state",
        "body position exceeds reasonable bounds",
        "body velocity exceeds reasonable bounds"
    };
    constexpr long long noError = std::numeric_limits<long long>::max();
    const auto finite = [](const Vec3& v) {
        return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
    };
    const auto outOfBounds = [](const Vec3& v) {
        constexpr double maxMagnitude = 1e6;
        return std::abs(v.x) > maxMagnitude || std::abs(v.y) > maxMagnitude ||
               std::abs(v.z) > maxMagnitude;
    };

    long long localError = noError;
    for (size_t i = 0; i < velocities.size(); ++i) {
        const Vec3& pos = positions[offset + i];
        const Vec3& vel = velocities[i];
        const auto bodyError = 3LL * static_cast<long long>(offset + i);
        if (!finite(pos) || !finite(vel)) {
            localError = bodyError;
        } else if (outOfBounds(pos)) {
            localError = bodyError + 1;
        } else if (outOfBounds(vel)) {
            localError = bodyError + 2;
        }
        if (localError != noError) {
            break;
        }
    }

    long long firstError;
    MPI_Allreduce(&localError, &firstError, 1, MPI_LONG_LONG_INT, MPI_MIN, MPI_COMM_WORLD);
    if (firstError != noError) {
        if (rank == 0) {
            printf("Validation failed: %s\n", errors[firstError % 3]);
            printf("Validation: FAILED\n");
        }
        return false;
    }

    const double localEnergy = computeLocalEnergy(positions, velocities, rank, ranks);
    double finalEnergy;
    MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Final energy: %.6f\n", finalEnergy);
        printf("Validation: PASSED\n");
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

struct Options {
    int numBodies = 1024;
    int numSteps = 10;
    int validate = 0;
    int printResults = 0;
};

// Return -1 to run the simulation, or the exit status for help and errors.
int parseOptions(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            options.numBodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            options.numSteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            options.validate = 1;
        } else if (strcmp(argv[i], "-r") == 0) {
            options.printResults = 1;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    if (options.numBodies < 0) {
        printf("Number of bodies must be nonnegative\n");
        return 1;
    }
    return -1;
}

int runSimulation(const Options& options, int rank, int ranks) {
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", options.numBodies);
        printf("Number of steps: %d\n", options.numSteps);
        printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
    }

    const BodyPartition partition(options.numBodies, ranks);
    std::vector<Vec3> positions(options.numBodies);
    std::vector<Vec3> velocities(partition.counts[rank]);
    const size_t offset = partition.offsets[rank];
    const MpiDoubleRecord vecType(3);
    initializeBodies(positions, velocities, partition, rank, vecType.type);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < options.numSteps; ++step) {
        computeForces(positions, velocities, offset);
        integrateBodies(positions, velocities, offset);
        // Each rank already has its own final positions. Only another force step
        // or the optional energy calculation needs the other ranks' positions.
        if (step + 1 < options.numSteps) {
            exchangePositions(positions, partition, vecType.type);
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %lld ms\n", static_cast<long long>(1000.0 * maxElapsed));
    }
    if (options.printResults) {
        printBodyResults(positions, velocities, partition, rank);
    }
    if (options.validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
        }
        if (options.numSteps > 0) {
            exchangePositions(positions, partition, vecType.type);
        }
        if (!validateSimulation(positions, velocities, offset, rank, ranks)) {
            return 1;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        return 1;
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int config[5];
    if (rank == 0) {
        Options options;
        config[0] = parseOptions(argc, argv, options);
        config[1] = options.numBodies;
        config[2] = options.numSteps;
        config[3] = options.validate;
        config[4] = options.printResults;
    }
    MPI_Bcast(config, 5, MPI_INT, 0, MPI_COMM_WORLD);

    int status = config[0];
    if (status == -1) {
        const Options options{config[1], config[2], config[3], config[4]};
        status = runSimulation(options, rank, ranks);
    }
    MPI_Finalize();
    return status;
}
