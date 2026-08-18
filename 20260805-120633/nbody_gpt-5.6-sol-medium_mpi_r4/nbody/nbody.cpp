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
};

static_assert(std::is_standard_layout_v<Vec3>);
static_assert(sizeof(Vec3) == 3 * sizeof(double));

// Each rank owns [displacements[rank], displacements[rank] + counts[rank]).
// Keeping the ranges contiguous makes both the force kernel and collectives
// operate on contiguous memory, including when N is not divisible by P.
void makeDecomposition(int n, int ranks, std::vector<int>& counts,
                       std::vector<int>& displacements) {
    counts.resize(ranks);
    displacements.resize(ranks);
    const int base = n / ranks;
    const int remainder = n % ranks;
    int offset = 0;
    for (int rank = 0; rank < ranks; ++rank) {
        counts[rank] = base + (rank < remainder ? 1 : 0);
        displacements[rank] = offset;
        offset += counts[rank];
    }
}

// Generate in exactly the same body/component order as the serial program.
void randomizeBodies(std::vector<Vec3>& positions,
                     std::vector<Vec3>& velocities, unsigned int seed = 42) {
    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i].x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        positions[i].y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        positions[i].z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velocities[i].x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velocities[i].y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velocities[i].z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// All ranks hold the current positions, but calculate only their owned bodies.
// The j loop deliberately retains serial order, so a body's force has the same
// floating-point evaluation order regardless of the number of MPI ranks.
void computeForces(const Vec3* __restrict positions,
                   Vec3* __restrict localVelocities, int n, int first,
                   int localCount) {
    for (int local = 0; local < localCount; ++local) {
        const Vec3 pi = positions[first + local];
        double fx = 0.0;
        double fy = 0.0;
        double fz = 0.0;

        for (int j = 0; j < n; ++j) {
            const double dx = positions[j].x - pi.x;
            const double dy = positions[j].y - pi.y;
            const double dz = positions[j].z - pi.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            fx += dx * invDist3;
            fy += dy * invDist3;
            fz += dz * invDist3;
        }

        localVelocities[local].x += DT * fx;
        localVelocities[local].y += DT * fy;
        localVelocities[local].z += DT * fz;
    }
}

void integrateBodies(Vec3* __restrict positions,
                     const Vec3* __restrict localVelocities, int first,
                     int localCount) {
    for (int local = 0; local < localCount; ++local) {
        Vec3& position = positions[first + local];
        position.x += localVelocities[local].x * DT;
        position.y += localVelocities[local].y * DT;
        position.z += localVelocities[local].z * DT;
    }
}

double computeTotalEnergy(const std::vector<Vec3>& positions,
                          const std::vector<Vec3>& velocities) {
    double energy = 0.0;
    const size_t n = positions.size();
    for (const Vec3& velocity : velocities) {
        energy += 0.5 * (velocity.x * velocity.x + velocity.y * velocity.y +
                        velocity.z * velocity.z);
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = positions[j].x - positions[i].x;
            const double dy = positions[j].y - positions[i].y;
            const double dz = positions[j].z - positions[i].z;
            const double distance =
                std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / distance;
        }
    }
    return energy;
}

// Return a stable reason code so only rank zero emits validation diagnostics.
int validateLocal(const std::vector<Vec3>& positions,
                  const std::vector<Vec3>& velocities, int first) {
    constexpr double maxPosition = 1e6;
    constexpr double maxVelocity = 1e6;
    for (size_t local = 0; local < velocities.size(); ++local) {
        const Vec3& p = positions[first + local];
        const Vec3& v = velocities[local];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
            !std::isfinite(p.z) || !std::isfinite(v.x) ||
            !std::isfinite(v.y) || !std::isfinite(v.z)) {
            return 3;
        }
        if (std::abs(p.x) > maxPosition || std::abs(p.y) > maxPosition ||
            std::abs(p.z) > maxPosition) {
            return 2;
        }
        if (std::abs(v.x) > maxVelocity || std::abs(v.y) > maxVelocity ||
            std::abs(v.z) > maxVelocity) {
            return 1;
        }
    }
    return 0;
}

void printUsage(const char* programName) {
    printf("Usage: %s [options]\n", programName);
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
    int parseStatus = 0;

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
            parseStatus = 2;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
            parseStatus = 1;
            break;
        }
    }

    if (numBodies < 0 || numSteps < 0) {
        if (rank == 0) {
            printf("Number of bodies and steps must be non-negative\n");
        }
        parseStatus = 1;
    }
    if (parseStatus != 0) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> counts;
    std::vector<int> displacements;
    makeDecomposition(numBodies, rankCount, counts, displacements);
    const int localCount = counts[rank];
    const int first = displacements[rank];

    MPI_Datatype vec3Type;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);

    std::vector<Vec3> positions(static_cast<size_t>(numBodies));
    std::vector<Vec3> allVelocities;
    if (rank == 0) {
        allVelocities.resize(static_cast<size_t>(numBodies));
        randomizeBodies(positions, allVelocities);
    }
    std::vector<Vec3> localVelocities(static_cast<size_t>(localCount));

    MPI_Bcast(positions.data(), numBodies, vec3Type, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? allVelocities.data() : nullptr, counts.data(),
                 displacements.data(), vec3Type, localVelocities.data(),
                 localCount, vec3Type, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        allVelocities.clear();
        allVelocities.shrink_to_fit();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(positions.data(), localVelocities.data(), numBodies,
                      first, localCount);
        integrateBodies(positions.data(), localVelocities.data(), first,
                        localCount);

        // Updated owned positions already reside in their receive locations.
        MPI_Allgatherv(MPI_IN_PLACE, 0, vec3Type, positions.data(),
                       counts.data(), displacements.data(), vec3Type,
                       MPI_COMM_WORLD);
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Simulation time: %lld ms\n",
               static_cast<long long>(elapsed * 1000.0));
    }

    int validationCode = 0;
    if (validate) {
        const int localCode =
            validateLocal(positions, localVelocities, first);
        MPI_Allreduce(&localCode, &validationCode, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
    }

    // Velocities need not be communicated during the simulation. Gather them
    // only for the optional serial-format output and energy calculation.
    if (printResults || validate) {
        if (rank == 0) {
            allVelocities.resize(static_cast<size_t>(numBodies));
        }
        MPI_Gatherv(localVelocities.data(), localCount, vec3Type,
                    rank == 0 ? allVelocities.data() : nullptr, counts.data(),
                    displacements.data(), vec3Type, 0, MPI_COMM_WORLD);
    }

    if (rank == 0 && printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
        for (int i = 0; i < numBodies; ++i) {
            bodyData.push_back(positions[i].x);
            bodyData.push_back(positions[i].y);
            bodyData.push_back(positions[i].z);
            bodyData.push_back(allVelocities[i].x);
            bodyData.push_back(allVelocities[i].y);
            bodyData.push_back(allVelocities[i].z);
        }
        print_results(bodyData, "Bodies");
    }

    int exitCode = 0;
    if (rank == 0 && validate) {
        printf("Validating simulation results...\n");
        if (validationCode == 0) {
            const double finalEnergy =
                computeTotalEnergy(positions, allVelocities);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            if (validationCode == 3) {
                printf("Validation failed: found NaN or Inf value in body state\n");
            } else if (validationCode == 2) {
                printf("Validation failed: body position exceeds reasonable bounds\n");
            } else {
                printf("Validation failed: body velocity exceeds reasonable bounds\n");
            }
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Type_free(&vec3Type);
    MPI_Finalize();
    return exitCode;
}
