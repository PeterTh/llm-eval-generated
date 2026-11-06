#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};
static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must contain three contiguous doubles");

// Every rank generates the same position snapshot. Only the velocities of its
// own bodies are retained; rand_r's sequence is identical to the serial code.
void randomizeBodies(std::vector<Vec3>& positions, std::vector<Vec3>& velocities,
                     int numBodies, int firstBody, unsigned int seed = 42) {
    for (int i = 0; i < numBodies; ++i) {
        Vec3 pos, vel;
        pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        positions[i] = pos;
        if (i >= firstBody && i - firstBody < static_cast<int>(velocities.size())) {
            velocities[i - firstBody] = vel;
        }
    }
}

void computeForces(const std::vector<Vec3>& positions, std::vector<Vec3>& velocities,
                   int numBodies, int firstBody) {
    for (size_t local = 0; local < velocities.size(); ++local) {
        const Vec3& pos = positions[firstBody + local];
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        for (int j = 0; j < numBodies; ++j) {
            const double dx = positions[j].x - pos.x;
            const double dy = positions[j].y - pos.y;
            const double dz = positions[j].z - pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        velocities[local].x += DT * Fx;
        velocities[local].y += DT * Fy;
        velocities[local].z += DT * Fz;
    }
}

void integrateBodies(std::vector<Vec3>& positions, const std::vector<Vec3>& velocities,
                     int firstBody) {
    for (size_t i = 0; i < velocities.size(); ++i) {
        positions[firstBody + i].x += velocities[i].x * DT;
        positions[firstBody + i].y += velocities[i].y * DT;
        positions[firstBody + i].z += velocities[i].z * DT;
    }
}

// The potential-energy rows are assigned cyclically to balance the triangular
// iteration space, independently of the contiguous simulation partition.
double computeLocalEnergy(const std::vector<Vec3>& positions, const std::vector<Vec3>& velocities,
                          int numBodies, int rank, int ranks) {
    double energy = 0.0;
    for (const auto& vel : velocities) {
        energy += 0.5 * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
    }
    for (int i = rank; i < numBodies; ) {
        for (int j = i + 1; j < numBodies; ++j) {
            const double dx = positions[j].x - positions[i].x;
            const double dy = positions[j].y - positions[i].y;
            const double dz = positions[j].z - positions[i].z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
        if (numBodies - i <= ranks) break;
        i += ranks;
    }
    return energy;
}

// Return the same first failure and diagnostic as the serial validation.
int validationFailure(const Vec3& pos, const Vec3& vel) {
    if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
        !std::isfinite(vel.x) || !std::isfinite(vel.y) || !std::isfinite(vel.z)) return 1;
    if (std::abs(pos.x) > 1e6 || std::abs(pos.y) > 1e6 || std::abs(pos.z) > 1e6) return 2;
    if (std::abs(vel.x) > 1e6 || std::abs(vel.y) > 1e6 || std::abs(vel.z) > 1e6) return 3;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    bool stop = false;
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
            if (rank == 0) printUsage(argv[0]);
            stop = true;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            stop = true;
            break;
        }
    }
    if (!stop && numBodies < 0) {
        if (rank == 0) printf("Number of bodies must be nonnegative\n");
        exitCode = 1;
        stop = true;
    }
    if (stop) {
        MPI_Finalize();
        return exitCode;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks ? 1 : 0);
        offsets[r] = r == 0 ? 0 : offsets[r - 1] + counts[r - 1];
    }
    const int firstBody = offsets[rank];
    // Keep a valid receive buffer even for an empty simulation.
    std::vector<Vec3> positions(std::max(1, numBodies));
    std::vector<Vec3> velocities(counts[rank]);
    randomizeBodies(positions, velocities, numBodies, firstBody);

    MPI_Datatype vec3Type;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        computeForces(positions, velocities, numBodies, firstBody);
        integrateBodies(positions, velocities, firstBody);
        // Validation needs the final snapshot on all ranks.
        if (step + 1 < numSteps || validate) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, vec3Type, positions.data(), counts.data(),
                           offsets.data(), vec3Type, MPI_COMM_WORLD);
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));

    if (printResults) {
        // Without validation, only rank 0 needs the final positions.
        if (!validate && numSteps > 0) {
            MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : positions.data() + firstBody,
                        counts[rank], vec3Type, positions.data(), counts.data(),
                        offsets.data(), vec3Type, 0, MPI_COMM_WORLD);
        }
        std::vector<Vec3> allVelocities(rank == 0 ? std::max(1, numBodies) : 0);
        MPI_Gatherv(velocities.data(), counts[rank], vec3Type, allVelocities.data(), counts.data(),
                    offsets.data(), vec3Type, 0, MPI_COMM_WORLD);
        if (rank == 0) {
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
    }
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        int localFailure[2] = {numBodies, 0};
        for (int i = 0; i < counts[rank]; ++i) {
            const int reason = validationFailure(positions[firstBody + i], velocities[i]);
            if (reason != 0) {
                localFailure[0] = firstBody + i;
                localFailure[1] = reason;
                break;
            }
        }
        int firstFailure[2];
        MPI_Allreduce(localFailure, firstFailure, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
        if (firstFailure[0] == numBodies) {
            const double localEnergy = computeLocalEnergy(positions, velocities, numBodies, rank, ranks);
            double finalEnergy = 0.0;
            MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                if (firstFailure[1] == 1) printf("Validation failed: found NaN or Inf value in body state\n");
                else if (firstFailure[1] == 2) printf("Validation failed: body position exceeds reasonable bounds\n");
                else printf("Validation failed: body velocity exceeds reasonable bounds\n");
                printf("Validation: FAILED\n");
            }
            exitCode = 1;
        }
    }
    MPI_Type_free(&vec3Type);
    MPI_Finalize();
    return exitCode;
}
