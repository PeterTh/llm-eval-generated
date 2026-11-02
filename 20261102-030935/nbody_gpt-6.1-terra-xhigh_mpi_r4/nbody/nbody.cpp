#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must match the MPI datatype");

struct Body {
    Vec3 pos;
    Vec3 vel;
};

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

void computeForces(const std::vector<Vec3>& positions, std::vector<Vec3>& velocities, int first) {
    const size_t n = positions.size();
    for (size_t local = 0; local < velocities.size(); ++local) {
        const Vec3 pos = positions[first + local];
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        for (size_t j = 0; j < n; ++j) {
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

// Return the first invalid body's index and its error, so rank zero can report
// the same failure that a serial traversal would have found.
int validationError(const Vec3& pos, const Vec3& vel) {
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (numBodies < 0) {
        if (rank == 0) fprintf(stderr, "Number of bodies must be nonnegative\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> counts(size), displs(size);
    for (int r = 0, offset = 0; r < size; ++r) {
        counts[r] = numBodies / size + (r < numBodies % size);
        displs[r] = offset;
        offset += counts[r];
    }
    const int localCount = counts[rank];
    const int first = displs[rank];
    std::vector<Vec3> positions(numBodies), velocities(localCount);
    std::vector<Vec3> initialVelocities;
    if (rank == 0) {
        std::vector<Body> bodies(numBodies);
        randomizeBodies(bodies);
        initialVelocities.resize(numBodies);
        for (int i = 0; i < numBodies; ++i) {
            positions[i] = bodies[i].pos;
            initialVelocities[i] = bodies[i].vel;
        }
    }
    MPI_Datatype vec3Type;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);
    MPI_Bcast(positions.data(), numBodies, vec3Type, 0, MPI_COMM_WORLD);
    MPI_Scatterv(initialVelocities.data(), counts.data(), displs.data(), vec3Type,
                 velocities.data(), localCount, vec3Type, 0, MPI_COMM_WORLD);
    std::vector<Vec3>().swap(initialVelocities);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        computeForces(positions, velocities, first);
        for (int i = 0; i < localCount; ++i) {
            positions[first + i].x += velocities[i].x * DT;
            positions[first + i].y += velocities[i].y * DT;
            positions[first + i].z += velocities[i].z * DT;
        }
        // All ranks need the same position snapshot before the next force step.
        if (step + 1 < numSteps || printResults || validate) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, vec3Type, positions.data(), counts.data(),
                           displs.data(), vec3Type, MPI_COMM_WORLD);
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));

    if (printResults) {
        std::vector<Vec3> allVelocities(rank == 0 ? numBodies : 0);
        MPI_Gatherv(velocities.data(), localCount, vec3Type, allVelocities.data(),
                    counts.data(), displs.data(), vec3Type, 0, MPI_COMM_WORLD);
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

    int exitCode = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        int localFailure[2] = {numBodies, 0};
        for (int i = 0; i < localCount; ++i) {
            const int error = validationError(positions[first + i], velocities[i]);
            if (error) {
                localFailure[0] = first + i;
                localFailure[1] = error;
                break;
            }
        }
        int failure[2];
        MPI_Allreduce(localFailure, failure, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
        if (failure[0] != numBodies) {
            if (rank == 0) {
                if (failure[1] == 1) printf("Validation failed: found NaN or Inf value in body state\n");
                else if (failure[1] == 2) printf("Validation failed: body position exceeds reasonable bounds\n");
                else printf("Validation failed: body velocity exceeds reasonable bounds\n");
                printf("Validation: FAILED\n");
            }
            exitCode = 1;
        } else {
            double localEnergy = 0.0;
            for (const Vec3& vel : velocities) {
                localEnergy += 0.5 * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
            }
            // Cyclic rows balance the triangular pairwise potential calculation.
            for (size_t i = static_cast<size_t>(rank); i < positions.size(); i += static_cast<size_t>(size)) {
                for (size_t j = i + 1; j < positions.size(); ++j) {
                    const double dx = positions[j].x - positions[i].x;
                    const double dy = positions[j].y - positions[i].y;
                    const double dz = positions[j].z - positions[i].z;
                    const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                    localEnergy -= 1.0 / dist;
                }
            }
            double energy = 0.0;
            MPI_Reduce(&localEnergy, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Final energy: %.6f\n", energy);
                printf("Validation: PASSED\n");
            }
        }
    }
    MPI_Type_free(&vec3Type);
    MPI_Finalize();
    return exitCode;
}
