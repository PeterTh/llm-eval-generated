#include <mpi.h>

#include <algorithm>
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

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Vec3) == 3 * sizeof(double));
static_assert(sizeof(Body) == 6 * sizeof(double));

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

// Each rank evaluates a disjoint set of force rows.  Positions are replicated,
// but velocities remain local because no other rank needs them during a step.
void computeForces(const std::vector<Vec3>& positions,
                   std::vector<Vec3>& localVelocities,
                   const int globalOffset) {
    const size_t n = positions.size();
    const size_t localN = localVelocities.size();

    for (size_t localI = 0; localI < localN; ++localI) {
        const Vec3 pi = positions[static_cast<size_t>(globalOffset) + localI];
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        // Keep the same global j order as the serial implementation.  Besides
        // reproducibility, the single flat loop is friendly to vectorization.
        for (size_t j = 0; j < n; ++j) {
            const double dx = positions[j].x - pi.x;
            const double dy = positions[j].y - pi.y;
            const double dz = positions[j].z - pi.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        localVelocities[localI].x += DT * Fx;
        localVelocities[localI].y += DT * Fy;
        localVelocities[localI].z += DT * Fz;
    }
}

void integrateBodies(std::vector<Vec3>& localPositions,
                     const std::vector<Vec3>& localVelocities) {
    for (size_t i = 0; i < localPositions.size(); ++i) {
        localPositions[i].x += localVelocities[i].x * DT;
        localPositions[i].y += localVelocities[i].y * DT;
        localPositions[i].z += localVelocities[i].z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    return energy;
}

bool validateLocalSimulation(const std::vector<Vec3>& positions,
                             const std::vector<Vec3>& velocities) {
    constexpr double maxPos = 1e6;
    constexpr double maxVel = 1e6;
    for (size_t i = 0; i < positions.size(); ++i) {
        const Vec3& pos = positions[i];
        const Vec3& vel = velocities[i];
        if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
            !std::isfinite(vel.x) || !std::isfinite(vel.y) || !std::isfinite(vel.z)) {
            return false;
        }
        if (std::abs(pos.x) > maxPos || std::abs(pos.y) > maxPos || std::abs(pos.z) > maxPos ||
            std::abs(vel.x) > maxVel || std::abs(vel.y) > maxVel || std::abs(vel.z) > maxVel) {
            return false;
        }
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

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
            showHelp = true;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            argumentsValid = false;
        }
    }

    if (numBodies < 0 || numSteps < 0) {
        if (rank == 0) printf("The number of bodies and steps must be non-negative.\n");
        argumentsValid = false;
    }
    if (showHelp || !argumentsValid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", worldSize);
    }

    // Balanced contiguous ownership also supplies the counts/displacements for
    // all collectives.  A rank may own zero bodies when P > N.
    std::vector<int> counts(worldSize), displacements(worldSize);
    const int quotient = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    int offset = 0;
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = quotient + (r < remainder ? 1 : 0);
        displacements[r] = offset;
        offset += counts[r];
    }
    const int localN = counts[rank];

    MPI_Datatype vec3Type, bodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);

    std::vector<Body> initialBodies;
    if (rank == 0) {
        initialBodies.resize(static_cast<size_t>(numBodies));
        randomizeBodies(initialBodies);
    }
    std::vector<Body> localBodies(static_cast<size_t>(localN));
    MPI_Scatterv(rank == 0 ? initialBodies.data() : nullptr,
                 counts.data(), displacements.data(), bodyType,
                 localBodies.data(), localN, bodyType, 0, MPI_COMM_WORLD);

    std::vector<Vec3> localPositions(static_cast<size_t>(localN));
    std::vector<Vec3> localVelocities(static_cast<size_t>(localN));
    for (int i = 0; i < localN; ++i) {
        localPositions[i] = localBodies[i].pos;
        localVelocities[i] = localBodies[i].vel;
    }
    // Release the initialization buffers before entering the memory-intensive
    // force loop.  Only root will allocate a full Body array again if needed.
    std::vector<Body>().swap(localBodies);
    std::vector<Body>().swap(initialBodies);

    std::vector<Vec3> positions(static_cast<size_t>(numBodies));
    MPI_Allgatherv(localPositions.data(), localN, vec3Type,
                   positions.data(), counts.data(), displacements.data(), vec3Type,
                   MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        computeForces(positions, localVelocities, displacements[rank]);
        integrateBodies(localPositions, localVelocities);
        MPI_Allgatherv(localPositions.data(), localN, vec3Type,
                       positions.data(), counts.data(), displacements.data(), vec3Type,
                       MPI_COMM_WORLD);
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));

    // Gather complete bodies only for output/energy calculation; the timed path
    // communicates positions alone.
    std::vector<Body> gatheredBodies;
    if (printResults || validate) {
        localBodies.resize(static_cast<size_t>(localN));
        for (int i = 0; i < localN; ++i) {
            localBodies[i] = Body{localPositions[i], localVelocities[i]};
        }
        if (rank == 0) gatheredBodies.resize(static_cast<size_t>(numBodies));
        MPI_Gatherv(localBodies.data(), localN, bodyType,
                    rank == 0 ? gatheredBodies.data() : nullptr,
                    counts.data(), displacements.data(), bodyType, 0, MPI_COMM_WORLD);
    }

    if (printResults && rank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
        for (const auto& body : gatheredBodies) {
            bodyData.push_back(body.pos.x);
            bodyData.push_back(body.pos.y);
            bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x);
            bodyData.push_back(body.vel.y);
            bodyData.push_back(body.vel.z);
        }
        print_results(bodyData, "Bodies");
    }

    int exitCode = 0;
    if (validate) {
        const int localValid = validateLocalSimulation(localPositions, localVelocities) ? 1 : 0;
        int globallyValid = 0;
        MPI_Reduce(&localValid, &globallyValid, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating simulation results...\n");
            if (globallyValid) {
                printf("Final energy: %.6f\n", computeTotalEnergy(gatheredBodies));
                printf("Validation: PASSED\n");
            } else {
                printf("Validation failed: found non-finite or out-of-bounds body state\n");
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Type_free(&bodyType);
    MPI_Type_free(&vec3Type);
    MPI_Finalize();
    return exitCode;
}
