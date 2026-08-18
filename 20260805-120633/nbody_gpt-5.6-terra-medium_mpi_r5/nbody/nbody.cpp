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

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Vec3) == 3 * sizeof(double));
static_assert(sizeof(Body) == 6 * sizeof(double));

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

void computeForces(std::vector<Body>& localBodies, const std::vector<Vec3>& positions) {
    const int n = static_cast<int>(positions.size());
    for (Body& body : localBodies) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        const double px = body.pos.x;
        const double py = body.pos.y;
        const double pz = body.pos.z;

        for (int j = 0; j < n; ++j) {
            const double dx = positions[j].x - px;
            const double dy = positions[j].y - py;
            const double dz = positions[j].z - pz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            fx += dx * invDist3;
            fy += dy * invDist3;
            fz += dz * invDist3;
        }

        body.vel.x += DT * fx;
        body.vel.y += DT * fy;
        body.vel.z += DT * fz;
    }
}

void integrateBodies(std::vector<Body>& bodies) {
    for (Body& body : bodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

double computeLocalEnergy(const std::vector<Body>& localBodies, const std::vector<Vec3>& positions,
                          int globalOffset) {
    double energy = 0.0;
    const int n = static_cast<int>(positions.size());
    for (int local = 0; local < static_cast<int>(localBodies.size()); ++local) {
        const Body& body = localBodies[local];
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
        for (int j = globalOffset + local + 1; j < n; ++j) {
            const double dx = positions[j].x - body.pos.x;
            const double dy = positions[j].y - body.pos.y;
            const double dz = positions[j].z - body.pos.z;
            energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    return energy;
}

bool validateLocalSimulation(const std::vector<Body>& bodies) {
    for (const Body& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z) ||
            std::abs(body.pos.x) > 1e6 || std::abs(body.pos.y) > 1e6 || std::abs(body.pos.z) > 1e6 ||
            std::abs(body.vel.x) > 1e6 || std::abs(body.vel.y) > 1e6 || std::abs(body.vel.z) > 1e6) {
            return false;
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks energy conservation)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (numBodies < 0 || numSteps < 0) {
        if (rank == 0) std::printf("Number of bodies and steps must be non-negative\n");
        MPI_Finalize();
        return 1;
    }

    std::vector<int> counts(worldSize), displacements(worldSize);
    const int baseCount = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    for (int r = 0, offset = 0; r < worldSize; ++r) {
        counts[r] = baseCount + (r < remainder ? 1 : 0);
        displacements[r] = offset;
        offset += counts[r];
    }
    const int localCount = counts[rank];
    const int localOffset = displacements[rank];

    MPI_Datatype vec3Type;
    MPI_Datatype bodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);

    std::vector<Body> initialBodies;
    if (rank == 0) {
        initialBodies.resize(numBodies);
        randomizeBodies(initialBodies);
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<Body> localBodies(localCount);
    MPI_Scatterv(rank == 0 ? initialBodies.data() : nullptr, counts.data(), displacements.data(), bodyType,
                 localBodies.data(), localCount, bodyType, 0, MPI_COMM_WORLD);

    std::vector<Vec3> positions(numBodies);
    std::vector<Vec3> localPositions(localCount);
    for (int i = 0; i < localCount; ++i) localPositions[i] = localBodies[i].pos;
    MPI_Allgatherv(localPositions.data(), localCount, vec3Type, positions.data(), counts.data(),
                   displacements.data(), vec3Type, MPI_COMM_WORLD);

    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        computeForces(localBodies, positions);
        integrateBodies(localBodies);
        for (int i = 0; i < localCount; ++i) localPositions[i] = localBodies[i].pos;
        MPI_Allgatherv(localPositions.data(), localCount, vec3Type, positions.data(), counts.data(),
                       displacements.data(), vec3Type, MPI_COMM_WORLD);
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) std::printf("Simulation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));

    if (printResults) {
        std::vector<Body> finalBodies;
        if (rank == 0) finalBodies.resize(numBodies);
        MPI_Gatherv(localBodies.data(), localCount, bodyType, rank == 0 ? finalBodies.data() : nullptr,
                    counts.data(), displacements.data(), bodyType, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<double> bodyData;
            bodyData.reserve(static_cast<size_t>(numBodies) * 6);
            for (const Body& body : finalBodies) {
                bodyData.insert(bodyData.end(), {body.pos.x, body.pos.y, body.pos.z,
                                                  body.vel.x, body.vel.y, body.vel.z});
            }
            print_results(bodyData, "Bodies");
        }
    }

    if (validate) {
        const int localValid = validateLocalSimulation(localBodies) ? 1 : 0;
        int globallyValid = 0;
        MPI_Reduce(&localValid, &globallyValid, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
        const double localEnergy = computeLocalEnergy(localBodies, positions, localOffset);
        double finalEnergy = 0.0;
        MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::printf("Validating simulation results...\n");
            if (globallyValid) {
                std::printf("Final energy: %.6f\n", finalEnergy);
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
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
