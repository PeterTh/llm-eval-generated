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

void packBodies(const std::vector<Body>& bodies, std::vector<double>& data) {
    data.resize(bodies.size() * 6);
    for (size_t i = 0; i < bodies.size(); ++i) {
        const size_t offset = i * 6;
        data[offset] = bodies[i].pos.x;
        data[offset + 1] = bodies[i].pos.y;
        data[offset + 2] = bodies[i].pos.z;
        data[offset + 3] = bodies[i].vel.x;
        data[offset + 4] = bodies[i].vel.y;
        data[offset + 5] = bodies[i].vel.z;
    }
}

void unpackBodies(const std::vector<double>& data, std::vector<Body>& bodies) {
    const size_t count = data.size() / 6;
    bodies.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const size_t offset = i * 6;
        bodies[i].pos.x = data[offset];
        bodies[i].pos.y = data[offset + 1];
        bodies[i].pos.z = data[offset + 2];
        bodies[i].vel.x = data[offset + 3];
        bodies[i].vel.y = data[offset + 4];
        bodies[i].vel.z = data[offset + 5];
    }
}

void computeForces(std::vector<Body>& localBodies, const std::vector<double>& positions, const size_t globalStart, const size_t numBodies) {
    for (size_t i = 0; i < localBodies.size(); ++i) {
        const size_t globalIndex = globalStart + i;
        const size_t baseIndex = globalIndex * 3;
        const double ix = positions[baseIndex];
        const double iy = positions[baseIndex + 1];
        const double iz = positions[baseIndex + 2];
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (size_t j = 0; j < numBodies; ++j) {
            const size_t jIndex = j * 3;
            const double dx = positions[jIndex] - ix;
            const double dy = positions[jIndex + 1] - iy;
            const double dz = positions[jIndex + 2] - iz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        localBodies[i].vel.x += DT * Fx;
        localBodies[i].vel.y += DT * Fy;
        localBodies[i].vel.z += DT * Fz;
    }
}

void integrateBodies(std::vector<Body>& bodies) {
    for (auto& body : bodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
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

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            printf("Validation failed: body velocity exceeds reasonable bounds\n");
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

    int worldSize = 0;
    int worldRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int exitCode = -1;

    // Parse command line arguments on rank 0
    if (worldRank == 0) {
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
                printUsage(argv[0]);
                exitCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitCode = 1;
                break;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exitCode != -1) {
        MPI_Finalize();
        return exitCode;
    }

    int settings[4];
    if (worldRank == 0) {
        settings[0] = numBodies;
        settings[1] = numSteps;
        settings[2] = validate ? 1 : 0;
        settings[3] = printResults ? 1 : 0;
    }
    MPI_Bcast(settings, 4, MPI_INT, 0, MPI_COMM_WORLD);
    if (worldRank != 0) {
        numBodies = settings[0];
        numSteps = settings[1];
        validate = settings[2] != 0;
        printResults = settings[3] != 0;
    }

    if (worldRank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int baseBodies = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    std::vector<int> counts(worldSize);
    std::vector<int> displs(worldSize);
    int offset = 0;
    for (int rank = 0; rank < worldSize; ++rank) {
        counts[rank] = baseBodies + (rank < remainder ? 1 : 0);
        displs[rank] = offset;
        offset += counts[rank];
    }

    const int localCount = counts[worldRank];
    const size_t globalStart = static_cast<size_t>(displs[worldRank]);

    std::vector<int> counts3(worldSize);
    std::vector<int> displs3(worldSize);
    std::vector<int> counts6(worldSize);
    std::vector<int> displs6(worldSize);
    for (int rank = 0; rank < worldSize; ++rank) {
        counts3[rank] = counts[rank] * 3;
        displs3[rank] = displs[rank] * 3;
        counts6[rank] = counts[rank] * 6;
        displs6[rank] = displs[rank] * 6;
    }

    std::vector<double> allData;
    if (worldRank == 0) {
        std::vector<Body> bodies(numBodies);
        randomizeBodies(bodies);
        packBodies(bodies, allData);
    }

    std::vector<double> localData(static_cast<size_t>(counts6[worldRank]));
    MPI_Scatterv(worldRank == 0 ? allData.data() : nullptr, counts6.data(), displs6.data(), MPI_DOUBLE,
                 localData.empty() ? nullptr : localData.data(), counts6[worldRank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    std::vector<Body> localBodies;
    unpackBodies(localData, localBodies);

    std::vector<double> allPositions(static_cast<size_t>(numBodies) * 3);
    std::vector<double> localPositions(static_cast<size_t>(counts3[worldRank]));

    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        for (int i = 0; i < localCount; ++i) {
            const size_t baseIndex = static_cast<size_t>(i) * 3;
            localPositions[baseIndex] = localBodies[i].pos.x;
            localPositions[baseIndex + 1] = localBodies[i].pos.y;
            localPositions[baseIndex + 2] = localBodies[i].pos.z;
        }

        MPI_Allgatherv(localPositions.empty() ? nullptr : localPositions.data(), counts3[worldRank], MPI_DOUBLE,
                       allPositions.empty() ? nullptr : allPositions.data(), counts3.data(), displs3.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        computeForces(localBodies, allPositions, globalStart, static_cast<size_t>(numBodies));
        integrateBodies(localBodies);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - startTime;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        const long timeMs = static_cast<long>(maxElapsed * 1000.0);
        printf("Simulation time: %ld ms\n", timeMs);
    }

    int finalExit = 0;
    if (printResults || validate) {
        for (int i = 0; i < localCount; ++i) {
            const size_t offsetIndex = static_cast<size_t>(i) * 6;
            localData[offsetIndex] = localBodies[i].pos.x;
            localData[offsetIndex + 1] = localBodies[i].pos.y;
            localData[offsetIndex + 2] = localBodies[i].pos.z;
            localData[offsetIndex + 3] = localBodies[i].vel.x;
            localData[offsetIndex + 4] = localBodies[i].vel.y;
            localData[offsetIndex + 5] = localBodies[i].vel.z;
        }

        std::vector<double> gatheredData;
        if (worldRank == 0) {
            gatheredData.resize(static_cast<size_t>(numBodies) * 6);
        }

        MPI_Gatherv(localData.empty() ? nullptr : localData.data(), counts6[worldRank], MPI_DOUBLE,
                    worldRank == 0 ? gatheredData.data() : nullptr, counts6.data(), displs6.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (worldRank == 0) {
            std::vector<Body> bodies;
            unpackBodies(gatheredData, bodies);

            if (printResults) {
                std::vector<double> bodyData;
                bodyData.reserve(static_cast<size_t>(numBodies) * 6);
                for (const auto& body : bodies) {
                    bodyData.push_back(body.pos.x);
                    bodyData.push_back(body.pos.y);
                    bodyData.push_back(body.pos.z);
                    bodyData.push_back(body.vel.x);
                    bodyData.push_back(body.vel.y);
                    bodyData.push_back(body.vel.z);
                }
                print_results(bodyData, "Bodies");
            }

            if (validate) {
                printf("Validating simulation results...\n");

                if (validateSimulation(bodies)) {
                    double finalEnergy = computeTotalEnergy(bodies);
                    printf("Final energy: %.6f\n", finalEnergy);
                    printf("Validation: PASSED\n");
                    finalExit = 0;
                } else {
                    printf("Validation: FAILED\n");
                    finalExit = 1;
                }
            }
        }
    }

    MPI_Bcast(&finalExit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return finalExit;
}
