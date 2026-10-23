#include <chrono>
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

void randomizeBodies(std::vector<Body>& bodies, size_t start, size_t count, size_t n, unsigned int seed = 42) {
    for (size_t b = 0; b < n; ++b) {
        const double px = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double py = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double pz = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double vx = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double vy = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double vz = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        if (b >= start && b < start + count) {
            Body& body = bodies[b - start];
            body.pos = Vec3(px, py, pz);
            body.vel = Vec3(vx, vy, vz);
        }
    }
}

void computeForces(std::vector<Body>& bodies, const std::vector<double>& positions, size_t n) {
    const double* __restrict__ pos = positions.data();
    const size_t localCount = bodies.size();

    for (size_t i = 0; i < localCount; ++i) {
        const double xi = bodies[i].pos.x;
        const double yi = bodies[i].pos.y;
        const double zi = bodies[i].pos.z;
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (size_t j = 0; j < n; ++j) {
            const size_t k = 3 * j;
            const double dx = pos[k] - xi;
            const double dy = pos[k + 1] - yi;
            const double dz = pos[k + 2] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        bodies[i].vel.x += DT * Fx;
        bodies[i].vel.y += DT * Fy;
        bodies[i].vel.z += DT * Fz;
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

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
        numBodies = 0;
    }

    const size_t n = static_cast<size_t>(numBodies);

    // Contiguous block distribution of bodies over ranks.
    std::vector<int> counts(numRanks);
    std::vector<int> displs(numRanks);
    std::vector<int> posCounts(numRanks);
    std::vector<int> posDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        const size_t start = n * static_cast<size_t>(r) / static_cast<size_t>(numRanks);
        const size_t end = n * static_cast<size_t>(r + 1) / static_cast<size_t>(numRanks);
        counts[r] = static_cast<int>(end - start);
        displs[r] = static_cast<int>(start);
        posCounts[r] = 3 * counts[r];
        posDispls[r] = 3 * displs[r];
    }

    const size_t localStart = static_cast<size_t>(displs[rank]);
    const size_t localCount = static_cast<size_t>(counts[rank]);

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Initialize bodies (each rank deterministically reproduces its own block).
    std::vector<Body> bodies(localCount);
    randomizeBodies(bodies, localStart, localCount, n);

    // Full position buffer, replicated on every rank so the all-pairs force
    // loop can read every body's position without further communication.
    std::vector<double> positions(3 * n);
    for (size_t i = 0; i < localCount; ++i) {
        const size_t k = 3 * (localStart + i);
        positions[k] = bodies[i].pos.x;
        positions[k + 1] = bodies[i].pos.y;
        positions[k + 2] = bodies[i].pos.z;
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE,
                   positions.data(), posCounts.data(), posDispls.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, positions, n);
        integrateBodies(bodies);

        if (step + 1 < numSteps) {
            for (size_t i = 0; i < localCount; ++i) {
                const size_t k = 3 * (localStart + i);
                positions[k] = bodies[i].pos.x;
                positions[k + 1] = bodies[i].pos.y;
                positions[k + 2] = bodies[i].pos.z;
            }
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE,
                           positions.data(), posCounts.data(), posDispls.data(), MPI_DOUBLE,
                           MPI_COMM_WORLD);
        }
    }

    const double elapsed = MPI_Wtime() - startTime;

    // Collect the complete final state on rank 0 for output/validation.
    int localReturnCode = 0;
    if (printResults || validate) {
        std::vector<Body> allBodies;
        if (rank == 0) {
            allBodies.resize(n);
        }

        std::vector<int> bodyCounts(numRanks);
        std::vector<int> bodyDispls(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            bodyCounts[r] = counts[r] * static_cast<int>(sizeof(Body));
            bodyDispls[r] = displs[r] * static_cast<int>(sizeof(Body));
        }

        MPI_Gatherv(bodies.data(), static_cast<int>(localCount * sizeof(Body)), MPI_BYTE,
                    rank == 0 ? allBodies.data() : nullptr,
                    bodyCounts.data(), bodyDispls.data(), MPI_BYTE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Simulation time: %ld ms\n",
                   static_cast<long>(elapsed * 1000.0));

            if (printResults) {
                std::vector<double> bodyData;
                bodyData.reserve(n * 6);
                for (const auto& body : allBodies) {
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

                if (validateSimulation(allBodies)) {
                    double finalEnergy = computeTotalEnergy(allBodies);
                    printf("Final energy: %.6f\n", finalEnergy);
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    localReturnCode = 1;
                }
            }
        }

        MPI_Bcast(&localReturnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        if (rank == 0) {
            printf("Simulation time: %ld ms\n",
                   static_cast<long>(elapsed * 1000.0));
        }
    }

    MPI_Finalize();
    return localReturnCode;
}
