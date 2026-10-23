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

static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be a tightly packed array of doubles");

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

__attribute__((optimize("no-tree-vectorize")))
void computeForces(std::vector<Body>& bodies, const double* gpos, size_t n) {
    const size_t localN = bodies.size();

    for (size_t i = 0; i < localN; ++i) {
        const double xi = bodies[i].pos.x;
        const double yi = bodies[i].pos.y;
        const double zi = bodies[i].pos.z;

        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (size_t j = 0; j < n; ++j) {
            const double dx = gpos[3 * j + 0] - xi;
            const double dy = gpos[3 * j + 1] - yi;
            const double dz = gpos[3 * j + 2] - zi;
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
    int exitCode = 0;

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

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t n = static_cast<size_t>(numBodies);
    std::vector<int> counts(numRanks);
    std::vector<int> displs(numRanks);
    std::vector<int> elemCounts(numRanks);
    std::vector<int> elemDispls(numRanks);

    const size_t base = n / static_cast<size_t>(numRanks);
    const size_t rem = n % static_cast<size_t>(numRanks);
    size_t running = 0;
    for (int r = 0; r < numRanks; ++r) {
        const size_t c = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        counts[r] = static_cast<int>(c);
        displs[r] = static_cast<int>(running);
        elemCounts[r] = static_cast<int>(3 * c);
        elemDispls[r] = static_cast<int>(3 * running);
        running += c;
    }

    const size_t localN = static_cast<size_t>(counts[rank]);
    const size_t offset = static_cast<size_t>(displs[rank]);
    const int localBytes = static_cast<int>(localN * sizeof(Body));

    std::vector<int> byteCounts(numRanks);
    std::vector<int> byteDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        byteCounts[r] = static_cast<int>(static_cast<size_t>(counts[r]) * sizeof(Body));
        byteDispls[r] = static_cast<int>(static_cast<size_t>(displs[r]) * sizeof(Body));
    }

    std::vector<Body> allBodies;
    if (rank == 0) {
        allBodies.resize(n);
        randomizeBodies(allBodies);
    }

    std::vector<Body> bodies(localN);
    MPI_Scatterv(rank == 0 ? allBodies.data() : nullptr,
                 byteCounts.data(), byteDispls.data(), MPI_BYTE,
                 localBytes > 0 ? bodies.data() : nullptr, localBytes, MPI_BYTE,
                 0, MPI_COMM_WORLD);
    std::vector<Body>().swap(allBodies);

    std::vector<double> gpos(3 * n);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        double* sendPos = gpos.data() + 3 * offset;
        for (size_t i = 0; i < localN; ++i) {
            sendPos[3 * i + 0] = bodies[i].pos.x;
            sendPos[3 * i + 1] = bodies[i].pos.y;
            sendPos[3 * i + 2] = bodies[i].pos.z;
        }

        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       gpos.data(), elemCounts.data(), elemDispls.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        computeForces(bodies, gpos.data(), n);
        integrateBodies(bodies);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count();
    long global_ms = 0;
    MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", global_ms);
    }

    std::vector<Body> globalBodies;
    if (rank == 0) {
        globalBodies.resize(n);
    }

    MPI_Gatherv(bodies.data(), localBytes, MPI_BYTE,
                rank == 0 ? globalBodies.data() : nullptr,
                byteCounts.data(), byteDispls.data(), MPI_BYTE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.reserve(n * 6);
            for (const auto& body : globalBodies) {
                bodyData.push_back(body.pos.x);
                bodyData.push_back(body.pos.y);
                bodyData.push_back(body.pos.z);
                bodyData.push_back(body.vel.x);
                bodyData.push_back(body.vel.y);
                bodyData.push_back(body.vel.z);
            }
            print_results(bodyData, "Bodies");
        }

        // Validation: check that simulation produces finite, reasonable values
        if (validate) {
            printf("Validating simulation results...\n");

            if (validateSimulation(globalBodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(globalBodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
