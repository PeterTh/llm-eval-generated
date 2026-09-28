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

// Computes forces for the local slice of bodies [offset, offset + localBodies.size())
// against the full, globally up-to-date position array `allPos`. Looping j over the
// entire [0, n) range in order keeps the floating-point summation identical to the
// original single-process algorithm regardless of how bodies are distributed across ranks.
void computeForces(std::vector<Body>& localBodies, const std::vector<Vec3>& allPos, size_t offset) {
    const size_t n = allPos.size();
    const size_t localN = localBodies.size();

    for (size_t li = 0; li < localN; ++li) {
        const size_t i = offset + li;
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (size_t j = 0; j < n; ++j) {
            const double dx = allPos[j].x - allPos[i].x;
            const double dy = allPos[j].y - allPos[i].y;
            const double dz = allPos[j].z - allPos[i].z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        localBodies[li].vel.x += DT * Fx;
        localBodies[li].vel.y += DT * Fy;
        localBodies[li].vel.z += DT * Fz;
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

    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical argv on every rank under mpirun)
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

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", worldSize);
    }

    // Every rank generates the identical, full initial condition deterministically
    // from the same seed so results do not depend on the number of ranks.
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Distribute bodies as evenly as possible across ranks (contiguous blocks).
    std::vector<int> counts(worldSize), displs(worldSize);
    {
        const int base = numBodies / worldSize;
        const int rem = numBodies % worldSize;
        int off = 0;
        for (int r = 0; r < worldSize; ++r) {
            counts[r] = base + (r < rem ? 1 : 0);
            displs[r] = off;
            off += counts[r];
        }
    }
    const size_t localOffset = static_cast<size_t>(displs[rank]);
    const size_t localCount = static_cast<size_t>(counts[rank]);

    std::vector<Body> localBodies(localCount);
    for (size_t i = 0; i < localCount; ++i) {
        localBodies[i] = bodies[localOffset + i];
    }

    // Full, globally shared position array kept in sync across ranks each step.
    std::vector<Vec3> allPos(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        allPos[i] = bodies[i].pos;
    }

    std::vector<int> countsD(worldSize), displsD(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        countsD[r] = counts[r] * 3;
        displsD[r] = displs[r] * 3;
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(localBodies, allPos, localOffset);
        integrateBodies(localBodies);

        // Extract updated local positions into a contiguous buffer and share
        // the refreshed global position array with every rank.
        std::vector<Vec3> localPos(localCount);
        for (size_t i = 0; i < localCount; ++i) {
            localPos[i] = localBodies[i].pos;
        }
        MPI_Allgatherv(localPos.data(), static_cast<int>(localCount * 3), MPI_DOUBLE,
                       allPos.data(), countsD.data(), displsD.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }

    double end = MPI_Wtime();
    double elapsed = end - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
    }

    // Gather the full, final body state (positions and velocities) back to rank 0.
    std::vector<int> counts6(worldSize), displs6(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        counts6[r] = counts[r] * 6;
        displs6[r] = displs[r] * 6;
    }
    std::vector<Body> finalBodies(rank == 0 ? numBodies : 0);
    MPI_Gatherv(localBodies.data(), static_cast<int>(localCount * 6), MPI_DOUBLE,
                finalBodies.data(), counts6.data(), displs6.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int result = 0;

    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.reserve(numBodies * 6);
            for (const auto& body : finalBodies) {
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

            if (validateSimulation(finalBodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(finalBodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
                result = 0;
            } else {
                printf("Validation: FAILED\n");
                result = 1;
            }
        }
    }

    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return result;
}
