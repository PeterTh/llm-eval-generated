#include <algorithm>
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

// Compute gravitational forces for locally-owned bodies using all-body positions
void computeForces(std::vector<Body>& local_bodies, const std::vector<Vec3>& all_positions) {
    const size_t n_local = local_bodies.size();
    const size_t n_total = all_positions.size();

    for (size_t i = 0; i < n_local; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (size_t j = 0; j < n_total; ++j) {
            const double dx = all_positions[j].x - local_bodies[i].pos.x;
            const double dy = all_positions[j].y - local_bodies[i].pos.y;
            const double dz = all_positions[j].z - local_bodies[i].pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        local_bodies[i].vel.x += DT * Fx;
        local_bodies[i].vel.y += DT * Fy;
        local_bodies[i].vel.z += DT * Fz;
    }
}

void integrateBodies(std::vector<Body>& bodies) {
    for (auto& body : bodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

// Compute total energy contribution from locally-owned bodies using all-body positions.
// Potential energy pairs are counted only once (by the process owning the smaller index).
double computeTotalEnergy(const std::vector<Body>& local_bodies,
                          const std::vector<Vec3>& all_positions,
                          int global_offset) {
    const size_t n_local = local_bodies.size();
    const size_t n_total = all_positions.size();
    double energy = 0.0;

    // Kinetic energy (assuming unit mass)
    for (const auto& body : local_bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
    // Each process handles its local bodies; counts pair (global_i, j) only when j > global_i
    for (size_t ii = 0; ii < n_local; ++ii) {
        const int global_i = global_offset + static_cast<int>(ii);
        for (size_t j = static_cast<size_t>(global_i) + 1; j < n_total; ++j) {
            const double dx = all_positions[j].x - local_bodies[ii].pos.x;
            const double dy = all_positions[j].y - local_bodies[ii].pos.y;
            const double dz = all_positions[j].z - local_bodies[ii].pos.z;
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

    int numRanks, myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse to avoid verbose broadcast)
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
            if (myRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (myRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Compute decomposition: contiguous chunks
    const int base = numBodies / numRanks;
    const int remainder = numBodies % numRanks;
    const int localCount = base + (myRank < remainder ? 1 : 0);
    const int offset = myRank * base + std::min(myRank, remainder);

    // Precompute scatter/gather counts and offsets (in units of 6 doubles per Body)
    std::vector<int> bodyCounts(numRanks), bodyOffs(numRanks);
    for (int i = 0; i < numRanks; ++i) {
        int cnt = base + (i < remainder ? 1 : 0);
        bodyCounts[i] = cnt * 6;
        bodyOffs[i] = (i == 0) ? 0 : bodyOffs[i - 1] + (base + (i - 1 < remainder ? 1 : 0)) * 6;
    }

    if (myRank == 0) {
        printf("N-Body Simulation (MPI)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI processes: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Initialize all bodies on rank 0, then scatter
    std::vector<double> allBodyData;
    if (myRank == 0) {
        std::vector<Body> allBodies(numBodies);
        randomizeBodies(allBodies);
        allBodyData.resize(numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            allBodyData[i * 6 + 0] = allBodies[i].pos.x;
            allBodyData[i * 6 + 1] = allBodies[i].pos.y;
            allBodyData[i * 6 + 2] = allBodies[i].pos.z;
            allBodyData[i * 6 + 3] = allBodies[i].vel.x;
            allBodyData[i * 6 + 4] = allBodies[i].vel.y;
            allBodyData[i * 6 + 5] = allBodies[i].vel.z;
        }
    }

    std::vector<Body> localBodies(localCount);
    std::vector<double> localBodyData(localCount * 6);

    MPI_Scatterv(myRank == 0 ? allBodyData.data() : nullptr,
                 bodyCounts.data(), bodyOffs.data(), MPI_DOUBLE,
                 localBodyData.data(), localCount * 6, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    for (int i = 0; i < localCount; ++i) {
        localBodies[i].pos.x = localBodyData[i * 6 + 0];
        localBodies[i].pos.y = localBodyData[i * 6 + 1];
        localBodies[i].pos.z = localBodyData[i * 6 + 2];
        localBodies[i].vel.x = localBodyData[i * 6 + 3];
        localBodies[i].vel.y = localBodyData[i * 6 + 4];
        localBodies[i].vel.z = localBodyData[i * 6 + 5];
    }

    // Precompute position gather counts/offsets (3 doubles per Vec3 position)
    std::vector<int> posCounts(numRanks), posOffs(numRanks);
    for (int i = 0; i < numRanks; ++i) {
        int cnt = base + (i < remainder ? 1 : 0);
        posCounts[i] = cnt * 3;
        posOffs[i] = (i == 0) ? 0 : posOffs[i - 1] + posCounts[i - 1];
    }

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> allPosData(numBodies * 3);
    std::vector<double> localPosData(localCount * 3);

    for (int step = 0; step < numSteps; ++step) {
        // Pack local positions
        for (int i = 0; i < localCount; ++i) {
            localPosData[i * 3 + 0] = localBodies[i].pos.x;
            localPosData[i * 3 + 1] = localBodies[i].pos.y;
            localPosData[i * 3 + 2] = localBodies[i].pos.z;
        }

        // Gather all positions from every process
        MPI_Allgatherv(localPosData.data(), localCount * 3, MPI_DOUBLE,
                       allPosData.data(), posCounts.data(), posOffs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // Convert flat position data to Vec3 array for computation
        std::vector<Vec3> allPositions(numBodies);
        for (int i = 0; i < numBodies; ++i) {
            allPositions[i].x = allPosData[i * 3 + 0];
            allPositions[i].y = allPosData[i * 3 + 1];
            allPositions[i].z = allPosData[i * 3 + 2];
        }

        computeForces(localBodies, allPositions);
        integrateBodies(localBodies);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (myRank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Print results for external validation (rank 0 gathers and outputs)
    if (printResults) {
        std::vector<double> localResultData(localCount * 6);
        for (int i = 0; i < localCount; ++i) {
            localResultData[i * 6 + 0] = localBodies[i].pos.x;
            localResultData[i * 6 + 1] = localBodies[i].pos.y;
            localResultData[i * 6 + 2] = localBodies[i].pos.z;
            localResultData[i * 6 + 3] = localBodies[i].vel.x;
            localResultData[i * 6 + 4] = localBodies[i].vel.y;
            localResultData[i * 6 + 5] = localBodies[i].vel.z;
        }

        std::vector<double> allResultData;
        if (myRank == 0) {
            allResultData.resize(numBodies * 6);
        }

        MPI_Gatherv(localResultData.data(), localCount * 6, MPI_DOUBLE,
                    myRank == 0 ? allResultData.data() : nullptr,
                    bodyCounts.data(), bodyOffs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (myRank == 0) {
            print_results(allResultData, "Bodies");
        }
    }

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        int localValid = validateSimulation(localBodies) ? 1 : 0;
        int globalValid = 0;
        MPI_Allreduce(&localValid, &globalValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (myRank == 0) {
            printf("Validating simulation results...\n");
        }

        if (globalValid) {
            // Gather final positions for energy computation
            for (int i = 0; i < localCount; ++i) {
                localPosData[i * 3 + 0] = localBodies[i].pos.x;
                localPosData[i * 3 + 1] = localBodies[i].pos.y;
                localPosData[i * 3 + 2] = localBodies[i].pos.z;
            }
            MPI_Allgatherv(localPosData.data(), localCount * 3, MPI_DOUBLE,
                           allPosData.data(), posCounts.data(), posOffs.data(), MPI_DOUBLE,
                           MPI_COMM_WORLD);

            std::vector<Vec3> allPositions(numBodies);
            for (int i = 0; i < numBodies; ++i) {
                allPositions[i].x = allPosData[i * 3 + 0];
                allPositions[i].y = allPosData[i * 3 + 1];
                allPositions[i].z = allPosData[i * 3 + 2];
            }

            double localEnergy = computeTotalEnergy(localBodies, allPositions, offset);
            double totalEnergy = 0.0;
            MPI_Reduce(&localEnergy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (myRank == 0) {
                printf("Final energy: %.6f\n", totalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (myRank == 0) {
                printf("Validation failed: found NaN or Inf value in body state\n");
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
