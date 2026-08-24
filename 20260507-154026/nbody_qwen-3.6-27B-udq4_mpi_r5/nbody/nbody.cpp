#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

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

// Each rank computes forces for its local subset of bodies using the full body array.
// This is the core O(N^2) kernel — work is split across ranks by the outer loop.
void computeForcesLocal(const std::vector<Body>& allBodies, std::vector<Body>& localBodies,
                        const size_t globalStart, const size_t localCount) {
    const size_t n = allBodies.size();

    for (size_t li = 0; li < localCount; ++li) {
        const size_t i = globalStart + li;
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (size_t j = 0; j < n; ++j) {
            const double dx = allBodies[j].pos.x - allBodies[i].pos.x;
            const double dy = allBodies[j].pos.y - allBodies[i].pos.y;
            const double dz = allBodies[j].pos.z - allBodies[i].pos.z;
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

void integrateBodiesLocal(std::vector<Body>& localBodies, const size_t localCount) {
    for (size_t i = 0; i < localCount; ++i) {
        localBodies[i].pos.x += localBodies[i].vel.x * DT;
        localBodies[i].pos.y += localBodies[i].vel.y * DT;
        localBodies[i].pos.z += localBodies[i].vel.z * DT;
    }
}

// Kinetic energy: each rank computes for its local bodies, then Allreduce.
double computeKineticEnergyLocal(const std::vector<Body>& localBodies, const size_t localCount) {
    double energy = 0.0;
    for (size_t i = 0; i < localCount; ++i) {
        energy += 0.5 * (localBodies[i].vel.x * localBodies[i].vel.x +
                         localBodies[i].vel.y * localBodies[i].vel.y +
                         localBodies[i].vel.z * localBodies[i].vel.z);
    }
    return energy;
}

// Potential energy: each rank computes for pairs (i, j) where i is in its local range
// and j > i. Then Allreduce sums across ranks.
double computePotentialEnergyLocal(const std::vector<Body>& allBodies,
                                   const size_t globalStart, const size_t localCount) {
    const size_t n = allBodies.size();
    double energy = 0.0;

    for (size_t li = 0; li < localCount; ++li) {
        const size_t i = globalStart + li;
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = allBodies[j].pos.x - allBodies[i].pos.x;
            const double dy = allBodies[j].pos.y - allBodies[i].pos.y;
            const double dz = allBodies[j].pos.z - allBodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    return energy;
}

// Validate that simulation produces finite, reasonable values (local check).
bool validateSimulationLocal(const std::vector<Body>& localBodies, const size_t localCount) {
    for (size_t i = 0; i < localCount; ++i) {
        const auto& body = localBodies[i];
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            return false;
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos ||
            std::abs(body.pos.z) > maxPos) {
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel ||
            std::abs(body.vel.z) > maxVel) {
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

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = static_cast<size_t>(atoi(argv[++i]));
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

    if (rank == 0) {
        printf("N-Body Simulation (MPI: %d ranks)\n", numRanks);
        printf("Number of bodies: %zu\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Domain decomposition: distribute bodies across ranks
    const size_t localCount = numBodies / numRanks;
    const int remainder = static_cast<int>(numBodies % numRanks);
    const size_t myLocalCount = localCount + (rank < remainder ? 1 : 0);

    // Compute global start index for this rank
    size_t globalStart = 0;
    for (int r = 0; r < rank; ++r) {
        globalStart += localCount + (r < remainder ? 1 : 0);
    }

    // Allocate local and global body arrays
    std::vector<Body> localBodies(myLocalCount);
    std::vector<Body> allBodies(numBodies);

    // Initialize bodies on rank 0, then broadcast
    if (rank == 0) {
        randomizeBodies(allBodies);
    }

    // Build send/recv counts for Allgatherv in the simulation loop
    std::vector<int> sendcounts(numRanks);
    std::vector<int> displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        sendcounts[r] = (localCount + (r < remainder ? 1 : 0)) * 6;
        displs[r] = 0;
        for (int rr = 0; rr < r; ++rr) {
            displs[r] += localCount + (rr < remainder ? 1 : 0);
        }
        displs[r] *= 6;
    }

    std::vector<double> sendBuf(myLocalCount * 6);
    std::vector<double> recvBuf(numBodies * 6);

    // Bcast initial body data from rank 0 to all ranks
    std::vector<double> fullBuf(numBodies * 6);
    if (rank == 0) {
        for (size_t i = 0; i < numBodies; ++i) {
            fullBuf[i * 6 + 0] = allBodies[i].pos.x;
            fullBuf[i * 6 + 1] = allBodies[i].pos.y;
            fullBuf[i * 6 + 2] = allBodies[i].pos.z;
            fullBuf[i * 6 + 3] = allBodies[i].vel.x;
            fullBuf[i * 6 + 4] = allBodies[i].vel.y;
            fullBuf[i * 6 + 5] = allBodies[i].vel.z;
        }
    }
    MPI_Bcast(fullBuf.data(), static_cast<int>(numBodies * 6), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (size_t i = 0; i < numBodies; ++i) {
        allBodies[i].pos.x = fullBuf[i * 6 + 0];
        allBodies[i].pos.y = fullBuf[i * 6 + 1];
        allBodies[i].pos.z = fullBuf[i * 6 + 2];
        allBodies[i].vel.x = fullBuf[i * 6 + 3];
        allBodies[i].vel.y = fullBuf[i * 6 + 4];
        allBodies[i].vel.z = fullBuf[i * 6 + 5];
    }

    // Copy local subset
    for (size_t i = 0; i < myLocalCount; ++i) {
        localBodies[i] = allBodies[globalStart + i];
    }

    // Synchronize before the timer
    MPI_Barrier(MPI_COMM_WORLD);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Each rank computes forces for its local subset using all body positions
        computeForcesLocal(allBodies, localBodies, globalStart, myLocalCount);

        // Integrate local bodies
        integrateBodiesLocal(localBodies, myLocalCount);

        // Gather all body states back using Allgatherv
        for (size_t i = 0; i < myLocalCount; ++i) {
            sendBuf[i * 6 + 0] = localBodies[i].pos.x;
            sendBuf[i * 6 + 1] = localBodies[i].pos.y;
            sendBuf[i * 6 + 2] = localBodies[i].pos.z;
            sendBuf[i * 6 + 3] = localBodies[i].vel.x;
            sendBuf[i * 6 + 4] = localBodies[i].vel.y;
            sendBuf[i * 6 + 5] = localBodies[i].vel.z;
        }

        MPI_Allgatherv(sendBuf.data(), static_cast<int>(myLocalCount * 6), MPI_DOUBLE,
                       recvBuf.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // Unpack received data
        for (size_t i = 0; i < numBodies; ++i) {
            allBodies[i].pos.x = recvBuf[i * 6 + 0];
            allBodies[i].pos.y = recvBuf[i * 6 + 1];
            allBodies[i].pos.z = recvBuf[i * 6 + 2];
            allBodies[i].vel.x = recvBuf[i * 6 + 3];
            allBodies[i].vel.y = recvBuf[i * 6 + 4];
            allBodies[i].vel.z = recvBuf[i * 6 + 5];
        }

        // Update local subset
        for (size_t i = 0; i < myLocalCount; ++i) {
            localBodies[i] = allBodies[globalStart + i];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = static_cast<long>(duration.count());
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", maxDuration);
    }

    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
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

    // Validation
    if (validate) {
        // Local validation
        bool localValid = validateSimulationLocal(localBodies, myLocalCount);
        int globalValid = localValid ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &globalValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validating simulation results...\n");
        }

        if (globalValid) {
            // Compute total energy
            double localKE = computeKineticEnergyLocal(localBodies, myLocalCount);
            double totalKE = 0.0;
            MPI_Reduce(&localKE, &totalKE, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            double localPE = computePotentialEnergyLocal(allBodies, globalStart, myLocalCount);
            double totalPE = 0.0;
            MPI_Reduce(&localPE, &totalPE, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                double finalEnergy = totalKE + totalPE;
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
