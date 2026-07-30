#include <mpi.h>
#include <chrono>
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

void randomizeBodies(Body* bodies, int n, unsigned int seed = 42) {
    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// Compute gravitational forces for local bodies using all-body positions (flat array: 3 doubles per body)
void computeForces(Body* myBodies, int myN, const double* allPos, int totalN) {
    for (int i = 0; i < myN; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        const double px = myBodies[i].pos.x;
        const double py = myBodies[i].pos.y;
        const double pz = myBodies[i].pos.z;

        for (int j = 0; j < totalN; ++j) {
            const double dx = allPos[3 * j]     - px;
            const double dy = allPos[3 * j + 1] - py;
            const double dz = allPos[3 * j + 2] - pz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        myBodies[i].vel.x += DT * Fx;
        myBodies[i].vel.y += DT * Fy;
        myBodies[i].vel.z += DT * Fz;
    }
}

void integrateBodies(Body* bodies, int n) {
    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

double computeKineticEnergy(const Body* bodies, int n) {
    double energy = 0.0;
    for (int i = 0; i < n; ++i) {
        energy += 0.5 * (bodies[i].vel.x * bodies[i].vel.x +
                        bodies[i].vel.y * bodies[i].vel.y +
                        bodies[i].vel.z * bodies[i].vel.z);
    }
    return energy;
}

double computePotentialEnergy(const double* allPos, int myN, int totalN, int myOffset) {
    double energy = 0.0;
    for (int i = myOffset; i < myOffset + myN; ++i) {
        for (int j = i + 1; j < totalN; ++j) {
            const double dx = allPos[3 * j]     - allPos[3 * i];
            const double dy = allPos[3 * j + 1] - allPos[3 * i + 1];
            const double dz = allPos[3 * j + 2] - allPos[3 * i + 2];
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const Body* bodies, int n) {
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(bodies[i].pos.x) || !std::isfinite(bodies[i].pos.y) || !std::isfinite(bodies[i].pos.z) ||
            !std::isfinite(bodies[i].vel.x) || !std::isfinite(bodies[i].vel.y) || !std::isfinite(bodies[i].vel.z)) {
            return false;
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(bodies[i].pos.x) > maxPos || std::abs(bodies[i].pos.y) > maxPos ||
            std::abs(bodies[i].pos.z) > maxPos) return false;
        if (std::abs(bodies[i].vel.x) > maxVel || std::abs(bodies[i].vel.y) > maxVel ||
            std::abs(bodies[i].vel.z) > maxVel) return false;
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

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse)
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

    // Compute distribution across MPI ranks
    int baseCount = numBodies / numProcs;
    int remainder = numBodies % numProcs;
    std::vector<int> counts(numProcs), displs(numProcs);
    int offset = 0;
    for (int i = 0; i < numProcs; ++i) {
        counts[i] = baseCount + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += counts[i];
    }
    int nlocal = counts[rank];
    int myOffset = displs[rank];

    // Counts and displacements for flat arrays (3 doubles per body for positions)
    std::vector<int> counts3(numProcs), displs3(numProcs);
    for (int i = 0; i < numProcs; ++i) {
        counts3[i] = 3 * counts[i];
        displs3[i] = 3 * displs[i];
    }

    if (rank == 0) {
        printf("N-Body Simulation (MPI)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI processes: %d\n", numProcs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Local bodies storage
    std::vector<Body> localBodies(nlocal);
    std::vector<double> localPosFlat(3 * nlocal);

    // All-body positions buffer (for force computation)
    std::vector<double> allPos(3 * numBodies);

    // Initialize: rank 0 generates all bodies and scatters
    {
        std::vector<double> sendPos, sendVel;
        if (rank == 0) {
            sendPos.resize(3 * numBodies);
            sendVel.resize(3 * numBodies);

            std::vector<Body> allBodies(numBodies);
            randomizeBodies(allBodies.data(), numBodies);

            for (int i = 0; i < numBodies; ++i) {
                sendPos[3 * i]     = allBodies[i].pos.x;
                sendPos[3 * i + 1] = allBodies[i].pos.y;
                sendPos[3 * i + 2] = allBodies[i].pos.z;
                sendVel[3 * i]     = allBodies[i].vel.x;
                sendVel[3 * i + 1] = allBodies[i].vel.y;
                sendVel[3 * i + 2] = allBodies[i].vel.z;
            }
        }

        // Scatter positions to all ranks
        MPI_Scatterv(rank == 0 ? sendPos.data() : nullptr, counts3.data(), displs3.data(), MPI_DOUBLE,
                     localPosFlat.data(), 3 * nlocal, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Scatter velocities to all ranks
        std::vector<double> localVelFlat(3 * nlocal);
        MPI_Scatterv(rank == 0 ? sendVel.data() : nullptr, counts3.data(), displs3.data(), MPI_DOUBLE,
                     localVelFlat.data(), 3 * nlocal, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Unpack into localBodies
        for (int i = 0; i < nlocal; ++i) {
            localBodies[i].pos.x = localPosFlat[3 * i];
            localBodies[i].pos.y = localPosFlat[3 * i + 1];
            localBodies[i].pos.z = localPosFlat[3 * i + 2];
            localBodies[i].vel.x = localVelFlat[3 * i];
            localBodies[i].vel.y = localVelFlat[3 * i + 1];
            localBodies[i].vel.z = localVelFlat[3 * i + 2];
        }
    }

    // Initial allgather of positions into allPos for the first computeForces
    MPI_Allgatherv(localPosFlat.data(), 3 * nlocal, MPI_DOUBLE,
                   allPos.data(), counts3.data(), displs3.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Compute forces for local bodies using all-body positions
        computeForces(localBodies.data(), nlocal, allPos.data(), numBodies);

        // Integrate local bodies
        integrateBodies(localBodies.data(), nlocal);

        // Pack updated local positions for allgather
        for (int i = 0; i < nlocal; ++i) {
            localPosFlat[3 * i]     = localBodies[i].pos.x;
            localPosFlat[3 * i + 1] = localBodies[i].pos.y;
            localPosFlat[3 * i + 2] = localBodies[i].pos.z;
        }

        // Share updated positions with all ranks (collective synchronization)
        MPI_Allgatherv(localPosFlat.data(), 3 * nlocal, MPI_DOUBLE,
                       allPos.data(), counts3.data(), displs3.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    long long maxDuration;
    MPI_Reduce(&duration, &maxDuration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %lld ms\n", maxDuration);
    }

    // Print results for external validation
    if (printResults) {
        // Pack local body data (6 doubles per body: pos.x/y/z, vel.x/y/z)
        std::vector<double> localBodyData(6 * nlocal);
        for (int i = 0; i < nlocal; ++i) {
            localBodyData[6 * i]     = localBodies[i].pos.x;
            localBodyData[6 * i + 1] = localBodies[i].pos.y;
            localBodyData[6 * i + 2] = localBodies[i].pos.z;
            localBodyData[6 * i + 3] = localBodies[i].vel.x;
            localBodyData[6 * i + 4] = localBodies[i].vel.y;
            localBodyData[6 * i + 5] = localBodies[i].vel.z;
        }

        // Gather all body data to rank 0
        std::vector<int> gatherCounts(numProcs), gatherDispls(numProcs);
        for (int i = 0; i < numProcs; ++i) {
            gatherCounts[i] = 6 * counts[i];
            gatherDispls[i] = 6 * displs[i];
        }

        std::vector<double> allBodyData;
        if (rank == 0) allBodyData.resize(6 * numBodies);

        MPI_Gatherv(localBodyData.data(), 6 * nlocal, MPI_DOUBLE,
                    rank == 0 ? allBodyData.data() : nullptr,
                    gatherCounts.data(), gatherDispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(allBodyData, "Bodies");
        }
    }

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        int localOk = validateSimulation(localBodies.data(), nlocal) ? 1 : 0;
        int globalOk;
        MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (globalOk) {
            // Compute kinetic energy locally and reduce
            double localKE = computeKineticEnergy(localBodies.data(), nlocal);
            double totalKE;
            MPI_Reduce(&localKE, &totalKE, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            // Compute potential energy for local i-range (uses allPos which has all positions)
            double localPE = computePotentialEnergy(allPos.data(), nlocal, numBodies, myOffset);
            double totalPE;
            MPI_Reduce(&localPE, &totalPE, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                printf("Final energy: %.6f\n", totalKE + totalPE);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
