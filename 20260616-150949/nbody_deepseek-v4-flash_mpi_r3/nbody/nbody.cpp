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

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int exitFlag = 0;

    if (rank == 0) {
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
                exitFlag = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitFlag = 2;
            }
        }
    }

    MPI_Bcast(&exitFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exitFlag) {
        MPI_Finalize();
        return exitFlag == 1 ? 0 : 1;
    }

    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int validateInt = validate ? 1 : 0;
    int printResultsInt = printResults ? 1 : 0;
    MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResultsInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = validateInt != 0;
    printResults = printResultsInt != 0;

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", nprocs);
    }

    // Distribute bodies across ranks
    int nLocal = numBodies / nprocs;
    int remainder = numBodies % nprocs;
    std::vector<int> counts(nprocs), displs(nprocs);
    for (int p = 0, offset = 0; p < nprocs; ++p) {
        counts[p] = nLocal + (p < remainder ? 1 : 0);
        displs[p] = offset;
        offset += counts[p];
    }
    nLocal = counts[rank];

    // Initialize all bodies on rank 0, then scatter
    std::vector<Body> allBodies;
    std::vector<double> scatterBuf;
    if (rank == 0) {
        allBodies.resize(numBodies);
        randomizeBodies(allBodies);

        scatterBuf.resize(numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            scatterBuf[i * 6 + 0] = allBodies[i].pos.x;
            scatterBuf[i * 6 + 1] = allBodies[i].pos.y;
            scatterBuf[i * 6 + 2] = allBodies[i].pos.z;
            scatterBuf[i * 6 + 3] = allBodies[i].vel.x;
            scatterBuf[i * 6 + 4] = allBodies[i].vel.y;
            scatterBuf[i * 6 + 5] = allBodies[i].vel.z;
        }
    }

    std::vector<double> localBuf(nLocal * 6);
    std::vector<int> scatterCounts(nprocs), scatterDispls(nprocs);
    for (int p = 0; p < nprocs; ++p) {
        scatterCounts[p] = counts[p] * 6;
        scatterDispls[p] = displs[p] * 6;
    }
    MPI_Scatterv(rank == 0 ? scatterBuf.data() : nullptr,
                 scatterCounts.data(), scatterDispls.data(), MPI_DOUBLE,
                 localBuf.data(), nLocal * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Unpack local bodies
    std::vector<Body> localBodies(nLocal);
    for (int i = 0; i < nLocal; ++i) {
        localBodies[i].pos.x = localBuf[i * 6 + 0];
        localBodies[i].pos.y = localBuf[i * 6 + 1];
        localBodies[i].pos.z = localBuf[i * 6 + 2];
        localBodies[i].vel.x = localBuf[i * 6 + 3];
        localBodies[i].vel.y = localBuf[i * 6 + 4];
        localBodies[i].vel.z = localBuf[i * 6 + 5];
    }

    // Allgather parameters for position data (3 doubles per body)
    std::vector<double> allPos(3 * numBodies);
    std::vector<double> localPos(3 * nLocal);
    std::vector<int> posCounts(nprocs), posDispls(nprocs);
    for (int p = 0; p < nprocs; ++p) {
        posCounts[p] = counts[p] * 3;
        posDispls[p] = displs[p] * 3;
    }

    // Main simulation loop
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Pack local positions
        for (int i = 0; i < nLocal; ++i) {
            localPos[i * 3 + 0] = localBodies[i].pos.x;
            localPos[i * 3 + 1] = localBodies[i].pos.y;
            localPos[i * 3 + 2] = localBodies[i].pos.z;
        }

        // Allgather all positions so each rank can compute forces
        MPI_Allgatherv(localPos.data(), nLocal * 3, MPI_DOUBLE,
                       allPos.data(), posCounts.data(), posDispls.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // Compute forces for locally-owned bodies
        for (int li = 0; li < nLocal; ++li) {
            double Fx = 0.0, Fy = 0.0, Fz = 0.0;
            const double px = localBodies[li].pos.x;
            const double py = localBodies[li].pos.y;
            const double pz = localBodies[li].pos.z;

            for (int gj = 0; gj < numBodies; ++gj) {
                const double dx = allPos[gj * 3 + 0] - px;
                const double dy = allPos[gj * 3 + 1] - py;
                const double dz = allPos[gj * 3 + 2] - pz;
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

        // Integrate local bodies
        for (int li = 0; li < nLocal; ++li) {
            localBodies[li].pos.x += localBodies[li].vel.x * DT;
            localBodies[li].pos.y += localBodies[li].vel.y * DT;
            localBodies[li].pos.z += localBodies[li].vel.z * DT;
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

    // Repack local bodies for gathering to rank 0
    for (int i = 0; i < nLocal; ++i) {
        localBuf[i * 6 + 0] = localBodies[i].pos.x;
        localBuf[i * 6 + 1] = localBodies[i].pos.y;
        localBuf[i * 6 + 2] = localBodies[i].pos.z;
        localBuf[i * 6 + 3] = localBodies[i].vel.x;
        localBuf[i * 6 + 4] = localBodies[i].vel.y;
        localBuf[i * 6 + 5] = localBodies[i].vel.z;
    }

    // Gather final results to rank 0
    std::vector<double> finalBuf;
    if (rank == 0) finalBuf.resize(numBodies * 6);

    MPI_Gatherv(localBuf.data(), nLocal * 6, MPI_DOUBLE,
                rank == 0 ? finalBuf.data() : nullptr,
                scatterCounts.data(), scatterDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Reconstruct all bodies on rank 0
    if (rank == 0) {
        for (int i = 0; i < numBodies; ++i) {
            allBodies[i].pos.x = finalBuf[i * 6 + 0];
            allBodies[i].pos.y = finalBuf[i * 6 + 1];
            allBodies[i].pos.z = finalBuf[i * 6 + 2];
            allBodies[i].vel.x = finalBuf[i * 6 + 3];
            allBodies[i].vel.y = finalBuf[i * 6 + 4];
            allBodies[i].vel.z = finalBuf[i * 6 + 5];
        }
    }

    // Print results for external validation
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
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");

        if (validateSimulation(allBodies)) {
            double finalEnergy = computeTotalEnergy(allBodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    MPI_Finalize();
    return 0;
}
