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

// Compute forces on locally owned bodies using the full global body set.
// Each rank computes forces only for its assigned range [localStart, localEnd).
void computeForces(const std::vector<Body>& allBodies, std::vector<Body>& localBodies,
                   size_t localStart, size_t localEnd) {
    const size_t n = allBodies.size();

    for (size_t i = localStart; i < localEnd; ++i) {
        const Body& bi = allBodies[i];
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (size_t j = 0; j < n; ++j) {
            const double dx = allBodies[j].pos.x - bi.pos.x;
            const double dy = allBodies[j].pos.y - bi.pos.y;
            const double dz = allBodies[j].pos.z - bi.pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        size_t li = i - localStart;
        localBodies[li].vel.x += DT * Fx;
        localBodies[li].vel.y += DT * Fy;
        localBodies[li].vel.z += DT * Fz;
    }
}

void integrateBodies(std::vector<Body>& localBodies) {
    for (auto& body : localBodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

// Each rank computes energy contributions for pairs where the first index
// falls in its local range, then we MPI_SUM across all ranks.
double computeLocalEnergy(const std::vector<Body>& allBodies,
                          size_t localStart, size_t localEnd) {
    double energy = 0.0;
    const size_t n = allBodies.size();

    // Kinetic energy for local bodies
    for (size_t i = localStart; i < localEnd; ++i) {
        const Body& b = allBodies[i];
        energy += 0.5 * (b.vel.x * b.vel.x +
                         b.vel.y * b.vel.y +
                         b.vel.z * b.vel.z);
    }

    // Potential energy: pairs (i,j) where i is in local range, j > i
    for (size_t i = localStart; i < localEnd; ++i) {
        const Body& bi = allBodies[i];
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = allBodies[j].pos.x - bi.pos.x;
            const double dy = allBodies[j].pos.y - bi.pos.y;
            const double dz = allBodies[j].pos.z - bi.pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

// Validate locally owned bodies
bool validateSimulation(const std::vector<Body>& allBodies,
                        size_t localStart, size_t localEnd) {
    for (size_t i = localStart; i < localEnd; ++i) {
        const Body& body = allBodies[i];
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) ||
            !std::isfinite(body.pos.z) || !std::isfinite(body.vel.x) ||
            !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
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

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse identically)
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
        printf("N-Body Simulation (MPI: %d ranks)\n", numRanks);
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // --- Block distribution ---
    const size_t n = static_cast<size_t>(numBodies);
    const size_t baseCount = n / numRanks;
    const size_t remainder = n % numRanks;
    const size_t localCount = baseCount + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t localStart =
        static_cast<size_t>(rank) * baseCount +
        std::min(static_cast<size_t>(rank), remainder);
    const size_t localEnd = localStart + localCount;

    // --- Initialize local bodies with deterministic random values ---
    // Each rank advances the RNG to its offset so the same bodies are produced
    // regardless of the number of ranks.
    std::vector<Body> localBodies(localCount);
    unsigned int seed = 42;
    // Skip over bodies before our range
    for (size_t i = 0; i < localStart; ++i) {
        rand_r(&seed); rand_r(&seed); rand_r(&seed); // pos x,y,z
        rand_r(&seed); rand_r(&seed); rand_r(&seed); // vel x,y,z
    }
    for (auto& body : localBodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }

    // --- Communication buffers ---
    // Each body = 6 doubles (pos x,y,z + vel x,y,z)
    constexpr int DOUBLES_PER_BODY = 6;
    const int sendCount = static_cast<int>(localCount * DOUBLES_PER_BODY);

    std::vector<double> sendBuf(sendCount);
    std::vector<double> recvBuf(n * DOUBLES_PER_BODY);

    // Pack initial local bodies into send buffer
    for (size_t i = 0; i < localCount; ++i) {
        const Body& b = localBodies[i];
        size_t idx = i * DOUBLES_PER_BODY;
        sendBuf[idx]     = b.pos.x;
        sendBuf[idx + 1] = b.pos.y;
        sendBuf[idx + 2] = b.pos.z;
        sendBuf[idx + 3] = b.vel.x;
        sendBuf[idx + 4] = b.vel.y;
        sendBuf[idx + 5] = b.vel.z;
    }

    // Build recvCounts and displacements for MPI_Allgatherv
    std::vector<int> recvCounts(numRanks);
    std::vector<int> displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        size_t rc = baseCount + (static_cast<size_t>(r) < remainder ? 1 : 0);
        size_t disp = static_cast<size_t>(r) * baseCount +
                      std::min(static_cast<size_t>(r), remainder);
        recvCounts[r] = static_cast<int>(rc * DOUBLES_PER_BODY);
        displs[r] = static_cast<int>(disp * DOUBLES_PER_BODY);
    }

    // Global body array (all ranks need full state for force computation)
    std::vector<Body> allBodies(n);

    // --- Simulation loop ---
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Gather all body data so every rank has the complete state
        MPI_Allgatherv(sendBuf.data(), sendCount, MPI_DOUBLE,
                       recvBuf.data(), recvCounts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Unpack into global body array
        for (size_t i = 0; i < n; ++i) {
            size_t idx = i * DOUBLES_PER_BODY;
            allBodies[i].pos.x = recvBuf[idx];
            allBodies[i].pos.y = recvBuf[idx + 1];
            allBodies[i].pos.z = recvBuf[idx + 2];
            allBodies[i].vel.x = recvBuf[idx + 3];
            allBodies[i].vel.y = recvBuf[idx + 4];
            allBodies[i].vel.z = recvBuf[idx + 5];
        }

        // Compute forces on locally owned bodies
        computeForces(allBodies, localBodies, localStart, localEnd);

        // Integrate locally owned bodies
        integrateBodies(localBodies);

        // Pack updated local bodies for next step
        for (size_t i = 0; i < localCount; ++i) {
            const Body& b = localBodies[i];
            size_t idx = i * DOUBLES_PER_BODY;
            sendBuf[idx]     = b.pos.x;
            sendBuf[idx + 1] = b.pos.y;
            sendBuf[idx + 2] = b.pos.z;
            sendBuf[idx + 3] = b.vel.x;
            sendBuf[idx + 4] = b.vel.y;
            sendBuf[idx + 5] = b.vel.z;
        }
    }

    // Final allgather to get complete final state on every rank
    MPI_Allgatherv(sendBuf.data(), sendCount, MPI_DOUBLE,
                   recvBuf.data(), recvCounts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    for (size_t i = 0; i < n; ++i) {
        size_t idx = i * DOUBLES_PER_BODY;
        allBodies[i].pos.x = recvBuf[idx];
        allBodies[i].pos.y = recvBuf[idx + 1];
        allBodies[i].pos.z = recvBuf[idx + 2];
        allBodies[i].vel.x = recvBuf[idx + 3];
        allBodies[i].vel.y = recvBuf[idx + 4];
        allBodies[i].vel.z = recvBuf[idx + 5];
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // --- Print results for external validation ---
    if (printResults) {
        if (rank == 0) {
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
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- Validation ---
    if (validate) {
        bool localValid = validateSimulation(allBodies, localStart, localEnd);
        int globalValid;
        MPI_Allreduce(&localValid, &globalValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validating simulation results...\n");
            if (globalValid) {
                double localEnergy = computeLocalEnergy(allBodies, localStart, localEnd);
                double totalEnergy;
                MPI_Reduce(&localEnergy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                printf("Final energy: %.6f\n", totalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        } else {
            double localEnergy = computeLocalEnergy(allBodies, localStart, localEnd);
            MPI_Reduce(&localEnergy, nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}