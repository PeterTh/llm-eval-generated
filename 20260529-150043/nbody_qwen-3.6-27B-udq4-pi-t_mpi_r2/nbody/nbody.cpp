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

// Compute forces for bodies in [localStart, localEnd) against ALL bodies.
// Each rank computes forces for its assigned subset using the full body list,
// preserving the exact same iteration order as the sequential version.
void computeForcesLocal(const std::vector<Body>& allBodies, std::vector<Body>& outBodies,
                        size_t localStart, size_t localEnd) {
    const size_t n = allBodies.size();

    for (size_t i = localStart; i < localEnd; ++i) {
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

        outBodies[i].vel.x += DT * Fx;
        outBodies[i].vel.y += DT * Fy;
        outBodies[i].vel.z += DT * Fz;
    }
}

// Integrate positions for bodies in [localStart, localEnd)
void integrateBodiesLocal(std::vector<Body>& bodies, size_t localStart, size_t localEnd) {
    for (size_t i = localStart; i < localEnd; ++i) {
        auto& body = bodies[i];
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

// Compute energy contributions for this rank's partition.
// Kinetic energy: each rank computes for its local bodies.
// Potential energy: each rank computes pairs where i is in local range and j > i.
// Summing all ranks' contributions yields the full energy.
double computeLocalEnergy(const std::vector<Body>& allBodies, size_t localStart, size_t localEnd) {
    double energy = 0.0;
    const size_t n = allBodies.size();

    // Kinetic energy for local bodies
    for (size_t i = localStart; i < localEnd; ++i) {
        const auto& body = allBodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x +
                         body.vel.y * body.vel.y +
                         body.vel.z * body.vel.z);
    }

    // Potential energy: pairs (i, j) where i is in local range and j > i
    for (size_t i = localStart; i < localEnd; ++i) {
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

// Validate bodies in [localStart, localEnd)
bool validateLocal(const std::vector<Body>& bodies, size_t localStart, size_t localEnd) {
    for (size_t i = localStart; i < localEnd; ++i) {
        const auto& body = bodies[i];
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            return false;
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
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
    }

    // Each rank holds a full copy of all bodies for force computation.
    std::vector<Body> bodies(numBodies);

    // Initialize bodies on rank 0, then broadcast to all ranks.
    if (rank == 0) {
        randomizeBodies(bodies);
    }
    MPI_Bcast(reinterpret_cast<char*>(bodies.data()),
              numBodies * static_cast<int>(sizeof(Body)), MPI_BYTE,
              0, MPI_COMM_WORLD);

    // Compute local work partition for this rank.
    const size_t bodiesPerRank = numBodies / numRanks;
    const size_t localStart = rank * bodiesPerRank;
    const size_t localEnd = (rank == numRanks - 1) ? numBodies : (rank + 1) * bodiesPerRank;
    const size_t localCount = localEnd - localStart;

    // Precompute Allgatherv receive counts and displacements (in bytes).
    std::vector<int> recvCounts(numRanks);
    std::vector<int> recvDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        size_t rStart = r * bodiesPerRank;
        size_t rEnd = (r == numRanks - 1) ? numBodies : (r + 1) * bodiesPerRank;
        recvCounts[r] = static_cast<int>((rEnd - rStart) * sizeof(Body));
        recvDispls[r] = static_cast<int>(rStart * sizeof(Body));
    }

    // Temporary receive buffer for Allgatherv sync.
    std::vector<Body> recvBuffer(numBodies);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Each rank computes forces for its local bodies against ALL bodies.
        // All ranks operate on the same snapshot of body positions.
        computeForcesLocal(bodies, bodies, localStart, localEnd);

        // Each rank integrates its local bodies.
        integrateBodiesLocal(bodies, localStart, localEnd);

        // Synchronize: gather each rank's local updates into the full body array.
        MPI_Allgatherv(
            reinterpret_cast<char*>(bodies.data() + localStart),
            static_cast<int>(localCount * sizeof(Body)), MPI_BYTE,
            reinterpret_cast<char*>(recvBuffer.data()),
            recvCounts.data(), recvDispls.data(), MPI_BYTE,
            MPI_COMM_WORLD);
        std::memcpy(bodies.data(), recvBuffer.data(), numBodies * sizeof(Body));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Print results for external validation (only rank 0).
    if (printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (const auto& body : bodies) {
            bodyData.push_back(body.pos.x);
            bodyData.push_back(body.pos.y);
            bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x);
            bodyData.push_back(body.vel.y);
            bodyData.push_back(body.vel.z);
        }
        if (rank == 0) {
            print_results(bodyData, "Bodies");
        }
    }

    // Validation
    if (validate) {
        bool localValid = validateLocal(bodies, localStart, localEnd);
        int globalValid;
        MPI_Allreduce(&localValid, &globalValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validating simulation results...\n");
        }

        if (globalValid) {
            // Compute total energy via reduction.
            double localEnergy = computeLocalEnergy(bodies, localStart, localEnd);
            double totalEnergy;
            MPI_Reduce(&localEnergy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM,
                       0, MPI_COMM_WORLD);

            if (rank == 0) {
                printf("Final energy: %.6f\n", totalEnergy);
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