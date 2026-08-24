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

// Compute gravitational forces on local bodies from all bodies in the system.
// Each rank computes forces only on its owned bodies, but uses all bodies
// as sources. This gives O(N^2 / P) compute per rank with O(N) communication
// per step for the position gather.
void computeForcesLocal(std::vector<Body>& localBodies,
                        const std::vector<Body>& allBodies,
                        size_t localCount, size_t totalCount) {
    for (size_t i = 0; i < localCount; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (size_t j = 0; j < totalCount; ++j) {
            const double dx = allBodies[j].pos.x - localBodies[i].pos.x;
            const double dy = allBodies[j].pos.y - localBodies[i].pos.y;
            const double dz = allBodies[j].pos.z - localBodies[i].pos.z;
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

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (same on all ranks)
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
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t n = static_cast<size_t>(numBodies);

    // Block distribution: bodies are split contiguously across ranks.
    // Ranks 0..remainder-1 get baseCount+1 bodies, the rest get baseCount.
    const size_t baseCount = n / static_cast<size_t>(numRanks);
    const size_t remainder = n % static_cast<size_t>(numRanks);
    const size_t localCount = baseCount + (static_cast<size_t>(rank) < remainder ? 1 : 0);

    // Build send/recv count and displacement arrays (in bytes for MPI_BYTE)
    std::vector<int> counts(numRanks);
    std::vector<int> displs(numRanks);
    {
        size_t offset = 0;
        for (int r = 0; r < numRanks; ++r) {
            size_t cnt = baseCount + (static_cast<size_t>(r) < remainder ? 1 : 0);
            counts[r] = static_cast<int>(cnt * sizeof(Body));
            displs[r] = static_cast<int>(offset * sizeof(Body));
            offset += cnt;
        }
    }

    // Allocate local body storage
    std::vector<Body> localBodies(localCount);

    // Rank 0 initializes all bodies, then scatters to all ranks.
    // This guarantees identical initial conditions regardless of rank count.
    std::vector<Body> allBodies(n);
    if (rank == 0) {
        randomizeBodies(allBodies);
    }

    // Scatter bodies: rank 0 sends, all ranks receive their chunk.
    MPI_Scatterv(allBodies.data(), counts.data(), displs.data(), MPI_BYTE,
                 localBodies.data(), static_cast<int>(localCount * sizeof(Body)), MPI_BYTE,
                 0, MPI_COMM_WORLD);

    // allBodies is used as the receive buffer for Allgatherv on every rank.
    // It is already sized to n.

    // ---- Simulation loop ----
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Gather all body states to every rank so each rank can compute
        // forces on its local bodies against all other bodies.
        MPI_Allgatherv(localBodies.data(), static_cast<int>(localCount * sizeof(Body)), MPI_BYTE,
                       allBodies.data(), counts.data(), displs.data(), MPI_BYTE,
                       MPI_COMM_WORLD);

        // Compute forces on local bodies using the full body set.
        computeForcesLocal(localBodies, allBodies, localCount, n);

        // Integrate local bodies.
        integrateBodies(localBodies);
    }

    auto end = std::chrono::high_resolution_clock::now();
    long long localDurationMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long durationMs = 0;
    MPI_Reduce(&localDurationMs, &durationMs, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %lld ms\n", durationMs);
    }

    // Gather all bodies back to rank 0 for results / validation.
    // On non-root ranks, allBodies is not used after this call.
    MPI_Gatherv(localBodies.data(), static_cast<int>(localCount * sizeof(Body)), MPI_BYTE,
                allBodies.data(), counts.data(), displs.data(), MPI_BYTE,
                0, MPI_COMM_WORLD);

    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
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

    // Validation (rank 0 only)
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");

        if (validateSimulation(allBodies)) {
            double finalEnergy = computeTotalEnergy(allBodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
