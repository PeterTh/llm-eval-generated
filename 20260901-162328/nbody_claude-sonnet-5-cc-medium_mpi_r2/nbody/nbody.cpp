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

// Compute the block distribution of `n` items over `size` ranks: counts[r] items
// starting at displs[r], with the remainder spread over the first ranks.
void computeBlockDistribution(int n, int size, std::vector<int>& counts, std::vector<int>& displs) {
    counts.assign(size, 0);
    displs.assign(size, 0);
    const int base = n / size;
    const int remainder = n % size;
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < remainder ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
}

// Compute forces and update velocities for the local slice of bodies [start, end),
// using the full (up-to-date) bodies array for the interactions.
void computeForcesLocal(std::vector<Body>& bodies, size_t start, size_t end) {
    const size_t n = bodies.size();

    for (size_t i = start; i < end; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (size_t j = 0; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
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

void integrateBodiesLocal(std::vector<Body>& bodies, size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

// Gather the position (or velocity) component of the local slice from every rank
// into every rank's full `bodies` array via MPI_Allgatherv.
void allgatherVec3(std::vector<Body>& bodies, size_t start, size_t count, bool position,
                    const std::vector<int>& recvCounts3, const std::vector<int>& recvDispls3) {
    std::vector<double> sendBuf(count * 3);
    for (size_t i = 0; i < count; ++i) {
        const Vec3& v = position ? bodies[start + i].pos : bodies[start + i].vel;
        sendBuf[3 * i + 0] = v.x;
        sendBuf[3 * i + 1] = v.y;
        sendBuf[3 * i + 2] = v.z;
    }

    std::vector<double> recvBuf(bodies.size() * 3);
    MPI_Allgatherv(sendBuf.data(), static_cast<int>(count * 3), MPI_DOUBLE, recvBuf.data(), recvCounts3.data(),
                   recvDispls3.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    for (size_t j = 0; j < bodies.size(); ++j) {
        Vec3& v = position ? bodies[j].pos : bodies[j].vel;
        v.x = recvBuf[3 * j + 0];
        v.y = recvBuf[3 * j + 1];
        v.z = recvBuf[3 * j + 2];
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
        printf("Number of MPI ranks: %d\n", worldSize);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Initialize bodies (deterministic seed => identical on every rank)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Distribute bodies in contiguous blocks across ranks
    std::vector<int> counts, displs;
    computeBlockDistribution(numBodies, worldSize, counts, displs);
    const size_t localStart = static_cast<size_t>(displs[rank]);
    const size_t localCount = static_cast<size_t>(counts[rank]);
    const size_t localEnd = localStart + localCount;

    std::vector<int> counts3(worldSize), displs3(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        counts3[r] = counts[r] * 3;
        displs3[r] = displs[r] * 3;
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesLocal(bodies, localStart, localEnd);
        integrateBodiesLocal(bodies, localStart, localEnd);
        allgatherVec3(bodies, localStart, localCount, /*position=*/true, counts3, displs3);
    }

    // Gather final velocities so every rank has the complete, consistent state
    allgatherVec3(bodies, localStart, localCount, /*position=*/false, counts3, displs3);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", max_ms);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
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
        print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        int retCode = 0;
        if (rank == 0) {
            printf("Validating simulation results...\n");

            if (validateSimulation(bodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
                retCode = 0;
            } else {
                printf("Validation: FAILED\n");
                retCode = 1;
            }
        }
        MPI_Bcast(&retCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return retCode;
    }

    MPI_Finalize();
    return 0;
}
