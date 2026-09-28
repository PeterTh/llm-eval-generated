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

// Computes forces (and updates velocities) only for the local range of
// bodies [begin, end); positions of all bodies (including remote ones) must
// already be up to date in `bodies`.
void computeForces(std::vector<Body>& bodies, size_t begin, size_t end) {
    const size_t n = bodies.size();

    for (size_t i = begin; i < end; ++i) {
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

// Integrates positions only for the local range of bodies [begin, end).
void integrateBodies(std::vector<Body>& bodies, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
        auto& body = bodies[i];
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

// Computes the per-rank block decomposition of `n` bodies across `numRanks`
// ranks: counts[r] is the number of bodies owned by rank r, displs[r] is the
// starting index of that block. Bodies are split as evenly as possible.
void computeDistribution(size_t n, int numRanks, std::vector<int>& counts, std::vector<int>& displs) {
    counts.assign(numRanks, 0);
    displs.assign(numRanks, 0);
    const size_t base = n / numRanks;
    const size_t remainder = n % numRanks;
    size_t offset = 0;
    for (int r = 0; r < numRanks; ++r) {
        const size_t count = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
        counts[r] = static_cast<int>(count);
        displs[r] = static_cast<int>(offset);
        offset += count;
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

    int rank = 0, numRanks = 1;
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
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
    }

    // Initialize bodies. Every rank generates the identical full array (same
    // seed), so no broadcast of the initial state is required.
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Distribute bodies across ranks in contiguous blocks. Each rank only
    // computes forces/integration for its own block; the resulting bodies
    // are then synchronized to every rank via MPI_Allgatherv.
    std::vector<int> counts, displs;
    computeDistribution(static_cast<size_t>(numBodies), numRanks, counts, displs);
    const size_t begin = static_cast<size_t>(displs[rank]);
    const size_t end = begin + static_cast<size_t>(counts[rank]);

    // Scale block counts/displacements by 6 doubles per Body (pos + vel).
    std::vector<int> doubleCounts(numRanks), doubleDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        doubleCounts[r] = counts[r] * 6;
        doubleDispls[r] = displs[r] * 6;
    }
    double* bodyData = reinterpret_cast<double*>(bodies.data());

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, begin, end);
        integrateBodies(bodies, begin, end);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bodyData, doubleCounts.data(),
                       doubleDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", maxDuration);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> resultData;
        resultData.reserve(numBodies * 6);
        for (const auto& body : bodies) {
            resultData.push_back(body.pos.x);
            resultData.push_back(body.pos.y);
            resultData.push_back(body.pos.z);
            resultData.push_back(body.vel.x);
            resultData.push_back(body.vel.y);
            resultData.push_back(body.vel.z);
        }
        print_results(resultData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values.
    // Every rank holds an identical, fully synchronized copy of `bodies`, so
    // each rank validates independently and returns a consistent exit code.
    int result = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
        }

        if (validateSimulation(bodies)) {
            if (rank == 0) {
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            result = 0;
        } else {
            if (rank == 0) {
                printf("Validation: FAILED\n");
            }
            result = 1;
        }
    }

    MPI_Finalize();
    return result;
}
