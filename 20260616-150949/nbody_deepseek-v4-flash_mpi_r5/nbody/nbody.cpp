#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
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

// Generate bodies deterministically: each rank skips ahead in the RNG stream
// to produce the same bodies as the sequential version, regardless of rank count.
void randomizeBodies(std::vector<Body>& bodies, size_t globalOffset) {
    unsigned int seed = 42;
    size_t skip = globalOffset * 6;
    for (size_t s = 0; s < skip; ++s) {
        rand_r(&seed);
    }
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// Compute forces on local bodies using globally-gathered positions
void computeForces(std::vector<Body>& localBodies, const double* allPos, size_t n) {
    const size_t local_n = localBodies.size();

    for (size_t i = 0; i < local_n; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        const double px = localBodies[i].pos.x;
        const double py = localBodies[i].pos.y;
        const double pz = localBodies[i].pos.z;

        for (size_t j = 0; j < n; ++j) {
            const double dx = allPos[3 * j] - px;
            const double dy = allPos[3 * j + 1] - py;
            const double dz = allPos[3 * j + 2] - pz;
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

    // Parse command line arguments (all ranks parse independently)
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

    // Determine block distribution of bodies across ranks
    const int base = numBodies / numRanks;
    const int remainder = numBodies % numRanks;
    const int local_n = base + (rank < remainder ? 1 : 0);
    const int localStart = rank * base + std::min(rank, remainder);

    // Precompute MPI gather counts / displacements
    std::vector<int> posCounts(numRanks), posDispls(numRanks);
    std::vector<int> bodyCounts(numRanks), bodyDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        int n = base + (r < remainder ? 1 : 0);
        posCounts[r] = 3 * n;
        posDispls[r] = (r == 0) ? 0 : posDispls[r - 1] + posCounts[r - 1];
        bodyCounts[r] = 6 * n;
        bodyDispls[r] = (r == 0) ? 0 : bodyDispls[r - 1] + bodyCounts[r - 1];
    }

    // Initialize local bodies (deterministic regardless of rank count)
    std::vector<Body> localBodies(local_n);
    randomizeBodies(localBodies, localStart);

    // MPI communication buffers
    std::vector<double> localPos(3 * local_n);
    std::vector<double> allPos(3 * numBodies);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Pack local positions
        for (int i = 0; i < local_n; ++i) {
            localPos[3 * i]     = localBodies[i].pos.x;
            localPos[3 * i + 1] = localBodies[i].pos.y;
            localPos[3 * i + 2] = localBodies[i].pos.z;
        }

        // Gather all positions so each rank can compute forces
        MPI_Allgatherv(localPos.data(), 3 * local_n, MPI_DOUBLE,
                       allPos.data(), posCounts.data(), posDispls.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Compute forces on local bodies using all positions
        computeForces(localBodies, allPos.data(), numBodies);

        // Integrate local bodies
        integrateBodies(localBodies);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Gather all bodies to rank 0 for validation / results output
    if (validate || printResults) {
        std::vector<double> localBodyData(6 * local_n);
        for (int i = 0; i < local_n; ++i) {
            localBodyData[6 * i]     = localBodies[i].pos.x;
            localBodyData[6 * i + 1] = localBodies[i].pos.y;
            localBodyData[6 * i + 2] = localBodies[i].pos.z;
            localBodyData[6 * i + 3] = localBodies[i].vel.x;
            localBodyData[6 * i + 4] = localBodies[i].vel.y;
            localBodyData[6 * i + 5] = localBodies[i].vel.z;
        }

        std::vector<double> allBodyData;
        if (rank == 0) allBodyData.resize(6 * numBodies);

        MPI_Gatherv(localBodyData.data(), 6 * local_n, MPI_DOUBLE,
                    allBodyData.data(), bodyCounts.data(), bodyDispls.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Reconstruct full bodies on rank 0
            std::vector<Body> allBodies(numBodies);
            for (int i = 0; i < numBodies; ++i) {
                allBodies[i].pos.x = allBodyData[6 * i];
                allBodies[i].pos.y = allBodyData[6 * i + 1];
                allBodies[i].pos.z = allBodyData[6 * i + 2];
                allBodies[i].vel.x = allBodyData[6 * i + 3];
                allBodies[i].vel.y = allBodyData[6 * i + 4];
                allBodies[i].vel.z = allBodyData[6 * i + 5];
            }

            // Print results for external validation
            if (printResults) {
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

            // Validation: check that simulation produces finite, reasonable values
            if (validate) {
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
        }
    }

    MPI_Finalize();
    return 0;
}
