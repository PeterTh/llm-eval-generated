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

static inline void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

static inline void computeForcesLocal(std::vector<Body>& localBodies,
                                     const std::vector<double>& allPos,
                                     size_t globalOffset) {
    const size_t localN = localBodies.size();
    const size_t n = allPos.size() / 3;

    for (size_t li = 0; li < localN; ++li) {
        const size_t gi = globalOffset + li;
        const double ix = allPos[3 * gi + 0];
        const double iy = allPos[3 * gi + 1];
        const double iz = allPos[3 * gi + 2];

        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        for (size_t j = 0; j < n; ++j) {
            const double dx = allPos[3 * j + 0] - ix;
            const double dy = allPos[3 * j + 1] - iy;
            const double dz = allPos[3 * j + 2] - iz;
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

static inline void integrateBodiesLocal(std::vector<Body>& localBodies) {
    for (auto& body : localBodies) {
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

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks see the same argv under mpirun)
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
        printf("N-Body Simulation (MPI)\n");
        printf("MPI ranks: %d\n", size);
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Block distribution (works for any numBodies, including numBodies < size)
    const int base = numBodies / size;
    const int rem = numBodies % size;
    const int localN = base + (rank < rem ? 1 : 0);
    const int globalOffset = rank * base + (rank < rem ? rank : rem);

    std::vector<int> counts(size), displs(size), counts6(size), displs6(size), counts3(size), displs3(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = r * base + (r < rem ? r : rem);
        counts6[r] = counts[r] * 6;
        displs6[r] = displs[r] * 6;
        counts3[r] = counts[r] * 3;
        displs3[r] = displs[r] * 3;
    }

    // Initialize bodies on rank 0 to preserve exact serial semantics, then Scatterv.
    std::vector<double> allInit;
    if (rank == 0) {
        std::vector<Body> bodies(numBodies);
        randomizeBodies(bodies);
        allInit.resize((size_t)numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            allInit[(size_t)i * 6 + 0] = bodies[i].pos.x;
            allInit[(size_t)i * 6 + 1] = bodies[i].pos.y;
            allInit[(size_t)i * 6 + 2] = bodies[i].pos.z;
            allInit[(size_t)i * 6 + 3] = bodies[i].vel.x;
            allInit[(size_t)i * 6 + 4] = bodies[i].vel.y;
            allInit[(size_t)i * 6 + 5] = bodies[i].vel.z;
        }
    }

    std::vector<double> localInit((size_t)localN * 6);
    MPI_Scatterv(rank == 0 ? allInit.data() : nullptr,
                 counts6.data(), displs6.data(), MPI_DOUBLE,
                 localInit.data(), (int)localInit.size(), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    std::vector<Body> localBodies((size_t)localN);
    for (int i = 0; i < localN; ++i) {
        localBodies[i].pos.x = localInit[(size_t)i * 6 + 0];
        localBodies[i].pos.y = localInit[(size_t)i * 6 + 1];
        localBodies[i].pos.z = localInit[(size_t)i * 6 + 2];
        localBodies[i].vel.x = localInit[(size_t)i * 6 + 3];
        localBodies[i].vel.y = localInit[(size_t)i * 6 + 4];
        localBodies[i].vel.z = localInit[(size_t)i * 6 + 5];
    }

    // Run simulation
    std::vector<double> localPos((size_t)localN * 3);
    std::vector<double> allPos((size_t)numBodies * 3);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        for (int i = 0; i < localN; ++i) {
            localPos[(size_t)i * 3 + 0] = localBodies[i].pos.x;
            localPos[(size_t)i * 3 + 1] = localBodies[i].pos.y;
            localPos[(size_t)i * 3 + 2] = localBodies[i].pos.z;
        }

        // Allgather positions so every rank can compute forces with identical j-ordering as serial.
        MPI_Allgatherv(localPos.data(), (int)localPos.size(), MPI_DOUBLE,
                       allPos.data(), counts3.data(), displs3.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        computeForcesLocal(localBodies, allPos, (size_t)globalOffset);
        integrateBodiesLocal(localBodies);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    const double msLocal = (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double msMax = 0.0;
    MPI_Reduce(&msLocal, &msMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %.0f ms\n", msMax);
    }

    // Gather full state to rank 0 for results/validation
    const bool needGather = printResults || validate;
    std::vector<double> localState((size_t)localN * 6);
    for (int i = 0; i < localN; ++i) {
        localState[(size_t)i * 6 + 0] = localBodies[i].pos.x;
        localState[(size_t)i * 6 + 1] = localBodies[i].pos.y;
        localState[(size_t)i * 6 + 2] = localBodies[i].pos.z;
        localState[(size_t)i * 6 + 3] = localBodies[i].vel.x;
        localState[(size_t)i * 6 + 4] = localBodies[i].vel.y;
        localState[(size_t)i * 6 + 5] = localBodies[i].vel.z;
    }

    std::vector<double> allState;
    if (needGather && rank == 0) allState.resize((size_t)numBodies * 6);

    if (needGather) {
        MPI_Gatherv(localState.data(), (int)localState.size(), MPI_DOUBLE,
                    rank == 0 ? allState.data() : nullptr,
                    counts6.data(), displs6.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (needGather && rank == 0) {
        if (printResults) {
            print_results(allState, "Bodies");
        }

        if (validate) {
            printf("Validating simulation results...\n");
            std::vector<Body> bodies(numBodies);
            for (int i = 0; i < numBodies; ++i) {
                bodies[i].pos.x = allState[(size_t)i * 6 + 0];
                bodies[i].pos.y = allState[(size_t)i * 6 + 1];
                bodies[i].pos.z = allState[(size_t)i * 6 + 2];
                bodies[i].vel.x = allState[(size_t)i * 6 + 3];
                bodies[i].vel.y = allState[(size_t)i * 6 + 4];
                bodies[i].vel.z = allState[(size_t)i * 6 + 5];
            }

            if (validateSimulation(bodies)) {
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
