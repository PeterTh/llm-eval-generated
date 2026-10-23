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

static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be tightly packed");
static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be tightly packed");

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

void computeForces(std::vector<Body>& localBodies, const std::vector<Body>& bodies, size_t localOffset) {
    const size_t n = bodies.size();
    const size_t localCount = localBodies.size();

    for (size_t li = 0; li < localCount; ++li) {
        const size_t i = localOffset + li;
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

        localBodies[li].vel.x += DT * Fx;
        localBodies[li].vel.y += DT * Fy;
        localBodies[li].vel.z += DT * Fz;
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

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank parses the same options)
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
    }

    // Contiguous, balanced distribution of bodies across MPI ranks.  Keeping each
    // body's inner loop summation in the original order makes the floating point
    // results identical to the serial implementation.
    std::vector<int> counts(numRanks);
    std::vector<int> displs(numRanks);
    const int base = numBodies / numRanks;
    const int remainder = numBodies % numRanks;
    int offset = 0;
    for (int r = 0; r < numRanks; ++r) {
        counts[r] = base + (r < remainder ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }

    const int localCount = counts[rank];

    MPI_Datatype mpiBody;
    MPI_Type_contiguous(6, MPI_DOUBLE, &mpiBody);
    MPI_Type_commit(&mpiBody);

    // Positions-only view of a Body: three doubles at offset 0, stride 48 bytes.
    MPI_Datatype mpiPosBase;
    MPI_Datatype mpiPos;
    MPI_Type_contiguous(3, MPI_DOUBLE, &mpiPosBase);
    MPI_Type_create_resized(mpiPosBase, 0, 6 * sizeof(double), &mpiPos);
    MPI_Type_commit(&mpiPos);
    MPI_Type_free(&mpiPosBase);

    // Rank 0 performs the deterministic initialization, then scatters the bodies.
    // globalBodies doubles as the gather buffer for the per-step position exchange.
    std::vector<Body> globalBodies(numBodies);
    if (rank == 0) {
        randomizeBodies(globalBodies);
    }
    std::vector<Body> localBodies(localCount);
    MPI_Scatterv(rank == 0 ? globalBodies.data() : nullptr, counts.data(), displs.data(), mpiBody,
                 localBodies.data(), localCount, mpiBody, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Publish positions to every rank before computing the forces.  Velocities
        // stay local because each rank only integrates its own bodies.
        MPI_Allgatherv(localBodies.data(), localCount, mpiPos,
                       globalBodies.data(), counts.data(), displs.data(), mpiPos,
                       MPI_COMM_WORLD);

        computeForces(localBodies, globalBodies, static_cast<size_t>(displs[rank]));
        integrateBodies(localBodies);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Only rank 0 needs the complete state for result printing and validation.
    if (printResults || validate) {
        MPI_Gatherv(localBodies.data(), localCount, mpiBody,
                    rank == 0 ? globalBodies.data() : nullptr, counts.data(), displs.data(), mpiBody,
                    0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.reserve(static_cast<size_t>(numBodies) * 6);
            for (const auto& body : globalBodies) {
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

            if (validateSimulation(globalBodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(globalBodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Type_free(&mpiPos);
    MPI_Type_free(&mpiBody);
    MPI_Finalize();
    return exitCode;
}
