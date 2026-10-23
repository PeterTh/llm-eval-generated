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

static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be three contiguous doubles");
static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be six contiguous doubles");

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

// Compute forces (and update velocities) for the contiguous body range
// [begin, end) owned by this rank.  Positions of all bodies are replicated on
// every rank, so each rank evaluates the same per-body expression as the
// original serial implementation with the same accumulation order.
void computeForces(std::vector<Body>& bodies, const size_t begin, const size_t end) {
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

void integrateBodies(std::vector<Body>& bodies, const size_t begin, const size_t end) {
    for (size_t i = begin; i < end; ++i) {
        Body& body = bodies[i];
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

    if (numBodies < 0) {
        numBodies = 0;
    }

    const size_t n = static_cast<size_t>(numBodies);

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Distribute the bodies contiguously over the ranks.
    std::vector<int> counts(numRanks), displs(numRanks), bodyCounts(numRanks), bodyDispls(numRanks);
    const int base = numBodies / numRanks;
    const int rem = numBodies % numRanks;
    int offset = 0;
    for (int r = 0; r < numRanks; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        bodyCounts[r] = 6 * counts[r];
        bodyDispls[r] = 6 * displs[r];
        offset += counts[r];
    }
    const size_t begin = static_cast<size_t>(displs[rank]);
    const size_t end = begin + static_cast<size_t>(counts[rank]);

    // Every rank holds the full state: positions are replicated for force
    // evaluation, while velocities are only updated for the owned range.
    std::vector<Body> bodies(n);

    // Rank 0 generates the complete initial state and broadcasts it.  A Body
    // is laid out as six consecutive doubles, so it can be sent as a flat
    // array.  Starting from the identical state on every rank guarantees
    // deterministic, rank-count-independent results.
    if (rank == 0) {
        randomizeBodies(bodies);
    }
    if (n > 0) {
        MPI_Bcast(reinterpret_cast<double*>(bodies.data()), static_cast<int>(6 * n),
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Buffers used to publish updated positions.  Positions are packed as
    // three consecutive doubles per body so that the gather is a plain
    // contiguous MPI_DOUBLE operation.
    std::vector<double> posSend(3 * (end - begin));
    std::vector<double> posRecv(3 * n);
    std::vector<int> posCounts(numRanks), posDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        posCounts[r] = 3 * counts[r];
        posDispls[r] = 3 * displs[r];
    }

    // Run simulation.
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, begin, end);
        integrateBodies(bodies, begin, end);

        // Publish this rank's updated positions to all other ranks for the
        // next force evaluation.
        if (n > 0) {
            for (size_t i = begin; i < end; ++i) {
                const size_t k = 3 * (i - begin);
                posSend[k + 0] = bodies[i].pos.x;
                posSend[k + 1] = bodies[i].pos.y;
                posSend[k + 2] = bodies[i].pos.z;
            }
            MPI_Allgatherv(posSend.data(), posCounts[rank], MPI_DOUBLE,
                           posRecv.data(), posCounts.data(), posDispls.data(), MPI_DOUBLE,
                           MPI_COMM_WORLD);
            for (size_t i = 0; i < n; ++i) {
                const size_t k = 3 * i;
                bodies[i].pos.x = posRecv[k + 0];
                bodies[i].pos.y = posRecv[k + 1];
                bodies[i].pos.z = posRecv[k + 2];
            }
        }
    }

    const double tEnd = MPI_Wtime();
    const long durationMs = static_cast<long>((tEnd - tStart) * 1000.0);
    long maxDurationMs;
    MPI_Reduce(&durationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", maxDurationMs);
    }

    // Gather the final state on rank 0 for result output / validation.
    std::vector<double> bodyData;
    if ((printResults || validate) && n > 0) {
        if (rank == 0) {
            bodyData.resize(6 * n);
        }
        MPI_Gatherv(reinterpret_cast<const double*>(bodies.data() + begin), bodyCounts[rank], MPI_DOUBLE,
                    rank == 0 ? bodyData.data() : nullptr, bodyCounts.data(), bodyDispls.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int exitCode = 0;

    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(bodyData, "Bodies");
        }

        // Validation: check that simulation produces finite, reasonable values
        if (validate) {
            printf("Validating simulation results...\n");

            std::vector<Body> allBodies(n);
            for (size_t i = 0; i < n; ++i) {
                allBodies[i].pos.x = bodyData[6 * i + 0];
                allBodies[i].pos.y = bodyData[6 * i + 1];
                allBodies[i].pos.z = bodyData[6 * i + 2];
                allBodies[i].vel.x = bodyData[6 * i + 3];
                allBodies[i].vel.y = bodyData[6 * i + 4];
                allBodies[i].vel.z = bodyData[6 * i + 5];
            }

            if (validateSimulation(allBodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(allBodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exitCode;
}
