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

// Number of target bodies processed simultaneously in the force kernel.
// The inner (source) loop stays sequential per target body, so the summation
// order -- and therefore the result -- is identical to the serial code, while
// the compiler can vectorize across the block lanes.
constexpr size_t BLOCK = 4;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// Positions of all bodies are replicated on every rank as {x,y,z} triples so
// that they can be exchanged with a single Allgatherv; velocities are only
// stored for the locally owned slice.
using Coords = std::vector<double>;

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

// Computes the forces acting on the local bodies [localOffset, localOffset+localN)
// from all n bodies and advances the local velocities.
static void computeForcesLocal(const double* __restrict pos, double* __restrict vel, const size_t n,
                               const size_t localOffset, const size_t localN) {
    size_t i = 0;

    for (; i + BLOCK <= localN; i += BLOCK) {
        double xi[BLOCK], yi[BLOCK], zi[BLOCK];
        double Fx[BLOCK] = {}, Fy[BLOCK] = {}, Fz[BLOCK] = {};

        for (size_t b = 0; b < BLOCK; ++b) {
            xi[b] = pos[3 * (localOffset + i + b) + 0];
            yi[b] = pos[3 * (localOffset + i + b) + 1];
            zi[b] = pos[3 * (localOffset + i + b) + 2];
        }

        for (size_t j = 0; j < n; ++j) {
            const double xj = pos[3 * j + 0];
            const double yj = pos[3 * j + 1];
            const double zj = pos[3 * j + 2];

            for (size_t b = 0; b < BLOCK; ++b) {
                const double dx = xj - xi[b];
                const double dy = yj - yi[b];
                const double dz = zj - zi[b];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx[b] += dx * invDist3;
                Fy[b] += dy * invDist3;
                Fz[b] += dz * invDist3;
            }
        }

        for (size_t b = 0; b < BLOCK; ++b) {
            vel[3 * (i + b) + 0] += DT * Fx[b];
            vel[3 * (i + b) + 1] += DT * Fy[b];
            vel[3 * (i + b) + 2] += DT * Fz[b];
        }
    }

    for (; i < localN; ++i) {
        const double xi = pos[3 * (localOffset + i) + 0];
        const double yi = pos[3 * (localOffset + i) + 1];
        const double zi = pos[3 * (localOffset + i) + 2];
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (size_t j = 0; j < n; ++j) {
            const double dx = pos[3 * j + 0] - xi;
            const double dy = pos[3 * j + 1] - yi;
            const double dz = pos[3 * j + 2] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        vel[3 * i + 0] += DT * Fx;
        vel[3 * i + 1] += DT * Fy;
        vel[3 * i + 2] += DT * Fz;
    }
}

// Advances the positions of the locally owned bodies into the local send buffer.
static void integrateBodiesLocal(const double* __restrict pos, const double* __restrict vel,
                                 double* __restrict posOut, const size_t localOffset, const size_t localN) {
    for (size_t i = 0; i < 3 * localN; ++i) {
        posOut[i] = pos[3 * localOffset + i] + vel[i] * DT;
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

    const size_t n = numBodies > 0 ? static_cast<size_t>(numBodies) : 0;

    // Block-distribute the bodies over the ranks; every rank derives the same
    // decomposition and initial state, so no initial communication is needed.
    // The distribution is done in units of BLOCK bodies so that each rank's
    // slice is aligned with the force kernel's blocking: the results are then
    // bit-identical regardless of the number of ranks used.
    const size_t numChunks = (n + BLOCK - 1) / BLOCK;
    std::vector<int> posCounts(numRanks), posDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        const size_t begin = std::min(n, BLOCK * ((numChunks * static_cast<size_t>(r)) / static_cast<size_t>(numRanks)));
        const size_t end = std::min(n, BLOCK * ((numChunks * static_cast<size_t>(r + 1)) / static_cast<size_t>(numRanks)));
        posDispls[r] = static_cast<int>(3 * begin);
        posCounts[r] = static_cast<int>(3 * (end - begin));
    }
    const size_t localOffset = static_cast<size_t>(posDispls[rank]) / 3;
    const size_t localN = static_cast<size_t>(posCounts[rank]) / 3;

    // Initialize bodies (replicated, deterministic on every rank)
    std::vector<Body> bodies(n);
    randomizeBodies(bodies);

    Coords pos(3 * n);
    Coords localVel(3 * localN);
    Coords localPos(3 * localN);
    for (size_t i = 0; i < n; ++i) {
        pos[3 * i + 0] = bodies[i].pos.x;
        pos[3 * i + 1] = bodies[i].pos.y;
        pos[3 * i + 2] = bodies[i].pos.z;
    }
    for (size_t i = 0; i < localN; ++i) {
        localVel[3 * i + 0] = bodies[localOffset + i].vel.x;
        localVel[3 * i + 1] = bodies[localOffset + i].vel.y;
        localVel[3 * i + 2] = bodies[localOffset + i].vel.z;
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesLocal(pos.data(), localVel.data(), n, localOffset, localN);
        integrateBodiesLocal(pos.data(), localVel.data(), localPos.data(), localOffset, localN);
        MPI_Allgatherv(localPos.data(), posCounts[rank], MPI_DOUBLE, pos.data(), posCounts.data(),
                       posDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long elapsed = static_cast<long>(duration.count());
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", elapsed);
    }

    // Collect the final state on the root for output and validation. Positions
    // are already replicated, only the velocities need to be gathered.
    if (printResults || validate) {
        Coords vel(rank == 0 ? 3 * n : 0);
        MPI_Gatherv(localVel.data(), posCounts[rank], MPI_DOUBLE, vel.data(), posCounts.data(),
                    posDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (size_t i = 0; i < n; ++i) {
                bodies[i].pos = Vec3(pos[3 * i + 0], pos[3 * i + 1], pos[3 * i + 2]);
                bodies[i].vel = Vec3(vel[3 * i + 0], vel[3 * i + 1], vel[3 * i + 2]);
            }
        }
    }

    int exitCode = 0;

    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.reserve(n * 6);
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
            printf("Validating simulation results...\n");

            if (validateSimulation(bodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies);
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
