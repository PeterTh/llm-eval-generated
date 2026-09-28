#include <algorithm>
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

// Number of bodies whose force is accumulated simultaneously. The lane loop below
// is what the compiler turns into SIMD code, so this should match the machine's
// vector width in doubles (4 for AVX2).
constexpr size_t LANES = 4;

// Distributed force computation: this rank owns bodies [begin, end) and updates
// their velocities. Positions of *all* bodies are replicated in `pos` (interleaved
// x,y,z).
//
// The bodies are processed in groups of LANES: the j loop broadcasts one source
// body to all lanes, and every lane keeps its own force accumulator. Each
// accumulator therefore still sees the interactions in ascending j order exactly
// as in the serial version, which makes the result bit-for-bit identical while
// vectorizing across i instead of across the (order-bound) reduction.
void computeForces(const double* __restrict pos, double* __restrict vel, const size_t n, const size_t begin,
    const size_t end) {
    for (size_t i0 = begin; i0 < end; i0 += LANES) {
        const size_t nl = std::min(LANES, end - i0);

        double xi[LANES], yi[LANES], zi[LANES];
        double Fx[LANES], Fy[LANES], Fz[LANES];
        for (size_t l = 0; l < LANES; ++l) {
            // Unused lanes of a partial group repeat body i0; their result is dropped.
            const size_t i = (l < nl) ? i0 + l : i0;
            xi[l] = pos[3 * i + 0];
            yi[l] = pos[3 * i + 1];
            zi[l] = pos[3 * i + 2];
            Fx[l] = 0.0;
            Fy[l] = 0.0;
            Fz[l] = 0.0;
        }

        for (size_t j = 0; j < n; ++j) {
            const double xj = pos[3 * j + 0];
            const double yj = pos[3 * j + 1];
            const double zj = pos[3 * j + 2];

            for (size_t l = 0; l < LANES; ++l) {
                const double dx = xj - xi[l];
                const double dy = yj - yi[l];
                const double dz = zj - zi[l];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx[l] = std::fma(dx, invDist3, Fx[l]);
                Fy[l] = std::fma(dy, invDist3, Fy[l]);
                Fz[l] = std::fma(dz, invDist3, Fz[l]);
            }
        }

        for (size_t l = 0; l < nl; ++l) {
            const size_t k = i0 + l - begin;
            // volatile keeps DT * F a separately rounded product, matching the code
            // the serial version generates for this (out of the hot loop) update.
            const volatile double dvx = DT * Fx[l];
            const volatile double dvy = DT * Fy[l];
            const volatile double dvz = DT * Fz[l];
            vel[3 * k + 0] += dvx;
            vel[3 * k + 1] += dvy;
            vel[3 * k + 2] += dvz;
        }
    }
}

// Integrate the locally owned bodies in place.
void integrateBodies(double* __restrict pos, const double* __restrict vel, const size_t begin, const size_t end) {
    for (size_t i = begin; i < end; ++i) {
        const size_t l = i - begin;
        pos[3 * i + 0] += vel[3 * l + 0] * DT;
        pos[3 * i + 1] += vel[3 * l + 1] * DT;
        pos[3 * i + 2] += vel[3 * l + 2] * DT;
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

    if (numBodies < 0) numBodies = 0;

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t n = static_cast<size_t>(numBodies);

    // Block-distribute the bodies over the ranks. Each rank owns [begin, end) and
    // is responsible for their velocity update and position integration. The work
    // per body is identical, so equal-sized blocks balance the load; blocks are
    // handed out in units of LANES so that only the very last one can be partial.
    std::vector<int> counts(numRanks), displs(numRanks);
    {
        const size_t numGroups = (n + LANES - 1) / LANES;
        const size_t base = numGroups / numRanks;
        const size_t rem = numGroups % numRanks;
        size_t off = 0;
        for (int r = 0; r < numRanks; ++r) {
            const size_t groups = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            const size_t cnt = std::min(groups * LANES, n - off);
            displs[r] = static_cast<int>(3 * off);
            counts[r] = static_cast<int>(3 * cnt);
            off += cnt;
        }
    }
    const size_t begin = static_cast<size_t>(displs[rank]) / 3;
    const size_t localN = static_cast<size_t>(counts[rank]) / 3;
    const size_t end = begin + localN;

    // Replicated positions (interleaved x,y,z) and locally owned velocities.
    std::vector<double> pos(3 * n);
    std::vector<double> vel(3 * localN);
    {
        // Deterministic initialization, performed redundantly on every rank so
        // that no communication is required to get started.
        std::vector<Body> bodies(n);
        randomizeBodies(bodies);
        for (size_t i = 0; i < n; ++i) {
            pos[3 * i + 0] = bodies[i].pos.x;
            pos[3 * i + 1] = bodies[i].pos.y;
            pos[3 * i + 2] = bodies[i].pos.z;
        }
        for (size_t l = 0; l < localN; ++l) {
            vel[3 * l + 0] = bodies[begin + l].vel.x;
            vel[3 * l + 1] = bodies[begin + l].vel.y;
            vel[3 * l + 2] = bodies[begin + l].vel.z;
        }
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(pos.data(), vel.data(), n, begin, end);
        integrateBodies(pos.data(), vel.data(), begin, end);
        // Republish the updated positions; every rank needs all of them for the
        // next force evaluation.
        MPI_Allgatherv(MPI_IN_PLACE, counts[rank], MPI_DOUBLE, pos.data(), counts.data(), displs.data(), MPI_DOUBLE,
            MPI_COMM_WORLD);
    }

    auto end_t = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start);

    long localMs = static_cast<long>(duration.count());
    long maxMs = localMs;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", maxMs);
    }

    // Collect the final state on rank 0 for output and validation. Positions are
    // already replicated, velocities live on their owning rank.
    std::vector<Body> bodies;
    if (printResults || validate) {
        std::vector<double> allVel(rank == 0 ? 3 * n : 0);
        MPI_Gatherv(vel.data(), counts[rank], MPI_DOUBLE, allVel.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
            MPI_COMM_WORLD);
        if (rank == 0) {
            bodies.resize(n);
            for (size_t i = 0; i < n; ++i) {
                bodies[i].pos = Vec3(pos[3 * i + 0], pos[3 * i + 1], pos[3 * i + 2]);
                bodies[i].vel = Vec3(allVel[3 * i + 0], allVel[3 * i + 1], allVel[3 * i + 2]);
            }
        }
    }

    // Print results for external validation
    if (printResults && rank == 0) {
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
        int status = 0;
        if (rank == 0) {
            printf("Validating simulation results...\n");

            if (validateSimulation(bodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return status;
    }

    MPI_Finalize();
    return 0;
}
