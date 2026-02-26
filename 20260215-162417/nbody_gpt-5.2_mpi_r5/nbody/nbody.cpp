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

static void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

static double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
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
static bool validateSimulation(const std::vector<Body>& bodies) {
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

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks energy conservation)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static void computeCountsDispls(int nBodies, int nRanks, std::vector<int>& countsBodies, std::vector<int>& displsBodies,
                                std::vector<int>& countsDoubles3, std::vector<int>& displsDoubles3) {
    countsBodies.assign(nRanks, 0);
    displsBodies.assign(nRanks, 0);
    countsDoubles3.assign(nRanks, 0);
    displsDoubles3.assign(nRanks, 0);

    const int base = nBodies / nRanks;
    const int rem = nBodies % nRanks;

    int disp = 0;
    for (int r = 0; r < nRanks; ++r) {
        const int cnt = base + (r < rem ? 1 : 0);
        countsBodies[r] = cnt;
        displsBodies[r] = disp;
        countsDoubles3[r] = cnt * 3;
        displsDoubles3[r] = disp * 3;
        disp += cnt;
    }
}

static void computeForcesLocal(const std::vector<double>& posGlobal, std::vector<double>& velLocal, int localOffset,
                              int localCountBodies, int nBodies) {
    const double* __restrict__ pg = posGlobal.data();
    double* __restrict__ vl = velLocal.data();

    for (int li = 0; li < localCountBodies; ++li) {
        const int gi = localOffset + li;
        const double xi = pg[3LL * gi + 0];
        const double yi = pg[3LL * gi + 1];
        const double zi = pg[3LL * gi + 2];

        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (int j = 0; j < nBodies; ++j) {
            const double dx = pg[3LL * j + 0] - xi;
            const double dy = pg[3LL * j + 1] - yi;
            const double dz = pg[3LL * j + 2] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        vl[3LL * li + 0] += DT * Fx;
        vl[3LL * li + 1] += DT * Fy;
        vl[3LL * li + 2] += DT * Fz;
    }
}

static void integrateBodiesLocal(std::vector<double>& posLocal, const std::vector<double>& velLocal, int localCountBodies) {
    double* __restrict__ pl = posLocal.data();
    const double* __restrict__ vl = velLocal.data();

    for (int i = 0; i < localCountBodies; ++i) {
        pl[3LL * i + 0] += vl[3LL * i + 0] * DT;
        pl[3LL * i + 1] += vl[3LL * i + 1] * DT;
        pl[3LL * i + 2] += vl[3LL * i + 2] * DT;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBodies = 1024;
    int numSteps = 10;
    int validate_i = 0;
    int printResults_i = 0;

    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numBodies = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                numSteps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate_i = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults_i = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }

        printf("N-Body Simulation (MPI)\n");
        printf("MPI ranks: %d\n", size);
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate_i ? "enabled" : "disabled");
    }

    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_i, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> countsBodies, displsBodies, countsDoubles3, displsDoubles3;
    computeCountsDispls(numBodies, size, countsBodies, displsBodies, countsDoubles3, displsDoubles3);

    const int localCountBodies = countsBodies[rank];
    const int localOffset = displsBodies[rank];

    std::vector<double> posAll;
    std::vector<double> velAll;

    if (rank == 0) {
        std::vector<Body> bodies(numBodies);
        randomizeBodies(bodies);

        posAll.resize(3LL * numBodies);
        velAll.resize(3LL * numBodies);
        for (int i = 0; i < numBodies; ++i) {
            posAll[3LL * i + 0] = bodies[i].pos.x;
            posAll[3LL * i + 1] = bodies[i].pos.y;
            posAll[3LL * i + 2] = bodies[i].pos.z;
            velAll[3LL * i + 0] = bodies[i].vel.x;
            velAll[3LL * i + 1] = bodies[i].vel.y;
            velAll[3LL * i + 2] = bodies[i].vel.z;
        }
    }

    std::vector<double> posLocal(3LL * localCountBodies);
    std::vector<double> velLocal(3LL * localCountBodies);

    MPI_Scatterv(rank == 0 ? posAll.data() : nullptr, countsDoubles3.data(), displsDoubles3.data(), MPI_DOUBLE,
                 posLocal.data(), localCountBodies * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? velAll.data() : nullptr, countsDoubles3.data(), displsDoubles3.data(), MPI_DOUBLE,
                 velLocal.data(), localCountBodies * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> posGlobal(3LL * numBodies);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        MPI_Allgatherv(posLocal.data(), localCountBodies * 3, MPI_DOUBLE, posGlobal.data(), countsDoubles3.data(),
                       displsDoubles3.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        computeForcesLocal(posGlobal, velLocal, localOffset, localCountBodies, numBodies);
        integrateBodiesLocal(posLocal, velLocal, localCountBodies);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double elapsed = t1 - t0;

    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", (long)llround(maxElapsed * 1000.0));
    }

    // Gather final state to rank 0 for printing/validation
    if (rank == 0) {
        posAll.resize(3LL * numBodies);
        velAll.resize(3LL * numBodies);
    }

    MPI_Gatherv(posLocal.data(), localCountBodies * 3, MPI_DOUBLE, rank == 0 ? posAll.data() : nullptr,
                countsDoubles3.data(), displsDoubles3.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(velLocal.data(), localCountBodies * 3, MPI_DOUBLE, rank == 0 ? velAll.data() : nullptr,
                countsDoubles3.data(), displsDoubles3.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int ret = 0;

    if (rank == 0) {
        std::vector<Body> bodies(numBodies);
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos.x = posAll[3LL * i + 0];
            bodies[i].pos.y = posAll[3LL * i + 1];
            bodies[i].pos.z = posAll[3LL * i + 2];
            bodies[i].vel.x = velAll[3LL * i + 0];
            bodies[i].vel.y = velAll[3LL * i + 1];
            bodies[i].vel.z = velAll[3LL * i + 2];
        }

        // Print results for external validation
        if (printResults_i) {
            std::vector<double> bodyData;
            bodyData.reserve((size_t)numBodies * 6);
            for (int i = 0; i < numBodies; ++i) {
                bodyData.push_back(bodies[i].pos.x);
                bodyData.push_back(bodies[i].pos.y);
                bodyData.push_back(bodies[i].pos.z);
                bodyData.push_back(bodies[i].vel.x);
                bodyData.push_back(bodies[i].vel.y);
                bodyData.push_back(bodies[i].vel.z);
            }
            print_results(bodyData, "Bodies");
        }

        // Validation: check that simulation produces finite, reasonable values
        if (validate_i) {
            printf("Validating simulation results...\n");

            if (validateSimulation(bodies)) {
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
                ret = 0;
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }

    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return ret;
}
