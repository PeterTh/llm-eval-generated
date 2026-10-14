#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
// Bodies processed together in the force kernel. Each lane keeps its own accumulator and
// sums over j in the same order as the serial code, so results are bit-identical.
constexpr size_t FORCE_BLOCK = 8;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};
static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be tightly packed for MPI transfers");

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// Half-open range [begin, end) of body indices owned by one rank
struct Range {
    size_t begin;
    size_t end;
};

Range blockRange(size_t n, int rank, int size) {
    return { n * rank / size, n * (rank + 1) / size };
}

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

// Updates velocities of bodies [lo, hi) using the positions of all bodies.
// The fused multiply-adds are written out explicitly so every build produces the same bits.
void computeForces(const std::vector<Vec3>& pos, std::vector<Vec3>& vel, size_t lo, size_t hi) {
    const size_t n = pos.size();

    for (size_t ib = lo; ib < hi; ib += FORCE_BLOCK) {
        const size_t count = std::min(FORCE_BLOCK, hi - ib);

        double xi[FORCE_BLOCK], yi[FORCE_BLOCK], zi[FORCE_BLOCK];
        double Fx[FORCE_BLOCK] = {}, Fy[FORCE_BLOCK] = {}, Fz[FORCE_BLOCK] = {};
        // Lanes past the end of the block repeat the last body; their results are discarded
        for (size_t k = 0; k < FORCE_BLOCK; ++k) {
            const Vec3& p = pos[ib + std::min(k, count - 1)];
            xi[k] = p.x;
            yi[k] = p.y;
            zi[k] = p.z;
        }

        for (size_t j = 0; j < n; ++j) {
            const double xj = pos[j].x;
            const double yj = pos[j].y;
            const double zj = pos[j].z;
            for (size_t k = 0; k < FORCE_BLOCK; ++k) {
                const double dx = xj - xi[k];
                const double dy = yj - yi[k];
                const double dz = zj - zi[k];
                const double distSqr = std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx[k] = std::fma(dx, invDist3, Fx[k]);
                Fy[k] = std::fma(dy, invDist3, Fy[k]);
                Fz[k] = std::fma(dz, invDist3, Fz[k]);
            }
        }

        for (size_t k = 0; k < count; ++k) {
            vel[ib + k].x += DT * Fx[k];
            vel[ib + k].y += DT * Fy[k];
            vel[ib + k].z += DT * Fz[k];
        }
    }
}

void integrateBodies(std::vector<Vec3>& pos, const std::vector<Vec3>& vel, size_t lo, size_t hi) {
    for (size_t i = lo; i < hi; ++i) {
        pos[i].x = std::fma(vel[i].x, DT, pos[i].x);
        pos[i].y = std::fma(vel[i].y, DT, pos[i].y);
        pos[i].z = std::fma(vel[i].z, DT, pos[i].z);
    }
}

// This rank's share of the total energy. Kinetic terms cover the owned bodies; pair terms
// cover rows i = rank, rank + size, ... so the triangular pair loop is spread evenly.
double partialEnergy(const std::vector<Vec3>& pos, const std::vector<Vec3>& vel, Range mine, int rank, int size) {
    const size_t n = pos.size();
    double energy = 0.0;

    // Kinetic energy (assuming unit mass)
    for (size_t i = mine.begin; i < mine.end; ++i) {
        energy += 0.5 * (vel[i].x * vel[i].x +
                         vel[i].y * vel[i].y +
                         vel[i].z * vel[i].z);
    }

    // Potential energy (assuming unit mass for all bodies)
    for (size_t i = rank; i < n; i += size) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = pos[j].x - pos[i].x;
            const double dy = pos[j].y - pos[i].y;
            const double dz = pos[j].z - pos[i].z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Vec3>& pos, const std::vector<Vec3>& vel) {
    for (size_t i = 0; i < pos.size(); ++i) {
        // Check for NaN or Inf values
        if (!std::isfinite(pos[i].x) || !std::isfinite(pos[i].y) || !std::isfinite(pos[i].z) ||
            !std::isfinite(vel[i].x) || !std::isfinite(vel[i].y) || !std::isfinite(vel[i].z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(pos[i].x) > maxPos || std::abs(pos[i].y) > maxPos || std::abs(pos[i].z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(vel[i].x) > maxVel || std::abs(vel[i].y) > maxVel || std::abs(vel[i].z) > maxVel) {
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

int run(int argc, char** argv, int rank, int size) {
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (numBodies < 0) {
        if (rank == 0) printf("Number of bodies must be non-negative\n");
        return 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t n = static_cast<size_t>(numBodies);
    const Range mine = blockRange(n, rank, size);

    // Every rank generates the same initial state, so no initial scatter is needed
    std::vector<Body> init(n);
    randomizeBodies(init);
    std::vector<Vec3> pos(n), vel(n);
    for (size_t i = 0; i < n; ++i) {
        pos[i] = init[i].pos;
        vel[i] = init[i].vel;
    }

    // Per-rank counts and displacements in doubles (3 per body) for the position exchange
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        const Range range = blockRange(n, r, size);
        counts[r] = static_cast<int>(3 * (range.end - range.begin));
        displs[r] = static_cast<int>(3 * range.begin);
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(pos, vel, mine.begin, mine.end);
        integrateBodies(pos, vel, mine.begin, mine.end);
        // Share the updated positions of owned bodies with all ranks
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, pos.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", static_cast<long>((end - start) * 1000.0));
    }

    // Velocities are only held by their owner; gather them on rank 0 for output and validation
    if (printResults || validate) {
        if (rank == 0) {
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, vel.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(vel.data() + mine.begin, counts[rank], MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(n * 6);
        for (size_t i = 0; i < n; ++i) {
            bodyData.push_back(pos[i].x);
            bodyData.push_back(pos[i].y);
            bodyData.push_back(pos[i].z);
            bodyData.push_back(vel[i].x);
            bodyData.push_back(vel[i].y);
            bodyData.push_back(vel[i].z);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        int ok = 1;
        if (rank == 0) {
            printf("Validating simulation results...\n");
            ok = validateSimulation(pos, vel) ? 1 : 0;
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (ok) {
            // Report final energy for reference
            double localEnergy = partialEnergy(pos, vel, mine, rank, size);
            double finalEnergy = 0.0;
            MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            return 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rc = run(argc, argv, rank, size);

    MPI_Finalize();
    return rc;
}
