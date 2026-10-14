#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
// Bodies per tile in computeForces; keeps the tile's force accumulators in L1
constexpr size_t TILE = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// Contiguous block of bodies owned by one rank
struct Range {
    size_t begin;
    size_t count;
};

Range blockRange(size_t n, int rank, int size) {
    const size_t r = static_cast<size_t>(rank);
    const size_t p = static_cast<size_t>(size);
    const size_t base = n / p;
    const size_t rem = n % p;
    return {r * base + std::min(r, rem), base + (r < rem ? 1 : 0)};
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

// Velocity update kept as a separate multiply and add. GCC would otherwise fuse it into an
// FMA, which changes the last bits relative to the serial reference (the position update
// and force accumulation are still fused, exactly as in the serial code).
__attribute__((optimize("fp-contract=off"))) void applyForces(Vec3* vel, const double* Fx, const double* Fy,
                                                             const double* Fz, size_t len) {
    for (size_t k = 0; k < len; ++k) {
        vel[k].x += DT * Fx[k];
        vel[k].y += DT * Fy[k];
        vel[k].z += DT * Fz[k];
    }
}

// Updates velocities of the local bodies [begin, begin + count) against all n bodies.
// Each body's force accumulates over j = 0..n-1 in the same order as the serial loop,
// so results are bit-identical for any number of ranks. The inner loop runs over the
// local bodies, which vectorizes without reordering any floating-point sum.
void computeForces(const Vec3* pos, size_t n, size_t begin, size_t count, Vec3* vel) {
    std::vector<double> xi(count), yi(count), zi(count);
    for (size_t i = 0; i < count; ++i) {
        xi[i] = pos[begin + i].x;
        yi[i] = pos[begin + i].y;
        zi[i] = pos[begin + i].z;
    }

    for (size_t tile = 0; tile < count; tile += TILE) {
        const size_t len = std::min(TILE, count - tile);
        double Fx[TILE] = {}, Fy[TILE] = {}, Fz[TILE] = {};

        for (size_t j = 0; j < n; ++j) {
            const double xj = pos[j].x;
            const double yj = pos[j].y;
            const double zj = pos[j].z;
            for (size_t k = 0; k < len; ++k) {
                const double dx = xj - xi[tile + k];
                const double dy = yj - yi[tile + k];
                const double dz = zj - zi[tile + k];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx[k] += dx * invDist3;
                Fy[k] += dy * invDist3;
                Fz[k] += dz * invDist3;
            }
        }

        applyForces(vel + tile, Fx, Fy, Fz, len);
    }
}

void integrateBodies(Vec3* pos, const Vec3* vel, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        pos[i].x += vel[i].x * DT;
        pos[i].y += vel[i].y * DT;
        pos[i].z += vel[i].z * DT;
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

int runBenchmark(int argc, char** argv, int rank, int size) {
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t n = static_cast<size_t>(numBodies);

    // Every rank builds the same initial state; each keeps only its own block
    std::vector<Body> bodies(n);
    randomizeBodies(bodies);

    const Range mine = blockRange(n, rank, size);
    // Per-rank element counts and displacements in doubles (3 per body) for the collectives
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        const Range range = blockRange(n, r, size);
        counts[r] = static_cast<int>(3 * range.count);
        displs[r] = static_cast<int>(3 * range.begin);
    }

    // Positions of all bodies are replicated on every rank; velocities are local only
    std::vector<Vec3> pos(n);
    for (size_t i = 0; i < n; ++i) {
        pos[i] = bodies[i].pos;
    }
    std::vector<Vec3> vel(mine.count);
    for (size_t i = 0; i < mine.count; ++i) {
        vel[i] = bodies[mine.begin + i].vel;
    }
    double* posData = reinterpret_cast<double*>(pos.data());

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(pos.data(), n, mine.begin, mine.count, vel.data());
        integrateBodies(pos.data() + mine.begin, vel.data(), mine.count);
        // Share the updated positions of every block so all ranks see the full state
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, posData, counts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long local_ms = duration.count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", max_ms);
    }

    if (!printResults && !validate) {
        return 0;
    }

    // Gather the full velocity state on rank 0 for output and validation
    std::vector<Vec3> velAll(rank == 0 ? n : 0);
    MPI_Gatherv(vel.data(), static_cast<int>(3 * mine.count), MPI_DOUBLE,
                reinterpret_cast<double*>(velAll.data()), counts.data(), displs.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        for (size_t i = 0; i < n; ++i) {
            bodies[i].pos = pos[i];
            bodies[i].vel = velAll[i];
        }
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
        int failed = 0;
        if (rank == 0) {
            printf("Validating simulation results...\n");

            if (validateSimulation(bodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                failed = 1;
            }
        }
        // Share the verdict so every rank exits with the same status
        MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return failed;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int status = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return status;
}
