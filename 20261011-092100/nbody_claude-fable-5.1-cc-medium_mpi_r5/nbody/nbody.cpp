#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// MPI-parallel N-body simulation.
//
// Decomposition: the N bodies are split into contiguous, (almost) equal-sized
// blocks, one per rank. Every rank keeps a full copy of all body positions
// (needed for the all-pairs force evaluation) but only owns the velocities of
// its own block. Each step a rank computes the forces on its own bodies
// against all N positions, integrates its own block, and then the updated
// positions are exchanged with a single MPI_Allgatherv. Because the force
// evaluation per body is O(N) regardless of index, a plain contiguous block
// decomposition is perfectly load balanced.
//
// The per-body arithmetic (inner j-loop order, operations, softening) is kept
// identical to the original serial code so that results are bitwise equal.
// The original build (-O3 -march=native, default -ffp-contract=fast) fuses
// specific multiply-adds into FMAs; those fusions are written out explicitly
// with std::fma here and implicit contraction is disabled in CMakeLists.txt,
// so the floating-point results do not depend on how the compiler happens to
// contract the restructured loops.
// The only restructuring is that a tile of TILE bodies (i) is processed
// together inside the j-loop, which lets the compiler vectorise across i
// without changing the sequential accumulation order of any individual body.

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int TILE = 16;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// Structure-of-arrays state. Positions are global (size n); velocities are
// only valid for the locally owned block [begin, end).
struct State {
    int n = 0;
    int begin = 0;
    int end = 0;
    std::vector<double> px, py, pz;  // global positions
    std::vector<double> vx, vy, vz;  // local velocities (size end - begin)
    std::vector<int> counts, displs; // block sizes / offsets for Allgatherv
};

static void blockRange(int n, int size, int rank, int& begin, int& end) {
    const int base = n / size;
    const int rem = n % size;
    begin = rank * base + std::min(rank, rem);
    end = begin + base + (rank < rem ? 1 : 0);
}

// Generates exactly the same sequence as the original serial randomizeBodies,
// but keeps only the locally owned velocities.
static void randomizeBodies(State& s, unsigned int seed = 42) {
    for (int i = 0; i < s.n; ++i) {
        const double x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double vx = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double vy = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double vz = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        s.px[i] = x;
        s.py[i] = y;
        s.pz[i] = z;
        if (i >= s.begin && i < s.end) {
            s.vx[i - s.begin] = vx;
            s.vy[i - s.begin] = vy;
            s.vz[i - s.begin] = vz;
        }
    }
}

// Computes forces on the local block of bodies against all n bodies and
// applies the velocity kick. Arithmetic order per body matches the original.
static void computeForces(State& s) {
    const int n = s.n;
    const double* __restrict px = s.px.data();
    const double* __restrict py = s.py.data();
    const double* __restrict pz = s.pz.data();

    for (int i0 = s.begin; i0 < s.end; i0 += TILE) {
        const int nb = std::min(TILE, s.end - i0);

        alignas(64) double xi[TILE], yi[TILE], zi[TILE];
        alignas(64) double Fx[TILE], Fy[TILE], Fz[TILE];
        for (int k = 0; k < TILE; ++k) {
            // Pad the tail of the tile with the last valid body; results for
            // padded lanes are simply discarded.
            const int idx = i0 + std::min(k, nb - 1);
            xi[k] = px[idx];
            yi[k] = py[idx];
            zi[k] = pz[idx];
            Fx[k] = 0.0;
            Fy[k] = 0.0;
            Fz[k] = 0.0;
        }

        for (int j = 0; j < n; ++j) {
            const double xj = px[j];
            const double yj = py[j];
            const double zj = pz[j];
            for (int k = 0; k < TILE; ++k) {
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

        const int l0 = i0 - s.begin;
        for (int k = 0; k < nb; ++k) {
            s.vx[l0 + k] += DT * Fx[k];
            s.vy[l0 + k] += DT * Fy[k];
            s.vz[l0 + k] += DT * Fz[k];
        }
    }
}

static void integrateBodies(State& s) {
    const int local = s.end - s.begin;
    double* __restrict px = s.px.data() + s.begin;
    double* __restrict py = s.py.data() + s.begin;
    double* __restrict pz = s.pz.data() + s.begin;
    const double* __restrict vx = s.vx.data();
    const double* __restrict vy = s.vy.data();
    const double* __restrict vz = s.vz.data();
    for (int i = 0; i < local; ++i) {
        px[i] = std::fma(vx[i], DT, px[i]);
        py[i] = std::fma(vy[i], DT, py[i]);
        pz[i] = std::fma(vz[i], DT, pz[i]);
    }
}

// Exchange the updated positions of every block with all ranks.
static void exchangePositions(State& s) {
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, s.px.data(), s.counts.data(), s.displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, s.py.data(), s.counts.data(), s.displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, s.pz.data(), s.counts.data(), s.displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
}

// Total energy (unit masses). Kinetic part over local velocities, potential
// part over the local block of i against j > i, reduced over all ranks.
static double computeTotalEnergy(const State& s) {
    double energy = 0.0;
    const int n = s.n;
    const int local = s.end - s.begin;

    for (int i = 0; i < local; ++i) {
        const double v2 = std::fma(s.vz[i], s.vz[i], std::fma(s.vx[i], s.vx[i], s.vy[i] * s.vy[i]));
        energy = std::fma(0.5, v2, energy);
    }

    for (int i = s.begin; i < s.end; ++i) {
        const double xi = s.px[i], yi = s.py[i], zi = s.pz[i];
        double pe = 0.0;
        for (int j = i + 1; j < n; ++j) {
            const double dx = s.px[j] - xi;
            const double dy = s.py[j] - yi;
            const double dz = s.pz[j] - zi;
            const double dist = std::sqrt(std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING);
            pe -= 1.0 / dist;
        }
        energy += pe;
    }

    double total = 0.0;
    MPI_Allreduce(&energy, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return total;
}

// Validate that simulation produces finite, reasonable values (rank 0, on the
// fully gathered body array).
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

// Gather the complete body state (positions + velocities) onto rank 0.
static std::vector<Body> gatherBodies(const State& s, int rank) {
    std::vector<double> vx, vy, vz;
    if (rank == 0) {
        vx.resize(s.n);
        vy.resize(s.n);
        vz.resize(s.n);
    }
    const int local = s.end - s.begin;
    MPI_Gatherv(s.vx.data(), local, MPI_DOUBLE, vx.data(), s.counts.data(), s.displs.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    MPI_Gatherv(s.vy.data(), local, MPI_DOUBLE, vy.data(), s.counts.data(), s.displs.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    MPI_Gatherv(s.vz.data(), local, MPI_DOUBLE, vz.data(), s.counts.data(), s.displs.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    std::vector<Body> bodies;
    if (rank == 0) {
        bodies.resize(s.n);
        for (int i = 0; i < s.n; ++i) {
            bodies[i].pos = Vec3(s.px[i], s.py[i], s.pz[i]);
            bodies[i].vel = Vec3(vx[i], vy[i], vz[i]);
        }
    }
    return bodies;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank sees the same argv).
    int parseStatus = -1;  // -1: continue, otherwise exit with this code
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
            parseStatus = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
            break;
        }
    }
    if (parseStatus >= 0) {
        MPI_Finalize();
        return parseStatus;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    // Set up decomposition and initialize bodies
    State s;
    s.n = numBodies;
    blockRange(s.n, size, rank, s.begin, s.end);
    s.px.assign(s.n, 0.0);
    s.py.assign(s.n, 0.0);
    s.pz.assign(s.n, 0.0);
    s.vx.assign(s.end - s.begin, 0.0);
    s.vy.assign(s.end - s.begin, 0.0);
    s.vz.assign(s.end - s.begin, 0.0);
    s.counts.resize(size);
    s.displs.resize(size);
    for (int r = 0; r < size; ++r) {
        int b, e;
        blockRange(s.n, size, r, b, e);
        s.counts[r] = e - b;
        s.displs[r] = b;
    }
    randomizeBodies(s);

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(s);
        integrateBodies(s);
        exchangePositions(s);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", maxDuration);
    }

    std::vector<Body> bodies;
    if (printResults || validate) {
        bodies = gatherBodies(s, rank);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve((size_t)numBodies * 6);
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
    int exitCode = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        int ok = (rank == 0) ? (validateSimulation(bodies) ? 1 : 0) : 1;
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (ok) {
            // Report final energy for reference (computed collectively)
            double finalEnergy = computeTotalEnergy(s);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Finalize();
    return exitCode;
}
