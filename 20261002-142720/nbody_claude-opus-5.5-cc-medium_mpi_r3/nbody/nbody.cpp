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

// Distributed state: every rank holds all positions (interleaved x,y,z, replicated
// via Allgatherv each step) and the velocities of its own contiguous block of bodies.
struct Decomp {
    int rank = 0, size = 1;
    std::vector<int> counts, displs;  // in bodies
    std::vector<int> counts3, displs3;  // in doubles (3 per body)
    std::vector<int> counts6, displs6;  // in doubles (6 per body)
    int lo = 0, hi = 0;

    Decomp(int n, int r, int s) : rank(r), size(s), counts(s), displs(s), counts3(s), displs3(s), counts6(s), displs6(s) {
        const int base = n / s, rem = n % s;
        int off = 0;
        for (int p = 0; p < s; ++p) {
            counts[p] = base + (p < rem ? 1 : 0);
            displs[p] = off;
            counts3[p] = 3 * counts[p];
            displs3[p] = 3 * off;
            counts6[p] = 6 * counts[p];
            displs6[p] = 6 * off;
            off += counts[p];
        }
        lo = displs[rank];
        hi = lo + counts[rank];
    }
};

constexpr int TILE = 8;

// Floating-point contraction is disabled for this file (see CMakeLists.txt); the FMAs below
// are spelled out explicitly to reproduce the reference build's rounding bit-for-bit, no
// matter how the compiler vectorizes the restructured loops.

// Computes the force on local bodies [lo, hi) from all n bodies and updates their velocity.
// The per-body summation order over j matches the sequential code exactly; vectorization
// is done across i (TILE target bodies at once), which keeps results bit-identical. A partial
// last tile duplicates its final body and discards the extra lanes, so every body goes
// through the same (vectorized) code path.
void computeForces(const double* __restrict pos, double* __restrict vel, int lo, int hi, int n) {
    for (int i = lo; i < hi; i += TILE) {
        const int cnt = (hi - i < TILE) ? hi - i : TILE;
        double xi[TILE], yi[TILE], zi[TILE];
        double Fx[TILE], Fy[TILE], Fz[TILE];
        for (int k = 0; k < TILE; ++k) {
            const int b = i + (k < cnt ? k : cnt - 1);
            xi[k] = pos[3 * b + 0];
            yi[k] = pos[3 * b + 1];
            zi[k] = pos[3 * b + 2];
            Fx[k] = 0.0;
            Fy[k] = 0.0;
            Fz[k] = 0.0;
        }
        for (int j = 0; j < n; ++j) {
            const double xj = pos[3 * j + 0];
            const double yj = pos[3 * j + 1];
            const double zj = pos[3 * j + 2];
#pragma GCC ivdep
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
        for (int k = 0; k < cnt; ++k) {
            const int li = i + k - lo;
            vel[3 * li + 0] += DT * Fx[k];
            vel[3 * li + 1] += DT * Fy[k];
            vel[3 * li + 2] += DT * Fz[k];
        }
    }
}

// Advances local bodies' positions (in the global position array) using local velocities.
void integrateBodies(double* __restrict pos, const double* __restrict vel, int lo, int hi) {
    const int m = 3 * (hi - lo);
    double* p = pos + 3 * lo;
    for (int k = 0; k < m; ++k) {
        p[k] = std::fma(vel[k], DT, p[k]);
    }
}

// Collective: every rank holds the full body set. Rows of the i < j triangle are split
// into contiguous blocks with roughly equal pair counts. Each rank accumulates in the
// original order (rank 0 starts from the kinetic energy), and the partial sums are
// reduced onto rank 0; with one rank this is exactly the sequential computation.
double computeTotalEnergy(const std::vector<Body>& bodies, int rank, int size) {
    const size_t n = bodies.size();
    const double totalPairs = 0.5 * (double)n * ((double)n - 1.0);

    // First row i such that the pairs in rows [0, i) reach fraction p/size of the total
    auto rowBoundary = [&](int p) -> size_t {
        if (p <= 0) return 0;
        if (p >= size) return n;
        const double target = totalPairs * p / size;
        size_t i = 0;
        double acc = 0.0;
        while (i < n && acc < target) {
            acc += (double)(n - 1 - i);
            ++i;
        }
        return i;
    };
    const size_t rlo = rowBoundary(rank), rhi = rowBoundary(rank + 1);

    double energy = 0.0;
    if (rank == 0) {
        // Kinetic energy (assuming unit mass)
        for (const auto& body : bodies) {
            const double v2 = std::fma(body.vel.z, body.vel.z, std::fma(body.vel.x, body.vel.x, body.vel.y * body.vel.y));
            energy = std::fma(0.5, v2, energy);
        }
    }
    
    // Potential energy (assuming unit mass for all bodies)
    for (size_t i = rlo; i < rhi; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    if (size == 1) return energy;
    std::vector<double> parts(rank == 0 ? size : 0);
    MPI_Gather(&energy, 1, MPI_DOUBLE, parts.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double total = 0.0;
    if (rank == 0) {
        total = parts[0];
        for (int p = 1; p < size; ++p) total += parts[p];
    }
    return total;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies, bool verbose) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            if (verbose) printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            if (verbose) printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            if (verbose) printf("Validation failed: body velocity exceeds reasonable bounds\n");
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
    const bool root = (rank == 0);

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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (root) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Initialize bodies (deterministic, so every rank generates the same set)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    const Decomp d(numBodies, rank, size);
    std::vector<double> pos(3 * (size_t)numBodies);
    std::vector<double> vel(3 * (size_t)(d.hi - d.lo));
    for (int i = 0; i < numBodies; ++i) {
        pos[3 * i + 0] = bodies[i].pos.x;
        pos[3 * i + 1] = bodies[i].pos.y;
        pos[3 * i + 2] = bodies[i].pos.z;
    }
    for (int i = d.lo; i < d.hi; ++i) {
        const int li = i - d.lo;
        vel[3 * li + 0] = bodies[i].vel.x;
        vel[3 * li + 1] = bodies[i].vel.y;
        vel[3 * li + 2] = bodies[i].vel.z;
    }
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(pos.data(), vel.data(), d.lo, d.hi, numBodies);
        integrateBodies(pos.data(), vel.data(), d.lo, d.hi);
        // Replicate updated positions to all ranks
        if (size > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pos.data(), d.counts3.data(), d.displs3.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (root) printf("Simulation time: %ld ms\n", duration.count());

    if (!printResults && !validate) {
        MPI_Finalize();
        return 0;
    }

    // Collect final state (positions are already replicated; gather velocities)
    std::vector<double> allVel(3 * (size_t)numBodies);
    MPI_Allgatherv(vel.data(), d.counts3[rank], MPI_DOUBLE, allVel.data(), d.counts3.data(), d.displs3.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos = Vec3(pos[3 * i + 0], pos[3 * i + 1], pos[3 * i + 2]);
        bodies[i].vel = Vec3(allVel[3 * i + 0], allVel[3 * i + 1], allVel[3 * i + 2]);
    }
    
    // Print results for external validation
    if (printResults && root) {
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
    int rc = 0;
    if (validate) {
        if (root) printf("Validating simulation results...\n");
        
        // Every rank holds identical state, so all ranks reach the same verdict
        if (validateSimulation(bodies, root)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies, rank, size);
            if (root) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (root) printf("Validation: FAILED\n");
            rc = 1;
        }
    }
    
    MPI_Finalize();
    return rc;
}
