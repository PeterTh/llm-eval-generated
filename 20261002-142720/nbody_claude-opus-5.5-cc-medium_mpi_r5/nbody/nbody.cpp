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

// Distributed state: every rank holds all positions (interleaved x,y,z) and
// owns a contiguous block [lo, hi) of bodies whose velocities it updates.
struct DistState {
    int n = 0;
    int lo = 0, hi = 0;
    std::vector<double> pos;      // 3*n, all bodies
    std::vector<double> vel;      // 3*(hi-lo), owned bodies
    std::vector<int> counts;      // per-rank element counts (3*bodies)
    std::vector<int> displs;      // per-rank element displacements
};

constexpr int IBLOCK = 16;

// Floating-point contraction is disabled at build time; FMAs are explicit and
// mirror the contraction the reference (serial) build performs.
// Forces for owned bodies; per-body accumulation order over j is identical to
// the serial code, the i-block is processed in SIMD lanes.
void computeForces(DistState& s) {
    const int n = s.n;
    const double* __restrict pos = s.pos.data();
    double* __restrict vel = s.vel.data();
    int i0 = s.lo;
    for (; i0 + IBLOCK <= s.hi; i0 += IBLOCK) {
        double xi[IBLOCK], yi[IBLOCK], zi[IBLOCK];
        double Fx[IBLOCK], Fy[IBLOCK], Fz[IBLOCK];
        for (int k = 0; k < IBLOCK; ++k) {
            xi[k] = pos[3 * (i0 + k) + 0];
            yi[k] = pos[3 * (i0 + k) + 1];
            zi[k] = pos[3 * (i0 + k) + 2];
            Fx[k] = 0.0; Fy[k] = 0.0; Fz[k] = 0.0;
        }
        for (int j = 0; j < n; ++j) {
            const double xj = pos[3 * j + 0];
            const double yj = pos[3 * j + 1];
            const double zj = pos[3 * j + 2];
#pragma GCC ivdep
            for (int k = 0; k < IBLOCK; ++k) {
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
        for (int k = 0; k < IBLOCK; ++k) {
            const int li = i0 + k - s.lo;
            vel[3 * li + 0] += DT * Fx[k];
            vel[3 * li + 1] += DT * Fy[k];
            vel[3 * li + 2] += DT * Fz[k];
        }
    }
    for (int i = i0; i < s.hi; ++i) {
        const double xi = pos[3 * i + 0], yi = pos[3 * i + 1], zi = pos[3 * i + 2];
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        for (int j = 0; j < n; ++j) {
            const double dx = pos[3 * j + 0] - xi;
            const double dy = pos[3 * j + 1] - yi;
            const double dz = pos[3 * j + 2] - zi;
            const double distSqr = std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx = std::fma(dx, invDist3, Fx);
            Fy = std::fma(dy, invDist3, Fy);
            Fz = std::fma(dz, invDist3, Fz);
        }
        const int li = i - s.lo;
        vel[3 * li + 0] += DT * Fx;
        vel[3 * li + 1] += DT * Fy;
        vel[3 * li + 2] += DT * Fz;
    }
}

// Integrate owned bodies, then share updated positions with all ranks.
void integrateBodies(DistState& s) {
    double* __restrict p = s.pos.data() + 3 * static_cast<size_t>(s.lo);
    const double* __restrict v = s.vel.data();
    const int m = 3 * (s.hi - s.lo);
    for (int k = 0; k < m; ++k) {
        p[k] = std::fma(v[k], DT, p[k]);
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   s.pos.data(), s.counts.data(), s.displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        const double v2 = std::fma(body.vel.z, body.vel.z,
                                   std::fma(body.vel.x, body.vel.x, body.vel.y * body.vel.y));
        energy = std::fma(v2, 0.5, energy);
    }
    
    // Potential energy (assuming unit mass for all bodies)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING);
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
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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
    
    // Initialize bodies (deterministic, identical on every rank)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Block decomposition of bodies across ranks
    DistState s;
    s.n = numBodies;
    s.counts.resize(nranks);
    s.displs.resize(nranks);
    {
        const int base = numBodies / nranks, rem = numBodies % nranks;
        int off = 0;
        for (int r = 0; r < nranks; ++r) {
            const int cnt = base + (r < rem ? 1 : 0);
            if (r == rank) { s.lo = off; s.hi = off + cnt; }
            s.counts[r] = 3 * cnt;
            s.displs[r] = 3 * off;
            off += cnt;
        }
    }
    s.pos.resize(3 * static_cast<size_t>(numBodies));
    for (int i = 0; i < numBodies; ++i) {
        s.pos[3 * i + 0] = bodies[i].pos.x;
        s.pos[3 * i + 1] = bodies[i].pos.y;
        s.pos[3 * i + 2] = bodies[i].pos.z;
    }
    s.vel.resize(3 * static_cast<size_t>(s.hi - s.lo));
    for (int i = s.lo; i < s.hi; ++i) {
        s.vel[3 * (i - s.lo) + 0] = bodies[i].vel.x;
        s.vel[3 * (i - s.lo) + 1] = bodies[i].vel.y;
        s.vel[3 * (i - s.lo) + 2] = bodies[i].vel.z;
    }
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(s);
        integrateBodies(s);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count(), global_ms = 0;
    MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Collect velocities on rank 0 (positions are already replicated)
    std::vector<double> allVel;
    if (rank == 0) allVel.resize(3 * static_cast<size_t>(numBodies));
    MPI_Gatherv(s.vel.data(), 3 * (s.hi - s.lo), MPI_DOUBLE,
                allVel.data(), s.counts.data(), s.displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos = Vec3(s.pos[3 * i + 0], s.pos[3 * i + 1], s.pos[3 * i + 2]);
            bodies[i].vel = Vec3(allVel[3 * i + 0], allVel[3 * i + 1], allVel[3 * i + 2]);
        }

        printf("Simulation time: %ld ms\n", global_ms);
        
        // Print results for external validation
        if (printResults) {
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
            printf("Validating simulation results...\n");
            
            if (validateSimulation(bodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        fflush(stdout);
    }
    
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
