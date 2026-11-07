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

// Distributed simulation state.
//
// Every rank holds the positions of *all* bodies (needed for the O(n^2)
// force evaluation) in structure-of-arrays form, plus an interleaved copy
// used for the per-step MPI_Allgatherv.  Each rank owns a contiguous block
// of bodies [offset, offset + count) and keeps only the velocities of those.
struct DistributedState {
    int n = 0;            // global number of bodies
    int rank = 0, size = 1;
    int offset = 0;       // first body owned by this rank
    int count = 0;        // number of bodies owned by this rank
    std::vector<int> counts3, displs3;  // Allgatherv layout (in doubles, 3 per body)

    std::vector<double> posAll;         // interleaved x,y,z for all bodies (3n)
    std::vector<double> px, py, pz;     // SoA copies of posAll
    std::vector<double> vx, vy, vz;     // local velocities (count)
};

static void partition(int n, int size, int rank, int& offset, int& count) {
    const int base = n / size;
    const int rem = n % size;
    count = base + (rank < rem ? 1 : 0);
    offset = rank * base + (rank < rem ? rank : rem);
}

static void unpackPositions(DistributedState& s) {
    const int n = s.n;
    const double* __restrict p = s.posAll.data();
    double* __restrict px = s.px.data();
    double* __restrict py = s.py.data();
    double* __restrict pz = s.pz.data();
    for (int i = 0; i < n; ++i) {
        px[i] = p[3 * i];
        py[i] = p[3 * i + 1];
        pz[i] = p[3 * i + 2];
    }
}

// Force kernel tile width: the inner loop over a tile of TILE target bodies
// vectorizes across targets, while the accumulation for each target still
// runs over j = 0..n-1 in the original order, so the results are bit-identical
// to the sequential implementation.
constexpr int TILE = 16;

static void computeForcesTile(const double* __restrict px, const double* __restrict py,
                              const double* __restrict pz, int n,
                              const double* __restrict tx, const double* __restrict ty,
                              const double* __restrict tz,
                              double* __restrict fx, double* __restrict fy, double* __restrict fz) {
    double Fx[TILE], Fy[TILE], Fz[TILE];
    for (int k = 0; k < TILE; ++k) {
        Fx[k] = 0.0;
        Fy[k] = 0.0;
        Fz[k] = 0.0;
    }
    for (int j = 0; j < n; ++j) {
        const double xj = px[j], yj = py[j], zj = pz[j];
        for (int k = 0; k < TILE; ++k) {
            const double dx = xj - tx[k];
            const double dy = yj - ty[k];
            const double dz = zj - tz[k];
            // Explicit fused multiply-adds reproduce the contraction pattern the
            // compiler applies to the scalar reference loop, keeping results
            // bit-identical to the original implementation.
            const double distSqr = std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx[k] = std::fma(dx, invDist3, Fx[k]);
            Fy[k] = std::fma(dy, invDist3, Fy[k]);
            Fz[k] = std::fma(dz, invDist3, Fz[k]);
        }
    }
    for (int k = 0; k < TILE; ++k) {
        fx[k] = Fx[k];
        fy[k] = Fy[k];
        fz[k] = Fz[k];
    }
}

void computeForces(DistributedState& s) {
    const int n = s.n;
    const double* px = s.px.data();
    const double* py = s.py.data();
    const double* pz = s.pz.data();

    alignas(64) double tx[TILE], ty[TILE], tz[TILE];
    alignas(64) double fx[TILE], fy[TILE], fz[TILE];

    for (int i0 = 0; i0 < s.count; i0 += TILE) {
        const int m = (s.count - i0 < TILE) ? (s.count - i0) : TILE;
        for (int k = 0; k < TILE; ++k) {
            const int gi = s.offset + i0 + (k < m ? k : m - 1);  // pad with last valid body
            tx[k] = px[gi];
            ty[k] = py[gi];
            tz[k] = pz[gi];
        }
        computeForcesTile(px, py, pz, n, tx, ty, tz, fx, fy, fz);
        for (int k = 0; k < m; ++k) {
            s.vx[i0 + k] += DT * fx[k];
            s.vy[i0 + k] += DT * fy[k];
            s.vz[i0 + k] += DT * fz[k];
        }
    }
}

void integrateBodies(DistributedState& s) {
    double* p = s.posAll.data() + 3 * static_cast<size_t>(s.offset);
    for (int i = 0; i < s.count; ++i) {
        // Fused as in the reference build (pos += vel * DT contracted to an FMA).
        p[3 * i] = std::fma(s.vx[i], DT, p[3 * i]);
        p[3 * i + 1] = std::fma(s.vy[i], DT, p[3 * i + 1]);
        p[3 * i + 2] = std::fma(s.vz[i], DT, p[3 * i + 2]);
    }
}

// Exchange the locally-updated positions so every rank sees the full state.
void exchangePositions(DistributedState& s) {
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, s.posAll.data(), s.counts3.data(),
                   s.displs3.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    unpackPositions(s);
}

// Total energy, computed cooperatively: the pair loop is distributed over
// ranks in a cyclic fashion for load balance, then reduced to rank 0.
double computeTotalEnergy(const std::vector<Body>& bodies, int rank, int size) {
    const size_t n = bodies.size();
    double local = 0.0;

    // Potential energy (assuming unit mass for all bodies)
    for (size_t i = rank; i < n; i += size) {
        double rowSum = 0.0;
        const double xi = bodies[i].pos.x, yi = bodies[i].pos.y, zi = bodies[i].pos.z;
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - xi;
            const double dy = bodies[j].pos.y - yi;
            const double dz = bodies[j].pos.z - zi;
            const double dist = std::sqrt(std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING);
            rowSum += 1.0 / dist;
        }
        local -= rowSum;
    }

    double energy = 0.0;
    MPI_Reduce(&local, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Kinetic energy (assuming unit mass)
        double kinetic = 0.0;
        for (const auto& body : bodies) {
            kinetic += 0.5 * (body.vel.x * body.vel.x +
                              body.vel.y * body.vel.y +
                              body.vel.z * body.vel.z);
        }
        energy += kinetic;
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identically on every rank)
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

    // Initialize bodies: the deterministic generator is cheap, so every rank
    // produces the full initial state itself rather than scattering it.
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    DistributedState s;
    s.n = numBodies;
    s.rank = rank;
    s.size = size;
    partition(numBodies, size, rank, s.offset, s.count);
    s.counts3.resize(size);
    s.displs3.resize(size);
    for (int r = 0; r < size; ++r) {
        int off, cnt;
        partition(numBodies, size, r, off, cnt);
        s.counts3[r] = 3 * cnt;
        s.displs3[r] = 3 * off;
    }
    s.posAll.resize(3 * static_cast<size_t>(numBodies));
    s.px.resize(numBodies);
    s.py.resize(numBodies);
    s.pz.resize(numBodies);
    s.vx.resize(s.count);
    s.vy.resize(s.count);
    s.vz.resize(s.count);
    for (int i = 0; i < numBodies; ++i) {
        s.posAll[3 * i] = bodies[i].pos.x;
        s.posAll[3 * i + 1] = bodies[i].pos.y;
        s.posAll[3 * i + 2] = bodies[i].pos.z;
    }
    for (int i = 0; i < s.count; ++i) {
        s.vx[i] = bodies[s.offset + i].vel.x;
        s.vy[i] = bodies[s.offset + i].vel.y;
        s.vz[i] = bodies[s.offset + i].vel.z;
    }
    unpackPositions(s);

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

    if (rank == 0) printf("Simulation time: %ld ms\n", duration.count());

    // Gather the final velocities so every rank holds the complete final state
    // (positions are already replicated after the last exchange).
    if (printResults || validate) {
        std::vector<double> velLocal(3 * static_cast<size_t>(s.count));
        for (int i = 0; i < s.count; ++i) {
            velLocal[3 * i] = s.vx[i];
            velLocal[3 * i + 1] = s.vy[i];
            velLocal[3 * i + 2] = s.vz[i];
        }
        std::vector<double> velAll(3 * static_cast<size_t>(numBodies));
        MPI_Allgatherv(velLocal.data(), 3 * s.count, MPI_DOUBLE, velAll.data(), s.counts3.data(),
                       s.displs3.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos.x = s.posAll[3 * i];
            bodies[i].pos.y = s.posAll[3 * i + 1];
            bodies[i].pos.z = s.posAll[3 * i + 2];
            bodies[i].vel.x = velAll[3 * i];
            bodies[i].vel.y = velAll[3 * i + 1];
            bodies[i].vel.z = velAll[3 * i + 2];
        }
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
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
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies, rank, size);
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
