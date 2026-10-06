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
        body.pos.x = std::fma(2.0, rand_r(&seed) / (double)RAND_MAX, -1.0);
        body.pos.y = std::fma(2.0, rand_r(&seed) / (double)RAND_MAX, -1.0);
        body.pos.z = std::fma(2.0, rand_r(&seed) / (double)RAND_MAX, -1.0);
        body.vel.x = std::fma(2.0, rand_r(&seed) / (double)RAND_MAX, -1.0);
        body.vel.y = std::fma(2.0, rand_r(&seed) / (double)RAND_MAX, -1.0);
        body.vel.z = std::fma(2.0, rand_r(&seed) / (double)RAND_MAX, -1.0);
    }
}

// Block distribution of bodies across ranks.
struct Decomp {
    int rank = 0, size = 1;
    std::vector<int> counts, displs;  // in bodies
    int begin = 0, count = 0;
};

static Decomp makeDecomp(int n, int rank, int size) {
    Decomp d;
    d.rank = rank;
    d.size = size;
    d.counts.resize(size);
    d.displs.resize(size);
    const int base = n / size, rem = n % size;
    int off = 0;
    for (int r = 0; r < size; ++r) {
        d.counts[r] = base + (r < rem ? 1 : 0);
        d.displs[r] = off;
        off += d.counts[r];
    }
    d.begin = d.displs[rank];
    d.count = d.counts[rank];
    return d;
}

constexpr int TILE = 8;

// Floating-point contraction is disabled at build time; the fused multiply-adds
// below are explicit so results are bit-identical regardless of vectorization
// or rank count.

// Computes forces for local bodies [begin, begin+count) against all bodies
// (positions in pos, packed xyz), updating local velocities (vel, packed xyz,
// indexed locally). Per-body summation over j follows the original order.
static void computeForces(const double* __restrict pos, double* __restrict vel, int n, int begin, int count) {
    // A partial last tile is padded with copies of its last body so that every
    // body goes through identical code (bit-identical results for any rank count).
    for (int t = 0; t < count; t += TILE) {
        const int valid = count - t < TILE ? count - t : TILE;
        double xi[TILE], yi[TILE], zi[TILE];
        double Fx[TILE] = {}, Fy[TILE] = {}, Fz[TILE] = {};
        for (int k = 0; k < TILE; ++k) {
            const int i = begin + t + (k < valid ? k : valid - 1);
            xi[k] = pos[3 * i];
            yi[k] = pos[3 * i + 1];
            zi[k] = pos[3 * i + 2];
        }
        for (int j = 0; j < n; ++j) {
            const double xj = pos[3 * j], yj = pos[3 * j + 1], zj = pos[3 * j + 2];
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
        for (int k = 0; k < valid; ++k) {
            const int l = t + k;
            vel[3 * l] += DT * Fx[k];
            vel[3 * l + 1] += DT * Fy[k];
            vel[3 * l + 2] += DT * Fz[k];
        }
    }
}

static void integrateBodies(double* __restrict pos, const double* __restrict vel, int begin, int count) {
    double* p = pos + 3 * (size_t)begin;
    for (int k = 0; k < 3 * count; ++k) {
        p[k] = std::fma(vel[k], DT, p[k]);
    }
}

// Distributed total energy; requires full body state on every rank.
double computeTotalEnergy(const std::vector<Body>& bodies, int rank, int size) {
    double local = 0.0;
    const size_t n = bodies.size();

    if (rank == 0) {
        for (const auto& body : bodies) {
            local += 0.5 * (body.vel.x * body.vel.x +
                            body.vel.y * body.vel.y +
                            body.vel.z * body.vel.z);
        }
    }

    // Cyclic row distribution balances the triangular workload.
    for (size_t i = rank; i < n; i += size) {
        double pot = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING);
            pot += 1.0 / dist;
        }
        local -= pot;
    }

    double energy = 0.0;
    MPI_Allreduce(&local, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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
    
    // Initialize bodies (deterministically, identically on every rank)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    const Decomp d = makeDecomp(numBodies, rank, size);
    std::vector<int> counts3(size), displs3(size);
    for (int r = 0; r < size; ++r) {
        counts3[r] = 3 * d.counts[r];
        displs3[r] = 3 * d.displs[r];
    }

    // Replicated positions (packed xyz), local velocities (packed xyz)
    std::vector<double> pos(3 * (size_t)numBodies);
    std::vector<double> vel(3 * (size_t)d.count);
    for (int i = 0; i < numBodies; ++i) {
        pos[3 * i] = bodies[i].pos.x;
        pos[3 * i + 1] = bodies[i].pos.y;
        pos[3 * i + 2] = bodies[i].pos.z;
    }
    for (int l = 0; l < d.count; ++l) {
        const Body& b = bodies[d.begin + l];
        vel[3 * l] = b.vel.x;
        vel[3 * l + 1] = b.vel.y;
        vel[3 * l + 2] = b.vel.z;
    }
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(pos.data(), vel.data(), numBodies, d.begin, d.count);
        integrateBodies(pos.data(), vel.data(), d.begin, d.count);
        if (size > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pos.data(), counts3.data(), displs3.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) printf("Simulation time: %ld ms\n", duration.count());

    // Gather velocities so every rank holds the full final state
    std::vector<double> allVel(3 * (size_t)numBodies);
    MPI_Allgatherv(vel.data(), 3 * d.count, MPI_DOUBLE, allVel.data(), counts3.data(), displs3.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos = Vec3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
        bodies[i].vel = Vec3(allVel[3 * i], allVel[3 * i + 1], allVel[3 * i + 2]);
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
    int ret = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        
        int ok = 0;
        if (rank == 0) ok = validateSimulation(bodies) ? 1 : 0;
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
            ret = 1;
        }
    }
    
    MPI_Finalize();
    return ret;
}
