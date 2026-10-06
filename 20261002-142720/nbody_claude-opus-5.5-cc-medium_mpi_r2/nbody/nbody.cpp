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

// Block decomposition of [0, n) over nprocs ranks (sizes differ by at most one)
struct Decomp {
    std::vector<int> counts, displs;
    Decomp(int n, int nprocs) : counts(nprocs), displs(nprocs) {
        const int base = n / nprocs, rem = n % nprocs;
        int off = 0;
        for (int r = 0; r < nprocs; ++r) {
            counts[r] = base + (r < rem ? 1 : 0);
            displs[r] = off;
            off += counts[r];
        }
    }
};

constexpr int IBLK = 8;

// Computes forces for local bodies [lo, lo+cnt) against all n bodies (positions
// stored interleaved xyz in pos) and updates local velocities (interleaved xyz).
// The j-summation order per body i is identical to the sequential code, and the
// vectorization is across i, so results are bit-identical.
typedef double vd4 __attribute__((vector_size(32)));

static inline vd4 vsqrt(vd4 v) {
    return vd4{std::sqrt(v[0]), std::sqrt(v[1]), std::sqrt(v[2]), std::sqrt(v[3])};
}

// Accumulates forces on IBLK bodies (xi, yi, zi) from all n bodies, vectorized
// across the IBLK bodies. Kept out of line so that every body, including padded
// remainder blocks, runs exactly the same code.
__attribute__((noinline)) static void forceBlock(const double* __restrict pos, int n, const double* __restrict xi,
                                                 const double* __restrict yi, const double* __restrict zi,
                                                 double* __restrict Fx, double* __restrict Fy,
                                                 double* __restrict Fz) {
    constexpr int NV = IBLK / 4;
    vd4 ax[NV], ay[NV], az[NV], fx[NV], fy[NV], fz[NV];
    for (int v = 0; v < NV; ++v) {
        __builtin_memcpy(&ax[v], xi + 4 * v, sizeof(vd4));
        __builtin_memcpy(&ay[v], yi + 4 * v, sizeof(vd4));
        __builtin_memcpy(&az[v], zi + 4 * v, sizeof(vd4));
        fx[v] = fy[v] = fz[v] = vd4{0.0, 0.0, 0.0, 0.0};
    }
    for (int j = 0; j < n; ++j) {
        const double xj = pos[3 * j], yj = pos[3 * j + 1], zj = pos[3 * j + 2];
        for (int v = 0; v < NV; ++v) {
            const vd4 dx = xj - ax[v];
            const vd4 dy = yj - ay[v];
            const vd4 dz = zj - az[v];
            const vd4 distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const vd4 invDist = 1.0 / vsqrt(distSqr);
            const vd4 invDist3 = invDist * invDist * invDist;
            fx[v] += dx * invDist3;
            fy[v] += dy * invDist3;
            fz[v] += dz * invDist3;
        }
    }
    for (int v = 0; v < NV; ++v) {
        __builtin_memcpy(Fx + 4 * v, &fx[v], sizeof(vd4));
        __builtin_memcpy(Fy + 4 * v, &fy[v], sizeof(vd4));
        __builtin_memcpy(Fz + 4 * v, &fz[v], sizeof(vd4));
    }
}

static void computeForcesLocal(const double* __restrict pos, double* __restrict vel, int n, int lo, int cnt) {
    for (int ii = 0; ii < cnt; ii += IBLK) {
        // Remainder lanes duplicate the last local body and are discarded
        const int valid = (cnt - ii < IBLK) ? cnt - ii : IBLK;
        alignas(64) double xi[IBLK], yi[IBLK], zi[IBLK], Fx[IBLK], Fy[IBLK], Fz[IBLK];
        for (int k = 0; k < IBLK; ++k) {
            const int i = lo + ii + (k < valid ? k : valid - 1);
            xi[k] = pos[3 * i];
            yi[k] = pos[3 * i + 1];
            zi[k] = pos[3 * i + 2];
        }
        forceBlock(pos, n, xi, yi, zi, Fx, Fy, Fz);
        for (int k = 0; k < valid; ++k) {
            const int li = ii + k;
            vel[3 * li] += DT * Fx[k];
            vel[3 * li + 1] += DT * Fy[k];
            vel[3 * li + 2] += DT * Fz[k];
        }
    }
}

// Integrates the local bodies' positions (written in place into the global array).
static void integrateLocal(double* __restrict pos, const double* __restrict vel, int lo, int cnt) {
    double* p = pos + 3 * (size_t)lo;
    for (int k = 0; k < 3 * cnt; ++k) {
        p[k] += vel[k] * DT;
    }
}

// Total energy computed in parallel; pairs (i, j>i) are distributed cyclically
// over i to balance the triangular workload.
double computeTotalEnergy(const std::vector<Body>& bodies, int rank, int nprocs) {
    const size_t n = bodies.size();
    double local = 0.0;

    // Kinetic energy (assuming unit mass)
    if (rank == 0) {
        for (const auto& body : bodies) {
            local += 0.5 * (body.vel.x * body.vel.x +
                            body.vel.y * body.vel.y +
                            body.vel.z * body.vel.z);
        }
    }

    // Potential energy (assuming unit mass for all bodies)
    for (size_t i = rank; i < n; i += nprocs) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            local -= 1.0 / dist;
        }
    }

    double energy = 0.0;
    MPI_Reduce(&local, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
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
    
    // Initialize bodies (deterministically, identically on every rank)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    const int n = numBodies;
    const Decomp dec(n, nprocs);
    const int lo = dec.displs[rank];
    const int cnt = dec.counts[rank];
    std::vector<int> counts3(nprocs), displs3(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        counts3[r] = 3 * dec.counts[r];
        displs3[r] = 3 * dec.displs[r];
    }

    // Global positions (replicated), local velocities; interleaved xyz
    std::vector<double> pos(3 * (size_t)n);
    std::vector<double> vel(3 * (size_t)cnt);
    for (int i = 0; i < n; ++i) {
        pos[3 * i] = bodies[i].pos.x;
        pos[3 * i + 1] = bodies[i].pos.y;
        pos[3 * i + 2] = bodies[i].pos.z;
    }
    for (int k = 0; k < cnt; ++k) {
        vel[3 * k] = bodies[lo + k].vel.x;
        vel[3 * k + 1] = bodies[lo + k].vel.y;
        vel[3 * k + 2] = bodies[lo + k].vel.z;
    }
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForcesLocal(pos.data(), vel.data(), n, lo, cnt);
        integrateLocal(pos.data(), vel.data(), lo, cnt);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pos.data(), counts3.data(), displs3.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
    }

    // Collect velocities on root
    std::vector<double> allVel(root ? 3 * (size_t)n : 0);
    MPI_Gatherv(vel.data(), 3 * cnt, MPI_DOUBLE, allVel.data(), counts3.data(), displs3.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    long local_ms = duration.count(), max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (root) printf("Simulation time: %ld ms\n", max_ms);

    // Write back final state (positions on all ranks, velocities on root)
    for (int i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
        if (root) bodies[i].vel = Vec3(allVel[3 * i], allVel[3 * i + 1], allVel[3 * i + 2]);
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
    
    int ret = 0;
    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        int ok = 0;
        if (root) {
            printf("Validating simulation results...\n");
            ok = validateSimulation(bodies) ? 1 : 0;
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (ok) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies, rank, nprocs);
            if (root) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (root) printf("Validation: FAILED\n");
            ret = 1;
        }
    }
    
    MPI_Finalize();
    return ret;
}
