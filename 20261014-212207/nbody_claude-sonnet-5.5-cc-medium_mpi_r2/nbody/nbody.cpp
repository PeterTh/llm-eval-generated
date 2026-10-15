#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <mpi.h>
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

// Positions are stored in a block-structured SoA layout: the block of rank r
// starts at 3*displ[r] and holds cnt[r] x values, then cnt[r] y values, then
// cnt[r] z values. This lets a single in-place MPI_Allgatherv distribute them.
struct Partition {
    int size, rank;
    std::vector<int> cnt, displ;  // bodies per rank / first body of rank
    std::vector<int> rcnt, rdisp; // doubles per rank / offset in doubles
    Partition(int n, int size_, int rank_) : size(size_), rank(rank_), cnt(size_), displ(size_), rcnt(size_), rdisp(size_) {
        const int base = n / size, rem = n % size;
        int off = 0;
        for (int r = 0; r < size; ++r) {
            cnt[r] = base + (r < rem ? 1 : 0);
            displ[r] = off;
            rcnt[r] = 3 * cnt[r];
            rdisp[r] = 3 * off;
            off += cnt[r];
        }
    }
    int lo() const { return displ[rank]; }
    int local() const { return cnt[rank]; }
};

constexpr int TILE = 8;

// Update velocities of the locally owned bodies. The summation order over j is
// identical to the serial code (global index order), so results are bit-identical.
void computeForces(const std::vector<double>& pos, std::vector<double>& vel, const Partition& part) {
    const int nl = part.local();
    const int lo = part.lo();
    const double* mine = &pos[3 * (size_t)lo];

    for (int i0 = 0; i0 < nl; i0 += TILE) {
        const int t = std::min(TILE, nl - i0);
        double xi[TILE], yi[TILE], zi[TILE], Fx[TILE], Fy[TILE], Fz[TILE];
        for (int k = 0; k < TILE; ++k) {
            const int kk = k < t ? k : t - 1;
            xi[k] = mine[i0 + kk];
            yi[k] = mine[nl + i0 + kk];
            zi[k] = mine[2 * nl + i0 + kk];
            Fx[k] = Fy[k] = Fz[k] = 0.0;
        }
        for (int r = 0; r < part.size; ++r) {
            const int c = part.cnt[r];
            const double* bx = &pos[part.rdisp[r]];
            const double* by = bx + c;
            const double* bz = by + c;
            for (int j = 0; j < c; ++j) {
                const double xj = bx[j], yj = by[j], zj = bz[j];
#pragma GCC unroll 8
                for (int k = 0; k < TILE; ++k) {
                    const double dx = xj - xi[k];
                    const double dy = yj - yi[k];
                    const double dz = zj - zi[k];
                    const double distSqr = std::fma(dz, dz, std::fma(dy, dy, dx * dx)) + SOFTENING;
                    const double invDist = 1.0 / std::sqrt(distSqr);
                    const double invDist3 = invDist * invDist * invDist;
                    Fx[k] = std::fma(dx, invDist3, Fx[k]);
                    Fy[k] = std::fma(dy, invDist3, Fy[k]);
                    Fz[k] = std::fma(dz, invDist3, Fz[k]);
                }
            }
        }
        // Fixed-width update on padded copies.
        double vx[TILE], vy[TILE], vz[TILE];
        for (int k = 0; k < TILE; ++k) {
            const int kk = k < t ? k : t - 1;
            vx[k] = vel[i0 + kk];
            vy[k] = vel[nl + i0 + kk];
            vz[k] = vel[2 * nl + i0 + kk];
        }
        for (int k = 0; k < TILE; ++k) {
            vx[k] = std::fma(DT, Fx[k], vx[k]);
            vy[k] = std::fma(DT, Fy[k], vy[k]);
            vz[k] = std::fma(DT, Fz[k], vz[k]);
        }
        for (int k = 0; k < t; ++k) {
            vel[i0 + k] = vx[k];
            vel[nl + i0 + k] = vy[k];
            vel[2 * nl + i0 + k] = vz[k];
        }
    }
}

// Advance owned positions (velocities use the same block-local SoA layout).
void integrateBodies(std::vector<double>& pos, const std::vector<double>& vel, const Partition& part) {
    const int nl = part.local();
    double* mine = &pos[3 * (size_t)part.lo()];
    for (int i = 0; i < 3 * nl; ++i) mine[i] = std::fma(vel[i], DT, mine[i]);
}

// Energy: kinetic over owned bodies, potential over cyclically distributed rows.
double computeTotalEnergy(const std::vector<double>& pos, const std::vector<double>& vel, const Partition& part, int n) {
    const int nl = part.local();
    double energy = 0.0;
    for (int i = 0; i < nl; ++i) {
        energy += 0.5 * (vel[i] * vel[i] + vel[nl + i] * vel[nl + i] + vel[2 * nl + i] * vel[2 * nl + i]);
    }

    // global index -> position lookup
    std::vector<double> gx(n), gy(n), gz(n);
    for (int r = 0; r < part.size; ++r) {
        const double* b = &pos[part.rdisp[r]];
        const int c = part.cnt[r];
        for (int k = 0; k < c; ++k) {
            gx[part.displ[r] + k] = b[k];
            gy[part.displ[r] + k] = b[c + k];
            gz[part.displ[r] + k] = b[2 * c + k];
        }
    }
    for (int i = part.rank; i < n; i += part.size) {
        double e = 0.0;
        for (int j = i + 1; j < n; ++j) {
            const double dx = gx[j] - gx[i];
            const double dy = gy[j] - gy[i];
            const double dz = gz[j] - gz[i];
            e += 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
        energy -= e;
    }
    double total = 0.0;
    MPI_Allreduce(&energy, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return total;
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
    int rc = 0;

    // Parse command line arguments (every rank parses identically)
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

    const int n = numBodies;
    Partition part(n, size, rank);
    const int nl = part.local();

    // Every rank generates the same initial state (cheap, deterministic) and keeps its slice.
    std::vector<double> pos(3 * (size_t)n), vel(3 * (size_t)nl);
    {
        std::vector<Body> bodies(n);
        randomizeBodies(bodies);
        for (int r = 0; r < size; ++r) {
            double* b = &pos[part.rdisp[r]];
            const int c = part.cnt[r], d = part.displ[r];
            for (int k = 0; k < c; ++k) {
                b[k] = bodies[d + k].pos.x;
                b[c + k] = bodies[d + k].pos.y;
                b[2 * c + k] = bodies[d + k].pos.z;
            }
        }
        for (int k = 0; k < nl; ++k) {
            const Body& bd = bodies[part.lo() + k];
            vel[k] = bd.vel.x;
            vel[nl + k] = bd.vel.y;
            vel[2 * nl + k] = bd.vel.z;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(pos, vel, part);
        integrateBodies(pos, vel, part);
        if (size > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pos.data(), part.rcnt.data(), part.rdisp.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) printf("Simulation time: %ld ms\n", (long)duration.count());

    if (printResults || validate) {
        // Gather velocities (block-SoA per rank) on rank 0 and rebuild the Body array.
        std::vector<double> allVel;
        if (rank == 0) allVel.resize(3 * (size_t)n);
        MPI_Gatherv(vel.data(), 3 * nl, MPI_DOUBLE, allVel.data(), part.rcnt.data(), part.rdisp.data(), MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);

        int ok = 1;
        if (rank == 0) {
            std::vector<Body> bodies(n);
            for (int r = 0; r < size; ++r) {
                const double* p = &pos[part.rdisp[r]];
                const double* v = &allVel[part.rdisp[r]];
                const int c = part.cnt[r], d = part.displ[r];
                for (int k = 0; k < c; ++k) {
                    bodies[d + k].pos = Vec3(p[k], p[c + k], p[2 * c + k]);
                    bodies[d + k].vel = Vec3(v[k], v[c + k], v[2 * c + k]);
                }
            }

            if (printResults) {
                std::vector<double> bodyData;
                bodyData.reserve((size_t)n * 6);
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
            if (validate) {
                printf("Validating simulation results...\n");
                ok = validateSimulation(bodies) ? 1 : 0;
            }
        }

        if (validate) {
            MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (ok) {
                double finalEnergy = computeTotalEnergy(pos, vel, part, n);
                if (rank == 0) {
                    printf("Final energy: %.6f\n", finalEnergy);
                    printf("Validation: PASSED\n");
                }
            } else {
                if (rank == 0) printf("Validation: FAILED\n");
                rc = 1;
            }
        }
    }

    MPI_Finalize();
    return rc;
}
