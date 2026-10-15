#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
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

constexpr int LANES = 8;

// Global positions live in a rank-blocked SoA layout: for rank r the block starts at 3*displs[r]
// and holds cnt[r] x values, then cnt[r] y values, then cnt[r] z values.
// Iterating blocks in rank order and elements in order preserves the original j summation order.
static void computeForces(double* vel, const double* gpos, const int* cnt, const int* displs,
                          int nranks, int myCnt, const double* myPos) {
    for (int i0 = 0; i0 < myCnt; i0 += LANES) {
        const int m = std::min(LANES, myCnt - i0);
        double xi[LANES] = {}, yi[LANES] = {}, zi[LANES] = {};
        double Fx[LANES] = {}, Fy[LANES] = {}, Fz[LANES] = {};
        for (int l = 0; l < m; ++l) {
            xi[l] = myPos[i0 + l];
            yi[l] = myPos[myCnt + i0 + l];
            zi[l] = myPos[2 * myCnt + i0 + l];
        }
        for (int r = 0; r < nranks; ++r) {
            const int c = cnt[r];
            const double* px = gpos + 3 * (size_t)displs[r];
            const double* py = px + c;
            const double* pz = py + c;
            for (int j = 0; j < c; ++j) {
                const double xj = px[j], yj = py[j], zj = pz[j];
                for (int l = 0; l < LANES; ++l) {
                    const double dx = xj - xi[l];
                    const double dy = yj - yi[l];
                    const double dz = zj - zi[l];
                    const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    const double invDist = 1.0 / std::sqrt(distSqr);
                    const double invDist3 = invDist * invDist * invDist;
                    Fx[l] += dx * invDist3;
                    Fy[l] += dy * invDist3;
                    Fz[l] += dz * invDist3;
                }
            }
        }
        for (int l = 0; l < m; ++l) {
            vel[3 * (i0 + l) + 0] += DT * Fx[l];
            vel[3 * (i0 + l) + 1] += DT * Fy[l];
            vel[3 * (i0 + l) + 2] += DT * Fz[l];
        }
    }
}

// Potential energy distributed cyclically over ranks; returns this rank's partial sum.
double computePartialPotential(const std::vector<Body>& bodies, int rank, int nranks) {
    double energy = 0.0;
    const size_t n = bodies.size();
    for (size_t i = rank; i < n; i += nranks) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    return energy;
}

double computeKinetic(const std::vector<Body>& bodies) {
    double energy = 0.0;
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
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
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

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

    // Every rank generates the identical initial state (cheap, avoids a broadcast).
    std::vector<Body> init(numBodies);
    randomizeBodies(init);

    std::vector<int> cnt(nranks), displs(nranks);
    for (int r = 0, off = 0; r < nranks; ++r) {
        cnt[r] = numBodies / nranks + (r < numBodies % nranks ? 1 : 0);
        displs[r] = off;
        off += cnt[r];
    }
    const int myCnt = cnt[rank], myOff = displs[rank];

    std::vector<double> gpos(3 * (size_t)numBodies + 1);
    std::vector<double> vel(3 * (size_t)myCnt + 1);
    for (int r = 0; r < nranks; ++r) {
        double* b = gpos.data() + 3 * (size_t)displs[r];
        for (int k = 0; k < cnt[r]; ++k) {
            const Body& bd = init[displs[r] + k];
            b[k] = bd.pos.x;
            b[cnt[r] + k] = bd.pos.y;
            b[2 * cnt[r] + k] = bd.pos.z;
        }
    }
    for (int k = 0; k < myCnt; ++k) {
        vel[3 * k] = init[myOff + k].vel.x;
        vel[3 * k + 1] = init[myOff + k].vel.y;
        vel[3 * k + 2] = init[myOff + k].vel.z;
    }

    std::vector<int> rcnt(nranks), rdisp(nranks);
    for (int r = 0; r < nranks; ++r) {
        rcnt[r] = 3 * cnt[r];
        rdisp[r] = 3 * displs[r];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    double* myPos = gpos.data() + 3 * (size_t)myOff;
    for (int step = 0; step < numSteps; ++step) {
        // Own block of gpos holds the old positions here, so forces can read it directly.
        computeForces(vel.data(), gpos.data(), cnt.data(), displs.data(), nranks, myCnt, myPos);
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < myCnt; ++k)
                myPos[c * myCnt + k] += vel[3 * k + c] * DT;
        if (nranks > 1)
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, gpos.data(), rcnt.data(), rdisp.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long local_ms = duration.count(), max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", max_ms);

    if (!printResults && !validate) {
        MPI_Finalize();
        return 0;
    }

    // Reassemble full state (positions are replicated; gather velocities to all ranks).
    std::vector<double> gvel(3 * (size_t)numBodies + 1);
    MPI_Allgatherv(vel.data(), 3 * myCnt, MPI_DOUBLE, gvel.data(), rcnt.data(), rdisp.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    std::vector<Body> bodies(numBodies);
    for (int r = 0; r < nranks; ++r) {
        const double* b = gpos.data() + 3 * (size_t)displs[r];
        for (int k = 0; k < cnt[r]; ++k) {
            Body& bd = bodies[displs[r] + k];
            bd.pos.x = b[k];
            bd.pos.y = b[cnt[r] + k];
            bd.pos.z = b[2 * cnt[r] + k];
            bd.vel.x = gvel[3 * (displs[r] + k)];
            bd.vel.y = gvel[3 * (displs[r] + k) + 1];
            bd.vel.z = gvel[3 * (displs[r] + k) + 2];
        }
    }

    if (printResults && rank == 0) {
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

    int status = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        const bool ok = validateSimulation(bodies);  // identical data on all ranks
        double partial = computePartialPotential(bodies, rank, nranks);
        double potential = 0.0;
        MPI_Reduce(&partial, &potential, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (ok) {
                printf("Final energy: %.6f\n", computeKinetic(bodies) + potential);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        status = ok ? 0 : 1;
    }

    MPI_Finalize();
    return status;
}
