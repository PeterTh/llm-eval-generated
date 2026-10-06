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

// Block distribution of bodies across ranks: rank r owns [displs[r], displs[r] + counts[r]).
struct Decomposition {
    int rank = 0;
    int size = 1;
    std::vector<int> counts;
    std::vector<int> displs;
    int lo = 0;
    int hi = 0;

    Decomposition(int n, MPI_Comm comm) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        counts.resize(size);
        displs.resize(size);
        const int base = n / size;
        const int rem = n % size;
        int off = 0;
        for (int r = 0; r < size; ++r) {
            counts[r] = base + (r < rem ? 1 : 0);
            displs[r] = off;
            off += counts[r];
        }
        lo = displs[rank];
        hi = lo + counts[rank];
    }
};

// Distributed state: every rank keeps all positions (structure of arrays),
// but only the velocities of the bodies it owns.
struct DistState {
    int n = 0;
    std::vector<double> px, py, pz;  // global, size n
    std::vector<double> vx, vy, vz;  // local, size hi - lo
};

// Velocity update for owned bodies. Several i bodies are processed together
// across SIMD lanes; each lane accumulates over j in the original order. FMA
// usage is spelled out explicitly (and automatic contraction is disabled in the
// build) to reproduce the serial reference's rounding bit for bit.
constexpr int LANES = 8;

void computeForces(DistState& s, const Decomposition& d) {
    const int n = s.n;
    const double* __restrict px = s.px.data();
    const double* __restrict py = s.py.data();
    const double* __restrict pz = s.pz.data();
    double* __restrict vx = s.vx.data();
    double* __restrict vy = s.vy.data();
    double* __restrict vz = s.vz.data();
    const int lo = d.lo;
    const int nloc = d.hi - d.lo;

    for (int i0 = 0; i0 < nloc; i0 += LANES) {
        const int cnt = std::min(LANES, nloc - i0);
        alignas(64) double xi[LANES], yi[LANES], zi[LANES];
        alignas(64) double Fx[LANES], Fy[LANES], Fz[LANES];
        for (int k = 0; k < LANES; ++k) {
            const int gi = lo + i0 + (k < cnt ? k : cnt - 1);
            xi[k] = px[gi];
            yi[k] = py[gi];
            zi[k] = pz[gi];
            Fx[k] = 0.0;
            Fy[k] = 0.0;
            Fz[k] = 0.0;
        }

        for (int j = 0; j < n; ++j) {
            const double xj = px[j], yj = py[j], zj = pz[j];
#pragma GCC ivdep
            for (int k = 0; k < LANES; ++k) {
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
            const int li = i0 + k;
            vx[li] += DT * Fx[k];
            vy[li] += DT * Fy[k];
            vz[li] += DT * Fz[k];
        }
    }
}

// Advance owned positions, then share them with all ranks.
void integrateBodies(DistState& s, const Decomposition& d, MPI_Comm comm) {
    const int lo = d.lo;
    const int nloc = d.hi - d.lo;
    double* __restrict px = s.px.data() + lo;
    double* __restrict py = s.py.data() + lo;
    double* __restrict pz = s.pz.data() + lo;
    const double* __restrict vx = s.vx.data();
    const double* __restrict vy = s.vy.data();
    const double* __restrict vz = s.vz.data();
    for (int i = 0; i < nloc; ++i) {
        px[i] = std::fma(vx[i], DT, px[i]);
        py[i] = std::fma(vy[i], DT, py[i]);
        pz[i] = std::fma(vz[i], DT, pz[i]);
    }

    if (d.size > 1) {
        MPI_Request reqs[3];
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, s.px.data(), d.counts.data(), d.displs.data(),
                        MPI_DOUBLE, comm, &reqs[0]);
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, s.py.data(), d.counts.data(), d.displs.data(),
                        MPI_DOUBLE, comm, &reqs[1]);
        MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, s.pz.data(), d.counts.data(), d.displs.data(),
                        MPI_DOUBLE, comm, &reqs[2]);
        MPI_Waitall(3, reqs, MPI_STATUSES_IGNORE);
    }
}

// Total energy computed collectively; the result is valid on all ranks.
double computeTotalEnergy(const DistState& s, const Decomposition& d, MPI_Comm comm) {
    double energy = 0.0;
    const int n = s.n;
    const int nloc = d.hi - d.lo;

    // Kinetic energy (assuming unit mass) of owned bodies
    for (int i = 0; i < nloc; ++i) {
        energy += 0.5 * (s.vx[i] * s.vx[i] + s.vy[i] * s.vy[i] + s.vz[i] * s.vz[i]);
    }

    // Potential energy (assuming unit mass for all bodies); rows are dealt
    // out cyclically so the triangular workload is balanced.
    const double* px = s.px.data();
    const double* py = s.py.data();
    const double* pz = s.pz.data();
    for (int i = d.rank; i < n; i += d.size) {
        const double xi = px[i], yi = py[i], zi = pz[i];
        double row = 0.0;
        for (int j = i + 1; j < n; ++j) {
            const double dx = px[j] - xi;
            const double dy = py[j] - yi;
            const double dz = pz[j] - zi;
            const double dist = std::sqrt(std::fma(dz, dz, std::fma(dy, dy, dx * dx)) + SOFTENING);
            row -= 1.0 / dist;
        }
        energy += row;
    }

    double total = 0.0;
    MPI_Allreduce(&energy, &total, 1, MPI_DOUBLE, MPI_SUM, comm);
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
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    const bool root = (rank == 0);

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
    if (numBodies < 0) numBodies = 0;
    
    if (root) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        fflush(stdout);
    }
    
    // Initialize bodies (deterministically on every rank, so no broadcast is needed)
    const Decomposition d(numBodies, comm);
    DistState s;
    s.n = numBodies;
    {
        std::vector<Body> bodies(numBodies);
        randomizeBodies(bodies);
        s.px.resize(numBodies);
        s.py.resize(numBodies);
        s.pz.resize(numBodies);
        for (int i = 0; i < numBodies; ++i) {
            s.px[i] = bodies[i].pos.x;
            s.py[i] = bodies[i].pos.y;
            s.pz[i] = bodies[i].pos.z;
        }
        const int nloc = d.hi - d.lo;
        s.vx.resize(nloc);
        s.vy.resize(nloc);
        s.vz.resize(nloc);
        for (int i = 0; i < nloc; ++i) {
            s.vx[i] = bodies[d.lo + i].vel.x;
            s.vy[i] = bodies[d.lo + i].vel.y;
            s.vz[i] = bodies[d.lo + i].vel.z;
        }
    }
    
    // Run simulation
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(s, d);
        integrateBodies(s, d, comm);
    }
    
    MPI_Barrier(comm);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    long local_ms = duration.count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, comm);
    if (root) printf("Simulation time: %ld ms\n", max_ms);
    
    // Collect the full body state on rank 0 for output and validation
    std::vector<Body> bodies;
    if (printResults || validate) {
        std::vector<double> gvx, gvy, gvz;
        if (root) {
            gvx.resize(numBodies);
            gvy.resize(numBodies);
            gvz.resize(numBodies);
        }
        const int nloc = d.hi - d.lo;
        MPI_Gatherv(s.vx.data(), nloc, MPI_DOUBLE, gvx.data(), d.counts.data(), d.displs.data(), MPI_DOUBLE, 0, comm);
        MPI_Gatherv(s.vy.data(), nloc, MPI_DOUBLE, gvy.data(), d.counts.data(), d.displs.data(), MPI_DOUBLE, 0, comm);
        MPI_Gatherv(s.vz.data(), nloc, MPI_DOUBLE, gvz.data(), d.counts.data(), d.displs.data(), MPI_DOUBLE, 0, comm);
        if (root) {
            bodies.resize(numBodies);
            for (int i = 0; i < numBodies; ++i) {
                bodies[i].pos = Vec3(s.px[i], s.py[i], s.pz[i]);
                bodies[i].vel = Vec3(gvx[i], gvy[i], gvz[i]);
            }
        }
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
    int ret = 0;
    if (validate) {
        if (root) printf("Validating simulation results...\n");
        
        int valid = root ? (validateSimulation(bodies) ? 1 : 0) : 0;
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
        if (valid) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(s, d, comm);
            if (root) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            ret = 0;
        } else {
            if (root) printf("Validation: FAILED\n");
            ret = 1;
        }
    }
    
    MPI_Finalize();
    return ret;
}
