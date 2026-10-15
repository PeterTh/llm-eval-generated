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

// Distributed state: every rank holds all positions (SoA), but velocities only
// for its own slice [lo, hi).
struct Dist {
    int rank, size;
    int n;
    int lo, hi;
    std::vector<int> counts, displs;
};

// Each rank handles a contiguous slice of bodies; per-body summation order over j
// is identical to the serial code, so results match it bit for bit.
void computeForces(const Dist& d, const double* __restrict px, const double* __restrict py,
                   const double* __restrict pz, double* __restrict vx, double* __restrict vy,
                   double* __restrict vz) {
    const int n = d.n;
    constexpr int B = 4;
    int i = d.lo;
    for (; i + B <= d.hi; i += B) {
        double xi[B], yi[B], zi[B], Fx[B] = {}, Fy[B] = {}, Fz[B] = {};
        for (int k = 0; k < B; ++k) {
            xi[k] = px[i + k];
            yi[k] = py[i + k];
            zi[k] = pz[i + k];
        }
        for (int j = 0; j < n; ++j) {
            const double xj = px[j], yj = py[j], zj = pz[j];
#pragma omp simd
            for (int k = 0; k < B; ++k) {
                const double dx = xj - xi[k];
                const double dy = yj - yi[k];
                const double dz = zj - zi[k];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx[k] += dx * invDist3;
                Fy[k] += dy * invDist3;
                Fz[k] += dz * invDist3;
            }
        }
        for (int k = 0; k < B; ++k) {
            vx[i + k - d.lo] += DT * Fx[k];
            vy[i + k - d.lo] += DT * Fy[k];
            vz[i + k - d.lo] += DT * Fz[k];
        }
    }
    for (; i < d.hi; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        const double xi = px[i], yi = py[i], zi = pz[i];
        for (int j = 0; j < n; ++j) {
            const double dx = px[j] - xi;
            const double dy = py[j] - yi;
            const double dz = pz[j] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        vx[i - d.lo] += DT * Fx;
        vy[i - d.lo] += DT * Fy;
        vz[i - d.lo] += DT * Fz;
    }
}

void integrateBodies(const Dist& d, double* px, double* py, double* pz, const double* vx,
                     const double* vy, const double* vz) {
    for (int i = d.lo; i < d.hi; ++i) {
        px[i] += vx[i - d.lo] * DT;
        py[i] += vy[i - d.lo] * DT;
        pz[i] += vz[i - d.lo] * DT;
    }
}

// Total energy; potential part is distributed over ranks (result valid on all ranks).
double computeTotalEnergy(const Dist& d, const double* px, const double* py, const double* pz,
                          const double* vx, const double* vy, const double* vz) {
    double local = 0.0;
    for (int i = d.lo; i < d.hi; ++i) {
        const double a = vx[i - d.lo], b = vy[i - d.lo], c = vz[i - d.lo];
        local += 0.5 * (a * a + b * b + c * c);
    }
    for (int i = d.lo; i < d.hi; ++i) {
        for (int j = i + 1; j < d.n; ++j) {
            const double dx = px[j] - px[i];
            const double dy = py[j] - py[i];
            const double dz = pz[j] - pz[i];
            local -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
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
    Dist d;
    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.size);
    const bool root = d.rank == 0;

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
    if (numBodies < 0) numBodies = 0;
    
    if (root) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Initialize bodies (deterministic; every rank generates the same data)
    const int n = numBodies;
    d.n = n;
    d.counts.resize(d.size);
    d.displs.resize(d.size);
    for (int r = 0, off = 0; r < d.size; ++r) {
        d.counts[r] = n / d.size + (r < n % d.size ? 1 : 0);
        d.displs[r] = off;
        off += d.counts[r];
    }
    d.lo = d.displs[d.rank];
    d.hi = d.lo + d.counts[d.rank];
    const int nLocal = d.counts[d.rank];

    std::vector<double> px(n), py(n), pz(n);
    std::vector<double> vx(nLocal), vy(nLocal), vz(nLocal);
    {
        std::vector<Body> bodies(n);
        randomizeBodies(bodies);
        for (int i = 0; i < n; ++i) {
            px[i] = bodies[i].pos.x;
            py[i] = bodies[i].pos.y;
            pz[i] = bodies[i].pos.z;
        }
        for (int i = d.lo; i < d.hi; ++i) {
            vx[i - d.lo] = bodies[i].vel.x;
            vy[i - d.lo] = bodies[i].vel.y;
            vz[i - d.lo] = bodies[i].vel.z;
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(d, px.data(), py.data(), pz.data(), vx.data(), vy.data(), vz.data());
        integrateBodies(d, px.data(), py.data(), pz.data(), vx.data(), vy.data(), vz.data());
        if (d.size > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, px.data(), d.counts.data(), d.displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, py.data(), d.counts.data(), d.displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pz.data(), d.counts.data(), d.displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (root) printf("Simulation time: %ld ms\n", (long)duration.count());
    
    // Gather velocities on all ranks (needed for validation and printing)
    std::vector<double> gvx(n), gvy(n), gvz(n);
    if (printResults || validate) {
        MPI_Allgatherv(vx.data(), nLocal, MPI_DOUBLE, gvx.data(), d.counts.data(), d.displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(vy.data(), nLocal, MPI_DOUBLE, gvy.data(), d.counts.data(), d.displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(vz.data(), nLocal, MPI_DOUBLE, gvz.data(), d.counts.data(), d.displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (printResults && root) {
        std::vector<double> bodyData;
        bodyData.reserve((size_t)n * 6);
        for (int i = 0; i < n; ++i) {
            bodyData.push_back(px[i]);
            bodyData.push_back(py[i]);
            bodyData.push_back(pz[i]);
            bodyData.push_back(gvx[i]);
            bodyData.push_back(gvy[i]);
            bodyData.push_back(gvz[i]);
        }
        print_results(bodyData, "Bodies");
    }
    
    int rc = 0;
    if (validate) {
        if (root) printf("Validating simulation results...\n");
        
        std::vector<Body> bodies(n);
        for (int i = 0; i < n; ++i) {
            bodies[i].pos = Vec3(px[i], py[i], pz[i]);
            bodies[i].vel = Vec3(gvx[i], gvy[i], gvz[i]);
        }
        const bool ok = validateSimulation(bodies);  // identical on all ranks
        if (ok) {
            double finalEnergy = computeTotalEnergy(d, px.data(), py.data(), pz.data(), vx.data(), vy.data(), vz.data());
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
