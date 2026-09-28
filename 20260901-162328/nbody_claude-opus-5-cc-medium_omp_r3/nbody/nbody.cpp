#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#include <sched.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// Bodies processed simultaneously by one SIMD-vectorized i-tile. Each lane
// keeps its own force accumulator and walks j in ascending order, so the
// floating-point summation order per body is identical to the original scalar
// implementation. The wide tile amortizes the sqrt/divide latency better; the
// narrow one is used when the body count is too small to keep every thread
// busy otherwise.
constexpr size_t WIDE_TILE = 8;
constexpr size_t NARROW_TILE = 4;

// Structure-of-arrays body storage: contiguous coordinate streams let the
// inner j-loop broadcast a single body and the i-tile load with plain
// vector loads instead of gathers.
struct BodySystem {
    std::vector<double> px, py, pz;
    std::vector<double> vx, vy, vz;

    explicit BodySystem(const size_t n) : px(n), py(n), pz(n), vx(n), vy(n), vz(n) {}

    size_t size() const { return px.size(); }
};

void randomizeBodies(BodySystem& bodies, unsigned int seed = 42) {
    // Kept sequential: rand_r carries a strict dependency through the seed.
    const size_t n = bodies.size();
    for (size_t i = 0; i < n; ++i) {
        bodies.px[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.py[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.pz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vx[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vy[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// Force kernel body; called from inside an already-open parallel region.
template <size_t TILE>
void computeForcesTiled(BodySystem& bodies) {
    const size_t n = bodies.size();

    const double* __restrict const px = bodies.px.data();
    const double* __restrict const py = bodies.py.data();
    const double* __restrict const pz = bodies.pz.data();
    double* __restrict const vx = bodies.vx.data();
    double* __restrict const vy = bodies.vy.data();
    double* __restrict const vz = bodies.vz.data();

#pragma omp for schedule(static)
    for (size_t ii = 0; ii < n; ii += TILE) {
        const size_t m = std::min(TILE, n - ii);

        double ix[TILE], iy[TILE], iz[TILE];
        double Fx[TILE], Fy[TILE], Fz[TILE];
        for (size_t k = 0; k < m; ++k) {
            ix[k] = px[ii + k];
            iy[k] = py[ii + k];
            iz[k] = pz[ii + k];
            Fx[k] = 0.0;
            Fy[k] = 0.0;
            Fz[k] = 0.0;
        }

        for (size_t j = 0; j < n; ++j) {
            const double jx = px[j], jy = py[j], jz = pz[j];

#pragma omp simd simdlen(TILE)
            for (size_t k = 0; k < m; ++k) {
                const double dx = jx - ix[k];
                const double dy = jy - iy[k];
                const double dz = jz - iz[k];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx[k] += dx * invDist3;
                Fy[k] += dy * invDist3;
                Fz[k] += dz * invDist3;
            }
        }

        for (size_t k = 0; k < m; ++k) {
            vx[ii + k] += DT * Fx[k];
            vy[ii + k] += DT * Fy[k];
            vz[ii + k] += DT * Fz[k];
        }
    }
}

// Dispatches to the tile width picked for this problem size. The predicate is
// uniform across the team, so every thread enters the same worksharing loop.
void computeForces(BodySystem& bodies, const bool wide) {
    if (wide) {
        computeForcesTiled<WIDE_TILE>(bodies);
    } else {
        computeForcesTiled<NARROW_TILE>(bodies);
    }
}

// Position update; called from inside an already-open parallel region.
void integrateBodies(BodySystem& bodies) {
    const size_t n = bodies.size();

    double* __restrict const px = bodies.px.data();
    double* __restrict const py = bodies.py.data();
    double* __restrict const pz = bodies.pz.data();
    const double* __restrict const vx = bodies.vx.data();
    const double* __restrict const vy = bodies.vy.data();
    const double* __restrict const vz = bodies.vz.data();

#pragma omp for simd schedule(static)
    for (size_t i = 0; i < n; ++i) {
        px[i] += vx[i] * DT;
        py[i] += vy[i] * DT;
        pz[i] += vz[i] * DT;
    }
}

double computeTotalEnergy(const BodySystem& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    const double* __restrict const px = bodies.px.data();
    const double* __restrict const py = bodies.py.data();
    const double* __restrict const pz = bodies.pz.data();

    // Kinetic energy (assuming unit mass)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies.vx[i] * bodies.vx[i] + bodies.vy[i] * bodies.vy[i] + bodies.vz[i] * bodies.vz[i]);
    }

    // Potential energy (assuming unit mass for all bodies). Each i contributes
    // its own partial sum; the partials are folded together in ascending i
    // order so the result is reproducible regardless of the thread count.
    std::vector<double> partial(n, 0.0);

#pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < n; ++i) {
        double local = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = px[j] - px[i];
            const double dy = py[j] - py[i];
            const double dz = pz[j] - pz[i];
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            local += 1.0 / dist;
        }
        partial[i] = local;
    }

    for (size_t i = 0; i < n; ++i) {
        energy -= partial[i];
    }

    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const BodySystem& bodies) {
    const size_t n = bodies.size();

    // Index of the first body failing each check, so the message reported is
    // the one the original sequential scan would have produced.
    size_t firstNonFinite = n, firstPos = n, firstVel = n;

    // Check for extreme values (bodies shouldn't fly off to infinity)
    const double maxPos = 1e6;
    const double maxVel = 1e6;

#pragma omp parallel for schedule(static) reduction(min : firstNonFinite, firstPos, firstVel)
    for (size_t i = 0; i < n; ++i) {
        // Check for NaN or Inf values
        if (!std::isfinite(bodies.px[i]) || !std::isfinite(bodies.py[i]) || !std::isfinite(bodies.pz[i]) ||
            !std::isfinite(bodies.vx[i]) || !std::isfinite(bodies.vy[i]) || !std::isfinite(bodies.vz[i])) {
            firstNonFinite = std::min(firstNonFinite, i);
            continue;
        }

        if (std::abs(bodies.px[i]) > maxPos || std::abs(bodies.py[i]) > maxPos || std::abs(bodies.pz[i]) > maxPos) {
            firstPos = std::min(firstPos, i);
        } else if (std::abs(bodies.vx[i]) > maxVel || std::abs(bodies.vy[i]) > maxVel ||
                   std::abs(bodies.vz[i]) > maxVel) {
            firstVel = std::min(firstVel, i);
        }
    }

    const size_t first = std::min({firstNonFinite, firstPos, firstVel});
    if (first == n) {
        return true;
    }

    if (first == firstNonFinite) {
        printf("Validation failed: found NaN or Inf value in body state\n");
    } else if (first == firstPos) {
        printf("Validation failed: body position exceeds reasonable bounds\n");
    } else {
        printf("Validation failed: body velocity exceeds reasonable bounds\n");
    }
    return false;
}

// libgomp leaves the team unbound unless OMP_PROC_BIND/OMP_PLACES are set, and
// letting the scheduler migrate threads costs several x at high thread counts.
// Spread the team over the CPUs this process is already allowed to run on, one
// thread per CPU, unless the user configured binding explicitly.
void bindThreads() {
    if (getenv("OMP_PROC_BIND") != nullptr || getenv("OMP_PLACES") != nullptr || omp_get_num_places() > 0) {
        return;
    }

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return;
    }

    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) {
            cpus.push_back(cpu);
        }
    }
    if (cpus.empty()) {
        return;
    }

#pragma omp parallel
    {
        cpu_set_t self;
        CPU_ZERO(&self);
        CPU_SET(cpus[omp_get_thread_num() % cpus.size()], &self);
        sched_setaffinity(0, sizeof(self), &self);
    }
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Initialize bodies
    BodySystem bodies(numBodies);
    randomizeBodies(bodies);

    // Pin the team before timing so the threads do not migrate between cores
    bindThreads();

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    // One parallel region for the whole simulation: the per-step work shares
    // the same team and only pays for the implicit barriers between phases.
    const bool wideTile = static_cast<size_t>(numBodies) >= WIDE_TILE * omp_get_max_threads();

#pragma omp parallel
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, wideTile);
        integrateBodies(bodies);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            bodyData.push_back(bodies.px[i]);
            bodyData.push_back(bodies.py[i]);
            bodyData.push_back(bodies.pz[i]);
            bodyData.push_back(bodies.vx[i]);
            bodyData.push_back(bodies.vy[i]);
            bodyData.push_back(bodies.vz[i]);
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}
