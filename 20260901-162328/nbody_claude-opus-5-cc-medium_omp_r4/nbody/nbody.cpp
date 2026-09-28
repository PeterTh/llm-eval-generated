#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// Number of bodies processed simultaneously by the vectorized inner kernel.
// The force accumulation for each body still runs over j in ascending order,
// and every rounding step is spelled out explicitly (std::fma, built with
// -ffp-contract=off), so results stay bit-identical to the scalar reference.
constexpr size_t TILE = 8;

// Structure-of-arrays body storage: halves the bytes streamed by the force
// kernel (positions only) and lets the i-direction vectorize cleanly.
struct BodySystem {
    std::vector<double> px, py, pz;
    std::vector<double> vx, vy, vz;

    explicit BodySystem(const size_t n) : px(n), py(n), pz(n), vx(n), vy(n), vz(n) {}
    size_t size() const { return px.size(); }
};

void randomizeBodies(BodySystem& bodies, unsigned int seed = 42) {
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

// Both kernels below use orphaned work-sharing constructs: they must be called
// from inside an active parallel region (see runSimulation).
void computeForces(BodySystem& bodies) {
    const size_t n = bodies.size();
    const double* __restrict px = bodies.px.data();
    const double* __restrict py = bodies.py.data();
    const double* __restrict pz = bodies.pz.data();
    double* __restrict vx = bodies.vx.data();
    double* __restrict vy = bodies.vy.data();
    double* __restrict vz = bodies.vz.data();

    const size_t numTiles = n / TILE;
    const size_t tail = numTiles * TILE;

#pragma omp for schedule(static) nowait
    for (size_t t = 0; t < numTiles; ++t) {
        const size_t base = t * TILE;

        double Fx[TILE] = {}, Fy[TILE] = {}, Fz[TILE] = {};
        double ix[TILE], iy[TILE], iz[TILE];
        for (size_t k = 0; k < TILE; ++k) {
            ix[k] = px[base + k];
            iy[k] = py[base + k];
            iz[k] = pz[base + k];
        }

        for (size_t j = 0; j < n; ++j) {
            const double jx = px[j], jy = py[j], jz = pz[j];
#pragma omp simd
            for (size_t k = 0; k < TILE; ++k) {
                const double dx = jx - ix[k];
                const double dy = jy - iy[k];
                const double dz = jz - iz[k];
                const double distSqr = std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx[k] = std::fma(dx, invDist3, Fx[k]);
                Fy[k] = std::fma(dy, invDist3, Fy[k]);
                Fz[k] = std::fma(dz, invDist3, Fz[k]);
            }
        }

        for (size_t k = 0; k < TILE; ++k) {
            vx[base + k] += DT * Fx[k];
            vy[base + k] += DT * Fy[k];
            vz[base + k] += DT * Fz[k];
        }
    }

    // Remainder bodies (n % TILE), scalar but with the identical j order.
    // The implicit barrier at the end of this loop also covers the tile loop
    // above, so all velocity updates are visible before integration.
#pragma omp for schedule(static)
    for (size_t i = tail; i < n; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        for (size_t j = 0; j < n; ++j) {
            const double dx = px[j] - px[i];
            const double dy = py[j] - py[i];
            const double dz = pz[j] - pz[i];
            const double distSqr = std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx = std::fma(dx, invDist3, Fx);
            Fy = std::fma(dy, invDist3, Fy);
            Fz = std::fma(dz, invDist3, Fz);
        }
        vx[i] += DT * Fx;
        vy[i] += DT * Fy;
        vz[i] += DT * Fz;
    }
}

void integrateBodies(BodySystem& bodies) {
    const size_t n = bodies.size();
    double* __restrict px = bodies.px.data();
    double* __restrict py = bodies.py.data();
    double* __restrict pz = bodies.pz.data();
    const double* __restrict vx = bodies.vx.data();
    const double* __restrict vy = bodies.vy.data();
    const double* __restrict vz = bodies.vz.data();

#pragma omp for simd schedule(static)
    for (size_t i = 0; i < n; ++i) {
        px[i] = std::fma(vx[i], DT, px[i]);
        py[i] = std::fma(vy[i], DT, py[i]);
        pz[i] = std::fma(vz[i], DT, pz[i]);
    }
}

// One parallel region spans the whole time-step loop: threads are forked once
// instead of twice per step, which matters for small body counts where each
// step is short compared to fork/join cost.
void runSimulation(BodySystem& bodies, const int numSteps) {
#pragma omp parallel proc_bind(spread)
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies);
        integrateBodies(bodies);
    }
}

double computeTotalEnergy(const BodySystem& bodies) {
    const size_t n = bodies.size();
    const double* __restrict px = bodies.px.data();
    const double* __restrict py = bodies.py.data();
    const double* __restrict pz = bodies.pz.data();

    double energy = 0.0;

    // Kinetic energy (assuming unit mass)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies.vx[i] * bodies.vx[i] + bodies.vy[i] * bodies.vy[i] + bodies.vz[i] * bodies.vz[i]);
    }

    // Potential energy (assuming unit mass for all bodies). Per-i partial sums
    // are reduced in index order afterwards so the result is independent of the
    // thread count.
    std::vector<double> partial(n, 0.0);
#pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        double local = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = px[j] - px[i];
            const double dy = py[j] - py[i];
            const double dz = pz[j] - pz[i];
            const double dist = std::sqrt(std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING);
            local -= 1.0 / dist;
        }
        partial[i] = local;
    }
    for (size_t i = 0; i < n; ++i) {
        energy += partial[i];
    }

    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const BodySystem& bodies) {
    const size_t n = bodies.size();
    // Check for NaN/Inf and extreme values (bodies shouldn't fly off to infinity)
    const double maxPos = 1e6;
    const double maxVel = 1e6;
    int badState = 0, badPos = 0, badVel = 0;

#pragma omp parallel for schedule(static) reduction(| : badState, badPos, badVel)
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(bodies.px[i]) || !std::isfinite(bodies.py[i]) || !std::isfinite(bodies.pz[i]) ||
            !std::isfinite(bodies.vx[i]) || !std::isfinite(bodies.vy[i]) || !std::isfinite(bodies.vz[i])) {
            badState = 1;
            continue;
        }
        if (std::abs(bodies.px[i]) > maxPos || std::abs(bodies.py[i]) > maxPos || std::abs(bodies.pz[i]) > maxPos) {
            badPos = 1;
        }
        if (std::abs(bodies.vx[i]) > maxVel || std::abs(bodies.vy[i]) > maxVel || std::abs(bodies.vz[i]) > maxVel) {
            badVel = 1;
        }
    }

    if (badState) {
        printf("Validation failed: found NaN or Inf value in body state\n");
        return false;
    }
    if (badPos) {
        printf("Validation failed: body position exceeds reasonable bounds\n");
        return false;
    }
    if (badVel) {
        printf("Validation failed: body velocity exceeds reasonable bounds\n");
        return false;
    }
    return true;
}

// The force kernel is compute bound, so running two threads per physical core
// only adds barrier traffic (measurably slower on SMT machines). Unless the
// user pinned a thread count via OMP_NUM_THREADS, default to one thread per
// physical core.
void selectDefaultThreadCount() {
    if (getenv("OMP_NUM_THREADS") != nullptr) {
        return;
    }

    std::set<std::pair<int, int>> cores;
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    int packageId = -1;
    while (std::getline(cpuinfo, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const int value = atoi(line.c_str() + colon + 1);
        if (line.rfind("physical id", 0) == 0) {
            packageId = value;
        } else if (line.rfind("core id", 0) == 0) {
            cores.emplace(packageId, value);
        }
    }

    if (!cores.empty() && static_cast<int>(cores.size()) < omp_get_num_procs()) {
        omp_set_num_threads(static_cast<int>(cores.size()));
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

    selectDefaultThreadCount();
    printf("OpenMP threads: %d\n", omp_get_max_threads());

    // Initialize bodies
    BodySystem bodies(numBodies);
    randomizeBodies(bodies);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(bodies, numSteps);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < bodies.size(); ++i) {
            bodyData[i * 6 + 0] = bodies.px[i];
            bodyData[i * 6 + 1] = bodies.py[i];
            bodyData[i * 6 + 2] = bodies.pz[i];
            bodyData[i * 6 + 3] = bodies.vx[i];
            bodyData[i * 6 + 4] = bodies.vy[i];
            bodyData[i * 6 + 5] = bodies.vz[i];
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
