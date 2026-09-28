#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// The file is compiled with -ffp-contract=off and all fused multiply-adds are
// spelled out with std::fma(), so the floating-point rounding of the kernels is
// fixed regardless of vectorization: results are bit-identical to the original
// sequential code (which GCC contracted in exactly this pattern).
static inline double distSqrSoft(const double dx, const double dy, const double dz) {
    return std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING;
}

// Number of bodies processed simultaneously by the vectorized inner kernel.
// Each lane accumulates the forces of one body over all j in the original
// order, so results are bit-identical to the sequential code.
constexpr size_t TILE = 8;

// Structure-of-arrays body storage. Padded up to a multiple of TILE so the
// vectorized kernel never needs a scalar remainder loop; padding entries are
// never used as force sources and their results are discarded.
//
// Positions are double buffered: a step reads the positions of buffer `cur`
// and writes the integrated positions to the other one. That removes the
// write-after-read dependency between the force loop and the integration, so
// a step needs a single barrier instead of two.
struct BodySystem {
    size_t n = 0;    // number of real bodies
    size_t nPad = 0; // n rounded up to a multiple of TILE
    size_t cur = 0;  // position buffer holding the current positions
    std::vector<double> px[2], py[2], pz[2];
    std::vector<double> vx, vy, vz;

    explicit BodySystem(const size_t numBodies)
        : n(numBodies), nPad(((numBodies + TILE - 1) / TILE) * TILE), px{std::vector<double>(nPad, 0.0),
                                                                        std::vector<double>(nPad, 0.0)},
          py{std::vector<double>(nPad, 0.0), std::vector<double>(nPad, 0.0)},
          pz{std::vector<double>(nPad, 0.0), std::vector<double>(nPad, 0.0)}, vx(nPad, 0.0), vy(nPad, 0.0),
          vz(nPad, 0.0) {}
};

void randomizeBodies(BodySystem& bodies, unsigned int seed = 42) {
    // Kept sequential: reproduces the exact rand_r() stream of the original.
    for (size_t i = 0; i < bodies.n; ++i) {
        bodies.px[0][i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.py[0][i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.pz[0][i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vx[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vy[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// Force computation of one step, fused with the position integration of the
// bodies owned by the calling thread. Must be called from inside a parallel
// region; the `omp for` distributes the tiles and the caller's barrier
// separates the position writes from the next step's position reads.
static void computeStep(BodySystem& bodies, const size_t readBuf) {
    const size_t n = bodies.n;
    const size_t nPad = bodies.nPad;

    const double* __restrict px = bodies.px[readBuf].data();
    const double* __restrict py = bodies.py[readBuf].data();
    const double* __restrict pz = bodies.pz[readBuf].data();
    double* __restrict qx = bodies.px[readBuf ^ 1].data();
    double* __restrict qy = bodies.py[readBuf ^ 1].data();
    double* __restrict qz = bodies.pz[readBuf ^ 1].data();
    double* __restrict vx = bodies.vx.data();
    double* __restrict vy = bodies.vy.data();
    double* __restrict vz = bodies.vz.data();

#pragma omp for schedule(static) nowait
    for (size_t i0 = 0; i0 < nPad; i0 += TILE) {
        double xi[TILE], yi[TILE], zi[TILE];
        double Fx[TILE], Fy[TILE], Fz[TILE];

#pragma omp simd
        for (size_t t = 0; t < TILE; ++t) {
            xi[t] = px[i0 + t];
            yi[t] = py[i0 + t];
            zi[t] = pz[i0 + t];
            Fx[t] = 0.0;
            Fy[t] = 0.0;
            Fz[t] = 0.0;
        }

        for (size_t j = 0; j < n; ++j) {
            const double xj = px[j];
            const double yj = py[j];
            const double zj = pz[j];

#pragma omp simd
            for (size_t t = 0; t < TILE; ++t) {
                const double dx = xj - xi[t];
                const double dy = yj - yi[t];
                const double dz = zj - zi[t];
                const double distSqr = distSqrSoft(dx, dy, dz);
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx[t] = std::fma(dx, invDist3, Fx[t]);
                Fy[t] = std::fma(dy, invDist3, Fy[t]);
                Fz[t] = std::fma(dz, invDist3, Fz[t]);
            }
        }

        const size_t tEnd = (i0 + TILE <= n) ? TILE : n - i0;
#pragma omp simd
        for (size_t t = 0; t < tEnd; ++t) {
            const size_t i = i0 + t;
            // Velocity update (computeForces) followed by the integration of
            // this body's position; both only touch body i.
            vx[i] += DT * Fx[t];
            vy[i] += DT * Fy[t];
            vz[i] += DT * Fz[t];
            qx[i] = std::fma(vx[i], DT, xi[t]);
            qy[i] = std::fma(vy[i], DT, yi[t]);
            qz[i] = std::fma(vz[i], DT, zi[t]);
        }
    }
}

void runSimulation(BodySystem& bodies, const int numSteps) {
    const size_t startBuf = bodies.cur;

#pragma omp parallel
    {
        for (int step = 0; step < numSteps; ++step) {
            computeStep(bodies, (startBuf + (size_t)step) & 1);
#pragma omp barrier
        }
    }

    bodies.cur = (startBuf + (size_t)numSteps) & 1;
}

double computeTotalEnergy(const BodySystem& bodies) {
    const size_t n = bodies.n;

    const double* __restrict px = bodies.px[bodies.cur].data();
    const double* __restrict py = bodies.py[bodies.cur].data();
    const double* __restrict pz = bodies.pz[bodies.cur].data();
    const double* __restrict vx = bodies.vx.data();
    const double* __restrict vy = bodies.vy.data();
    const double* __restrict vz = bodies.vz.data();

    double energy = 0.0;

    // Kinetic energy (assuming unit mass)
#pragma omp parallel for simd reduction(+ : energy) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * std::fma(vz[i], vz[i], std::fma(vx[i], vx[i], vy[i] * vy[i]));
    }

    // Potential energy (assuming unit mass for all bodies).
    // The i-loop is triangular, so a dynamic schedule balances the work.
#pragma omp parallel for reduction(+ : energy) schedule(dynamic, 16)
    for (size_t i = 0; i < n; ++i) {
        double local = 0.0;
#pragma omp simd reduction(+ : local)
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = px[j] - px[i];
            const double dy = py[j] - py[i];
            const double dz = pz[j] - pz[i];
            const double dist = std::sqrt(distSqrSoft(dx, dy, dz));
            local -= 1.0 / dist;
        }
        energy += local;
    }

    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const BodySystem& bodies) {
    const size_t n = bodies.n;

    const double* __restrict px = bodies.px[bodies.cur].data();
    const double* __restrict py = bodies.py[bodies.cur].data();
    const double* __restrict pz = bodies.pz[bodies.cur].data();
    const double* __restrict vx = bodies.vx.data();
    const double* __restrict vy = bodies.vy.data();
    const double* __restrict vz = bodies.vz.data();

    // 1: non-finite, 2: position bounds, 3: velocity bounds, 0: ok
    const auto check = [&](const size_t i) {
        // Check for NaN or Inf values
        if (!std::isfinite(px[i]) || !std::isfinite(py[i]) || !std::isfinite(pz[i]) || !std::isfinite(vx[i]) ||
            !std::isfinite(vy[i]) || !std::isfinite(vz[i])) {
            return 1;
        }
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(px[i]) > maxPos || std::abs(py[i]) > maxPos || std::abs(pz[i]) > maxPos) {
            return 2;
        }
        if (std::abs(vx[i]) > maxVel || std::abs(vy[i]) > maxVel || std::abs(vz[i]) > maxVel) {
            return 3;
        }
        return 0;
    };

    // Report the first (lowest-index) failure, as the sequential version did.
    long long firstBad = (long long)n;
#pragma omp parallel for reduction(min : firstBad) schedule(static)
    for (long long i = 0; i < (long long)n; ++i) {
        if (check((size_t)i) != 0 && i < firstBad) {
            firstBad = i;
        }
    }

    const int failure = firstBad < (long long)n ? check((size_t)firstBad) : 0;
    switch (failure) {
        case 1: printf("Validation failed: found NaN or Inf value in body state\n"); return false;
        case 2: printf("Validation failed: body position exceeds reasonable bounds\n"); return false;
        case 3: printf("Validation failed: body velocity exceeds reasonable bounds\n"); return false;
        default: return true;
    }
}

// Number of hardware threads sharing one physical core (SMT degree), 1 if it
// cannot be determined. The kernel is limited by the divide/square-root units,
// which are shared by the SMT siblings of a core, so using more than one
// thread per core only adds barrier cost.
static int smtDegree() {
    FILE* f = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
    if (f == nullptr) {
        return 1;
    }
    char buf[512] = {};
    const size_t len = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);

    // The list is comma separated and may contain ranges, e.g. "0,128" or "0-1"
    int count = 0;
    for (size_t i = 0; i < len;) {
        char* endA = nullptr;
        const long a = strtol(buf + i, &endA, 10);
        if (endA == buf + i) {
            break;
        }
        i = (size_t)(endA - buf);
        long b = a;
        if (i < len && buf[i] == '-') {
            char* endB = nullptr;
            b = strtol(buf + i + 1, &endB, 10);
            i = (size_t)(endB - buf);
        }
        count += (int)(b - a + 1);
        while (i < len && (buf[i] == ',' || buf[i] == ' ' || buf[i] == '\n')) {
            ++i;
        }
    }
    return count > 0 ? count : 1;
}

// Pick the number of threads: never more than the environment allows, one
// thread per physical core at most, and enough work per thread that the
// per-step barrier does not dominate.
static int chooseThreadCount(const size_t n) {
    constexpr size_t MIN_INTERACTIONS_PER_THREAD = 32768;

    long long threads = omp_get_max_threads();
    threads = std::min(threads, (long long)std::max(1, omp_get_num_procs() / smtDegree()));
    threads = std::min(threads, (long long)std::max<size_t>(1, (n * n) / MIN_INTERACTIONS_PER_THREAD));
    threads = std::min(threads, (long long)std::max<size_t>(1, (n + TILE - 1) / TILE));
    return (int)std::max(1LL, threads);
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

    omp_set_num_threads(chooseThreadCount(numBodies > 0 ? (size_t)numBodies : 0));
    printf("OpenMP threads: %d\n", omp_get_max_threads());

    // Initialize bodies
    BodySystem bodies(numBodies > 0 ? (size_t)numBodies : 0);
    randomizeBodies(bodies);

    // Warm up the thread pool so its creation is not attributed to the kernels
#pragma omp parallel
    {}

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(bodies, numSteps);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData(bodies.n * 6);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < bodies.n; ++i) {
            bodyData[i * 6 + 0] = bodies.px[bodies.cur][i];
            bodyData[i * 6 + 1] = bodies.py[bodies.cur][i];
            bodyData[i * 6 + 2] = bodies.pz[bodies.cur][i];
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
