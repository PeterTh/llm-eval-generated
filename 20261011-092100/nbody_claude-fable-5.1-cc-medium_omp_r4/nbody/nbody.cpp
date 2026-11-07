#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <omp.h>
#include <sched.h>

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

// Structure-of-arrays copy of the body positions. Two buffers are used:
// one is read by all threads during a step while the updated positions are
// written into the other, so a single barrier per step suffices.
struct PosSoA {
    std::vector<double> x, y, z;
    void resize(size_t n) { x.resize(n); y.resize(n); z.resize(n); }
};

// Must be called by all threads of an enclosing parallel region.
void gatherPositions(const std::vector<Body>& bodies, PosSoA& pos) {
    const size_t n = bodies.size();
    const Body* __restrict b = bodies.data();
    double* __restrict px = pos.x.data();
    double* __restrict py = pos.y.data();
    double* __restrict pz = pos.z.data();
    #pragma omp for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        px[i] = b[i].pos.x;
        py[i] = b[i].pos.y;
        pz[i] = b[i].pos.z;
    }
    // implicit barrier
}

// Force + integration kernel for a block of B consecutive i-bodies
// [i0, i0+cnt). Each i body keeps its own accumulator and iterates j in the
// original order 0..n-1, so the floating point results are bitwise identical
// to the sequential version; SIMD is applied across the B i-bodies instead
// of across j.
template <size_t B>
static inline void stepBlock(const double* __restrict px, const double* __restrict py,
                             const double* __restrict pz, double* __restrict nx,
                             double* __restrict ny, double* __restrict nz,
                             Body* __restrict b, size_t n, size_t i0, size_t cnt) {
    alignas(64) double xi[B], yi[B], zi[B];
    alignas(64) double Fx[B], Fy[B], Fz[B];
    for (size_t k = 0; k < B; ++k) {
        const size_t idx = (k < cnt) ? i0 + k : i0;  // pad with a valid body
        xi[k] = px[idx]; yi[k] = py[idx]; zi[k] = pz[idx];
        Fx[k] = 0.0; Fy[k] = 0.0; Fz[k] = 0.0;
    }

    for (size_t j = 0; j < n; ++j) {
        const double xj = px[j], yj = py[j], zj = pz[j];
        #pragma omp simd aligned(xi, yi, zi, Fx, Fy, Fz : 64)
        for (size_t k = 0; k < B; ++k) {
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

    for (size_t k = 0; k < cnt; ++k) {
        Body& body = b[i0 + k];
        double dvx = DT * Fx[k];
        double dvy = DT * Fy[k];
        double dvz = DT * Fz[k];
        // Prevent contraction of (DT*F) + vel into an FMA, matching the
        // rounding of the original scalar code exactly.
        asm volatile("" : "+x"(dvx), "+x"(dvy), "+x"(dvz));
        body.vel.x += dvx;
        body.vel.y += dvy;
        body.vel.z += dvz;

        // Integration (equivalent to the original integrateBodies step).
        // Explicit FMA matches the contraction the compiler performs for
        // `pos += vel * DT` in the original loop.
        body.pos.x = std::fma(body.vel.x, DT, body.pos.x);
        body.pos.y = std::fma(body.vel.y, DT, body.pos.y);
        body.pos.z = std::fma(body.vel.z, DT, body.pos.z);
        nx[i0 + k] = body.pos.x;
        ny[i0 + k] = body.pos.y;
        nz[i0 + k] = body.pos.z;
    }
}

template <size_t B>
static void stepBlocked(const PosSoA& cur, PosSoA& next, Body* __restrict b, size_t n) {
    const double* __restrict px = cur.x.data();
    const double* __restrict py = cur.y.data();
    const double* __restrict pz = cur.z.data();
    double* __restrict nx = next.x.data();
    double* __restrict ny = next.y.data();
    double* __restrict nz = next.z.data();
    #pragma omp for schedule(dynamic, 1)
    for (size_t i0 = 0; i0 < n; i0 += B) {
        const size_t cnt = (n - i0 < B) ? (n - i0) : B;
        stepBlock<B>(px, py, pz, nx, ny, nz, b, n, i0, cnt);
    }
    // implicit barrier: all reads of `cur` and writes of `next` are complete
}

// One simulation step: compute forces, update velocities and integrate
// positions. Reads positions from `cur`, writes updated positions to `next`
// and to `bodies`. Must be called by all threads of an enclosing parallel
// region.
void simulationStep(std::vector<Body>& bodies, const PosSoA& cur, PosSoA& next) {
    const size_t n = bodies.size();
    Body* __restrict b = bodies.data();

    // Pick the i-block width: wide blocks give more independent SIMD chains
    // (hiding sqrt/div latency), narrow blocks give more work items so that
    // all threads stay busy for small n. Results are identical either way.
    const size_t nthreads = static_cast<size_t>(omp_get_num_threads());
    if (n >= 16 * 4 * nthreads) {
        stepBlocked<16>(cur, next, b, n);
    } else if (n >= 8 * 4 * nthreads) {
        stepBlocked<8>(cur, next, b, n);
    } else {
        stepBlocked<4>(cur, next, b, n);
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
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

// Count physical cores available to this process (Linux sysfs topology).
// Returns 0 if the information is unavailable. The force kernel is bound by
// the FP sqrt/div throughput of each core, so a second SMT thread per core
// adds barrier/scheduling overhead without adding throughput.
static int countPhysicalCores() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return 0;

    int cores = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &mask)) continue;
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                        "/topology/thread_siblings_list");
        if (!f) return 0;
        std::string list;
        std::getline(f, list);
        // Count the lowest-numbered sibling of each core that we may run on.
        // The lowest sibling is always first in the list ("0,128" or "0-1").
        const int first = atoi(list.c_str());
        if (first == cpu) ++cores;
    }
    return cores;
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
    // Unless the user chose a thread count explicitly, run one thread per
    // physical core (see countPhysicalCores).
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        const int cores = countPhysicalCores();
        if (cores > 0 && cores < omp_get_max_threads()) omp_set_num_threads(cores);
    }
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    PosSoA posBuf[2];
    posBuf[0].resize(numBodies);
    posBuf[1].resize(numBodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    // One parallel region for the whole simulation; a single barrier per
    // step (at the end of the worksharing loop) separates the steps.
    #pragma omp parallel proc_bind(close)
    {
        gatherPositions(bodies, posBuf[0]);
        for (int step = 0; step < numSteps; ++step) {
            simulationStep(bodies, posBuf[step & 1], posBuf[(step + 1) & 1]);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
    // Print results for external validation
    if (printResults) {
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
