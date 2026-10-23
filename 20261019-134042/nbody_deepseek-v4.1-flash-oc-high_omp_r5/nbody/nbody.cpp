#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#if defined(__AVX__)
#include <immintrin.h>
#endif

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

// Softened squared distance with the same operation grouping as the reference
// build: dy*dy is rounded on its own and the remaining terms are fused.
static inline double softenedDistSqr(const double dx, const double dy, const double dz) {
    const double dy2 = dy * dy;
    const double dxy2 = std::fma(dx, dx, dy2);
    return std::fma(dz, dz, dxy2) + SOFTENING;
}

// Computes 1 / |d|^3, preserving the reference rounding exactly.
static inline double invDistCubed(const double dx, const double dy, const double dz) {
    const double invDist = 1.0 / std::sqrt(softenedDistSqr(dx, dy, dz));
    return invDist * invDist * invDist;
}

// Accumulates the force on a single body. Every body's force is computed
// independently, so the outer loop over bodies can be distributed across
// threads without changing any floating-point result.
static inline void computeForceOnBody(std::vector<Body>& bodies, const size_t n, const size_t i) {
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    const Body& bodyI = bodies[i];

    // Batch the square roots and divisions for four bodies at a time. Each
    // SIMD lane performs exactly the same IEEE operation as the scalar
    // reference, so the results are bit-for-bit identical. The force
    // accumulators are still updated in the original j order.
    size_t j = 0;
#if defined(__AVX__)
    for (; j + 3 < n; j += 4) {
        const double dx0 = bodies[j + 0].pos.x - bodyI.pos.x;
        const double dy0 = bodies[j + 0].pos.y - bodyI.pos.y;
        const double dz0 = bodies[j + 0].pos.z - bodyI.pos.z;
        const double dx1 = bodies[j + 1].pos.x - bodyI.pos.x;
        const double dy1 = bodies[j + 1].pos.y - bodyI.pos.y;
        const double dz1 = bodies[j + 1].pos.z - bodyI.pos.z;
        const double dx2 = bodies[j + 2].pos.x - bodyI.pos.x;
        const double dy2 = bodies[j + 2].pos.y - bodyI.pos.y;
        const double dz2 = bodies[j + 2].pos.z - bodyI.pos.z;
        const double dx3 = bodies[j + 3].pos.x - bodyI.pos.x;
        const double dy3 = bodies[j + 3].pos.y - bodyI.pos.y;
        const double dz3 = bodies[j + 3].pos.z - bodyI.pos.z;

        const double distSqr0 = softenedDistSqr(dx0, dy0, dz0);
        const double distSqr1 = softenedDistSqr(dx1, dy1, dz1);
        const double distSqr2 = softenedDistSqr(dx2, dy2, dz2);
        const double distSqr3 = softenedDistSqr(dx3, dy3, dz3);

        const __m256d distSqr = _mm256_set_pd(distSqr3, distSqr2, distSqr1, distSqr0);
        const __m256d invDist = _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(distSqr));
        const __m256d invCube = _mm256_mul_pd(_mm256_mul_pd(invDist, invDist), invDist);

        alignas(32) double cube[4];
        _mm256_store_pd(cube, invCube);

        Fx = std::fma(dx0, cube[0], Fx);
        Fy = std::fma(dy0, cube[0], Fy);
        Fz = std::fma(dz0, cube[0], Fz);
        Fx = std::fma(dx1, cube[1], Fx);
        Fy = std::fma(dy1, cube[1], Fy);
        Fz = std::fma(dz1, cube[1], Fz);
        Fx = std::fma(dx2, cube[2], Fx);
        Fy = std::fma(dy2, cube[2], Fy);
        Fz = std::fma(dz2, cube[2], Fz);
        Fx = std::fma(dx3, cube[3], Fx);
        Fy = std::fma(dy3, cube[3], Fy);
        Fz = std::fma(dz3, cube[3], Fz);
    }
#endif

    for (; j < n; ++j) {
        const double dx = bodies[j].pos.x - bodyI.pos.x;
        const double dy = bodies[j].pos.y - bodyI.pos.y;
        const double dz = bodies[j].pos.z - bodyI.pos.z;
        const double invCube = invDistCubed(dx, dy, dz);

        Fx = std::fma(dx, invCube, Fx);
        Fy = std::fma(dy, invCube, Fy);
        Fz = std::fma(dz, invCube, Fz);
    }

    // Force the intermediate product to be rounded before the addition so the
    // result matches the reference build (which does not fuse this update)
    // regardless of how the OpenMP outlined loop is compiled.
    volatile double dtx = DT * Fx;
    volatile double dty = DT * Fy;
    volatile double dtz = DT * Fz;

    bodies[i].vel.x += dtx;
    bodies[i].vel.y += dty;
    bodies[i].vel.z += dtz;
}

static inline void integrateBody(Body& body) {
    body.pos.x = std::fma(body.vel.x, DT, body.pos.x);
    body.pos.y = std::fma(body.vel.y, DT, body.pos.y);
    body.pos.z = std::fma(body.vel.z, DT, body.pos.z);
}

// Runs all time steps inside a single parallel region. Keeping the worker
// threads resident across steps avoids repeated fork/join overhead, which
// dominates for small particle counts.
void simulate(std::vector<Body>& bodies, const int numSteps) {
    const size_t n = bodies.size();

    #pragma omp parallel
    {
        for (int step = 0; step < numSteps; ++step) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                computeForceOnBody(bodies, n, i);
            }

            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                integrateBody(bodies[i]);
            }
        }
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

// Number of CPUs represented by a Linux cpu-list string such as "0,128" or
// "0-3,8-11".
static int cpuListCount(const char* list) {
    int total = 0;
    const char* p = list;
    while (*p != '\0' && *p != '\n') {
        int first = 0;
        int last = 0;
        const int matched = std::sscanf(p, "%d-%d", &first, &last);
        if (matched == 2) {
            total += last - first + 1;
        } else if (std::sscanf(p, "%d", &first) == 1) {
            total += 1;
        }
        while (*p != '\0' && *p != ',' && *p != '\n') {
            ++p;
        }
        if (*p == ',') {
            ++p;
        }
    }
    return total;
}

// Uses the physical core count rather than the logical CPU count so that the
// default configuration does not oversubscribe SMT siblings. The thread count
// is also limited by the amount of work, since thread startup would otherwise
// dominate very small simulations.
static int defaultThreadCount(const int numBodies, const int numSteps) {
    const int logical = omp_get_num_procs();
    if (logical <= 1) {
        return logical;
    }

    int smtWidth = 1;
    if (FILE* f = std::fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r")) {
        char buffer[256] = {};
        if (std::fgets(buffer, sizeof(buffer), f) != nullptr) {
            const int siblings = cpuListCount(buffer);
            if (siblings > 0) {
                smtWidth = siblings;
            }
        }
        std::fclose(f);
    }

    const int physical = logical / smtWidth > 0 ? logical / smtWidth : logical;

    const double bodies = static_cast<double>(numBodies > 0 ? numBodies : 0);
    const double steps = static_cast<double>(numSteps > 0 ? numSteps : 0);
    const double work = bodies * bodies * steps;
    const double wanted = work / 4.0e5;
    int threads = wanted >= static_cast<double>(physical) ? physical : static_cast<int>(wanted);
    if (threads < 1) {
        threads = 1;
    }
    return threads;
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

    // Respect an explicit thread count, otherwise avoid oversubscribing SMT.
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(defaultThreadCount(numBodies, numSteps));
    }

    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    simulate(bodies, numSteps);
    
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
