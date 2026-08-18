#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// Retain the original state layout so the OpenMP implementation remains
// bit-for-bit compatible with the serial floating-point trajectory.
struct Bodies {
    explicit Bodies(const size_t count)
        : items(count) {}

    size_t size() const noexcept { return items.size(); }

    std::vector<Body> items;
};

void randomizeBodies(Bodies& bodies, unsigned int seed = 42) {
    for (size_t i = 0; i < bodies.size(); ++i) {
        // Retain the original per-body random-number ordering.
        bodies.items[i].pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.items[i].pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.items[i].pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.items[i].vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.items[i].vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.items[i].vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

void simulateBodies(Bodies& bodies, const int numSteps) {
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(bodies.size());
    Body* const data = bodies.items.data();

    // Very small chunks lose more time at the two step barriers than they
    // gain from another worker.  Respect the user's OpenMP limit while
    // avoiding oversubscription on machines with large default teams.
    constexpr std::ptrdiff_t MIN_BODIES_PER_THREAD = 64;
    const std::ptrdiff_t usefulThreads =
        n > 0 ? (n + MIN_BODIES_PER_THREAD - 1) / MIN_BODIES_PER_THREAD : 1;
    const int maxThreads = omp_get_max_threads();
    const int teamSize = usefulThreads < maxThreads
        ? static_cast<int>(usefulThreads)
        : maxThreads;

    // Keep a single team alive for all steps.  Both implicit barriers are
    // required: all force reads must precede integration, and every position
    // update must complete before the next step begins.
#pragma omp parallel num_threads(teamSize) default(none) \
    shared(data) firstprivate(n, numSteps)
    {
        for (int step = 0; step < numSteps; ++step) {
#pragma omp for schedule(static)
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                const double x = data[i].pos.x;
                const double y = data[i].pos.y;
                const double z = data[i].pos.z;
                double forceX = 0.0;
                double forceY = 0.0;
                double forceZ = 0.0;

                for (std::ptrdiff_t j = 0; j < n; ++j) {
                    const double dx = data[j].pos.x - x;
                    const double dy = data[j].pos.y - y;
                    const double dz = data[j].pos.z - z;
                    const double distSqr =
                        dx * dx + dy * dy + dz * dz + SOFTENING;
                    const double invDist = 1.0 / std::sqrt(distSqr);
                    const double invDist3 = invDist * invDist * invDist;

                    forceX += dx * invDist3;
                    forceY += dy * invDist3;
                    forceZ += dz * invDist3;
                }

                data[i].vel.x += DT * forceX;
                data[i].vel.y += DT * forceY;
                data[i].vel.z += DT * forceZ;
            }

#pragma omp for schedule(static)
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                data[i].pos.x += data[i].vel.x * DT;
                data[i].pos.y += data[i].vel.y * DT;
                data[i].pos.z += data[i].vel.z * DT;
            }
        }
    }
}

double computeTotalEnergy(const Bodies& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies.items[i].vel.x * bodies.items[i].vel.x +
                        bodies.items[i].vel.y * bodies.items[i].vel.y +
                        bodies.items[i].vel.z * bodies.items[i].vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies.items[j].pos.x - bodies.items[i].pos.x;
            const double dy = bodies.items[j].pos.y - bodies.items[i].pos.y;
            const double dz = bodies.items[j].pos.z - bodies.items[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    
    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const Bodies& bodies) {
    for (size_t i = 0; i < bodies.size(); ++i) {
        // Check for NaN or Inf values
        if (!std::isfinite(bodies.items[i].pos.x) ||
            !std::isfinite(bodies.items[i].pos.y) ||
            !std::isfinite(bodies.items[i].pos.z) ||
            !std::isfinite(bodies.items[i].vel.x) ||
            !std::isfinite(bodies.items[i].vel.y) ||
            !std::isfinite(bodies.items[i].vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(bodies.items[i].pos.x) > maxPos ||
            std::abs(bodies.items[i].pos.y) > maxPos ||
            std::abs(bodies.items[i].pos.z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(bodies.items[i].vel.x) > maxVel ||
            std::abs(bodies.items[i].vel.y) > maxVel ||
            std::abs(bodies.items[i].vel.z) > maxVel) {
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
    Bodies bodies(static_cast<size_t>(numBodies));
    randomizeBodies(bodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    simulateBodies(bodies, numSteps);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (size_t i = 0; i < bodies.size(); ++i) {
            bodyData.push_back(bodies.items[i].pos.x);
            bodyData.push_back(bodies.items[i].pos.y);
            bodyData.push_back(bodies.items[i].pos.z);
            bodyData.push_back(bodies.items[i].vel.x);
            bodyData.push_back(bodies.items[i].vel.y);
            bodyData.push_back(bodies.items[i].vel.z);
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
