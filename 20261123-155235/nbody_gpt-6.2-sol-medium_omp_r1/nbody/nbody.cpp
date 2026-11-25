#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <utility>
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

int teamSize(size_t n) {
    // Give each worker enough interactions and target bodies to amortize barriers.
    constexpr double interactionsPerThread = 65536.0;
    constexpr size_t bodiesPerThread = 64;
    const double workLimit = std::ceil(n * static_cast<double>(n) / interactionsPerThread);
    const double targetLimit = static_cast<double>(n / bodiesPerThread);
    const double maxThreads = omp_get_max_threads();
    return static_cast<int>(std::max(1.0, std::min({workLimit, targetLimit, maxThreads})));
}

void simulate(std::vector<Body>& bodies, int numSteps) {
    const size_t n = bodies.size();
    if (n == 0 || numSteps <= 0) {
        return;
    }

    constexpr size_t tileSize = 4;
    std::vector<Body> nextBodies(n);

    // Each step reads a complete snapshot and writes the next one. The loop
    // barrier publishes that snapshot without a barrier between force and integration.
    #pragma omp parallel num_threads(teamSize(n))
    {
        Body* current = bodies.data();
        Body* next = nextBodies.data();
        for (int step = 0; step < numSteps; ++step) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; i += tileSize) {
                const size_t count = std::min(tileSize, n - i);
                double x[tileSize] = {}, y[tileSize] = {}, z[tileSize] = {};
                double fx[tileSize] = {}, fy[tileSize] = {}, fz[tileSize] = {};
                for (size_t lane = 0; lane < count; ++lane) {
                    x[lane] = current[i + lane].pos.x;
                    y[lane] = current[i + lane].pos.y;
                    z[lane] = current[i + lane].pos.z;
                }

                // Vectorize across targets so that each force still sums sources in order.
                for (size_t j = 0; j < n; ++j) {
                    const Vec3 source = current[j].pos;
                    #pragma omp simd
                    for (size_t lane = 0; lane < tileSize; ++lane) {
                        const double dx = source.x - x[lane];
                        const double dy = source.y - y[lane];
                        const double dz = source.z - z[lane];
                        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                        const double invDist = 1.0 / std::sqrt(distSqr);
                        const double invDist3 = invDist * invDist * invDist;
                        fx[lane] += dx * invDist3;
                        fy[lane] += dy * invDist3;
                        fz[lane] += dz * invDist3;
                    }
                }

                for (size_t lane = 0; lane < count; ++lane) {
                    Body body = current[i + lane];
                    // Round the velocity increments before adding them, as in the serial kernel.
                    const volatile Vec3 kick{DT * fx[lane], DT * fy[lane], DT * fz[lane]};
                    body.vel.x += kick.x;
                    body.vel.y += kick.y;
                    body.vel.z += kick.z;
                    body.pos.x += body.vel.x * DT;
                    body.pos.y += body.vel.y * DT;
                    body.pos.z += body.vel.z * DT;
                    next[i + lane] = body;
                }
            }
            std::swap(current, next);
        }
    }

    if (numSteps % 2 != 0) {
        bodies.swap(nextBodies);
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
    
    // Distribute rows cyclically to balance the decreasing number of pairs per row.
    std::vector<double> potentialEnergy(n);
    #pragma omp parallel for schedule(static, 1) num_threads(teamSize(n))
    for (size_t i = 0; i < n; ++i) {
        double potential = 0.0;
        #pragma omp simd reduction(+:potential)
        for (size_t j = i + 1; j < n; ++j) {
            const Vec3& source = bodies[j].pos;
            const double dx = source.x - bodies[i].pos.x;
            const double dy = source.y - bodies[i].pos.y;
            const double dz = source.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potential -= 1.0 / dist;
        }
        potentialEnergy[i] = potential;
    }
    for (double potential : potentialEnergy) {
        energy += potential;
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
