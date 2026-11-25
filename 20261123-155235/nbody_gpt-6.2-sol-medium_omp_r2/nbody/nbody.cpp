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
constexpr size_t FORCE_BLOCK_SIZE = 8;

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

struct Positions {
    std::vector<double> x, y, z;

    explicit Positions(size_t n) : x(n), y(n), z(n) {}
};

int teamSize(size_t targets, size_t workItems) {
    // Keep enough work per thread to amortize team startup and step barriers.
    constexpr size_t targetsPerThread = 64;
    const size_t usefulThreads = std::max(size_t{2}, targets / targetsPerThread);
    return static_cast<int>(std::min({usefulThreads, workItems, size_t(omp_get_max_threads())}));
}

void advanceBodies(std::vector<Body>& bodies, const Positions& current, Positions& next, size_t first) {
    double fx[FORCE_BLOCK_SIZE]{}, fy[FORCE_BLOCK_SIZE]{}, fz[FORCE_BLOCK_SIZE]{};
    const size_t n = bodies.size();

    // Each SIMD lane accumulates one body's forces in the original source order.
    for (size_t j = 0; j < n; ++j) {
        const double x = current.x[j], y = current.y[j], z = current.z[j];
        #pragma omp simd
        for (size_t lane = 0; lane < FORCE_BLOCK_SIZE; ++lane) {
            const size_t i = first + lane;
            const double dx = x - current.x[i];
            const double dy = y - current.y[i];
            const double dz = z - current.z[i];
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            fx[lane] += dx * invDist3;
            fy[lane] += dy * invDist3;
            fz[lane] += dz * invDist3;
        }
    }

    const size_t active = std::min(FORCE_BLOCK_SIZE, n - first);
    for (size_t lane = 0; lane < active; ++lane) {
        const size_t i = first + lane;
        auto& vel = bodies[i].vel;
        #if defined(__GNUC__) && !defined(__clang__)
        // GCC's original force loop rounds the timestep product before the velocity addition.
        const volatile double dvx = DT * fx[lane], dvy = DT * fy[lane], dvz = DT * fz[lane];
        vel.x += dvx;
        vel.y += dvy;
        vel.z += dvz;
        #else
        vel.x += DT * fx[lane];
        vel.y += DT * fy[lane];
        vel.z += DT * fz[lane];
        #endif
        next.x[i] = current.x[i] + vel.x * DT;
        next.y[i] = current.y[i] + vel.y * DT;
        next.z[i] = current.z[i] + vel.z * DT;
    }
}

void simulate(std::vector<Body>& bodies, int numSteps) {
    const size_t n = bodies.size();
    if (n == 0 || numSteps <= 0) {
        return;
    }

    const size_t blocks = (n + FORCE_BLOCK_SIZE - 1) / FORCE_BLOCK_SIZE;
    const size_t paddedSize = blocks * FORCE_BLOCK_SIZE;
    // Pad the target coordinates so the last SIMD block can use the same force loop.
    Positions positions[] = {Positions(paddedSize), Positions(paddedSize)};
    for (size_t i = 0; i < n; ++i) {
        positions[0].x[i] = bodies[i].pos.x;
        positions[0].y[i] = bodies[i].pos.y;
        positions[0].z[i] = bodies[i].pos.z;
    }

    #pragma omp parallel num_threads(teamSize(n, blocks))
    {
        for (int step = 0; step < numSteps; ++step) {
            const Positions& current = positions[step % 2];
            Positions& next = positions[1 - step % 2];
            // The loop barrier publishes every next position before the buffers exchange roles.
            #pragma omp for schedule(static)
            for (size_t block = 0; block < blocks; ++block) {
                advanceBodies(bodies, current, next, block * FORCE_BLOCK_SIZE);
            }
        }
    }

    const Positions& final = positions[numSteps % 2];
    for (size_t i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(final.x[i], final.y[i], final.z[i]);
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
    
    // Cyclic scheduling balances the decreasing number of pairs in each row.
    // Combine the row sums in a fixed order to keep the result independent of team size.
    std::vector<double> rowEnergy(n);
    #pragma omp parallel for schedule(static, 1) num_threads(teamSize(n / 2, std::max(n, size_t{1})))
    for (size_t i = 0; i < n; ++i) {
        double potential = 0.0;
        #pragma omp simd reduction(+:potential)
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potential -= 1.0 / dist;
        }
        rowEnergy[i] = potential;
    }
    for (double potential : rowEnergy) {
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
