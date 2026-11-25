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

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

void randomizeBodies(std::vector<Vec3>& positions, std::vector<Vec3>& velocities,
                     unsigned int seed = 42) {
    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i].x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        positions[i].y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        positions[i].z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velocities[i].x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velocities[i].y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velocities[i].z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

constexpr size_t TARGETS_PER_BLOCK = 8;

double updateVelocity(double velocity, double force) {
    // GCC's original force loop rounded the velocity increment before adding it.
#if defined(__GNUC__) && !defined(__clang__)
    const volatile double increment = DT * force;
    return velocity + increment;
#else
    return velocity + DT * force;
#endif
}

void advanceBlock(const Vec3* positions, Vec3* nextPositions, Vec3* velocities,
                  size_t n, size_t first) {
    const size_t count = std::min(TARGETS_PER_BLOCK, n - first);
    double x[TARGETS_PER_BLOCK] = {}, y[TARGETS_PER_BLOCK] = {}, z[TARGETS_PER_BLOCK] = {};
    double fx[TARGETS_PER_BLOCK] = {}, fy[TARGETS_PER_BLOCK] = {}, fz[TARGETS_PER_BLOCK] = {};
    for (size_t lane = 0; lane < count; ++lane) {
        x[lane] = positions[first + lane].x;
        y[lane] = positions[first + lane].y;
        z[lane] = positions[first + lane].z;
    }

    // Vectorize across targets so each body's force still sums sources in order.
    for (size_t j = 0; j < n; ++j) {
        const Vec3 source = positions[j];
        #pragma omp simd
        for (size_t lane = 0; lane < TARGETS_PER_BLOCK; ++lane) {
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
        Vec3& vel = velocities[first + lane];
        vel.x = updateVelocity(vel.x, fx[lane]);
        vel.y = updateVelocity(vel.y, fy[lane]);
        vel.z = updateVelocity(vel.z, fz[lane]);
        nextPositions[first + lane] = {
            x[lane] + vel.x * DT, y[lane] + vel.y * DT, z[lane] + vel.z * DT
        };
    }
}

int teamSize(size_t n) {
    // Keep team overhead proportional to the amount of pairwise work.
    constexpr size_t BODIES_PER_THREAD = 64;
    return std::min<size_t>(omp_get_max_threads(), 1 + (n - 1) / BODIES_PER_THREAD);
}

void simulate(std::vector<Vec3>& positions, std::vector<Vec3>& velocities, int steps) {
    const size_t n = positions.size();
    if (steps <= 0 || n == 0) {
        return;
    }

    std::vector<Vec3> scratch(n);
    auto* current = positions.data();
    auto* next = scratch.data();
    auto* velocity = velocities.data();
    const int threads = teamSize(n);

    // Each step reads a shared snapshot and writes the next one. The worksharing
    // barrier completes all writes before each thread swaps its private pointers.
    #pragma omp parallel num_threads(threads) default(none) shared(n, steps, velocity) firstprivate(current, next)
    {
        for (int step = 0; step < steps; ++step) {
            #pragma omp for schedule(static)
            for (size_t first = 0; first < n; first += TARGETS_PER_BLOCK) {
                advanceBlock(current, next, velocity, n, first);
            }
            std::swap(current, next);
        }
    }

    if (steps % 2 != 0) {
        positions.swap(scratch);
    }
}

double computeTotalEnergy(const std::vector<Vec3>& positions, const std::vector<Vec3>& velocities) {
    const size_t n = positions.size();
    double energy = 0.0;
    for (const Vec3& vel : velocities) {
        energy += 0.5 * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
    }
    if (n < 2) {
        return energy;
    }

    std::vector<double> potential(n);
    // Interleave rows because the number of potential-energy pairs drops with i.
    #pragma omp parallel for schedule(static, 1) num_threads(teamSize(n))
    for (size_t i = 0; i < n; ++i) {
        const Vec3 target = positions[i];
        double rowEnergy = 0.0;
        #pragma omp simd reduction(+:rowEnergy)
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = positions[j].x - target.x;
            const double dy = positions[j].y - target.y;
            const double dz = positions[j].z - target.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            rowEnergy -= 1.0 / dist;
        }
        potential[i] = rowEnergy;
    }

    // Use a fixed order so the reported energy is independent of the team size.
    for (double rowEnergy : potential) {
        energy += rowEnergy;
    }
    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Vec3>& positions, const std::vector<Vec3>& velocities) {
    for (size_t i = 0; i < positions.size(); ++i) {
        const Vec3& pos = positions[i];
        const Vec3& vel = velocities[i];
        // Check for NaN or Inf values
        if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
            !std::isfinite(vel.x) || !std::isfinite(vel.y) || !std::isfinite(vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(pos.x) > maxPos || std::abs(pos.y) > maxPos || std::abs(pos.z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(vel.x) > maxVel || std::abs(vel.y) > maxVel || std::abs(vel.z) > maxVel) {
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
    std::vector<Vec3> positions(numBodies);
    std::vector<Vec3> velocities(numBodies);
    randomizeBodies(positions, velocities);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    simulate(positions, velocities, numSteps);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(positions.size() * 6);
        for (size_t i = 0; i < positions.size(); ++i) {
            bodyData.push_back(positions[i].x);
            bodyData.push_back(positions[i].y);
            bodyData.push_back(positions[i].z);
            bodyData.push_back(velocities[i].x);
            bodyData.push_back(velocities[i].y);
            bodyData.push_back(velocities[i].z);
        }
        print_results(bodyData, "Bodies");
    }
    
    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(positions, velocities)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(positions, velocities);
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
