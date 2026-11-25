#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

struct Bodies {
    std::vector<Vec3> positions;
    std::vector<Vec3> velocities;
};

void randomizeBodies(Bodies& bodies, unsigned int seed = 42) {
    for (size_t i = 0; i < bodies.positions.size(); ++i) {
        Vec3& pos = bodies.positions[i];
        Vec3& vel = bodies.velocities[i];
        pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

constexpr int TARGETS_PER_BLOCK = 8;

int workerCount(size_t n, uint64_t work) {
    // Amortize team and barrier overhead on small inputs.
    constexpr size_t targetsPerThread = 64;
    constexpr uint64_t workPerThread = 32768;
    const auto threads = std::min<uint64_t>({static_cast<uint64_t>(omp_get_max_threads()),
                                            n / targetsPerThread, work / workPerThread});
    return static_cast<int>(std::max<uint64_t>(1, threads));
}

void advanceBlock(const Vec3* positions, Vec3* nextPositions, Vec3* velocities,
                  size_t n, size_t begin) {
    const int count = static_cast<int>(std::min(n - begin, static_cast<size_t>(TARGETS_PER_BLOCK)));
    double x[TARGETS_PER_BLOCK]{}, y[TARGETS_PER_BLOCK]{}, z[TARGETS_PER_BLOCK]{};
    double fx[TARGETS_PER_BLOCK]{}, fy[TARGETS_PER_BLOCK]{}, fz[TARGETS_PER_BLOCK]{};

    for (int lane = 0; lane < count; ++lane) {
        const Vec3& pos = positions[begin + lane];
        x[lane] = pos.x;
        y[lane] = pos.y;
        z[lane] = pos.z;
    }

    for (size_t j = 0; j < n; ++j) {
        const Vec3 source = positions[j];
        // Each lane owns one target and visits sources in the original order.
        #pragma omp simd
        for (int lane = 0; lane < TARGETS_PER_BLOCK; ++lane) {
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

#if defined(__GNUC__) && !defined(__clang__)
    // Preserve the separate multiplication emitted by GCC in the original force pass.
    #pragma omp simd
    for (int lane = 0; lane < TARGETS_PER_BLOCK; ++lane) {
        fx[lane] *= DT;
        fy[lane] *= DT;
        fz[lane] *= DT;
    }
    constexpr double velocityScale = 1.0;
#else
    constexpr double velocityScale = DT;
#endif

    for (int lane = 0; lane < count; ++lane) {
        const size_t i = begin + lane;
        Vec3& vel = velocities[i];
        vel.x += velocityScale * fx[lane];
        vel.y += velocityScale * fy[lane];
        vel.z += velocityScale * fz[lane];
        nextPositions[i] = {x[lane] + vel.x * DT, y[lane] + vel.y * DT, z[lane] + vel.z * DT};
    }
}

void simulate(Bodies& bodies, int numSteps) {
    const size_t n = bodies.positions.size();
    if (n == 0 || numSteps <= 0) {
        return;
    }

    std::vector<Vec3> nextPositions(n);
    Vec3* const positions[] = {bodies.positions.data(), nextPositions.data()};
    Vec3* const velocities = bodies.velocities.data();
    const size_t blocks = (n + TARGETS_PER_BLOCK - 1) / TARGETS_PER_BLOCK;
    const int threads = workerCount(n, static_cast<uint64_t>(n) * n);

    #pragma omp parallel num_threads(threads)
    {
        for (int step = 0; step < numSteps; ++step) {
            const Vec3* current = positions[step % 2];
            Vec3* next = positions[1 - step % 2];
            // All reads use one position buffer while the next step is written to the other.
            #pragma omp for schedule(static)
            for (size_t block = 0; block < blocks; ++block) {
                advanceBlock(current, next, velocities, n, block * TARGETS_PER_BLOCK);
            }
        }
    }

    if (numSteps % 2 != 0) {
        bodies.positions.swap(nextPositions);
    }
}

double computeTotalEnergy(const Bodies& bodies) {
    double energy = 0.0;
    const size_t n = bodies.positions.size();
    
    // Kinetic energy (assuming unit mass)
    for (const Vec3& vel : bodies.velocities) {
        energy += 0.5 * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
    }

    if (n < 2) {
        return energy;
    }
    const uint64_t pairs = static_cast<uint64_t>(n) * (n - 1) / 2;
    const int threads = workerCount(n, pairs);
    std::vector<double> rowEnergy(n);
    // Interleave outer iterations to balance the decreasing row lengths.
    #pragma omp parallel for schedule(static, 1) num_threads(threads)
    for (size_t i = 0; i < n; ++i) {
        const Vec3 target = bodies.positions[i];
        double potential = 0.0;
        #pragma omp simd reduction(+:potential)
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies.positions[j].x - target.x;
            const double dy = bodies.positions[j].y - target.y;
            const double dz = bodies.positions[j].z - target.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potential -= 1.0 / dist;
        }
        rowEnergy[i] = potential;
    }

    // Keep the final sum independent of the number of workers.
    for (double potential : rowEnergy) {
        energy += potential;
    }
    
    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const Bodies& bodies) {
    for (size_t i = 0; i < bodies.positions.size(); ++i) {
        const Vec3& pos = bodies.positions[i];
        const Vec3& vel = bodies.velocities[i];
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
    Bodies bodies{std::vector<Vec3>(numBodies), std::vector<Vec3>(numBodies)};
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
        bodyData.reserve(bodies.positions.size() * 6);
        for (size_t i = 0; i < bodies.positions.size(); ++i) {
            const Vec3& pos = bodies.positions[i];
            const Vec3& vel = bodies.velocities[i];
            bodyData.push_back(pos.x);
            bodyData.push_back(pos.y);
            bodyData.push_back(pos.z);
            bodyData.push_back(vel.x);
            bodyData.push_back(vel.y);
            bodyData.push_back(vel.z);
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
