#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// Separate component arrays keep the hot position streams contiguous.  Besides
// reducing the amount of data brought into cache by the force kernel, this
// layout lets the compiler evaluate several target bodies with packed SIMD
// instructions without changing the summation order for any one body.
struct Bodies {
    explicit Bodies(const size_t count)
        : posX(count), posY(count), posZ(count), velX(count), velY(count), velZ(count) {}

    [[nodiscard]] size_t size() const noexcept { return posX.size(); }

    std::vector<double> posX;
    std::vector<double> posY;
    std::vector<double> posZ;
    std::vector<double> velX;
    std::vector<double> velY;
    std::vector<double> velZ;
};

void randomizeBodies(Bodies& bodies, unsigned int seed = 42) {
    for (size_t i = 0; i < bodies.size(); ++i) {
        bodies.posX[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        bodies.posY[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        bodies.posZ[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        bodies.velX[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        bodies.velY[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        bodies.velZ[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

void simulate(Bodies& bodies, const int numSteps) {
    constexpr size_t FORCE_BLOCK_SIZE = 8;
    constexpr size_t MIN_BODIES_PER_THREAD = 64;
    const size_t n = bodies.size();
    const size_t usefulThreads = std::max<size_t>(1, (n + MIN_BODIES_PER_THREAD - 1) /
                                                        MIN_BODIES_PER_THREAD);
    const int threadCount = static_cast<int>(
        std::min(usefulThreads, static_cast<size_t>(omp_get_max_threads())));

    double* const __restrict posX = bodies.posX.data();
    double* const __restrict posY = bodies.posY.data();
    double* const __restrict posZ = bodies.posZ.data();
    double* const __restrict velX = bodies.velX.data();
    double* const __restrict velY = bodies.velY.data();
    double* const __restrict velZ = bodies.velZ.data();

    // One long-lived team avoids two fork/join cycles on every time step.
    // Static scheduling is balanced because every target body performs the
    // same number of interactions and it also keeps scheduling overhead tiny.
#pragma omp parallel default(none) shared(posX, posY, posZ, velX, velY, velZ)             \
    firstprivate(n, numSteps, threadCount, FORCE_BLOCK_SIZE) num_threads(threadCount)
    {
        for (int step = 0; step < numSteps; ++step) {
#pragma omp for schedule(static)
            for (size_t block = 0; block < n; block += FORCE_BLOCK_SIZE) {
                const size_t blockSize = std::min(FORCE_BLOCK_SIZE, n - block);
                alignas(64) double bodyX[FORCE_BLOCK_SIZE];
                alignas(64) double bodyY[FORCE_BLOCK_SIZE];
                alignas(64) double bodyZ[FORCE_BLOCK_SIZE];
                alignas(64) double forceX[FORCE_BLOCK_SIZE] = {};
                alignas(64) double forceY[FORCE_BLOCK_SIZE] = {};
                alignas(64) double forceZ[FORCE_BLOCK_SIZE] = {};

                for (size_t lane = 0; lane < blockSize; ++lane) {
                    bodyX[lane] = posX[block + lane];
                    bodyY[lane] = posY[block + lane];
                    bodyZ[lane] = posZ[block + lane];
                }

                for (size_t j = 0; j < n; ++j) {
                    const double sourceX = posX[j];
                    const double sourceY = posY[j];
                    const double sourceZ = posZ[j];

#pragma omp simd aligned(bodyX, bodyY, bodyZ, forceX, forceY, forceZ : 64)
                    for (size_t lane = 0; lane < blockSize; ++lane) {
                        const double dx = sourceX - bodyX[lane];
                        const double dy = sourceY - bodyY[lane];
                        const double dz = sourceZ - bodyZ[lane];
                        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                        const double invDist = 1.0 / std::sqrt(distSqr);
                        const double invDist3 = invDist * invDist * invDist;

                        forceX[lane] += dx * invDist3;
                        forceY[lane] += dy * invDist3;
                        forceZ[lane] += dz * invDist3;
                    }
                }

                for (size_t lane = 0; lane < blockSize; ++lane) {
                    const size_t i = block + lane;
                    // Preserve the original multiply-then-add rounding here;
                    // the original AoS kernel was compiled as two operations.
                    volatile double deltaVelocityX = DT * forceX[lane];
                    volatile double deltaVelocityY = DT * forceY[lane];
                    volatile double deltaVelocityZ = DT * forceZ[lane];
                    velX[i] += deltaVelocityX;
                    velY[i] += deltaVelocityY;
                    velZ[i] += deltaVelocityZ;
                }
            }

            // The force loop's implicit barrier keeps positions read-only until
            // every velocity is ready.  This loop's barrier likewise ensures
            // the next step sees a complete position update.
#pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                posX[i] += velX[i] * DT;
                posY[i] += velY[i] * DT;
                posZ[i] += velZ[i] * DT;
            }
        }
    }
}

double computeTotalEnergy(const Bodies& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies.velX[i] * bodies.velX[i] +
                        bodies.velY[i] * bodies.velY[i] +
                        bodies.velZ[i] * bodies.velZ[i]);
    }

    // Potential energy (assuming unit mass for all bodies)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies.posX[j] - bodies.posX[i];
            const double dy = bodies.posY[j] - bodies.posY[i];
            const double dz = bodies.posZ[j] - bodies.posZ[i];
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
        if (!std::isfinite(bodies.posX[i]) || !std::isfinite(bodies.posY[i]) ||
            !std::isfinite(bodies.posZ[i]) || !std::isfinite(bodies.velX[i]) ||
            !std::isfinite(bodies.velY[i]) || !std::isfinite(bodies.velZ[i])) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        constexpr double maxPos = 1e6;
        constexpr double maxVel = 1e6;
        if (std::abs(bodies.posX[i]) > maxPos || std::abs(bodies.posY[i]) > maxPos ||
            std::abs(bodies.posZ[i]) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(bodies.velX[i]) > maxVel || std::abs(bodies.velY[i]) > maxVel ||
            std::abs(bodies.velZ[i]) > maxVel) {
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
    const auto start = std::chrono::high_resolution_clock::now();
    simulate(bodies, numSteps);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities in the original body order.
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
        for (size_t i = 0; i < bodies.size(); ++i) {
            bodyData.push_back(bodies.posX[i]);
            bodyData.push_back(bodies.posY[i]);
            bodyData.push_back(bodies.posZ[i]);
            bodyData.push_back(bodies.velX[i]);
            bodyData.push_back(bodies.velY[i]);
            bodyData.push_back(bodies.velZ[i]);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
            // Report final energy for reference
            const double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            return 0;
        }

        printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
