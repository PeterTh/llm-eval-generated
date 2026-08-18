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
constexpr size_t MIN_BODIES_PER_THREAD = 128;
#if defined(__AVX512F__)
constexpr size_t FORCE_BLOCK_SIZE = 8;
#elif defined(__AVX__)
constexpr size_t FORCE_BLOCK_SIZE = 4;
#else
constexpr size_t FORCE_BLOCK_SIZE = 2;
#endif

struct Bodies {
    explicit Bodies(const size_t count)
        : positionX(count), positionY(count), positionZ(count),
          velocityX(count), velocityY(count), velocityZ(count) {}

    [[nodiscard]] size_t size() const noexcept { return positionX.size(); }

    // Structure-of-arrays storage gives the O(n^2) loop dense, independent
    // streams and prevents unused velocities from entering the read-only
    // position working set shared by all cores.
    std::vector<double> positionX;
    std::vector<double> positionY;
    std::vector<double> positionZ;
    std::vector<double> velocityX;
    std::vector<double> velocityY;
    std::vector<double> velocityZ;
};

void randomizeBodies(Bodies& bodies, unsigned int seed = 42) {
    for (size_t i = 0; i < bodies.size(); ++i) {
        bodies.positionX[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.positionY[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.positionZ[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.velocityX[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.velocityY[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.velocityZ[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// Called collectively by every thread in the simulation's OpenMP team.
void computeForces(Bodies& bodies) {
    const size_t n = bodies.size();
    const double* const positionX = bodies.positionX.data();
    const double* const positionY = bodies.positionY.data();
    const double* const positionZ = bodies.positionZ.data();
    double* const velocityX = bodies.velocityX.data();
    double* const velocityY = bodies.velocityY.data();
    double* const velocityZ = bodies.velocityZ.data();
    const size_t numBlocks = (n + FORCE_BLOCK_SIZE - 1) / FORCE_BLOCK_SIZE;

    #pragma omp for schedule(static)
    for (size_t block = 0; block < numBlocks; ++block) {
        const size_t first = block * FORCE_BLOCK_SIZE;
        const int blockSize = static_cast<int>(std::min(FORCE_BLOCK_SIZE, n - first));
        double Fx[FORCE_BLOCK_SIZE] = {};
        double Fy[FORCE_BLOCK_SIZE] = {};
        double Fz[FORCE_BLOCK_SIZE] = {};

        // Vectorize across independent destination bodies. Each lane retains
        // the original j-order of accumulation, avoiding a reordered force
        // reduction while still evaluating several square roots at once.
        for (size_t j = 0; j < n; ++j) {
            const double sourceX = positionX[j];
            const double sourceY = positionY[j];
            const double sourceZ = positionZ[j];

            #pragma omp simd simdlen(FORCE_BLOCK_SIZE)
            for (int lane = 0; lane < blockSize; ++lane) {
                const double dx = sourceX - positionX[first + lane];
                const double dy = sourceY - positionY[first + lane];
                const double dz = sourceZ - positionZ[first + lane];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx[lane] += dx * invDist3;
                Fy[lane] += dy * invDist3;
                Fz[lane] += dz * invDist3;
            }
        }

        for (int lane = 0; lane < blockSize; ++lane) {
            // The original update rounds the scaled force before adding it to
            // velocity. The volatile temporaries prevent contraction into a
            // fused multiply-add after the force loop is vectorized.
            const volatile double deltaX = DT * Fx[lane];
            const volatile double deltaY = DT * Fy[lane];
            const volatile double deltaZ = DT * Fz[lane];
            velocityX[first + lane] += deltaX;
            velocityY[first + lane] += deltaY;
            velocityZ[first + lane] += deltaZ;
        }
    }
}

void integrateBodies(Bodies& bodies) {
    const size_t n = bodies.size();

    #pragma omp for simd schedule(static)
    for (size_t i = 0; i < n; ++i) {
        bodies.positionX[i] += bodies.velocityX[i] * DT;
        bodies.positionY[i] += bodies.velocityY[i] * DT;
        bodies.positionZ[i] += bodies.velocityZ[i] * DT;
    }
}

void simulate(Bodies& bodies, const int numSteps) {
    // Very small chunks spend more time in barriers than in force evaluation.
    // Treat OMP_NUM_THREADS as an upper bound and keep enough destinations on
    // each thread to amortize worksharing overhead.
    const size_t workLimitedThreads = std::max<size_t>(
        1, (bodies.size() + MIN_BODIES_PER_THREAD - 1) / MIN_BODIES_PER_THREAD);
    const int numThreads = static_cast<int>(
        std::min(static_cast<size_t>(omp_get_max_threads()), workLimitedThreads));

    // Keep one thread team alive for the complete simulation. The implicit
    // barrier at the end of each worksharing loop separates force evaluation
    // from integration and each timestep from the next one.
    #pragma omp parallel default(none) shared(bodies, numSteps) num_threads(numThreads)
    {
        for (int step = 0; step < numSteps; ++step) {
            computeForces(bodies);
            integrateBodies(bodies);
        }
    }
}

double computeTotalEnergy(const Bodies& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies.velocityX[i] * bodies.velocityX[i] +
                        bodies.velocityY[i] * bodies.velocityY[i] +
                        bodies.velocityZ[i] * bodies.velocityZ[i]);
    }
    
    // Potential energy (assuming unit mass for all bodies)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies.positionX[j] - bodies.positionX[i];
            const double dy = bodies.positionY[j] - bodies.positionY[i];
            const double dz = bodies.positionZ[j] - bodies.positionZ[i];
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
        if (!std::isfinite(bodies.positionX[i]) || !std::isfinite(bodies.positionY[i]) ||
            !std::isfinite(bodies.positionZ[i]) || !std::isfinite(bodies.velocityX[i]) ||
            !std::isfinite(bodies.velocityY[i]) || !std::isfinite(bodies.velocityZ[i])) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(bodies.positionX[i]) > maxPos || std::abs(bodies.positionY[i]) > maxPos ||
            std::abs(bodies.positionZ[i]) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(bodies.velocityX[i]) > maxVel || std::abs(bodies.velocityY[i]) > maxVel ||
            std::abs(bodies.velocityZ[i]) > maxVel) {
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
    Bodies bodies(numBodies);
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
        for (size_t i = 0; i < bodies.size(); ++i) {
            bodyData.push_back(bodies.positionX[i]);
            bodyData.push_back(bodies.positionY[i]);
            bodyData.push_back(bodies.positionZ[i]);
            bodyData.push_back(bodies.velocityX[i]);
            bodyData.push_back(bodies.velocityY[i]);
            bodyData.push_back(bodies.velocityZ[i]);
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
