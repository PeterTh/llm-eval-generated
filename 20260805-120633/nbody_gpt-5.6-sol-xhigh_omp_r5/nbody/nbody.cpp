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

struct Bodies {
  private:
    static constexpr size_t CACHE_LINE_BYTES = 64;
    static constexpr size_t DOUBLES_PER_CACHE_LINE = CACHE_LINE_BYTES / sizeof(double);

    static double* alignToCacheLine(double* pointer) noexcept {
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        const auto aligned = (address + CACHE_LINE_BYTES - 1) & ~(CACHE_LINE_BYTES - 1);
        return reinterpret_cast<double*>(aligned);
    }

    size_t count_;
    size_t stride_;
    std::vector<double> storage_;

  public:
    explicit Bodies(const size_t count)
        : count_(count),
          stride_((count + DOUBLES_PER_CACHE_LINE - 1) & ~(DOUBLES_PER_CACHE_LINE - 1)),
          storage_(6 * stride_ + DOUBLES_PER_CACHE_LINE - 1),
          posX(alignToCacheLine(storage_.data())),
          posY(posX + stride_),
          posZ(posY + stride_),
          velX(posZ + stride_),
          velY(velX + stride_),
          velZ(velY + stride_) {}

    [[nodiscard]] size_t size() const noexcept { return count_; }

    double* const posX;
    double* const posY;
    double* const posZ;
    double* const velX;
    double* const velY;
    double* const velZ;
};

void randomizeBodies(Bodies& bodies, unsigned int seed = 42) {
    for (size_t i = 0; i < bodies.size(); ++i) {
        bodies.posX[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.posY[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.posZ[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.velX[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.velY[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.velZ[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

void simulateBodies(Bodies& bodies, const int numSteps) {
    const size_t n = bodies.size();
    double* const posX = bodies.posX;
    double* const posY = bodies.posY;
    double* const posZ = bodies.posZ;
    double* const velX = bodies.velX;
    double* const velY = bodies.velY;
    double* const velZ = bodies.velZ;
    constexpr size_t SIMD_WIDTH = 8;
    constexpr size_t MIN_BODIES_PER_THREAD = 16;
    const size_t forceBlocks = (n + SIMD_WIDTH - 1) / SIMD_WIDTH;
    const size_t usefulThreads = std::max<size_t>(1, (n + MIN_BODIES_PER_THREAD - 1) /
                                                    MIN_BODIES_PER_THREAD);
    const int threadCount = static_cast<int>(
        std::min(usefulThreads, static_cast<size_t>(omp_get_max_threads())));

    // A single long-lived team avoids paying thread start-up costs at every
    // force and integration phase.  The implicit barriers are both required:
    // all forces must observe one position snapshot, and all positions must be
    // integrated before the next snapshot is used.
    #pragma omp parallel num_threads(threadCount)
    {
        for (int step = 0; step < numSteps; ++step) {
            #pragma omp for schedule(static)
            for (size_t block = 0; block < forceBlocks; ++block) {
                const size_t begin = block * SIMD_WIDTH;
                const size_t active = std::min(SIMD_WIDTH, n - begin);
                alignas(64) double x[SIMD_WIDTH];
                alignas(64) double y[SIMD_WIDTH];
                alignas(64) double z[SIMD_WIDTH];
                alignas(64) double fx[SIMD_WIDTH] = {};
                alignas(64) double fy[SIMD_WIDTH] = {};
                alignas(64) double fz[SIMD_WIDTH] = {};

                #pragma omp simd aligned(posX, posY, posZ, x, y, z:64) simdlen(8)
                for (size_t lane = 0; lane < active; ++lane) {
                    x[lane] = posX[begin + lane];
                    y[lane] = posY[begin + lane];
                    z[lane] = posZ[begin + lane];
                }

                for (size_t j = 0; j < n; ++j) {
                    const double sourceX = posX[j];
                    const double sourceY = posY[j];
                    const double sourceZ = posZ[j];

                    // Vectorize across independent destination bodies.  Each
                    // body's j loop remains ordered exactly as in the scalar
                    // implementation, avoiding a reassociated force sum.
                    #pragma omp simd aligned(x, y, z, fx, fy, fz:64) simdlen(8)
                    for (size_t lane = 0; lane < active; ++lane) {
                        const double dx = sourceX - x[lane];
                        const double dy = sourceY - y[lane];
                        const double dz = sourceZ - z[lane];
                        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                        const double invDist = 1.0 / std::sqrt(distSqr);
                        const double invDist3 = invDist * invDist * invDist;

                        fx[lane] += dx * invDist3;
                        fy[lane] += dy * invDist3;
                        fz[lane] += dz * invDist3;
                    }
                }

                #pragma omp simd aligned(velX, velY, velZ, fx, fy, fz:64) simdlen(8)
                for (size_t lane = 0; lane < active; ++lane) {
                    velX[begin + lane] += DT * fx[lane];
                    velY[begin + lane] += DT * fy[lane];
                    velZ[begin + lane] += DT * fz[lane];
                }
            }

            #pragma omp for simd schedule(static) aligned(posX, posY, posZ, velX, velY, velZ:64)
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
        if (!std::isfinite(bodies.posX[i]) || !std::isfinite(bodies.posY[i]) || !std::isfinite(bodies.posZ[i]) ||
            !std::isfinite(bodies.velX[i]) || !std::isfinite(bodies.velY[i]) || !std::isfinite(bodies.velZ[i])) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(bodies.posX[i]) > maxPos || std::abs(bodies.posY[i]) > maxPos || std::abs(bodies.posZ[i]) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(bodies.velX[i]) > maxVel || std::abs(bodies.velY[i]) > maxVel || std::abs(bodies.velZ[i]) > maxVel) {
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
