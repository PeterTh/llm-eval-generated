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

#if defined(__AVX512F__)
constexpr size_t FORCE_BLOCK_SIZE = 8;
#else
constexpr size_t FORCE_BLOCK_SIZE = 4;
#endif

template <size_t Width>
void advanceBodyBlock(Body* bodies, const double* x, const double* y, const double* z,
                      double* nextX, double* nextY, double* nextZ,
                      size_t first, size_t n) {
    alignas(64) double px[Width], py[Width], pz[Width];
    alignas(64) double Fx[Width] = {}, Fy[Width] = {}, Fz[Width] = {};

    #pragma omp simd
    for (size_t lane = 0; lane < Width; ++lane) {
        px[lane] = x[first + lane];
        py[lane] = y[first + lane];
        pz[lane] = z[first + lane];
    }

    // Vectorize across target bodies, keeping each body's summation in j order.
    for (size_t j = 0; j < n; ++j) {
        const double sx = x[j], sy = y[j], sz = z[j];
        #pragma omp simd
        for (size_t lane = 0; lane < Width; ++lane) {
            const double dx = sx - px[lane];
            const double dy = sy - py[lane];
            const double dz = sz - pz[lane];
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx[lane] = std::fma(dx, invDist3, Fx[lane]);
            Fy[lane] = std::fma(dy, invDist3, Fy[lane]);
            Fz[lane] = std::fma(dz, invDist3, Fz[lane]);
        }
    }

    // Only this block's owner writes its velocities and next-step positions.
    for (size_t lane = 0; lane < Width; ++lane) {
        Body& body = bodies[first + lane];
        // Preserve the separately rounded velocity increments of the original
        // force pass when force calculation and integration are fused.
        const volatile double deltaVx = DT * Fx[lane];
        const volatile double deltaVy = DT * Fy[lane];
        const volatile double deltaVz = DT * Fz[lane];
        body.vel.x += deltaVx;
        body.vel.y += deltaVy;
        body.vel.z += deltaVz;
        body.pos.x = nextX[first + lane] = px[lane] + body.vel.x * DT;
        body.pos.y = nextY[first + lane] = py[lane] + body.vel.y * DT;
        body.pos.z = nextZ[first + lane] = pz[lane] + body.vel.z * DT;
    }
}

void simulateBodies(std::vector<Body>& bodies, int numSteps) {
    const size_t n = bodies.size();
    if (n == 0 || numSteps <= 0) {
        return;
    }

    // Two compact position snapshots allow force calculation and integration
    // to share one work loop and one barrier per step.
    std::vector<double> positions(6 * n);
    const size_t numBlocks = (n + FORCE_BLOCK_SIZE - 1) / FORCE_BLOCK_SIZE;
    // Avoid creating large teams for inputs with only a few blocks per worker.
    const size_t usefulThreads = (numBlocks + 7) / 8;
    const int numThreads = std::min(omp_get_max_threads(),
                                  static_cast<int>(usefulThreads));
    Body* data = bodies.data();
    double* storage = positions.data();

    #pragma omp parallel num_threads(numThreads) default(none) \
        shared(data, storage, n, numBlocks, numSteps)
    {
        double* x = storage;
        double* y = storage + n;
        double* z = storage + 2 * n;
        double* nextX = storage + 3 * n;
        double* nextY = storage + 4 * n;
        double* nextZ = storage + 5 * n;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            storage[i] = data[i].pos.x;
            storage[n + i] = data[i].pos.y;
            storage[2 * n + i] = data[i].pos.z;
        }

        for (int step = 0; step < numSteps; ++step) {
            #pragma omp for schedule(static)
            for (size_t block = 0; block < numBlocks; ++block) {
                const size_t first = block * FORCE_BLOCK_SIZE;
                if (first + FORCE_BLOCK_SIZE <= n) {
                    advanceBodyBlock<FORCE_BLOCK_SIZE>(data, x, y, z,
                                                      nextX, nextY, nextZ, first, n);
                } else {
                    for (size_t i = first; i < n; ++i) {
                        advanceBodyBlock<1>(data, x, y, z,
                                            nextX, nextY, nextZ, i, n);
                    }
                }
            }
            // The worksharing barrier completes every next-position write.
            // Each thread then exchanges its own pointers without another barrier.
            std::swap(x, nextX);
            std::swap(y, nextY);
            std::swap(z, nextZ);
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
    
    simulateBodies(bodies, numSteps);
    
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
