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

    void set(size_t i, const Vec3& pos) {
        x[i] = pos.x;
        y[i] = pos.y;
        z[i] = pos.z;
    }
};

static inline double updateVelocity(double velocity, double force) {
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
    // GCC on x86-64 rounded the kick separately in the original force loop.
    const volatile double kick = DT * force;
    return velocity + kick;
#else
    return velocity + DT * force;
#endif
}

void simulate(std::vector<Body>& bodies, int steps) {
    const size_t n = bodies.size();
    if (steps <= 0 || n == 0) {
        return;
    }

    constexpr size_t blockSize = 8;
    const size_t blocks = (n + blockSize - 1) / blockSize;
    // Amortize team and barrier costs on small problems.
    constexpr size_t interactionsPerThread = 65536;
    const size_t minBodiesPerThread = (interactionsPerThread + n - 1) / n;
    const size_t workThreads = 1 + (n - 1) / minBodiesPerThread;
    const int threads = std::min({static_cast<size_t>(omp_get_max_threads()), blocks, workThreads});
    // Pad the targets so the last block can use the same SIMD loop as the others.
    Positions first(blocks * blockSize), second(blocks * blockSize);

    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            first.set(i, bodies[i].pos);
        }

        // Every update reads the same position snapshot. The worksharing
        // barrier completes the next snapshot before any thread can use it.
        const Positions* current = &first;
        Positions* next = &second;
        for (int step = 0; step < steps; ++step) {
            #pragma omp for schedule(static)
            for (size_t block = 0; block < blocks; ++block) {
                const size_t begin = block * blockSize;
                const size_t count = std::min(blockSize, n - begin);
                const double* x = current->x.data();
                const double* y = current->y.data();
                const double* z = current->z.data();
                double fx[blockSize] = {}, fy[blockSize] = {}, fz[blockSize] = {};

                for (size_t j = 0; j < n; ++j) {
                    const double sourceX = x[j], sourceY = y[j], sourceZ = z[j];
                    // Vectorize across targets to preserve each body's sum order.
                    #pragma omp simd
                    for (size_t lane = 0; lane < blockSize; ++lane) {
                        const double dx = sourceX - x[begin + lane];
                        const double dy = sourceY - y[begin + lane];
                        const double dz = sourceZ - z[begin + lane];
                        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                        const double invDist = 1.0 / std::sqrt(distSqr);
                        const double invDist3 = invDist * invDist * invDist;

                        fx[lane] += dx * invDist3;
                        fy[lane] += dy * invDist3;
                        fz[lane] += dz * invDist3;
                    }
                }

                for (size_t lane = 0; lane < count; ++lane) {
                    const size_t i = begin + lane;
                    auto& vel = bodies[i].vel;
                    vel.x = updateVelocity(vel.x, fx[lane]);
                    vel.y = updateVelocity(vel.y, fy[lane]);
                    vel.z = updateVelocity(vel.z, fz[lane]);
                    next->x[i] = x[i] + vel.x * DT;
                    next->y[i] = y[i] + vel.y * DT;
                    next->z[i] = z[i] + vel.z * DT;
                }
            }
            current = next;
            next = next == &first ? &second : &first;
        }

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < n; ++i) {
            bodies[i].pos = {current->x[i], current->y[i], current->z[i]};
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
    if (n < 2) {
        return energy;
    }

    // Number the pairs in the same order as the original triangular loop.
    const auto rowStart = [n](std::uint64_t i) { return i * (2 * std::uint64_t{n} - i - 1) / 2; };
    constexpr size_t pairsPerBlock = 4096;
    const std::uint64_t pairs = rowStart(n - 1);
    const std::uint64_t blocks = (pairs + pairsPerBlock - 1) / pairsPerBlock;
    // The ordered sum limits how much parallel pair evaluation can help.
    const int threads = std::min({std::uint64_t{8}, static_cast<std::uint64_t>(omp_get_max_threads()), blocks});
    Positions positions(n);
    const double* x = positions.x.data();
    const double* y = positions.y.data();
    const double* z = positions.z.data();

    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            positions.set(i, bodies[i].pos);
        }

        double potential[pairsPerBlock];
        #pragma omp for ordered schedule(static, 1)
        for (std::uint64_t block = 0; block < blocks; ++block) {
            const std::uint64_t begin = block * pairsPerBlock;
            const size_t count = std::min<std::uint64_t>(pairsPerBlock, pairs - begin);

            size_t i = 0, upper = n - 1;
            while (i + 1 < upper) {
                const size_t mid = i + (upper - i) / 2;
                if (rowStart(mid) <= begin) {
                    i = mid;
                } else {
                    upper = mid;
                }
            }

            size_t j = i + 1 + (begin - rowStart(i));
            for (size_t offset = 0; offset < count; ++i, j = i + 1) {
                const double targetX = x[i], targetY = y[i], targetZ = z[i];
                const size_t run = std::min(n - j, count - offset);
                #pragma omp simd
                for (size_t lane = 0; lane < run; ++lane) {
                    const double dx = x[j + lane] - targetX;
                    const double dy = y[j + lane] - targetY;
                    const double dz = z[j + lane] - targetZ;
                    const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                    potential[offset + lane] = 1.0 / dist;
                }
                offset += run;
            }

            #pragma omp ordered
            {
                for (size_t k = 0; k < count; ++k) {
                    energy -= potential[k];
                }
            }
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
