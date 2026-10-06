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

// Number of target bodies processed together; each lane accumulates its own
// forces in the original j order, so results are identical to the scalar code.
constexpr size_t LANES = 8;

// Structure-of-arrays position buffers (double-buffered so that positions can be
// integrated right after the force pass without a separate synchronization step).
struct PosSoA {
    std::vector<double> x, y, z;
    void resize(size_t n) { x.assign(n, 0.0); y.assign(n, 0.0); z.assign(n, 0.0); }
};

static inline void forceBlock(const double* __restrict px, const double* __restrict py,
                              const double* __restrict pz, size_t n, size_t i0,
                              double* __restrict Fx, double* __restrict Fy, double* __restrict Fz) {
    double xi[LANES], yi[LANES], zi[LANES];
    double fx[LANES], fy[LANES], fz[LANES];
#pragma omp simd
    for (size_t k = 0; k < LANES; ++k) {
        xi[k] = px[i0 + k]; yi[k] = py[i0 + k]; zi[k] = pz[i0 + k];
        fx[k] = 0.0; fy[k] = 0.0; fz[k] = 0.0;
    }
    for (size_t j = 0; j < n; ++j) {
        const double xj = px[j], yj = py[j], zj = pz[j];
#pragma omp simd
        for (size_t k = 0; k < LANES; ++k) {
            const double dx = xj - xi[k];
            const double dy = yj - yi[k];
            const double dz = zj - zi[k];
            // Explicit FMAs reproduce the contraction of the reference build bit-for-bit.
            const double distSqr = std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            fx[k] = std::fma(dx, invDist3, fx[k]);
            fy[k] = std::fma(dy, invDist3, fy[k]);
            fz[k] = std::fma(dz, invDist3, fz[k]);
        }
    }
    for (size_t k = 0; k < LANES; ++k) { Fx[k] = fx[k]; Fy[k] = fy[k]; Fz[k] = fz[k]; }
}

// Runs all simulation steps (force computation followed by integration) in a
// single parallel region with one barrier per step.
void simulate(std::vector<Body>& bodies, int numSteps) {
    const size_t n = bodies.size();
    if (n == 0 || numSteps <= 0) return;
    const size_t numBlocks = (n + LANES - 1) / LANES;
    const size_t padded = numBlocks * LANES;
    PosSoA buf[2];
    buf[0].resize(padded);
    buf[1].resize(padded);

    // Never spawn more threads than there are blocks of work: idle threads would
    // only add barrier overhead.
    const size_t maxThreads = static_cast<size_t>(omp_get_max_threads());
    const int numThreads = static_cast<int>(numBlocks < maxThreads ? numBlocks : maxThreads);

#pragma omp parallel num_threads(numThreads)
    {
#pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            buf[0].x[i] = bodies[i].pos.x;
            buf[0].y[i] = bodies[i].pos.y;
            buf[0].z[i] = bodies[i].pos.z;
        }
        // implicit barrier

        for (int step = 0; step < numSteps; ++step) {
            const PosSoA& cur = buf[step & 1];
            PosSoA& nxt = buf[(step + 1) & 1];
            const double* px = cur.x.data();
            const double* py = cur.y.data();
            const double* pz = cur.z.data();

#pragma omp for schedule(static)
            for (size_t b = 0; b < numBlocks; ++b) {
                const size_t i0 = b * LANES;
                double Fx[LANES], Fy[LANES], Fz[LANES];
                forceBlock(px, py, pz, n, i0, Fx, Fy, Fz);
                const size_t iend = (i0 + LANES < n) ? i0 + LANES : n;
                for (size_t i = i0; i < iend; ++i) {
                    Body& body = bodies[i];
                    const size_t k = i - i0;
                    body.vel.x += DT * Fx[k];
                    body.vel.y += DT * Fy[k];
                    body.vel.z += DT * Fz[k];
                    body.pos.x = std::fma(body.vel.x, DT, body.pos.x);
                    body.pos.y = std::fma(body.vel.y, DT, body.pos.y);
                    body.pos.z = std::fma(body.vel.z, DT, body.pos.z);
                    nxt.x[i] = body.pos.x;
                    nxt.y[i] = body.pos.y;
                    nxt.z[i] = body.pos.z;
                }
            }
            // implicit barrier: next step reads nxt
        }
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
#pragma omp parallel for reduction(+:energy) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const Body& body = bodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
#pragma omp parallel for reduction(-:energy) schedule(dynamic, 16)
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
