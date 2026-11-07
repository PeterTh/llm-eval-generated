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

// Number of i-bodies processed together against each j-body. The j-loop stays
// the innermost sequential reduction per body (identical summation order to the
// serial code), while the compiler vectorizes across the IBLOCK independent
// accumulators.
constexpr size_t IBLOCK = 16;

void computeForces(std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    if (n == 0) return;
    const size_t nPadded = (n + IBLOCK - 1) / IBLOCK * IBLOCK;

    // Structure-of-arrays snapshot of positions (contiguous, cache friendly).
    // Padding entries replicate body 0; their results are discarded.
    static std::vector<double> px, py, pz;
    px.resize(nPadded); py.resize(nPadded); pz.resize(nPadded);

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (size_t i = 0; i < nPadded; ++i) {
            const Body& b = bodies[i < n ? i : 0];
            px[i] = b.pos.x; py[i] = b.pos.y; pz[i] = b.pos.z;
        }

        const double* __restrict X = px.data();
        const double* __restrict Y = py.data();
        const double* __restrict Z = pz.data();

#pragma omp for schedule(static)
        for (size_t i0 = 0; i0 < nPadded; i0 += IBLOCK) {
            alignas(64) double xi[IBLOCK], yi[IBLOCK], zi[IBLOCK];
            alignas(64) double Fx[IBLOCK], Fy[IBLOCK], Fz[IBLOCK];
            for (size_t k = 0; k < IBLOCK; ++k) {
                xi[k] = X[i0 + k]; yi[k] = Y[i0 + k]; zi[k] = Z[i0 + k];
                Fx[k] = 0.0; Fy[k] = 0.0; Fz[k] = 0.0;
            }

            for (size_t j = 0; j < n; ++j) {
                const double xj = X[j], yj = Y[j], zj = Z[j];
#pragma omp simd aligned(xi, yi, zi, Fx, Fy, Fz : 64)
                for (size_t k = 0; k < IBLOCK; ++k) {
                    const double dx = xj - xi[k];
                    const double dy = yj - yi[k];
                    const double dz = zj - zi[k];
                    // Explicit FMA pattern (matches the contraction GCC applies
                    // to the reference scalar kernel); -ffp-contract=off keeps
                    // the compiler from introducing any other fusions.
                    const double distSqr = std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING;
                    const double invDist = 1.0 / std::sqrt(distSqr);
                    const double invDist3 = invDist * invDist * invDist;

                    Fx[k] = std::fma(dx, invDist3, Fx[k]);
                    Fy[k] = std::fma(dy, invDist3, Fy[k]);
                    Fz[k] = std::fma(dz, invDist3, Fz[k]);
                }
            }

            const size_t iEnd = (i0 + IBLOCK < n) ? i0 + IBLOCK : n;
            for (size_t i = i0; i < iEnd; ++i) {
                const size_t k = i - i0;
                bodies[i].vel.x += DT * Fx[k];
                bodies[i].vel.y += DT * Fy[k];
                bodies[i].vel.z += DT * Fz[k];
            }
        }
    }
}

void integrateBodies(std::vector<Body>& bodies) {
    const size_t n = bodies.size();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        Body& body = bodies[i];
        body.pos.x = std::fma(body.vel.x, DT, body.pos.x);
        body.pos.y = std::fma(body.vel.y, DT, body.pos.y);
        body.pos.z = std::fma(body.vel.z, DT, body.pos.z);
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
    
    // Potential energy (assuming unit mass for all bodies).
    // Per-row partial sums are computed in parallel (triangular workload, hence
    // dynamic scheduling) and reduced serially so the result is deterministic
    // regardless of thread count.
    std::vector<double> rowPotential(n, 0.0);
#pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < n; ++i) {
        const double xi = bodies[i].pos.x, yi = bodies[i].pos.y, zi = bodies[i].pos.z;
        double rowSum = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - xi;
            const double dy = bodies[j].pos.y - yi;
            const double dz = bodies[j].pos.z - zi;
            const double dist = std::sqrt(std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING);
            rowSum -= 1.0 / dist;
        }
        rowPotential[i] = rowSum;
    }
    for (size_t i = 0; i < n; ++i) {
        energy += rowPotential[i];
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
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies);
        integrateBodies(bodies);
    }
    
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
