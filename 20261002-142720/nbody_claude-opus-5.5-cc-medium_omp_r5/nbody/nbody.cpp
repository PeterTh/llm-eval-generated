#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// Structure-of-arrays copy of the body positions, used by the force kernel for
// unit-stride (vectorizable) access. Kept in sync by integrateBodies().
// Floating-point contraction is disabled at build time (-ffp-contract=off); the
// explicit std::fma calls below reproduce the FMA pattern GCC emits for the
// original serial code at -O3 -march=native, so results are bitwise identical.
struct PosSoA {
    std::vector<double> x, y, z;
};

// Number of bodies i processed together; each lane accumulates its own force
// over j in the original sequential order, so results are bitwise identical.
constexpr size_t IBLOCK = 8;

static inline void forceBlock(const PosSoA& p, std::vector<Body>& bodies, size_t i0, size_t n) {
    const double* __restrict px = p.x.data();
    const double* __restrict py = p.y.data();
    const double* __restrict pz = p.z.data();

    if (i0 + IBLOCK <= n) {
        double xi[IBLOCK], yi[IBLOCK], zi[IBLOCK];
        double Fx[IBLOCK], Fy[IBLOCK], Fz[IBLOCK];
        for (size_t k = 0; k < IBLOCK; ++k) {
            xi[k] = px[i0 + k];
            yi[k] = py[i0 + k];
            zi[k] = pz[i0 + k];
            Fx[k] = 0.0;
            Fy[k] = 0.0;
            Fz[k] = 0.0;
        }
        for (size_t j = 0; j < n; ++j) {
            const double xj = px[j], yj = py[j], zj = pz[j];
#pragma omp simd
            for (size_t k = 0; k < IBLOCK; ++k) {
                const double dx = xj - xi[k];
                const double dy = yj - yi[k];
                const double dz = zj - zi[k];
                const double distSqr = std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx[k] = std::fma(dx, invDist3, Fx[k]);
                Fy[k] = std::fma(dy, invDist3, Fy[k]);
                Fz[k] = std::fma(dz, invDist3, Fz[k]);
            }
        }
        for (size_t k = 0; k < IBLOCK; ++k) {
            bodies[i0 + k].vel.x += DT * Fx[k];
            bodies[i0 + k].vel.y += DT * Fy[k];
            bodies[i0 + k].vel.z += DT * Fz[k];
        }
    } else {
        for (size_t i = i0; i < n; ++i) {
            double Fx = 0.0, Fy = 0.0, Fz = 0.0;
            const double xi = px[i], yi = py[i], zi = pz[i];
            for (size_t j = 0; j < n; ++j) {
                const double dx = px[j] - xi;
                const double dy = py[j] - yi;
                const double dz = pz[j] - zi;
                const double distSqr = std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx = std::fma(dx, invDist3, Fx);
                Fy = std::fma(dy, invDist3, Fy);
                Fz = std::fma(dz, invDist3, Fz);
            }
            bodies[i].vel.x += DT * Fx;
            bodies[i].vel.y += DT * Fy;
            bodies[i].vel.z += DT * Fz;
        }
    }
}

// Must be called from within an OpenMP parallel region (orphaned worksharing).
void computeForces(std::vector<Body>& bodies, const PosSoA& p) {
    const size_t n = bodies.size();
    const size_t nblocks = (n + IBLOCK - 1) / IBLOCK;

#pragma omp for schedule(static)
    for (size_t b = 0; b < nblocks; ++b) {
        forceBlock(p, bodies, b * IBLOCK, n);
    }
}

// Must be called from within an OpenMP parallel region (orphaned worksharing).
void integrateBodies(std::vector<Body>& bodies, PosSoA& p) {
    const size_t n = bodies.size();
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        Body& body = bodies[i];
        body.pos.x = std::fma(body.vel.x, DT, body.pos.x);
        body.pos.y = std::fma(body.vel.y, DT, body.pos.y);
        body.pos.z = std::fma(body.vel.z, DT, body.pos.z);
        p.x[i] = body.pos.x;
        p.y[i] = body.pos.y;
        p.z[i] = body.pos.z;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
#pragma omp parallel for reduction(+ : energy) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const Body& body = bodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
#pragma omp parallel for reduction(+ : energy) schedule(dynamic, 16)
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
    
    PosSoA soa;
    soa.x.resize(numBodies);
    soa.y.resize(numBodies);
    soa.z.resize(numBodies);

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            soa.x[i] = bodies[i].pos.x;
            soa.y[i] = bodies[i].pos.y;
            soa.z[i] = bodies[i].pos.z;
        }
        for (int step = 0; step < numSteps; ++step) {
            computeForces(bodies, soa);
            integrateBodies(bodies, soa);
        }
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
