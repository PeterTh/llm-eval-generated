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

// Number of bodies "i" processed together in SIMD lanes. Each lane accumulates
// its forces over j in exactly the original sequential order, so results are
// bitwise identical to the serial code while still vectorizing.
constexpr size_t TILE = 8;

// Structure-of-arrays state used during the simulation. Arrays are padded to a
// multiple of TILE so the inner tile loop needs no remainder handling.
struct BodiesSoA {
    size_t n = 0, padded = 0;
    std::vector<double> px, py, pz, vx, vy, vz;
};

// Velocity update for bodies [i0, i0 + TILE) using current positions.
static inline void computeForcesTile(BodiesSoA& s, const size_t i0) {
    const size_t n = s.n;
    const double* __restrict px = s.px.data();
    const double* __restrict py = s.py.data();
    const double* __restrict pz = s.pz.data();

    alignas(64) double xi[TILE], yi[TILE], zi[TILE];
    alignas(64) double Fx[TILE], Fy[TILE], Fz[TILE];
    #pragma omp simd
    for (size_t k = 0; k < TILE; ++k) {
        xi[k] = px[i0 + k];
        yi[k] = py[i0 + k];
        zi[k] = pz[i0 + k];
        Fx[k] = 0.0;
        Fy[k] = 0.0;
        Fz[k] = 0.0;
    }

    for (size_t j = 0; j < n; ++j) {
        const double xj = px[j], yj = py[j], zj = pz[j];
        #pragma omp simd aligned(xi, yi, zi, Fx, Fy, Fz : 64)
        for (size_t k = 0; k < TILE; ++k) {
            const double dx = xj - xi[k];
            const double dy = yj - yi[k];
            const double dz = zj - zi[k];
            // Explicit FMAs reproduce the rounding of the reference build
            // (which relies on compiler contraction) bit for bit.
            const double distSqr = std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx[k] = std::fma(dx, invDist3, Fx[k]);
            Fy[k] = std::fma(dy, invDist3, Fy[k]);
            Fz[k] = std::fma(dz, invDist3, Fz[k]);
        }
    }

    const size_t cnt = (i0 + TILE <= n) ? TILE : n - i0;
    for (size_t k = 0; k < cnt; ++k) {
        s.vx[i0 + k] += DT * Fx[k];
        s.vy[i0 + k] += DT * Fy[k];
        s.vz[i0 + k] += DT * Fz[k];
    }
}

// Runs the full simulation inside a single parallel region to avoid repeated
// fork/join overhead. Per step: all velocities are updated from the current
// positions (barrier), then all positions are integrated (barrier).
void simulate(std::vector<Body>& bodies, const int numSteps) {
    BodiesSoA s;
    s.n = bodies.size();
    s.padded = (s.n + TILE - 1) / TILE * TILE;
    s.px.resize(s.padded); s.py.resize(s.padded); s.pz.resize(s.padded);
    s.vx.resize(s.padded); s.vy.resize(s.padded); s.vz.resize(s.padded);
    const size_t n = s.n;
    const long numTiles = static_cast<long>(s.padded / TILE);

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < s.padded; ++i) {
            if (i < n) {
                s.px[i] = bodies[i].pos.x; s.py[i] = bodies[i].pos.y; s.pz[i] = bodies[i].pos.z;
                s.vx[i] = bodies[i].vel.x; s.vy[i] = bodies[i].vel.y; s.vz[i] = bodies[i].vel.z;
            } else {
                s.px[i] = s.py[i] = s.pz[i] = 0.0;
                s.vx[i] = s.vy[i] = s.vz[i] = 0.0;
            }
        }

        for (int step = 0; step < numSteps; ++step) {
            #pragma omp for schedule(static)
            for (long t = 0; t < numTiles; ++t) {
                computeForcesTile(s, static_cast<size_t>(t) * TILE);
            }

            #pragma omp for simd schedule(static)
            for (size_t i = 0; i < n; ++i) {
                s.px[i] = std::fma(s.vx[i], DT, s.px[i]);
                s.py[i] = std::fma(s.vy[i], DT, s.py[i]);
                s.pz[i] = std::fma(s.vz[i], DT, s.pz[i]);
            }
        }

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            bodies[i].pos = Vec3(s.px[i], s.py[i], s.pz[i]);
            bodies[i].vel = Vec3(s.vx[i], s.vy[i], s.vz[i]);
        }
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass); O(n), kept serial for exact ordering
    for (const auto& body : bodies) {
        energy = std::fma(0.5, std::fma(body.vel.z, body.vel.z,
                                        std::fma(body.vel.y, body.vel.y, body.vel.x * body.vel.x)),
                          energy);
    }
    
    // Potential energy (assuming unit mass for all bodies).
    // Each row's pair terms are computed in parallel; rows are then combined in
    // order so the result is independent of the number of threads.
    std::vector<double> rowEnergy(n, 0.0);
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < n; ++i) {
        double e = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(std::fma(dz, dz, std::fma(dy, dy, dx * dx)) + SOFTENING);
            e -= 1.0 / dist;
        }
        rowEnergy[i] = e;
    }
    for (size_t i = 0; i < n; ++i) {
        energy += rowEnergy[i];
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
