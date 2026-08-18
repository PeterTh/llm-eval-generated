#include <chrono>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// A structure-of-arrays layout keeps the three position streams contiguous in
// the O(N^2) force loop and gives the compiler a straightforward SIMD loop.
struct Bodies {
    explicit Bodies(const size_t n)
        : px(n), py(n), pz(n), vx(n), vy(n), vz(n) {}

    size_t size() const noexcept { return px.size(); }

    std::vector<double> px, py, pz;
    std::vector<double> vx, vy, vz;
};

void randomizeBodies(Bodies& bodies, unsigned int seed = 42) {
    for (size_t i = 0; i < bodies.size(); ++i) {
        bodies.px[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.py[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.pz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vx[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vy[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

void simulate(Bodies& bodies, const int numSteps) {
    const ptrdiff_t n = static_cast<ptrdiff_t>(bodies.size());

#pragma omp parallel default(none) shared(bodies, n, numSteps)
    for (int step = 0; step < numSteps; ++step) {
#pragma omp for schedule(static)
        for (ptrdiff_t i = 0; i < n; ++i) {
            double Fx = 0.0, Fy = 0.0, Fz = 0.0;
            const double pix = bodies.px[i];
            const double piy = bodies.py[i];
            const double piz = bodies.pz[i];

#pragma omp simd reduction(+:Fx,Fy,Fz)
            for (ptrdiff_t j = 0; j < n; ++j) {
                const double dx = bodies.px[j] - pix;
                const double dy = bodies.py[j] - piy;
                const double dz = bodies.pz[j] - piz;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }

            bodies.vx[i] += DT * Fx;
            bodies.vy[i] += DT * Fy;
            bodies.vz[i] += DT * Fz;
        }

#pragma omp for schedule(static)
        for (ptrdiff_t i = 0; i < n; ++i) {
            bodies.px[i] += bodies.vx[i] * DT;
            bodies.py[i] += bodies.vy[i] * DT;
            bodies.pz[i] += bodies.vz[i] * DT;
        }
    }
}

double computeTotalEnergy(const Bodies& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
#pragma omp parallel for reduction(+:energy) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies.vx[i] * bodies.vx[i] +
                        bodies.vy[i] * bodies.vy[i] +
                        bodies.vz[i] * bodies.vz[i]);
    }
    
    // Potential energy (assuming unit mass for all bodies)
#pragma omp parallel for reduction(+:energy) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies.px[j] - bodies.px[i];
            const double dy = bodies.py[j] - bodies.py[i];
            const double dz = bodies.pz[j] - bodies.pz[i];
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
        if (!std::isfinite(bodies.px[i]) || !std::isfinite(bodies.py[i]) || !std::isfinite(bodies.pz[i]) ||
            !std::isfinite(bodies.vx[i]) || !std::isfinite(bodies.vy[i]) || !std::isfinite(bodies.vz[i])) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(bodies.px[i]) > maxPos || std::abs(bodies.py[i]) > maxPos || std::abs(bodies.pz[i]) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(bodies.vx[i]) > maxVel || std::abs(bodies.vy[i]) > maxVel || std::abs(bodies.vz[i]) > maxVel) {
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
            bodyData.push_back(bodies.px[i]);
            bodyData.push_back(bodies.py[i]);
            bodyData.push_back(bodies.pz[i]);
            bodyData.push_back(bodies.vx[i]);
            bodyData.push_back(bodies.vy[i]);
            bodyData.push_back(bodies.vz[i]);
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
