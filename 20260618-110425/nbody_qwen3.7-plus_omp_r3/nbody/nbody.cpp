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

// Structure-of-Arrays layout for optimal cache utilization and SIMD vectorization
struct Bodies {
    size_t n;
    std::vector<double> px, py, pz;
    std::vector<double> vx, vy, vz;

    explicit Bodies(size_t n) : n(n), px(n), py(n), pz(n), vx(n), vy(n), vz(n) {}
};

void randomizeBodies(Bodies& bodies, unsigned int seed = 42) {
    for (size_t i = 0; i < bodies.n; ++i) {
        bodies.px[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.py[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.pz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vx[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vy[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

void computeForces(Bodies& bodies) {
    const size_t n = bodies.n;
    const double* __restrict__ px = bodies.px.data();
    const double* __restrict__ py = bodies.py.data();
    const double* __restrict__ pz = bodies.pz.data();
    double* __restrict__ vx = bodies.vx.data();
    double* __restrict__ vy = bodies.vy.data();
    double* __restrict__ vz = bodies.vz.data();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        const double pxi = px[i], pyi = py[i], pzi = pz[i];

        #pragma omp simd reduction(+:Fx,Fy,Fz)
        for (size_t j = 0; j < n; ++j) {
            const double dx = px[j] - pxi;
            const double dy = py[j] - pyi;
            const double dz = pz[j] - pzi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        vx[i] += DT * Fx;
        vy[i] += DT * Fy;
        vz[i] += DT * Fz;
    }
}

void integrateBodies(Bodies& bodies) {
    const size_t n = bodies.n;
    double* __restrict__ px = bodies.px.data();
    double* __restrict__ py = bodies.py.data();
    double* __restrict__ pz = bodies.pz.data();
    const double* __restrict__ vx = bodies.vx.data();
    const double* __restrict__ vy = bodies.vy.data();
    const double* __restrict__ vz = bodies.vz.data();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; ++i) {
        px[i] += vx[i] * DT;
        py[i] += vy[i] * DT;
        pz[i] += vz[i] * DT;
    }
}

double computeTotalEnergy(const Bodies& bodies) {
    double energy = 0.0;
    const size_t n = bodies.n;
    const double* __restrict__ px = bodies.px.data();
    const double* __restrict__ py = bodies.py.data();
    const double* __restrict__ pz = bodies.pz.data();
    const double* __restrict__ vx = bodies.vx.data();
    const double* __restrict__ vy = bodies.vy.data();
    const double* __restrict__ vz = bodies.vz.data();

    // Kinetic energy (assuming unit mass)
    #pragma omp parallel for simd reduction(+:energy) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    }

    // Potential energy (assuming unit mass for all bodies)
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const double pxi = px[i], pyi = py[i], pzi = pz[i];
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = px[j] - pxi;
            const double dy = py[j] - pyi;
            const double dz = pz[j] - pzi;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const Bodies& bodies) {
    for (size_t i = 0; i < bodies.n; ++i) {
        if (!std::isfinite(bodies.px[i]) || !std::isfinite(bodies.py[i]) || !std::isfinite(bodies.pz[i]) ||
            !std::isfinite(bodies.vx[i]) || !std::isfinite(bodies.vy[i]) || !std::isfinite(bodies.vz[i])) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

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

    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies);
        integrateBodies(bodies);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Print results for external validation
    if (printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
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
