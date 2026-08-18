#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// Keep each component contiguous.  The force calculation streams through all
// positions for every body, so an SoA layout gives the inner loop substantially
// better cache and SIMD behavior than an array of interleaved Body objects.
struct Bodies {
    explicit Bodies(const size_t n)
        : posX(n), posY(n), posZ(n), velX(n), velY(n), velZ(n) {}

    [[nodiscard]] size_t size() const noexcept { return posX.size(); }

    std::vector<double> posX, posY, posZ;
    std::vector<double> velX, velY, velZ;
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

// Called by every thread in the persistent parallel region in main.
void computeForces(Bodies& bodies) {
    const size_t n = bodies.size();
    const double* const __restrict px = bodies.posX.data();
    const double* const __restrict py = bodies.posY.data();
    const double* const __restrict pz = bodies.posZ.data();
    double* const __restrict vx = bodies.velX.data();
    double* const __restrict vy = bodies.velY.data();
    double* const __restrict vz = bodies.velZ.data();

#pragma omp for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        const double pix = px[i];
        const double piy = py[i];
        const double piz = pz[i];

#pragma omp simd reduction(+ : Fx, Fy, Fz)
        for (size_t j = 0; j < n; ++j) {
            const double dx = px[j] - pix;
            const double dy = py[j] - piy;
            const double dz = pz[j] - piz;
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
    const size_t n = bodies.size();

#pragma omp for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        bodies.posX[i] += bodies.velX[i] * DT;
        bodies.posY[i] += bodies.velY[i] * DT;
        bodies.posZ[i] += bodies.velZ[i] * DT;
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
    
#pragma omp parallel default(none) shared(bodies, numSteps)
    {
        for (int step = 0; step < numSteps; ++step) {
            computeForces(bodies);
            integrateBodies(bodies);
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
