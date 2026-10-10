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

// These worksharing routines are called by every thread in the simulation team.
void computeForces(std::vector<Body>& bodies,
                   const double* positionX, const double* positionY, const double* positionZ) {
    const size_t n = bodies.size();
    constexpr size_t blockSize = 4;

    #pragma omp for schedule(static)
    for (size_t first = 0; first < n; first += blockSize) {
        const size_t count = std::min(blockSize, n - first);
        alignas(64) double x[blockSize] = {}, y[blockSize] = {}, z[blockSize] = {};
        alignas(64) double Fx[blockSize] = {}, Fy[blockSize] = {}, Fz[blockSize] = {};

        for (size_t lane = 0; lane < count; ++lane) {
            x[lane] = positionX[first + lane];
            y[lane] = positionY[first + lane];
            z[lane] = positionZ[first + lane];
        }

        // Vectorize across target bodies, keeping each body's j summation order.
        // Zero-filled tail lanes let the hot loop always use a full SIMD block.
        for (size_t j = 0; j < n; ++j) {
            const double sourceX = positionX[j];
            const double sourceY = positionY[j];
            const double sourceZ = positionZ[j];

            #pragma omp simd aligned(x, y, z, Fx, Fy, Fz : 64)
            for (size_t lane = 0; lane < blockSize; ++lane) {
                const double dx = sourceX - x[lane];
                const double dy = sourceY - y[lane];
                const double dz = sourceZ - z[lane];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx[lane] += dx * invDist3;
                Fy[lane] += dy * invDist3;
                Fz[lane] += dz * invDist3;
            }
        }

        // Scale first so the velocity update retains separate multiply/add rounding.
        #pragma omp simd aligned(Fx, Fy, Fz : 64)
        for (size_t lane = 0; lane < blockSize; ++lane) {
            Fx[lane] *= DT;
            Fy[lane] *= DT;
            Fz[lane] *= DT;
        }

        for (size_t lane = 0; lane < count; ++lane) {
            bodies[first + lane].vel.x += Fx[lane];
            bodies[first + lane].vel.y += Fy[lane];
            bodies[first + lane].vel.z += Fz[lane];
        }
    }
}

void integrateBodies(std::vector<Body>& bodies,
                     double* positionX, double* positionY, double* positionZ) {
    const size_t n = bodies.size();

    #pragma omp for simd schedule(static)
    for (size_t i = 0; i < n; ++i) {
        auto& body = bodies[i];
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
        positionX[i] = body.pos.x;
        positionY[i] = body.pos.y;
        positionZ[i] = body.pos.z;
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

    // Compact, read-only positions avoid streaming velocities and false sharing
    // between force reads and velocity writes. Refresh them during integration.
    std::vector<double> x(numBodies), y(numBodies), z(numBodies);
    double* positionX = x.data();
    double* positionY = y.data();
    double* positionZ = z.data();
    const size_t n = bodies.size();
    // Keep enough work per worker to amortize synchronization for small inputs.
    constexpr size_t minBodiesPerThread = 64;
    const int usefulThreads = static_cast<int>(std::max<size_t>(
        1, (n + minBodiesPerThread - 1) / minBodiesPerThread));
    const int numThreads = std::min(omp_get_max_threads(), usefulThreads);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel num_threads(numThreads) default(none) \
        shared(bodies, positionX, positionY, positionZ, n, numSteps)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            positionX[i] = bodies[i].pos.x;
            positionY[i] = bodies[i].pos.y;
            positionZ[i] = bodies[i].pos.z;
        }

        for (int step = 0; step < numSteps; ++step) {
            computeForces(bodies, positionX, positionY, positionZ);
            // The force-loop barrier completes all reads before positions change.
            integrateBodies(bodies, positionX, positionY, positionZ);
            // The integration barrier publishes positions for the next timestep.
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
