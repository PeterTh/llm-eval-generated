#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// Structure-of-arrays body storage: the force kernel only touches the position
// arrays, so keeping them contiguous maximizes useful bytes per cache line and
// gives the hardware prefetchers a clean, unit-stride access pattern.
struct Bodies {
    std::vector<double> posX, posY, posZ;
    std::vector<double> velX, velY, velZ;

    explicit Bodies(const size_t n)
        : posX(n), posY(n), posZ(n), velX(n), velY(n), velZ(n) {}

    size_t size() const noexcept { return posX.size(); }
};

void randomizeBodies(Bodies& bodies, unsigned int seed = 42) {
    const size_t n = bodies.size();
    for (size_t i = 0; i < n; ++i) {
        bodies.posX[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.posY[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.posZ[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.velX[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.velY[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.velZ[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// Must be called from inside a parallel region: the work-sharing constructs are
// orphaned so that a single region can span the whole simulation loop.
void computeForces(Bodies& bodies) {
    const size_t n = bodies.size();
    const double* __restrict const px = bodies.posX.data();
    const double* __restrict const py = bodies.posY.data();
    const double* __restrict const pz = bodies.posZ.data();
    double* __restrict const vx = bodies.velX.data();
    double* __restrict const vy = bodies.velY.data();
    double* __restrict const vz = bodies.velZ.data();

    // Each i is independent and every i costs exactly the same, so a static
    // schedule gives perfect balance with no scheduling overhead. The inner
    // loop is SIMD-vectorized; the reduction is the only place where the
    // summation order differs from the serial code, and it is fixed at compile
    // time, so results are independent of the number of threads.
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const double xi = px[i], yi = py[i], zi = pz[i];
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

#pragma omp simd reduction(+ : Fx, Fy, Fz)
        for (size_t j = 0; j < n; ++j) {
            const double dx = px[j] - xi;
            const double dy = py[j] - yi;
            const double dz = pz[j] - zi;
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

// Also called from inside the parallel region. The implicit barrier of the
// preceding force loop guarantees all positions have been read before any of
// them is updated here.
void integrateBodies(Bodies& bodies) {
    const size_t n = bodies.size();
    double* __restrict const px = bodies.posX.data();
    double* __restrict const py = bodies.posY.data();
    double* __restrict const pz = bodies.posZ.data();
    const double* __restrict const vx = bodies.velX.data();
    const double* __restrict const vy = bodies.velY.data();
    const double* __restrict const vz = bodies.velZ.data();

#pragma omp for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        px[i] += vx[i] * DT;
        py[i] += vy[i] * DT;
        pz[i] += vz[i] * DT;
    }
}

double computeTotalEnergy(const Bodies& bodies) {
    const size_t n = bodies.size();
    const double* __restrict const px = bodies.posX.data();
    const double* __restrict const py = bodies.posY.data();
    const double* __restrict const pz = bodies.posZ.data();

    double energy = 0.0;

    // Kinetic energy (assuming unit mass)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies.velX[i] * bodies.velX[i] +
                         bodies.velY[i] * bodies.velY[i] +
                         bodies.velZ[i] * bodies.velZ[i]);
    }

    // Potential energy (assuming unit mass for all bodies). Per-row partial
    // sums are accumulated separately and combined in index order, so the
    // result is independent of the thread count and of the schedule used.
    std::vector<double> rowEnergy(n, 0.0);

#pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        double e = 0.0;
#pragma omp simd reduction(+ : e)
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = px[j] - px[i];
            const double dy = py[j] - py[i];
            const double dz = pz[j] - pz[i];
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
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
bool validateSimulation(const Bodies& bodies) {
    const size_t n = bodies.size();
    for (size_t i = 0; i < n; ++i) {
        const double posx = bodies.posX[i], posy = bodies.posY[i], posz = bodies.posZ[i];
        const double velx = bodies.velX[i], vely = bodies.velY[i], velz = bodies.velZ[i];

        // Check for NaN or Inf values
        if (!std::isfinite(posx) || !std::isfinite(posy) || !std::isfinite(posz) ||
            !std::isfinite(velx) || !std::isfinite(vely) || !std::isfinite(velz)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(posx) > maxPos || std::abs(posy) > maxPos || std::abs(posz) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(velx) > maxVel || std::abs(vely) > maxVel || std::abs(velz) > maxVel) {
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

    // A single parallel region spans all steps: threads are forked once and the
    // implicit barriers of the work-sharing loops provide the required
    // force/integrate synchronization.
#pragma omp parallel
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
        for (int i = 0; i < numBodies; ++i) {
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
