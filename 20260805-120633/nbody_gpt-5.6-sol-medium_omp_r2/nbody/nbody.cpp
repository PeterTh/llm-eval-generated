#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// A structure-of-arrays layout keeps the six independent streams contiguous.
// The O(N^2) force loop consequently reads only position data from memory.
struct Bodies {
    explicit Bodies(const std::size_t count)
        : x(count), y(count), z(count), vx(count), vy(count), vz(count) {}

    std::size_t size() const noexcept { return x.size(); }

    std::vector<double> x, y, z;
    std::vector<double> vx, vy, vz;
};

void randomizeBodies(Bodies& bodies, unsigned int seed = 42) {
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        // Keep the calls in the original order so initialization is unchanged.
        bodies.x[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        bodies.y[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        bodies.z[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        bodies.vx[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        bodies.vy[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        bodies.vz[i] = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

void runSimulation(Bodies& bodies, const int numSteps) {
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(bodies.size());
    double* const x = bodies.x.data();
    double* const y = bodies.y.data();
    double* const z = bodies.z.data();
    double* const vx = bodies.vx.data();
    double* const vy = bodies.vy.data();
    double* const vz = bodies.vz.data();

    // One long-lived team avoids two parallel-region startups per timestep.
    // The implicit barrier after each loop is required: force calculation must
    // finish before integration, and integration before the next force pass.
#pragma omp parallel default(none) shared(x, y, z, vx, vy, vz, n, numSteps)
    {
        for (int step = 0; step < numSteps; ++step) {
#pragma omp for schedule(static)
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                const double xi = x[i];
                const double yi = y[i];
                const double zi = z[i];
                double fx = 0.0;
                double fy = 0.0;
                double fz = 0.0;

                // Each i is independent. SIMD reductions expose the innermost
                // arithmetic while the outer loop provides thread scalability.
#pragma omp simd reduction(+ : fx, fy, fz)
                for (std::ptrdiff_t j = 0; j < n; ++j) {
                    const double dx = x[j] - xi;
                    const double dy = y[j] - yi;
                    const double dz = z[j] - zi;
                    const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    const double invDist = 1.0 / std::sqrt(distSqr);
                    const double invDist3 = invDist * invDist * invDist;

                    fx += dx * invDist3;
                    fy += dy * invDist3;
                    fz += dz * invDist3;
                }

                vx[i] += DT * fx;
                vy[i] += DT * fy;
                vz[i] += DT * fz;
            }

#pragma omp for schedule(static)
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                x[i] += vx[i] * DT;
                y[i] += vy[i] * DT;
                z[i] += vz[i] * DT;
            }
        }
    }
}

double computeTotalEnergy(const Bodies& bodies) {
    double energy = 0.0;
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(bodies.size());

#pragma omp parallel for reduction(+ : energy) schedule(static)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies.vx[i] * bodies.vx[i] +
                        bodies.vy[i] * bodies.vy[i] +
                        bodies.vz[i] * bodies.vz[i]);

        double potential = 0.0;
#pragma omp simd reduction(+ : potential)
        for (std::ptrdiff_t j = i + 1; j < n; ++j) {
            const double dx = bodies.x[j] - bodies.x[i];
            const double dy = bodies.y[j] - bodies.y[i];
            const double dz = bodies.z[j] - bodies.z[i];
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potential -= 1.0 / dist;
        }
        energy += potential;
    }

    return energy;
}

// Validate that simulation produces finite, reasonable values.
bool validateSimulation(const Bodies& bodies) {
    int finite = 1;
    int boundedPosition = 1;
    int boundedVelocity = 1;
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(bodies.size());

#pragma omp parallel for reduction(& : finite, boundedPosition, boundedVelocity) schedule(static)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        finite &= std::isfinite(bodies.x[i]) && std::isfinite(bodies.y[i]) &&
                  std::isfinite(bodies.z[i]) && std::isfinite(bodies.vx[i]) &&
                  std::isfinite(bodies.vy[i]) && std::isfinite(bodies.vz[i]);
        boundedPosition &= std::abs(bodies.x[i]) <= 1e6 &&
                           std::abs(bodies.y[i]) <= 1e6 &&
                           std::abs(bodies.z[i]) <= 1e6;
        boundedVelocity &= std::abs(bodies.vx[i]) <= 1e6 &&
                           std::abs(bodies.vy[i]) <= 1e6 &&
                           std::abs(bodies.vz[i]) <= 1e6;
    }

    if (!finite) {
        std::printf("Validation failed: found NaN or Inf value in body state\n");
        return false;
    }
    if (!boundedPosition) {
        std::printf("Validation failed: body position exceeds reasonable bounds\n");
        return false;
    }
    if (!boundedVelocity) {
        std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks energy conservation)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (numBodies < 0 || numSteps < 0) {
        std::printf("Number of bodies and steps must be non-negative\n");
        return 1;
    }

    std::printf("N-Body Simulation\n");
    std::printf("Number of bodies: %d\n", numBodies);
    std::printf("Number of steps: %d\n", numSteps);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    Bodies bodies(static_cast<std::size_t>(numBodies));
    randomizeBodies(bodies);

    const auto start = std::chrono::high_resolution_clock::now();
    runSimulation(bodies, numSteps);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::printf("Simulation time: %ld ms\n", static_cast<long>(duration.count()));

    if (printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<std::size_t>(numBodies) * 6);
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            bodyData.push_back(bodies.x[i]);
            bodyData.push_back(bodies.y[i]);
            bodyData.push_back(bodies.z[i]);
            bodyData.push_back(bodies.vx[i]);
            bodyData.push_back(bodies.vy[i]);
            bodyData.push_back(bodies.vz[i]);
        }
        print_results(bodyData, "Bodies");
    }

    if (validate) {
        std::printf("Validating simulation results...\n");
        if (validateSimulation(bodies)) {
            const double finalEnergy = computeTotalEnergy(bodies);
            std::printf("Final energy: %.6f\n", finalEnergy);
            std::printf("Validation: PASSED\n");
            return 0;
        }
        std::printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
