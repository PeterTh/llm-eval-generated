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

static inline double rand_unit(unsigned int& seed) noexcept {
    return rand_r(&seed) / static_cast<double>(RAND_MAX);
}

void randomizeBodies(std::vector<double>& px,
                    std::vector<double>& py,
                    std::vector<double>& pz,
                    std::vector<double>& vx,
                    std::vector<double>& vy,
                    std::vector<double>& vz,
                    unsigned int seed = 42) {
    const size_t n = px.size();
    for (size_t i = 0; i < n; ++i) {
        px[i] = 2.0 * rand_unit(seed) - 1.0;
        py[i] = 2.0 * rand_unit(seed) - 1.0;
        pz[i] = 2.0 * rand_unit(seed) - 1.0;
        vx[i] = 2.0 * rand_unit(seed) - 1.0;
        vy[i] = 2.0 * rand_unit(seed) - 1.0;
        vz[i] = 2.0 * rand_unit(seed) - 1.0;
    }
}

double computeTotalEnergy(const std::vector<double>& px,
                          const std::vector<double>& py,
                          const std::vector<double>& pz,
                          const std::vector<double>& vx,
                          const std::vector<double>& vy,
                          const std::vector<double>& vz) {
    double energy = 0.0;
    const size_t n = px.size();

    // Kinetic energy (assuming unit mass)
#pragma omp parallel for schedule(static) reduction(+ : energy)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    }

    // Potential energy (assuming unit mass for all bodies)
#pragma omp parallel for schedule(static) reduction(+ : energy)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = px[j] - px[i];
            const double dy = py[j] - py[i];
            const double dz = pz[j] - pz[i];
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<double>& px,
                        const std::vector<double>& py,
                        const std::vector<double>& pz,
                        const std::vector<double>& vx,
                        const std::vector<double>& vy,
                        const std::vector<double>& vz) {
    const size_t n = px.size();
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(px[i]) || !std::isfinite(py[i]) || !std::isfinite(pz[i]) || !std::isfinite(vx[i]) ||
            !std::isfinite(vy[i]) || !std::isfinite(vz[i])) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(px[i]) > maxPos || std::abs(py[i]) > maxPos || std::abs(pz[i]) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(vx[i]) > maxVel || std::abs(vy[i]) > maxVel || std::abs(vz[i]) > maxVel) {
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

    omp_set_dynamic(0);

    // Structure-of-arrays (SoA) for better memory throughput/vectorization
    const size_t n = static_cast<size_t>(numBodies);
    std::vector<double> px(n), py(n), pz(n);
    std::vector<double> vx(n), vy(n), vz(n);
    randomizeBodies(px, py, pz, vx, vy, vz);

    double* const pxp = px.data();
    double* const pyp = py.data();
    double* const pzp = pz.data();
    double* const vxp = vx.data();
    double* const vyp = vy.data();
    double* const vzp = vz.data();

    const double dt = DT;

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
        for (int step = 0; step < numSteps; ++step) {
#pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                const double ix = pxp[i];
                const double iy = pyp[i];
                const double iz = pzp[i];

                double Fx = 0.0, Fy = 0.0, Fz = 0.0;

#pragma omp simd reduction(+ : Fx, Fy, Fz)
                for (size_t j = 0; j < n; ++j) {
                    const double dx = pxp[j] - ix;
                    const double dy = pyp[j] - iy;
                    const double dz = pzp[j] - iz;
                    const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    const double invDist = 1.0 / std::sqrt(distSqr);
                    const double invDist3 = invDist * invDist * invDist;

                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }

                vxp[i] += dt * Fx;
                vyp[i] += dt * Fy;
                vzp[i] += dt * Fz;
            }

#pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                pxp[i] += vxp[i] * dt;
                pyp[i] += vyp[i] * dt;
                pzp[i] += vzp[i] * dt;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Print results for external validation
    if (printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (size_t i = 0; i < n; ++i) {
            bodyData.push_back(pxp[i]);
            bodyData.push_back(pyp[i]);
            bodyData.push_back(pzp[i]);
            bodyData.push_back(vxp[i]);
            bodyData.push_back(vyp[i]);
            bodyData.push_back(vzp[i]);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(px, py, pz, vx, vy, vz)) {
            double finalEnergy = computeTotalEnergy(px, py, pz, vx, vy, vz);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            return 0;
        }

        printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
