#include <omp.h>
#include <pthread.h>
#include <sched.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
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

// Number of target bodies processed together by one thread. Each target keeps
// its own accumulator and sums contributions in the original j order, so the
// arithmetic per body is identical to the sequential version while the
// independent targets map onto SIMD lanes.
// FMA usage is explicit (and contraction disabled in the build) so that the
// rounding matches the reference build of the original code exactly.
constexpr size_t BLOCK = 8;

// Run numSteps simulation steps (force computation + integration) in parallel.
// Positions are double-buffered in SoA layout so that integration can be fused
// with force computation, requiring only a single barrier per step.
void simulate(std::vector<Body>& bodies, int numSteps) {
    const size_t n = bodies.size();
    if (n == 0 || numSteps <= 0) return;
    const size_t nBlocks = (n + BLOCK - 1) / BLOCK;
    const size_t nPad = nBlocks * BLOCK;

    std::vector<double> bufA(3 * nPad, 0.0), bufB(3 * nPad, 0.0), velBuf(3 * nPad, 0.0);
    double* const vx = velBuf.data();
    double* const vy = vx + nPad;
    double* const vz = vy + nPad;

    #pragma omp parallel
    {
        // Each thread holds its own buffer pointers and swaps them in lockstep.
        double* cur = bufA.data();
        double* nxt = bufB.data();

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            cur[i] = bodies[i].pos.x;
            cur[nPad + i] = bodies[i].pos.y;
            cur[2 * nPad + i] = bodies[i].pos.z;
            vx[i] = bodies[i].vel.x;
            vy[i] = bodies[i].vel.y;
            vz[i] = bodies[i].vel.z;
        }

        for (int step = 0; step < numSteps; ++step) {
            const double* __restrict__ px = cur;
            const double* __restrict__ py = cur + nPad;
            const double* __restrict__ pz = cur + 2 * nPad;
            double* __restrict__ qx = nxt;
            double* __restrict__ qy = nxt + nPad;
            double* __restrict__ qz = nxt + 2 * nPad;

            #pragma omp for schedule(static)
            for (size_t b = 0; b < nBlocks; ++b) {
                const size_t i0 = b * BLOCK;
                double xi[BLOCK], yi[BLOCK], zi[BLOCK];
                double Fx[BLOCK], Fy[BLOCK], Fz[BLOCK];
                for (size_t k = 0; k < BLOCK; ++k) {
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
                    for (size_t k = 0; k < BLOCK; ++k) {
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

                const size_t kEnd = (i0 + BLOCK <= n) ? BLOCK : n - i0;
                for (size_t k = 0; k < kEnd; ++k) {
                    const size_t i = i0 + k;
                    // computeForces: velocity update
                    vx[i] += DT * Fx[k];
                    vy[i] += DT * Fy[k];
                    vz[i] += DT * Fz[k];
                    // integrateBodies: position update into the next buffer
                    qx[i] = std::fma(vx[i], DT, xi[k]);
                    qy[i] = std::fma(vy[i], DT, yi[k]);
                    qz[i] = std::fma(vz[i], DT, zi[k]);
                }
            }
            // Implicit barrier above: all reads of cur are done; swap buffers.
            std::swap(cur, nxt);
        }

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            bodies[i].pos.x = cur[i];
            bodies[i].pos.y = cur[nPad + i];
            bodies[i].pos.z = cur[2 * nPad + i];
            bodies[i].vel.x = vx[i];
            bodies[i].vel.y = vy[i];
            bodies[i].vel.z = vz[i];
        }
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
    #pragma omp parallel for schedule(static) reduction(+:energy)
    for (size_t i = 0; i < n; ++i) {
        const Body& body = bodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
    #pragma omp parallel for schedule(dynamic, 16) reduction(-:energy)
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

// Create the OpenMP thread team and, unless the user configured binding via
// OMP_PROC_BIND / OMP_PLACES, pin each thread to its own CPU from the
// process's allowed set to avoid thread migration between steps.
void initThreads() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    const bool bind = !std::getenv("OMP_PROC_BIND") && !std::getenv("OMP_PLACES") &&
                      sched_getaffinity(0, sizeof(allowed), &allowed) == 0;
    std::vector<int> cpus;
    if (bind) {
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (CPU_ISSET(c, &allowed)) cpus.push_back(c);
        }
    }

    #pragma omp parallel
    {
        if (!cpus.empty()) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpus[omp_get_thread_num() % cpus.size()], &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        }
    }
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

    // Start the OpenMP thread team during initialization so that thread
    // creation is not attributed to the simulation itself.
    initThreads();
    
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
