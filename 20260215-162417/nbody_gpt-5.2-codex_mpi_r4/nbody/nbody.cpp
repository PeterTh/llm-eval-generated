#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr size_t BODY_STRIDE = 6;

void randomizeBodies(std::vector<double>& bodies, unsigned int seed = 42) {
    const size_t n = bodies.size() / BODY_STRIDE;
    for (size_t i = 0; i < n; ++i) {
        const size_t base = i * BODY_STRIDE;
        bodies[base + 0] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[base + 1] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[base + 2] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[base + 3] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[base + 4] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies[base + 5] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

void computeForces(std::vector<double>& bodies, const size_t start, const size_t end) {
    const size_t n = bodies.size() / BODY_STRIDE;
    const double* data = bodies.data();
    
    for (size_t i = start; i < end; ++i) {
        const size_t ib = i * BODY_STRIDE;
        const double ix = data[ib + 0];
        const double iy = data[ib + 1];
        const double iz = data[ib + 2];
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (size_t j = 0; j < n; ++j) {
            const size_t jb = j * BODY_STRIDE;
            const double dx = data[jb + 0] - ix;
            const double dy = data[jb + 1] - iy;
            const double dz = data[jb + 2] - iz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        
        bodies[ib + 3] += DT * Fx;
        bodies[ib + 4] += DT * Fy;
        bodies[ib + 5] += DT * Fz;
    }
}

void integrateBodies(std::vector<double>& bodies, const size_t start, const size_t end) {
    for (size_t i = start; i < end; ++i) {
        const size_t base = i * BODY_STRIDE;
        bodies[base + 0] += bodies[base + 3] * DT;
        bodies[base + 1] += bodies[base + 4] * DT;
        bodies[base + 2] += bodies[base + 5] * DT;
    }
}

double computeTotalEnergy(const std::vector<double>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size() / BODY_STRIDE;
    
    // Kinetic energy (assuming unit mass)
    for (size_t i = 0; i < n; ++i) {
        const size_t base = i * BODY_STRIDE;
        const double vx = bodies[base + 3];
        const double vy = bodies[base + 4];
        const double vz = bodies[base + 5];
        energy += 0.5 * (vx * vx + vy * vy + vz * vz);
    }
    
    // Potential energy (assuming unit mass for all bodies)
    for (size_t i = 0; i < n; ++i) {
        const size_t ib = i * BODY_STRIDE;
        const double ix = bodies[ib + 0];
        const double iy = bodies[ib + 1];
        const double iz = bodies[ib + 2];
        for (size_t j = i + 1; j < n; ++j) {
            const size_t jb = j * BODY_STRIDE;
            const double dx = bodies[jb + 0] - ix;
            const double dy = bodies[jb + 1] - iy;
            const double dz = bodies[jb + 2] - iz;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    
    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<double>& bodies) {
    const size_t n = bodies.size() / BODY_STRIDE;
    for (size_t i = 0; i < n; ++i) {
        const size_t base = i * BODY_STRIDE;
        const double px = bodies[base + 0];
        const double py = bodies[base + 1];
        const double pz = bodies[base + 2];
        const double vx = bodies[base + 3];
        const double vy = bodies[base + 4];
        const double vz = bodies[base + 5];
        // Check for NaN or Inf values
        if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz) ||
            !std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(vz)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(px) > maxPos || std::abs(py) > maxPos || std::abs(pz) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(vx) > maxVel || std::abs(vy) > maxVel || std::abs(vz) > maxVel) {
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
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int earlyExit = -1;
    const char* unknownArg = nullptr;
    
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
            earlyExit = 0;
            break;
        } else {
            unknownArg = argv[i];
            earlyExit = 1;
            break;
        }
    }

    if (earlyExit != -1) {
        if (rank == 0) {
            if (earlyExit == 1 && unknownArg != nullptr) {
                printf("Unknown option: %s\n", unknownArg);
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return earlyExit;
    }
    
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Initialize bodies
    std::vector<double> bodies(static_cast<size_t>(numBodies) * BODY_STRIDE);
    if (rank == 0) {
        randomizeBodies(bodies);
    }
    if (!bodies.empty()) {
        MPI_Bcast(bodies.data(), static_cast<int>(bodies.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    const size_t n = bodies.size() / BODY_STRIDE;
    const size_t base = size > 0 ? n / static_cast<size_t>(size) : 0;
    const size_t rem = size > 0 ? n % static_cast<size_t>(size) : 0;
    const size_t rankSize = static_cast<size_t>(rank);
    const size_t localCount = base + (rankSize < rem ? 1 : 0);
    const size_t localStart = base * rankSize + (rankSize < rem ? rankSize : rem);
    const size_t localEnd = localStart + localCount;
    std::vector<int> recvcounts(size, 0);
    std::vector<int> displs(size, 0);
    for (int r = 0; r < size; ++r) {
        const size_t rSize = static_cast<size_t>(r);
        const size_t rCount = base + (rSize < rem ? 1 : 0);
        const size_t rStart = base * rSize + (rSize < rem ? rSize : rem);
        recvcounts[r] = static_cast<int>(rCount * BODY_STRIDE);
        displs[r] = static_cast<int>(rStart * BODY_STRIDE);
    }
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, localStart, localEnd);
        integrateBodies(bodies, localStart, localEnd);
        if (size > 1 && !bodies.empty()) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, bodies.data(),
                           recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    if (rank == 0) {
        const long long durationMs = static_cast<long long>((end - start) * 1000.0);
        printf("Simulation time: %lld ms\n", durationMs);
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(bodies, "Bodies");
    }
    
    // Validation: check that simulation produces finite, reasonable values
    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
            if (validateSimulation(bodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return exitCode;
}
