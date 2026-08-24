#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

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

// Compute forces on local bodies [localStart, localStart+localCount) against all bodies
void computeForces(std::vector<Body>& bodies, const std::vector<double>& allPos,
                   size_t localStart, size_t localCount) {
    const size_t n = bodies.size();

    for (size_t li = 0; li < localCount; ++li) {
        const size_t i = localStart + li;
        const double px = allPos[i * 3 + 0];
        const double py = allPos[i * 3 + 1];
        const double pz = allPos[i * 3 + 2];
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (size_t j = 0; j < n; ++j) {
            const double dx = allPos[j * 3 + 0] - px;
            const double dy = allPos[j * 3 + 1] - py;
            const double dz = allPos[j * 3 + 2] - pz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        bodies[i].vel.x += DT * Fx;
        bodies[i].vel.y += DT * Fy;
        bodies[i].vel.z += DT * Fz;
    }
}

void integrateBodies(std::vector<Body>& bodies, size_t localStart, size_t localCount) {
    for (size_t li = 0; li < localCount; ++li) {
        const size_t i = localStart + li;
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
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
    MPI_Init(&argc, &argv);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Initialize bodies identically on all ranks
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Compute partition: each rank owns a contiguous chunk of bodies
    const size_t n = static_cast<size_t>(numBodies);
    const size_t baseCount = n / numProcs;
    const size_t remainder = n % numProcs;
    // Ranks [0, remainder) get baseCount+1, others get baseCount
    const size_t localCount = baseCount + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    size_t localStart = 0;
    for (int r = 0; r < rank; ++r) {
        localStart += baseCount + (static_cast<size_t>(r) < remainder ? 1 : 0);
    }

    // Flat buffer for all positions (3 doubles per body) used in force computation
    std::vector<double> allPos(n * 3);
    // Precompute position gather counts/displs (3 doubles per body)
    std::vector<int> posCounts(numProcs);
    std::vector<int> posDispls(numProcs);
    {
        size_t off = 0;
        for (int r = 0; r < numProcs; ++r) {
            size_t cnt = baseCount + (static_cast<size_t>(r) < remainder ? 1 : 0);
            posCounts[r] = static_cast<int>(cnt * 3);
            posDispls[r] = static_cast<int>(off * 3);
            off += cnt;
        }
    }

    // Velocity gather counts/displs for final gather (3 doubles per body)
    std::vector<double> allVel(n * 3);
    std::vector<int> velCounts(posCounts);
    std::vector<int> velDispls(posDispls);

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Pack initial positions into allPos
    for (size_t i = 0; i < n; ++i) {
        allPos[i * 3 + 0] = bodies[i].pos.x;
        allPos[i * 3 + 1] = bodies[i].pos.y;
        allPos[i * 3 + 2] = bodies[i].pos.z;
    }

    for (int step = 0; step < numSteps; ++step) {
        // Compute forces on local bodies using global positions
        computeForces(bodies, allPos, localStart, localCount);

        // Integrate local bodies
        integrateBodies(bodies, localStart, localCount);

        // Pack local updated positions and gather for next step
        for (size_t li = 0; li < localCount; ++li) {
            const size_t i = localStart + li;
            allPos[i * 3 + 0] = bodies[i].pos.x;
            allPos[i * 3 + 1] = bodies[i].pos.y;
            allPos[i * 3 + 2] = bodies[i].pos.z;
        }
        MPI_Allgatherv(MPI_IN_PLACE,
                        0, MPI_DATATYPE_NULL,
                        allPos.data(), posCounts.data(), posDispls.data(),
                        MPI_DOUBLE, MPI_COMM_WORLD);
    }

    // Final gather of velocities for validation/results
    for (size_t li = 0; li < localCount; ++li) {
        const size_t i = localStart + li;
        allVel[i * 3 + 0] = bodies[i].vel.x;
        allVel[i * 3 + 1] = bodies[i].vel.y;
        allVel[i * 3 + 2] = bodies[i].vel.z;
    }
    MPI_Allgatherv(MPI_IN_PLACE,
                    0, MPI_DATATYPE_NULL,
                    allVel.data(), velCounts.data(), velDispls.data(),
                    MPI_DOUBLE, MPI_COMM_WORLD);

    // Reconstruct full body state from gathered data
    for (size_t i = 0; i < n; ++i) {
        bodies[i].pos.x = allPos[i * 3 + 0];
        bodies[i].pos.y = allPos[i * 3 + 1];
        bodies[i].pos.z = allPos[i * 3 + 2];
        bodies[i].vel.x = allVel[i * 3 + 0];
        bodies[i].vel.y = allVel[i * 3 + 1];
        bodies[i].vel.z = allVel[i * 3 + 2];
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long localDuration = static_cast<long long>(duration.count());
    long long globalDuration = 0;
    MPI_Reduce(&localDuration, &globalDuration, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Simulation time: %lld ms\n", globalDuration);
    }
    
    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
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
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
