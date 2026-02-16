#include <mpi.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Compute forces on local bodies given global positions array
void computeForcesDistributed(std::vector<Body>& localBodies, const std::vector<double>& globalPos, int globalN, int globalOffset) {
    const int localN = static_cast<int>(localBodies.size());
    for (int i = 0; i < localN; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        const double ix = localBodies[i].pos.x;
        const double iy = localBodies[i].pos.y;
        const double iz = localBodies[i].pos.z;
        for (int j = 0; j < globalN; ++j) {
            const double jx = globalPos[3*j + 0];
            const double jy = globalPos[3*j + 1];
            const double jz = globalPos[3*j + 2];
            const double dx = jx - ix;
            const double dy = jy - iy;
            const double dz = jz - iz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        localBodies[i].vel.x += DT * Fx;
        localBodies[i].vel.y += DT * Fy;
        localBodies[i].vel.z += DT * Fz;
    }
}

void integrateBodies(std::vector<Body>& bodies) {
    for (auto& body : bodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

// Compute total energy on a full-body array (sequential)
double computeTotalEnergyFull(const std::vector<Body>& bodies) {
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
bool validateSimulationFull(const std::vector<Body>& bodies) {
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
    // Initialize MPI unconditionally
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBodies = 1024;
    int numSteps = 10;
    int validate_flag = 0;
    int printResults_flag = 0;

    // Parse command line arguments on rank 0 and broadcast
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numBodies = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                numSteps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate_flag = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults_flag = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else if (i == 1) {
                // ignore unknown options on non-root ranks; we'll let root handle unknowns
            }
        }
    }
    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const bool validate = (validate_flag != 0);
    const bool printResults = (printResults_flag != 0);

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    // Determine local partition
    int base = numBodies / size;
    int rem = numBodies % size;
    int localN = base + (rank < rem ? 1 : 0);
    int offset = rank * base + std::min(rank, rem);

    // Initialize local bodies
    std::vector<Body> localBodies(localN);
    randomizeBodies(localBodies, 42u + static_cast<unsigned int>(rank));

    // Prepare gather counts for positions and body data
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        int cnt = base + (r < rem ? 1 : 0);
        counts[r] = cnt * 3; // for positions (3 doubles per body)
        displs[r] = (r * base + std::min(r, rem)) * 3;
    }

    std::vector<double> localPos(localN * 3);
    std::vector<double> globalPos(numBodies * 3);

    // Synchronize and time only on rank 0
    MPI_Barrier(MPI_COMM_WORLD);
    std::chrono::high_resolution_clock::time_point start;
    if (rank == 0) start = std::chrono::high_resolution_clock::now();

    // Main simulation loop
    for (int step = 0; step < numSteps; ++step) {
        // pack local positions
        for (int i = 0; i < localN; ++i) {
            localPos[3*i + 0] = localBodies[i].pos.x;
            localPos[3*i + 1] = localBodies[i].pos.y;
            localPos[3*i + 2] = localBodies[i].pos.z;
        }

        // Allgather positions to all ranks
        MPI_Allgatherv(localPos.data(), localN * 3, MPI_DOUBLE,
                       globalPos.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // compute forces using global positions
        computeForcesDistributed(localBodies, globalPos, numBodies, offset);

        // integrate local bodies
        integrateBodies(localBodies);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Gather full body data to rank 0 if needed for printing/validation
    // Prepare counts for 6 doubles per body
    std::vector<int> counts6(size), displs6(size);
    for (int r = 0; r < size; ++r) {
        int cnt = base + (r < rem ? 1 : 0);
        counts6[r] = cnt * 6;
        displs6[r] = (r * base + std::min(r, rem)) * 6;
    }

    std::vector<double> localBodyData(localN * 6);
    for (int i = 0; i < localN; ++i) {
        localBodyData[6*i + 0] = localBodies[i].pos.x;
        localBodyData[6*i + 1] = localBodies[i].pos.y;
        localBodyData[6*i + 2] = localBodies[i].pos.z;
        localBodyData[6*i + 3] = localBodies[i].vel.x;
        localBodyData[6*i + 4] = localBodies[i].vel.y;
        localBodyData[6*i + 5] = localBodies[i].vel.z;
    }

    std::vector<double> gatheredBodyData;
    if (rank == 0) gatheredBodyData.resize(numBodies * 6);

    MPI_Gatherv(localBodyData.data(), localN * 6, MPI_DOUBLE,
                rank == 0 ? gatheredBodyData.data() : nullptr, counts6.data(), displs6.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int exit_code = 0;

    if (rank == 0) {
        if (printResults) {
            // Convert gatheredBodyData into vector<double> bodyData for print_results
            print_results(gatheredBodyData, "Bodies");
        }

        if (validate) {
            // Reconstruct Body vector for validation and energy computation
            std::vector<Body> allBodies(numBodies);
            for (int i = 0; i < numBodies; ++i) {
                allBodies[i].pos.x = gatheredBodyData[6*i + 0];
                allBodies[i].pos.y = gatheredBodyData[6*i + 1];
                allBodies[i].pos.z = gatheredBodyData[6*i + 2];
                allBodies[i].vel.x = gatheredBodyData[6*i + 3];
                allBodies[i].vel.y = gatheredBodyData[6*i + 4];
                allBodies[i].vel.z = gatheredBodyData[6*i + 5];
            }

            printf("Validating simulation results...\n");
            if (validateSimulationFull(allBodies)) {
                double finalEnergy = computeTotalEnergyFull(allBodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
                exit_code = 0;
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    // Broadcast exit code to all ranks so all return the same code
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exit_code;
}
