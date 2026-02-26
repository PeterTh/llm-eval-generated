#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

void computeCounts(const int numBodies, const int numRanks, std::vector<int>& counts, std::vector<int>& displs) {
    counts.assign(numRanks, 0);
    displs.assign(numRanks, 0);
    const int base = numBodies / numRanks;
    const int rem = numBodies % numRanks;
    int offset = 0;
    for (int i = 0; i < numRanks; ++i) {
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = offset;
        offset += counts[i];
    }
}

void randomizeBodies(std::vector<double>& positions, std::vector<double>& velocities, unsigned int seed = 42) {
    const size_t n = positions.size() / 3;
    for (size_t i = 0; i < n; ++i) {
        const size_t idx = 3 * i;
        positions[idx] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        positions[idx + 1] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        positions[idx + 2] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velocities[idx] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velocities[idx + 1] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        velocities[idx + 2] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

void computeForces(const std::vector<double>& positions, std::vector<double>& velocities, const size_t offset, const size_t count) {
    const size_t n = positions.size() / 3;
    const double* pos = positions.data();
    double* vel = velocities.data();

    for (size_t i = 0; i < count; ++i) {
        const size_t bodyIndex = offset + i;
        const size_t p = 3 * bodyIndex;
        const double xi = pos[p];
        const double yi = pos[p + 1];
        const double zi = pos[p + 2];
        double Fx = 0.0;
        double Fy = 0.0;
        double Fz = 0.0;

        for (size_t j = 0; j < n; ++j) {
            const size_t pj = 3 * j;
            const double dx = pos[pj] - xi;
            const double dy = pos[pj + 1] - yi;
            const double dz = pos[pj + 2] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        const size_t v = 3 * i;
        vel[v] += DT * Fx;
        vel[v + 1] += DT * Fy;
        vel[v + 2] += DT * Fz;
    }
}

void integrateBodies(std::vector<double>& positions, const std::vector<double>& velocities, const size_t offset, const size_t count) {
    double* pos = positions.data();
    const double* vel = velocities.data();

    for (size_t i = 0; i < count; ++i) {
        const size_t bodyIndex = offset + i;
        const size_t p = 3 * bodyIndex;
        const size_t v = 3 * i;
        pos[p] += vel[v] * DT;
        pos[p + 1] += vel[v + 1] * DT;
        pos[p + 2] += vel[v + 2] * DT;
    }
}

double computeTotalEnergy(const std::vector<double>& positions, const std::vector<double>& velocities) {
    double energy = 0.0;
    const size_t n = positions.size() / 3;
    const double* pos = positions.data();
    const double* vel = velocities.data();

    // Kinetic energy (assuming unit mass)
    for (size_t i = 0; i < n; ++i) {
        const size_t v = 3 * i;
        energy += 0.5 * (vel[v] * vel[v] + vel[v + 1] * vel[v + 1] + vel[v + 2] * vel[v + 2]);
    }

    // Potential energy (assuming unit mass for all bodies)
    for (size_t i = 0; i < n; ++i) {
        const size_t pi = 3 * i;
        const double xi = pos[pi];
        const double yi = pos[pi + 1];
        const double zi = pos[pi + 2];
        for (size_t j = i + 1; j < n; ++j) {
            const size_t pj = 3 * j;
            const double dx = pos[pj] - xi;
            const double dy = pos[pj + 1] - yi;
            const double dz = pos[pj + 2] - zi;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<double>& positions, const std::vector<double>& velocities) {
    const size_t n = positions.size() / 3;
    const double* pos = positions.data();
    const double* vel = velocities.data();

    for (size_t i = 0; i < n; ++i) {
        const size_t p = 3 * i;
        // Check for NaN or Inf values
        if (!std::isfinite(pos[p]) || !std::isfinite(pos[p + 1]) || !std::isfinite(pos[p + 2]) ||
            !std::isfinite(vel[p]) || !std::isfinite(vel[p + 1]) || !std::isfinite(vel[p + 2])) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(pos[p]) > maxPos || std::abs(pos[p + 1]) > maxPos || std::abs(pos[p + 2]) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(vel[p]) > maxVel || std::abs(vel[p + 1]) > maxVel || std::abs(vel[p + 2]) > maxVel) {
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int numBodies = 1024;
    int numSteps = 10;
    int validate = 0;
    int printResults = 0;
    int showHelp = 0;
    int exitCode = 0;
    
    // Parse command line arguments
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numBodies = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                numSteps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitCode = 1;
                break;
            }
        }

        if (showHelp) {
            printUsage(argv[0]);
        }
    }

    int params[6] = {numBodies, numSteps, validate, printResults, showHelp, exitCode};
    MPI_Bcast(params, 6, MPI_INT, 0, MPI_COMM_WORLD);
    numBodies = params[0];
    numSteps = params[1];
    validate = params[2];
    printResults = params[3];
    showHelp = params[4];
    exitCode = params[5];

    if (showHelp || exitCode != 0) {
        MPI_Finalize();
        return exitCode;
    }
    
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> counts;
    std::vector<int> displs;
    computeCounts(numBodies, numRanks, counts, displs);
    std::vector<int> counts3(numRanks, 0);
    std::vector<int> displs3(numRanks, 0);
    for (int i = 0; i < numRanks; ++i) {
        counts3[i] = counts[i] * 3;
        displs3[i] = displs[i] * 3;
    }
    const int localCount = counts[rank];
    const int localOffset = displs[rank];
    
    // Initialize bodies
    std::vector<double> positions(static_cast<size_t>(numBodies) * 3);
    std::vector<double> localVelocities(static_cast<size_t>(localCount) * 3);
    std::vector<double> fullVelocities;
    if (rank == 0) {
        fullVelocities.resize(static_cast<size_t>(numBodies) * 3);
        randomizeBodies(positions, fullVelocities);
    }
    MPI_Bcast(positions.data(), static_cast<int>(positions.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const double* velSend = (rank == 0 && !fullVelocities.empty()) ? fullVelocities.data() : nullptr;
    MPI_Scatterv(velSend, counts3.data(), displs3.data(), MPI_DOUBLE,
                 localVelocities.data(), localCount * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(positions, localVelocities, static_cast<size_t>(localOffset), static_cast<size_t>(localCount));
        integrateBodies(positions, localVelocities, static_cast<size_t>(localOffset), static_cast<size_t>(localCount));
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       positions.data(), counts3.data(), displs3.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double localTime = end - start;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", static_cast<long>(maxTime * 1000.0));
    }
    
    // Print results for external validation
    int validationStatus = 0;
    if (printResults || validate) {
        if (rank == 0 && fullVelocities.empty()) {
            fullVelocities.resize(static_cast<size_t>(numBodies) * 3);
        }
        MPI_Gatherv(localVelocities.data(), localCount * 3, MPI_DOUBLE,
                    rank == 0 ? fullVelocities.data() : nullptr, counts3.data(), displs3.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                // Serialize body positions and velocities for hashing
                std::vector<double> bodyData;
                bodyData.reserve(static_cast<size_t>(numBodies) * 6);
                for (int i = 0; i < numBodies; ++i) {
                    const size_t p = static_cast<size_t>(i) * 3;
                    bodyData.push_back(positions[p]);
                    bodyData.push_back(positions[p + 1]);
                    bodyData.push_back(positions[p + 2]);
                    bodyData.push_back(fullVelocities[p]);
                    bodyData.push_back(fullVelocities[p + 1]);
                    bodyData.push_back(fullVelocities[p + 2]);
                }
                print_results(bodyData, "Bodies");
            }

            if (validate) {
                printf("Validating simulation results...\n");
                if (validateSimulation(positions, fullVelocities)) {
                    // Report final energy for reference
                    double finalEnergy = computeTotalEnergy(positions, fullVelocities);
                    printf("Final energy: %.6f\n", finalEnergy);
                    printf("Validation: PASSED\n");
                    validationStatus = 1;
                } else {
                    printf("Validation: FAILED\n");
                    validationStatus = 0;
                }
            }
        }

        if (validate) {
            MPI_Bcast(&validationStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Finalize();
            return validationStatus ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
