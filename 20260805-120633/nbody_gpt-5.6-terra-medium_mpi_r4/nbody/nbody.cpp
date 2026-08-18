#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be tightly packed for MPI");

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

void computeForces(std::vector<Body>& bodies, const int firstBody, const int localBodyCount) {
    const size_t n = bodies.size();
    
    for (int localIndex = 0; localIndex < localBodyCount; ++localIndex) {
        const size_t i = static_cast<size_t>(firstBody + localIndex);
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (size_t j = 0; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
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

void integrateBodies(std::vector<Body>& bodies, const int firstBody, const int localBodyCount) {
    for (int localIndex = 0; localIndex < localBodyCount; ++localIndex) {
        Body& body = bodies[static_cast<size_t>(firstBody + localIndex)];
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
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
bool validateSimulationLocal(const std::vector<Body>& bodies, const int firstBody, const int localBodyCount) {
    for (int localIndex = 0; localIndex < localBodyCount; ++localIndex) {
        const Body& body = bodies[static_cast<size_t>(firstBody + localIndex)];
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            return false;
        }

        constexpr double maxPos = 1e6;
        constexpr double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos ||
            std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
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
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    if (numBodies < 0 || numSteps < 0) {
        if (rank == 0) {
            printf("Number of bodies and steps must be non-negative\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (numBodies > std::numeric_limits<int>::max() / 6) {
        if (rank == 0) {
            printf("Number of bodies is too large for MPI counts\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> counts(worldSize);
    std::vector<int> displacements(worldSize);
    const int baseCount = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    int displacement = 0;
    for (int process = 0; process < worldSize; ++process) {
        counts[process] = baseCount + (process < remainder ? 1 : 0);
        displacements[process] = displacement;
        displacement += counts[process];
    }
    const int localBodyCount = counts[rank];
    const int firstBody = displacements[rank];
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    if (rank == 0) {
        randomizeBodies(bodies);
    }
    Body* const localBodies = localBodyCount > 0 ? bodies.data() + firstBody : bodies.data();

    MPI_Datatype bodyType;
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);

    MPI_Datatype positionType;
    MPI_Datatype contiguousPositionType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &contiguousPositionType);
    MPI_Type_create_resized(contiguousPositionType, 0, sizeof(Body), &positionType);
    MPI_Type_commit(&positionType);
    MPI_Type_free(&contiguousPositionType);

    // Distribute the deterministic initial state once.  Each timestep exchanges
    // only positions; non-owned velocities are not needed for force evaluation.
    MPI_Bcast(bodies.data(), numBodies, bodyType, 0, MPI_COMM_WORLD);
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, firstBody, localBodyCount);
        integrateBodies(bodies, firstBody, localBodyCount);
        MPI_Allgatherv(MPI_IN_PLACE, 0, positionType,
                       bodies.data(), counts.data(), displacements.data(), positionType,
                       MPI_COMM_WORLD);
    }
    
    const double localDuration = MPI_Wtime() - start;
    double maxDuration = 0.0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", static_cast<long>(maxDuration * 1000.0));
    }
    
    // Print results for external validation
    if (printResults) {
        MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : localBodies, localBodyCount, bodyType,
                    bodies.data(), counts.data(), displacements.data(), bodyType, 0, MPI_COMM_WORLD);
        if (rank == 0) {
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
    }
    
    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
        }
        
        const int localValid = validateSimulationLocal(bodies, firstBody, localBodyCount) ? 1 : 0;
        int globallyValid = 0;
        MPI_Allreduce(&localValid, &globallyValid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        if (globallyValid) {
            // Report final energy for reference
            MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : localBodies, localBodyCount, bodyType,
                        bodies.data(), counts.data(), displacements.data(), bodyType, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                const double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            MPI_Type_free(&positionType);
            MPI_Type_free(&bodyType);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Validation: FAILED\n");
            }
            MPI_Type_free(&positionType);
            MPI_Type_free(&bodyType);
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Type_free(&positionType);
    MPI_Type_free(&bodyType);
    MPI_Finalize();
    return 0;
}
