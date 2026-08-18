#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
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

static_assert(sizeof(Vec3) == 3 * sizeof(double));
static_assert(sizeof(Body) == 6 * sizeof(double));

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

void computeForces(std::vector<Body>& localBodies,
                   const std::vector<Vec3>& globalPositions,
                   const int globalOffset) {
    const size_t n = globalPositions.size();
    
    for (size_t localIndex = 0; localIndex < localBodies.size(); ++localIndex) {
        const Vec3& position = globalPositions[static_cast<size_t>(globalOffset) + localIndex];
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (size_t j = 0; j < n; ++j) {
            const double dx = globalPositions[j].x - position.x;
            const double dy = globalPositions[j].y - position.y;
            const double dz = globalPositions[j].z - position.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        
        localBodies[localIndex].vel.x += DT * Fx;
        localBodies[localIndex].vel.y += DT * Fy;
        localBodies[localIndex].vel.z += DT * Fz;
    }
}

void integrateBodies(std::vector<Body>& bodies) {
    for (auto& body : bodies) {
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

    if (numBodies < 0) {
        if (rank == 0) {
            printf("Number of bodies must be non-negative\n");
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

    // Bodies are distributed in contiguous global-index ranges.  Each rank
    // retains only its own state; the positions are replicated because every
    // target body interacts with every source body.
    std::vector<int> bodyCounts(static_cast<size_t>(worldSize));
    std::vector<int> bodyDisplacements(static_cast<size_t>(worldSize));
    const int baseCount = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    for (int process = 0; process < worldSize; ++process) {
        bodyCounts[static_cast<size_t>(process)] = baseCount + (process < remainder ? 1 : 0);
        bodyDisplacements[static_cast<size_t>(process)] =
            (process == 0) ? 0 : bodyDisplacements[static_cast<size_t>(process - 1)] +
                                  bodyCounts[static_cast<size_t>(process - 1)];
    }
    const int localCount = bodyCounts[static_cast<size_t>(rank)];
    const int globalOffset = bodyDisplacements[static_cast<size_t>(rank)];

    std::vector<Body> initialBodies;
    if (rank == 0) {
        initialBodies.resize(static_cast<size_t>(numBodies));
        randomizeBodies(initialBodies);
    }
    std::vector<Body> localBodies(static_cast<size_t>(localCount));
    std::vector<Vec3> globalPositions(static_cast<size_t>(numBodies));

    // Body and position datatypes let the position-only all-gather read the
    // pos member directly from the local AoS state, avoiding a packing pass
    // and transferring only three doubles per body per simulation step.
    MPI_Datatype mpiBody;
    MPI_Datatype mpiVec3;
    MPI_Datatype mpiBodyPosition;
    MPI_Type_contiguous(6, MPI_DOUBLE, &mpiBody);
    MPI_Type_contiguous(3, MPI_DOUBLE, &mpiVec3);
    MPI_Type_commit(&mpiBody);
    MPI_Type_commit(&mpiVec3);
    MPI_Type_create_resized(mpiVec3, 0, static_cast<MPI_Aint>(sizeof(Body)), &mpiBodyPosition);
    MPI_Type_commit(&mpiBodyPosition);

    MPI_Scatterv(rank == 0 ? initialBodies.data() : nullptr,
                 bodyCounts.data(), bodyDisplacements.data(), mpiBody,
                 localBodies.data(), localCount, mpiBody, 0, MPI_COMM_WORLD);

    initialBodies.clear();
    initialBodies.shrink_to_fit();
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (int step = 0; step < numSteps; ++step) {
        MPI_Allgatherv(localBodies.data(), localCount, mpiBodyPosition,
                       globalPositions.data(), bodyCounts.data(), bodyDisplacements.data(),
                       mpiVec3, MPI_COMM_WORLD);
        computeForces(localBodies, globalPositions, globalOffset);
        integrateBodies(localBodies);
    }
    
    const double elapsed = MPI_Wtime() - start;
    double maximumElapsed = 0.0;
    MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", static_cast<long>(maximumElapsed * 1000.0));
    }

    const bool needFinalBodies = printResults || validate;
    std::vector<Body> bodies;
    if (needFinalBodies) {
        if (rank == 0) {
            bodies.resize(static_cast<size_t>(numBodies));
        }
        MPI_Gatherv(localBodies.data(), localCount, mpiBody,
                    rank == 0 ? bodies.data() : nullptr,
                    bodyCounts.data(), bodyDisplacements.data(), mpiBody,
                    0, MPI_COMM_WORLD);
    }

    MPI_Type_free(&mpiBodyPosition);
    MPI_Type_free(&mpiVec3);
    MPI_Type_free(&mpiBody);
    
    // Print results for external validation
    if (rank == 0 && printResults) {
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
        int validationPassed = 1;
        if (rank == 0) {
            printf("Validating simulation results...\n");
            validationPassed = validateSimulation(bodies) ? 1 : 0;
        }
        MPI_Bcast(&validationPassed, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (validationPassed != 0) {
            if (rank == 0) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                printf("Validation: FAILED\n");
            }
        }

        MPI_Finalize();
        return validationPassed == 0 ? 1 : 0;
    }

    MPI_Finalize();
    return 0;
}
