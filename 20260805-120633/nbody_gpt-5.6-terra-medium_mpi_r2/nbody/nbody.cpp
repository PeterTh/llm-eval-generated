#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <type_traits>
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
static_assert(std::is_standard_layout_v<Body>);

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

void computeForces(std::vector<Body>& localBodies, const std::vector<Vec3>& positions) {
    const size_t n = positions.size();
    
    for (Body& body : localBodies) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (size_t j = 0; j < n; ++j) {
            const double dx = positions[j].x - body.pos.x;
            const double dy = positions[j].y - body.pos.y;
            const double dz = positions[j].z - body.pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        
        body.vel.x += DT * Fx;
        body.vel.y += DT * Fy;
        body.vel.z += DT * Fz;
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

    if (numBodies < 0 || numSteps < 0) {
        if (rank == 0) {
            printf("Number of bodies and steps must be non-negative\n");
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
    
    // Bodies are owned by one rank.  Every rank receives the global positions
    // each step, which is the only remote data required for its force updates.
    std::vector<int> bodyCounts(worldSize), bodyDisplacements(worldSize);
    std::vector<int> scalarCounts(worldSize), scalarDisplacements(worldSize);
    const int baseCount = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    int displacement = 0;
    for (int r = 0; r < worldSize; ++r) {
        bodyCounts[r] = baseCount + (r < remainder ? 1 : 0);
        bodyDisplacements[r] = displacement;
        scalarCounts[r] = 3 * bodyCounts[r];
        scalarDisplacements[r] = 3 * displacement;
        displacement += bodyCounts[r];
    }

    const int localCount = bodyCounts[rank];
    std::vector<Body> localBodies(localCount);
    std::vector<Body> allBodies;
    if (rank == 0) {
        allBodies.resize(numBodies);
        randomizeBodies(allBodies);
    }

    std::vector<int> bodyScalarCounts(worldSize), bodyScalarDisplacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        bodyScalarCounts[r] = 6 * bodyCounts[r];
        bodyScalarDisplacements[r] = 6 * bodyDisplacements[r];
    }
    MPI_Scatterv(rank == 0 ? reinterpret_cast<double*>(allBodies.data()) : nullptr,
                 bodyScalarCounts.data(), bodyScalarDisplacements.data(), MPI_DOUBLE,
                 reinterpret_cast<double*>(localBodies.data()), 6 * localCount, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // The resized datatype gathers the three position doubles from each Body
    // without communicating its velocity fields.
    MPI_Datatype contiguousPosition;
    MPI_Datatype bodyPosition;
    MPI_Type_contiguous(3, MPI_DOUBLE, &contiguousPosition);
    MPI_Type_create_resized(contiguousPosition, 0, sizeof(Body), &bodyPosition);
    MPI_Type_commit(&bodyPosition);
    MPI_Type_free(&contiguousPosition);

    std::vector<Vec3> positions(numBodies);
    MPI_Allgatherv(localBodies.data(), localCount, bodyPosition,
                   positions.data(), scalarCounts.data(), scalarDisplacements.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(localBodies, positions);
        integrateBodies(localBodies);
        MPI_Allgatherv(localBodies.data(), localCount, bodyPosition,
                       positions.data(), scalarCounts.data(), scalarDisplacements.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }
    
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Simulation time: %.0f ms\n", duration * 1000.0);
    }

    // Avoid a full-state gather in benchmark-only runs.  The root needs it
    // only for output hashing and the serial-equivalent validation routines.
    if (printResults || validate) {
        if (rank == 0) {
            allBodies.resize(numBodies);
        }
        MPI_Gatherv(reinterpret_cast<double*>(localBodies.data()), 6 * localCount, MPI_DOUBLE,
                    rank == 0 ? reinterpret_cast<double*>(allBodies.data()) : nullptr,
                    bodyScalarCounts.data(), bodyScalarDisplacements.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (const auto& body : allBodies) {
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
    int result = 0;
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(allBodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(allBodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            result = 0;
        } else {
            printf("Validation: FAILED\n");
            result = 1;
        }
    }

    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Type_free(&bodyPosition);
    MPI_Finalize();
    return result;
}
