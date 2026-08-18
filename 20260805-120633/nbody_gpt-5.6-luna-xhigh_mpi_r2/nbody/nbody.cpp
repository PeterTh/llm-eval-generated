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

static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must contain three contiguous doubles");
static_assert(sizeof(Body) == 6 * sizeof(double), "Body must contain six contiguous doubles");

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

void computeForces(std::vector<Body>& bodies, const size_t firstBody, const size_t lastBody) {
    const size_t n = bodies.size();
    const Body* const allBodies = bodies.data();
    
    for (size_t i = firstBody; i < lastBody; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        const double xi = allBodies[i].pos.x;
        const double yi = allBodies[i].pos.y;
        const double zi = allBodies[i].pos.z;
        
        for (size_t j = 0; j < n; ++j) {
            const double dx = allBodies[j].pos.x - xi;
            const double dy = allBodies[j].pos.y - yi;
            const double dz = allBodies[j].pos.z - zi;
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

void integrateBodies(std::vector<Body>& bodies, const size_t firstBody, const size_t lastBody) {
    for (size_t i = firstBody; i < lastBody; ++i) {
        auto& body = bodies[i];
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

    const size_t bodyCount = static_cast<size_t>(numBodies);
    const size_t baseBodyCount = bodyCount / static_cast<size_t>(worldSize);
    const size_t remainder = bodyCount % static_cast<size_t>(worldSize);
    const size_t rankIndex = static_cast<size_t>(rank);
    const size_t firstBody = rankIndex * baseBodyCount +
                             (rankIndex < remainder ? rankIndex : remainder);
    const size_t localBodyCount = baseBodyCount + (rankIndex < remainder ? 1 : 0);
    const size_t lastBody = firstBody + localBodyCount;

    std::vector<int> bodyCounts(static_cast<size_t>(worldSize));
    std::vector<int> bodyDisplacements(static_cast<size_t>(worldSize));
    for (int process = 0; process < worldSize; ++process) {
        const size_t processIndex = static_cast<size_t>(process);
        const size_t processCount = baseBodyCount + (processIndex < remainder ? 1 : 0);
        const size_t processStart = processIndex * baseBodyCount +
                                    (processIndex < remainder ? processIndex : remainder);
        if (processCount > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            processStart > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) {
                printf("Too many bodies for MPI distribution\n");
            }
            MPI_Finalize();
            return 1;
        }
        bodyCounts[static_cast<size_t>(process)] = static_cast<int>(processCount);
        bodyDisplacements[static_cast<size_t>(process)] = static_cast<int>(processStart);
    }

    MPI_Datatype mpiBodyType;
    MPI_Type_contiguous(6, MPI_DOUBLE, &mpiBodyType);
    MPI_Type_commit(&mpiBodyType);

    MPI_Datatype contiguousPositionType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &contiguousPositionType);
    MPI_Type_commit(&contiguousPositionType);

    // A position occupies the first three doubles of Body. Resizing the
    // datatype gives MPI the stride between consecutive Body objects while
    // keeping the velocity fields out of the per-step collective.
    MPI_Datatype mpiPositionType;
    MPI_Type_create_resized(contiguousPositionType, 0, sizeof(Body), &mpiPositionType);
    MPI_Type_commit(&mpiPositionType);
    MPI_Type_free(&contiguousPositionType);
    
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Initialize bodies
    std::vector<Body> bodies(bodyCount);
    if (rank == 0) {
        randomizeBodies(bodies);
    }
    MPI_Bcast(bodies.data(), numBodies, mpiBodyType, 0, MPI_COMM_WORLD);
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, firstBody, lastBody);
        integrateBodies(bodies, firstBody, lastBody);
        // All ranks need current positions for the next force calculation.
        // Velocities are only needed by the rank that owns each body.
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       bodies.data(), bodyCounts.data(), bodyDisplacements.data(),
                       mpiPositionType, MPI_COMM_WORLD);
    }
    
    const double elapsed = MPI_Wtime() - start;
    double maximumElapsed = 0.0;
    MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMilliseconds = static_cast<long>(maximumElapsed * 1000.0);
        printf("Simulation time: %ld ms\n", durationMilliseconds);
    }

    // Reassemble complete body records only once, for root-side result
    // serialization and validation. During the simulation, remote velocity
    // fields are intentionally not communicated because they are never read.
    Body* localSendBuffer = localBodyCount == 0 ? nullptr : bodies.data() + firstBody;
    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : localSendBuffer,
                rank == 0 ? 0 : static_cast<int>(localBodyCount),
                rank == 0 ? MPI_DATATYPE_NULL : mpiBodyType,
                bodies.data(), bodyCounts.data(), bodyDisplacements.data(),
                mpiBodyType, 0, MPI_COMM_WORLD);
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(bodyCount * 6);
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
        bool valid = true;
        if (rank == 0) {
            printf("Validating simulation results...\n");
            valid = validateSimulation(bodies);
        }
        int validOnRoot = valid ? 1 : 0;
        MPI_Bcast(&validOnRoot, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (validOnRoot != 0) {
            // Report final energy for reference
            if (rank == 0) {
                const double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                printf("Validation: FAILED\n");
            }
            MPI_Type_free(&mpiPositionType);
            MPI_Type_free(&mpiBodyType);
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Type_free(&mpiPositionType);
    MPI_Type_free(&mpiBodyType);
    MPI_Finalize();
    return 0;
}
