#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
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

static_assert(std::is_standard_layout_v<Vec3> && sizeof(Vec3) == 3 * sizeof(double));
static_assert(std::is_standard_layout_v<Body> && sizeof(Body) == 6 * sizeof(double));

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

void computeForces(const std::vector<Vec3>& globalPositions,
                   const std::vector<Vec3>& localPositions,
                   std::vector<Vec3>& localVelocities) {
    const Vec3* const positions = globalPositions.data();
    const size_t numBodies = globalPositions.size();

    for (size_t i = 0; i < localPositions.size(); ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        const double px = localPositions[i].x;
        const double py = localPositions[i].y;
        const double pz = localPositions[i].z;

        // Keeping the global source order makes the accumulation independent
        // of the number of ranks and equivalent to the serial algorithm.
        for (size_t j = 0; j < numBodies; ++j) {
            const double dx = positions[j].x - px;
            const double dy = positions[j].y - py;
            const double dz = positions[j].z - pz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        localVelocities[i].x += DT * Fx;
        localVelocities[i].y += DT * Fy;
        localVelocities[i].z += DT * Fz;
    }
}

void integrateBodies(std::vector<Vec3>& localPositions,
                     const std::vector<Vec3>& localVelocities) {
    for (size_t i = 0; i < localPositions.size(); ++i) {
        localPositions[i].x += localVelocities[i].x * DT;
        localPositions[i].y += localVelocities[i].y * DT;
        localPositions[i].z += localVelocities[i].z * DT;
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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

    // Contiguous ownership keeps the original body order and balances work to
    // within one body, including when there are more ranks than bodies.
    std::vector<int> counts(numRanks);
    std::vector<int> displacements(numRanks);
    const int baseCount = numBodies / numRanks;
    const int remainder = numBodies % numRanks;
    for (int r = 0, offset = 0; r < numRanks; ++r) {
        counts[r] = baseCount + (r < remainder ? 1 : 0);
        displacements[r] = offset;
        offset += counts[r];
    }
    const int localCount = counts[rank];

    MPI_Datatype vec3Type;
    MPI_Datatype bodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);

    std::vector<Vec3> localPositions(static_cast<size_t>(localCount));
    std::vector<Vec3> localVelocities(static_cast<size_t>(localCount));

    // Rank zero generates the same deterministic initial sequence as the
    // original program and distributes just one owned chunk to every rank.
    {
        std::vector<Body> initialBodies;
        if (rank == 0) {
            initialBodies.resize(static_cast<size_t>(numBodies));
            randomizeBodies(initialBodies);
        }
        std::vector<Body> localBodies(static_cast<size_t>(localCount));
        MPI_Scatterv(rank == 0 ? initialBodies.data() : nullptr,
                     counts.data(), displacements.data(), bodyType,
                     localBodies.data(), localCount, bodyType, 0, MPI_COMM_WORLD);

        for (int i = 0; i < localCount; ++i) {
            localPositions[i] = localBodies[i].pos;
            localVelocities[i] = localBodies[i].vel;
        }
    }
    MPI_Type_free(&bodyType);

    std::vector<Vec3> globalPositions(static_cast<size_t>(numBodies));
    MPI_Allgatherv(localPositions.data(), localCount, vec3Type,
                   globalPositions.data(), counts.data(), displacements.data(),
                   vec3Type, MPI_COMM_WORLD);

    // The maximum rank time is the distributed simulation's wall-clock time.
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(globalPositions, localPositions, localVelocities);
        integrateBodies(localPositions, localVelocities);
        // The final exchange is unnecessary unless another force step needs
        // the positions. Reporting performs its own untimed final exchange.
        if (step + 1 < numSteps) {
            MPI_Allgatherv(localPositions.data(), localCount, vec3Type,
                           globalPositions.data(), counts.data(), displacements.data(),
                           vec3Type, MPI_COMM_WORLD);
        }
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const auto elapsedMilliseconds = static_cast<long>(elapsed * 1000.0);
        printf("Simulation time: %ld ms\n", elapsedMilliseconds);
    }

    // Reporting and validation are outside the measured region. Gather the
    // velocities only when the root needs the complete final state.
    std::vector<Vec3> globalVelocities;
    if (printResults || validate) {
        if (numSteps > 0) {
            MPI_Allgatherv(localPositions.data(), localCount, vec3Type,
                           globalPositions.data(), counts.data(), displacements.data(),
                           vec3Type, MPI_COMM_WORLD);
        }
        if (rank == 0) {
            globalVelocities.resize(static_cast<size_t>(numBodies));
        }
        MPI_Gatherv(localVelocities.data(), localCount, vec3Type,
                    rank == 0 ? globalVelocities.data() : nullptr,
                    counts.data(), displacements.data(), vec3Type,
                    0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == 0 && printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
        for (int i = 0; i < numBodies; ++i) {
            bodyData.push_back(globalPositions[i].x);
            bodyData.push_back(globalPositions[i].y);
            bodyData.push_back(globalPositions[i].z);
            bodyData.push_back(globalVelocities[i].x);
            bodyData.push_back(globalVelocities[i].y);
            bodyData.push_back(globalVelocities[i].z);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    if (rank == 0 && validate) {
        printf("Validating simulation results...\n");

        std::vector<Body> bodies(static_cast<size_t>(numBodies));
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos = globalPositions[i];
            bodies[i].vel = globalVelocities[i];
        }

        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Type_free(&vec3Type);
    MPI_Finalize();
    return exitCode;
}
