#include <cmath>
#include <cstddef>
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

void computeForces(std::vector<Body>& bodies, const size_t firstBody, const size_t lastBody) {
    const size_t n = bodies.size();

    for (size_t i = firstBody; i < lastBody; ++i) {
        const double posX = bodies[i].pos.x;
        const double posY = bodies[i].pos.y;
        const double posZ = bodies[i].pos.z;
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (size_t j = 0; j < n; ++j) {
            const double dx = bodies[j].pos.x - posX;
            const double dy = bodies[j].pos.y - posY;
            const double dz = bodies[j].pos.z - posZ;
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

    // Bodies are distributed in contiguous ranges.  Every rank keeps the
    // complete source position state because each force calculation needs
    // every body.
    std::vector<int> bodyCounts(worldSize);
    std::vector<int> bodyDisplacements(worldSize);
    const int bodiesPerRank = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    for (int process = 0; process < worldSize; ++process) {
        bodyCounts[process] = bodiesPerRank + (process < remainder ? 1 : 0);
        bodyDisplacements[process] = process == 0
            ? 0
            : bodyDisplacements[process - 1] + bodyCounts[process - 1];
    }

    std::vector<Body> bodies(static_cast<size_t>(numBodies));
    if (rank == 0) {
        randomizeBodies(bodies);
    }

    // Describe the two Vec3 members without assuming that Body is an MPI
    // primitive type.  Its extent is exactly sizeof(Body), so counts and
    // displacements below are expressed in bodies rather than bytes.
    MPI_Datatype mpiBody;
    int blockLengths[2] = {3, 3};
    MPI_Aint memberDisplacements[2] = {
        static_cast<MPI_Aint>(offsetof(Body, pos)),
        static_cast<MPI_Aint>(offsetof(Body, vel))
    };
    MPI_Datatype memberTypes[2] = {MPI_DOUBLE, MPI_DOUBLE};
    MPI_Type_create_struct(2, blockLengths, memberDisplacements, memberTypes, &mpiBody);
    MPI_Type_commit(&mpiBody);

    // Force evaluation reads positions only.  Keep the datatype extent equal
    // to Body so body counts/displacements remain directly usable while each
    // timestep exchanges only the position member.
    MPI_Datatype mpiPositionMember;
    MPI_Datatype mpiBodyPositions;
    int positionBlockLength = 3;
    MPI_Aint positionDisplacement = static_cast<MPI_Aint>(offsetof(Body, pos));
    MPI_Type_create_struct(1, &positionBlockLength, &positionDisplacement,
                           memberTypes, &mpiPositionMember);
    MPI_Type_create_resized(mpiPositionMember, 0, static_cast<MPI_Aint>(sizeof(Body)),
                            &mpiBodyPositions);
    MPI_Type_commit(&mpiBodyPositions);
    MPI_Type_free(&mpiPositionMember);

    const int localCount = bodyCounts[rank];
    const size_t firstBody = static_cast<size_t>(bodyDisplacements[rank]);
    const size_t lastBody = firstBody + static_cast<size_t>(localCount);

    // Scatter the initial state, then make all positions available to every
    // rank.  The root uses MPI_IN_PLACE because its initialized array is also
    // its receive buffer for the root-owned range.
    if (numBodies > 0) {
        if (rank == 0) {
            MPI_Scatterv(bodies.data(), bodyCounts.data(), bodyDisplacements.data(), mpiBody,
                         MPI_IN_PLACE, 0, mpiBody, 0, MPI_COMM_WORLD);
        } else {
            MPI_Scatterv(bodies.data(), bodyCounts.data(), bodyDisplacements.data(), mpiBody,
                         bodies.data() + firstBody, localCount, mpiBody, 0, MPI_COMM_WORLD);
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, mpiBodyPositions,
                       bodies.data(), bodyCounts.data(), bodyDisplacements.data(), mpiBodyPositions,
                       MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, firstBody, lastBody);
        integrateBodies(bodies, firstBody, lastBody);

        // The next timestep needs the newly integrated positions from every
        // rank.  In-place allgather avoids an extra local-to-global copy on
        // the critical path.
        MPI_Allgatherv(MPI_IN_PLACE, 0, mpiBodyPositions,
                       bodies.data(), bodyCounts.data(), bodyDisplacements.data(), mpiBodyPositions,
                       MPI_COMM_WORLD);
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMilliseconds = static_cast<long>(elapsed * 1000.0);
        printf("Simulation time: %ld ms\n", durationMilliseconds);
    }

    // Velocities are private to their owning rank during the simulation.  A
    // full-state gather is needed only for rank 0's optional result output or
    // validation, and is deliberately outside the timed region.
    if ((printResults || validate) && numBodies > 0) {
        if (rank == 0) {
            MPI_Gatherv(MPI_IN_PLACE, 0, mpiBody,
                        bodies.data(), bodyCounts.data(), bodyDisplacements.data(), mpiBody,
                        0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(bodies.data() + firstBody, localCount, mpiBody,
                        bodies.data(), bodyCounts.data(), bodyDisplacements.data(), mpiBody,
                        0, MPI_COMM_WORLD);
        }
    }

    // Print results for external validation
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
    int exitCode = 0;
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");

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
    MPI_Type_free(&mpiBodyPositions);
    MPI_Type_free(&mpiBody);
    MPI_Finalize();
    return exitCode;
}
