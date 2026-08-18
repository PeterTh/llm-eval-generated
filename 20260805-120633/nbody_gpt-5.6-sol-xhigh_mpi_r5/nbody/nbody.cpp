#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-loop-vectorize")))
#endif
void computeForces(std::vector<Body>& localBodies, const std::vector<Vec3>& globalPositions) {
    const size_t n = globalPositions.size();

    for (auto& body : localBodies) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        const double px = body.pos.x;
        const double py = body.pos.y;
        const double pz = body.pos.z;

        // All ranks traverse sources in the same global order. Besides making
        // the access pattern contiguous, this keeps results independent of the
        // number of MPI ranks used for the run.
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
        for (size_t j = 0; j < n; ++j) {
            const double dx = globalPositions[j].x - px;
            const double dy = globalPositions[j].y - py;
            const double dz = globalPositions[j].z - pz;
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

void integrateBodies(std::vector<Body>& localBodies, std::vector<Vec3>& globalPositions,
                     const int globalOffset) {
    for (size_t i = 0; i < localBodies.size(); ++i) {
        Body& body = localBodies[i];
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
        globalPositions[static_cast<size_t>(globalOffset) + i] = body.pos;
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
    int parseResult = 0;

    // Parse command line arguments on every rank. MPI_Init has already removed
    // any launcher-specific arguments supplied by the MPI implementation.
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
            parseResult = 2;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
            parseResult = 1;
            break;
        }
    }

    if (parseResult == 0 && (numBodies < 0 || numSteps < 0)) {
        if (rank == 0) {
            printf("Number of bodies and simulation steps must be non-negative\n");
        }
        parseResult = 1;
    }

    if (parseResult != 0) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseResult == 2 ? 0 : 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Assign one contiguous, near-equal target range to every rank.
    std::vector<int> counts(static_cast<size_t>(worldSize));
    std::vector<int> offsets(static_cast<size_t>(worldSize));
    const int baseCount = numBodies / worldSize;
    const int remainder = numBodies % worldSize;
    int nextOffset = 0;
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = baseCount + (r < remainder ? 1 : 0);
        offsets[r] = nextOffset;
        nextOffset += counts[r];
    }

    MPI_Datatype vec3Type;
    MPI_Datatype bodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);

    // Only rank zero ever stores the complete body state. During timesteps,
    // ranks replicate positions (the read-only source data) but retain only
    // their own velocities and writable bodies.
    std::vector<Body> allBodies;
    if (rank == 0) {
        allBodies.resize(static_cast<size_t>(numBodies));
        randomizeBodies(allBodies);
    }

    const int localCount = counts[rank];
    const int localOffset = offsets[rank];
    std::vector<Body> localBodies(static_cast<size_t>(localCount));
    MPI_Scatterv(rank == 0 ? allBodies.data() : nullptr, counts.data(), offsets.data(), bodyType,
                 localBodies.data(), localCount, bodyType, 0, MPI_COMM_WORLD);

    std::vector<Vec3> globalPositions(static_cast<size_t>(numBodies));
    for (int i = 0; i < localCount; ++i) {
        globalPositions[static_cast<size_t>(localOffset + i)] = localBodies[static_cast<size_t>(i)].pos;
    }
    MPI_Allgatherv(MPI_IN_PLACE, localCount, vec3Type, globalPositions.data(), counts.data(),
                   offsets.data(), vec3Type, MPI_COMM_WORLD);

    // Time the synchronized distributed simulation. Rank zero reports the
    // slowest rank because that determines the application's elapsed time.
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(localBodies, globalPositions);
        integrateBodies(localBodies, globalPositions, localOffset);

        // The last exchange is unnecessary: final reporting gathers Body,
        // while every earlier exchange supplies the next step's source state.
        if (step + 1 < numSteps) {
            MPI_Allgatherv(MPI_IN_PLACE, localCount, vec3Type, globalPositions.data(), counts.data(),
                           offsets.data(), vec3Type, MPI_COMM_WORLD);
        }
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long elapsedMilliseconds = static_cast<long>(elapsed * 1000.0);
        printf("Simulation time: %ld ms\n", elapsedMilliseconds);
    }

    if (printResults || validate) {
        MPI_Gatherv(localBodies.data(), localCount, bodyType,
                    rank == 0 ? allBodies.data() : nullptr, counts.data(), offsets.data(), bodyType,
                    0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == 0) {
        // Print results for external validation.
        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(static_cast<size_t>(numBodies) * 6);
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

        // Validation: check that simulation produces finite, reasonable values.
        if (validate) {
            printf("Validating simulation results...\n");

            if (validateSimulation(allBodies)) {
                const double finalEnergy = computeTotalEnergy(allBodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Type_free(&bodyType);
    MPI_Type_free(&vec3Type);
    MPI_Finalize();
    return exitCode;
}
