#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
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

// The MPI types below use one element per position/body, so counts and
// displacements do not overflow through multiplication by three or six.
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

template <size_t Width>
void computeForceBlock(Body* bodies, const std::vector<Vec3>& positions) {
    alignas(64) double x[Width], y[Width], z[Width];
    alignas(64) double Fx[Width] = {}, Fy[Width] = {}, Fz[Width] = {};
    for (size_t k = 0; k < Width; ++k) {
        x[k] = bodies[k].pos.x;
        y[k] = bodies[k].pos.y;
        z[k] = bodies[k].pos.z;
    }

    // SIMD lanes are independent target bodies. Every lane visits sources in
    // their original order, without reassociating the floating-point reduction.
    for (const auto& source : positions) {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 1
#endif
        for (size_t k = 0; k < Width; ++k) {
            const double dx = source.x - x[k];
            const double dy = source.y - y[k];
            const double dz = source.z - z[k];
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

#if defined(__FMA__)
            // Keep fused accumulation even for scalar tail bodies; otherwise
            // vectorizing source products can change the rounding of the sum.
            Fx[k] = std::fma(dx, invDist3, Fx[k]);
            Fy[k] = std::fma(dy, invDist3, Fy[k]);
            Fz[k] = std::fma(dz, invDist3, Fz[k]);
#else
            Fx[k] += dx * invDist3;
            Fy[k] += dy * invDist3;
            Fz[k] += dz * invDist3;
#endif
        }
    }

    for (size_t k = 0; k < Width; ++k) {
        // Match the original velocity update's separately rounded products.
        const volatile double dvx = DT * Fx[k];
        const volatile double dvy = DT * Fy[k];
        const volatile double dvz = DT * Fz[k];
        bodies[k].vel.x += dvx;
        bodies[k].vel.y += dvy;
        bodies[k].vel.z += dvz;
    }
}

void computeForces(std::vector<Body>& bodies, const std::vector<Vec3>& positions) {
#if defined(__AVX512F__)
    constexpr size_t width = 8;
#elif defined(__AVX__)
    constexpr size_t width = 4;
#else
    constexpr size_t width = 2;
#endif
    size_t i = 0;
    for (; i + width <= bodies.size(); i += width) {
        computeForceBlock<width>(bodies.data() + i, positions);
    }
    for (; i < bodies.size(); ++i) {
        computeForceBlock<1>(bodies.data() + i, positions);
    }
}

void exchangePositions(const std::vector<Body>& bodies, std::vector<Vec3>& positions,
                       int offset, const std::vector<int>& counts,
                       const std::vector<int>& offsets, MPI_Datatype positionType,
                       MPI_Comm comm) {
    for (size_t i = 0; i < bodies.size(); ++i) {
        positions[static_cast<size_t>(offset) + i] = bodies[i].pos;
    }
    if (counts.size() > 1) {
        MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, positions.data(), counts.data(),
                       offsets.data(), positionType, comm);
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

int runSimulation(int argc, char** argv) {
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;

    // Parse and print only on the root, then give every rank the same settings
    // and exit status, including help and malformed command lines.
    if (rank == 0) {
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
                printUsage(argv[0]);
                parseStatus = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseStatus = 1;
                break;
            }
        }
        if (parseStatus == 0 && numBodies < 0) {
            printf("Number of bodies must be non-negative\n");
            parseStatus = 1;
        }
    }

    int settings[] = {numBodies, numSteps, validate, printResults, parseStatus};
    MPI_Bcast(settings, 5, MPI_INT, 0, MPI_COMM_WORLD);
    numBodies = settings[0];
    numSteps = settings[1];
    validate = settings[2] != 0;
    printResults = settings[3] != 0;
    if (settings[4] != 0) {
        return settings[4] == 2 ? 0 : 1;
    }

    // Empty ranks do not participate in the simulation collectives. This also
    // handles more MPI ranks than bodies, and the zero-body case.
    const int activeRanks = std::min(worldSize, std::max(numBodies, 1));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED, rank, &comm);
    int exitCode = 0;
    if (comm == MPI_COMM_NULL) {
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return exitCode;
    }

    std::vector<int> counts(activeRanks), offsets(activeRanks);
    int offset = 0;
    for (int p = 0; p < activeRanks; ++p) {
        counts[p] = numBodies / activeRanks + (p < numBodies % activeRanks ? 1 : 0);
        offsets[p] = offset;
        offset += counts[p];
    }

    MPI_Datatype positionType, bodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &positionType);
    MPI_Type_commit(&positionType);
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);

    std::vector<Body> bodies;
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        bodies.resize(numBodies);
        randomizeBodies(bodies);
    }
    std::vector<Body> localBodies(counts[rank]);
    std::vector<Vec3> positions(numBodies);
    MPI_Scatterv(bodies.data(), counts.data(), offsets.data(), bodyType,
                 localBodies.data(), counts[rank], bodyType, 0, comm);
    // Keep only local velocities and a global snapshot of positions in the
    // simulation. The full state is gathered only if results are requested.
    std::vector<Body>().swap(bodies);
    if (numSteps > 0) {
        exchangePositions(localBodies, positions, offsets[rank], counts, offsets,
                          positionType, comm);
    }

    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        computeForces(localBodies, positions);
        integrateBodies(localBodies);
        // The final positions need no further replication: all forces have
        // been computed, and optional output uses the gathered complete state.
        if (step + 1 < numSteps) {
            exchangePositions(localBodies, positions, offsets[rank], counts, offsets,
                              positionType, comm);
        }
    }
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Simulation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
    }

    if (printResults || validate) {
        if (rank == 0) {
            bodies.resize(numBodies);
        }
        MPI_Gatherv(localBodies.data(), counts[rank], bodyType, bodies.data(),
                    counts.data(), offsets.data(), bodyType, 0, comm);
    }

    // Print results for external validation
    if (rank == 0 && printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
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
    if (rank == 0 && validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            // Retain the original diagnostic summation order on the root.
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Type_free(&bodyType);
    MPI_Type_free(&positionType);
    MPI_Comm_free(&comm);
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return exitCode;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int exitCode = 1;
    try {
        exitCode = runSimulation(argc, argv);
    } catch (const std::exception& error) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        // A failed allocation must not leave other ranks blocked in a collective.
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return exitCode;
}
