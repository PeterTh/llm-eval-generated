#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr size_t FORCE_BLOCK_SIZE = 8;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// The MPI datatypes below describe doubles, rather than transmitting raw bytes.
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

void exchangePositions(std::vector<Vec3>& positions, const std::vector<int>& counts,
                       const std::vector<int>& offsets, MPI_Datatype positionType, MPI_Comm comm) {
    // Use MPI's fixed-count algorithms when the partition is uniform.
    if (positions.size() % counts.size() == 0) {
        MPI_Allgather(MPI_IN_PLACE, 0, positionType, positions.data(), counts[0], positionType, comm);
    } else {
        MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, positions.data(), counts.data(),
                       offsets.data(), positionType, comm);
    }
}

void computeForces(std::vector<Body>& bodies, const std::vector<Vec3>& positions) {
    // SIMD lanes represent different target bodies. Each lane still accumulates
    // sources in their original order, without a reordered floating-point sum.
    constexpr size_t blockSize = FORCE_BLOCK_SIZE;
    size_t i = 0;
    for (; i + blockSize <= bodies.size(); i += blockSize) {
        alignas(64) double x[blockSize], y[blockSize], z[blockSize];
        alignas(64) double fx[blockSize] = {}, fy[blockSize] = {}, fz[blockSize] = {};
        for (size_t lane = 0; lane < blockSize; ++lane) {
            x[lane] = bodies[i + lane].pos.x;
            y[lane] = bodies[i + lane].pos.y;
            z[lane] = bodies[i + lane].pos.z;
        }
        for (const auto& source : positions) {
            for (size_t lane = 0; lane < blockSize; ++lane) {
                const double dx = source.x - x[lane];
                const double dy = source.y - y[lane];
                const double dz = source.z - z[lane];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx[lane] += dx * invDist3;
                fy[lane] += dy * invDist3;
                fz[lane] += dz * invDist3;
            }
        }
        for (size_t lane = 0; lane < blockSize; ++lane) {
            bodies[i + lane].vel.x += DT * fx[lane];
            bodies[i + lane].vel.y += DT * fy[lane];
            bodies[i + lane].vel.z += DT * fz[lane];
        }
    }

    for (; i < bodies.size(); ++i) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        for (const auto& source : positions) {
            const double dx = source.x - bodies[i].pos.x;
            const double dy = source.y - bodies[i].pos.y;
            const double dz = source.z - bodies[i].pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            // Keep the fused accumulation used by the SIMD lanes instead of
            // allowing the compiler to vectorize a reduction over sources.
            fx = std::fma(dx, invDist3, fx);
            fy = std::fma(dy, invDist3, fy);
            fz = std::fma(dz, invDist3, fz);
        }
        const volatile double dvx = DT * fx;
        const volatile double dvy = DT * fy;
        const volatile double dvz = DT * fz;
        bodies[i].vel.x += dvx;
        bodies[i].vel.y += dvy;
        bodies[i].vel.z += dvz;
    }
}

void integrateBodies(std::vector<Body>& bodies, std::vector<Vec3>& positions, size_t first) {
    for (size_t i = 0; i < bodies.size(); ++i) {
        auto& body = bodies[i];
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
        positions[first + i] = body.pos;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies, const std::vector<Vec3>& positions,
                          int rank, int ranks, MPI_Comm comm) {
    double energy = 0.0;
    const size_t n = positions.size();
    
    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
    // Cyclic ownership balances the triangular pair loop independently of the
    // contiguous body ownership used for the simulation.
    for (size_t i = static_cast<size_t>(rank); i < n; i += static_cast<size_t>(ranks)) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = positions[j].x - positions[i].x;
            const double dy = positions[j].y - positions[i].y;
            const double dz = positions[j].z - positions[i].z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    
    double totalEnergy = 0.0;
    MPI_Reduce(&energy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    return totalEnergy;
}

// Validate that simulation produces finite, reasonable values
long long validateSimulation(const std::vector<Body>& bodies, size_t first) {
    for (size_t i = 0; i < bodies.size(); ++i) {
        const auto& body = bodies[i];
        // Encoding the body index preserves the original first-failure message
        // when validation results from different ranks are combined with MIN.
        const long long failure = 3 * static_cast<long long>(first + i);
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            return failure;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            return failure + 1;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            return failure + 2;
        }
    }
    return std::numeric_limits<long long>::max();
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

int runSimulation(int argc, char** argv, int worldRank, int worldSize) {
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
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (numBodies < 0) {
        if (worldRank == 0) printf("Number of bodies must be nonnegative\n");
        return 1;
    }
    if (worldRank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Ranks without bodies do not participate in per-step communication. Keep
    // rank zero active even for an empty simulation so it can produce output.
    const int ranks = std::max(1, std::min(numBodies, worldSize));
    MPI_Comm comm = MPI_COMM_WORLD;
    if (ranks < worldSize) {
        MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED, worldRank, &comm);
        if (comm == MPI_COMM_NULL) return 0;
    }
    const int rank = worldRank;
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks ? 1 : 0);
        offsets[r] = r * (numBodies / ranks) + std::min(r, numBodies % ranks);
    }
    const size_t first = static_cast<size_t>(offsets[rank]);
    std::vector<Body> bodies(counts[rank]);
    std::vector<Vec3> positions(numBodies);

    MPI_Datatype positionType, bodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &positionType);
    MPI_Type_commit(&positionType);
    MPI_Type_contiguous(2, positionType, &bodyType);
    MPI_Type_commit(&bodyType);
    {
        // Generate the original random stream once; scattering contiguous
        // segments keeps initialization independent of the number of ranks.
        std::vector<Body> initialBodies;
        if (rank == 0) {
            initialBodies.resize(numBodies);
            randomizeBodies(initialBodies);
        }
        MPI_Scatterv(initialBodies.data(), counts.data(), offsets.data(), bodyType,
                     bodies.data(), counts[rank], bodyType, 0, comm);
    }
    for (size_t i = 0; i < bodies.size(); ++i) positions[first + i] = bodies[i].pos;
    exchangePositions(positions, counts, offsets, positionType, comm);

    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, positions);
        integrateBodies(bodies, positions, first);
        // The final positions are exchanged only if validation needs them.
        if (step + 1 < numSteps) {
            exchangePositions(positions, counts, offsets, positionType, comm);
        }
    }
    
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) printf("Simulation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
    
    // Print results for external validation
    if (printResults) {
        // The packed MPI datatype and rank order reproduce the original body
        // serialization, including its hash and sample indexing.
        std::vector<double> bodyData(rank == 0 ? static_cast<size_t>(numBodies) * 6 : 0);
        MPI_Gatherv(bodies.data(), counts[rank], bodyType, bodyData.data(), counts.data(),
                    offsets.data(), bodyType, 0, comm);
        if (rank == 0) print_results(bodyData, "Bodies");
    }

    int result = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        const long long localFailure = validateSimulation(bodies, first);
        long long failure;
        MPI_Allreduce(&localFailure, &failure, 1, MPI_LONG_LONG_INT, MPI_MIN, comm);
        if (failure == std::numeric_limits<long long>::max()) {
            if (numSteps > 0) {
                exchangePositions(positions, counts, offsets, positionType, comm);
            }
            const double finalEnergy = computeTotalEnergy(bodies, positions, rank, ranks, comm);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                const char* messages[] = {
                    "Validation failed: found NaN or Inf value in body state",
                    "Validation failed: body position exceeds reasonable bounds",
                    "Validation failed: body velocity exceeds reasonable bounds"
                };
                printf("%s\n", messages[failure % 3]);
                printf("Validation: FAILED\n");
            }
            result = 1;
        }
    }

    MPI_Type_free(&bodyType);
    MPI_Type_free(&positionType);
    if (comm != MPI_COMM_WORLD) MPI_Comm_free(&comm);
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = runSimulation(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        // A failed allocation on one rank must not leave its peers blocked in
        // a collective operation.
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
