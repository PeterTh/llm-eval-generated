#include <algorithm>
#include <array>
#include <exception>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
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

// Vectorize across targets, retaining the original source-body summation order.
// No reassociation or approximate reciprocal/square root is needed.
void computeForces(std::span<Body> bodies, const std::vector<Vec3>& positions) {
#if defined(__AVX512F__)
    constexpr size_t width = 8;
#else
    constexpr size_t width = 4;
#endif
    const size_t n = positions.size();
    size_t i = 0;
    for (; i + width <= bodies.size(); i += width) {
        double px[width], py[width], pz[width];
        double fx[width] = {}, fy[width] = {}, fz[width] = {};
        for (size_t k = 0; k < width; ++k) {
            px[k] = bodies[i + k].pos.x;
            py[k] = bodies[i + k].pos.y;
            pz[k] = bodies[i + k].pos.z;
        }
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < width; ++k) {
                const double dx = positions[j].x - px[k];
                const double dy = positions[j].y - py[k];
                const double dz = positions[j].z - pz[k];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx[k] += dx * invDist3;
                fy[k] += dy * invDist3;
                fz[k] += dz * invDist3;
            }
        }
        for (size_t k = 0; k < width; ++k) {
            // Round the velocity increments before adding, as in the original
            // force kernel; prevent contraction with the velocity addition.
            const volatile double dvx = DT * fx[k];
            const volatile double dvy = DT * fy[k];
            const volatile double dvz = DT * fz[k];
            bodies[i + k].vel.x += dvx;
            bodies[i + k].vel.y += dvy;
            bodies[i + k].vel.z += dvz;
        }
    }
    if (i < bodies.size()) {
        // Pad the final tile so every real body uses the same vectorized
        // arithmetic, independent of the rank count or partition boundary.
        std::array<Body, width> tail{};
        std::copy(bodies.begin() + i, bodies.end(), tail.begin());
        computeForces(tail, positions);
        for (size_t k = 0; i + k < bodies.size(); ++k) {
            bodies[i + k].vel = tail[k].vel;
        }
    }
}

void integrateBodies(std::vector<Body>& bodies, Vec3* positions) {
    for (size_t i = 0; i < bodies.size(); ++i) {
        auto& body = bodies[i];
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
        positions[i] = body.pos;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies,
                          const std::vector<Vec3>& positions, int rank, int ranks) {
    double energy = 0.0;
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }
    // Cyclic ownership balances the triangular potential-energy calculation.
    for (size_t i = rank; i < positions.size(); i += ranks) {
        for (size_t j = i + 1; j < positions.size(); ++j) {
            const double dx = positions[j].x - positions[i].x;
            const double dy = positions[j].y - positions[i].y;
            const double dz = positions[j].z - positions[i].z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    double total = 0.0;
    MPI_Reduce(&energy, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return total;
}

// Validate that simulation produces finite, reasonable values
int validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            return 1;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            return 2;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            return 3;
        }
    }
    return 0;
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

int runSimulation(int argc, char** argv, int rank, int ranks) {
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
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (numBodies < 0 || numSteps < 0) {
        if (rank == 0) fprintf(stderr, "Body and step counts must be nonnegative\n");
        return 1;
    }
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Counts are expressed in bodies/positions rather than doubles, avoiding
    // multiplication overflow in MPI's int counts and displacements.
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = (numBodies / ranks) * r + std::min(r, numBodies % ranks);
    }
    static_assert(sizeof(Vec3) == 3 * sizeof(double));
    static_assert(sizeof(Body) == 2 * sizeof(Vec3));
    MPI_Datatype positionType, bodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &positionType);
    MPI_Type_commit(&positionType);
    MPI_Type_contiguous(2, positionType, &bodyType);
    MPI_Type_commit(&bodyType);

    std::vector<Body> bodies;
    std::vector<Body> localBodies(counts[rank]);
    std::vector<Vec3> positions(numBodies);
    if (rank == 0) {
        bodies.resize(numBodies);
        randomizeBodies(bodies);
        for (int i = 0; i < numBodies; ++i) positions[i] = bodies[i].pos;
    }
    MPI_Scatterv(bodies.data(), counts.data(), offsets.data(), bodyType,
                 localBodies.data(), counts[rank], bodyType, 0, MPI_COMM_WORLD);
    MPI_Bcast(positions.data(), numBodies, positionType, 0, MPI_COMM_WORLD);
    // Complete body states are only needed on root when printing results.
    std::vector<Body>().swap(bodies);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        computeForces(localBodies, positions);
        if (!localBodies.empty()) {
            integrateBodies(localBodies, positions.data() + offsets[rank]);
        }
        if (step + 1 < numSteps) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, positions.data(),
                           counts.data(), offsets.data(), positionType, MPI_COMM_WORLD);
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %lld ms\n", static_cast<long long>(duration * 1000));

    if (printResults) {
        if (rank == 0) bodies.resize(numBodies);
        MPI_Gatherv(localBodies.data(), counts[rank], bodyType, bodies.data(),
                    counts.data(), offsets.data(), bodyType, 0, MPI_COMM_WORLD);
        if (rank == 0) {
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
    }

    int status = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        const int localError = validateSimulation(localBodies);
        int error = 0;
        MPI_Allreduce(&localError, &error, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (error == 0) {
            // The last step does not exchange positions unless energy is needed.
            if (numSteps > 0) {
                MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, positions.data(),
                               counts.data(), offsets.data(), positionType, MPI_COMM_WORLD);
            }
            const double finalEnergy = computeTotalEnergy(localBodies, positions, rank, ranks);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                const char* reasons[] = {"", "found NaN or Inf value in body state",
                                         "body position exceeds reasonable bounds",
                                         "body velocity exceeds reasonable bounds"};
                printf("Validation failed: %s\n", reasons[error]);
                printf("Validation: FAILED\n");
            }
            status = 1;
        }
    }
    MPI_Type_free(&bodyType);
    MPI_Type_free(&positionType);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        status = runSimulation(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
