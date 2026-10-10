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

// MPI communicates doubles rather than bytes, including on heterogeneous hosts.
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
    double x[Width], y[Width], z[Width];
    double Fx[Width] = {}, Fy[Width] = {}, Fz[Width] = {};
    for (size_t k = 0; k < Width; ++k) {
        x[k] = bodies[k].pos.x;
        y[k] = bodies[k].pos.y;
        z[k] = bodies[k].pos.z;
    }

    // SIMD lanes represent independent target bodies. Every lane still visits
    // sources in the original order, avoiding floating-point reduction changes.
    for (const auto& pos : positions) {
#if defined(__GNUC__)
        // Keep the inner loop available for SIMD across targets rather than
        // unrolling it into redundant calculations for individual components.
#pragma GCC unroll 1
#endif
        for (size_t k = 0; k < Width; ++k) {
            const double dx = pos.x - x[k];
            const double dy = pos.y - y[k];
            const double dz = pos.z - z[k];
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx[k] += dx * invDist3;
            Fy[k] += dy * invDist3;
            Fz[k] += dz * invDist3;
        }
    }

    for (size_t k = 0; k < Width; ++k) {
        // Round the increments before adding, matching the original kernel.
        // These stores are outside the quadratic force loop.
        const volatile double vx = DT * Fx[k];
        const volatile double vy = DT * Fy[k];
        const volatile double vz = DT * Fz[k];
        bodies[k].vel.x += vx;
        bodies[k].vel.y += vy;
        bodies[k].vel.z += vz;
    }
}

void computeForces(std::vector<Body>& bodies, const std::vector<Vec3>& positions) {
    constexpr size_t Width = 4;
    size_t i = 0;
    for (; i + Width <= bodies.size(); i += Width) {
        computeForceBlock<Width>(bodies.data() + i, positions);
    }
    if (i < bodies.size()) {
        // Use the same arithmetic for partial blocks on every rank.
        Body tail[Width] = {};
        const size_t remaining = bodies.size() - i;
        for (size_t k = 0; k < remaining; ++k) tail[k] = bodies[i + k];
        computeForceBlock<Width>(tail, positions);
        for (size_t k = 0; k < remaining; ++k) bodies[i + k] = tail[k];
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

int runSimulation(int argc, char** argv, int rank, int numRanks) {
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

    if (numBodies < 0) {
        if (rank == 0) printf("Number of bodies must be nonnegative\n");
        return 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Contiguous, balanced ownership also preserves the original output order.
    // Counts are in bodies, so multiplying large counts by 3 or 6 cannot overflow.
    std::vector<int> counts(numRanks), offsets(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        counts[r] = numBodies / numRanks + (r < numBodies % numRanks);
        offsets[r] = r * (numBodies / numRanks) + std::min(r, numBodies % numRanks);
    }

    MPI_Datatype positionType, bodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &positionType);
    MPI_Type_commit(&positionType);
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);

    std::vector<Body> bodies;
    if (rank == 0) {
        bodies.resize(numBodies);
        randomizeBodies(bodies);
    }
    std::vector<Body> localBodies(counts[rank]);
    MPI_Scatterv(bodies.data(), counts.data(), offsets.data(), bodyType,
                 localBodies.data(), counts[rank], bodyType, 0, MPI_COMM_WORLD);
    if (!printResults && !validate) std::vector<Body>().swap(bodies);

    // Only positions are replicated and exchanged; velocities remain local.
    std::vector<Vec3> positions(numBodies);
    for (size_t i = 0; i < localBodies.size(); ++i) {
        positions[static_cast<size_t>(offsets[rank]) + i] = localBodies[i].pos;
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, positions.data(), counts.data(),
                   offsets.data(), positionType, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        computeForces(localBodies, positions);
        integrateBodies(localBodies);
        if (step + 1 < numSteps) {
            for (size_t i = 0; i < localBodies.size(); ++i) {
                positions[static_cast<size_t>(offsets[rank]) + i] = localBodies[i].pos;
            }
            MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, positions.data(), counts.data(),
                           offsets.data(), positionType, MPI_COMM_WORLD);
        }
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %lld ms\n", static_cast<long long>(duration * 1000.0));

    // Gather complete states only when needed for the original diagnostics.
    if (printResults || validate) {
        MPI_Gatherv(localBodies.data(), counts[rank], bodyType, bodies.data(),
                    counts.data(), offsets.data(), bodyType, 0, MPI_COMM_WORLD);
    }
    MPI_Type_free(&positionType);
    MPI_Type_free(&bodyType);

    int exitCode = 0;
    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
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

        // Preserve the original order of the diagnostic energy summation.
        // Diagnostics run outside the timed, distributed simulation.
        if (validate) {
            printf("Validating simulation results...\n");

            if (validateSimulation(bodies)) {
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return exitCode;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    try {
        const int exitCode = runSimulation(argc, argv, rank, numRanks);
        MPI_Finalize();
        return exitCode;
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
