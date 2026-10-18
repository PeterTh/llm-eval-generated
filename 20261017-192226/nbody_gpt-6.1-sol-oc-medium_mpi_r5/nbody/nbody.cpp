#include <algorithm>
#include <cmath>
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

void computeForces(std::vector<Body>& bodies, const std::vector<Vec3>& positions) {
    // Vectorize across independent target bodies, not across the force
    // reduction: every target still sums sources in their original order.
    constexpr size_t tileSize = 8;
    for (size_t i = 0; i < bodies.size(); i += tileSize) {
        const size_t count = std::min(tileSize, bodies.size() - i);
        alignas(64) double x[tileSize], y[tileSize], z[tileSize];
        alignas(64) double Fx[tileSize] = {}, Fy[tileSize] = {}, Fz[tileSize] = {};
        for (size_t k = 0; k < count; ++k) {
            x[k] = bodies[i + k].pos.x;
            y[k] = bodies[i + k].pos.y;
            z[k] = bodies[i + k].pos.z;
        }
        for (const auto& source : positions) {
            for (size_t k = 0; k < count; ++k) {
                const double dx = source.x - x[k];
                const double dy = source.y - y[k];
                const double dz = source.z - z[k];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx[k] += dx * invDist3;
                Fy[k] += dy * invDist3;
                Fz[k] += dz * invDist3;
            }
        }
        for (size_t k = 0; k < count; ++k) {
            // Round the timestep multiplication before adding velocity, as
            // in the original kernel (rather than contracting this to FMA).
            const volatile double dvx = DT * Fx[k];
            const volatile double dvy = DT * Fy[k];
            const volatile double dvz = DT * Fz[k];
            bodies[i + k].vel.x += dvx;
            bodies[i + k].vel.y += dvy;
            bodies[i + k].vel.z += dvz;
        }
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
    int rank, numRanks;
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
            if (rank == 0) printUsage(argv[0]);
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
        if (rank == 0) fprintf(stderr, "Body and step counts must be nonnegative\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Contiguous, balanced ownership also handles ranks with no bodies.
    std::vector<int> counts(numRanks), offsets(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        counts[r] = numBodies / numRanks + (r < numBodies % numRanks);
        offsets[r] = r * (numBodies / numRanks) + std::min(r, numBodies % numRanks);
    }
    MPI_Datatype positionType, bodyType;
    static_assert(sizeof(Vec3) == 3 * sizeof(double));
    static_assert(sizeof(Body) == 6 * sizeof(double));
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
    // Only positions are replicated; velocities remain on their owning rank.
    std::vector<Vec3> positions(numBodies);
    for (int i = 0; i < counts[rank]; ++i)
        positions[offsets[rank] + i] = localBodies[i].pos;
    MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, positions.data(),
                   counts.data(), offsets.data(), positionType, MPI_COMM_WORLD);
    std::vector<Body>().swap(bodies);
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(localBodies, positions);
        integrateBodies(localBodies);
        // The final positions are not needed by another force evaluation.
        if (step + 1 < numSteps) {
            for (int i = 0; i < counts[rank]; ++i)
                positions[offsets[rank] + i] = localBodies[i].pos;
            MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, positions.data(),
                           counts.data(), offsets.data(), positionType, MPI_COMM_WORLD);
        }
    }
    
    const double elapsed = MPI_Wtime() - start;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(duration * 1000));

    if (printResults || validate) {
        if (rank == 0) bodies.resize(numBodies);
        MPI_Gatherv(localBodies.data(), counts[rank], bodyType, bodies.data(),
                    counts.data(), offsets.data(), bodyType, 0, MPI_COMM_WORLD);
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
    int status = 0;
    if (rank == 0 && validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }
    
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Type_free(&bodyType);
    MPI_Type_free(&positionType);
    MPI_Finalize();
    return status;
}
