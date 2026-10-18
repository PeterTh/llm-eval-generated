#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <type_traits>

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
    // Vectorize across independent targets, retaining the original source
    // summation order for every body. Each source is reused across a tile.
    constexpr size_t tileSize = 32;
    for (size_t base = 0; base < bodies.size(); base += tileSize) {
        const size_t count = std::min(tileSize, bodies.size() - base);
        alignas(64) double x[tileSize], y[tileSize], z[tileSize];
        alignas(64) double fx[tileSize] = {}, fy[tileSize] = {}, fz[tileSize] = {};
        for (size_t k = 0; k < count; ++k) {
            x[k] = bodies[base + k].pos.x;
            y[k] = bodies[base + k].pos.y;
            z[k] = bodies[base + k].pos.z;
        }
        for (const auto& source : positions) {
            for (size_t k = 0; k < count; ++k) {
                const double dx = source.x - x[k];
                const double dy = source.y - y[k];
                const double dz = source.z - z[k];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx[k] += dx * invDist3;
                fy[k] += dy * invDist3;
                fz[k] += dz * invDist3;
            }
        }
        for (size_t k = 0; k < count; ++k) {
            bodies[base + k].vel.x += DT * fx[k];
            bodies[base + k].vel.y += DT * fy[k];
            bodies[base + k].vel.z += DT * fz[k];
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
    
    if (numBodies < 0) {
        if (rank == 0) fprintf(stderr, "Number of bodies must be nonnegative\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Balanced contiguous ownership also handles ranks with no bodies.
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = r == 0 ? 0 : offsets[r - 1] + counts[r - 1];
    }

    static_assert(std::is_standard_layout_v<Vec3> && sizeof(Vec3) == 3 * sizeof(double));
    static_assert(std::is_standard_layout_v<Body> && sizeof(Body) == 6 * sizeof(double));
    MPI_Datatype positionType, bodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &positionType);
    MPI_Type_commit(&positionType);
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);

    std::vector<Body> bodies(counts[rank]);
    std::vector<Body> allBodies;
    if (rank == 0) {
        allBodies.resize(numBodies);
        randomizeBodies(allBodies);
    }
    MPI_Scatterv(allBodies.data(), counts.data(), offsets.data(), bodyType,
                 bodies.data(), counts[rank], bodyType, 0, MPI_COMM_WORLD);
    // Full states are needed again only for optional output/validation.
    std::vector<Body>().swap(allBodies);
    std::vector<Vec3> positions(numBodies);
    for (size_t i = 0; i < bodies.size(); ++i)
        positions[offsets[rank] + i] = bodies[i].pos;
    MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, positions.data(),
                   counts.data(), offsets.data(), positionType, MPI_COMM_WORLD);
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, positions);
        integrateBodies(bodies);
        if (step + 1 < numSteps) {
            for (size_t i = 0; i < bodies.size(); ++i)
                positions[offsets[rank] + i] = bodies[i].pos;
            MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, positions.data(),
                           counts.data(), offsets.data(), positionType, MPI_COMM_WORLD);
        }
    }
    
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(duration * 1000));

    if (printResults || validate) {
        if (rank == 0) allBodies.resize(numBodies);
        MPI_Gatherv(bodies.data(), counts[rank], bodyType, allBodies.data(),
                    counts.data(), offsets.data(), bodyType, 0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        // Serialize body positions and velocities for hashing
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
    
    // Validation: check that simulation produces finite, reasonable values
    int result = 0;
    if (rank == 0 && validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(allBodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(allBodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            result = 1;
        }
    }
    
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Type_free(&positionType);
    MPI_Type_free(&bodyType);
    MPI_Finalize();
    return result;
}
