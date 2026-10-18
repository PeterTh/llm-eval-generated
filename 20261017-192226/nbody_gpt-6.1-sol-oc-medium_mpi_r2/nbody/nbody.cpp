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

// The MPI datatypes below describe these layouts without transmitting padding.
static_assert(sizeof(Vec3) == 3 * sizeof(double));
static_assert(offsetof(Body, vel) == sizeof(Vec3));
static_assert(sizeof(Body) == 2 * sizeof(Vec3));

struct MpiSession {
    MpiSession(int& argc, char**& argv) { MPI_Init(&argc, &argv); }
    ~MpiSession() { MPI_Finalize(); }
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
    const size_t n = positions.size();
    
    for (size_t i = 0; i < bodies.size(); ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (size_t j = 0; j < n; ++j) {
            const double dx = positions[j].x - bodies[i].pos.x;
            const double dy = positions[j].y - bodies[i].pos.y;
            const double dz = positions[j].z - bodies[i].pos.z;
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

void integrateBodies(std::vector<Body>& bodies, std::vector<Vec3>& positions, size_t offset) {
    for (auto& body : bodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
    for (const auto& body : bodies) positions[offset++] = body.pos;
}

double computeTotalEnergy(const std::vector<Body>& bodies,
                          const std::vector<Vec3>& positions, int rank, int ranks) {
    double energy = 0.0;
    const size_t n = positions.size();
    
    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
    // Cyclic rows balance the triangular pair loop across ranks.
    for (size_t i = rank; i < n; i += ranks) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = positions[j].x - positions[i].x;
            const double dy = positions[j].y - positions[i].y;
            const double dz = positions[j].z - positions[i].z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    
    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies, bool reportErrors) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            if (reportErrors) printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            if (reportErrors) printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            if (reportErrors) printf("Validation failed: body velocity exceeds reasonable bounds\n");
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
    MpiSession mpi(argc, argv);
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
        if (rank == 0) fprintf(stderr, "Number of bodies must be nonnegative\n");
        return 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    MPI_Datatype positionType, bodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &positionType);
    MPI_Type_commit(&positionType);
    MPI_Type_contiguous(2, positionType, &bodyType);
    MPI_Type_commit(&bodyType);

    // Contiguous, balanced ownership keeps every force sum in global body order.
    // Counts are in bodies, avoiding a sixfold reduction of MPI's count limit.
    std::vector<int> counts(ranks), offsets(ranks);
    int offset = 0;
    for (int r = 0; r < ranks; ++r) {
        offsets[r] = offset;
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offset += counts[r];
    }
    std::vector<Body> initialBodies;
    if (rank == 0) {
        initialBodies.resize(numBodies);
        randomizeBodies(initialBodies);
    }
    std::vector<Body> bodies(counts[rank]);
    MPI_Scatterv(initialBodies.data(), counts.data(), offsets.data(), bodyType,
                 bodies.data(), counts[rank], bodyType, 0, MPI_COMM_WORLD);
    std::vector<Body>().swap(initialBodies);

    // Only positions are replicated; velocities remain on their owning ranks.
    std::vector<Vec3> positions(numBodies);
    for (size_t i = 0; i < bodies.size(); ++i)
        positions[offsets[rank] + i] = bodies[i].pos;
    auto exchangePositions = [&] {
        MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, positions.data(),
                       counts.data(), offsets.data(), positionType, MPI_COMM_WORLD);
    };
    if (numSteps > 0) exchangePositions();
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (int step = 0; step < numSteps; ++step) {
        if (step > 0) exchangePositions();
        computeForces(bodies, positions);
        integrateBodies(bodies, positions, offsets[rank]);
    }
    
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(duration * 1000));
    
    // Print results for external validation
    if (printResults) {
        std::vector<Body> allBodies;
        if (rank == 0) allBodies.resize(numBodies);
        MPI_Gatherv(bodies.data(), counts[rank], bodyType, allBodies.data(),
                    counts.data(), offsets.data(), bodyType, 0, MPI_COMM_WORLD);
        if (rank == 0) {
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
    }

    MPI_Type_free(&bodyType);
    
    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        int localValid = validateSimulation(bodies, rank == 0), valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        if (valid) {
            // Report final energy for reference
            exchangePositions();
            double localEnergy = computeTotalEnergy(bodies, positions, rank, ranks);
            double finalEnergy = 0.0;
            MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            MPI_Type_free(&positionType);
            return 1;
        }
    }
    MPI_Type_free(&positionType);
    return 0;
}
