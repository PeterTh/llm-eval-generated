#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// SIMD lanes are independent target bodies. Every lane visits sources in the
// original order, so vectorization needs neither approximate math nor a reduction.
void computeForces(std::vector<Body>& bodies, const std::vector<Vec3>& positions) {
    constexpr size_t lanes = 4;
    for (size_t i = 0; i < bodies.size(); i += lanes) {
        const size_t count = std::min(lanes, bodies.size() - i);
        double px[lanes] = {}, py[lanes] = {}, pz[lanes] = {};
        double fx[lanes] = {}, fy[lanes] = {}, fz[lanes] = {};
        for (size_t k = 0; k < count; ++k) {
            px[k] = bodies[i + k].pos.x;
            py[k] = bodies[i + k].pos.y;
            pz[k] = bodies[i + k].pos.z;
        }
        for (const auto& source : positions) {
            for (size_t k = 0; k < lanes; ++k) {
                const double dx = source.x - px[k];
                const double dy = source.y - py[k];
                const double dz = source.z - pz[k];
                const double distSqr = std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx[k] = std::fma(dx, invDist3, fx[k]);
                fy[k] = std::fma(dy, invDist3, fy[k]);
                fz[k] = std::fma(dz, invDist3, fz[k]);
            }
        }
        for (size_t k = 0; k < count; ++k) {
            // Round the velocity increment separately, as in the original
            // kernel; partial batches must use the same arithmetic as full ones.
            volatile double vx = DT * fx[k];
            volatile double vy = DT * fy[k];
            volatile double vz = DT * fz[k];
            bodies[i + k].vel.x += vx;
            bodies[i + k].vel.y += vy;
            bodies[i + k].vel.z += vz;
        }
    }

}

void integrateBodies(std::vector<Body>& bodies, std::vector<Vec3>& positions,
                     size_t offset) {
    for (size_t i = 0; i < bodies.size(); ++i) {
        auto& body = bodies[i];
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
        positions[offset + i] = body.pos;
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

// Finalization also covers help and argument-error exits on every rank.
struct MpiSession {
    MpiSession(int& argc, char**& argv) { MPI_Init(&argc, &argv); }
    ~MpiSession() { MPI_Finalize(); }
};

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

    // Exclude ranks without bodies from the per-step collectives.
    const int activeRanks = std::min(ranks, std::max(numBodies, 1));
    MPI_Comm simulation;
    MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED,
                   rank, &simulation);
    int result = 0;
    if (rank >= activeRanks) {
        MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return result;
    }
    std::vector<int> counts(activeRanks), offsets(activeRanks);
    for (int r = 0, offset = 0; r < activeRanks; ++r) {
        counts[r] = numBodies / activeRanks + (r < numBodies % activeRanks);
        offsets[r] = offset;
        offset += counts[r];
    }

    // Datatypes count bodies/positions, avoiding overflow from multiplying MPI
    // counts by the number of coordinates. These structs contain only doubles.
    static_assert(sizeof(Vec3) == 3 * sizeof(double));
    static_assert(sizeof(Body) == 6 * sizeof(double));
    MPI_Datatype bodyType, positionType;
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);
    MPI_Type_contiguous(3, MPI_DOUBLE, &positionType);
    MPI_Type_commit(&positionType);

    std::vector<Body> bodies;
    if (rank == 0) {
        bodies.resize(numBodies);
        randomizeBodies(bodies);
    }
    std::vector<Body> localBodies(counts[rank]);
    MPI_Scatterv(bodies.data(), counts.data(), offsets.data(), bodyType,
                 localBodies.data(), counts[rank], bodyType, 0, simulation);
    std::vector<Vec3> positions(numBodies);
    for (size_t i = 0; i < localBodies.size(); ++i)
        positions[offsets[rank] + i] = localBodies[i].pos;
    if (rank == 0) std::vector<Body>().swap(bodies);
    MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, positions.data(),
                   counts.data(), offsets.data(), positionType, simulation);

    MPI_Barrier(simulation);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        computeForces(localBodies, positions);
        integrateBodies(localBodies, positions, offsets[rank]);
        // Velocities stay local. The final step needs no position exchange.
        if (step + 1 < numSteps) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, positions.data(),
                           counts.data(), offsets.data(), positionType, simulation);
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, simulation);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(duration * 1000.0));

    // Gather the full state only when requested, preserving global body order.
    if (printResults || validate) {
        if (rank == 0) bodies.resize(numBodies);
        MPI_Gatherv(localBodies.data(), counts[rank], bodyType, bodies.data(),
                    counts.data(), offsets.data(), bodyType, 0, simulation);
    }
    MPI_Type_free(&positionType);
    MPI_Type_free(&bodyType);
    MPI_Comm_free(&simulation);

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
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            result = 1;
        }
    }

    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return result;
}
