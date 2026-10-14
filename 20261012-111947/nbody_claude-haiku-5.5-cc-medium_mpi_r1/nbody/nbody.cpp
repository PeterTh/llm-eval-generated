#include <mpi.h>

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

// Body is exchanged as raw doubles in the final gather
static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be packed as six doubles");

// Contiguous range of bodies [begin, begin + count) owned by one rank
struct BlockRange {
    int begin;
    int count;
};

BlockRange blockRange(const int numBodies, const int rank, const int size) {
    const long long n = numBodies;
    const int begin = static_cast<int>(n * rank / size);
    const int end = static_cast<int>(n * (rank + 1) / size);
    return {begin, end - begin};
}

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

// Updates velocities of bodies [begin, end) only. Each body's force sum runs over
// all j in the same order as the serial code, so results are bit-identical.
void computeForces(std::vector<Body>& bodies, size_t begin, size_t end) {
    const size_t n = bodies.size();

    for (size_t i = begin; i < end; ++i) {
        const Vec3 pi = bodies[i].pos;
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (size_t j = 0; j < n; ++j) {
            const double dx = bodies[j].pos.x - pi.x;
            const double dy = bodies[j].pos.y - pi.y;
            const double dz = bodies[j].pos.z - pi.z;
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

// Advances positions of bodies [begin, end) only
void integrateBodies(std::vector<Body>& bodies, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
        Body& body = bodies[i];
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
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool isRoot = rank == 0;

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank, so all ranks exit together)
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
            if (isRoot) printUsage(argv[0]);
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (isRoot) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Every rank holds the full state so force sums can read all positions
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Block decomposition of bodies across ranks
    const BlockRange local = blockRange(numBodies, rank, size);
    const size_t begin = static_cast<size_t>(local.begin);
    const size_t end = begin + static_cast<size_t>(local.count);

    // Per-rank counts and displacements (in bodies) for the position exchange
    std::vector<int> counts(size), displs(size);
    std::vector<int> counts6(size), displs6(size);
    for (int r = 0; r < size; ++r) {
        const BlockRange range = blockRange(numBodies, r, size);
        counts[r] = range.count;
        displs[r] = range.begin;
        counts6[r] = 6 * range.count;
        displs6[r] = 6 * range.begin;
    }

    // Datatype selecting only the pos field of each Body, with extent sizeof(Body),
    // so one Allgatherv moves positions in place without packing velocities
    MPI_Datatype vec3Type, posType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_create_resized(vec3Type, 0, static_cast<MPI_Aint>(sizeof(Body)), &posType);
    MPI_Type_commit(&posType);
    MPI_Type_free(&vec3Type);

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, begin, end);
        integrateBodies(bodies, begin, end);
        // Share updated positions of every block with all ranks
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       bodies.data(), counts.data(), displs.data(), posType, MPI_COMM_WORLD);
    }

    const double elapsed = MPI_Wtime() - start;

    // Gather final velocities so the full state is available for output and validation
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   bodies.data(), counts6.data(), displs6.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Type_free(&posType);

    if (isRoot) {
        printf("Simulation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
    }

    // Print results for external validation
    if (printResults && isRoot) {
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
    if (validate) {
        if (!isRoot) return 0;

        printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int rc = runSimulation(argc, argv);
    MPI_Finalize();
    return rc;
}
