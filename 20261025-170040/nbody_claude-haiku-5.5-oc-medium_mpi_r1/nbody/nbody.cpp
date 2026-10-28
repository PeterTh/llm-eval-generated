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

static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be packed as 6 doubles for MPI transfer");

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

// Block distribution of n bodies over `size` ranks: rank r owns [start(r), start(r+1)).
inline size_t blockStart(size_t n, int rank, int size) {
    return n * static_cast<size_t>(rank) / static_cast<size_t>(size);
}

// Accumulates the force-induced velocity change for bodies [begin, begin+count),
// using positions of all n bodies (px/py/pz in structure-of-arrays layout).
// Each body's force sum runs over j = 0..n-1 in the original order. Only the
// owned bodies' velocities are modified.
void computeForcesBlock(const double* px, const double* py, const double* pz, size_t n,
                        size_t begin, size_t count, Body* bodies) {
    for (size_t i = begin; i < begin + count; ++i) {
        const double xi = px[i], yi = py[i], zi = pz[i];
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (size_t j = 0; j < n; ++j) {
            const double dx = px[j] - xi;
            const double dy = py[j] - yi;
            const double dz = pz[j] - zi;
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

// Advances positions of bodies [begin, begin+count) by their velocities.
void integrateBlock(Body* bodies, size_t begin, size_t count) {
    for (size_t i = begin; i < begin + count; ++i) {
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
bool validateSimulation(const std::vector<Body>& bodies, bool report) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            if (report) printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            if (report) printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            if (report) printf("Validation failed: body velocity exceeds reasonable bounds\n");
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

// Runs the distributed simulation. All ranks hold identical copies of the body
// state at every point where control leaves a collective, so all ranks take the
// same control-flow decisions; only rank 0 performs console output.
int runBenchmark(int argc, char** argv, int rank, int size) {
    const bool root = (rank == 0);

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
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (root) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Initialize bodies (deterministic, so every rank generates the same state)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    const size_t n = bodies.size();
    const size_t myBegin = blockStart(n, rank, size);
    const size_t myEnd = blockStart(n, rank + 1, size);
    const size_t myCount = myEnd - myBegin;

    // MPI datatype describing one Body (6 contiguous doubles)
    MPI_Datatype bodyType;
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);

    // Allgatherv layout: contiguous block per rank, counted in bodies
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        const size_t b = blockStart(n, r, size);
        const size_t e = blockStart(n, r + 1, size);
        counts[r] = static_cast<int>(e - b);
        displs[r] = static_cast<int>(b);
    }

    // Structure-of-arrays position copy used by the force kernel
    std::vector<double> px(n), py(n), pz(n);

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        for (size_t i = 0; i < n; ++i) {
            px[i] = bodies[i].pos.x;
            py[i] = bodies[i].pos.y;
            pz[i] = bodies[i].pos.z;
        }

        computeForcesBlock(px.data(), py.data(), pz.data(), n, myBegin, myCount, bodies.data());
        integrateBlock(bodies.data(), myBegin, myCount);

        // Share the updated state of every rank's block with all ranks
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bodies.data(), counts.data(), displs.data(),
                       bodyType, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double localElapsed = t1 - t0;
    double maxElapsed = 0.0;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Simulation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
    }
    
    // Print results for external validation
    if (printResults && root) {
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
    int rc = 0;
    if (validate) {
        if (root) printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies, root)) {
            // Report final energy for reference (serial, identical to the original)
            if (root) {
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            rc = 0;
        } else {
            if (root) printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    MPI_Type_free(&bodyType);
    return rc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rc = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return rc;
}
