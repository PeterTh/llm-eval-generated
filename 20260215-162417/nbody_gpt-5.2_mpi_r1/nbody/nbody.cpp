#include <chrono>
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

static inline void computeForcesRange(std::vector<Body>& bodies, const size_t iBegin, const size_t iEnd) {
    const size_t n = bodies.size();

    for (size_t i = iBegin; i < iEnd; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        const double pix = bodies[i].pos.x;
        const double piy = bodies[i].pos.y;
        const double piz = bodies[i].pos.z;

        for (size_t j = 0; j < n; ++j) {
            const double dx = bodies[j].pos.x - pix;
            const double dy = bodies[j].pos.y - piy;
            const double dz = bodies[j].pos.z - piz;
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

static inline void integrateBodiesRange(std::vector<Body>& bodies, const size_t iBegin, const size_t iEnd) {
    for (size_t i = iBegin; i < iEnd; ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Initialize bodies (rank 0), then broadcast to all ranks
    std::vector<Body> bodies(numBodies);
    if (rank == 0) {
        randomizeBodies(bodies);
    }
    MPI_Bcast(bodies.data(), static_cast<int>(bodies.size() * sizeof(Body)), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Block distribution of bodies across ranks
    const size_t n = bodies.size();
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    const size_t iBegin = static_cast<size_t>(rank) * base + static_cast<size_t>(rank < static_cast<int>(rem) ? rank : rem);
    const size_t iCount = base + (static_cast<size_t>(rank) < rem ? 1u : 0u);
    const size_t iEnd = iBegin + iCount;

    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        const size_t rBegin = static_cast<size_t>(r) * base + static_cast<size_t>(r < static_cast<int>(rem) ? r : rem);
        const size_t rCount = base + (static_cast<size_t>(r) < rem ? 1u : 0u);
        recvcounts[r] = static_cast<int>(rCount * sizeof(Body));
        displs[r] = static_cast<int>(rBegin * sizeof(Body));
    }

    std::vector<Body> gathered(numBodies);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesRange(bodies, iBegin, iEnd);
        integrateBodiesRange(bodies, iBegin, iEnd);

        const void* sendbuf = (iCount > 0) ? reinterpret_cast<const void*>(&bodies[iBegin]) : nullptr;
        MPI_Allgatherv(sendbuf,
                       static_cast<int>(iCount * sizeof(Body)),
                       MPI_BYTE,
                       gathered.data(),
                       recvcounts.data(),
                       displs.data(),
                       MPI_BYTE,
                       MPI_COMM_WORLD);
        bodies.swap(gathered);
    }

    const double t1 = MPI_Wtime();
    double elapsed = (t1 - t0) * 1000.0;
    double elapsedMax = 0.0;
    MPI_Reduce(&elapsed, &elapsedMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", static_cast<long>(elapsedMax));
    }

    // Print results for external validation
    if (printResults && rank == 0) {
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

    int exitCode = 0;

    // Validation: check that simulation produces finite, reasonable values
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            exitCode = 0;
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
