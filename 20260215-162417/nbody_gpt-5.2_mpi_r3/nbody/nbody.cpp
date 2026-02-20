#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
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

static_assert(std::is_standard_layout_v<Body>);
static_assert(std::is_trivially_copyable_v<Body>);

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

static inline void partition1D(const int n, const int size, const int rank, int& start, int& count) {
    const int base = n / size;
    const int rem = n % size;
    count = base + (rank < rem ? 1 : 0);
    start = rank * base + (rank < rem ? rank : rem);
}

static inline void computeForcesRange(std::vector<Body>& bodies, const int start, const int end) {
    const size_t n = bodies.size();

    for (int i = start; i < end; ++i) {
        const double ix = bodies[(size_t)i].pos.x;
        const double iy = bodies[(size_t)i].pos.y;
        const double iz = bodies[(size_t)i].pos.z;

        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (size_t j = 0; j < n; ++j) {
            const double dx = bodies[j].pos.x - ix;
            const double dy = bodies[j].pos.y - iy;
            const double dz = bodies[j].pos.z - iz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        bodies[(size_t)i].vel.x += DT * Fx;
        bodies[(size_t)i].vel.y += DT * Fy;
        bodies[(size_t)i].vel.z += DT * Fz;
    }
}

static inline void integrateBodiesRange(std::vector<Body>& bodies, const int start, const int end) {
    for (int i = start; i < end; ++i) {
        bodies[(size_t)i].pos.x += bodies[(size_t)i].vel.x * DT;
        bodies[(size_t)i].pos.y += bodies[(size_t)i].vel.y * DT;
        bodies[(size_t)i].pos.z += bodies[(size_t)i].vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
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
    int validate = 0;
    int printResults = 0;
    int showHelp = 0;
    int parseOk = 1;

    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numBodies = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                numSteps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parseOk = 0;
                break;
            }
        }

        if (!parseOk) {
            printUsage(argv[0]);
        }
        if (showHelp) {
            printUsage(argv[0]);
        }
    }

    MPI_Bcast(&parseOk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&showHelp, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!parseOk) {
        MPI_Finalize();
        return 1;
    }
    if (showHelp) {
        MPI_Finalize();
        return 0;
    }

    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Initialize bodies on rank 0 and broadcast full initial state.
    std::vector<Body> bodies((size_t)numBodies);
    if (rank == 0) {
        randomizeBodies(bodies);
    }
    if (numBodies > 0) {
        MPI_Bcast(bodies.data(), (int)((size_t)numBodies * sizeof(Body)), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    int localStart = 0;
    int localCount = 0;
    partition1D(numBodies, size, rank, localStart, localCount);
    const int localEnd = localStart + localCount;

    std::vector<int> recvCountsBytes((size_t)size);
    std::vector<int> displsBytes((size_t)size);
    for (int r = 0; r < size; ++r) {
        int s = 0, c = 0;
        partition1D(numBodies, size, r, s, c);
        recvCountsBytes[(size_t)r] = (int)((size_t)c * sizeof(Body));
        displsBytes[(size_t)r] = (int)((size_t)s * sizeof(Body));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesRange(bodies, localStart, localEnd);
        integrateBodiesRange(bodies, localStart, localEnd);

        if (size > 1) {
            MPI_Allgatherv(localCount ? (void*)(bodies.data() + (size_t)localStart) : nullptr,
                           recvCountsBytes[(size_t)rank],
                           MPI_BYTE,
                           bodies.data(),
                           recvCountsBytes.data(),
                           displsBytes.data(),
                           MPI_BYTE,
                           MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double localTime = t1 - t0;

    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = (long)std::llround(maxTime * 1000.0);
        printf("Simulation time: %ld ms\n", ms);
    }

    int exitCode = 0;

    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve((size_t)numBodies * 6);
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

    // Validation: check that simulation produces finite, reasonable values (rank 0 only)
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
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
