#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// Number of target bodies processed together in the force kernel. Each lane keeps its own
// accumulator and sums over j in increasing order, so results match the serial algorithm bit-for-bit,
// while the fixed-width lane loop maps onto SIMD registers.
constexpr size_t TILE = 8;

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

// Updates velocities of bodies [first, first + count) using the positions of all n bodies.
// pos must hold the full, up-to-date position array; vel is indexed globally.
void computeForces(const Vec3* pos, Vec3* vel, size_t n, size_t first, size_t count) {
    for (size_t ib = 0; ib < count; ib += TILE) {
        const size_t bs = std::min(TILE, count - ib);

        double xi[TILE] = {}, yi[TILE] = {}, zi[TILE] = {};
        double Fx[TILE] = {}, Fy[TILE] = {}, Fz[TILE] = {};
        for (size_t k = 0; k < bs; ++k) {
            const Vec3& p = pos[first + ib + k];
            xi[k] = p.x;
            yi[k] = p.y;
            zi[k] = p.z;
        }

        for (size_t j = 0; j < n; ++j) {
            const double xj = pos[j].x;
            const double yj = pos[j].y;
            const double zj = pos[j].z;
            for (size_t k = 0; k < TILE; ++k) {
                const double dx = xj - xi[k];
                const double dy = yj - yi[k];
                const double dz = zj - zi[k];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx[k] += dx * invDist3;
                Fy[k] += dy * invDist3;
                Fz[k] += dz * invDist3;
            }
        }

        for (size_t k = 0; k < bs; ++k) {
            Vec3& v = vel[first + ib + k];
            v.x += DT * Fx[k];
            v.y += DT * Fy[k];
            v.z += DT * Fz[k];
        }
    }
}

// Advances positions of bodies [first, first + count).
void integrateBodies(Vec3* pos, const Vec3* vel, size_t first, size_t count) {
    for (size_t i = first; i < first + count; ++i) {
        pos[i].x += vel[i].x * DT;
        pos[i].y += vel[i].y * DT;
        pos[i].z += vel[i].z * DT;
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

static int runBenchmark(int argc, char** argv) {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool isRoot = (rank == 0);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (identical on all ranks)
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

    if (numBodies < 0) {
        if (isRoot) printf("Error: number of bodies must be non-negative\n");
        return 1;
    }
    
    if (isRoot) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t n = static_cast<size_t>(numBodies);

    // Deterministic initial state, generated identically on every rank
    std::vector<Body> init(n);
    randomizeBodies(init);
    std::vector<Vec3> pos(n), vel(n);
    for (size_t i = 0; i < n; ++i) {
        pos[i] = init[i].pos;
        vel[i] = init[i].vel;
    }
    std::vector<Body>().swap(init);

    // Block distribution of bodies across ranks
    std::vector<int> counts(size), displs(size);
    {
        const size_t base = n / size, rem = n % size;
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            counts[r] = static_cast<int>(base + (static_cast<size_t>(r) < rem ? 1 : 0));
            displs[r] = offset;
            offset += counts[r];
        }
    }
    const size_t first = static_cast<size_t>(displs[rank]);
    const size_t count = static_cast<size_t>(counts[rank]);

    MPI_Datatype vec3Type;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        // Forces use positions from the previous step (all ranks hold the full array)
        computeForces(pos.data(), vel.data(), n, first, count);
        integrateBodies(pos.data(), vel.data(), first, count);
        // Share updated positions with all ranks (in place, each rank's block is already placed)
        MPI_Allgatherv(MPI_IN_PLACE, 0, vec3Type, pos.data(), counts.data(), displs.data(), vec3Type,
                       MPI_COMM_WORLD);
    }

    // Velocities are only needed in full for output/validation
    MPI_Allgatherv(MPI_IN_PLACE, 0, vec3Type, vel.data(), counts.data(), displs.data(), vec3Type,
                   MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    long localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    int rc = 0;
    if (isRoot) {
        printf("Simulation time: %ld ms\n", maxDuration);

        std::vector<Body> bodies(n);
        for (size_t i = 0; i < n; ++i) {
            bodies[i].pos = pos[i];
            bodies[i].vel = vel[i];
        }

        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.reserve(n * 6);
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
            printf("Validating simulation results...\n");

            if (validateSimulation(bodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
    }

    // Propagate the root's exit status so all ranks agree
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Type_free(&vec3Type);
    return rc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int rc = runBenchmark(argc, argv);
    MPI_Finalize();
    return rc;
}
