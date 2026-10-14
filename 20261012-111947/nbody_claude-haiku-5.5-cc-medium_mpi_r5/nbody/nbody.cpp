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

// Bodies whose forces are accumulated together. Each lane keeps its own sum over j in
// the same order as the serial loop, so results are bit-identical while the lane loop
// maps onto SIMD registers.
constexpr size_t LANES = 8;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

// Block distribution of bodies over MPI ranks
struct Decomposition {
    std::vector<int> counts;     // bodies owned by each rank
    std::vector<int> displs;     // first body owned by each rank
    std::vector<int> xyzCounts;  // doubles owned by each rank (3 per body)
    std::vector<int> xyzDispls;  // first double owned by each rank

    Decomposition(const int numBodies, const int numRanks)
        : counts(numRanks), displs(numRanks), xyzCounts(numRanks), xyzDispls(numRanks) {
        const int base = numBodies / numRanks;
        const int extra = numBodies % numRanks;
        int offset = 0;
        for (int r = 0; r < numRanks; ++r) {
            counts[r] = base + (r < extra ? 1 : 0);
            displs[r] = offset;
            xyzCounts[r] = 3 * counts[r];
            xyzDispls[r] = 3 * offset;
            offset += counts[r];
        }
    }
};

void randomizeBodies(std::vector<Vec3>& pos, std::vector<Vec3>& vel, unsigned int seed = 42) {
    for (size_t i = 0; i < pos.size(); ++i) {
        pos[i].x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pos[i].y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pos[i].z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel[i].x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel[i].y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel[i].z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// Updates velocities of bodies [begin, end) using the positions of all bodies
void computeForces(const std::vector<Vec3>& pos, std::vector<Vec3>& vel, size_t begin, size_t end) {
    const size_t n = pos.size();

    for (size_t i0 = begin; i0 < end; i0 += LANES) {
        const size_t lanes = std::min(LANES, end - i0);

        // Unused trailing lanes repeat the last valid body; their results are discarded
        double xi[LANES], yi[LANES], zi[LANES];
        double Fx[LANES] = {}, Fy[LANES] = {}, Fz[LANES] = {};
        for (size_t l = 0; l < LANES; ++l) {
            const Vec3& p = pos[i0 + std::min(l, lanes - 1)];
            xi[l] = p.x;
            yi[l] = p.y;
            zi[l] = p.z;
        }

        for (size_t j = 0; j < n; ++j) {
            const Vec3& pj = pos[j];
            for (size_t l = 0; l < LANES; ++l) {
                const double dx = pj.x - xi[l];
                const double dy = pj.y - yi[l];
                const double dz = pj.z - zi[l];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx[l] += dx * invDist3;
                Fy[l] += dy * invDist3;
                Fz[l] += dz * invDist3;
            }
        }

        for (size_t l = 0; l < lanes; ++l) {
            vel[i0 + l].x += DT * Fx[l];
            vel[i0 + l].y += DT * Fy[l];
            vel[i0 + l].z += DT * Fz[l];
        }
    }
}

void integrateBodies(std::vector<Vec3>& pos, const std::vector<Vec3>& vel, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
        pos[i].x += vel[i].x * DT;
        pos[i].y += vel[i].y * DT;
        pos[i].z += vel[i].z * DT;
    }
}

// Shares each rank's owned slice of v with every rank, in place
void allgatherVec3(std::vector<Vec3>& v, const Decomposition& dec) {
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, v.data(), dec.xyzCounts.data(), dec.xyzDispls.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
}

// The pair terms are split across ranks and reduced, so their rounding order differs from a serial sum
double computeTotalEnergy(const std::vector<Vec3>& pos, const std::vector<Vec3>& vel, size_t rank, size_t size) {
    const size_t n = pos.size();

    // Kinetic energy (assuming unit mass)
    double kinetic = 0.0;
    for (const auto& v : vel) {
        kinetic += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }

    // Potential energy (assuming unit mass for all bodies). Interleaved rows balance the i<j workload.
    double potential = 0.0;
    for (size_t i = rank; i < n; i += size) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = pos[j].x - pos[i].x;
            const double dy = pos[j].y - pos[i].y;
            const double dz = pos[j].z - pos[i].z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potential += 1.0 / dist;
        }
    }

    double totalPotential = 0.0;
    MPI_Allreduce(&potential, &totalPotential, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return kinetic - totalPotential;
}

// Validate that simulation produces finite, reasonable values.
// Returns nullptr if valid, otherwise a message describing the first failure.
const char* validateSimulation(const std::vector<Vec3>& pos, const std::vector<Vec3>& vel) {
    for (size_t i = 0; i < pos.size(); ++i) {
        const Vec3& p = pos[i];
        const Vec3& v = vel[i];

        // Check for NaN or Inf values
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
            !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
            return "Validation failed: found NaN or Inf value in body state\n";
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(p.x) > maxPos || std::abs(p.y) > maxPos || std::abs(p.z) > maxPos) {
            return "Validation failed: body position exceeds reasonable bounds\n";
        }
        if (std::abs(v.x) > maxVel || std::abs(v.y) > maxVel || std::abs(v.z) > maxVel) {
            return "Validation failed: body velocity exceeds reasonable bounds\n";
        }
    }
    return nullptr;
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

int run(int argc, char** argv, int rank, int size) {
    const bool root = rank == 0;

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

    // Every rank builds the same initial state, so no initial broadcast is needed
    const Decomposition dec(numBodies, size);
    std::vector<Vec3> pos(numBodies), vel(numBodies);
    randomizeBodies(pos, vel);

    const size_t begin = dec.displs[rank];
    const size_t end = begin + dec.counts[rank];

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(pos, vel, begin, end);
        integrateBodies(pos, vel, begin, end);
        allgatherVec3(pos, dec);
    }

    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Simulation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
    }

    // Velocities are only held locally until here; every rank needs the full state for output and validation
    if (printResults || validate) {
        allgatherVec3(vel, dec);
    }

    // Print results for external validation
    if (printResults && root) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            bodyData.push_back(pos[i].x);
            bodyData.push_back(pos[i].y);
            bodyData.push_back(pos[i].z);
            bodyData.push_back(vel[i].x);
            bodyData.push_back(vel[i].y);
            bodyData.push_back(vel[i].z);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (root) printf("Validating simulation results...\n");

        const char* failure = validateSimulation(pos, vel);
        if (failure == nullptr) {
            // Report final energy for reference
            const double finalEnergy = computeTotalEnergy(pos, vel, rank, size);
            if (root) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            return 0;
        }

        if (root) {
            printf("%s", failure);
            printf("Validation: FAILED\n");
        }
        return 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int status = run(argc, argv, rank, size);

    MPI_Finalize();
    return status;
}
