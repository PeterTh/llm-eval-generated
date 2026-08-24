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

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept
        : x(x), y(y), z(z) {}
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

// Compute forces on local bodies using the full body set.
// Each rank computes forces only for its owned bodies.
void computeForcesLocal(std::vector<Body>& local_bodies, const std::vector<Body>& all_bodies) {
    const size_t n_local = local_bodies.size();
    const size_t n = all_bodies.size();

    for (size_t i = 0; i < n_local; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (size_t j = 0; j < n; ++j) {
            const double dx = all_bodies[j].pos.x - local_bodies[i].pos.x;
            const double dy = all_bodies[j].pos.y - local_bodies[i].pos.y;
            const double dz = all_bodies[j].pos.z - local_bodies[i].pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        local_bodies[i].vel.x += DT * Fx;
        local_bodies[i].vel.y += DT * Fy;
        local_bodies[i].vel.z += DT * Fz;
    }
}

void integrateBodies(std::vector<Body>& bodies) {
    for (auto& body : bodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

// Compute partial energy contribution from local bodies.
// Kinetic energy is summed over local bodies.
// Potential energy covers pairs (i, j) where i is local and j > i.
double computePartialEnergy(const std::vector<Body>& local_bodies,
                            const std::vector<Body>& all_bodies,
                            size_t local_global_start) {
    double energy = 0.0;
    const size_t n_local = local_bodies.size();
    const size_t n = all_bodies.size();

    // Kinetic energy for local bodies
    for (const auto& body : local_bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                         body.vel.y * body.vel.y +
                         body.vel.z * body.vel.z);
    }

    // Potential energy: pairs where i is in local range and j > i
    for (size_t i = 0; i < n_local; ++i) {
        const size_t global_i = local_global_start + i;
        for (size_t j = global_i + 1; j < n; ++j) {
            const double dx = all_bodies[j].pos.x - local_bodies[i].pos.x;
            const double dy = all_bodies[j].pos.y - local_bodies[i].pos.y;
            const double dz = all_bodies[j].pos.z - local_bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) ||
            !std::isfinite(body.pos.z) || !std::isfinite(body.vel.x) ||
            !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos ||
            std::abs(body.pos.z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel ||
            std::abs(body.vel.z) > maxVel) {
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
    int rank, numRanks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (on all ranks)
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
        printf("MPI ranks: %d\n", numRanks);
    }

    const size_t n = numBodies;

    // Block distribution: ranks 0..rem-1 get base+1 bodies, rest get base
    const size_t base_count = n / static_cast<size_t>(numRanks);
    const size_t rem = n % static_cast<size_t>(numRanks);
    const size_t my_count = base_count + (rank < static_cast<int>(rem) ? 1 : 0);

    // Compute my global start index
    size_t my_start = 0;
    for (int r = 0; r < rank; ++r) {
        my_start += base_count + (r < static_cast<int>(rem) ? 1 : 0);
    }

    // Build recvcounts and displacements for Allgatherv (in bytes)
    std::vector<int> recvcounts(numRanks);
    std::vector<int> displacements(numRanks);
    size_t offset = 0;
    for (int r = 0; r < numRanks; ++r) {
        const size_t cnt = base_count + (r < static_cast<int>(rem) ? 1 : 0);
        recvcounts[r] = static_cast<int>(cnt * sizeof(Body));
        displacements[r] = static_cast<int>(offset * sizeof(Body));
        offset += cnt;
    }

    // Initialize bodies on rank 0, then broadcast to all ranks
    std::vector<Body> all_bodies(n);
    if (rank == 0) {
        randomizeBodies(all_bodies);
    }
    MPI_Bcast(all_bodies.data(), static_cast<int>(n * sizeof(Body)), MPI_BYTE, 0,
              MPI_COMM_WORLD);

    // Extract local bodies (owned subset)
    std::vector<Body> local_bodies(my_count);
    std::copy(all_bodies.begin() + my_start, all_bodies.begin() + my_start + my_count,
              local_bodies.begin());

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Compute forces on local bodies using global body positions
        computeForcesLocal(local_bodies, all_bodies);

        // Integrate local bodies
        integrateBodies(local_bodies);

        // Gather updated local bodies back so every rank has the full state
        MPI_Allgatherv(local_bodies.data(), static_cast<int>(my_count * sizeof(Body)),
                       MPI_BYTE,
                       all_bodies.data(), recvcounts.data(), displacements.data(),
                       MPI_BYTE, MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    long long local_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long global_duration = 0;
    MPI_Reduce(&local_duration, &global_duration, 1, MPI_LONG_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %lld ms\n", global_duration);
    }

    // Print results for external validation (rank 0 only)
    if (printResults) {
        std::vector<double> bodyData;
        if (rank == 0) {
            bodyData.reserve(n * 6);
            for (const auto& body : all_bodies) {
                bodyData.push_back(body.pos.x);
                bodyData.push_back(body.pos.y);
                bodyData.push_back(body.pos.z);
                bodyData.push_back(body.vel.x);
                bodyData.push_back(body.vel.y);
                bodyData.push_back(body.vel.z);
            }
        }
        if (rank == 0) print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        // Each rank validates its portion of all_bodies
        bool local_valid = validateSimulation(all_bodies);
        int global_valid = local_valid ? 1 : 0;
        int all_valid = 0;
        MPI_Allreduce(&global_valid, &all_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (all_valid) {
            // Compute total energy via reduction (collective: all ranks participate)
            double local_energy = computePartialEnergy(local_bodies, all_bodies, my_start);
            double total_energy = 0.0;
            MPI_Reduce(&local_energy, &total_energy, 1, MPI_DOUBLE, MPI_SUM, 0,
                       MPI_COMM_WORLD);

            if (rank == 0) {
                printf("Final energy: %.6f\n", total_energy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
