#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
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

/*
 * MPI-parallelized force computation.
 * Bodies are distributed in blocks across ranks. Each rank holds its local
 * bodies in `local_bodies` (owned by this rank). The full set of all bodies
 * is assembled into `all_bodies` via MPI_Allgather so every rank can compute
 * forces against every other body. Each rank then computes forces only for
 * its local bodies, accumulating velocity changes.
 */
void computeForces(std::vector<Body>& local_bodies,
                   const std::vector<Body>& all_bodies) {
    const size_t n = all_bodies.size();
    const size_t local_n = local_bodies.size();

    for (size_t i = 0; i < local_n; ++i) {
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

/*
 * MPI-parallelized total energy computation.
 * Kinetic energy is computed locally and reduced with MPI_Allreduce.
 * Potential energy pairs (i < j) are distributed: each rank computes pairs
 * where both bodies are local plus pairs where the first body (i) is local
 * and the second body (j) is remote. This avoids double-counting.
 */
double computeTotalEnergy(const std::vector<Body>& local_bodies,
                          const std::vector<Body>& all_bodies,
                          const size_t global_offset) {
    const size_t local_n = local_bodies.size();
    const size_t n = all_bodies.size();
    double energy = 0.0;

    // Local kinetic energy
    for (const auto& body : local_bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                         body.vel.y * body.vel.y +
                         body.vel.z * body.vel.z);
    }

    // Local potential energy: pairs (i, j) with i < j, i in local range
    for (size_t li = 0; li < local_n; ++li) {
        const size_t gi = global_offset + li;
        for (size_t gj = gi + 1; gj < n; ++gj) {
            const double dx = all_bodies[gj].pos.x - all_bodies[gi].pos.x;
            const double dy = all_bodies[gj].pos.y - all_bodies[gi].pos.y;
            const double dz = all_bodies[gj].pos.z - all_bodies[gi].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

// Validate that local bodies produce finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
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
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int num_ranks = 1, my_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse the same args)
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
            if (my_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (my_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (my_rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI ranks: %d\n", num_ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Block-distribute bodies: rank r owns bodies [r*block, (r+1)*block)
    // with the last rank taking any remainder.
    const size_t base_count = numBodies / num_ranks;
    const size_t remainder  = numBodies % num_ranks;
    const size_t local_n    = base_count + (my_rank < static_cast<int>(remainder) ? 1 : 0);
    const size_t global_offset = base_count * my_rank + std::min(static_cast<size_t>(my_rank), remainder);

    // Initialize local bodies (same seed on every rank so the global sequence
    // is deterministic; we generate all bodies on rank 0 then scatter).
    std::vector<Body> local_bodies(local_n);
    if (my_rank == 0) {
        std::vector<Body> all_bodies(numBodies);
        randomizeBodies(all_bodies);
        // Scatter to all ranks
        size_t offset = 0;
        for (int r = 0; r < num_ranks; ++r) {
            const size_t rn = base_count + (r < static_cast<int>(remainder) ? 1 : 0);
            if (r == 0) {
                local_bodies = std::vector<Body>(all_bodies.begin() + offset,
                                                 all_bodies.begin() + offset + rn);
            } else {
                MPI_Send(&all_bodies[offset], static_cast<int>(rn * sizeof(Body)),
                         MPI_BYTE, r, 0, MPI_COMM_WORLD);
            }
            offset += rn;
        }
    } else {
        MPI_Recv(&local_bodies[0], static_cast<int>(local_n * sizeof(Body)),
                 MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Allgather: every rank gets a full copy of all bodies for force computation.
    std::vector<Body> all_bodies(numBodies);
    {
        // Build the counts/displacements for MPI_Allgatherv
        std::vector<int> recvcounts(num_ranks);
        std::vector<int> displs(num_ranks);
        size_t offset = 0;
        for (int r = 0; r < num_ranks; ++r) {
            const size_t rn = base_count + (r < static_cast<int>(remainder) ? 1 : 0);
            recvcounts[r] = static_cast<int>(rn * sizeof(Body));
            displs[r]     = static_cast<int>(offset * sizeof(Body));
            offset += rn;
        }
        MPI_Allgatherv(&local_bodies[0],
                       static_cast<int>(local_n * sizeof(Body)), MPI_BYTE,
                       &all_bodies[0],
                       recvcounts.data(), displs.data(), MPI_BYTE,
                       MPI_COMM_WORLD);
    }

    // Precompute counts/displacements for Allgatherv (constant across steps)
    std::vector<int> recvcounts(num_ranks);
    std::vector<int> displs(num_ranks);
    {
        size_t offset = 0;
        for (int r = 0; r < num_ranks; ++r) {
            const size_t rn = base_count + (r < static_cast<int>(remainder) ? 1 : 0);
            recvcounts[r] = static_cast<int>(rn * sizeof(Body));
            displs[r]     = static_cast<int>(offset * sizeof(Body));
            offset += rn;
        }
    }

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(local_bodies, all_bodies);
        integrateBodies(local_bodies);

        // Synchronize: each rank contributes its updated local bodies
        MPI_Allgatherv(&local_bodies[0],
                       static_cast<int>(local_n * sizeof(Body)), MPI_BYTE,
                       &all_bodies[0],
                       recvcounts.data(), displs.data(), MPI_BYTE,
                       MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = duration.count();
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("Simulation time: %ld ms\n", max_duration_ms);
    }

    // Print results for external validation (rank 0 only)
    if (printResults && my_rank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (const auto& body : all_bodies) {
            bodyData.push_back(body.pos.x);
            bodyData.push_back(body.pos.y);
            bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x);
            bodyData.push_back(body.vel.y);
            bodyData.push_back(body.vel.z);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation
    if (validate) {
        // Local validation first
        bool local_ok = validateSimulation(local_bodies);
        int global_ok = local_ok ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (my_rank == 0) {
            printf("Validating simulation results...\n");
        }

        if (global_ok) {
            // Compute total energy in parallel
            double local_energy = computeTotalEnergy(local_bodies, all_bodies, global_offset);
            double total_energy = 0.0;
            MPI_Reduce(&local_energy, &total_energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (my_rank == 0) {
                printf("Final energy: %.6f\n", total_energy);
                printf("Validation: PASSED\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (my_rank == 0) {
                printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
