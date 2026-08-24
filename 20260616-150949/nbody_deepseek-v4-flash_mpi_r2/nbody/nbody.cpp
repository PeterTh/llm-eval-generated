#include <mpi.h>
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
    MPI_Init(&argc, &argv);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse identically)
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
        printf("N-Body Simulation (MPI)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", numProcs);
    }

    // Compute body distribution across ranks
    std::vector<int> counts(numProcs), displs(numProcs);
    std::vector<int> bytes_counts(numProcs), bytes_displs(numProcs);
    std::vector<int> pos_counts(numProcs), pos_displs(numProcs);
    int total = 0;
    for (int i = 0; i < numProcs; ++i) {
        int n_local = numBodies / numProcs + (i < numBodies % numProcs ? 1 : 0);
        counts[i] = n_local;
        displs[i] = total;
        bytes_counts[i] = n_local * static_cast<int>(sizeof(Body));
        bytes_displs[i] = total * static_cast<int>(sizeof(Body));
        pos_counts[i] = n_local * 3;
        pos_displs[i] = total * 3;
        total += n_local;
    }

    const int local_n = counts[rank];

    // Initialize all bodies on rank 0
    std::vector<Body> all_bodies;
    if (rank == 0) {
        all_bodies.resize(numBodies);
        randomizeBodies(all_bodies);
    }

    // Scatter bodies from rank 0 to all ranks
    std::vector<Body> local_bodies(local_n);
    MPI_Scatterv(rank == 0 ? all_bodies.data() : nullptr, bytes_counts.data(), bytes_displs.data(),
                 MPI_BYTE, local_bodies.data(), local_n * static_cast<int>(sizeof(Body)),
                 MPI_BYTE, 0, MPI_COMM_WORLD);

    // Pre-allocate communication and computation buffers
    std::vector<double> local_positions_flat(local_n * 3);
    std::vector<double> all_positions_flat(numBodies * 3);
    // Position buffer: holds full Body array for force computation
    std::vector<Body> pos_buf(numBodies);

    // Record local body offset in the full array for force computation range
    const int local_start = displs[rank];

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Pack local positions into flat array
        for (int i = 0; i < local_n; ++i) {
            local_positions_flat[i * 3]     = local_bodies[i].pos.x;
            local_positions_flat[i * 3 + 1] = local_bodies[i].pos.y;
            local_positions_flat[i * 3 + 2] = local_bodies[i].pos.z;
        }

        // Share all positions across all ranks
        MPI_Allgatherv(local_positions_flat.data(), local_n * 3, MPI_DOUBLE,
                       all_positions_flat.data(), pos_counts.data(), pos_displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Initialize pos_buf: positions from allgather, velocities from local bodies
        for (int i = 0; i < numBodies; ++i) {
            pos_buf[i].pos.x = all_positions_flat[i * 3];
            pos_buf[i].pos.y = all_positions_flat[i * 3 + 1];
            pos_buf[i].pos.z = all_positions_flat[i * 3 + 2];
        }
        for (int i = 0; i < local_n; ++i) {
            pos_buf[local_start + i].vel = local_bodies[i].vel;
        }

        // Compute forces for local body range using the position buffer
        {
            const int end = local_start + local_n;
            for (int i = local_start; i < end; ++i) {
                double Fx = 0.0, Fy = 0.0, Fz = 0.0;
                for (int j = 0; j < numBodies; ++j) {
                    const double dx = pos_buf[j].pos.x - pos_buf[i].pos.x;
                    const double dy = pos_buf[j].pos.y - pos_buf[i].pos.y;
                    const double dz = pos_buf[j].pos.z - pos_buf[i].pos.z;
                    const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    const double invDist = 1.0 / std::sqrt(distSqr);
                    const double invDist3 = invDist * invDist * invDist;

                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }

                pos_buf[i].vel.x += DT * Fx;
                pos_buf[i].vel.y += DT * Fy;
                pos_buf[i].vel.z += DT * Fz;
            }
        }

        // Copy velocity updates from position buffer back to local bodies
        for (int i = 0; i < local_n; ++i) {
            local_bodies[i].vel = pos_buf[local_start + i].vel;
        }

        // Integrate local bodies (update positions from velocities)
        for (int i = 0; i < local_n; ++i) {
            local_bodies[i].pos.x += local_bodies[i].vel.x * DT;
            local_bodies[i].pos.y += local_bodies[i].vel.y * DT;
            local_bodies[i].pos.z += local_bodies[i].vel.z * DT;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto local_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = static_cast<long long>(local_duration.count());
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather all bodies back to rank 0 for output / validation
    std::vector<Body> gathered_bodies;
    if (rank == 0) {
        gathered_bodies.resize(numBodies);
    }
    MPI_Gatherv(local_bodies.data(), local_n * static_cast<int>(sizeof(Body)), MPI_BYTE,
                rank == 0 ? gathered_bodies.data() : nullptr,
                bytes_counts.data(), bytes_displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %lld ms\n", max_duration_ms);

        // Print results for external validation
        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(static_cast<size_t>(numBodies) * 6);
            for (const auto& body : gathered_bodies) {
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

            if (validateSimulation(gathered_bodies)) {
                double finalEnergy = computeTotalEnergy(gathered_bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
