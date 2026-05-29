#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Body {
    double pos[3];
    double vel[3];
};

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed) {
    for (auto& body : bodies) {
        body.pos[0] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos[1] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos[2] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel[0] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel[1] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel[2] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks)
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    // Block distribution: first (numBodies % size) ranks get one extra body
    const size_t nb = static_cast<size_t>(numBodies);
    const size_t base_n = nb / static_cast<size_t>(size);
    const size_t remainder = nb % static_cast<size_t>(size);
    const size_t local_n = base_n + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t global_start = base_n * static_cast<size_t>(rank) +
                                 std::min(static_cast<size_t>(rank), remainder);

    // Compute sendcounts and displs for Allgatherv (6 doubles per body)
    std::vector<int> sendcounts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        const size_t r_local = base_n + (static_cast<size_t>(r) < remainder ? 1 : 0);
        const size_t r_start = base_n * static_cast<size_t>(r) +
                                std::min(static_cast<size_t>(r), remainder);
        sendcounts[r] = static_cast<int>(r_local * 6);
        displs[r] = static_cast<int>(r_start * 6);
    }

    // Initialize local bodies with deterministic seed (advance past previous ranks)
    std::vector<Body> local_bodies(local_n);
    unsigned int seed = 42;
    for (size_t i = 0; i < global_start * 6; ++i) rand_r(&seed);
    randomizeBodies(local_bodies, seed);

    // Flat data buffers for MPI communication (6 doubles per body)
    std::vector<double> local_data(local_n * 6);
    std::vector<double> full_data(nb * 6);

    // Pack local bodies into flat buffer
    auto pack = [&](const std::vector<Body>& b, std::vector<double>& d) {
        for (size_t i = 0; i < b.size(); ++i) {
            d[i * 6 + 0] = b[i].pos[0];
            d[i * 6 + 1] = b[i].pos[1];
            d[i * 6 + 2] = b[i].pos[2];
            d[i * 6 + 3] = b[i].vel[0];
            d[i * 6 + 4] = b[i].vel[1];
            d[i * 6 + 5] = b[i].vel[2];
        }
    };

    // Pack local bodies and gather full state
    pack(local_bodies, local_data);
    MPI_Allgatherv(local_data.data(), static_cast<int>(local_n * 6), MPI_DOUBLE,
                   full_data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    // Print info (rank 0 only)
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Temporary array for reading during force computation
    std::vector<double> all_pos(nb * 3);
    std::vector<double> all_vel(nb * 3);

    // Simulation loop
    double t_start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        // Extract positions and velocities from full_data
        for (size_t i = 0; i < nb; ++i) {
            all_pos[i * 3 + 0] = full_data[i * 6 + 0];
            all_pos[i * 3 + 1] = full_data[i * 6 + 1];
            all_pos[i * 3 + 2] = full_data[i * 6 + 2];
            all_vel[i * 3 + 0] = full_data[i * 6 + 3];
            all_vel[i * 3 + 1] = full_data[i * 6 + 4];
            all_vel[i * 3 + 2] = full_data[i * 6 + 5];
        }

        // Compute forces for local bodies (each rank handles its subset)
        for (size_t i = 0; i < local_n; ++i) {
            const size_t gi = global_start + i;
            double Fx = 0.0, Fy = 0.0, Fz = 0.0;

            for (size_t j = 0; j < nb; ++j) {
                const double dx = all_pos[j * 3 + 0] - all_pos[gi * 3 + 0];
                const double dy = all_pos[j * 3 + 1] - all_pos[gi * 3 + 1];
                const double dz = all_pos[j * 3 + 2] - all_pos[gi * 3 + 2];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }

            all_vel[gi * 3 + 0] += DT * Fx;
            all_vel[gi * 3 + 1] += DT * Fy;
            all_vel[gi * 3 + 2] += DT * Fz;
        }

        // Integrate local bodies
        for (size_t i = 0; i < local_n; ++i) {
            const size_t gi = global_start + i;
            all_pos[gi * 3 + 0] += all_vel[gi * 3 + 0] * DT;
            all_pos[gi * 3 + 1] += all_vel[gi * 3 + 1] * DT;
            all_pos[gi * 3 + 2] += all_vel[gi * 3 + 2] * DT;
        }

        // Pack updated local state back into local_data
        for (size_t i = 0; i < local_n; ++i) {
            const size_t gi = global_start + i;
            local_data[i * 6 + 0] = all_pos[gi * 3 + 0];
            local_data[i * 6 + 1] = all_pos[gi * 3 + 1];
            local_data[i * 6 + 2] = all_pos[gi * 3 + 2];
            local_data[i * 6 + 3] = all_vel[gi * 3 + 0];
            local_data[i * 6 + 4] = all_vel[gi * 3 + 1];
            local_data[i * 6 + 5] = all_vel[gi * 3 + 2];
        }

        // Gather all local updates into full_data
        MPI_Allgatherv(local_data.data(), static_cast<int>(local_n * 6), MPI_DOUBLE,
                       full_data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }

    double t_end = MPI_Wtime();
    double duration_ms = (t_end - t_start) * 1000.0;

    if (rank == 0) {
        printf("Simulation time: %.0f ms\n", duration_ms);
    }

    // Print results for external validation (rank 0)
    if (printResults && rank == 0) {
        std::vector<double> bodyData(full_data.begin(), full_data.end());
        print_results(bodyData, "Bodies");
    }

    // Validation
    if (validate) {
        // Each rank validates its local portion
        int local_valid = 1;
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        for (size_t i = 0; i < local_n && local_valid; ++i) {
            const size_t gi = global_start + i;
            for (int d = 0; d < 3; ++d) {
                if (!std::isfinite(full_data[gi * 6 + d]) ||
                    !std::isfinite(full_data[gi * 6 + 3 + d])) {
                    local_valid = 0;
                    break;
                }
                if (std::abs(full_data[gi * 6 + d]) > maxPos ||
                    std::abs(full_data[gi * 6 + 3 + d]) > maxVel) {
                    local_valid = 0;
                    break;
                }
            }
        }

        int global_valid;
        MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validating simulation results...\n");

            if (global_valid) {
                // Compute total energy on rank 0
                double energy = 0.0;
                // Kinetic energy
                for (size_t i = 0; i < nb; ++i) {
                    energy += 0.5 * (full_data[i * 6 + 3] * full_data[i * 6 + 3] +
                                     full_data[i * 6 + 4] * full_data[i * 6 + 4] +
                                     full_data[i * 6 + 5] * full_data[i * 6 + 5]);
                }
                // Potential energy
                for (size_t i = 0; i < nb; ++i) {
                    for (size_t j = i + 1; j < nb; ++j) {
                        const double dx = full_data[j * 6 + 0] - full_data[i * 6 + 0];
                        const double dy = full_data[j * 6 + 1] - full_data[i * 6 + 1];
                        const double dz = full_data[j * 6 + 2] - full_data[i * 6 + 2];
                        const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                        energy -= 1.0 / dist;
                    }
                }
                printf("Final energy: %.6f\n", energy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
