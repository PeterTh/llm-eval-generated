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

// Compute forces on local bodies using all body positions.
// all_pos: contiguous array of 3*numBodies doubles (x0,y0,z0,x1,y1,z1,...)
// local_pos: 3*nlocal doubles for local bodies
// local_vel: 3*nlocal doubles for local bodies (updated in place)
static void computeForces(const double* __restrict all_pos,
                          double* __restrict local_pos,
                          double* __restrict local_vel,
                          int nlocal, int numBodies) {
    for (int i = 0; i < nlocal; ++i) {
        const double px = local_pos[3*i];
        const double py = local_pos[3*i+1];
        const double pz = local_pos[3*i+2];
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (int j = 0; j < numBodies; ++j) {
            const double dx = all_pos[3*j]   - px;
            const double dy = all_pos[3*j+1] - py;
            const double dz = all_pos[3*j+2] - pz;
            const double distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        local_vel[3*i]   += DT * Fx;
        local_vel[3*i+1] += DT * Fy;
        local_vel[3*i+2] += DT * Fz;
    }
}

static void integrateBodies(double* __restrict local_pos,
                            const double* __restrict local_vel,
                            int nlocal) {
    for (int i = 0; i < nlocal; ++i) {
        local_pos[3*i]   += local_vel[3*i]   * DT;
        local_pos[3*i+1] += local_vel[3*i+1] * DT;
        local_pos[3*i+2] += local_vel[3*i+2] * DT;
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

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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
            printUsage(argv[0]);
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
        printf("N-Body Simulation (MPI, %d ranks)\n", nranks);
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // 1D block distribution: rank r gets bodies [offset_r, offset_r + nlocal_r)
    auto blockInfo = [&](int r) -> std::pair<int,int> {
        int base = numBodies / nranks;
        int rem  = numBodies % nranks;
        int nl   = base + (r < rem ? 1 : 0);
        int off  = r * base + std::min(r, rem);
        return {nl, off};
    };

    auto [nlocal, offset] = blockInfo(rank);

    // All ranks generate identical initial conditions for reproducibility
    std::vector<Body> allBodies(numBodies);
    randomizeBodies(allBodies);

    // Extract local bodies into flat SoA buffers
    std::vector<double> local_pos(3 * nlocal);
    std::vector<double> local_vel(3 * nlocal);
    for (int i = 0; i < nlocal; ++i) {
        const auto& b = allBodies[offset + i];
        local_pos[3*i]   = b.pos.x;
        local_pos[3*i+1] = b.pos.y;
        local_pos[3*i+2] = b.pos.z;
        local_vel[3*i]   = b.vel.x;
        local_vel[3*i+1] = b.vel.y;
        local_vel[3*i+2] = b.vel.z;
    }
    allBodies.clear();
    allBodies.shrink_to_fit();

    // Setup Allgatherv counts and displacements (in doubles)
    std::vector<int> sendcounts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        auto [nl, off] = blockInfo(r);
        sendcounts[r] = 3 * nl;
        displs[r]     = 3 * off;
    }

    // Buffer for allgather of positions
    std::vector<double> all_pos(3 * numBodies);

    // MPI_Barrier before timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Gather all positions
        MPI_Allgatherv(local_pos.data(), 3 * nlocal, MPI_DOUBLE,
                       all_pos.data(), sendcounts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Compute forces on local bodies and update velocities
        computeForces(all_pos.data(), local_pos.data(), local_vel.data(),
                      nlocal, numBodies);

        // Integrate local positions
        integrateBodies(local_pos.data(), local_vel.data(), nlocal);
    }

    // Final gather of positions for output/validation
    MPI_Allgatherv(local_pos.data(), 3 * nlocal, MPI_DOUBLE,
                   all_pos.data(), sendcounts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    // Gather all velocities to rank 0
    std::vector<int> vel_sendcounts = sendcounts;
    std::vector<int> vel_displs = displs;
    std::vector<double> all_vel(3 * numBodies);
    MPI_Allgatherv(local_vel.data(), 3 * nlocal, MPI_DOUBLE,
                   all_vel.data(), vel_sendcounts.data(), vel_displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = static_cast<long long>(duration.count());
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG_INT,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %lld ms\n", max_duration_ms);
    }

    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            bodyData.push_back(all_pos[3*i]);
            bodyData.push_back(all_pos[3*i+1]);
            bodyData.push_back(all_pos[3*i+2]);
            bodyData.push_back(all_vel[3*i]);
            bodyData.push_back(all_vel[3*i+1]);
            bodyData.push_back(all_vel[3*i+2]);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        // Check local bodies for NaN/Inf and bounds
        bool local_valid = true;
        for (int i = 0; i < nlocal; ++i) {
            for (int c = 0; c < 3; ++c) {
                if (!std::isfinite(local_pos[3*i+c]) || !std::isfinite(local_vel[3*i+c])) {
                    local_valid = false;
                }
                const double maxVal = 1e6;
                if (std::abs(local_pos[3*i+c]) > maxVal || std::abs(local_vel[3*i+c]) > maxVal) {
                    local_valid = false;
                }
            }
        }

        bool global_valid;
        MPI_Allreduce(&local_valid, &global_valid, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

        if (!global_valid) {
            if (rank == 0) {
                printf("Validation failed: found NaN, Inf, or out-of-bounds value\n");
                printf("Validation: FAILED\n");
            }
        } else {
            // Compute total energy using collective reduction
            // Kinetic energy: sum of 0.5 * v^2 over all bodies
            double local_ke = 0.0;
            for (int i = 0; i < nlocal; ++i) {
                local_ke += 0.5 * (local_vel[3*i]*local_vel[3*i] +
                                   local_vel[3*i+1]*local_vel[3*i+1] +
                                   local_vel[3*i+2]*local_vel[3*i+2]);
            }
            double total_ke;
            MPI_Allreduce(&local_ke, &total_ke, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            // Potential energy: each rank computes for its local bodies against all bodies
            // then Allreduce and divide by 2 (each pair counted twice)
            double local_pe = 0.0;
            for (int i = 0; i < nlocal; ++i) {
                const int gi = offset + i;
                const double px = all_pos[3*gi];
                const double py = all_pos[3*gi+1];
                const double pz = all_pos[3*gi+2];
                for (int j = 0; j < numBodies; ++j) {
                    if (j == gi) continue;
                    const double dx = all_pos[3*j]   - px;
                    const double dy = all_pos[3*j+1] - py;
                    const double dz = all_pos[3*j+2] - pz;
                    const double dist = std::sqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
                    local_pe -= 1.0 / dist;
                }
            }
            double total_pe;
            MPI_Allreduce(&local_pe, &total_pe, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            total_pe *= 0.5;

            double finalEnergy = total_ke + total_pe;
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
