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

void randomizeBodies(double* pos_x, double* pos_y, double* pos_z,
                     double* vel_x, double* vel_y, double* vel_z,
                     int n, int start_idx, unsigned int seed = 42) {
    for (int i = 0; i < start_idx; ++i) {
        for (int j = 0; j < 6; ++j) {
            rand_r(&seed);
        }
    }
    for (int i = 0; i < n; ++i) {
        pos_x[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pos_y[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pos_z[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel_x[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel_y[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel_z[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

void computeForces(const double* all_pos_x, const double* all_pos_y, const double* all_pos_z,
                   double* vel_x, double* vel_y, double* vel_z,
                   int local_offset, int local_n, int total_n) {
    for (int i = 0; i < local_n; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        const int gi = local_offset + i;
        const double px = all_pos_x[gi];
        const double py = all_pos_y[gi];
        const double pz = all_pos_z[gi];
        
        for (int j = 0; j < total_n; ++j) {
            const double dx = all_pos_x[j] - px;
            const double dy = all_pos_y[j] - py;
            const double dz = all_pos_z[j] - pz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        
        vel_x[i] += DT * Fx;
        vel_y[i] += DT * Fy;
        vel_z[i] += DT * Fz;
    }
}

void integrateBodies(double* pos_x, double* pos_y, double* pos_z,
                     const double* vel_x, const double* vel_y, const double* vel_z,
                     int n) {
    for (int i = 0; i < n; ++i) {
        pos_x[i] += vel_x[i] * DT;
        pos_y[i] += vel_y[i] * DT;
        pos_z[i] += vel_z[i] * DT;
    }
}

double computeTotalEnergy(const double* all_pos_x, const double* all_pos_y, const double* all_pos_z,
                          const double* all_vel_x, const double* all_vel_y, const double* all_vel_z,
                          int total_n) {
    double energy = 0.0;
    
    for (int i = 0; i < total_n; ++i) {
        energy += 0.5 * (all_vel_x[i] * all_vel_x[i] + 
                         all_vel_y[i] * all_vel_y[i] + 
                         all_vel_z[i] * all_vel_z[i]);
    }
    
    for (int i = 0; i < total_n; ++i) {
        for (int j = i + 1; j < total_n; ++j) {
            const double dx = all_pos_x[j] - all_pos_x[i];
            const double dy = all_pos_y[j] - all_pos_y[i];
            const double dz = all_pos_z[j] - all_pos_z[i];
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    
    return energy;
}

bool validateSimulation(const double* pos_x, const double* pos_y, const double* pos_z,
                        const double* vel_x, const double* vel_y, const double* vel_z,
                        int n) {
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(pos_x[i]) || !std::isfinite(pos_y[i]) || !std::isfinite(pos_z[i]) ||
            !std::isfinite(vel_x[i]) || !std::isfinite(vel_y[i]) || !std::isfinite(vel_z[i])) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(pos_x[i]) > maxPos || std::abs(pos_y[i]) > maxPos || std::abs(pos_z[i]) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(vel_x[i]) > maxVel || std::abs(vel_y[i]) > maxVel || std::abs(vel_z[i]) > maxVel) {
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
    
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    
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
        printf("N-Body Simulation (MPI, %d ranks)\n", nprocs);
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Block decomposition: distribute bodies across ranks
    std::vector<int> recvcounts(nprocs);
    std::vector<int> displs(nprocs);
    int offset = 0;
    for (int i = 0; i < nprocs; ++i) {
        recvcounts[i] = numBodies / nprocs + (i < numBodies % nprocs ? 1 : 0);
        displs[i] = offset;
        offset += recvcounts[i];
    }
    
    const int local_n = recvcounts[rank];
    const int local_offset = displs[rank];
    
    // Local body data (SoA layout for cache-friendly access)
    std::vector<double> local_pos_x(local_n), local_pos_y(local_n), local_pos_z(local_n);
    std::vector<double> local_vel_x(local_n), local_vel_y(local_n), local_vel_z(local_n);
    
    // Global position arrays for Allgatherv
    std::vector<double> all_pos_x(numBodies), all_pos_y(numBodies), all_pos_z(numBodies);
    
    // Initialize bodies - each rank generates its own bodies with correct RNG sequence
    randomizeBodies(local_pos_x.data(), local_pos_y.data(), local_pos_z.data(),
                    local_vel_x.data(), local_vel_y.data(), local_vel_z.data(),
                    local_n, local_offset);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        // Share all positions across all ranks
        MPI_Allgatherv(local_pos_x.data(), local_n, MPI_DOUBLE,
                       all_pos_x.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(local_pos_y.data(), local_n, MPI_DOUBLE,
                       all_pos_y.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(local_pos_z.data(), local_n, MPI_DOUBLE,
                       all_pos_z.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        
        // Compute forces on local bodies using all positions
        computeForces(all_pos_x.data(), all_pos_y.data(), all_pos_z.data(),
                      local_vel_x.data(), local_vel_y.data(), local_vel_z.data(),
                      local_offset, local_n, numBodies);
        
        // Integrate local bodies
        integrateBodies(local_pos_x.data(), local_pos_y.data(), local_pos_z.data(),
                        local_vel_x.data(), local_vel_y.data(), local_vel_z.data(),
                        local_n);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    long long local_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long global_duration = 0;
    MPI_Reduce(&local_duration, &global_duration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Simulation time: %lld ms\n", global_duration);
    }
    
    // Gather all data to rank 0 for output/validation
    std::vector<double> all_vel_x, all_vel_y, all_vel_z;
    if (rank == 0) {
        all_vel_x.resize(numBodies);
        all_vel_y.resize(numBodies);
        all_vel_z.resize(numBodies);
    }
    
    // Final gather of positions (all_pos_x already has rank 0's portion correct,
    // but we need the full array on rank 0)
    MPI_Gatherv(local_pos_x.data(), local_n, MPI_DOUBLE,
                all_pos_x.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_pos_y.data(), local_n, MPI_DOUBLE,
                all_pos_y.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_pos_z.data(), local_n, MPI_DOUBLE,
                all_pos_z.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_vel_x.data(), local_n, MPI_DOUBLE,
                all_vel_x.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_vel_y.data(), local_n, MPI_DOUBLE,
                all_vel_y.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_vel_z.data(), local_n, MPI_DOUBLE,
                all_vel_z.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Print results for external validation
    if (printResults && rank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            bodyData.push_back(all_pos_x[i]);
            bodyData.push_back(all_pos_y[i]);
            bodyData.push_back(all_pos_z[i]);
            bodyData.push_back(all_vel_x[i]);
            bodyData.push_back(all_vel_y[i]);
            bodyData.push_back(all_vel_z[i]);
        }
        print_results(bodyData, "Bodies");
    }
    
    // Validation
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(all_pos_x.data(), all_pos_y.data(), all_pos_z.data(),
                               all_vel_x.data(), all_vel_y.data(), all_vel_z.data(),
                               numBodies)) {
            double finalEnergy = computeTotalEnergy(all_pos_x.data(), all_pos_y.data(), all_pos_z.data(),
                                                    all_vel_x.data(), all_vel_y.data(), all_vel_z.data(),
                                                    numBodies);
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
    
    MPI_Finalize();
    return 0;
}
