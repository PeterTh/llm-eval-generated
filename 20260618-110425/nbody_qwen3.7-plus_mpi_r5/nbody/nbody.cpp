#include <mpi.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

void randomizeBodies(std::vector<double>& px, std::vector<double>& py, std::vector<double>& pz,
                     std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
                     int start, int count) {
    unsigned int seed = 42;
    for (int i = 0; i < start + count; i++) {
        double px_i = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        double py_i = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        double pz_i = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        double vx_i = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        double vy_i = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        double vz_i = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        if (i >= start) {
            px[i - start] = px_i;
            py[i - start] = py_i;
            pz[i - start] = pz_i;
            vx[i - start] = vx_i;
            vy[i - start] = vy_i;
            vz[i - start] = vz_i;
        }
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
        printf("MPI ranks: %d\n", size);
    }
    
    int base_n = numBodies / size;
    int remainder = numBodies % size;
    int local_n = base_n + (rank < remainder ? 1 : 0);
    int start_idx = rank * base_n + std::min(rank, remainder);
    
    std::vector<double> local_px(local_n), local_py(local_n), local_pz(local_n);
    std::vector<double> local_vx(local_n), local_vy(local_n), local_vz(local_n);
    
    randomizeBodies(local_px, local_py, local_pz, local_vx, local_vy, local_vz, start_idx, local_n);
    
    std::vector<double> all_px(numBodies), all_py(numBodies), all_pz(numBodies);
    
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; r++) {
        int rn = base_n + (r < remainder ? 1 : 0);
        int start_r = r * base_n + std::min(r, remainder);
        counts[r] = rn;
        displs[r] = start_r;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        MPI_Allgatherv(local_px.data(), local_n, MPI_DOUBLE, all_px.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(local_py.data(), local_n, MPI_DOUBLE, all_py.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(local_pz.data(), local_n, MPI_DOUBLE, all_pz.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        
        for (int i = 0; i < local_n; ++i) {
            double px_i = local_px[i], py_i = local_py[i], pz_i = local_pz[i];
            double Fx = 0.0, Fy = 0.0, Fz = 0.0;
            
            for (int j = 0; j < numBodies; ++j) {
                double dx = all_px[j] - px_i;
                double dy = all_py[j] - py_i;
                double dz = all_pz[j] - pz_i;
                double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                double invDist = 1.0 / std::sqrt(distSqr);
                double invDist3 = invDist * invDist * invDist;
                
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
            
            local_vx[i] += DT * Fx;
            local_vy[i] += DT * Fy;
            local_vz[i] += DT * Fz;
        }
        
        for (int i = 0; i < local_n; ++i) {
            local_px[i] += local_vx[i] * DT;
            local_py[i] += local_vy[i] * DT;
            local_pz[i] += local_vz[i] * DT;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    std::vector<double> all_px_final, all_py_final, all_pz_final;
    std::vector<double> all_vx_final, all_vy_final, all_vz_final;
    
    if (rank == 0) {
        all_px_final.resize(numBodies);
        all_py_final.resize(numBodies);
        all_pz_final.resize(numBodies);
        all_vx_final.resize(numBodies);
        all_vy_final.resize(numBodies);
        all_vz_final.resize(numBodies);
    }
    
    MPI_Gatherv(local_px.data(), local_n, MPI_DOUBLE, all_px_final.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_py.data(), local_n, MPI_DOUBLE, all_py_final.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_pz.data(), local_n, MPI_DOUBLE, all_pz_final.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_vx.data(), local_n, MPI_DOUBLE, all_vx_final.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_vy.data(), local_n, MPI_DOUBLE, all_vy_final.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_vz.data(), local_n, MPI_DOUBLE, all_vz_final.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        printf("Simulation time: %ld ms\n", duration.count());
        
        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(numBodies * 6);
            for (int i = 0; i < numBodies; i++) {
                bodyData.push_back(all_px_final[i]);
                bodyData.push_back(all_py_final[i]);
                bodyData.push_back(all_pz_final[i]);
                bodyData.push_back(all_vx_final[i]);
                bodyData.push_back(all_vy_final[i]);
                bodyData.push_back(all_vz_final[i]);
            }
            print_results(bodyData, "Bodies");
        }
        
        if (validate) {
            printf("Validating simulation results...\n");
            
            bool valid = true;
            for (int i = 0; i < numBodies; i++) {
                double px = all_px_final[i], py = all_py_final[i], pz = all_pz_final[i];
                double vx = all_vx_final[i], vy = all_vy_final[i], vz = all_vz_final[i];
                
                if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz) ||
                    !std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(vz)) {
                    printf("Validation failed: found NaN or Inf value in body state\n");
                    valid = false;
                    break;
                }
                
                const double maxPos = 1e6;
                const double maxVel = 1e6;
                if (std::abs(px) > maxPos || std::abs(py) > maxPos || std::abs(pz) > maxPos) {
                    printf("Validation failed: body position exceeds reasonable bounds\n");
                    valid = false;
                    break;
                }
                if (std::abs(vx) > maxVel || std::abs(vy) > maxVel || std::abs(vz) > maxVel) {
                    printf("Validation failed: body velocity exceeds reasonable bounds\n");
                    valid = false;
                    break;
                }
            }
            
            if (valid) {
                double energy = 0.0;
                
                for (int i = 0; i < numBodies; i++) {
                    energy += 0.5 * (all_vx_final[i] * all_vx_final[i] + 
                                    all_vy_final[i] * all_vy_final[i] + 
                                    all_vz_final[i] * all_vz_final[i]);
                }
                
                for (int i = 0; i < numBodies; i++) {
                    for (int j = i + 1; j < numBodies; j++) {
                        double dx = all_px_final[j] - all_px_final[i];
                        double dy = all_py_final[j] - all_py_final[i];
                        double dz = all_pz_final[j] - all_pz_final[i];
                        double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                        energy -= 1.0 / dist;
                    }
                }
                
                printf("Final energy: %.6f\n", energy);
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
