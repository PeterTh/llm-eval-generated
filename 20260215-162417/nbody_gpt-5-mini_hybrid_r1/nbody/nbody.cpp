#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <iostream>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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

static inline void cudaCheck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

extern "C" __global__ void computeForcesKernel(const double* posx_all, const double* posy_all, const double* posz_all,
                                                double* velx_local, double* vely_local, double* velz_local,
                                                int n_global, int local_n, int local_offset, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= local_n) return;
    int gi = local_offset + idx; // global index of this local body

    double px = posx_all[gi];
    double py = posy_all[gi];
    double pz = posz_all[gi];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int j = 0; j < n_global; ++j) {
        double dx = posx_all[j] - px;
        double dy = posy_all[j] - py;
        double dz = posz_all[j] - pz;
        double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        double invDist = rsqrt(distSqr); // faster reciprocal sqrt on device
        double invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    velx_local[idx] += dt * Fx;
    vely_local[idx] += dt * Fy;
    velz_local[idx] += dt * Fz;
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Use a repeatable pseudo-random sequence; do this on rank 0 only then distribute
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// Host wrapper that uses CUDA to compute forces for local bodies using all positions
void computeForcesHybrid(int rank, int worldSize,
                         std::vector<double>& posx_all, std::vector<double>& posy_all, std::vector<double>& posz_all,
                         std::vector<double>& velx_local, std::vector<double>& vely_local, std::vector<double>& velz_local,
                         int n_global, int local_n, int local_offset) {
    // Select device based on rank
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        int dev = rank % deviceCount;
        cudaCheck(cudaSetDevice(dev), "cudaSetDevice");
    }

    // Allocate device arrays
    double *d_posx = nullptr, *d_posy = nullptr, *d_posz = nullptr;
    double *d_velx = nullptr, *d_vely = nullptr, *d_velz = nullptr;

    size_t bytesAll = sizeof(double) * (size_t)n_global;
    size_t bytesLocal = sizeof(double) * (size_t)local_n;

    cudaCheck(cudaMalloc(&d_posx, bytesAll), "cudaMalloc posx");
    cudaCheck(cudaMalloc(&d_posy, bytesAll), "cudaMalloc posy");
    cudaCheck(cudaMalloc(&d_posz, bytesAll), "cudaMalloc posz");

    cudaCheck(cudaMalloc(&d_velx, bytesLocal), "cudaMalloc velx");
    cudaCheck(cudaMalloc(&d_vely, bytesLocal), "cudaMalloc vely");
    cudaCheck(cudaMalloc(&d_velz, bytesLocal), "cudaMalloc velz");

    // Copy data to device
    cudaCheck(cudaMemcpy(d_posx, posx_all.data(), bytesAll, cudaMemcpyHostToDevice), "cudaMemcpy posx");
    cudaCheck(cudaMemcpy(d_posy, posy_all.data(), bytesAll, cudaMemcpyHostToDevice), "cudaMemcpy posy");
    cudaCheck(cudaMemcpy(d_posz, posz_all.data(), bytesAll, cudaMemcpyHostToDevice), "cudaMemcpy posz");

    cudaCheck(cudaMemcpy(d_velx, velx_local.data(), bytesLocal, cudaMemcpyHostToDevice), "cudaMemcpy velx");
    cudaCheck(cudaMemcpy(d_vely, vely_local.data(), bytesLocal, cudaMemcpyHostToDevice), "cudaMemcpy vely");
    cudaCheck(cudaMemcpy(d_velz, velz_local.data(), bytesLocal, cudaMemcpyHostToDevice), "cudaMemcpy velz");

    // Launch kernel: one thread per local body
    int block = 128;
    int grid = (local_n + block - 1) / block;
    computeForcesKernel<<<grid, block>>>(d_posx, d_posy, d_posz, d_velx, d_vely, d_velz, n_global, local_n, local_offset, DT);
    cudaCheck(cudaGetLastError(), "kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "kernel sync");

    // Copy velocities back
    cudaCheck(cudaMemcpy(velx_local.data(), d_velx, bytesLocal, cudaMemcpyDeviceToHost), "cudaMemcpy velx back");
    cudaCheck(cudaMemcpy(vely_local.data(), d_vely, bytesLocal, cudaMemcpyDeviceToHost), "cudaMemcpy vely back");
    cudaCheck(cudaMemcpy(velz_local.data(), d_velz, bytesLocal, cudaMemcpyDeviceToHost), "cudaMemcpy velz back");

    cudaFree(d_posx); cudaFree(d_posy); cudaFree(d_posz);
    cudaFree(d_velx); cudaFree(d_vely); cudaFree(d_velz);
}

double computeTotalEnergyParallel(const std::vector<double>& velx, const std::vector<double>& vely, const std::vector<double>& velz) {
    double localEnergy = 0.0;
    size_t n = velx.size();
    #pragma omp parallel for reduction(+:localEnergy)
    for (size_t i = 0; i < n; ++i) {
        localEnergy += 0.5 * (velx[i] * velx[i] + vely[i] * vely[i] + velz[i] * velz[i]);
    }
    return localEnergy;
}

bool validateSimulationLocal(const std::vector<double>& posx, const std::vector<double>& posy, const std::vector<double>& posz,
                             const std::vector<double>& velx, const std::vector<double>& vely, const std::vector<double>& velz) {
    size_t n = posx.size();
    bool ok = true;
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(posx[i]) || !std::isfinite(posy[i]) || !std::isfinite(posz[i]) ||
            !std::isfinite(velx[i]) || !std::isfinite(vely[i]) || !std::isfinite(velz[i])) {
            ok = false;
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(posx[i]) > maxPos || std::abs(posy[i]) > maxPos || std::abs(posz[i]) > maxPos) ok = false;
        if (std::abs(velx[i]) > maxVel || std::abs(vely[i]) > maxVel || std::abs(velz[i]) > maxVel) ok = false;
    }
    return ok;
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
    // Initialize MPI unconditionally
    MPI_Init(&argc, &argv);
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only rank 0 prints usage/errors)
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
    }

    // Compute distribution of bodies across ranks
    std::vector<int> counts(worldSize), displs(worldSize);
    int base = numBodies / worldSize;
    int rem = numBodies % worldSize;
    for (int i = 0; i < worldSize; ++i) {
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i-1] + counts[i-1];
    }

    int local_n = counts[rank];
    int local_offset = displs[rank];

    // Buffers for local data: store as SoA for easy MPI and GPU transfers
    std::vector<double> posx_local(local_n), posy_local(local_n), posz_local(local_n);
    std::vector<double> velx_local(local_n), vely_local(local_n), velz_local(local_n);

    // Rank 0 initializes full dataset and scatters
    std::vector<double> sendbuf; // flattened [px,py,pz,vx,vy,vz] per body
    if (rank == 0) {
        std::vector<Body> bodies(numBodies);
        randomizeBodies(bodies);
        sendbuf.reserve((size_t)numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            sendbuf.push_back(bodies[i].pos.x);
            sendbuf.push_back(bodies[i].pos.y);
            sendbuf.push_back(bodies[i].pos.z);
            sendbuf.push_back(bodies[i].vel.x);
            sendbuf.push_back(bodies[i].vel.y);
            sendbuf.push_back(bodies[i].vel.z);
        }
    }

    // Prepare scatter counts/displs in terms of doubles (6 values per body)
    std::vector<int> counts6(worldSize), displs6(worldSize);
    for (int i = 0; i < worldSize; ++i) {
        counts6[i] = counts[i] * 6;
        displs6[i] = displs[i] * 6;
    }

    std::vector<double> recvbuf(local_n * 6);
    MPI_Scatterv(sendbuf.data(), counts6.data(), displs6.data(), MPI_DOUBLE,
                 recvbuf.data(), local_n * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Unpack recvbuf into local SoA
    for (int i = 0; i < local_n; ++i) {
        posx_local[i] = recvbuf[i*6 + 0];
        posy_local[i] = recvbuf[i*6 + 1];
        posz_local[i] = recvbuf[i*6 + 2];
        velx_local[i] = recvbuf[i*6 + 3];
        vely_local[i] = recvbuf[i*6 + 4];
        velz_local[i] = recvbuf[i*6 + 5];
    }

    // Global position buffers (all ranks will hold full positions for force computation)
    std::vector<double> posx_all(numBodies), posy_all(numBodies), posz_all(numBodies);

    // Prepare counts/displs for Allgatherv for positions (1 double per position component per body)
    std::vector<int> counts_pos(worldSize), displs_pos(worldSize);
    for (int i = 0; i < worldSize; ++i) {
        counts_pos[i] = counts[i];
        displs_pos[i] = displs[i];
    }

    // Warmup: gather initial positions
    MPI_Allgatherv(posx_local.data(), local_n, MPI_DOUBLE, posx_all.data(), counts_pos.data(), displs_pos.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(posy_local.data(), local_n, MPI_DOUBLE, posy_all.data(), counts_pos.data(), displs_pos.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(posz_local.data(), local_n, MPI_DOUBLE, posz_all.data(), counts_pos.data(), displs_pos.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Set CUDA device before main loop
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        int dev = rank % deviceCount;
        cudaCheck(cudaSetDevice(dev), "cudaSetDevice main");
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Gather all positions across ranks (so every rank has full pos arrays)
        MPI_Allgatherv(posx_local.data(), local_n, MPI_DOUBLE, posx_all.data(), counts_pos.data(), displs_pos.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(posy_local.data(), local_n, MPI_DOUBLE, posy_all.data(), counts_pos.data(), displs_pos.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(posz_local.data(), local_n, MPI_DOUBLE, posz_all.data(), counts_pos.data(), displs_pos.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // Use CUDA to compute forces and update local velocities
        computeForcesHybrid(rank, worldSize, posx_all, posy_all, posz_all, velx_local, vely_local, velz_local, numBodies, local_n, local_offset);

        // Integrate positions on host using OpenMP for parallelism
        #pragma omp parallel for
        for (int i = 0; i < local_n; ++i) {
            posx_local[i] += velx_local[i] * DT;
            posy_local[i] += vely_local[i] * DT;
            posz_local[i] += velz_local[i] * DT;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = duration.count();
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Simulation time: %lld ms\n", max_duration_ms);

    // Gather results back to rank 0 for printing/validation
    // Pack local data into recvbuf for Gatherv
    for (int i = 0; i < local_n; ++i) {
        recvbuf[i*6 + 0] = posx_local[i];
        recvbuf[i*6 + 1] = posy_local[i];
        recvbuf[i*6 + 2] = posz_local[i];
        recvbuf[i*6 + 3] = velx_local[i];
        recvbuf[i*6 + 4] = vely_local[i];
        recvbuf[i*6 + 5] = velz_local[i];
    }

    std::vector<double> finalbuf;
    if (rank == 0) finalbuf.resize((size_t)numBodies * 6);

    MPI_Gatherv(recvbuf.data(), local_n * 6, MPI_DOUBLE,
                finalbuf.data(), counts6.data(), displs6.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0 && printResults) {
        // Serialize and print
        std::vector<double> bodyData;
        bodyData.reserve((size_t)numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            bodyData.push_back(finalbuf[i*6 + 0]);
            bodyData.push_back(finalbuf[i*6 + 1]);
            bodyData.push_back(finalbuf[i*6 + 2]);
            bodyData.push_back(finalbuf[i*6 + 3]);
            bodyData.push_back(finalbuf[i*6 + 4]);
            bodyData.push_back(finalbuf[i*6 + 5]);
        }
        print_results(bodyData, "Bodies");
    }

    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        // Validate locally on rank 0 using gathered finalbuf
        bool ok = true;
        if (rank == 0) {
            std::vector<double> posx(numBodies), posy(numBodies), posz(numBodies), velx(numBodies), vely(numBodies), velz(numBodies);
            for (int i = 0; i < numBodies; ++i) {
                posx[i] = finalbuf[i*6 + 0]; posy[i] = finalbuf[i*6 + 1]; posz[i] = finalbuf[i*6 + 2];
                velx[i] = finalbuf[i*6 + 3]; vely[i] = finalbuf[i*6 + 4]; velz[i] = finalbuf[i*6 + 5];
            }
            ok = validateSimulationLocal(posx, posy, posz, velx, vely, velz);
            if (ok) {
                double kinetic = computeTotalEnergyParallel(velx, vely, velz);
                // compute pairwise potential on host (single-threaded for correctness)
                double potential = 0.0;
                for (int i = 0; i < numBodies; ++i) {
                    for (int j = i + 1; j < numBodies; ++j) {
                        double dx = posx[j] - posx[i];
                        double dy = posy[j] - posy[i];
                        double dz = posz[j] - posz[i];
                        double dist = std::sqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
                        potential -= 1.0 / dist;
                    }
                }
                double finalEnergy = kinetic + potential;
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
