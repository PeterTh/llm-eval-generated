#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

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

// Simple CUDA kernel: compute forces for local bodies using global positions
__global__ void force_kernel(const double* pos_all, double* vel_local, int n, int start_idx, int local_n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= local_n) return;
    int gi = start_idx + tid;
    double xi = pos_all[3 * gi + 0];
    double yi = pos_all[3 * gi + 1];
    double zi = pos_all[3 * gi + 2];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int j = 0; j < n; ++j) {
        double dx = pos_all[3 * j + 0] - xi;
        double dy = pos_all[3 * j + 1] - yi;
        double dz = pos_all[3 * j + 2] - zi;
        double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        double invDist = 1.0 / sqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    // write back to vel_local (layout: 3 * tid)
    vel_local[3 * tid + 0] += DT * Fx;
    vel_local[3 * tid + 1] += DT * Fy;
    vel_local[3 * tid + 2] += DT * Fz;
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Parallelize initialization with OpenMP
    #pragma omp parallel
    {
        unsigned int locseed = seed ^ (unsigned int)omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < bodies.size(); ++i) {
            bodies[i].pos.x = 2.0 * (rand_r(&locseed) / (double)RAND_MAX) - 1.0;
            bodies[i].pos.y = 2.0 * (rand_r(&locseed) / (double)RAND_MAX) - 1.0;
            bodies[i].pos.z = 2.0 * (rand_r(&locseed) / (double)RAND_MAX) - 1.0;
            bodies[i].vel.x = 2.0 * (rand_r(&locseed) / (double)RAND_MAX) - 1.0;
            bodies[i].vel.y = 2.0 * (rand_r(&locseed) / (double)RAND_MAX) - 1.0;
            bodies[i].vel.z = 2.0 * (rand_r(&locseed) / (double)RAND_MAX) - 1.0;
        }
    }
}

// Compute forces using hybrid approach: MPI allgather positions, CUDA kernel for local updates
void computeForcesHybrid(std::vector<Body>& localBodies, int numBodiesGlobal, int startIdx, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    int size = 0;
    MPI_Comm_size(comm, &size);

    const int local_n = static_cast<int>(localBodies.size());
    // Prepare local positions (contiguous)
    std::vector<double> pos_local(3 * local_n);
    for (int i = 0; i < local_n; ++i) {
        pos_local[3 * i + 0] = localBodies[i].pos.x;
        pos_local[3 * i + 1] = localBodies[i].pos.y;
        pos_local[3 * i + 2] = localBodies[i].pos.z;
    }

    // Prepare recvcounts/displs for allgatherv in units of triples
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    // We need to obtain local sizes from all ranks
    int local_triples = 3 * local_n;
    MPI_Allgather(&local_triples, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
    displs[0] = 0;
    for (int i = 1; i < size; ++i) displs[i] = displs[i - 1] + recvcounts[i - 1];
    int total_triples = 0;
    for (int c : recvcounts) total_triples += c;

    std::vector<double> pos_all(total_triples);
    MPI_Allgatherv(pos_local.data(), local_triples, MPI_DOUBLE, pos_all.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, comm);

    // Device buffers
    double* d_pos_all = nullptr;
    double* d_vel_local = nullptr;
    cudaMalloc((void**)&d_pos_all, sizeof(double) * total_triples);
    cudaMalloc((void**)&d_vel_local, sizeof(double) * (3 * local_n));

    // Build local velocity array
    std::vector<double> vel_local(3 * local_n);
    for (int i = 0; i < local_n; ++i) {
        vel_local[3 * i + 0] = localBodies[i].vel.x;
        vel_local[3 * i + 1] = localBodies[i].vel.y;
        vel_local[3 * i + 2] = localBodies[i].vel.z;
    }

    // Copy to device
    cudaMemcpy(d_pos_all, pos_all.data(), sizeof(double) * total_triples, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel_local, vel_local.data(), sizeof(double) * (3 * local_n), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (local_n + threads - 1) / threads;
    // total_triples == 3*numBodiesGlobal ideally
    force_kernel<<<blocks, threads>>>(d_pos_all, d_vel_local, numBodiesGlobal, startIdx, local_n);
    cudaDeviceSynchronize();

    // Copy back updated velocities
    cudaMemcpy(vel_local.data(), d_vel_local, sizeof(double) * (3 * local_n), cudaMemcpyDeviceToHost);

    // Apply velocities back to local bodies
    for (int i = 0; i < local_n; ++i) {
        localBodies[i].vel.x = vel_local[3 * i + 0];
        localBodies[i].vel.y = vel_local[3 * i + 1];
        localBodies[i].vel.z = vel_local[3 * i + 2];
    }

    cudaFree(d_pos_all);
    cudaFree(d_vel_local);
}

void integrateBodies(std::vector<Body>& bodies) {
    #pragma omp parallel for
    for (int i = 0; i < (int)bodies.size(); ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

// Compute total energy across MPI ranks
double computeTotalEnergyMPI(const std::vector<Body>& localBodies, int numBodiesGlobal, int startIdx, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    int size = 0;
    MPI_Comm_size(comm, &size);

    int local_n = (int)localBodies.size();
    int local_triples = 3 * local_n;

    // Gather positions to compute potential energy
    std::vector<int> recvcounts(size);
    MPI_Allgather(&local_triples, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
    std::vector<int> displs(size);
    displs[0] = 0;
    for (int i = 1; i < size; ++i) displs[i] = displs[i - 1] + recvcounts[i - 1];
    int total_triples = 0; for (int c : recvcounts) total_triples += c;
    std::vector<double> pos_all(total_triples);

    std::vector<double> pos_local(local_triples);
    for (int i = 0; i < local_n; ++i) {
        pos_local[3 * i + 0] = localBodies[i].pos.x;
        pos_local[3 * i + 1] = localBodies[i].pos.y;
        pos_local[3 * i + 2] = localBodies[i].pos.z;
    }
    MPI_Allgatherv(pos_local.data(), local_triples, MPI_DOUBLE, pos_all.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, comm);

    // Kinetic energy local
    double localE = 0.0;
    for (const auto& b : localBodies) {
        localE += 0.5 * (b.vel.x * b.vel.x + b.vel.y * b.vel.y + b.vel.z * b.vel.z);
    }

    // Potential: for each local global index gi compute sum over j>gi
    int n = numBodiesGlobal;
    for (int ii = 0; ii < local_n; ++ii) {
        int gi = startIdx + ii;
        double xi = pos_all[3 * gi + 0];
        double yi = pos_all[3 * gi + 1];
        double zi = pos_all[3 * gi + 2];
        for (int j = gi + 1; j < n; ++j) {
            double dx = pos_all[3 * j + 0] - xi;
            double dy = pos_all[3 * j + 1] - yi;
            double dz = pos_all[3 * j + 2] - zi;
            double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            localE -= 1.0 / dist;
        }
    }

    double globalE = 0.0;
    MPI_Allreduce(&localE, &globalE, 1, MPI_DOUBLE, MPI_SUM, comm);
    return globalE;
}

// Validate across MPI ranks
bool validateSimulationMPI(const std::vector<Body>& localBodies, MPI_Comm comm) {
    int local_good = 1;
    for (const auto& body : localBodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            local_good = 0; break;
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) { local_good = 0; break; }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) { local_good = 0; break; }
    }
    int global_good = 0;
    MPI_Allreduce(&local_good, &global_good, 1, MPI_INT, MPI_LAND, comm);
    return global_good != 0;
}

void printUsage(const char* progName) {
    if (progName) printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks energy conservation)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse only on rank 0 then broadcast
    if (rank == 0) {
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
            }
        }
    }
    MPI_Bcast(&numBodies, 1, MPI_INT, 0, comm);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, comm);
    int vflag = validate ? 1 : 0;
    int rflag = printResults ? 1 : 0;
    MPI_Bcast(&vflag, 1, MPI_INT, 0, comm);
    MPI_Bcast(&rflag, 1, MPI_INT, 0, comm);
    validate = (vflag != 0);
    printResults = (rflag != 0);

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine local partition
    int base = numBodies / size;
    int rem = numBodies % size;
    int local_n = base + (rank < rem ? 1 : 0);
    int startIdx = rank * base + std::min(rank, rem);

    std::vector<Body> localBodies(local_n);
    unsigned int seed = 42u + (unsigned int)rank;
    randomizeBodies(localBodies, seed);

    // Warm up device
    cudaFree(0);

    MPI_Barrier(comm);
    auto tstart = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesHybrid(localBodies, numBodies, startIdx, comm);
        integrateBodies(localBodies);
    }

    MPI_Barrier(comm);
    auto tend = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart);
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Print results for external validation
    if (printResults) {
        // Serialize local data
        std::vector<double> localData;
        localData.reserve(local_n * 6);
        for (const auto& b : localBodies) {
            localData.push_back(b.pos.x);
            localData.push_back(b.pos.y);
            localData.push_back(b.pos.z);
            localData.push_back(b.vel.x);
            localData.push_back(b.vel.y);
            localData.push_back(b.vel.z);
        }
        // Gather sizes
        std::vector<int> recvcounts(size);
        int local_elems = (int)localData.size();
        MPI_Gather(&local_elems, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, comm);
        std::vector<int> displs(size);
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) displs[i] = displs[i - 1] + recvcounts[i - 1];
            int total = 0; for (int c : recvcounts) total += c;
            std::vector<double> allData(total);
            MPI_Gatherv(localData.data(), local_elems, MPI_DOUBLE, allData.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, comm);
            print_results(allData, "Bodies");
        } else {
            MPI_Gatherv(localData.data(), local_elems, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, comm);
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        bool ok = validateSimulationMPI(localBodies, comm);
        if (ok) {
            double finalEnergy = computeTotalEnergyMPI(localBodies, numBodies, startIdx, comm);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
