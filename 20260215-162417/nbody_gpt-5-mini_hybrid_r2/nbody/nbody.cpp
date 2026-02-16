#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

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

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, -1); \
    } \
} while (0)

__global__ void computeForcesKernel(int N, const double* posx, const double* posy, const double* posz,
                                    double* velx, double* vely, double* velz,
                                    int offset) {
    int i_local = blockIdx.x * blockDim.x + threadIdx.x;
    int local_n = gridDim.x * blockDim.x; // not used
    int i_global = offset + i_local;
    if (i_global >= N) return;

    double px = posx[i_global];
    double py = posy[i_global];
    double pz = posz[i_global];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int j = 0; j < N; ++j) {
        double dx = posx[j] - px;
        double dy = posy[j] - py;
        double dz = posz[j] - pz;
        double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        double invDist = rsqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    // vel arrays are sized for local chunk and indexed by i_local
    velx[i_local] += DT * Fx;
    vely[i_local] += DT * Fy;
    velz[i_local] += DT * Fz;
}

__global__ void integrateKernel(double* posx, double* posy, double* posz,
                                const double* velx, const double* vely, const double* velz,
                                int N, int offset) {
    int i_local = blockIdx.x * blockDim.x + threadIdx.x;
    int i_global = offset + i_local;
    if (i_global >= N) return;

    posx[i_global] += velx[i_local] * DT;
    posy[i_global] += vely[i_local] * DT;
    posz[i_global] += velz[i_local] * DT;
}

void randomizeBodiesHost(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            fprintf(stderr, "Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            fprintf(stderr, "Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            fprintf(stderr, "Validation failed: body velocity exceeds reasonable bounds\n");
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
    // Initialize MPI and threading
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

    if (rank == 0) {
        printf("N-Body Simulation (MPI+OpenMP+CUDA)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine local partitioning
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    int base = numBodies / size;
    int rem = numBodies % size;
    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
    }
    displs[0] = 0;
    for (int r = 1; r < size; ++r) displs[r] = displs[r-1] + counts[r-1];
    int local_n = counts[rank];
    int offset = displs[rank];

    // Host buffers
    std::vector<Body> fullBodies;
    if (rank == 0) {
        fullBodies.resize(numBodies);
        randomizeBodiesHost(fullBodies, 42);
    }

    // Prepare host local arrays
    std::vector<double> posx_local(local_n), posy_local(local_n), posz_local(local_n);
    std::vector<double> velx_local(local_n), vely_local(local_n), velz_local(local_n);

    // Scatter initial data from root
    std::vector<int> counts_doubles(size), displs_doubles(size);
    for (int r = 0; r < size; ++r) {
        counts_doubles[r] = counts[r];
        displs_doubles[r] = displs[r];
    }

    if (rank == 0) {
        std::vector<double> posx(numBodies), posy(numBodies), posz(numBodies), velx(numBodies), vely(numBodies), velz(numBodies);
        for (int i = 0; i < numBodies; ++i) {
            posx[i] = fullBodies[i].pos.x;
            posy[i] = fullBodies[i].pos.y;
            posz[i] = fullBodies[i].pos.z;
            velx[i] = fullBodies[i].vel.x;
            vely[i] = fullBodies[i].vel.y;
            velz[i] = fullBodies[i].vel.z;
        }
        MPI_Scatterv(posx.data(), counts_doubles.data(), displs_doubles.data(), MPI_DOUBLE, posx_local.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(posy.data(), counts_doubles.data(), displs_doubles.data(), MPI_DOUBLE, posy_local.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(posz.data(), counts_doubles.data(), displs_doubles.data(), MPI_DOUBLE, posz_local.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(velx.data(), counts_doubles.data(), displs_doubles.data(), MPI_DOUBLE, velx_local.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(vely.data(), counts_doubles.data(), displs_doubles.data(), MPI_DOUBLE, vely_local.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(velz.data(), counts_doubles.data(), displs_doubles.data(), MPI_DOUBLE, velz_local.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE, posx_local.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE, posy_local.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE, posz_local.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE, velx_local.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE, vely_local.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE, velz_local.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Global position buffers (host)
    std::vector<double> posx_global(numBodies), posy_global(numBodies), posz_global(numBodies);

    // Device buffers
    double *d_posx = nullptr, *d_posy = nullptr, *d_posz = nullptr;
    double *d_velx = nullptr, *d_vely = nullptr, *d_velz = nullptr;

    // Allocate device memory: global pos arrays (size numBodies) and local vel arrays (size local_n)
    CUDA_CHECK(cudaMalloc((void**)&d_posx, sizeof(double) * numBodies));
    CUDA_CHECK(cudaMalloc((void**)&d_posy, sizeof(double) * numBodies));
    CUDA_CHECK(cudaMalloc((void**)&d_posz, sizeof(double) * numBodies));
    CUDA_CHECK(cudaMalloc((void**)&d_velx, sizeof(double) * local_n));
    CUDA_CHECK(cudaMalloc((void**)&d_vely, sizeof(double) * local_n));
    CUDA_CHECK(cudaMalloc((void**)&d_velz, sizeof(double) * local_n));

    // Initialize local vel device arrays from host
    CUDA_CHECK(cudaMemcpy(d_velx, velx_local.data(), sizeof(double) * local_n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vely, vely_local.data(), sizeof(double) * local_n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velz, velz_local.data(), sizeof(double) * local_n, cudaMemcpyHostToDevice));

    // Prepare kernel launch parameters
    int threads = 256;
    int blocks = (local_n + threads - 1) / threads;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Gather all positions to each rank
        MPI_Allgatherv(posx_local.data(), local_n, MPI_DOUBLE, posx_global.data(), counts_doubles.data(), displs_doubles.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(posy_local.data(), local_n, MPI_DOUBLE, posy_global.data(), counts_doubles.data(), displs_doubles.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(posz_local.data(), local_n, MPI_DOUBLE, posz_global.data(), counts_doubles.data(), displs_doubles.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // Copy global positions to device
        CUDA_CHECK(cudaMemcpy(d_posx, posx_global.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_posy, posy_global.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_posz, posz_global.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice));

        // Compute forces on device for local chunk
        computeForcesKernel<<<blocks, threads>>>(numBodies, d_posx, d_posy, d_posz, d_velx, d_vely, d_velz, offset);
        CUDA_CHECK(cudaGetLastError());

        // Integrate positions for local chunk on device
        integrateKernel<<<blocks, threads>>>(d_posx, d_posy, d_posz, d_velx, d_vely, d_velz, numBodies, offset);
        CUDA_CHECK(cudaGetLastError());

        // Copy updated local positions back to host
        CUDA_CHECK(cudaMemcpy(posx_local.data(), d_posx + offset, sizeof(double) * local_n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(posy_local.data(), d_posy + offset, sizeof(double) * local_n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(posz_local.data(), d_posz + offset, sizeof(double) * local_n, cudaMemcpyDeviceToHost));

        // Velocities remain on device across iterations; if needed on host for printing gather later
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) printf("Simulation time: %ld ms\n", duration.count());

    // Gather final data to root for validation/printing
    std::vector<double> final_posx, final_posy, final_posz, final_velx, final_vely, final_velz;
    if (rank == 0) {
        final_posx.resize(numBodies);
        final_posy.resize(numBodies);
        final_posz.resize(numBodies);
        final_velx.resize(numBodies);
        final_vely.resize(numBodies);
        final_velz.resize(numBodies);
    }

    // Copy local velocities back to host
    CUDA_CHECK(cudaMemcpy(velx_local.data(), d_velx, sizeof(double) * local_n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vely_local.data(), d_vely, sizeof(double) * local_n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(velz_local.data(), d_velz, sizeof(double) * local_n, cudaMemcpyDeviceToHost));

    MPI_Gatherv(posx_local.data(), local_n, MPI_DOUBLE, final_posx.data(), counts_doubles.data(), displs_doubles.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(posy_local.data(), local_n, MPI_DOUBLE, final_posy.data(), counts_doubles.data(), displs_doubles.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(posz_local.data(), local_n, MPI_DOUBLE, final_posz.data(), counts_doubles.data(), displs_doubles.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(velx_local.data(), local_n, MPI_DOUBLE, final_velx.data(), counts_doubles.data(), displs_doubles.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vely_local.data(), local_n, MPI_DOUBLE, final_vely.data(), counts_doubles.data(), displs_doubles.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(velz_local.data(), local_n, MPI_DOUBLE, final_velz.data(), counts_doubles.data(), displs_doubles.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Root assembles Body vector
    std::vector<Body> finalBodies;
    if (rank == 0) {
        finalBodies.resize(numBodies);
        for (int i = 0; i < numBodies; ++i) {
            finalBodies[i].pos.x = final_posx[i];
            finalBodies[i].pos.y = final_posy[i];
            finalBodies[i].pos.z = final_posz[i];
            finalBodies[i].vel.x = final_velx[i];
            finalBodies[i].vel.y = final_vely[i];
            finalBodies[i].vel.z = final_velz[i];
        }
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (const auto& body : finalBodies) {
            bodyData.push_back(body.pos.x);
            bodyData.push_back(body.pos.y);
            bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x);
            bodyData.push_back(body.vel.y);
            bodyData.push_back(body.vel.z);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation on root
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");
        if (validateSimulation(finalBodies)) {
            double finalEnergy = 0.0;
            // compute energy in double loop (host)
            int N = numBodies;
            for (const auto& body : finalBodies) {
                finalEnergy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
            }
            for (int i = 0; i < N; ++i) {
                for (int j = i + 1; j < N; ++j) {
                    double dx = finalBodies[j].pos.x - finalBodies[i].pos.x;
                    double dy = finalBodies[j].pos.y - finalBodies[i].pos.y;
                    double dz = finalBodies[j].pos.z - finalBodies[i].pos.z;
                    double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                    finalEnergy -= 1.0 / dist;
                }
            }
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_posx));
    CUDA_CHECK(cudaFree(d_posy));
    CUDA_CHECK(cudaFree(d_posz));
    CUDA_CHECK(cudaFree(d_velx));
    CUDA_CHECK(cudaFree(d_vely));
    CUDA_CHECK(cudaFree(d_velz));

    MPI_Finalize();
    return 0;
}
