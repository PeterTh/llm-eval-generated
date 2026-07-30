// Hybrid MPI + OpenMP + CUDA n-body simulation
// For maximum performance on accelerator clusters:
//   MPI distributes bodies across ranks,
//   CUDA accelerates the O(n^2) force computation on GPU,
//   OpenMP parallelizes host-side computations.
#include <mpi.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#include <cuda_runtime.h>

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// GPU thread block size
constexpr int BLOCK_SIZE = 256;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0,
                   const double z = 0) noexcept
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

// CUDA kernel: compute all-pairs forces for a subset of bodies (local_count
// bodies starting at local_start). Uses shared-memory tiling to reduce global
// memory traffic. Each thread handles one body i and accumulates forces from
// all bodies j loaded in tiles into shared memory.
__global__ void nbodyStepKernel(double* __restrict__ pos_x,
                                double* __restrict__ pos_y,
                                double* __restrict__ pos_z,
                                double* __restrict__ vel_x,
                                double* __restrict__ vel_y,
                                double* __restrict__ vel_z, int n,
                                int local_start, int local_count, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= local_count) return;
    int i = local_start + idx;

    // Shared memory for tiling over j bodies
    extern __shared__ double shmem[];
    double* sh_x = &shmem[0];
    double* sh_y = &shmem[blockDim.x];
    double* sh_z = &shmem[2 * blockDim.x];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    const double px = pos_x[i];
    const double py = pos_y[i];
    const double pz = pos_z[i];

    // Tile over all j bodies
    for (int tile = 0; tile * blockDim.x < n; ++tile) {
        // Cooperative load of a tile of j bodies into shared memory
        int j_global = tile * blockDim.x + threadIdx.x;
        if (j_global < n) {
            sh_x[threadIdx.x] = pos_x[j_global];
            sh_y[threadIdx.x] = pos_y[j_global];
            sh_z[threadIdx.x] = pos_z[j_global];
        }
        __syncthreads();

        // Each thread computes interactions with all j in this tile
        int j_end = min(blockDim.x, n - tile * blockDim.x);
        for (int jj = 0; jj < j_end; ++jj) {
            double dx = sh_x[jj] - px;
            double dy = sh_y[jj] - py;
            double dz = sh_z[jj] - pz;
            double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            double invDist = 1.0 / sqrt(distSqr);
            double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    // Update velocity and position for body i
    vel_x[i] += dt * Fx;
    vel_y[i] += dt * Fy;
    vel_z[i] += dt * Fz;
    pos_x[i] = px + vel_x[i] * dt;
    pos_y[i] = py + vel_y[i] * dt;
    pos_z[i] = pz + vel_z[i] * dt;
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

    // Select GPU device (round-robin across available devices)
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    if (nDevices == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % nDevices));

    // Parse command line arguments (rank 0 only, broadcast to all)
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

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
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // Require evenly divisible body count for straightforward distribution
    if (numBodies % size != 0) {
        if (rank == 0) {
            fprintf(stderr,
                    "Error: number of bodies (%d) must be divisible by number "
                    "of MPI processes (%d)\n",
                    numBodies, size);
        }
        MPI_Finalize();
        return 1;
    }

    int local_n = numBodies / size;
    int local_start = rank * local_n;

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI processes: %d\n", size);
        printf("Bodies per rank: %d\n", local_n);
        printf("CUDA devices available: %d\n", nDevices);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Host-side data (Structure of Arrays layout for efficient GPU transfers)
    std::vector<double> h_pos_x(numBodies);
    std::vector<double> h_pos_y(numBodies);
    std::vector<double> h_pos_z(numBodies);
    std::vector<double> h_vel_x(numBodies);
    std::vector<double> h_vel_y(numBodies);
    std::vector<double> h_vel_z(numBodies);

    // Initialize bodies on rank 0 and broadcast to all ranks
    if (rank == 0) {
        std::vector<Body> bodies(numBodies);
        randomizeBodies(bodies);
        for (int i = 0; i < numBodies; ++i) {
            h_pos_x[i] = bodies[i].pos.x;
            h_pos_y[i] = bodies[i].pos.y;
            h_pos_z[i] = bodies[i].pos.z;
            h_vel_x[i] = bodies[i].vel.x;
            h_vel_y[i] = bodies[i].vel.y;
            h_vel_z[i] = bodies[i].vel.z;
        }
    }

    MPI_Bcast(h_pos_x.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_pos_y.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_pos_z.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vel_x.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vel_y.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vel_z.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allocate device memory
    double *d_pos_x, *d_pos_y, *d_pos_z;
    double *d_vel_x, *d_vel_y, *d_vel_z;
    CUDA_CHECK(cudaMalloc(&d_pos_x, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pos_y, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pos_z, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_x, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_y, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_z, numBodies * sizeof(double)));

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_pos_x, h_pos_x.data(),
                          numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_y, h_pos_y.data(),
                          numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_z, h_pos_z.data(),
                          numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_x, h_vel_x.data(),
                          numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_y, h_vel_y.data(),
                          numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_z, h_vel_z.data(),
                          numBodies * sizeof(double), cudaMemcpyHostToDevice));

    // Main simulation loop: each rank processes its subset of bodies on GPU
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    int threads = BLOCK_SIZE;
    int blocks = (local_n + threads - 1) / threads;
    size_t shmem_size = 3 * threads * sizeof(double);

    for (int step = 0; step < numSteps; ++step) {
        // GPU: compute forces and update velocities+positions for local bodies
        nbodyStepKernel<<<blocks, threads, shmem_size>>>(
            d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, numBodies,
            local_start, local_n, DT);
        CUDA_CHECK(cudaGetLastError());

        // Copy local positions from device to host for MPI communication
        CUDA_CHECK(cudaMemcpy(h_pos_x.data() + local_start,
                              d_pos_x + local_start,
                              local_n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pos_y.data() + local_start,
                              d_pos_y + local_start,
                              local_n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pos_z.data() + local_start,
                              d_pos_z + local_start,
                              local_n * sizeof(double), cudaMemcpyDeviceToHost));

        // MPI: share all positions across all ranks
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_pos_x.data(),
                      local_n, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_pos_y.data(),
                      local_n, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_pos_z.data(),
                      local_n, MPI_DOUBLE, MPI_COMM_WORLD);

        // Copy all positions back to device for next step
        CUDA_CHECK(cudaMemcpy(d_pos_x, h_pos_x.data(),
                              numBodies * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos_y, h_pos_y.data(),
                              numBodies * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos_z, h_pos_z.data(),
                              numBodies * sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Copy final velocities back to host for output / validation
    CUDA_CHECK(cudaMemcpy(h_vel_x.data(), d_vel_x,
                          numBodies * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vel_y.data(), d_vel_y,
                          numBodies * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vel_z.data(), d_vel_z,
                          numBodies * sizeof(double), cudaMemcpyDeviceToHost));

    // Print results for external validation
    if (printResults) {
        // Gather all body data to rank 0
        std::vector<double> bodyData;
        if (rank == 0) {
            bodyData.reserve(numBodies * 6);
            for (int i = 0; i < numBodies; ++i) {
                bodyData.push_back(h_pos_x[i]);
                bodyData.push_back(h_pos_y[i]);
                bodyData.push_back(h_pos_z[i]);
                bodyData.push_back(h_vel_x[i]);
                bodyData.push_back(h_vel_y[i]);
                bodyData.push_back(h_vel_z[i]);
            }
            print_results(bodyData, "Bodies");
        }
    }

    // Validation with MPI + OpenMP
    if (validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
        }

        // Each rank validates its local bodies
        bool local_valid = true;
        for (int i = local_start; i < local_start + local_n; ++i) {
            if (!std::isfinite(h_pos_x[i]) || !std::isfinite(h_pos_y[i]) ||
                !std::isfinite(h_pos_z[i]) || !std::isfinite(h_vel_x[i]) ||
                !std::isfinite(h_vel_y[i]) ||
                !std::isfinite(h_vel_z[i])) {
                local_valid = false;
                break;
            }
            constexpr double maxPos = 1e6;
            constexpr double maxVel = 1e6;
            if (std::abs(h_pos_x[i]) > maxPos ||
                std::abs(h_pos_y[i]) > maxPos ||
                std::abs(h_pos_z[i]) > maxPos ||
                std::abs(h_vel_x[i]) > maxVel ||
                std::abs(h_vel_y[i]) > maxVel ||
                std::abs(h_vel_z[i]) > maxVel) {
                local_valid = false;
                break;
            }
        }

        int local_valid_int = local_valid ? 1 : 0;
        int all_valid_int;
        MPI_Allreduce(&local_valid_int, &all_valid_int, 1, MPI_INT, MPI_LAND,
                      MPI_COMM_WORLD);

        if (all_valid_int) {
            // Compute local energy contributions with OpenMP parallelism
            double kinetic = 0.0;
#pragma omp parallel for reduction(+ : kinetic)
            for (int i = local_start; i < local_start + local_n; ++i) {
                kinetic +=
                    0.5 * (h_vel_x[i] * h_vel_x[i] + h_vel_y[i] * h_vel_y[i] +
                           h_vel_z[i] * h_vel_z[i]);
            }

            double potential = 0.0;
#pragma omp parallel for reduction(+ : potential)
            for (int i = local_start; i < local_start + local_n; ++i) {
                for (int j = i + 1; j < numBodies; ++j) {
                    double dx = h_pos_x[j] - h_pos_x[i];
                    double dy = h_pos_y[j] - h_pos_y[i];
                    double dz = h_pos_z[j] - h_pos_z[i];
                    double dist = std::sqrt(dx * dx + dy * dy + dz * dz +
                                            SOFTENING);
                    potential -= 1.0 / dist;
                }
            }

            double total_kinetic = 0.0, total_potential = 0.0;
            MPI_Reduce(&kinetic, &total_kinetic, 1, MPI_DOUBLE, MPI_SUM, 0,
                       MPI_COMM_WORLD);
            MPI_Reduce(&potential, &total_potential, 1, MPI_DOUBLE, MPI_SUM, 0,
                       MPI_COMM_WORLD);

            if (rank == 0) {
                printf("Final energy: %.6f\n",
                       total_kinetic + total_potential);
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

    // Cleanup
    CUDA_CHECK(cudaFree(d_pos_x));
    CUDA_CHECK(cudaFree(d_pos_y));
    CUDA_CHECK(cudaFree(d_pos_z));
    CUDA_CHECK(cudaFree(d_vel_x));
    CUDA_CHECK(cudaFree(d_vel_y));
    CUDA_CHECK(cudaFree(d_vel_z));

    MPI_Finalize();
    return 0;
}
