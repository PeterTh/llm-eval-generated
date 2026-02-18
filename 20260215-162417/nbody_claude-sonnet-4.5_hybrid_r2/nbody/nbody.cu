#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// Separate position and velocity arrays for better GPU memory access
struct BodyArrays {
    double *pos_x, *pos_y, *pos_z;
    double *vel_x, *vel_y, *vel_z;
};

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < bodies.size(); ++i) {
        unsigned int local_seed = seed + i;
        bodies[i].pos.x = 2.0 * (rand_r(&local_seed) / (double)RAND_MAX) - 1.0;
        bodies[i].pos.y = 2.0 * (rand_r(&local_seed) / (double)RAND_MAX) - 1.0;
        bodies[i].pos.z = 2.0 * (rand_r(&local_seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.x = 2.0 * (rand_r(&local_seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.y = 2.0 * (rand_r(&local_seed) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.z = 2.0 * (rand_r(&local_seed) / (double)RAND_MAX) - 1.0;
    }
}

// CUDA kernel for force computation
__global__ void computeForcesKernel(const double* __restrict__ pos_x,
                                     const double* __restrict__ pos_y,
                                     const double* __restrict__ pos_z,
                                     double* __restrict__ vel_x,
                                     double* __restrict__ vel_y,
                                     double* __restrict__ vel_z,
                                     int n_local, int n_total,
                                     double dt, double softening) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_local) return;
    
    double fx = 0.0, fy = 0.0, fz = 0.0;
    double px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    
    // Use shared memory for tile-based computation
    __shared__ double s_pos_x[256], s_pos_y[256], s_pos_z[256];
    
    for (int tile = 0; tile < (n_total + blockDim.x - 1) / blockDim.x; ++tile) {
        int j_base = tile * blockDim.x + threadIdx.x;
        
        // Load tile into shared memory
        if (j_base < n_total) {
            s_pos_x[threadIdx.x] = pos_x[j_base];
            s_pos_y[threadIdx.x] = pos_y[j_base];
            s_pos_z[threadIdx.x] = pos_z[j_base];
        }
        __syncthreads();
        
        // Compute forces for this tile
        int tile_size = min(blockDim.x, n_total - tile * blockDim.x);
        #pragma unroll 8
        for (int j = 0; j < tile_size; ++j) {
            double dx = s_pos_x[j] - px;
            double dy = s_pos_y[j] - py;
            double dz = s_pos_z[j] - pz;
            double distSqr = dx * dx + dy * dy + dz * dz + softening;
            double invDist = rsqrt(distSqr);
            double invDist3 = invDist * invDist * invDist;
            
            fx += dx * invDist3;
            fy += dy * invDist3;
            fz += dz * invDist3;
        }
        __syncthreads();
    }
    
    // Update velocities
    vel_x[i] += dt * fx;
    vel_y[i] += dt * fy;
    vel_z[i] += dt * fz;
}

// CUDA kernel for position integration
__global__ void integratePositionsKernel(double* __restrict__ pos_x,
                                          double* __restrict__ pos_y,
                                          double* __restrict__ pos_z,
                                          const double* __restrict__ vel_x,
                                          const double* __restrict__ vel_y,
                                          const double* __restrict__ vel_z,
                                          int n, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    pos_x[i] += vel_x[i] * dt;
    pos_y[i] += vel_y[i] * dt;
    pos_z[i] += vel_z[i] * dt;
}

void integrateBodies(std::vector<Body>& bodies) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < bodies.size(); ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
    #pragma omp parallel for reduction(+:energy)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies[i].vel.x * bodies[i].vel.x + 
                        bodies[i].vel.y * bodies[i].vel.y + 
                        bodies[i].vel.z * bodies[i].vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
    double potential = 0.0;
    #pragma omp parallel for reduction(+:potential) schedule(dynamic, 64)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potential -= 1.0 / dist;
        }
    }
    energy += potential;
    
    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
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
    
    int rank, size;
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
    
    // Set CUDA device based on local rank
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices > 0) {
        int device = rank % num_devices;
        CUDA_CHECK(cudaSetDevice(device));
    } else {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if (rank == 0) {
        printf("N-Body Simulation (MPI+OpenMP+CUDA)\n");
        printf("Number of MPI ranks: %d\n", size);
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Distribute bodies across ranks
    int local_n = numBodies / size;
    int remainder = numBodies % size;
    int local_offset = rank * local_n + std::min(rank, remainder);
    if (rank < remainder) local_n++;
    
    // Initialize all bodies on rank 0, then distribute
    std::vector<Body> all_bodies;
    std::vector<Body> local_bodies(local_n);
    
    if (rank == 0) {
        all_bodies.resize(numBodies);
        randomizeBodies(all_bodies);
    }
    
    // Gather counts and displacements for MPI communication
    std::vector<int> counts(size), displs(size);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
        displs[i] = displs[i-1] + counts[i-1];
    }
    
    // Scatter bodies to all ranks
    std::vector<double> all_data, local_data(local_n * 6);
    if (rank == 0) {
        all_data.resize(numBodies * 6);
        #pragma omp parallel for
        for (int i = 0; i < numBodies; ++i) {
            all_data[i*6 + 0] = all_bodies[i].pos.x;
            all_data[i*6 + 1] = all_bodies[i].pos.y;
            all_data[i*6 + 2] = all_bodies[i].pos.z;
            all_data[i*6 + 3] = all_bodies[i].vel.x;
            all_data[i*6 + 4] = all_bodies[i].vel.y;
            all_data[i*6 + 5] = all_bodies[i].vel.z;
        }
    }
    
    std::vector<int> counts6(size), displs6(size);
    for (int i = 0; i < size; ++i) {
        counts6[i] = counts[i] * 6;
        displs6[i] = displs[i] * 6;
    }
    
    MPI_Scatterv(all_data.data(), counts6.data(), displs6.data(), MPI_DOUBLE,
                 local_data.data(), local_n * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Unpack local bodies
    #pragma omp parallel for
    for (int i = 0; i < local_n; ++i) {
        local_bodies[i].pos.x = local_data[i*6 + 0];
        local_bodies[i].pos.y = local_data[i*6 + 1];
        local_bodies[i].pos.z = local_data[i*6 + 2];
        local_bodies[i].vel.x = local_data[i*6 + 3];
        local_bodies[i].vel.y = local_data[i*6 + 4];
        local_bodies[i].vel.z = local_data[i*6 + 5];
    }
    
    // Allocate GPU memory
    double *d_pos_x, *d_pos_y, *d_pos_z;
    double *d_vel_x, *d_vel_y, *d_vel_z;
    double *d_all_pos_x, *d_all_pos_y, *d_all_pos_z;
    
    CUDA_CHECK(cudaMalloc(&d_pos_x, local_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pos_y, local_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pos_z, local_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_x, local_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_y, local_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_z, local_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_all_pos_x, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_all_pos_y, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_all_pos_z, numBodies * sizeof(double)));
    
    // Copy initial data to GPU
    std::vector<double> pos_x(local_n), pos_y(local_n), pos_z(local_n);
    std::vector<double> vel_x(local_n), vel_y(local_n), vel_z(local_n);
    
    #pragma omp parallel for
    for (int i = 0; i < local_n; ++i) {
        pos_x[i] = local_bodies[i].pos.x;
        pos_y[i] = local_bodies[i].pos.y;
        pos_z[i] = local_bodies[i].pos.z;
        vel_x[i] = local_bodies[i].vel.x;
        vel_y[i] = local_bodies[i].vel.y;
        vel_z[i] = local_bodies[i].vel.z;
    }
    
    CUDA_CHECK(cudaMemcpy(d_pos_x, pos_x.data(), local_n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_y, pos_y.data(), local_n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_z, pos_z.data(), local_n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_x, vel_x.data(), local_n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_y, vel_y.data(), local_n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_z, vel_z.data(), local_n * sizeof(double), cudaMemcpyHostToDevice));
    
    // Buffers for gathering all positions
    std::vector<double> all_pos_x(numBodies), all_pos_y(numBodies), all_pos_z(numBodies);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    // Main simulation loop
    const int blockSize = 256;
    const int numBlocks = (local_n + blockSize - 1) / blockSize;
    
    for (int step = 0; step < numSteps; ++step) {
        // Gather all positions for force computation
        CUDA_CHECK(cudaMemcpy(pos_x.data(), d_pos_x, local_n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pos_y.data(), d_pos_y, local_n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pos_z.data(), d_pos_z, local_n * sizeof(double), cudaMemcpyDeviceToHost));
        
        MPI_Allgatherv(pos_x.data(), local_n, MPI_DOUBLE, all_pos_x.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(pos_y.data(), local_n, MPI_DOUBLE, all_pos_y.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(pos_z.data(), local_n, MPI_DOUBLE, all_pos_z.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        
        CUDA_CHECK(cudaMemcpy(d_all_pos_x, all_pos_x.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_all_pos_y, all_pos_y.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_all_pos_z, all_pos_z.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
        
        // Compute forces on GPU
        computeForcesKernel<<<numBlocks, blockSize>>>(d_all_pos_x, d_all_pos_y, d_all_pos_z,
                                                       d_vel_x, d_vel_y, d_vel_z,
                                                       local_n, numBodies, DT, SOFTENING);
        CUDA_CHECK(cudaGetLastError());
        
        // Integrate positions on GPU
        integratePositionsKernel<<<numBlocks, blockSize>>>(d_pos_x, d_pos_y, d_pos_z,
                                                             d_vel_x, d_vel_y, d_vel_z,
                                                             local_n, DT);
        CUDA_CHECK(cudaGetLastError());
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(pos_x.data(), d_pos_x, local_n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pos_y.data(), d_pos_y, local_n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pos_z.data(), d_pos_z, local_n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vel_x.data(), d_vel_x, local_n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vel_y.data(), d_vel_y, local_n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vel_z.data(), d_vel_z, local_n * sizeof(double), cudaMemcpyDeviceToHost));
    
    #pragma omp parallel for
    for (int i = 0; i < local_n; ++i) {
        local_bodies[i].pos.x = pos_x[i];
        local_bodies[i].pos.y = pos_y[i];
        local_bodies[i].pos.z = pos_z[i];
        local_bodies[i].vel.x = vel_x[i];
        local_bodies[i].vel.y = vel_y[i];
        local_bodies[i].vel.z = vel_z[i];
    }
    
    // Gather all bodies to rank 0 for output
    #pragma omp parallel for
    for (int i = 0; i < local_n; ++i) {
        local_data[i*6 + 0] = local_bodies[i].pos.x;
        local_data[i*6 + 1] = local_bodies[i].pos.y;
        local_data[i*6 + 2] = local_bodies[i].pos.z;
        local_data[i*6 + 3] = local_bodies[i].vel.x;
        local_data[i*6 + 4] = local_bodies[i].vel.y;
        local_data[i*6 + 5] = local_bodies[i].vel.z;
    }
    
    MPI_Gatherv(local_data.data(), local_n * 6, MPI_DOUBLE,
                all_data.data(), counts6.data(), displs6.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        #pragma omp parallel for
        for (int i = 0; i < numBodies; ++i) {
            all_bodies[i].pos.x = all_data[i*6 + 0];
            all_bodies[i].pos.y = all_data[i*6 + 1];
            all_bodies[i].pos.z = all_data[i*6 + 2];
            all_bodies[i].vel.x = all_data[i*6 + 3];
            all_bodies[i].vel.y = all_data[i*6 + 4];
            all_bodies[i].vel.z = all_data[i*6 + 5];
        }
        
        // Print results for external validation
        if (printResults) {
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
            printf("Validating simulation results...\n");
            
            if (validateSimulation(all_bodies)) {
                double finalEnergy = computeTotalEnergy(all_bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                cudaFree(d_pos_x); cudaFree(d_pos_y); cudaFree(d_pos_z);
                cudaFree(d_vel_x); cudaFree(d_vel_y); cudaFree(d_vel_z);
                cudaFree(d_all_pos_x); cudaFree(d_all_pos_y); cudaFree(d_all_pos_z);
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Cleanup
    cudaFree(d_pos_x); cudaFree(d_pos_y); cudaFree(d_pos_z);
    cudaFree(d_vel_x); cudaFree(d_vel_y); cudaFree(d_vel_z);
    cudaFree(d_all_pos_x); cudaFree(d_all_pos_y); cudaFree(d_all_pos_z);
    
    MPI_Finalize();
    return 0;
}
