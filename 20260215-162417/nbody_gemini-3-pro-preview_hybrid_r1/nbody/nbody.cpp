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

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

__global__ void computeForcesKernel(const double* __restrict__ pos_x, 
                                    const double* __restrict__ pos_y, 
                                    const double* __restrict__ pos_z, 
                                    double* __restrict__ vel_x, 
                                    double* __restrict__ vel_y, 
                                    double* __restrict__ vel_z, 
                                    int n, int start_idx, int num_local) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_local) return;

    int global_i = start_idx + i;
    double my_px = pos_x[global_i];
    double my_py = pos_y[global_i];
    double my_pz = pos_z[global_i];
    
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;

    for (int j = 0; j < n; ++j) {
        double dx = pos_x[j] - my_px;
        double dy = pos_y[j] - my_py;
        double dz = pos_z[j] - my_pz;
        double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        double invDist = rsqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        
        fx += dx * invDist3;
        fy += dy * invDist3;
        fz += dz * invDist3;
    }

    vel_x[i] += DT * fx;
    vel_y[i] += DT * fy;
    vel_z[i] += DT * fz;
}

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

void integrateBodies(std::vector<Body>& bodies, int start_idx, int num_local) {
    #pragma omp parallel for
    for (int i = 0; i < num_local; ++i) {
        int global_i = start_idx + i;
        bodies[global_i].pos.x += bodies[global_i].vel.x * DT;
        bodies[global_i].pos.y += bodies[global_i].vel.y * DT;
        bodies[global_i].pos.z += bodies[global_i].vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy
    #pragma omp parallel for reduction(+:energy)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies[i].vel.x * bodies[i].vel.x + 
                        bodies[i].vel.y * bodies[i].vel.y + 
                        bodies[i].vel.z * bodies[i].vel.z);
    }
    
    // Potential energy
    #pragma omp parallel for reduction(-:energy)
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } 
    }

    if (rank == 0) {
        printf("N-Body Simulation (MPI + OpenMP + CUDA)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI Size: %d\n", size);
        int num_gpus = 0;
        cudaGetDeviceCount(&num_gpus);
        printf("CUDA Devices: %d\n", num_gpus);
    }

    // Set CUDA device
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > 0) {
        cudaSetDevice(rank % num_gpus);
    } else {
        if (rank == 0) fprintf(stderr, "No CUDA devices found!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize bodies on all ranks (replicated init for simplicity)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies); // All ranks generate same sequence if seed is same.
    // Ensure all ranks have exactly same data
    MPI_Bcast(bodies.data(), numBodies * sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Determine local work
    int bodies_per_rank = numBodies / size;
    int start_idx = rank * bodies_per_rank;
    int end_idx = (rank == size - 1) ? numBodies : start_idx + bodies_per_rank;
    int num_local = end_idx - start_idx;

    // SoA (Structure of Arrays) on Host for easy transfer
    std::vector<double> h_pos_x(numBodies), h_pos_y(numBodies), h_pos_z(numBodies);
    std::vector<double> h_vel_x(num_local), h_vel_y(num_local), h_vel_z(num_local);

    for (int i = 0; i < numBodies; ++i) {
        h_pos_x[i] = bodies[i].pos.x;
        h_pos_y[i] = bodies[i].pos.y;
        h_pos_z[i] = bodies[i].pos.z;
    }
    for (int i = 0; i < num_local; ++i) {
        h_vel_x[i] = bodies[start_idx + i].vel.x;
        h_vel_y[i] = bodies[start_idx + i].vel.y;
        h_vel_z[i] = bodies[start_idx + i].vel.z;
    }

    // Allocate Device Memory
    double *d_pos_x, *d_pos_y, *d_pos_z;
    double *d_vel_x, *d_vel_y, *d_vel_z;
    CUDA_CHECK(cudaMalloc(&d_pos_x, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pos_y, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pos_z, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_x, num_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_y, num_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_z, num_local * sizeof(double)));

    // Copy initial velocities to device
    CUDA_CHECK(cudaMemcpy(d_vel_x, h_vel_x.data(), num_local * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_y, h_vel_y.data(), num_local * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_z, h_vel_z.data(), num_local * sizeof(double), cudaMemcpyHostToDevice));

    // Temporary buffers for gathering positions
    std::vector<double> local_pos_x(num_local), local_pos_y(num_local), local_pos_z(num_local);
    // Gather counts and displacements for MPI_Allgatherv
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    for(int i=0; i<size; i++) {
        int r_start = i * bodies_per_rank;
        int r_end = (i == size - 1) ? numBodies : r_start + bodies_per_rank;
        recvcounts[i] = r_end - r_start;
        displs[i] = r_start;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    int blockSize = 256;
    int gridSize = (num_local + blockSize - 1) / blockSize;

    for (int step = 0; step < numSteps; ++step) {
        // 1. Copy current positions to device (all positions needed)
        CUDA_CHECK(cudaMemcpy(d_pos_x, h_pos_x.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos_y, h_pos_y.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos_z, h_pos_z.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));

        // 2. Compute Forces & Update Velocities (Integration part 1) on GPU
        computeForcesKernel<<<gridSize, blockSize>>>(d_pos_x, d_pos_y, d_pos_z, 
                                                     d_vel_x, d_vel_y, d_vel_z, 
                                                     numBodies, start_idx, num_local);
        CUDA_CHECK(cudaGetLastError());

        // 3. Update Positions (Integration part 2) on Host or Device
        // Since we need to share positions via MPI, we need them on Host eventually.
        // Let's do integration on GPU to save bandwidth, then copy back *only local* positions.
        // Wait, I need a kernel for integration. Or reuse existing structure.
        // Let's copy velocities back to host and integrate on host with OpenMP, 
        // OR implement integration kernel. Integration is O(N), fast.
        // Let's implement a simple integration kernel to keep data on GPU as much as possible.
        // Actually, for this step, let's just copy back velocities and positions, integrate on host (OpenMP), then gather.
        // Copying back velocities is needed for Energy check anyway? No, energy check is separate.
        // To minimize kernel launches/complexity, let's copy velocities back, integrate on host, then gather.
        
        CUDA_CHECK(cudaMemcpy(h_vel_x.data(), d_vel_x, num_local * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vel_y.data(), d_vel_y, num_local * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vel_z.data(), d_vel_z, num_local * sizeof(double), cudaMemcpyDeviceToHost));

        #pragma omp parallel for
        for (int i = 0; i < num_local; ++i) {
            local_pos_x[i] = h_pos_x[start_idx + i] + h_vel_x[i] * DT;
            local_pos_y[i] = h_pos_y[start_idx + i] + h_vel_y[i] * DT;
            local_pos_z[i] = h_pos_z[start_idx + i] + h_vel_z[i] * DT;
        }

        // 4. Gather all positions
        MPI_Allgatherv(local_pos_x.data(), num_local, MPI_DOUBLE, 
                       h_pos_x.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(local_pos_y.data(), num_local, MPI_DOUBLE, 
                       h_pos_y.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(local_pos_z.data(), num_local, MPI_DOUBLE, 
                       h_pos_z.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = duration.count();
    long simulation_time_ms = 0;
    MPI_Reduce(&local_duration_ms, &simulation_time_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) printf("Simulation time: %ld ms\n", simulation_time_ms);
    
    // Sync final state to bodies vector for validation/output
    // Need to gather velocities too
    std::vector<double> all_vel_x(numBodies), all_vel_y(numBodies), all_vel_z(numBodies);
    MPI_Gatherv(h_vel_x.data(), num_local, MPI_DOUBLE,
                all_vel_x.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                0, MPI_COMM_WORLD);
    MPI_Gatherv(h_vel_y.data(), num_local, MPI_DOUBLE,
                all_vel_y.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                0, MPI_COMM_WORLD);
    MPI_Gatherv(h_vel_z.data(), num_local, MPI_DOUBLE,
                all_vel_z.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos.x = h_pos_x[i];
            bodies[i].pos.y = h_pos_y[i];
            bodies[i].pos.z = h_pos_z[i];
            bodies[i].vel.x = all_vel_x[i];
            bodies[i].vel.y = all_vel_y[i];
            bodies[i].vel.z = all_vel_z[i];
        }

        // Print results for external validation
        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(numBodies * 6);
            for (const auto& body : bodies) {
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
            if (validateSimulation(bodies)) {
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    CUDA_CHECK(cudaFree(d_pos_x));
    CUDA_CHECK(cudaFree(d_pos_y));
    CUDA_CHECK(cudaFree(d_pos_z));
    CUDA_CHECK(cudaFree(d_vel_x));
    CUDA_CHECK(cudaFree(d_vel_y));
    CUDA_CHECK(cudaFree(d_vel_z));

    MPI_Finalize();
    return 0;
}
