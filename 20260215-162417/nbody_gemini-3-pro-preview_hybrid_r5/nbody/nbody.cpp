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
constexpr int BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
    // constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// CUDA error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void computeForcesKernel(const Body* __restrict__ all_bodies, 
                                    Body* __restrict__ local_bodies, 
                                    int n, int local_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= local_n) return;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double my_x = local_bodies[i].pos.x;
    double my_y = local_bodies[i].pos.y;
    double my_z = local_bodies[i].pos.z;

    for (int j = 0; j < n; ++j) {
        double dx = all_bodies[j].pos.x - my_x;
        double dy = all_bodies[j].pos.y - my_y;
        double dz = all_bodies[j].pos.z - my_z;
        double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        double invDist = 1.0 / sqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    local_bodies[i].vel.x += DT * Fx;
    local_bodies[i].vel.y += DT * Fy;
    local_bodies[i].vel.z += DT * Fz;
}

__global__ void integrateBodiesKernel(Body* __restrict__ bodies, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    bodies[i].pos.x += bodies[i].vel.x * DT;
    bodies[i].pos.y += bodies[i].vel.y * DT;
    bodies[i].pos.z += bodies[i].vel.z * DT;
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

// Validation function (CPU only)
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
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

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double kinetic_energy = 0.0;
    double potential_energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy
    #pragma omp parallel for reduction(+:kinetic_energy)
    for (size_t i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        kinetic_energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy
    #pragma omp parallel for reduction(+:potential_energy)
    for (size_t i = 0; i < n; ++i) {
        double local_pot = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            local_pot += 1.0 / dist;
        }
        potential_energy += local_pot;
    }
    
    return kinetic_energy - potential_energy;
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
            }
        }
    }

    // Broadcast arguments
    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    bool b_validate = validate;
    bool b_printResults = printResults;
    MPI_Bcast(&b_validate, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b_printResults, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    validate = b_validate;
    printResults = b_printResults;

    if (numBodies % size != 0) {
        if (rank == 0) printf("Error: Number of bodies must be divisible by number of ranks.\n");
        MPI_Finalize();
        return 1;
    }

    int local_n = numBodies / size;
    int offset = rank * local_n;

    if (rank == 0) {
        printf("N-Body Simulation (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI Ranks: %d\n", size);
        printf("Bodies per rank: %d\n", local_n);
    }

    // Set GPU device
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > 0) {
        cudaSetDevice(rank % num_gpus);
    }
    
    // Initialize bodies (on host)
    std::vector<Body> all_bodies_host(numBodies);
    randomizeBodies(all_bodies_host); 

    std::vector<Body> local_bodies_host(local_n);
    // Copy my portion
    std::memcpy(local_bodies_host.data(), &all_bodies_host[offset], local_n * sizeof(Body));

    // Allocate device memory
    Body *d_all_bodies, *d_local_bodies;
    cudaCheckError(cudaMalloc(&d_all_bodies, numBodies * sizeof(Body)));
    cudaCheckError(cudaMalloc(&d_local_bodies, local_n * sizeof(Body)));

    // Initial copy to device
    cudaCheckError(cudaMemcpy(d_all_bodies, all_bodies_host.data(), numBodies * sizeof(Body), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_local_bodies, local_bodies_host.data(), local_n * sizeof(Body), cudaMemcpyHostToDevice));

    // Create MPI type for Body
    MPI_Datatype mpi_body_type;
    MPI_Type_contiguous(6, MPI_DOUBLE, &mpi_body_type);
    MPI_Type_commit(&mpi_body_type);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (local_n + threadsPerBlock - 1) / threadsPerBlock;

    for (int step = 0; step < numSteps; ++step) {
        // 1. Compute forces on local bodies using all bodies
        computeForcesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_all_bodies, d_local_bodies, numBodies, local_n);
        cudaCheckError(cudaGetLastError());

        // 2. Integrate local bodies
        integrateBodiesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_local_bodies, local_n);
        cudaCheckError(cudaGetLastError());
        cudaDeviceSynchronize();

        // 3. Copy updated local bodies back to host to share
        cudaCheckError(cudaMemcpy(local_bodies_host.data(), d_local_bodies, local_n * sizeof(Body), cudaMemcpyDeviceToHost));

        // 4. Share bodies
        MPI_Allgather(local_bodies_host.data(), local_n, mpi_body_type,
                      all_bodies_host.data(), local_n, mpi_body_type, MPI_COMM_WORLD);
        
        // 5. Update all bodies on device
        cudaCheckError(cudaMemcpy(d_all_bodies, all_bodies_host.data(), numBodies * sizeof(Body), cudaMemcpyHostToDevice));
    }

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = static_cast<long>(duration.count());
    long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", global_duration_ms);
        
        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(numBodies * 6);
            for (const auto& body : all_bodies_host) {
                bodyData.push_back(body.pos.x);
                bodyData.push_back(body.pos.y);
                bodyData.push_back(body.pos.z);
                bodyData.push_back(body.vel.x);
                bodyData.push_back(body.vel.y);
                bodyData.push_back(body.vel.z);
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            printf("Validating simulation results...\n");
            if (validateSimulation(all_bodies_host)) {
                double finalEnergy = computeTotalEnergy(all_bodies_host);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    cudaFree(d_all_bodies);
    cudaFree(d_local_bodies);
    MPI_Type_free(&mpi_body_type);
    MPI_Finalize();
    return 0;
}
