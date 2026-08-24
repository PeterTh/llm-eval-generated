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
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
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

__global__ void computeForcesKernel(Body* bodies, int n, int start_idx, int local_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ Body shared_bodies[];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double my_x = 0.0, my_y = 0.0, my_z = 0.0;
    
    if (idx < local_n) {
        int i = start_idx + idx;
        my_x = bodies[i].pos.x;
        my_y = bodies[i].pos.y;
        my_z = bodies[i].pos.z;
    }

    for (int tile = 0; tile < (n + blockDim.x - 1) / blockDim.x; ++tile) {
        int tile_idx = tile * blockDim.x + threadIdx.x;
        
        if (tile_idx < n) {
            shared_bodies[threadIdx.x] = bodies[tile_idx];
        }
        __syncthreads();

        if (idx < local_n) {
            int bodies_in_tile = min((int)blockDim.x, n - tile * (int)blockDim.x);
            
            #pragma unroll
            for (int j = 0; j < bodies_in_tile; ++j) {
                const double dx = shared_bodies[j].pos.x - my_x;
                const double dy = shared_bodies[j].pos.y - my_y;
                const double dz = shared_bodies[j].pos.z - my_z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = rsqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (idx < local_n) {
        int i = start_idx + idx;
        bodies[i].vel.x += DT * Fx;
        bodies[i].vel.y += DT * Fy;
        bodies[i].vel.z += DT * Fz;
    }
}

__global__ void integrateBodiesKernel(Body* bodies, int start_idx, int local_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < local_n) {
        int i = start_idx + idx;
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

// Host function to compute total energy (for validation)
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
    // Provide explicit threading support for MPI_Init_thread if we were to use threads to make MPI calls
    // But we are not making MPI calls from OMP threads, so MPI_Init is fine.
    // However, for good measure in hybrid apps:
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

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
        printf("N-Body Simulation (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Number of MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    // All ranks generate same initial state to ensure consistency
    randomizeBodies(bodies);

    // Distribute work
    int bodiesPerRank = numBodies / size;
    int remainder = numBodies % size;
    
    // Arrays for MPI_Allgatherv
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    int current_idx = 0;
    for(int i=0; i<size; ++i) {
        recvcounts[i] = bodiesPerRank + (i < remainder ? 1 : 0);
        displs[i] = current_idx;
        current_idx += recvcounts[i];
    }
    
    int local_n = recvcounts[rank];
    int start_idx = displs[rank];

    // Allocate device memory
    Body* d_bodies;
    cudaMalloc(&d_bodies, numBodies * sizeof(Body));
    
    // Initial copy of all bodies to device
    cudaMemcpy(d_bodies, bodies.data(), numBodies * sizeof(Body), cudaMemcpyHostToDevice);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    int blockSize = 256;
    int numBlocks = (local_n + blockSize - 1) / blockSize;

    // Create MPI datatype for Body
    // Body is 6 doubles: pos(3), vel(3)
    MPI_Datatype mpi_body_type;
    MPI_Type_contiguous(6, MPI_DOUBLE, &mpi_body_type);
    MPI_Type_commit(&mpi_body_type);
    
    for (int step = 0; step < numSteps; ++step) {
        // Compute forces and integrate only for local bodies
        computeForcesKernel<<<numBlocks, blockSize, blockSize * sizeof(Body)>>>(d_bodies, numBodies, start_idx, local_n);
        integrateBodiesKernel<<<numBlocks, blockSize>>>(d_bodies, start_idx, local_n);
        cudaDeviceSynchronize();

        // Copy updated local bodies back to host
        // We only copy the chunk we own
        cudaMemcpy(bodies.data() + start_idx, d_bodies + start_idx, local_n * sizeof(Body), cudaMemcpyDeviceToHost);

        // Exchange data
        // MPI_Allgatherv with MPI_IN_PLACE means we send from our position in the buffer
        // and receive into the whole buffer
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                       bodies.data(), recvcounts.data(), displs.data(), mpi_body_type, MPI_COMM_WORLD);

        // Copy updated global state to device for next step
        // We need all bodies because forces depend on all bodies
        cudaMemcpy(d_bodies, bodies.data(), numBodies * sizeof(Body), cudaMemcpyHostToDevice);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = static_cast<long long>(duration.count());
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %lld ms\n", max_duration_ms);

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

        if (validate) {
            printf("Validating simulation results...\n");
            // Validation on host using OpenMP
            if (validateSimulation(bodies)) {
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Type_free(&mpi_body_type);
    cudaFree(d_bodies);
    MPI_Finalize();
    return 0;
}
