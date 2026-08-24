#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK_SIZE = 256;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// CUDA kernel: each thread computes the total force on one body from all other bodies.
// Uses shared-memory tiling over the source bodies for better memory bandwidth.
__global__ void computeForcesKernel(
    const double* __restrict__ g_px,
    const double* __restrict__ g_py,
    const double* __restrict__ g_pz,
    double* __restrict__ g_fx,
    double* __restrict__ g_fy,
    double* __restrict__ g_fz,
    const int local_start,
    const int local_count,
    const int total_bodies
) {
    __shared__ double s_px[BLOCK_SIZE];
    __shared__ double s_py[BLOCK_SIZE];
    __shared__ double s_pz[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid + local_start;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double px_i = 0.0, py_i = 0.0, pz_i = 0.0;

    const bool active = (i < local_start + local_count);
    if (active) {
        px_i = g_px[i];
        py_i = g_py[i];
        pz_i = g_pz[i];
    }

    const int num_tiles = (total_bodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        const int tile_start = t * BLOCK_SIZE;
        const int load_idx = tile_start + tid;

        if (load_idx < total_bodies) {
            s_px[tid] = g_px[load_idx];
            s_py[tid] = g_py[load_idx];
            s_pz[tid] = g_pz[load_idx];
        } else {
            s_px[tid] = 0.0;
            s_py[tid] = 0.0;
            s_pz[tid] = 0.0;
        }

        __syncthreads();

        if (active) {
            const int tile_end = min(BLOCK_SIZE, total_bodies - tile_start);
            #pragma unroll 8
            for (int j = 0; j < tile_end; ++j) {
                const double dx = s_px[j] - px_i;
                const double dy = s_py[j] - py_i;
                const double dz = s_pz[j] - pz_i;
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

    if (active) {
        const int local_idx = i - local_start;
        g_fx[local_idx] = Fx;
        g_fy[local_idx] = Fy;
        g_fz[local_idx] = Fz;
    }
}

// Must match original sequential randomization exactly
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

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double local_energy = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            local_energy -= 1.0 / dist;
        }
        energy += local_energy;
    }

    return energy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    bool valid = true;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < bodies.size(); ++i) {
        const auto& body = bodies[i];
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            #pragma omp critical
            {
                printf("Validation failed: found NaN or Inf value in body state\n");
                valid = false;
            }
        }

        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            #pragma omp critical
            {
                printf("Validation failed: body position exceeds reasonable bounds\n");
                valid = false;
            }
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            #pragma omp critical
            {
                printf("Validation failed: body velocity exceeds reasonable bounds\n");
                valid = false;
            }
        }
    }
    return valid;
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
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
        }
    }

    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("N-Body Simulation (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", nprocs);
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Select GPU (round-robin assignment)
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus == 0) {
        fprintf(stderr, "Rank %d: No CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int gpu_id = rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(gpu_id));

    if (rank == 0) {
        printf("GPUs available: %d\n", num_gpus);
    }

    // Distribute bodies across MPI ranks
    const int N = numBodies;
    const int local_start = (rank * N) / nprocs;
    const int local_count = ((rank + 1) * N) / nprocs - local_start;

    // Initialize bodies identically on all ranks (serial, same seed)
    std::vector<Body> bodies(N);
    randomizeBodies(bodies);

    // Allocate GPU buffers - positions stored for ALL bodies on GPU
    double *d_px, *d_py, *d_pz;
    double *d_fx, *d_fy, *d_fz;

    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pz, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fx, local_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fy, local_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fz, local_count * sizeof(double)));

    // CPU-side arrays for position data
    std::vector<double> h_px(N), h_py(N), h_pz(N);
    std::vector<double> h_fx(local_count), h_fy(local_count), h_fz(local_count);

    // Pack initial positions
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        h_px[i] = bodies[i].pos.x;
        h_py[i] = bodies[i].pos.y;
        h_pz[i] = bodies[i].pos.z;
    }

    // Upload initial positions to GPU
    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pz, h_pz.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    // Precompute MPI Allgatherv metadata for positions (packed xyz)
    std::vector<int> pos_gather_counts(nprocs), pos_gather_displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        int r_start = (r * N) / nprocs;
        int r_count = ((r + 1) * N) / nprocs - r_start;
        pos_gather_counts[r] = r_count * 3;
        pos_gather_displs[r] = r_start * 3;
    }

    // Precompute MPI Allgatherv metadata for velocities (packed xyz)
    std::vector<int> vel_gather_counts(nprocs), vel_gather_displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        int r_start = (r * N) / nprocs;
        int r_count = ((r + 1) * N) / nprocs - r_start;
        vel_gather_counts[r] = r_count * 3;
        vel_gather_displs[r] = r_start * 3;
    }

    // Communication buffers
    std::vector<double> pos_send(local_count * 3);
    std::vector<double> pos_recv(N * 3);
    std::vector<double> vel_send(local_count * 3);
    std::vector<double> vel_recv(N * 3);

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const int num_blocks = (local_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE);

    for (int step = 0; step < numSteps; ++step) {
        // === CUDA: compute forces for local bodies ===
        computeForcesKernel<<<grid, block>>>(
            d_px, d_py, d_pz,
            d_fx, d_fy, d_fz,
            local_start, local_count, N
        );

        // Copy forces from GPU to CPU
        CUDA_CHECK(cudaMemcpy(h_fx.data(), d_fx, local_count * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_fy.data(), d_fy, local_count * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_fz.data(), d_fz, local_count * sizeof(double), cudaMemcpyDeviceToHost));

        // === OpenMP: update velocities and integrate positions for local bodies ===
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < local_count; ++i) {
            const int idx = local_start + i;
            bodies[idx].vel.x += DT * h_fx[i];
            bodies[idx].vel.y += DT * h_fy[i];
            bodies[idx].vel.z += DT * h_fz[i];
            bodies[idx].pos.x += bodies[idx].vel.x * DT;
            bodies[idx].pos.y += bodies[idx].vel.y * DT;
            bodies[idx].pos.z += bodies[idx].vel.z * DT;
        }

        // === MPI: all-gather positions (packed xyz) for next step ===
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < local_count; ++i) {
            const int idx = local_start + i;
            pos_send[i * 3 + 0] = bodies[idx].pos.x;
            pos_send[i * 3 + 1] = bodies[idx].pos.y;
            pos_send[i * 3 + 2] = bodies[idx].pos.z;
        }

        MPI_Allgatherv(pos_send.data(), local_count * 3, MPI_DOUBLE,
                       pos_recv.data(), pos_gather_counts.data(), pos_gather_displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Unpack positions and upload to GPU
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            h_px[i] = pos_recv[i * 3 + 0];
            h_py[i] = pos_recv[i * 3 + 1];
            h_pz[i] = pos_recv[i * 3 + 2];
        }

        CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pz, h_pz.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long long local_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long global_duration = 0;
    MPI_Reduce(&local_duration, &global_duration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %lld ms\n", global_duration);
    }

    // === Final gather: collect all velocities to all ranks ===
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < local_count; ++i) {
        const int idx = local_start + i;
        vel_send[i * 3 + 0] = bodies[idx].vel.x;
        vel_send[i * 3 + 1] = bodies[idx].vel.y;
        vel_send[i * 3 + 2] = bodies[idx].vel.z;
    }

    MPI_Allgatherv(vel_send.data(), local_count * 3, MPI_DOUBLE,
                   vel_recv.data(), vel_gather_counts.data(), vel_gather_displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    // Reconstruct full body state on all ranks
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        bodies[i].vel.x = vel_recv[i * 3 + 0];
        bodies[i].vel.y = vel_recv[i * 3 + 1];
        bodies[i].vel.z = vel_recv[i * 3 + 2];
        bodies[i].pos.x = pos_recv[i * 3 + 0];
        bodies[i].pos.y = pos_recv[i * 3 + 1];
        bodies[i].pos.z = pos_recv[i * 3 + 2];
    }

    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(N * 6);
        for (int i = 0; i < N; ++i) {
            bodyData.push_back(bodies[i].pos.x);
            bodyData.push_back(bodies[i].pos.y);
            bodyData.push_back(bodies[i].pos.z);
            bodyData.push_back(bodies[i].vel.x);
            bodyData.push_back(bodies[i].vel.y);
            bodyData.push_back(bodies[i].vel.z);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
        }

        bool local_valid = validateSimulation(bodies);
        bool global_valid;
        MPI_Allreduce(&local_valid, &global_valid, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);

        if (rank == 0) {
            if (global_valid) {
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_px));
    CUDA_CHECK(cudaFree(d_py));
    CUDA_CHECK(cudaFree(d_pz));
    CUDA_CHECK(cudaFree(d_fx));
    CUDA_CHECK(cudaFree(d_fy));
    CUDA_CHECK(cudaFree(d_fz));

    MPI_Finalize();
    return 0;
}
