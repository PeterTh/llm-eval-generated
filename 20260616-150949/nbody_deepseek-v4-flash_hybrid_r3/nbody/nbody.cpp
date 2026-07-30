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

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        cudaError_t err = call;                                                      \
        if (err != cudaSuccess) {                                                    \
            fprintf(stderr, "[%d] CUDA error: %s\n", rank, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                            \
        }                                                                            \
    } while (0)

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// CUDA kernel: compute gravitational forces and update velocities for local bodies
// Uses tiled shared memory for coalesced global memory access.
__global__ void computeForcesKernel(
    const double* __restrict__ pos_x, const double* __restrict__ pos_y,
    const double* __restrict__ pos_z, double* __restrict__ vel_x,
    double* __restrict__ vel_y, double* __restrict__ vel_z, int n,
    int local_start, int local_count, double softening, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= local_count) return;

    int i = local_start + idx;
    double px = pos_x[i];
    double py = pos_y[i];
    double pz = pos_z[i];

    double fx = 0.0, fy = 0.0, fz = 0.0;

    extern __shared__ double shared[];
    double* sx = shared;
    double* sy = shared + blockDim.x;
    double* sz = shared + 2 * blockDim.x;

    for (int tile = 0; tile * blockDim.x < n; ++tile) {
        int j = tile * blockDim.x + threadIdx.x;
        if (j < n) {
            sx[threadIdx.x] = pos_x[j];
            sy[threadIdx.x] = pos_y[j];
            sz[threadIdx.x] = pos_z[j];
        }
        __syncthreads();

        int limit = blockDim.x;
        if ((tile + 1) * blockDim.x > n) {
            limit = n - tile * blockDim.x;
        }

        #pragma unroll
        for (int tj = 0; tj < limit; ++tj) {
            double dx = sx[tj] - px;
            double dy = sy[tj] - py;
            double dz = sz[tj] - pz;
            double distSqr = dx * dx + dy * dy + dz * dz + softening;
            double invDist = 1.0 / sqrt(distSqr);
            double invDist3 = invDist * invDist * invDist;
            fx += dx * invDist3;
            fy += dy * invDist3;
            fz += dz * invDist3;
        }
        __syncthreads();
    }

    vel_x[i] += dt * fx;
    vel_y[i] += dt * fy;
    vel_z[i] += dt * fz;
}

// CUDA kernel: integrate positions for local bodies (pos += vel * dt)
__global__ void integrateKernel(double* __restrict__ pos_x,
                                double* __restrict__ pos_y,
                                double* __restrict__ pos_z,
                                const double* __restrict__ vel_x,
                                const double* __restrict__ vel_y,
                                const double* __restrict__ vel_z,
                                int local_start, int local_count, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= local_count) return;
    int i = local_start + idx;
    pos_x[i] += vel_x[i] * dt;
    pos_y[i] += vel_y[i] * dt;
    pos_z[i] += vel_z[i] * dt;
}

// ---------------------------------------------------------------------------
// Host helper functions (used outside timed region)
// ---------------------------------------------------------------------------

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

    #pragma omp parallel for reduction(+ : energy)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies[i].vel.x * bodies[i].vel.x +
                         bodies[i].vel.y * bodies[i].vel.y +
                         bodies[i].vel.z * bodies[i].vel.z);
    }

    #pragma omp parallel for reduction(+ : energy)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double dx = bodies[j].pos.x - bodies[i].pos.x;
            double dy = bodies[j].pos.y - bodies[i].pos.y;
            double dz = bodies[j].pos.z - bodies[i].pos.z;
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    bool valid = true;
    const size_t n = bodies.size();

    #pragma omp parallel for reduction(&& : valid)
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(bodies[i].pos.x) || !std::isfinite(bodies[i].pos.y) ||
            !std::isfinite(bodies[i].pos.z) || !std::isfinite(bodies[i].vel.x) ||
            !std::isfinite(bodies[i].vel.y) || !std::isfinite(bodies[i].vel.z)) {
            valid = false;
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::fabs(bodies[i].pos.x) > maxPos || std::fabs(bodies[i].pos.y) > maxPos ||
            std::fabs(bodies[i].pos.z) > maxPos) {
            valid = false;
        }
        if (std::fabs(bodies[i].vel.x) > maxVel || std::fabs(bodies[i].vel.y) > maxVel ||
            std::fabs(bodies[i].vel.z) > maxVel) {
            valid = false;
        }
    }
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ---------------------------------------------------------------------------
// Main – hybrid MPI + OpenMP + CUDA n-body simulation
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int numBodies      = 1024;
    int numSteps       = 10;
    bool validate      = false;
    bool printResults  = false;

    // ---- parse arguments on rank 0, then broadcast ----
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
    // Use int for portability
    int val_int = validate ? 1 : 0;
    MPI_Bcast(&val_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (val_int != 0);
    val_int = printResults ? 1 : 0;
    MPI_Bcast(&val_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printResults = (val_int != 0);

    // ---- data distribution ----
    int base_local = numBodies / numRanks;
    int remainder  = numBodies % numRanks;

    std::vector<int> counts(numRanks), displs(numRanks);
    {
        int off = 0;
        for (int r = 0; r < numRanks; ++r) {
            counts[r] = base_local + (r < remainder ? 1 : 0);
            displs[r] = off;
            off += counts[r];
        }
    }
    int local_n     = counts[rank];
    int local_start = displs[rank];

    // ---- report configuration ----
    if (rank == 0) {
        printf("N-Body Simulation (MPI+OpenMP+CUDA)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps:  %d\n", numSteps);
        printf("Validation:       %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks:        %d\n", numRanks);
        int omp_threads;
        #pragma omp parallel
        {
            #pragma omp master
            omp_threads = omp_get_num_threads();
        }
        printf("OpenMP threads:   %d\n", omp_threads);

        int nDev;
        cudaGetDeviceCount(&nDev);
        printf("CUDA devices:     %d\n", nDev);
        if (nDev > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            printf("GPU:              %s\n", prop.name);
        }
        fflush(stdout);
    }

    // ---- select CUDA device (round-robin) ----
    int nCudaDevices = 0;
    cudaGetDeviceCount(&nCudaDevices);
    if (nCudaDevices == 0) {
        fprintf(stderr, "[%d] No CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % nCudaDevices));

    // ---- host memory (SoA layout) ----
    // Positions – all n bodies (filled by MPI exchange each step)
    std::vector<double> h_pos_x(numBodies);
    std::vector<double> h_pos_y(numBodies);
    std::vector<double> h_pos_z(numBodies);

    // Velocities – all-n allocation, but each rank owns its local segment
    std::vector<double> h_vel_x(numBodies);
    std::vector<double> h_vel_y(numBodies);
    std::vector<double> h_vel_z(numBodies);

    // ---- initialise bodies on rank 0, broadcast ----
    {
        std::vector<Body> allBodies;
        if (rank == 0) {
            allBodies.resize(numBodies);
            randomizeBodies(allBodies);
            for (int i = 0; i < numBodies; ++i) {
                h_pos_x[i] = allBodies[i].pos.x;
                h_pos_y[i] = allBodies[i].pos.y;
                h_pos_z[i] = allBodies[i].pos.z;
                h_vel_x[i] = allBodies[i].vel.x;
                h_vel_y[i] = allBodies[i].vel.y;
                h_vel_z[i] = allBodies[i].vel.z;
            }
        }
        // broadcast full state to every rank
        MPI_Bcast(h_pos_x.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_pos_y.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_pos_z.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_vel_x.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_vel_y.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_vel_z.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // ---- device memory ----
    double *d_pos_x, *d_pos_y, *d_pos_z;
    double *d_vel_x, *d_vel_y, *d_vel_z;

    CUDA_CHECK(cudaMalloc(&d_pos_x, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pos_y, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pos_z, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_x, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_y, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel_z, numBodies * sizeof(double)));

    // copy initial state to device
    CUDA_CHECK(cudaMemcpy(d_pos_x, h_pos_x.data(), numBodies * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_y, h_pos_y.data(), numBodies * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_z, h_pos_z.data(), numBodies * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_x, h_vel_x.data(), numBodies * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_y, h_vel_y.data(), numBodies * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_z, h_vel_z.data(), numBodies * sizeof(double),
                          cudaMemcpyHostToDevice));

    // ---- CUDA launch configuration ----
    constexpr int BLOCK_DIM = 256;
    dim3 block(BLOCK_DIM);
    dim3 grid((local_n + BLOCK_DIM - 1) / BLOCK_DIM);
    size_t shared_bytes = 3 * BLOCK_DIM * sizeof(double);

    // ---- timed simulation loop ----
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        // 1. Copy local positions back to host (blocking – syncs prior kernels)
        CUDA_CHECK(cudaMemcpy(&h_pos_x[local_start], d_pos_x + local_start,
                              local_n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_pos_y[local_start], d_pos_y + local_start,
                              local_n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_pos_z[local_start], d_pos_z + local_start,
                              local_n * sizeof(double), cudaMemcpyDeviceToHost));

        // 2. MPI_Allgatherv – exchange positions (IN_PLACE: local data already
        //    resides at correct offset in the receive buffer)
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_pos_x.data(),
                       counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_pos_y.data(),
                       counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_pos_z.data(),
                       counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // 3. Copy all positions to device
        CUDA_CHECK(cudaMemcpy(d_pos_x, h_pos_x.data(), numBodies * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos_y, h_pos_y.data(), numBodies * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pos_z, h_pos_z.data(), numBodies * sizeof(double),
                              cudaMemcpyHostToDevice));

        // 4. Force computation + velocity update (GPU)
        computeForcesKernel<<<grid, block, shared_bytes>>>(
            d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, numBodies,
            local_start, local_n, SOFTENING, DT);

        // 5. Position integration (GPU)
        integrateKernel<<<grid, block>>>(
            d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, local_start,
            local_n, DT);

        // NOTE: no explicit sync is needed here because the next
        // cudaMemcpy(D2H) at the top of the loop is implicitly blocking.
    }

    // Wait for any trailing GPU work, then barrier for timing
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    double elapsed = t_end - t_start;

    // ---- gather results to rank 0 ----
    // Copy local velocities from device to host
    CUDA_CHECK(cudaMemcpy(&h_vel_x[local_start], d_vel_x + local_start,
                          local_n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_vel_y[local_start], d_vel_y + local_start,
                          local_n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_vel_z[local_start], d_vel_z + local_start,
                          local_n * sizeof(double), cudaMemcpyDeviceToHost));

    // Gather velocities to rank 0 (positions are already replicated via Allgather)
    std::vector<double> g_vel_x, g_vel_y, g_vel_z;
    if (rank == 0) {
        g_vel_x.resize(numBodies);
        g_vel_y.resize(numBodies);
        g_vel_z.resize(numBodies);
    }

    MPI_Gatherv(h_vel_x.data() + local_start, local_n, MPI_DOUBLE,
                g_vel_x.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(h_vel_y.data() + local_start, local_n, MPI_DOUBLE,
                g_vel_y.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(h_vel_z.data() + local_start, local_n, MPI_DOUBLE,
                g_vel_z.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // ---- output ----
    if (rank == 0) {
        printf("Simulation time: %.3f s\n", elapsed);

        // Reconstruct Body vector for output / validation
        std::vector<Body> finalBodies(numBodies);
        for (int i = 0; i < numBodies; ++i) {
            finalBodies[i].pos.x = h_pos_x[i];
            finalBodies[i].pos.y = h_pos_y[i];
            finalBodies[i].pos.z = h_pos_z[i];
            finalBodies[i].vel.x = g_vel_x[i];
            finalBodies[i].vel.y = g_vel_y[i];
            finalBodies[i].vel.z = g_vel_z[i];
        }

        if (printResults) {
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

        if (validate) {
            printf("Validating simulation results...\n");
            if (validateSimulation(finalBodies)) {
                double finalEnergy = computeTotalEnergy(finalBodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    // ---- cleanup ----
    CUDA_CHECK(cudaFree(d_pos_x));
    CUDA_CHECK(cudaFree(d_pos_y));
    CUDA_CHECK(cudaFree(d_pos_z));
    CUDA_CHECK(cudaFree(d_vel_x));
    CUDA_CHECK(cudaFree(d_vel_y));
    CUDA_CHECK(cudaFree(d_vel_z));

    MPI_Finalize();
    return 0;
}
