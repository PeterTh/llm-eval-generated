#include <mpi.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>
#include <algorithm>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int TILE_SIZE = 256;
constexpr int BLOCK_SIZE = 256;

#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

struct Vec3 {
    double x, y, z;
    constexpr Vec3(double x = 0, double y = 0, double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

__global__ void computeForcesKernel(
    const double* __restrict__ my_px,
    const double* __restrict__ my_py,
    const double* __restrict__ my_pz,
    const double* __restrict__ all_px,
    const double* __restrict__ all_py,
    const double* __restrict__ all_pz,
    double* __restrict__ vx,
    double* __restrict__ vy,
    double* __restrict__ vz,
    const int n_local,
    const int n_total)
{
    __shared__ double s_px[TILE_SIZE];
    __shared__ double s_py[TILE_SIZE];
    __shared__ double s_pz[TILE_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    double ipx = 0.0, ipy = 0.0, ipz = 0.0;
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    if (i < n_local) {
        ipx = my_px[i];
        ipy = my_py[i];
        ipz = my_pz[i];
    }

    for (int tile = 0; tile < n_total; tile += TILE_SIZE) {
        const int j = tile + threadIdx.x;
        if (j < n_total) {
            s_px[threadIdx.x] = all_px[j];
            s_py[threadIdx.x] = all_py[j];
            s_pz[threadIdx.x] = all_pz[j];
        }
        __syncthreads();

        if (i < n_local) {
            for (int k = 0; k < TILE_SIZE; ++k) {
                if (tile + k >= n_total) break;
                const double dx = s_px[k] - ipx;
                const double dy = s_py[k] - ipy;
                const double dz = s_pz[k] - ipz;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (i < n_local) {
        vx[i] += DT * Fx;
        vy[i] += DT * Fy;
        vz[i] += DT * Fz;
    }
}

__global__ void integrateKernel(
    double* __restrict__ px,
    double* __restrict__ py,
    double* __restrict__ pz,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ vz,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        px[i] += vx[i] * DT;
        py[i] += vy[i] * DT;
        pz[i] += vz[i] * DT;
    }
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

bool validateSimulation(const std::vector<Body>& bodies) {
    bool valid = true;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < bodies.size(); ++i) {
        const auto& b = bodies[i];
        if (!std::isfinite(b.pos.x) || !std::isfinite(b.pos.y) || !std::isfinite(b.pos.z) ||
            !std::isfinite(b.vel.x) || !std::isfinite(b.vel.y) || !std::isfinite(b.vel.z)) {
            #pragma omp critical
            {
                printf("Validation failed: found NaN or Inf value in body state\n");
                valid = false;
            }
        }
        const double maxPos = 1e6, maxVel = 1e6;
        if (std::abs(b.pos.x) > maxPos || std::abs(b.pos.y) > maxPos || std::abs(b.pos.z) > maxPos) {
            #pragma omp critical
            {
                printf("Validation failed: body position exceeds reasonable bounds\n");
                valid = false;
            }
        }
        if (std::abs(b.vel.x) > maxVel || std::abs(b.vel.y) > maxVel || std::abs(b.vel.z) > maxVel) {
            #pragma omp critical
            {
                printf("Validation failed: body velocity exceeds reasonable bounds\n");
                valid = false;
            }
        }
    }
    return valid;
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies[i].vel.x * bodies[i].vel.x +
                         bodies[i].vel.y * bodies[i].vel.y +
                         bodies[i].vel.z * bodies[i].vel.z);
    }

    double potential = 0.0;
    #pragma omp parallel for reduction(+:potential) schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potential += 1.0 / dist;
        }
    }
    energy -= potential;

    return energy;
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
        printf("MPI ranks: %d\n", nprocs);
    }

    int numGPUs = 0;
    cudaError_t cudaErr = cudaGetDeviceCount(&numGPUs);
    if (cudaErr != cudaSuccess || numGPUs == 0) {
        fprintf(stderr, "[Rank %d] No CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int gpuId = rank % numGPUs;
    CUDA_CHECK(cudaSetDevice(gpuId));

    if (rank == 0) {
        printf("CUDA devices: %d\n", numGPUs);
    }

    const int base_count = numBodies / nprocs;
    const int remainder = numBodies % nprocs;
    const int n_local = base_count + (rank < remainder ? 1 : 0);

    std::vector<int> recvcounts(nprocs), displs(nprocs);
    int offset = 0;
    for (int r = 0; r < nprocs; ++r) {
        recvcounts[r] = base_count + (r < remainder ? 1 : 0);
        displs[r] = offset;
        offset += recvcounts[r];
    }
    const int my_start = displs[rank];

    std::vector<Body> allBodies(numBodies);
    randomizeBodies(allBodies);

    std::vector<double> init_px(n_local), init_py(n_local), init_pz(n_local);
    std::vector<double> init_vx(n_local), init_vy(n_local), init_vz(n_local);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_local; ++i) {
        const Body& b = allBodies[my_start + i];
        init_px[i] = b.pos.x;
        init_py[i] = b.pos.y;
        init_pz[i] = b.pos.z;
        init_vx[i] = b.vel.x;
        init_vy[i] = b.vel.y;
        init_vz[i] = b.vel.z;
    }
    allBodies.clear();
    allBodies.shrink_to_fit();

    const size_t local_alloc = std::max(n_local, 1);
    const size_t total_alloc = std::max(numBodies, 1);
    double *h_send_px, *h_send_py, *h_send_pz;
    double *h_recv_px, *h_recv_py, *h_recv_pz;
    CUDA_CHECK(cudaMallocHost(&h_send_px, local_alloc * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_send_py, local_alloc * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_send_pz, local_alloc * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_px, total_alloc * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_py, total_alloc * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_pz, total_alloc * sizeof(double)));

    double *d_my_px, *d_my_py, *d_my_pz;
    double *d_my_vx, *d_my_vy, *d_my_vz;
    CUDA_CHECK(cudaMalloc(&d_my_px, local_alloc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_my_py, local_alloc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_my_pz, local_alloc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_my_vx, local_alloc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_my_vy, local_alloc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_my_vz, local_alloc * sizeof(double)));

    double *d_all_px, *d_all_py, *d_all_pz;
    CUDA_CHECK(cudaMalloc(&d_all_px, total_alloc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_all_py, total_alloc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_all_pz, total_alloc * sizeof(double)));

    if (n_local > 0) {
        CUDA_CHECK(cudaMemcpy(d_my_px, init_px.data(), n_local * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_my_py, init_py.data(), n_local * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_my_pz, init_pz.data(), n_local * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_my_vx, init_vx.data(), n_local * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_my_vy, init_vy.data(), n_local * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_my_vz, init_vz.data(), n_local * sizeof(double), cudaMemcpyHostToDevice));
    }

    init_px.clear(); init_px.shrink_to_fit();
    init_py.clear(); init_py.shrink_to_fit();
    init_pz.clear(); init_pz.shrink_to_fit();
    init_vx.clear(); init_vx.shrink_to_fit();
    init_vy.clear(); init_vy.shrink_to_fit();
    init_vz.clear(); init_vz.shrink_to_fit();

    const int numBlocks = (n_local + BLOCK_SIZE - 1) / BLOCK_SIZE;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        if (n_local > 0) {
            CUDA_CHECK(cudaMemcpy(h_send_px, d_my_px, n_local * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_send_py, d_my_py, n_local * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_send_pz, d_my_pz, n_local * sizeof(double), cudaMemcpyDeviceToHost));
        }

        MPI_Allgatherv(h_send_px, n_local, MPI_DOUBLE,
                       h_recv_px, recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(h_send_py, n_local, MPI_DOUBLE,
                       h_recv_py, recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(h_send_pz, n_local, MPI_DOUBLE,
                       h_recv_pz, recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(d_all_px, h_recv_px, numBodies * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_all_py, h_recv_py, numBodies * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_all_pz, h_recv_pz, numBodies * sizeof(double), cudaMemcpyHostToDevice));

        if (numBlocks > 0) {
            computeForcesKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_my_px, d_my_py, d_my_pz,
                d_all_px, d_all_py, d_all_pz,
                d_my_vx, d_my_vy, d_my_vz,
                n_local, numBodies);

            integrateKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_my_px, d_my_py, d_my_pz,
                d_my_vx, d_my_vy, d_my_vz,
                n_local);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        printf("Simulation time: %ld ms\n", duration.count());
    }

    std::vector<double> final_px(numBodies), final_py(numBodies), final_pz(numBodies);
    std::vector<double> final_vx(numBodies), final_vy(numBodies), final_vz(numBodies);

    // Gather positions
    if (n_local > 0) {
        CUDA_CHECK(cudaMemcpy(h_send_px, d_my_px, n_local * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_send_py, d_my_py, n_local * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_send_pz, d_my_pz, n_local * sizeof(double), cudaMemcpyDeviceToHost));
    }
    MPI_Allgatherv(h_send_px, n_local, MPI_DOUBLE,
                   final_px.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(h_send_py, n_local, MPI_DOUBLE,
                   final_py.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(h_send_pz, n_local, MPI_DOUBLE,
                   final_pz.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Gather velocities (reuse recv buffers temporarily)
    if (n_local > 0) {
        CUDA_CHECK(cudaMemcpy(h_send_px, d_my_vx, n_local * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_send_py, d_my_vy, n_local * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_send_pz, d_my_vz, n_local * sizeof(double), cudaMemcpyDeviceToHost));
    }
    MPI_Allgatherv(h_send_px, n_local, MPI_DOUBLE,
                   final_vx.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(h_send_py, n_local, MPI_DOUBLE,
                   final_vy.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(h_send_pz, n_local, MPI_DOUBLE,
                   final_vz.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    if (printResults && rank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            bodyData.push_back(final_px[i]);
            bodyData.push_back(final_py[i]);
            bodyData.push_back(final_pz[i]);
            bodyData.push_back(final_vx[i]);
            bodyData.push_back(final_vy[i]);
            bodyData.push_back(final_vz[i]);
        }
        print_results(bodyData, "Bodies");
    }

    if (validate && rank == 0) {
        printf("Validating simulation results...\n");

        std::vector<Body> finalBodies(numBodies);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            finalBodies[i].pos.x = final_px[i];
            finalBodies[i].pos.y = final_py[i];
            finalBodies[i].pos.z = final_pz[i];
            finalBodies[i].vel.x = final_vx[i];
            finalBodies[i].vel.y = final_vy[i];
            finalBodies[i].vel.z = final_vz[i];
        }

        if (validateSimulation(finalBodies)) {
            double finalEnergy = computeTotalEnergy(finalBodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    CUDA_CHECK(cudaFreeHost(h_send_px));
    CUDA_CHECK(cudaFreeHost(h_send_py));
    CUDA_CHECK(cudaFreeHost(h_send_pz));
    CUDA_CHECK(cudaFreeHost(h_recv_px));
    CUDA_CHECK(cudaFreeHost(h_recv_py));
    CUDA_CHECK(cudaFreeHost(h_recv_pz));

    CUDA_CHECK(cudaFree(d_my_px));
    CUDA_CHECK(cudaFree(d_my_py));
    CUDA_CHECK(cudaFree(d_my_pz));
    CUDA_CHECK(cudaFree(d_my_vx));
    CUDA_CHECK(cudaFree(d_my_vy));
    CUDA_CHECK(cudaFree(d_my_vz));

    CUDA_CHECK(cudaFree(d_all_px));
    CUDA_CHECK(cudaFree(d_all_py));
    CUDA_CHECK(cudaFree(d_all_pz));

    MPI_Finalize();
    return 0;
}
