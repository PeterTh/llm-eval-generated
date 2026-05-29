#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                   cudaGetErrorString(err));                                     \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Body {
    double x, y, z;
    double vx, vy, vz;
};

/*
 * Combined force-computation + integration kernel.
 *
 * Layout: Structure-of-Arrays (SoA) on device for coalesced global memory
 *         accesses.  Each thread is responsible for one body.  The O(N^2)
 *         force loop is tiled over shared memory: every block loads a tile
 *         of body positions into shared memory, then all threads in the
 *         block compute their pairwise forces against that tile.
 */
__global__ void computeForcesAndIntegrate(
    const double* __restrict__ posx,
    const double* __restrict__ posy,
    const double* __restrict__ posz,
    double* __restrict__ velx,
    double* __restrict__ vely,
    double* __restrict__ velz,
    double* __restrict__ newposx,
    double* __restrict__ newposy,
    double* __restrict__ newposz,
    const int n)
{
    extern __shared__ double shared[];
    double* sx = shared;
    double* sy = shared + blockDim.x;
    double* sz = shared + 2 * blockDim.x;

    const int tile_size = blockDim.x;
    const int num_tiles = (n + tile_size - 1) / tile_size;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int t = 0; t < num_tiles; ++t) {
        // --- load tile into shared memory ---
        int j = t * tile_size + threadIdx.x;
        if (j < n) {
            sx[threadIdx.x] = posx[j];
            sy[threadIdx.x] = posy[j];
            sz[threadIdx.x] = posz[j];
        }
        __syncthreads();

        // --- accumulate forces from this tile ---
        for (int jj = 0; jj < tile_size; ++jj) {
            const int jdx = t * tile_size + jj;
            if (jdx < n) {
                const double dx = sx[jj] - posx[i];
                const double dy = sy[jj] - posy[i];
                const double dz = sz[jj] - posz[i];
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

    // --- integrate: update velocity then position ---
    const double newvx = velx[i] + DT * Fx;
    const double newvy = vely[i] + DT * Fy;
    const double newvz = velz[i] + DT * Fz;

    velx[i] = newvx;
    vely[i] = newvy;
    velz[i] = newvz;

    newposx[i] = posx[i] + newvx * DT;
    newposy[i] = posy[i] + newvy * DT;
    newposz[i] = posz[i] + newvz * DT;
}

static void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.x  = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.y  = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.z  = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vx = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vy = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vz = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

static double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vx * body.vx + body.vy * body.vy + body.vz * body.vz);
    }

    // Potential energy (unit mass)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].x - bodies[i].x;
            const double dy = bodies[j].y - bodies[i].y;
            const double dz = bodies[j].z - bodies[i].z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    return energy;
}

static bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.x) || !std::isfinite(body.y) ||
            !std::isfinite(body.z) || !std::isfinite(body.vx) ||
            !std::isfinite(body.vy) || !std::isfinite(body.vz)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        const double maxPos = 1e6, maxVel = 1e6;
        if (std::abs(body.x) > maxPos || std::abs(body.y) > maxPos ||
            std::abs(body.z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vx) > maxVel || std::abs(body.vy) > maxVel ||
            std::abs(body.vz) > maxVel) {
            printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks energy conservation)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // ---- host-side initialization ----
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // ---- device allocation (SoA layout) ----
    double *d_posx, *d_posy, *d_posz;
    double *d_velx, *d_vely, *d_velz;
    double *d_newposx, *d_newposy, *d_newposz;

    const size_t bytes = numBodies * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_posx, bytes));
    CUDA_CHECK(cudaMalloc(&d_posy, bytes));
    CUDA_CHECK(cudaMalloc(&d_posz, bytes));
    CUDA_CHECK(cudaMalloc(&d_velx, bytes));
    CUDA_CHECK(cudaMalloc(&d_vely, bytes));
    CUDA_CHECK(cudaMalloc(&d_velz, bytes));
    CUDA_CHECK(cudaMalloc(&d_newposx, bytes));
    CUDA_CHECK(cudaMalloc(&d_newposy, bytes));
    CUDA_CHECK(cudaMalloc(&d_newposz, bytes));

    // Prepare SoA arrays for device copy
    std::vector<double> h_posx(numBodies), h_posy(numBodies), h_posz(numBodies);
    std::vector<double> h_velx(numBodies), h_vely(numBodies), h_velz(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        h_posx[i] = bodies[i].x;
        h_posy[i] = bodies[i].y;
        h_posz[i] = bodies[i].z;
        h_velx[i] = bodies[i].vx;
        h_vely[i] = bodies[i].vy;
        h_velz[i] = bodies[i].vz;
    }

    CUDA_CHECK(cudaMemcpy(d_posx, h_posx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posy, h_posy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posz, h_posz.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velx, h_velx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vely, h_vely.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velz, h_velz.data(), bytes, cudaMemcpyHostToDevice));

    // ---- kernel launch configuration ----
    const int blockSize = 256;
    const int gridSize  = (numBodies + blockSize - 1) / blockSize;
    const size_t sharedMemSize = 3 * blockSize * sizeof(double);

    // ---- timed simulation loop ----
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesAndIntegrate<<<gridSize, blockSize, sharedMemSize>>>(
            d_posx, d_posy, d_posz,
            d_velx, d_vely, d_velz,
            d_newposx, d_newposy, d_newposz,
            numBodies);
        CUDA_CHECK(cudaGetLastError());

        // Pointer swap: old positions become the "new" buffer and vice versa
        double* tmp = d_posx; d_posx = d_newposx; d_newposx = tmp;
        tmp = d_posy; d_posy = d_newposy; d_newposy = tmp;
        tmp = d_posz; d_posz = d_newposz; d_newposz = tmp;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // ---- copy results back to host ----
    CUDA_CHECK(cudaMemcpy(h_posx.data(), d_posx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_posy.data(), d_posy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_posz.data(), d_posz, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_velx.data(), d_velx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vely.data(), d_vely, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_velz.data(), d_velz, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < numBodies; ++i) {
        bodies[i].x  = h_posx[i];
        bodies[i].y  = h_posy[i];
        bodies[i].z  = h_posz[i];
        bodies[i].vx = h_velx[i];
        bodies[i].vy = h_vely[i];
        bodies[i].vz = h_velz[i];
    }

    // ---- print results ----
    if (printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (const auto& body : bodies) {
            bodyData.push_back(body.x);
            bodyData.push_back(body.y);
            bodyData.push_back(body.z);
            bodyData.push_back(body.vx);
            bodyData.push_back(body.vy);
            bodyData.push_back(body.vz);
        }
        print_results(bodyData, "Bodies");
    }

    // ---- validation ----
    bool validationPassed = true;
    if (validate) {
        printf("Validating simulation results...\n");
        if (validateSimulation(bodies)) {
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            validationPassed = false;
            printf("Validation: FAILED\n");
        }
    }

    // ---- cleanup ----
    CUDA_CHECK(cudaFree(d_posx));
    CUDA_CHECK(cudaFree(d_posy));
    CUDA_CHECK(cudaFree(d_posz));
    CUDA_CHECK(cudaFree(d_velx));
    CUDA_CHECK(cudaFree(d_vely));
    CUDA_CHECK(cudaFree(d_velz));
    CUDA_CHECK(cudaFree(d_newposx));
    CUDA_CHECK(cudaFree(d_newposy));
    CUDA_CHECK(cudaFree(d_newposz));

    return validationPassed ? 0 : 1;
}
