#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0,
                   const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// ---------------------------------------------------------------------------
// CUDA error checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Combined force computation + integration kernel.
//
// Each thread processes one body.  Body positions are tiled through shared
// memory so that the O(n^2) interaction loop enjoys cache-line reuse on the
// "j" side.  After accumulating the total gravitational force, velocity and
// position are updated in the same kernel to avoid an extra global-memory
// round-trip.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256) nbodyStepKernel(
    double* __restrict__ pos_x, double* __restrict__ pos_y,
    double* __restrict__ pos_z, double* __restrict__ vel_x,
    double* __restrict__ vel_y, double* __restrict__ vel_z, int n, double dt) {
    // Shared memory for one tile of body positions (3 * blockDim.x doubles)
    extern __shared__ double sh_pos[];
    double* sh_px = sh_pos;
    double* sh_py = sh_pos + blockDim.x;
    double* sh_pz = sh_pos + 2 * blockDim.x;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = i < n;

    // Load this body's data into registers (one load per body per step)
    double px = 0.0, py = 0.0, pz = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    if (active) {
        px = pos_x[i];
        py = pos_y[i];
        pz = pos_z[i];
        vx = vel_x[i];
        vy = vel_y[i];
        vz = vel_z[i];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    // Sweep over all tiles of the j-loop
    for (int tile = 0; tile < gridDim.x; ++tile) {
        // Cooperative load: each thread brings one body into shared memory
        int j_base = tile * blockDim.x + threadIdx.x;
        if (j_base < n) {
            sh_px[threadIdx.x] = pos_x[j_base];
            sh_py[threadIdx.x] = pos_y[j_base];
            sh_pz[threadIdx.x] = pos_z[j_base];
        } else {
            sh_px[threadIdx.x] = 0.0;
            sh_py[threadIdx.x] = 0.0;
            sh_pz[threadIdx.x] = 0.0;
        }
        __syncthreads();

        int remaining = n - tile * (int)blockDim.x;
        int j_limit = remaining > (int)blockDim.x ? (int)blockDim.x
                      : (remaining > 0 ? remaining : 0);

        // Accumulate pairwise interactions within this tile
        #pragma unroll 8
        for (int j = 0; j < j_limit; ++j) {
            double dx = sh_px[j] - px;
            double dy = sh_py[j] - py;
            double dz = sh_pz[j] - pz;
            double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            double invDist = rsqrt(distSqr);
            double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    // Update velocity and step position (only for valid bodies)
    if (active) {
        vx += dt * Fx;
        vy += dt * Fy;
        vz += dt * Fz;

        vel_x[i] = vx;
        vel_y[i] = vy;
        vel_z[i] = vz;

        pos_x[i] = px + vx * dt;
        pos_y[i] = py + vy * dt;
        pos_z[i] = pz + vz * dt;
    }
}

// ---------------------------------------------------------------------------
// Energy computation kernel (used only for validation).
// Each thread computes the kinetic + (half-)potential energy for one body.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256) computeEnergyKernel(
    const double* __restrict__ pos_x, const double* __restrict__ pos_y,
    const double* __restrict__ pos_z, const double* __restrict__ vel_x,
    const double* __restrict__ vel_y, const double* __restrict__ vel_z, int n,
    double* __restrict__ energy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Kinetic energy contribution (unit mass)
    double e = 0.5 * (vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i] +
                      vel_z[i] * vel_z[i]);
    double px = pos_x[i];
    double py = pos_y[i];
    double pz = pos_z[i];

    // Potential energy contribution (j > i to avoid double counting)
    for (int j = i + 1; j < n; ++j) {
        double dx = pos_x[j] - px;
        double dy = pos_y[j] - py;
        double dz = pos_z[j] - pz;
        double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        e -= 1.0 / dist;
    }

    energy[i] = e;
}

// ---------------------------------------------------------------------------
// Host helper: random initialization
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

// ---------------------------------------------------------------------------
// Host helper: validate simulation
// ---------------------------------------------------------------------------
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) ||
            !std::isfinite(body.pos.z) || !std::isfinite(body.vel.x) ||
            !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos ||
            std::abs(body.pos.y) > maxPos ||
            std::abs(body.pos.z) > maxPos) {
            printf(
                "Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel ||
            std::abs(body.vel.y) > maxVel ||
            std::abs(body.vel.z) > maxVel) {
            printf(
                "Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Host helper: total energy (used in validation path)
// ---------------------------------------------------------------------------
double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    for (const auto& body : bodies) {
        energy += 0.5 *
                  (body.vel.x * body.vel.x + body.vel.y * body.vel.y +
                   body.vel.z * body.vel.z);
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double dx = bodies[j].pos.x - bodies[i].pos.x;
            double dy = bodies[j].pos.y - bodies[i].pos.y;
            double dz = bodies[j].pos.z - bodies[i].pos.z;
            double dist =
                std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

// ---------------------------------------------------------------------------
// Command-line help
// ---------------------------------------------------------------------------
void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks energy conservation)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
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

    // Select fastest GPU (device 0 by default, but prefer the one with most
    // free memory if multiple are present)
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        fprintf(stderr, "No CUDA-capable device found\n");
        return 1;
    }
    int bestDev = 0;
    size_t bestFree = 0;
    for (int d = 0; d < devCount; ++d) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, d));
        size_t freeMem, totalMem;
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
        if (freeMem > bestFree) {
            bestFree = freeMem;
            bestDev = d;
        }
        printf("  GPU %d: %s  (CC %d.%d,  %.1f GB free)\n", d, prop.name,
               prop.major, prop.minor, freeMem / 1.0e9);
    }
    CUDA_CHECK(cudaSetDevice(bestDev));
    printf("Using GPU device %d\n", bestDev);

    // Initialize bodies on host
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Extract body data into flat arrays for GPU transfer
    std::vector<double> h_pos_x(numBodies), h_pos_y(numBodies),
        h_pos_z(numBodies);
    std::vector<double> h_vel_x(numBodies), h_vel_y(numBodies),
        h_vel_z(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        h_pos_x[i] = bodies[i].pos.x;
        h_pos_y[i] = bodies[i].pos.y;
        h_pos_z[i] = bodies[i].pos.z;
        h_vel_x[i] = bodies[i].vel.x;
        h_vel_y[i] = bodies[i].vel.y;
        h_vel_z[i] = bodies[i].vel.z;
    }

    // Allocate GPU memory
    const size_t bytes = (size_t)numBodies * sizeof(double);
    double *d_pos_x, *d_pos_y, *d_pos_z;
    double *d_vel_x, *d_vel_y, *d_vel_z;
    CUDA_CHECK(cudaMalloc(&d_pos_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_pos_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_pos_z, bytes));
    CUDA_CHECK(cudaMalloc(&d_vel_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_vel_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_vel_z, bytes));

    // Copy initial data to GPU
    CUDA_CHECK(cudaMemcpy(d_pos_x, h_pos_x.data(), bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_y, h_pos_y.data(), bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_z, h_pos_z.data(), bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_x, h_vel_x.data(), bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_y, h_vel_y.data(), bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel_z, h_vel_z.data(), bytes,
                          cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const int blockSize = 256;
    const int gridSize = (numBodies + blockSize - 1) / blockSize;
    // 3 arrays of double per block for shared-memory tiling
    const size_t sharedMemBytes = 3 * blockSize * sizeof(double);

    // Warm-up run (excluded from timing)
    nbodyStepKernel<<<gridSize, blockSize, sharedMemBytes>>>(
        d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, numBodies, DT);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        nbodyStepKernel<<<gridSize, blockSize, sharedMemBytes>>>(
            d_pos_x, d_pos_y, d_pos_z, d_vel_x, d_vel_y, d_vel_z, numBodies,
            DT);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Simulation time: %ld ms\n", duration.count());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_pos_x.data(), d_pos_x, bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pos_y.data(), d_pos_y, bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pos_z.data(), d_pos_z, bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vel_x.data(), d_vel_x, bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vel_y.data(), d_vel_y, bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vel_z.data(), d_vel_z, bytes,
                          cudaMemcpyDeviceToHost));

    // Restore bodies vector for output / validation
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos.x = h_pos_x[i];
        bodies[i].pos.y = h_pos_y[i];
        bodies[i].pos.z = h_pos_z[i];
        bodies[i].vel.x = h_vel_x[i];
        bodies[i].vel.y = h_vel_y[i];
        bodies[i].vel.z = h_vel_z[i];
    }

    // Clean up GPU memory
    CUDA_CHECK(cudaFree(d_pos_x));
    CUDA_CHECK(cudaFree(d_pos_y));
    CUDA_CHECK(cudaFree(d_pos_z));
    CUDA_CHECK(cudaFree(d_vel_x));
    CUDA_CHECK(cudaFree(d_vel_y));
    CUDA_CHECK(cudaFree(d_vel_z));

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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}
