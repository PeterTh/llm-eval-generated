// N-Body benchmark, CUDA GPU-parallel version.
//
// Parallelization strategy:
//  * Bodies are stored on the device as structure-of-arrays double buffers
//    (coalesced loads/stores).
//  * The O(N^2) force computation is a tiled all-pairs kernel: each thread
//    owns one body i, and the j-loop is staged through shared memory one
//    tile at a time.  To keep the GPU saturated for small N, the j-range is
//    additionally split across gridDim.y slices; each slice writes a partial
//    force, and a fixed-order reduction kernel sums the slices
//    (deterministic, no atomics).
//  * Integration is a trivial per-body kernel.
//  * Total-energy (validation) uses a block-reduction kernel for the O(N^2)
//    potential term plus the O(N) kinetic term.
//  * Initialization (rand_r) stays sequential on the host to preserve the
//    exact original initial conditions.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                          \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

// ---------------------------------------------------------------------------
// Device data layout (structure of arrays)
// ---------------------------------------------------------------------------
struct DeviceBodies {
    double* px; double* py; double* pz;
    double* vx; double* vy; double* vz;
    int n;
};

constexpr int FORCE_BLOCK = 256;   // threads per block for the force kernel
constexpr int ENERGY_BLOCK = 256;  // threads per block for the energy kernel

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Computes, for each body i handled by this block, the force contribution of
// all bodies j in [jBegin, jEnd) where the j-range is slice blockIdx.y of
// gridDim.y equal slices.  Partial results go to f{x,y,z}[blockIdx.y * n + i].
__global__ void __launch_bounds__(FORCE_BLOCK)
forceKernel(const double* __restrict__ px, const double* __restrict__ py, const double* __restrict__ pz,
            double* __restrict__ fx, double* __restrict__ fy, double* __restrict__ fz, int n)
{
    __shared__ double sx[FORCE_BLOCK];
    __shared__ double sy[FORCE_BLOCK];
    __shared__ double sz[FORCE_BLOCK];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // j-slice handled by this block row
    const int slices = gridDim.y;
    const int jBegin = (int)(((long long)n * blockIdx.y) / slices);
    const int jEnd   = (int)(((long long)n * (blockIdx.y + 1)) / slices);

    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (i < n) { xi = px[i]; yi = py[i]; zi = pz[i]; }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = jBegin; tile < jEnd; tile += FORCE_BLOCK) {
        const int jl = tile + threadIdx.x;
        if (jl < jEnd) {
            sx[threadIdx.x] = px[jl];
            sy[threadIdx.x] = py[jl];
            sz[threadIdx.x] = pz[jl];
        }
        __syncthreads();

        const int count = min(FORCE_BLOCK, jEnd - tile);
        if (count == FORCE_BLOCK) {
            // full tile: unrolled, no bounds check
#pragma unroll 8
            for (int k = 0; k < FORCE_BLOCK; ++k) {
                const double dx = sx[k] - xi;
                const double dy = sy[k] - yi;
                const double dz = sz[k] - zi;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = rsqrt(distSqr);  // double-precision rsqrt: <=1 ulp, same as 1.0/sqrt() within rounding
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        } else {
            for (int k = 0; k < count; ++k) {
                const double dx = sx[k] - xi;
                const double dy = sy[k] - yi;
                const double dz = sz[k] - zi;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = rsqrt(distSqr);  // double-precision rsqrt: <=1 ulp, same as 1.0/sqrt() within rounding
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        const size_t out = (size_t)blockIdx.y * n + i;
        fx[out] = Fx;
        fy[out] = Fy;
        fz[out] = Fz;
    }
}

// Sums the per-slice partial forces in a fixed order (deterministic), applies
// the velocity kick, and then the position drift.  This fuses the tail of
// computeForces() with integrateBodies(); the semantics are identical because
// the original integrateBodies() runs only after *all* velocities are updated,
// and positions used by forceKernel are from before the step.
__global__ void reduceAndIntegrateKernel(const double* __restrict__ fx, const double* __restrict__ fy,
                                         const double* __restrict__ fz, int slices,
                                         double* __restrict__ px, double* __restrict__ py, double* __restrict__ pz,
                                         double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
                                         int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int s = 0; s < slices; ++s) {
        const size_t idx = (size_t)s * n + i;
        Fx += fx[idx];
        Fy += fy[idx];
        Fz += fz[idx];
    }

    const double nvx = vx[i] + DT * Fx;
    const double nvy = vy[i] + DT * Fy;
    const double nvz = vz[i] + DT * Fz;
    vx[i] = nvx; vy[i] = nvy; vz[i] = nvz;

    px[i] += nvx * DT;
    py[i] += nvy * DT;
    pz[i] += nvz * DT;
}

// Potential energy: each thread owns body i and sums -1/dist over j > i
// (tiles staged through shared memory); block-reduce to one partial per block.
__global__ void __launch_bounds__(ENERGY_BLOCK)
potentialEnergyKernel(const double* __restrict__ px, const double* __restrict__ py, const double* __restrict__ pz,
                      double* __restrict__ partial, int n)
{
    __shared__ double sx[ENERGY_BLOCK];
    __shared__ double sy[ENERGY_BLOCK];
    __shared__ double sz[ENERGY_BLOCK];
    __shared__ double red[ENERGY_BLOCK];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (i < n) { xi = px[i]; yi = py[i]; zi = pz[i]; }

    double e = 0.0;
    // Only tiles that can contain j > i for some i in this block
    const int firstTile = blockIdx.x * blockDim.x;
    for (int tile = firstTile; tile < n; tile += ENERGY_BLOCK) {
        const int jl = tile + threadIdx.x;
        if (jl < n) { sx[threadIdx.x] = px[jl]; sy[threadIdx.x] = py[jl]; sz[threadIdx.x] = pz[jl]; }
        __syncthreads();
        const int count = min(ENERGY_BLOCK, n - tile);
        if (i < n) {
            for (int k = 0; k < count; ++k) {
                const int j = tile + k;
                if (j > i) {
                    const double dx = sx[k] - xi;
                    const double dy = sy[k] - yi;
                    const double dz = sz[k] - zi;
                    const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                    e -= 1.0 / dist;
                }
            }
        }
        __syncthreads();
    }

    red[threadIdx.x] = e;
    __syncthreads();
    for (int s = ENERGY_BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = red[0];
}

// ---------------------------------------------------------------------------
// Host helpers
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

static inline int divUp(int a, int b) { return (a + b - 1) / b; }

// Choose how many j-slices to use so that the force grid has enough blocks
// to fill the GPU even for small N.
static int chooseSlices(int n, int numSMs) {
    if (n <= 0) return 1;
    const int iBlocks = divUp(n, FORCE_BLOCK);
    const int targetBlocks = numSMs * 4;
    int slices = 1;
    if (iBlocks < targetBlocks) {
        slices = divUp(targetBlocks, iBlocks);
        // Each slice should still have at least one full tile of work.
        const int maxSlices = std::max(1, n / FORCE_BLOCK);
        slices = std::min(slices, maxSlices);
    }
    return std::max(1, slices);
}

class GpuSimulation {
  public:
    explicit GpuSimulation(const std::vector<Body>& bodies) : n_((int)bodies.size()) {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        slices_ = chooseSlices(n_, prop.multiProcessorCount);

        const size_t bytes = (size_t)n_ * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_.px, bytes)); CUDA_CHECK(cudaMalloc(&d_.py, bytes)); CUDA_CHECK(cudaMalloc(&d_.pz, bytes));
        CUDA_CHECK(cudaMalloc(&d_.vx, bytes)); CUDA_CHECK(cudaMalloc(&d_.vy, bytes)); CUDA_CHECK(cudaMalloc(&d_.vz, bytes));
        d_.n = n_;
        const size_t fbytes = (size_t)slices_ * n_ * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_fx_, fbytes)); CUDA_CHECK(cudaMalloc(&d_fy_, fbytes)); CUDA_CHECK(cudaMalloc(&d_fz_, fbytes));

        upload(bodies);
    }

    ~GpuSimulation() {
        cudaFree(d_.px); cudaFree(d_.py); cudaFree(d_.pz);
        cudaFree(d_.vx); cudaFree(d_.vy); cudaFree(d_.vz);
        cudaFree(d_fx_); cudaFree(d_fy_); cudaFree(d_fz_);
    }

    void upload(const std::vector<Body>& bodies) {
        std::vector<double> h(6 * (size_t)n_);
        for (int i = 0; i < n_; ++i) {
            h[0 * n_ + i] = bodies[i].pos.x; h[1 * n_ + i] = bodies[i].pos.y; h[2 * n_ + i] = bodies[i].pos.z;
            h[3 * n_ + i] = bodies[i].vel.x; h[4 * n_ + i] = bodies[i].vel.y; h[5 * n_ + i] = bodies[i].vel.z;
        }
        const size_t bytes = (size_t)n_ * sizeof(double);
        CUDA_CHECK(cudaMemcpy(d_.px, h.data() + 0 * n_, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_.py, h.data() + 1 * n_, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_.pz, h.data() + 2 * n_, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_.vx, h.data() + 3 * n_, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_.vy, h.data() + 4 * n_, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_.vz, h.data() + 5 * n_, bytes, cudaMemcpyHostToDevice));
    }

    void download(std::vector<Body>& bodies) const {
        std::vector<double> h(6 * (size_t)n_);
        const size_t bytes = (size_t)n_ * sizeof(double);
        CUDA_CHECK(cudaMemcpy(h.data() + 0 * n_, d_.px, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.data() + 1 * n_, d_.py, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.data() + 2 * n_, d_.pz, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.data() + 3 * n_, d_.vx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.data() + 4 * n_, d_.vy, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.data() + 5 * n_, d_.vz, bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < n_; ++i) {
            bodies[i].pos.x = h[0 * n_ + i]; bodies[i].pos.y = h[1 * n_ + i]; bodies[i].pos.z = h[2 * n_ + i];
            bodies[i].vel.x = h[3 * n_ + i]; bodies[i].vel.y = h[4 * n_ + i]; bodies[i].vel.z = h[5 * n_ + i];
        }
    }

    // One full time step: computeForces (velocity kick) + integrateBodies.
    void step() {
        if (n_ == 0) return;
        const dim3 fgrid(divUp(n_, FORCE_BLOCK), slices_);
        forceKernel<<<fgrid, FORCE_BLOCK>>>(d_.px, d_.py, d_.pz, d_fx_, d_fy_, d_fz_, n_);
        reduceAndIntegrateKernel<<<divUp(n_, 256), 256>>>(d_fx_, d_fy_, d_fz_, slices_,
                                                          d_.px, d_.py, d_.pz, d_.vx, d_.vy, d_.vz, n_);
    }

    void synchronize() const { CUDA_CHECK(cudaDeviceSynchronize()); }

    // Potential-energy term (sum over i<j of -1/dist) computed on the GPU.
    double potentialEnergy() const {
        if (n_ == 0) return 0.0;
        const int blocks = divUp(n_, ENERGY_BLOCK);
        double* d_partial = nullptr;
        CUDA_CHECK(cudaMalloc(&d_partial, (size_t)blocks * sizeof(double)));
        potentialEnergyKernel<<<blocks, ENERGY_BLOCK>>>(d_.px, d_.py, d_.pz, d_partial, n_);
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> partial(blocks);
        CUDA_CHECK(cudaMemcpy(partial.data(), d_partial, (size_t)blocks * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_partial));
        double e = 0.0;
        for (double v : partial) e += v;
        return e;
    }

    int slices() const { return slices_; }

  private:
    int n_;
    int slices_ = 1;
    DeviceBodies d_{};
    double* d_fx_ = nullptr;
    double* d_fy_ = nullptr;
    double* d_fz_ = nullptr;
};

double computeTotalEnergy(const std::vector<Body>& bodies, const GpuSimulation& sim) {
    double energy = 0.0;

    // Kinetic energy (assuming unit mass) -- O(N), done on host
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies) -- O(N^2), on GPU
    energy += sim.potentialEnergy();

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

    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Set up GPU state (context creation, allocation, upload) outside the timed region
    GpuSimulation sim(bodies);
    sim.synchronize();

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        sim.step();
    }
    CUDA_CHECK(cudaGetLastError());
    sim.synchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    sim.download(bodies);

    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
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

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies, sim);
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
