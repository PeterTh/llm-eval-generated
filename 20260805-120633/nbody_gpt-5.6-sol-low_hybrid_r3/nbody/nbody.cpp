#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos, vel; };

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t e_ = (call);                                                     \
    if (e_ != cudaSuccess) {                                                     \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                     cudaGetErrorString(e_));                                    \
        MPI_Abort(MPI_COMM_WORLD, 2);                                            \
    }                                                                            \
} while (0)

static void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& b : bodies) {
        b.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// The read-only global state is tiled through shared memory.  Only the rank's
// contiguous output interval is updated, giving coalesced accesses and no atomics.
__global__ void forceKernel(Body* local, const Body* all, int count, int n) {
    extern __shared__ Body tile[];
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = li < count;
    const Vec3 pi = active ? local[li].pos : Vec3{0.0, 0.0, 0.0};
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += blockDim.x) {
        const int jload = base + threadIdx.x;
        if (jload < n) tile[threadIdx.x] = all[jload];
        __syncthreads();
        const int limit = min((int)blockDim.x, n - base);
#pragma unroll 4
        for (int j = 0; active && j < limit; ++j) {
            const double dx = tile[j].pos.x - pi.x;
            const double dy = tile[j].pos.y - pi.y;
            const double dz = tile[j].pos.z - pi.z;
            const double inv = 1.0 / sqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
            const double inv3 = inv * inv * inv;
            fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
        }
        __syncthreads();
    }
    if (active) {
        local[li].vel.x += DT * fx;
        local[li].vel.y += DT * fy;
        local[li].vel.z += DT * fz;
    }
}

__global__ void integrateKernel(Body* local, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        local[i].pos.x += local[i].vel.x * DT;
        local[i].pos.y += local[i].vel.y * DT;
        local[i].pos.z += local[i].vel.z * DT;
    }
}

static bool validateSimulation(const std::vector<Body>& bodies) {
    int ok = 1;
#pragma omp parallel for reduction(&:ok) schedule(static)
    for (long long i = 0; i < (long long)bodies.size(); ++i) {
        const Body& b = bodies[i];
        ok &= std::isfinite(b.pos.x) && std::isfinite(b.pos.y) && std::isfinite(b.pos.z) &&
              std::isfinite(b.vel.x) && std::isfinite(b.vel.y) && std::isfinite(b.vel.z) &&
              std::abs(b.pos.x) <= 1e6 && std::abs(b.pos.y) <= 1e6 && std::abs(b.pos.z) <= 1e6 &&
              std::abs(b.vel.x) <= 1e6 && std::abs(b.vel.y) <= 1e6 && std::abs(b.vel.z) <= 1e6;
    }
    return ok != 0;
}

static double computeTotalEnergy(const std::vector<Body>& b) {
    double energy = 0.0;
#pragma omp parallel for reduction(+:energy) schedule(static)
    for (long long i = 0; i < (long long)b.size(); ++i)
        energy += .5 * (b[i].vel.x*b[i].vel.x + b[i].vel.y*b[i].vel.y + b[i].vel.z*b[i].vel.z);
#pragma omp parallel for reduction(+:energy) schedule(dynamic,8)
    for (long long i = 0; i < (long long)b.size(); ++i)
        for (size_t j = i + 1; j < b.size(); ++j) {
            const double x=b[j].pos.x-b[i].pos.x, y=b[j].pos.y-b[i].pos.y, z=b[j].pos.z-b[i].pos.z;
            energy -= 1.0 / std::sqrt(x*x+y*y+z*z+SOFTENING);
        }
    return energy;
}

static void usage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num>  Number of bodies (default: 1024)\n"
                "  -s <num>  Number of steps (default: 10)\n  -v  Enable validation\n"
                "  -r  Print results for external validation\n  -h  Show help\n", p);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 3);

    int n = 1024, steps = 10; bool validate = false, results = false, help = false, bad = false;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i+1<argc) n=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-s") && i+1<argc) steps=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate=true;
        else if (!std::strcmp(argv[i], "-r")) results=true;
        else if (!std::strcmp(argv[i], "-h")) help=true;
        else bad=true;
    }
    if (help || bad || n < 1 || steps < 0) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize(); return bad || n < 1 || steps < 0;
    }

    // Select GPUs by node-local rank, not global rank.
    MPI_Comm localComm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, devices=0; MPI_Comm_rank(localComm, &localRank); CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) std::fprintf(stderr, "nbody requires CUDA accelerators\n"); MPI_Abort(MPI_COMM_WORLD, 4); }
    CUDA_CHECK(cudaSetDevice(localRank % devices)); MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), offsets(ranks), byteCounts(ranks), byteOffsets(ranks);
    for (int r=0; r<ranks; ++r) {
        counts[r] = n/ranks + (r < n%ranks); offsets[r] = r*(n/ranks) + (r < n%ranks ? r : n%ranks);
        byteCounts[r] = counts[r] * sizeof(Body); byteOffsets[r] = offsets[r] * sizeof(Body);
    }
    const int mine=counts[rank], begin=offsets[rank];
    std::vector<Body> all(n), local(mine);
    if (!rank) randomizeBodies(all);
    MPI_Bcast(all.data(), n*sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);
#pragma omp parallel for schedule(static)
    for (int i=0; i<mine; ++i) local[i] = all[begin+i];

    // Page-lock MPI staging buffers so PCIe/NVLink transfers use DMA directly.
    CUDA_CHECK(cudaHostRegister(all.data(), n*sizeof(Body), cudaHostRegisterPortable));
    if (mine) CUDA_CHECK(cudaHostRegister(local.data(), mine*sizeof(Body), cudaHostRegisterPortable));

    Body *dAll=nullptr, *dLocal=nullptr;
    CUDA_CHECK(cudaMalloc(&dAll, n*sizeof(Body))); CUDA_CHECK(cudaMalloc(&dLocal, (mine ? mine : 1)*sizeof(Body)));
    CUDA_CHECK(cudaMemcpy(dAll, all.data(), n*sizeof(Body), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dLocal, local.data(), mine*sizeof(Body), cudaMemcpyHostToDevice));
    constexpr int block=256; const int grid=(mine+block-1)/block;
    MPI_Barrier(MPI_COMM_WORLD); const double start=MPI_Wtime();
    for (int s=0; s<steps; ++s) {
        if (mine) {
            forceKernel<<<grid,block,block*sizeof(Body)>>>(dLocal,dAll,mine,n);
            integrateKernel<<<grid,block>>>(dLocal,mine);
            CUDA_CHECK(cudaMemcpy(local.data(),dLocal,mine*sizeof(Body),cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(local.data(), mine*sizeof(Body), MPI_BYTE, all.data(), byteCounts.data(), byteOffsets.data(), MPI_BYTE, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dAll,all.data(),n*sizeof(Body),cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize()); double elapsed=MPI_Wtime()-start, maximum=0;
    MPI_Reduce(&elapsed,&maximum,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    int rc=0;
    if (!rank) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n", n,steps,validate?"enabled":"disabled");
        std::printf("Simulation time: %ld ms\n", (long)(maximum*1000.0));
        if (results) { std::vector<double> data(6ULL*n);
#pragma omp parallel for schedule(static)
            for (int i=0;i<n;++i) { data[6*i]=all[i].pos.x; data[6*i+1]=all[i].pos.y; data[6*i+2]=all[i].pos.z; data[6*i+3]=all[i].vel.x; data[6*i+4]=all[i].vel.y; data[6*i+5]=all[i].vel.z; }
            print_results(data,"Bodies");
        }
        if (validate) { std::printf("Validating simulation results...\n"); rc=!validateSimulation(all);
            if (!rc) std::printf("Final energy: %.6f\nValidation: PASSED\n",computeTotalEnergy(all));
            else std::printf("Validation: FAILED\n"); }
    }
    MPI_Bcast(&rc,1,MPI_INT,0,MPI_COMM_WORLD);
    cudaFree(dLocal); cudaFree(dAll);
    if (mine) cudaHostUnregister(local.data());
    cudaHostUnregister(all.data());
    MPI_Finalize(); return rc;
}
