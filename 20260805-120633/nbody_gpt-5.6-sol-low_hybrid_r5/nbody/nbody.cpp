#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

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
    cudaError_t e_ = (call);                                                    \
    if (e_ != cudaSuccess) {                                                    \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(e_));                                   \
        MPI_Abort(MPI_COMM_WORLD, 2);                                           \
    }                                                                           \
} while (0)

static void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // rand_r is stateful, so generate exactly the original sequence first.
    for (Body& b : bodies) {
        b.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

__global__ void forceKernel(Body* __restrict__ bodies, int n, int begin, int count) {
    extern __shared__ double tile[];
    double* sx = tile;
    double* sy = tile + blockDim.x;
    double* sz = tile + 2 * blockDim.x;
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = begin + local;
    const bool active = local < count;
    const double ix = active ? bodies[i].pos.x : 0.0;
    const double iy = active ? bodies[i].pos.y : 0.0;
    const double iz = active ? bodies[i].pos.z : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += blockDim.x) {
        const int source = base + threadIdx.x;
        if (source < n) {
            sx[threadIdx.x] = bodies[source].pos.x;
            sy[threadIdx.x] = bodies[source].pos.y;
            sz[threadIdx.x] = bodies[source].pos.z;
        }
        __syncthreads();
        const int width = min((int)blockDim.x, n - base);
        if (active) for (int j = 0; j < width; ++j) {
            const double dx = sx[j] - ix, dy = sy[j] - iy, dz = sz[j] - iz;
            const double inv = rsqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            const double inv3 = inv * inv * inv;
            fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
        }
        __syncthreads();
    }
    if (active) {
        bodies[i].vel.x += DT * fx;
        bodies[i].vel.y += DT * fy;
        bodies[i].vel.z += DT * fz;
    }
}

__global__ void integrateKernel(Body* bodies, int begin, int count) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;
    Body& b = bodies[begin + local];
    b.pos.x += b.vel.x * DT;
    b.pos.y += b.vel.y * DT;
    b.pos.z += b.vel.z * DT;
}

static bool validateSimulation(const std::vector<Body>& bodies) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const Body& b = bodies[i];
        valid &= std::isfinite(b.pos.x) && std::isfinite(b.pos.y) && std::isfinite(b.pos.z) &&
                 std::isfinite(b.vel.x) && std::isfinite(b.vel.y) && std::isfinite(b.vel.z) &&
                 std::abs(b.pos.x) <= 1e6 && std::abs(b.pos.y) <= 1e6 && std::abs(b.pos.z) <= 1e6 &&
                 std::abs(b.vel.x) <= 1e6 && std::abs(b.vel.y) <= 1e6 && std::abs(b.vel.z) <= 1e6;
    }
    return valid != 0;
}

static double computeTotalEnergy(const std::vector<Body>& b) {
    double energy = 0.0;
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (std::size_t i = 0; i < b.size(); ++i) {
        energy += .5 * (b[i].vel.x*b[i].vel.x + b[i].vel.y*b[i].vel.y + b[i].vel.z*b[i].vel.z);
        for (std::size_t j = i + 1; j < b.size(); ++j) {
            const double dx=b[j].pos.x-b[i].pos.x, dy=b[j].pos.y-b[i].pos.y, dz=b[j].pos.z-b[i].pos.z;
            energy -= 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
        }
    }
    return energy;
}

static void usage(const char* p) {
    std::printf("Usage: %s [-n bodies] [-s steps] [-v] [-r] [-h]\n", p);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 3);

    int n = 1024, steps = 10; bool validate = false, results = false, bad = false, help = false;
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

    // Assign one rank per GPU on each node.
    MPI_Comm localComm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank=0, devices=0; MPI_Comm_rank(localComm, &localRank); CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if(rank==0) std::fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    CUDA_CHECK(cudaSetDevice(localRank % devices)); MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), offsets(ranks), byteCounts(ranks), byteOffsets(ranks);
    for (int r=0, off=0; r<ranks; ++r) {
        counts[r] = n/ranks + (r < n%ranks); offsets[r]=off; off += counts[r];
        byteCounts[r]=counts[r]*(int)sizeof(Body); byteOffsets[r]=offsets[r]*(int)sizeof(Body);
    }
    const int begin=offsets[rank], count=counts[rank];
    std::vector<Body> bodies(n);
    if (rank==0) randomizeBodies(bodies);
    MPI_Bcast(bodies.data(), n*(int)sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);

    Body* deviceBodies=nullptr; CUDA_CHECK(cudaMalloc(&deviceBodies, n*sizeof(Body)));
    CUDA_CHECK(cudaMemcpy(deviceBodies, bodies.data(), n*sizeof(Body), cudaMemcpyHostToDevice));
    if(rank==0) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\nMPI ranks: %d, OpenMP max threads: %d\n",
                    n, steps, validate?"enabled":"disabled", ranks, omp_get_max_threads());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start=std::chrono::high_resolution_clock::now();
    constexpr int block=256;
    for (int s=0; s<steps; ++s) {
        if (count) {
            forceKernel<<<(count+block-1)/block,block,3*block*sizeof(double)>>>(deviceBodies,n,begin,count);
            integrateKernel<<<(count+block-1)/block,block>>>(deviceBodies,begin,count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(bodies.data()+begin, deviceBodies+begin, count*sizeof(Body), cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bodies.data(), byteCounts.data(), byteOffsets.data(), MPI_BYTE, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(deviceBodies, bodies.data(), n*sizeof(Body), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize()); MPI_Barrier(MPI_COMM_WORLD);
    const auto end=std::chrono::high_resolution_clock::now(); CUDA_CHECK(cudaFree(deviceBodies));

    int rc=0;
    if(rank==0) {
        std::printf("Simulation time: %ld ms\n", (long)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
        if(results) {
            std::vector<double> data((std::size_t)n*6);
            #pragma omp parallel for schedule(static)
            for(int i=0;i<n;++i) { data[6*i]=bodies[i].pos.x; data[6*i+1]=bodies[i].pos.y; data[6*i+2]=bodies[i].pos.z; data[6*i+3]=bodies[i].vel.x; data[6*i+4]=bodies[i].vel.y; data[6*i+5]=bodies[i].vel.z; }
            print_results(data,"Bodies");
        }
        if(validate) {
            std::printf("Validating simulation results...\n");
            if(validateSimulation(bodies)) std::printf("Final energy: %.6f\nValidation: PASSED\n",computeTotalEnergy(bodies));
            else { std::printf("Validation: FAILED\n"); rc=1; }
        }
    }
    MPI_Bcast(&rc,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Finalize(); return rc;
}
