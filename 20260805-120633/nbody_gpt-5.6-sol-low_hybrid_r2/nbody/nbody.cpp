#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
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
    constexpr Vec3(double x = 0, double y = 0, double z = 0) noexcept : x(x), y(y), z(z) {}
};
struct Body { Vec3 pos; Vec3 vel; };

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t error_ = (call);                                                \
    if (error_ != cudaSuccess) {                                                \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                     cudaGetErrorString(error_));                               \
        MPI_Abort(MPI_COMM_WORLD, 2);                                           \
    }                                                                           \
} while (0)

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Keep the original serial PRNG sequence, making results independent of rank count.
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// One rank computes only its target bodies.  Tiling source positions in shared
// memory removes almost all redundant global loads while retaining the original
// per-target summation order.
__global__ void advanceBodies(const Body* __restrict__ all,
                              Body* __restrict__ local, int first, int count,
                              int n) {
    extern __shared__ Vec3 tile[];
    const int lane = threadIdx.x;
    const int li = blockIdx.x * blockDim.x + lane;
    const int i = first + li;
    double px = 0, py = 0, pz = 0, fx = 0, fy = 0, fz = 0;
    if (li < count) {
        px = all[i].pos.x; py = all[i].pos.y; pz = all[i].pos.z;
    }
    for (int base = 0; base < n; base += blockDim.x) {
        const int j = base + lane;
        if (j < n) tile[lane] = all[j].pos;
        __syncthreads();
        if (li < count) {
            const int limit = min(blockDim.x, n - base);
            #pragma unroll 4
            for (int k = 0; k < limit; ++k) {
                const double dx = tile[k].x - px;
                const double dy = tile[k].y - py;
                const double dz = tile[k].z - pz;
                const double inv = rsqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
                const double inv3 = inv * inv * inv;
                fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
            }
        }
        __syncthreads();
    }
    if (li < count) {
        Body b = all[i];
        b.vel.x += DT * fx; b.vel.y += DT * fy; b.vel.z += DT * fz;
        b.pos.x += DT * b.vel.x; b.pos.y += DT * b.vel.y; b.pos.z += DT * b.vel.z;
        local[li] = b;
    }
}

double computeTotalEnergy(const std::vector<Body>& b) {
    const long long n = static_cast<long long>(b.size());
    double kinetic = 0.0, potential = 0.0;
    #pragma omp parallel for reduction(+:kinetic) schedule(static)
    for (long long i = 0; i < n; ++i)
        kinetic += .5 * (b[i].vel.x*b[i].vel.x + b[i].vel.y*b[i].vel.y + b[i].vel.z*b[i].vel.z);
    #pragma omp parallel for reduction(+:potential) schedule(dynamic,8)
    for (long long i = 0; i < n; ++i)
        for (long long j = i + 1; j < n; ++j) {
            const double dx=b[j].pos.x-b[i].pos.x, dy=b[j].pos.y-b[i].pos.y, dz=b[j].pos.z-b[i].pos.z;
            potential -= 1.0/std::sqrt(dx*dx+dy*dy+dz*dz+SOFTENING);
        }
    return kinetic + potential;
}

bool validateSimulation(const std::vector<Body>& b) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid)
    for (long long i = 0; i < static_cast<long long>(b.size()); ++i) {
        const Body& x=b[i];
        valid &= std::isfinite(x.pos.x) && std::isfinite(x.pos.y) && std::isfinite(x.pos.z) &&
                 std::isfinite(x.vel.x) && std::isfinite(x.vel.y) && std::isfinite(x.vel.z) &&
                 std::abs(x.pos.x)<=1e6 && std::abs(x.pos.y)<=1e6 && std::abs(x.pos.z)<=1e6 &&
                 std::abs(x.vel.x)<=1e6 && std::abs(x.vel.y)<=1e6 && std::abs(x.vel.z)<=1e6;
    }
    if (!valid) std::printf("Validation failed: non-finite or out-of-bounds body state\n");
    return valid != 0;
}

void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num>  Number of bodies (default: 1024)\n"
                "  -s <num>  Number of simulation steps (default: 10)\n"
                "  -v        Enable validation\n  -r        Print results for external validation\n"
                "  -h        Show this help message\n", p);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank=0, ranks=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 3);

    int n=1024, steps=10; bool validate=false, printResults=false, help=false, bad=false;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i],"-n") && i+1<argc) n=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"-s") && i+1<argc) steps=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"-v")) validate=true;
        else if (!std::strcmp(argv[i],"-r")) printResults=true;
        else if (!std::strcmp(argv[i],"-h")) help=true;
        else bad=true;
    }
    if (help || bad || n<1 || steps<0) {
        if (rank==0) { if (bad) std::printf("Unknown or invalid option\n"); printUsage(argv[0]); }
        MPI_Finalize(); return bad || n<1 || steps<0;
    }

    MPI_Comm localComm; MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&localComm);
    int localRank=0, deviceCount=0; MPI_Comm_rank(localComm,&localRank); MPI_Comm_free(&localComm);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { if(rank==0) std::fprintf(stderr,"A CUDA device is required\n"); MPI_Abort(MPI_COMM_WORLD,4); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    std::vector<int> counts(ranks), offsets(ranks), byteCounts(ranks), byteOffsets(ranks);
    for (int r=0; r<ranks; ++r) {
        counts[r] = n/ranks + (r < n%ranks); offsets[r] = r*(n/ranks) + std::min(r,n%ranks);
        byteCounts[r]=counts[r]*static_cast<int>(sizeof(Body)); byteOffsets[r]=offsets[r]*static_cast<int>(sizeof(Body));
    }
    const int localN=counts[rank], first=offsets[rank];
    std::vector<Body> bodies(n); if(rank==0) randomizeBodies(bodies);
    MPI_Bcast(bodies.data(), n*static_cast<int>(sizeof(Body)), MPI_BYTE, 0, MPI_COMM_WORLD);

    Body *dAll=nullptr, *dLocal=nullptr; Body* hostLocal=nullptr;
    CUDA_CHECK(cudaMalloc(&dAll, n*sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&dLocal, std::max(1,localN)*sizeof(Body)));
    CUDA_CHECK(cudaMallocHost(&hostLocal, std::max(1,localN)*sizeof(Body)));
    CUDA_CHECK(cudaMemcpy(dAll,bodies.data(),n*sizeof(Body),cudaMemcpyHostToDevice));
    constexpr int threads=128; const int blocks=(localN+threads-1)/threads;
    MPI_Barrier(MPI_COMM_WORLD); const double start=MPI_Wtime();
    for (int step=0; step<steps; ++step) {
        if (localN) advanceBodies<<<blocks,threads,threads*sizeof(Vec3)>>>(dAll,dLocal,first,localN,n);
        CUDA_CHECK(cudaGetLastError());
        if (localN) CUDA_CHECK(cudaMemcpy(hostLocal,dLocal,localN*sizeof(Body),cudaMemcpyDeviceToHost));
        MPI_Allgatherv(hostLocal,byteCounts[rank],MPI_BYTE,bodies.data(),byteCounts.data(),byteOffsets.data(),MPI_BYTE,MPI_COMM_WORLD);
        if (step+1<steps) CUDA_CHECK(cudaMemcpy(dAll,bodies.data(),n*sizeof(Body),cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed=MPI_Wtime()-start, maxElapsed=0; MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",n,steps,validate?"enabled":"disabled");
        std::printf("Simulation time: %ld ms\n",static_cast<long>(maxElapsed*1000.0));
        if(printResults) {
            std::vector<double> data(static_cast<size_t>(n)*6);
            #pragma omp parallel for schedule(static)
            for(int i=0;i<n;++i) { data[6*i]=bodies[i].pos.x; data[6*i+1]=bodies[i].pos.y; data[6*i+2]=bodies[i].pos.z; data[6*i+3]=bodies[i].vel.x; data[6*i+4]=bodies[i].vel.y; data[6*i+5]=bodies[i].vel.z; }
            print_results(data,"Bodies");
        }
    }
    int result=0;
    if(rank==0 && validate) {
        std::printf("Validating simulation results...\n");
        if(validateSimulation(bodies)) { std::printf("Final energy: %.6f\nValidation: PASSED\n",computeTotalEnergy(bodies)); }
        else { std::printf("Validation: FAILED\n"); result=1; }
    }
    MPI_Bcast(&result,1,MPI_INT,0,MPI_COMM_WORLD);
    CUDA_CHECK(cudaFreeHost(hostLocal)); CUDA_CHECK(cudaFree(dLocal)); CUDA_CHECK(cudaFree(dAll));
    MPI_Finalize(); return result;
}
