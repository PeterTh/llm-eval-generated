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
static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be tightly packed");

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  std::fprintf(stderr, "CUDA failure at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
  MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)
#define MPI_CHECK(call) do { int e_ = (call); if (e_ != MPI_SUCCESS) { \
  std::fprintf(stderr, "MPI failure at %s:%d\n", __FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, e_); } } while (0)

void randomizeBodies(std::vector<Body>& bodies, unsigned seed = 42) {
  // Deliberately serial: preserving rand_r's original deterministic stream.
  for (Body& b : bodies) {
    double* p = &b.pos.x;
    for (int k = 0; k < 6; ++k) p[k] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
  }
}

template<int TILE>
__global__ void advanceBodies(Body* __restrict__ bodies, int n, int first, int count) {
  __shared__ double sx[TILE], sy[TILE], sz[TILE];
  const int local = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = first + local;
  double px = 0, py = 0, pz = 0;
  if (local < count) { px = bodies[i].pos.x; py = bodies[i].pos.y; pz = bodies[i].pos.z; }
  double fx = 0, fy = 0, fz = 0;

  for (int base = 0; base < n; base += TILE) {
    const int j = base + threadIdx.x;
    if (j < n) { sx[threadIdx.x] = bodies[j].pos.x; sy[threadIdx.x] = bodies[j].pos.y; sz[threadIdx.x] = bodies[j].pos.z; }
    __syncthreads();
    if (local < count) {
      const int end = min(TILE, n - base);
#pragma unroll 8
      for (int k = 0; k < end; ++k) {
        const double dx = sx[k] - px, dy = sy[k] - py, dz = sz[k] - pz;
        const double inv = rsqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
        const double inv3 = inv * inv * inv;
        fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
      }
    }
    __syncthreads();
  }
  if (local < count) {
    Body b = bodies[i];
    b.vel.x += DT * fx; b.vel.y += DT * fy; b.vel.z += DT * fz;
    b.pos.x += DT * b.vel.x; b.pos.y += DT * b.vel.y; b.pos.z += DT * b.vel.z;
    bodies[i] = b;
  }
}

double computeTotalEnergy(const std::vector<Body>& b) {
  double energy = 0.0;
#pragma omp parallel for reduction(+:energy) schedule(static)
  for (long long i = 0; i < (long long)b.size(); ++i) {
    energy += .5*(b[i].vel.x*b[i].vel.x+b[i].vel.y*b[i].vel.y+b[i].vel.z*b[i].vel.z);
    for (size_t j = i + 1; j < b.size(); ++j) {
      const double dx=b[j].pos.x-b[i].pos.x, dy=b[j].pos.y-b[i].pos.y, dz=b[j].pos.z-b[i].pos.z;
      energy -= 1.0/std::sqrt(dx*dx+dy*dy+dz*dz+SOFTENING);
    }
  }
  return energy;
}

bool validateSimulation(const std::vector<Body>& b) {
  int valid = 1;
#pragma omp parallel for reduction(&:valid) schedule(static)
  for (long long i=0; i<(long long)b.size(); ++i) {
    const double* p=&b[i].pos.x;
    int ok=1; for(int k=0;k<6;++k) ok &= std::isfinite(p[k]) && std::abs(p[k]) <= 1e6;
    valid &= ok;
  }
  return valid != 0;
}

void printUsage(const char* p) {
  std::printf("Usage: %s [options]\n  -n <num>  Number of bodies (default: 1024)\n  -s <num>  Steps (default: 10)\n  -v        Validate\n  -r        Print results\n  -h        Help\n",p);
}

int main(int argc, char** argv) {
  MPI_CHECK(MPI_Init(&argc, &argv));
  int rank, nranks; MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD,&rank)); MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD,&nranks));
  int n=1024, steps=10; bool validate=false, results=false; int parse_ok=1, help=0;
  for(int i=1;i<argc;++i) {
    if(!std::strcmp(argv[i],"-n")&&i+1<argc) n=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"-s")&&i+1<argc) steps=std::atoi(argv[++i]);
    else if(!std::strcmp(argv[i],"-v")) validate=true;
    else if(!std::strcmp(argv[i],"-r")) results=true;
    else if(!std::strcmp(argv[i],"-h")) help=1;
    else parse_ok=0;
  }
  if(help||!parse_ok||n<1||steps<0) { if(rank==0) printUsage(argv[0]); MPI_Finalize(); return parse_ok&&n>0&&steps>=0?0:1; }

  MPI_Comm local_comm; MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&local_comm));
  int local_rank=0, devices=0; MPI_CHECK(MPI_Comm_rank(local_comm,&local_rank)); CUDA_CHECK(cudaGetDeviceCount(&devices));
  if(devices==0) { if(rank==0) std::fprintf(stderr,"No CUDA devices available\n"); MPI_Abort(MPI_COMM_WORLD,2); }
  CUDA_CHECK(cudaSetDevice(local_rank % devices)); MPI_CHECK(MPI_Comm_free(&local_comm));

  const int q=n/nranks, rem=n%nranks, count=q+(rank<rem), first=rank*q+(rank<rem?rank:rem);
  std::vector<int> counts(nranks), offsets(nranks);
#pragma omp parallel for schedule(static)
  for(int r=0;r<nranks;++r) { int c=q+(r<rem); counts[r]=c*6; offsets[r]=(r*q+(r<rem?r:rem))*6; }
  std::vector<Body> bodies(n); if(rank==0) randomizeBodies(bodies);
  MPI_CHECK(MPI_Bcast(bodies.data(),n*6,MPI_DOUBLE,0,MPI_COMM_WORLD));
  Body* dev=nullptr; CUDA_CHECK(cudaMalloc(&dev,n*sizeof(Body))); CUDA_CHECK(cudaMemcpy(dev,bodies.data(),n*sizeof(Body),cudaMemcpyHostToDevice));

  if(rank==0) std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n",n,steps,validate?"enabled":"disabled",nranks,omp_get_max_threads());
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD)); const double start=MPI_Wtime();
  constexpr int block=256;
  for(int s=0;s<steps;++s) {
    if(count) advanceBodies<block><<<(count+block-1)/block,block>>>(dev,n,first,count);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    // Requires CUDA-aware MPI; this is intentionally the unconditional fast cluster path.
    MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,reinterpret_cast<double*>(dev),counts.data(),offsets.data(),MPI_DOUBLE,MPI_COMM_WORLD));
  }
  CUDA_CHECK(cudaDeviceSynchronize()); double elapsed=MPI_Wtime()-start, maximum=0;
  MPI_CHECK(MPI_Reduce(&elapsed,&maximum,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD));
  if(rank==0 && (results||validate)) CUDA_CHECK(cudaMemcpy(bodies.data(),dev,n*sizeof(Body),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(dev));

  int rc=0;
  if(rank==0) {
    std::printf("Simulation time: %ld ms\n",(long)(maximum*1000.0));
    if(results) { std::vector<double> data((size_t)n*6);
#pragma omp parallel for schedule(static)
      for(int i=0;i<n;++i) std::memcpy(data.data()+6LL*i,&bodies[i],6*sizeof(double));
      print_results(data,"Bodies"); }
    if(validate) { std::printf("Validating simulation results...\n"); if(validateSimulation(bodies)) std::printf("Final energy: %.6f\nValidation: PASSED\n",computeTotalEnergy(bodies)); else { std::printf("Validation: FAILED\n"); rc=1; } }
  }
  MPI_CHECK(MPI_Bcast(&rc,1,MPI_INT,0,MPI_COMM_WORLD)); MPI_CHECK(MPI_Finalize()); return rc;
}
