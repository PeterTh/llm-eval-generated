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
struct Body { Vec3 pos; Vec3 vel; };

__global__ void step_kernel(Body* b, int n, int first, int count) {
    const int i = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= first + count || i >= n) return;
    const Body self = b[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < n; ++j) {
        const double dx = b[j].pos.x - self.pos.x;
        const double dy = b[j].pos.y - self.pos.y;
        const double dz = b[j].pos.z - self.pos.z;
        const double r2 = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double inv = 1.0 / sqrt(r2);
        const double inv3 = inv * inv * inv;
        fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
    }
    b[i].vel.x = self.vel.x + DT * fx;
    b[i].vel.y = self.vel.y + DT * fy;
    b[i].vel.z = self.vel.z + DT * fz;
    b[i].pos.x = self.pos.x + b[i].vel.x * DT;
    b[i].pos.y = self.pos.y + b[i].vel.y * DT;
    b[i].pos.z = self.pos.z + b[i].vel.z * DT;
}

static void randomizeBodies(std::vector<Body>& bodies) {
    unsigned int seed = 42;
    std::vector<double> values(bodies.size() * 6);
    for (double& value : values) value = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        const double* v = values.data() + static_cast<size_t>(i) * 6;
        bodies[i].pos = {v[0], v[1], v[2]}; bodies[i].vel = {v[3], v[4], v[5]};
    }
}

static double computeTotalEnergy(const std::vector<Body>& b) {
    const int n = static_cast<int>(b.size());
    double kinetic = 0.0;
    #pragma omp parallel for reduction(+:kinetic) schedule(static)
    for (int i=0; i<n; ++i)
        kinetic += 0.5*(b[i].vel.x*b[i].vel.x+b[i].vel.y*b[i].vel.y+b[i].vel.z*b[i].vel.z);
    double potential = 0.0;
    #pragma omp parallel for reduction(+:potential) schedule(static)
    for (int i=0; i<n; ++i) for (int j=i+1; j<n; ++j) {
        const double x=b[j].pos.x-b[i].pos.x, y=b[j].pos.y-b[i].pos.y, z=b[j].pos.z-b[i].pos.z;
        potential -= 1.0/std::sqrt(x*x+y*y+z*z+SOFTENING);
    }
    return kinetic + potential;
}

static bool validateSimulation(const std::vector<Body>& b) {
    int bad = 0;
    #pragma omp parallel for reduction(+:bad) schedule(static)
    for (int i=0; i<static_cast<int>(b.size()); ++i) {
        const Body& x=b[i];
        const double v[6]={x.pos.x,x.pos.y,x.pos.z,x.vel.x,x.vel.y,x.vel.z};
        for (double q:v) if (!std::isfinite(q) || std::abs(q)>1e6) bad++;
    }
    return bad == 0;
}

static void usage(const char* p) { printf("Usage: %s [-n num] [-s num] [-v] [-r] [-h]\n",p); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank=0, ranks=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    int n=1024, steps=10; bool validate=false, results=false;
    for (int i=1;i<argc;++i) {
        if (!strcmp(argv[i],"-n") && i+1<argc) n=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-s") && i+1<argc) steps=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-v")) validate=true;
        else if (!strcmp(argv[i],"-r")) results=true;
        else if (!strcmp(argv[i],"-h")) { if(rank==0) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) usage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (n < 0 || steps < 0) { MPI_Finalize(); return 1; }
    if (rank==0) { printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",n,steps,validate?"enabled":"disabled"); }
    std::vector<Body> bodies(static_cast<size_t>(n)); randomizeBodies(bodies);
    const int first = (n*rank)/ranks, last=(n*(rank+1))/ranks, count=last-first;
    std::vector<int> counts(ranks), displs(ranks);
    for(int r=0;r<ranks;++r){ counts[r]=((n*(r+1))/ranks-(n*r)/ranks)*static_cast<int>(sizeof(Body)); displs[r]=(n*r/ranks)*static_cast<int>(sizeof(Body)); }
    int device_count=0; cudaGetDeviceCount(&device_count); if(device_count==0){ MPI_Abort(MPI_COMM_WORLD,2); }
    cudaSetDevice(rank % device_count);
    Body* device=nullptr; cudaMalloc(&device, static_cast<size_t>(n)*sizeof(Body));
    MPI_Barrier(MPI_COMM_WORLD); const auto start=std::chrono::high_resolution_clock::now();
    for(int s=0;s<steps;++s){
        cudaMemcpy(device,bodies.data(),static_cast<size_t>(n)*sizeof(Body),cudaMemcpyHostToDevice);
        step_kernel<<<(count+255)/256,256>>>(device,n,first,count); cudaDeviceSynchronize();
        cudaMemcpy(bodies.data()+first,device+first,static_cast<size_t>(count)*sizeof(Body),cudaMemcpyDeviceToHost);
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_BYTE,bodies.data(),counts.data(),displs.data(),MPI_BYTE,MPI_COMM_WORLD);
    }
    cudaFree(device); MPI_Barrier(MPI_COMM_WORLD);
    const auto end=std::chrono::high_resolution_clock::now(); double ms=std::chrono::duration<double,std::milli>(end-start).count(), maxms=0; MPI_Reduce(&ms,&maxms,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0) printf("Simulation time: %ld ms\n",static_cast<long>(maxms));
    if(rank==0 && results){ std::vector<double> d; d.reserve(static_cast<size_t>(n)*6); for(const auto& x:bodies){d.insert(d.end(),{x.pos.x,x.pos.y,x.pos.z,x.vel.x,x.vel.y,x.vel.z});} print_results(d,"Bodies"); }
    if(validate && rank==0){ printf("Validating simulation results...\n"); const bool ok=validateSimulation(bodies); if(ok) printf("Final energy: %.6f\n",computeTotalEnergy(bodies)); printf("Validation: %s\n",ok?"PASSED":"FAILED"); }
    MPI_Finalize(); return 0;
}
