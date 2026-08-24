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

static void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); }
}

__global__ void nbody_step(double* x, double* y, double* z, double* vx, double* vy, double* vz,
                           int first, int count, int n) {
    int i = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= first + count) return;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    const double xi = x[i], yi = y[i], zi = z[i];
    for (int j = 0; j < n; ++j) {
        double dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi;
        double inv = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        double inv3 = inv * inv * inv;
        fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
    }
    vx[i] += DT * fx; vy[i] += DT * fy; vz[i] += DT * fz;
    x[i] += vx[i] * DT; y[i] += vy[i] * DT; z[i] += vz[i] * DT;
}

static void randomize(std::vector<double>& x, std::vector<double>& y, std::vector<double>& z,
                      std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz) {
    unsigned seed = 42;
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        y[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        z[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vx[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vy[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

static bool valid(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z,
                  const std::vector<double>& vx, const std::vector<double>& vy, const std::vector<double>& vz) {
    bool ok = true;
    #pragma omp parallel for reduction(&:ok)
    for (int i = 0; i < (int)x.size(); ++i)
        ok &= std::isfinite(x[i]) && std::isfinite(y[i]) && std::isfinite(z[i]) &&
              std::isfinite(vx[i]) && std::isfinite(vy[i]) && std::isfinite(vz[i]) &&
              std::abs(x[i]) <= 1e6 && std::abs(y[i]) <= 1e6 && std::abs(z[i]) <= 1e6 &&
              std::abs(vx[i]) <= 1e6 && std::abs(vy[i]) <= 1e6 && std::abs(vz[i]) <= 1e6;
    return ok;
}

static double energy(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z,
                     const std::vector<double>& vx, const std::vector<double>& vy, const std::vector<double>& vz) {
    double e = 0.0; int n = (int)x.size();
    #pragma omp parallel for reduction(+:e)
    for (int i = 0; i < n; ++i) e += 0.5 * (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
    #pragma omp parallel for reduction(+:e)
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) {
        double dx=x[j]-x[i], dy=y[j]-y[i], dz=z[j]-z[i];
        e -= 1.0 / sqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
    }
    return e;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); int rank, nranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    int n=1024, steps=10; bool validate=false, results=false;
    for (int i=1;i<argc;++i) { if (!strcmp(argv[i],"-n") && i+1<argc) n=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-s") && i+1<argc) steps=atoi(argv[++i]); else if (!strcmp(argv[i],"-v")) validate=true;
        else if (!strcmp(argv[i],"-r")) results=true; else if (!strcmp(argv[i],"-h")) { if(rank==0) printf("Usage: %s [-n bodies] [-s steps] [-v] [-r]\n",argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) fprintf(stderr,"Unknown option: %s\n",argv[i]); MPI_Finalize(); return 1; } }
    if (n < 0 || steps < 0) { MPI_Finalize(); return 1; }
    std::vector<double> x(n),y(n),z(n),vx(n),vy(n),vz(n); if(rank==0) randomize(x,y,z,vx,vy,vz);
    MPI_Bcast(x.data(),n,MPI_DOUBLE,0,MPI_COMM_WORLD); MPI_Bcast(y.data(),n,MPI_DOUBLE,0,MPI_COMM_WORLD); MPI_Bcast(z.data(),n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(vx.data(),n,MPI_DOUBLE,0,MPI_COMM_WORLD); MPI_Bcast(vy.data(),n,MPI_DOUBLE,0,MPI_COMM_WORLD); MPI_Bcast(vz.data(),n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    int first = n*rank/nranks, end=n*(rank+1)/nranks, count=end-first;
    std::vector<int> counts(nranks), displs(nranks);
    for (int r=0; r<nranks; ++r) { displs[r]=n*r/nranks; counts[r]=n*(r+1)/nranks-displs[r]; }
    int device=0; cuda_check(cudaGetDeviceCount(&device) == cudaSuccess && device > 0 ? cudaSetDevice(rank % device) : cudaErrorNoDevice, "CUDA device");
    double *dx,*dy,*dz,*dvx,*dvy,*dvz; size_t bytes=(size_t)n*sizeof(double);
    for(double** p : {&dx,&dy,&dz,&dvx,&dvy,&dvz}) cuda_check(cudaMalloc(p,bytes),"cudaMalloc");
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    for(int s=0;s<steps;++s) { cuda_check(cudaMemcpy(dx,x.data(),bytes,cudaMemcpyHostToDevice),"H2D"); cuda_check(cudaMemcpy(dy,y.data(),bytes,cudaMemcpyHostToDevice),"H2D"); cuda_check(cudaMemcpy(dz,z.data(),bytes,cudaMemcpyHostToDevice),"H2D"); cuda_check(cudaMemcpy(dvx,vx.data(),bytes,cudaMemcpyHostToDevice),"H2D"); cuda_check(cudaMemcpy(dvy,vy.data(),bytes,cudaMemcpyHostToDevice),"H2D"); cuda_check(cudaMemcpy(dvz,vz.data(),bytes,cudaMemcpyHostToDevice),"H2D");
        nbody_step<<<(count+255)/256,256>>>(dx,dy,dz,dvx,dvy,dvz,first,count,n); cuda_check(cudaGetLastError(),"kernel"); cuda_check(cudaDeviceSynchronize(),"kernel sync");
        size_t local_bytes=(size_t)count*sizeof(double); cuda_check(cudaMemcpy(x.data()+first,dx+first,local_bytes,cudaMemcpyDeviceToHost),"D2H"); cuda_check(cudaMemcpy(y.data()+first,dy+first,local_bytes,cudaMemcpyDeviceToHost),"D2H"); cuda_check(cudaMemcpy(z.data()+first,dz+first,local_bytes,cudaMemcpyDeviceToHost),"D2H"); cuda_check(cudaMemcpy(vx.data()+first,dvx+first,local_bytes,cudaMemcpyDeviceToHost),"D2H"); cuda_check(cudaMemcpy(vy.data()+first,dvy+first,local_bytes,cudaMemcpyDeviceToHost),"D2H"); cuda_check(cudaMemcpy(vz.data()+first,dvz+first,local_bytes,cudaMemcpyDeviceToHost),"D2H");
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,x.data(),counts.data(),displs.data(),MPI_DOUBLE,MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,y.data(),counts.data(),displs.data(),MPI_DOUBLE,MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,z.data(),counts.data(),displs.data(),MPI_DOUBLE,MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,vx.data(),counts.data(),displs.data(),MPI_DOUBLE,MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,vy.data(),counts.data(),displs.data(),MPI_DOUBLE,MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,vz.data(),counts.data(),displs.data(),MPI_DOUBLE,MPI_COMM_WORLD);
    }
    auto endtime=std::chrono::high_resolution_clock::now(); for(double* p:{dx,dy,dz,dvx,dvy,dvz}) cudaFree(p);
    long local_time=(long)std::chrono::duration_cast<std::chrono::milliseconds>(endtime-start).count(), simulation_time=0;
    MPI_Reduce(&local_time,&simulation_time,1,MPI_LONG,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0) { printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nSimulation time: %ld ms\n",n,steps,simulation_time);
        if(results){std::vector<double>d;d.reserve(6*n);for(int i=0;i<n;++i){d.insert(d.end(),{x[i],y[i],z[i],vx[i],vy[i],vz[i]});}print_results(d,"Bodies");}
        if(validate){bool ok=valid(x,y,z,vx,vy,vz);if(ok)printf("Final energy: %.6f\nValidation: PASSED\n",energy(x,y,z,vx,vy,vz));else printf("Validation: FAILED\n"); MPI_Finalize();return ok?0:1;}}
    MPI_Finalize(); return 0;
}
