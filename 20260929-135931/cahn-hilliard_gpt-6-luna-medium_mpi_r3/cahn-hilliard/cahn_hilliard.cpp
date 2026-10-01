#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) { return (z * ny + y) * nx + x; }

// Exchange the planes adjacent to each rank's z slab. Ghost planes at the
// physical ends are filled by copying the boundary plane, matching clamping.
static void exchange(std::vector<double>& a, size_t plane, size_t localNz, int prev, int next, MPI_Comm comm) {
    if (prev != MPI_PROC_NULL)
        MPI_Sendrecv(a.data() + plane, (int)plane, MPI_DOUBLE, prev, 0,
                     a.data() + (localNz + 1) * plane, (int)plane, MPI_DOUBLE, next, 0, comm, MPI_STATUS_IGNORE);
    else std::copy_n(a.data() + plane, plane, a.data());
    if (next != MPI_PROC_NULL)
        MPI_Sendrecv(a.data() + localNz * plane, (int)plane, MPI_DOUBLE, next, 1,
                     a.data(), (int)plane, MPI_DOUBLE, prev, 1, comm, MPI_STATUS_IGNORE);
    else std::copy_n(a.data() + localNz * plane, plane, a.data() + (localNz + 1) * plane);
}

static double lap(const std::vector<double>& a, size_t x, size_t y, size_t z,
                  size_t nx, size_t ny, size_t globalZ, size_t globalNz) {
    const size_t q = idx3(x,y,z,nx,ny), xp=x+1<nx?x+1:x, xm=x?x-1:0, yp=y+1<ny?y+1:y, ym=y?y-1:0;
    const size_t zp = globalZ+1<globalNz ? z+1 : z;
    const size_t zm = globalZ ? z-1 : z;
    return a[idx3(xp,y,z,nx,ny)] + a[idx3(xm,y,z,nx,ny)] - 2*a[q]
         + a[idx3(x,yp,z,nx,ny)] + a[idx3(x,ym,z,nx,ny)] - 2*a[q]
         + a[(zp*ny+y)*nx+x] + a[(zm*ny+y)*nx+x] - 2*a[q];
}

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);
    int rank=0, tasks=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&tasks);
    size_t nx=64, ny=0, nz=0; int iterations=20; bool validate=false, printResults=false;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"-x")&&i+1<argc) nx=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-y")&&i+1<argc) ny=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-z")&&i+1<argc) nz=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-i")&&i+1<argc) iterations=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-v")) validate=true;
        else if(!strcmp(argv[i],"-r")) printResults=true;
        else if(!strcmp(argv[i],"-h")) { if(rank==0) { printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r] [-h]\n",argv[0]); } MPI_Finalize(); return 0; }
        else { if(rank==0) printf("Unknown option: %s\n",argv[i]); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    if(!ny) ny=nx; if(!nz) nz=nx;
    if(nz<(size_t)tasks) { if(rank==0) fprintf(stderr,"Number of MPI ranks cannot exceed z grid size\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    const size_t z0=nz*(size_t)rank/tasks, z1=nz*(size_t)(rank+1)/tasks, lz=z1-z0;
    const size_t plane=nx*ny, count=plane*lz;
    std::vector<double> c((lz+2)*plane), nextC((lz+2)*plane), mu((lz+2)*plane);
    for(size_t z=1;z<=lz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
        size_t id=((z0+z-1)*ny+y)*nx+x;
        c[idx3(x,y,z,nx,ny)]=-1.0+2.0*((((id+1)*1299709)%(nx*ny*nz))/double(nx*ny*nz));
    }
    const int prev=rank?rank-1:MPI_PROC_NULL, next=rank+1<tasks?rank+1:MPI_PROC_NULL;
    if(rank==0) { printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled"); printf("Initializing concentration field...\nRunning Cahn-Hilliard simulation...\n"); }
    MPI_Barrier(MPI_COMM_WORLD); const double start=MPI_Wtime();
    const double dt=.01, gamma=.5, D=1.;
    for(int t=0;t<iterations;++t) {
        exchange(c,plane,lz,prev,next,MPI_COMM_WORLD);
        for(size_t z=1;z<=lz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
            size_t q=idx3(x,y,z,nx,ny); double v=c[q];
            // Unit grid spacing, as in the original benchmark.
            mu[q]=-v+v*v*v-gamma*lap(c,x,y,z,nx,ny,z0+z-1,nz);
        }
        exchange(mu,plane,lz,prev,next,MPI_COMM_WORLD);
        for(size_t z=1;z<=lz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
            size_t q=idx3(x,y,z,nx,ny);
            nextC[q]=c[q]+dt*D*lap(mu,x,y,z,nx,ny,z0+z-1,nz);
        }
        c.swap(nextC);
    }
    double elapsed=MPI_Wtime()-start, maxElapsed=0; MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    std::vector<int> counts(tasks), displs(tasks);
    for(int r=0;r<tasks;++r) { size_t a=nz*(size_t)r/tasks, b=nz*(size_t)(r+1)/tasks; counts[r]=(int)(plane*(b-a)); displs[r]=(int)(plane*a); }
    std::vector<double> result(rank==0?nx*ny*nz:0);
    MPI_Gatherv(c.data()+plane,(int)count,MPI_DOUBLE,result.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    int ok=1;
    if(rank==0) {
        printf("Computation time: %ld ms\n",(long)(maxElapsed*1000));
        double updates=(double)nx*ny*nz*iterations, mcups=maxElapsed>0?updates/maxElapsed/1e6:0;
        printf("Performance: %.3f MCellUpdates/s\n",mcups);
        if(printResults) print_results(result,"Concentration");
        if(validate) {
            printf("Validating result...\n"); double mn=result[0],mx=result[0];
            for(double v:result) { if(!std::isfinite(v)) ok=0; mn=std::min(mn,v); mx=std::max(mx,v); }
            printf("Concentration range: [%.6f, %.6f]\n",mn,mx);
            if(mx>10||mn< -10) ok=0;
            printf("Validation: %s\n",ok?"PASSED":"FAILED");
        }
    }
    MPI_Bcast(&ok,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Finalize(); return validate&&!ok?1:0;
}
