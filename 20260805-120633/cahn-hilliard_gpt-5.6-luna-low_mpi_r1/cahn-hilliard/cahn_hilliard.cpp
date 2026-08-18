#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../common/results_output.hpp"

inline constexpr size_t id(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept { return (z * ny + y) * nx + x; }

static void exchange(std::vector<double>& a, size_t nx, size_t ny, size_t nz, int rank, int size, MPI_Comm comm) {
    const int face = static_cast<int>(nx * ny);
    MPI_Request req[4]; int n = 0;
    if (rank > 0) MPI_Irecv(a.data(), face, MPI_DOUBLE, rank - 1, 7, comm, &req[n++]);
    if (rank + 1 < size) MPI_Irecv(a.data() + (nz + 1) * face, face, MPI_DOUBLE, rank + 1, 7, comm, &req[n++]);
    if (rank > 0) MPI_Isend(a.data() + face, face, MPI_DOUBLE, rank - 1, 7, comm, &req[n++]);
    if (rank + 1 < size) MPI_Isend(a.data() + nz * face, face, MPI_DOUBLE, rank + 1, 7, comm, &req[n++]);
    MPI_Waitall(n, req, MPI_STATUSES_IGNORE);
    if (rank == 0) std::copy(a.data() + face, a.data() + 2 * face, a.data());
    if (rank + 1 == size) std::copy(a.data() + nz * face, a.data() + (nz + 1) * face, a.data() + (nz + 1) * face);
}

static void laplacian(const std::vector<double>& a, std::vector<double>& out, size_t nx, size_t ny, size_t nz,
                      double dx, double dy, double dz) {
    const double ix = 1.0/(dx*dx), iy = 1.0/(dy*dy), iz = 1.0/(dz*dz);
    for (size_t z=1; z<=nz; ++z) for (size_t y=0; y<ny; ++y) for (size_t x=0; x<nx; ++x) {
        size_t q=id(x,y,z,nx,ny); size_t xm=x?x-1:x, xp=x+1<nx?x+1:x, ym=y?y-1:y, yp=y+1<ny?y+1:y;
        out[q]=(a[id(xp,y,z,nx,ny)]+a[id(xm,y,z,nx,ny)]-2*a[q])*ix
             +(a[id(x,yp,z,nx,ny)]+a[id(x,ym,z,nx,ny)]-2*a[q])*iy
             +(a[id(x,y,z+1,nx,ny)]+a[id(x,y,z-1,nx,ny)]-2*a[q])*iz;
    }
}

static void usage(const char* p) { std::printf("Usage: %s [options]\n  -x/-y/-z <num> grid dimensions (default 64)\n  -i <num> time steps (default 20)\n  -v validate\n  -r print results\n  -h help\n",p); }

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv); int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    size_t nx=64,ny=0,nz=0; int iterations=20; bool validate=false, results=false;
    for(int i=1;i<argc;++i) { if(!std::strcmp(argv[i],"-x")&&i+1<argc) nx=std::strtoull(argv[++i],nullptr,10);
      else if(!std::strcmp(argv[i],"-y")&&i+1<argc) ny=std::strtoull(argv[++i],nullptr,10);
      else if(!std::strcmp(argv[i],"-z")&&i+1<argc) nz=std::strtoull(argv[++i],nullptr,10);
      else if(!std::strcmp(argv[i],"-i")&&i+1<argc) iterations=std::atoi(argv[++i]); else if(!std::strcmp(argv[i],"-v")) validate=true;
      else if(!std::strcmp(argv[i],"-r")) results=true; else if(!std::strcmp(argv[i],"-h")){if(rank==0)usage(argv[0]);MPI_Finalize();return 0;} else {if(rank==0)usage(argv[0]);MPI_Finalize();return 1;} }
    if(!ny)ny=nx; if(!nz)nz=nx;
    if(nz < static_cast<size_t>(size)) { if(rank==0) std::fprintf(stderr,"Z dimension must be at least MPI process count\n"); MPI_Finalize(); return 1; }
    size_t base=nz/size, rem=nz%size, localz=base+(static_cast<size_t>(rank)<rem), z0=rank*base+std::min<size_t>(rank,rem), face=nx*ny;
    std::vector<double> c((localz+2)*face), mu(c.size()), next(c.size());
    for(size_t z=1;z<=localz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) { size_t g=(z0+z-1)*face+y*nx+x; c[id(x,y,z,nx,ny)]=-1+2*((((g+1)*1299709)% (nx*ny*nz))/static_cast<double>(nx*ny*nz)); }
    if(rank==0){std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled");std::printf("Initializing concentration field...\nRunning Cahn-Hilliard simulation...\n");}
    MPI_Barrier(MPI_COMM_WORLD); double start=MPI_Wtime();
    for(int t=0;t<iterations;++t){ exchange(c,nx,ny,localz,rank,size,MPI_COMM_WORLD); laplacian(c,mu,nx,ny,localz,1,1,1);
      for(size_t z=1;z<=localz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x){size_t q=id(x,y,z,nx,ny),v=id(x,y,z,nx,ny);mu[q]=4.5*((c[v]+1)*(-2.0/9)+(c[v]-1)*(-2.0/9)-2*c[v]*(2.0/9))+3*c[v]+c[v]*c[v]*c[v]-0.5*mu[q];}
      exchange(mu,nx,ny,localz,rank,size,MPI_COMM_WORLD); laplacian(mu,next,nx,ny,localz,1,1,1);
      for(size_t z=1;z<=localz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x){size_t q=id(x,y,z,nx,ny);next[q]=c[q]+.01*next[q];} c.swap(next);
    }
    double elapsed=MPI_Wtime()-start, total; MPI_Reduce(&elapsed,&total,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0){double mc=double(nx)*ny*nz*iterations/total/1e6;std::printf("Computation time: %.0f ms\nPerformance: %.3f MCellUpdates/s\n",total*1000,mc);}
    std::vector<double> all; if(rank==0)all.resize(nx*ny*nz); std::vector<int> counts(size),displs(size); for(int r=0;r<size;++r){size_t n=base+(size_t(r)<rem);counts[r]=int(n*face);displs[r]=int((r*base+std::min<size_t>(r,rem))*face);}
    MPI_Gatherv(c.data()+face,int(localz*face),MPI_DOUBLE,rank==0?all.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    int ok=1; if(rank==0 && (results||validate)){if(results)print_results(all,"Concentration");double lo=*std::min_element(all.begin(),all.end()),hi=*std::max_element(all.begin(),all.end());std::printf("Concentration range: [%.6f, %.6f]\n",lo,hi);if(validate){std::printf("Validation: %s\n",(std::isfinite(lo)&&std::isfinite(hi)&&lo>=-10&&hi<=10)?"PASSED":"FAILED");ok=(std::isfinite(lo)&&std::isfinite(hi)&&lo>=-10&&hi<=10);}}
    MPI_Bcast(&ok,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Finalize(); return ok?0:1;
}
