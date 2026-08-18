#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../common/results_output.hpp"

inline size_t at(size_t x,size_t y,size_t z,size_t nx,size_t ny){return z*nx*ny+y*nx+x;}

static void exchange(std::vector<double>& a, size_t nx, size_t ny, size_t nz,
                     int rank, int size, MPI_Comm comm) {
    const int plane = static_cast<int>(nx*ny);
    const int down = rank ? rank-1 : MPI_PROC_NULL;
    const int up = rank+1 < size ? rank+1 : MPI_PROC_NULL;
    MPI_Sendrecv(a.data()+plane, plane, MPI_DOUBLE, down, 0,
                 a.data()+(nz+1)*plane, plane, MPI_DOUBLE, up, 0, comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(a.data()+nz*plane, plane, MPI_DOUBLE, up, 1,
                 a.data(), plane, MPI_DOUBLE, down, 1, comm, MPI_STATUS_IGNORE);
    // A clamped boundary repeats its boundary plane.
    if (rank == 0) std::copy(a.begin()+plane, a.begin()+2*plane, a.begin());
    if (rank == size-1) std::copy(a.begin()+nz*plane, a.begin()+(nz+1)*plane,
                                   a.begin()+(nz+1)*plane);
}

static inline double lap(const std::vector<double>& a,size_t x,size_t y,size_t z,
                         size_t nx,size_t ny,size_t nz,double dx,double dy,double dz) {
    const size_t xp=x+ (x+1<nx), xn=x-(x>0), yp=y+(y+1<ny), yn=y-(y>0);
    const size_t zp=z+1, zn=z-1;
    return (a[at(xp,y,z,nx,ny)]+a[at(xn,y,z,nx,ny)]-2*a[at(x,y,z,nx,ny)])/(dx*dx)
         + (a[at(x,yp,z,nx,ny)]+a[at(x,yn,z,nx,ny)]-2*a[at(x,y,z,nx,ny)])/(dy*dy)
         + (a[at(x,y,zp,nx,ny)]+a[at(x,y,zn,nx,ny)]-2*a[at(x,y,z,nx,ny)])/(dz*dz);
}

static void initialize(std::vector<double>& c,size_t nx,size_t ny,size_t localNz,
                       size_t globalZ,size_t z0) {
    const size_t vol=nx*ny*globalZ;
    for(size_t z=1;z<=localNz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
        const size_t id=(z0+z-1)*nx*ny+y*nx+x;
        c[at(x,y,z,nx,ny)]=-1.0+2.0*((((id+1)*1299709)%vol)/static_cast<double>(vol));
    }
}

static bool validate(const std::vector<double>& c,size_t nx,size_t ny,size_t nz,int rank,int size,MPI_Comm comm){
    double lo=1e300,hi=-1e300; bool bad=false;
    for(size_t z=1;z<=nz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x){double v=c[at(x,y,z,nx,ny)]; bad|=!std::isfinite(v); lo=std::min(lo,v); hi=std::max(hi,v);}
    double glo,ghi; int ibad=bad;
    MPI_Allreduce(&lo,&glo,1,MPI_DOUBLE,MPI_MIN,comm); MPI_Allreduce(&hi,&ghi,1,MPI_DOUBLE,MPI_MAX,comm); MPI_Allreduce(&ibad,&ibad,1,MPI_INT,MPI_MAX,comm);
    if(rank==0){printf("Concentration range: [%.6f, %.6f]\n",glo,ghi); if(ibad||ghi>10||glo<-10) {printf("Validation failed\n");return false;} printf("Validation: PASSED\n");} return ibad==0&&ghi<=10&&glo>=-10;
}

static void usage(const char* p){printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r] [-h]\n",p);}

int main(int argc,char** argv){
    MPI_Init(&argc,&argv); int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    size_t nx=64,ny=0,nz=0; int iterations=20; bool validateFlag=false,printResults=false;
    for(int i=1;i<argc;++i){if(!strcmp(argv[i],"-x")&&i+1<argc)nx=atoi(argv[++i]); else if(!strcmp(argv[i],"-y")&&i+1<argc)ny=atoi(argv[++i]); else if(!strcmp(argv[i],"-z")&&i+1<argc)nz=atoi(argv[++i]); else if(!strcmp(argv[i],"-i")&&i+1<argc)iterations=atoi(argv[++i]); else if(!strcmp(argv[i],"-v"))validateFlag=true; else if(!strcmp(argv[i],"-r"))printResults=true; else if(!strcmp(argv[i],"-h")){if(rank==0)usage(argv[0]);MPI_Finalize();return 0;} else {if(rank==0)usage(argv[0]);MPI_Finalize();return 1;}}
    if(!ny)ny=nx;if(!nz)nz=nx; if(nz<size){if(rank==0)printf("Z dimension must be at least MPI process count\n");MPI_Finalize();return 1;}
    const size_t base=nz/size, rem=nz%size, localNz=base+(static_cast<size_t>(rank)<rem), z0=static_cast<size_t>(rank)*base+std::min(static_cast<size_t>(rank),rem), plane=nx*ny;
    std::vector<double> cold((localNz+2)*plane),cnew((localNz+2)*plane),mu((localNz+2)*plane);
    initialize(cold,nx,ny,localNz,nz,z0); MPI_Barrier(MPI_COMM_WORLD); double start=MPI_Wtime();
    const double dt=.01,gamma=.5,D=1.,dx=1.,dy=1.,dz=1.,eAA=-2./9.,eBB=-2./9.,eAB=2./9.;
    for(int t=0;t<iterations;++t){
        exchange(cold,nx,ny,localNz,rank,size,MPI_COMM_WORLD);
        for(size_t z=1;z<=localNz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x){size_t q=at(x,y,z,nx,ny);double v=cold[q];mu[q]=4.5*((v+1)*eAA+(v-1)*eBB-2*v*eAB)+3*v+v*v*v-gamma*lap(cold,x,y,z,nx,ny,localNz,dx,dy,dz);}
        exchange(mu,nx,ny,localNz,rank,size,MPI_COMM_WORLD);
        for(size_t z=1;z<=localNz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x){size_t q=at(x,y,z,nx,ny);cnew[q]=cold[q]+dt*D*lap(mu,x,y,z,nx,ny,localNz,dx,dy,dz);}
        cold.swap(cnew);
    }
    double elapsed=MPI_Wtime()-start, worst; MPI_Reduce(&elapsed,&worst,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0){printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\n",nx,ny,nz,iterations);printf("Computation time: %.0f ms\nPerformance: %.3f MCellUpdates/s\n",worst*1000,(double)nx*ny*nz*iterations/worst/1e6);}
    if(printResults){
        std::vector<double> global;
        std::vector<int> counts,displs;
        if(rank==0){
            global.resize(nx*ny*nz); counts.resize(size); displs.resize(size);
            for(int r=0;r<size;++r){size_t rn=nz/size+(static_cast<size_t>(r)<nz%size); size_t rz=static_cast<size_t>(r)*(nz/size)+std::min(static_cast<size_t>(r),nz%size); counts[r]=static_cast<int>(rn*plane); displs[r]=static_cast<int>(rz*plane);}
        }
        MPI_Gatherv(cold.data()+plane,static_cast<int>(localNz*plane),MPI_DOUBLE,
                    rank==0?global.data():nullptr,rank==0?counts.data():nullptr,
                    rank==0?displs.data():nullptr,MPI_DOUBLE,0,MPI_COMM_WORLD);
        if(rank==0) print_results(global,"Concentration");
    }
    bool ok=true;if(validateFlag)ok=validate(cold,nx,ny,localNz,rank,size,MPI_COMM_WORLD); MPI_Finalize(); return ok?0:1;
}
