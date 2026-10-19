#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

static double lap(const std::vector<double>& a, size_t nx, size_t ny, size_t localNz,
                  size_t x, size_t y, size_t z, bool lowGlobal, bool highGlobal) {
    size_t xp = x + (x + 1 < nx), xn = x - (x > 0);
    size_t yp = y + (y + 1 < ny), yn = y - (y > 0);
    size_t zp = (z + 1 < localNz || !highGlobal) ? z + 1 : z;
    size_t zn = (z > 1 || !lowGlobal) ? z - 1 : z;
    const size_t i = idx3(x,y,z,nx,ny);
    return a[idx3(xp,y,z,nx,ny)] + a[idx3(xn,y,z,nx,ny)] - 2*a[i]
         + a[idx3(x,yp,z,nx,ny)] + a[idx3(x,yn,z,nx,ny)] - 2*a[i]
         + a[idx3(x,y,zp,nx,ny)] + a[idx3(x,y,zn,nx,ny)] - 2*a[i];
}

static bool validate(const std::vector<double>& c) {
    if (c.empty()) return false;
    double lo=c[0], hi=c[0];
    for(double v:c) { if(!std::isfinite(v)) return false; lo=std::min(lo,v); hi=std::max(hi,v); }
    printf("Concentration range: [%.6f, %.6f]\n",lo,hi);
    return lo >= -10 && hi <= 10;
}

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    size_t nx=64,ny=0,nz=0; int iterations=20; bool check=false,results=false;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"-x")&&i+1<argc) nx=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-y")&&i+1<argc) ny=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-z")&&i+1<argc) nz=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-i")&&i+1<argc) iterations=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-v")) check=true;
        else if(!strcmp(argv[i],"-r")) results=true;
        else if(!strcmp(argv[i],"-h")) { if(rank==0) printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r] [-h]\n",argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) fprintf(stderr,"Unknown option: %s\n",argv[i]); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    if(!ny) ny=nx;
    if(!nz) nz=nx;
    if(!nx||!ny||!nz||size>(int)nz||nx*ny>INT_MAX/(int)sizeof(double)) { if(rank==0) fprintf(stderr,"Invalid grid size or more ranks than Z planes\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    size_t base=nz/size,rem=nz%size,localNz=base+(rank<(int)rem);
    size_t z0=(size_t)rank*base+std::min((size_t)rank,rem), plane=nx*ny;
    // Two halo planes; interior planes are [1, localNz].
    std::vector<double> c((localNz+2)*plane), next(c.size()),mu(c.size());
    for(size_t z=1;z<=localNz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
        size_t id=(z0+z-1)*plane+y*nx+x;
        c[idx3(x,y,z,nx,ny)]=-1.0+2.0*((((id+1)*1299709)% (nx*ny*nz))/double(nx*ny*nz));
    }
    const int count=(int)plane;
    auto start=std::chrono::high_resolution_clock::now();
    for(int t=0;t<iterations;++t) {
        if(rank>0) MPI_Sendrecv(c.data()+plane, count,MPI_DOUBLE,rank-1,0,c.data(),count,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(rank+1<size) MPI_Sendrecv(c.data()+localNz*plane,count,MPI_DOUBLE,rank+1,0,c.data()+(localNz+1)*plane,count,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        for(size_t z=1;z<=localNz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
            size_t i=idx3(x,y,z,nx,ny); double v=c[i];
            mu[i]=4.5*((v+1)*(-2.0/9)+(v-1)*(-2.0/9)-2*v*(2.0/9))+3*v+v*v*v-0.5*lap(c,nx,ny,localNz,x,y,z,rank==0,rank==size-1);
        }
        if(rank>0) MPI_Sendrecv(mu.data()+plane,count,MPI_DOUBLE,rank-1,1,mu.data(),count,MPI_DOUBLE,rank-1,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(rank+1<size) MPI_Sendrecv(mu.data()+localNz*plane,count,MPI_DOUBLE,rank+1,1,mu.data()+(localNz+1)*plane,count,MPI_DOUBLE,rank+1,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        for(size_t z=1;z<=localNz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) { size_t i=idx3(x,y,z,nx,ny); next[i]=c[i]+0.01*lap(mu,nx,ny,localNz,x,y,z,rank==0,rank==size-1); }
        c.swap(next);
    }
    double elapsed=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-start).count(), maxElapsed;
    MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    std::vector<int> counts(size),displs(size);
    for(int r=0;r<size;++r) { size_t n=nz/size+(r<(int)(nz%size)); counts[r]=(int)(n*plane); size_t first=(size_t)r*(nz/size)+std::min((size_t)r,nz%size); displs[r]=(int)(first*plane); }
    std::vector<double> all; if(rank==0) all.resize(nx*ny*nz);
    MPI_Gatherv(c.data()+plane,count*(int)localNz,MPI_DOUBLE,all.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(rank==0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,check?"enabled":"disabled");
        printf("Computation time: %ld ms\n",(long)(maxElapsed*1000));
        printf("Performance: %.3f MCellUpdates/s\n",maxElapsed>0?nx*ny*nz*(double)iterations/maxElapsed/1e6:0.0);
        if(results) print_results(all,"Concentration");
        if(check) printf("Validating result...\n");
    }
    int valid=1; if(check&&rank==0) { valid=validate(all); printf("Validation: %s\n",valid?"PASSED":"FAILED"); } MPI_Bcast(&valid,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Finalize(); return check&&!valid?1:0;
}
