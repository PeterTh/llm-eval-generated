#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../common/results_output.hpp"

inline size_t at(size_t x, size_t y, size_t z, size_t nx, size_t ny) { return z*nx*ny+y*nx+x; }

static void exchange(std::vector<double>& a, size_t plane, size_t nz, int rank, int nr) {
    if (nr == 1) { std::copy_n(a.data()+plane, plane, a.data()); std::copy_n(a.data()+nz*plane, plane, a.data()+(nz+1)*plane); return; }
    MPI_Sendrecv(a.data()+plane, plane, MPI_DOUBLE, rank ? rank-1 : MPI_PROC_NULL, 0,
                 a.data()+(nz+1)*plane, plane, MPI_DOUBLE, rank+1<nr ? rank+1 : MPI_PROC_NULL, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(a.data()+nz*plane, plane, MPI_DOUBLE, rank+1<nr ? rank+1 : MPI_PROC_NULL, 1,
                 a.data(), plane, MPI_DOUBLE, rank ? rank-1 : MPI_PROC_NULL, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (!rank) std::copy_n(a.data()+plane, plane, a.data());
    if (rank == nr-1) std::copy_n(a.data()+nz*plane, plane, a.data()+(nz+1)*plane);
}

static void chemical(const std::vector<double>& c, std::vector<double>& mu, size_t nx, size_t ny, size_t nz) {
    constexpr double g=.5, dx2=1.;
    for (size_t z=1; z<=nz; ++z) for (size_t y=0;y<ny;++y) for (size_t x=0;x<nx;++x) {
        size_t q=at(x,y,z,nx,ny); double v=c[q];
        double lap=c[at(x+1<nx?x+1:x,y,z,nx,ny)]+c[at(x?x-1:0,y,z,nx,ny)]
          +c[at(x,y+1<ny?y+1:y,z,nx,ny)]+c[at(x,y?y-1:0,z,nx,ny)]
          +c[at(x,y,z+1,nx,ny)]+c[at(x,y,z-1,nx,ny)]-6*v;
        mu[q]=4.5*((v+1)*(-2./9)+(v-1)*(-2./9)-2*v*(2./9))+3*v+v*v*v-g*lap/dx2;
    }
}

static void update(std::vector<double>& n, const std::vector<double>& c, const std::vector<double>& m, size_t nx,size_t ny,size_t nz) {
    for(size_t z=1;z<=nz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
        size_t q=at(x,y,z,nx,ny); double v=m[q];
        double lap=m[at(x+1<nx?x+1:x,y,z,nx,ny)]+m[at(x?x-1:0,y,z,nx,ny)]+m[at(x,y+1<ny?y+1:y,z,nx,ny)]+m[at(x,y?y-1:0,z,nx,ny)]+m[at(x,y,z+1,nx,ny)]+m[at(x,y,z-1,nx,ny)]-6*v;
        n[q]=c[q]+.01*lap;
    }
}

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank,nr; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&nr);
    size_t nx=64,ny=0,nz=0; int iters=20; bool validate=false, results=false;
    for(int i=1;i<argc;++i) { if(!strcmp(argv[i],"-x")&&i+1<argc) nx=atoi(argv[++i]); else if(!strcmp(argv[i],"-y")&&i+1<argc) ny=atoi(argv[++i]); else if(!strcmp(argv[i],"-z")&&i+1<argc) nz=atoi(argv[++i]); else if(!strcmp(argv[i],"-i")&&i+1<argc) iters=atoi(argv[++i]); else if(!strcmp(argv[i],"-v")) validate=true; else if(!strcmp(argv[i],"-r")) results=true; else if(!strcmp(argv[i],"-h")){if(!rank) printf("Usage: %s [-x N] [-y N] [-z N] [-i N] [-v] [-r]\n",argv[0]); MPI_Finalize(); return 0;} else {if(!rank) printf("Unknown option: %s\n",argv[i]); MPI_Finalize(); return 1;} }
    if(!ny)ny=nx;if(!nz)nz=nx; size_t plane=nx*ny, base=nz/nr, rem=nz%nr, local=base+(size_t)(rank<rem), start=(size_t)rank*base+std::min((size_t)rank,rem);
    if(!rank){printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iters,validate?"enabled":"disabled");printf("MPI ranks: %d\n",nr);}
    std::vector<double> c((local+2)*plane), n(c.size()), mu(c.size());
    for(size_t z=1;z<=local;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x){size_t id=(start+z-1)*plane+y*nx+x; c[at(x,y,z,nx,ny)]=-1+2*((((id+1)*1299709)% (nx*ny*nz))/(double)(nx*ny*nz));}
    MPI_Barrier(MPI_COMM_WORLD); auto begin=std::chrono::steady_clock::now();
    for(int t=0;t<iters;++t){exchange(c,plane,local,rank,nr);chemical(c,mu,nx,ny,local);exchange(mu,plane,local,rank,nr);update(n,c,mu,nx,ny,local);c.swap(n);}
    MPI_Barrier(MPI_COMM_WORLD); double elapsed; auto end=std::chrono::steady_clock::now(); elapsed=std::chrono::duration<double>(end-begin).count(); double worst; MPI_Reduce(&elapsed,&worst,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(!rank){printf("Computation time: %ld ms\n",(long)(worst*1000));printf("Performance: %.3f MCellUpdates/s\n",(double)nx*ny*nz*iters/worst/1e6);}
    if(results){
        std::vector<int> counts,displs; std::vector<double> all;
        if(!rank){counts.resize(nr);displs.resize(nr);int d=0;for(int r=0;r<nr;++r){counts[r]=(int)((base+(size_t)(r<rem))*plane);displs[r]=d;d+=counts[r];}all.resize(nz*plane);}
        MPI_Gatherv(c.data()+plane,(int)(local*plane),MPI_DOUBLE,rank?nullptr:all.data(),rank?nullptr:counts.data(),rank?nullptr:displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
        if(!rank) print_results(all,"Concentration");
    }
    if(validate){double lo=1e300,hi=-1e300;int bad=0;for(size_t z=1;z<=local;++z)for(size_t q=0;q<plane;++q){double v=c[z*plane+q];lo=std::min(lo,v);hi=std::max(hi,v);bad|=!std::isfinite(v);}double glo,ghi;int gbad;MPI_Reduce(&lo,&glo,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);MPI_Reduce(&hi,&ghi,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);MPI_Reduce(&bad,&gbad,1,MPI_INT,MPI_MAX,0,MPI_COMM_WORLD);if(!rank){printf("Concentration range: [%.6f, %.6f]\n",glo,ghi);printf("Validation: %s\n",gbad||ghi>10||glo<-10?"FAILED":"PASSED");}}
    MPI_Finalize(); return 0;
}
