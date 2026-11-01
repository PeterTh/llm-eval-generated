#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

__global__ void stencilKernel(const Real* input, Real* output,
                             size_t nx, size_t ny, size_t nz_local,
                             size_t x0, size_t y0) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (nx-2)*(ny-2)*(nz_local-2);
    if (idx >= total) return;
    size_t plane = (nx-2)*(ny-2);
    size_t zl = idx / plane + 1;
    size_t rem = idx % plane;
    size_t yl = rem / (nx-2) + 1;
    size_t xl = rem % (nx-2) + 1;
    size_t gidx = zl*(nx*ny) + yl*nx + xl;
    Real center = input[gidx];
    Real left = input[gidx-1];
    Real right = input[gidx+1];
    Real front = input[gidx-nx];
    Real back = input[gidx+nx];
    Real bottom = input[gidx-nx*ny];
    Real top = input[gidx+nx*ny];
    output[gidx] = (center+left+right+front+back+bottom+top)/7.0;
}

static void initGrid(std::vector<Real>& grid, size_t nx, size_t ny, size_t nz,
                     size_t zGlobalOffset, size_t nzLocal) {
    #pragma omp parallel for collapse(3)
    for (size_t zl = 0; zl < nzLocal; ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t zg = zGlobalOffset + zl;
                size_t idx = idx3(x,y,zl,nx,ny);
                grid[idx] = ((zg*(nx*ny)+y*nx+x) % 19) * 1.0;
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    size_t nx=128, ny=0, nz=0;
    int iterations=10;
    bool validate=false, printResults=false;
    if (rank==0) {
        for (int i=1;i<argc;++i){
            if (strcmp(argv[i],"-x")==0 && i+1<argc) nx=atoi(argv[++i]);
            else if (strcmp(argv[i],"-y")==0 && i+1<argc) ny=atoi(argv[++i]);
            else if (strcmp(argv[i],"-z")==0 && i+1<argc) nz=atoi(argv[++i]);
            else if (strcmp(argv[i],"-i")==0 && i+1<argc) iterations=atoi(argv[++i]);
            else if (strcmp(argv[i],"-v")==0) validate=true;
            else if (strcmp(argv[i],"-r")==0) printResults=true;
        }
    }
    // broadcast params
    MPI_Bcast(&nx,1,MPI_UNSIGNED_LONG_LONG,0,MPI_COMM_WORLD);
    MPI_Bcast(&ny,1,MPI_UNSIGNED_LONG_LONG,0,MPI_COMM_WORLD);
    MPI_Bcast(&nz,1,MPI_UNSIGNED_LONG_LONG,0,MPI_COMM_WORLD);
    MPI_Bcast(&iterations,1,MPI_INT,0,MPI_COMM_WORLD);
    // bools as int
    int v=validate?1:0, r=printResults?1:0;
    MPI_Bcast(&v,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&r,1,MPI_INT,0,MPI_COMM_WORLD);
    validate = v!=0; printResults = r!=0;
    if (ny==0) ny=nx; if (nz==0) nz=nx;

    // partition interior z
    size_t interior = (nz>=2)? nz-2 : 0;
    size_t base = interior/size, rem = interior%size;
    size_t k = base + (rank < (int)rem ? 1 : 0);
    // z_start global interior index
    size_t zStart = 1;
    for (int i=0;i<rank;++i) zStart += base + (i<(int)rem?1:0);
    // local nz including halos
    size_t nzLocal = k + 2;
    size_t zGlobalOffset = (k>0)? zStart-1 : 0; // lower halo global z
    // handle empty
    if (k==0) { nzLocal=0; }

    std::vector<Real> grid1, grid2;
    if (k>0) {
        grid1.assign(nx*ny*nzLocal,0.0);
        grid2.assign(nx*ny*nzLocal,0.0);
        initGrid(grid1,nx,ny,nz,zGlobalOffset,nzLocal);
        initGrid(grid2,nx,ny,nz,zGlobalOffset,nzLocal);
    }

    // CUDA device
    int devCount=0; cudaGetDeviceCount(&devCount);
    int dev = rank % (devCount>0?devCount:1);
    if (devCount>0) cudaSetDevice(dev);

    Real *d_in=nullptr, *d_out=nullptr;
    if (k>0) {
        cudaMalloc(&d_in, grid1.size()*sizeof(Real));
        cudaMalloc(&d_out, grid2.size()*sizeof(Real));
    }

    if (rank==0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
        printf("Running stencil computation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter=0; iter<iterations && k>0; ++iter) {
        std::vector<Real>& in = (iter%2==0)? grid1 : grid2;
        std::vector<Real>& out = (iter%2==0)? grid2 : grid1;
        cudaMemcpy(d_in, in.data(), in.size()*sizeof(Real), cudaMemcpyHostToDevice);
        // launch
        size_t total = (nx-2)*(ny-2)*(k); // interior z count k, local interior indices 1..k
        // kernel uses nz_local including halos: nzLocal=k+2
        int threads=256;
        int blocks=(total+threads-1)/threads;
        stencilKernel<<<blocks,threads>>>(d_in,d_out,nx,ny,nzLocal,0,0);
        cudaDeviceSynchronize();
        // copy back
        cudaMemcpy(out.data(), d_out, out.size()*sizeof(Real), cudaMemcpyDeviceToHost);

        // copy x/y boundaries for owned interior rows
        #pragma omp parallel for
        for (size_t zl=1; zl<=k; ++zl) {
            for (size_t y=0; y<ny; ++y) {
                for (size_t x=0; x<nx; ++x) {
                    if (x==0||x==nx-1||y==0||y==ny-1) {
                        out[zl*nx*ny + y*nx + x] = in[zl*nx*ny + y*nx + x];
                    }
                }
            }
        }
        // global boundary halos already initialized; ensure output halo slices set to initial formula
        // lower halo
        if (zStart==1) {
            // lower halo global z=0 boundary: copy from initial (already in out? out halo not computed). Set formula.
            for (size_t y=0;y<ny;++y) for (size_t x=0;x<nx;++x) {
                out[0*nx*ny + y*nx + x] = ((0*(nx*ny)+y*nx+x)%19)*1.0;
            }
        }
        if (zStart + k == nz-1) { // last interior row z = nz-2? check: zStart+k-1 == nz-2 => zStart+k == nz-1
            size_t zg = nz-1;
            for (size_t y=0;y<ny;++y) for (size_t x=0;x<nx;++x) {
                out[(k+1)*nx*ny + y*nx + x] = ((zg*(nx*ny)+y*nx+x)%19)*1.0;
            }
        }
        // halo exchange
        // lower
        if (rank>0) {
            // send first interior row to prev, recv into lower halo
            MPI_Sendrecv(&out[1*nx*ny], nx*ny, MPI_DOUBLE, rank-1, 0,
                         &out[0*nx*ny], nx*ny, MPI_DOUBLE, rank-1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank<size-1) {
            MPI_Sendrecv(&out[k*nx*ny], nx*ny, MPI_DOUBLE, rank+1, 1,
                         &out[(k+1)*nx*ny], nx*ny, MPI_DOUBLE, rank+1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    long local_ms = duration.count(), max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // gather final grid to rank 0
    const std::vector<Real>& finalLocal = (iterations%2==0)? grid1 : grid2;
    // counts
    std::vector<int> counts(size), displs(size);
    size_t offset=0;
    for (int i=0;i<size;++i){
        size_t ki = base + (i<(int)rem?1:0);
        counts[i]=ki*nx*ny;
        displs[i]=offset*nx*ny;
        offset+=ki;
    }
    std::vector<Real> finalGrid;
    if (rank==0) finalGrid.assign(nx*ny*nz,0.0);
    // need send interior rows only (skip halos): finalLocal data + one plane offset
    std::vector<Real> sendbuf;
    if (k>0) {
        sendbuf.assign(k*nx*ny,0.0);
        for (size_t i=0;i<k;++i) memcpy(&sendbuf[i*nx*ny], &finalLocal[(i+1)*nx*ny], nx*ny*sizeof(Real));
    }
    MPI_Gatherv(sendbuf.data(), k*nx*ny, MPI_DOUBLE,
                rank==0?finalGrid.data()+nx*ny : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // fill boundaries z=0 and z=nz-1 on rank0
    if (rank==0) {
        for (size_t y=0;y<ny;++y) for (size_t x=0;x<nx;++x) {
            finalGrid[idx3(x,y,0,nx,ny)] = (idx3(x,y,0,nx,ny)%19)*1.0;
            finalGrid[idx3(x,y,nz-1,nx,ny)] = (idx3(x,y,nz-1,nx,ny)%19)*1.0;
        }
        printf("Computation time: %ld ms\n", max_ms);
        double cellUpdates = (double)((nx-2)*(ny-2)*(nz-2))*iterations;
        double mcups = cellUpdates/(max_ms/1000.0)/1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
        if (printResults) print_results(finalGrid,"Grid");
        if (validate) {
            printf("Validating result...\n");
            // reuse validateResult logic
            bool ok=true;
            for (auto v: finalGrid) if (std::isnan(v)||std::isinf(v)) { printf("Validation failed: found NaN or Inf value\n"); ok=false; break;}
            if (ok) {
                Real mn=finalGrid[0], mx=finalGrid[0];
                for (auto v: finalGrid){ mn=std::min(mn,v); mx=std::max(mx,v);}
                printf("Value range: [%.6f, %.6f]\n", mn, mx);
                if (mx>1e6 || mn<-1e6){ printf("Validation failed: values out of expected range\n"); ok=false;}
            }
            if (ok){ printf("Validation: PASSED\n"); }
            else { printf("Validation: FAILED\n"); }
        }
    }

    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    MPI_Finalize();
    return 0;
}
