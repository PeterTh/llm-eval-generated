#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Each MPI rank owns a contiguous slab in Z.  Two extra planes are halos.
__device__ __forceinline__ size_t cell(const size_t x, const size_t y, const size_t z,
                                       const size_t nx, const size_t ny) {
    return z * nx * ny + y * nx + x;
}

__global__ void chemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                  size_t nx, size_t ny, size_t localNz,
                                  double gamma, double eAA, double eBB, double eAB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > localNz) return;
    const size_t xp = (x + 1 < nx) ? x + 1 : x, xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y + 1 < ny) ? y + 1 : y, yn = (y > 0) ? y - 1 : 0;
    const size_t p = cell(x, y, z, nx, ny);
    const double v = c[p];
    const double lap = (c[cell(xp,y,z,nx,ny)] + c[cell(xn,y,z,nx,ny)] - 2.0*v)
                     + (c[cell(x,yp,z,nx,ny)] + c[cell(x,yn,z,nx,ny)] - 2.0*v)
                     + (c[cell(x,y,z+1,nx,ny)] + c[cell(x,y,z-1,nx,ny)] - 2.0*v);
    mu[p] = 4.5 * ((v + 1.0) * eAA + (v - 1.0) * eBB - 2.0 * v * eAB)
          + 3.0 * v + v * v * v - gamma * lap;
}

__global__ void updateConcentration(double* __restrict__ out, const double* __restrict__ old,
                                    const double* __restrict__ mu, size_t nx, size_t ny,
                                    size_t localNz, double dtD) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > localNz) return;
    const size_t xp = (x + 1 < nx) ? x + 1 : x, xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y + 1 < ny) ? y + 1 : y, yn = (y > 0) ? y - 1 : 0;
    const size_t p = cell(x, y, z, nx, ny);
    const double m = mu[p];
    const double lap = (mu[cell(xp,y,z,nx,ny)] + mu[cell(xn,y,z,nx,ny)] - 2.0*m)
                     + (mu[cell(x,yp,z,nx,ny)] + mu[cell(x,yn,z,nx,ny)] - 2.0*m)
                     + (mu[cell(x,y,z+1,nx,ny)] + mu[cell(x,y,z-1,nx,ny)] - 2.0*m);
    out[p] = old[p] + dtD * lap;
}

static void cudaCheck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

static void exchangeHalos(double* d, size_t plane, size_t localNz, int rank, int ranks,
                          MPI_Comm comm, std::vector<double>& lower, std::vector<double>& upper,
                          std::vector<double>& recvLower, std::vector<double>& recvUpper,
                          cudaStream_t stream) {
    const int below = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int above = (rank + 1 < ranks) ? rank + 1 : MPI_PROC_NULL;
    cudaCheck(cudaMemcpyAsync(lower.data(), d + plane, plane*sizeof(double), cudaMemcpyDeviceToHost, stream), "D2H lower");
    cudaCheck(cudaMemcpyAsync(upper.data(), d + localNz*plane, plane*sizeof(double), cudaMemcpyDeviceToHost, stream), "D2H upper");
    cudaCheck(cudaStreamSynchronize(stream), "halo D2H sync");
    MPI_Sendrecv(lower.data(), static_cast<int>(plane), MPI_DOUBLE, below, 41,
                 recvUpper.data(), static_cast<int>(plane), MPI_DOUBLE, above, 41, comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(upper.data(), static_cast<int>(plane), MPI_DOUBLE, above, 42,
                 recvLower.data(), static_cast<int>(plane), MPI_DOUBLE, below, 42, comm, MPI_STATUS_IGNORE);
    if (below == MPI_PROC_NULL) std::copy(lower.begin(), lower.end(), recvLower.begin());
    if (above == MPI_PROC_NULL) std::copy(upper.begin(), upper.end(), recvUpper.begin());
    cudaCheck(cudaMemcpyAsync(d, recvLower.data(), plane*sizeof(double), cudaMemcpyHostToDevice, stream), "H2D lower");
    cudaCheck(cudaMemcpyAsync(d + (localNz+1)*plane, recvUpper.data(), plane*sizeof(double), cudaMemcpyHostToDevice, stream), "H2D upper");
    cudaCheck(cudaStreamSynchronize(stream), "halo H2D sync");
}

static void initialize(std::vector<double>& c, size_t nx, size_t ny, size_t localNz,
                       size_t globalZ, size_t globalNz) {
    const size_t plane = nx * ny;
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < localNz; ++z) for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) {
        const size_t id = (globalZ + z) * plane + y * nx + x;
        const size_t pseudo = ((id + 1) * 1299709ULL) % (nx * ny * globalNz);
        c[z * plane + y * nx + x] = -1.0 + 2.0 * (static_cast<double>(pseudo) / (nx * ny * globalNz));
    }
}

static bool validate(const std::vector<double>& c) {
    bool ok = true; double lo = std::numeric_limits<double>::max(), hi = -lo;
    #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(&:ok)
    for (size_t i = 0; i < c.size(); ++i) { lo = std::min(lo, c[i]); hi = std::max(hi, c[i]); if (!std::isfinite(c[i])) ok = false; }
    printf("Concentration range: [%.6f, %.6f]\n", lo, hi);
    return ok && hi <= 10.0 && lo >= -10.0;
}

static void usage(const char* p) { printf("Usage: %s [options]\n  -x <num> Grid X (default 64)\n  -y <num> Grid Y (default X)\n  -z <num> Grid Z (default X)\n  -i <num> Time steps (default 20)\n  -v Validation\n  -r Print results\n  -h Help\n", p); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank, worldSize; MPI_Comm_rank(MPI_COMM_WORLD, &worldRank); MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx=64, ny=0, nz=0; int iterations=20; bool doValidate=false, printResults=false;
    for (int i=1; i<argc; ++i) {
        if (!strcmp(argv[i],"-x") && i+1<argc) nx=std::strtoull(argv[++i],nullptr,10);
        else if (!strcmp(argv[i],"-y") && i+1<argc) ny=std::strtoull(argv[++i],nullptr,10);
        else if (!strcmp(argv[i],"-z") && i+1<argc) nz=std::strtoull(argv[++i],nullptr,10);
        else if (!strcmp(argv[i],"-i") && i+1<argc) iterations=std::atoi(argv[++i]);
        else if (!strcmp(argv[i],"-v")) doValidate=true; else if (!strcmp(argv[i],"-r")) printResults=true;
        else if (!strcmp(argv[i],"-h")) { if(worldRank==0) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if(worldRank==0) usage(argv[0]); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    if (!ny) ny=nx; if (!nz) nz=nx;
    if (!nx || !ny || !nz || iterations < 0) MPI_Abort(MPI_COMM_WORLD,1);
    const int activeCount = static_cast<int>(std::min<size_t>(worldSize,nz));
    MPI_Comm activeComm; MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeCount ? 0 : MPI_UNDEFINED, worldRank, &activeComm);
    if (worldRank >= activeCount) { MPI_Finalize(); return 0; }
    int rank, ranks; MPI_Comm_rank(activeComm,&rank); MPI_Comm_size(activeComm,&ranks);
    const size_t base=nz/static_cast<size_t>(ranks), rem=nz%static_cast<size_t>(ranks);
    const size_t localNz=base+(static_cast<size_t>(rank)<rem), globalZ=static_cast<size_t>(rank)*base+std::min<size_t>(rank,rem);
    const size_t plane=nx*ny, localCells=(localNz+2)*plane;
    int deviceCount=0; cudaCheck(cudaGetDeviceCount(&deviceCount),"cudaGetDeviceCount");
    cudaCheck(cudaSetDevice(rank % deviceCount),"cudaSetDevice");
    cudaStream_t stream; cudaCheck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"cudaStreamCreate");
    std::vector<double> host(localNz*plane), result;
    std::vector<double> lower(plane), upper(plane), recvLower(plane), recvUpper(plane);
    initialize(host,nx,ny,localNz,globalZ,nz);
    double *dCold=nullptr,*dCnew=nullptr,*dMu=nullptr;
    cudaCheck(cudaMalloc(&dCold,localCells*sizeof(double)),"cudaMalloc cold"); cudaCheck(cudaMalloc(&dCnew,localCells*sizeof(double)),"cudaMalloc cnew"); cudaCheck(cudaMalloc(&dMu,localCells*sizeof(double)),"cudaMalloc mu");
    cudaCheck(cudaMemset(dCold,0,localCells*sizeof(double)),"cudaMemset");
    cudaCheck(cudaMemcpy(dCold+plane,host.data(),host.size()*sizeof(double),cudaMemcpyHostToDevice),"initial H2D");
    dim3 block(16,8,2), grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y,(localNz+block.z-1)/block.z);
    MPI_Barrier(activeComm); const auto start=std::chrono::steady_clock::now();
    for (int t=0;t<iterations;++t) {
        exchangeHalos(dCold,plane,localNz,rank,ranks,activeComm,lower,upper,recvLower,recvUpper,stream);
        chemicalPotential<<<grid,block,0,stream>>>(dCold,dMu,nx,ny,localNz,0.5,-2.0/9.0,-2.0/9.0,2.0/9.0);
        cudaCheck(cudaGetLastError(),"chemicalPotential");
        exchangeHalos(dMu,plane,localNz,rank,ranks,activeComm,lower,upper,recvLower,recvUpper,stream);
        updateConcentration<<<grid,block,0,stream>>>(dCnew,dCold,dMu,nx,ny,localNz,0.01);
        cudaCheck(cudaGetLastError(),"updateConcentration"); cudaCheck(cudaStreamSynchronize(stream),"update sync");
        std::swap(dCold,dCnew);
    }
    cudaCheck(cudaMemcpy(host.data(),dCold+plane,host.size()*sizeof(double),cudaMemcpyDeviceToHost),"final D2H");
    const auto end=std::chrono::steady_clock::now(); const double seconds=std::chrono::duration<double>(end-start).count();
    if (rank==0) result.resize(nx*ny*nz);
    std::vector<int> counts(ranks), displs(ranks); for(int r=0;r<ranks;++r){ const size_t n=base+(static_cast<size_t>(r)<rem); counts[r]=static_cast<int>(n*plane); displs[r]=static_cast<int>((static_cast<size_t>(r)*base+std::min<size_t>(r,rem))*plane); }
    MPI_Gatherv(host.data(),static_cast<int>(host.size()),MPI_DOUBLE,rank==0?result.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,activeComm);
    if (rank==0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,doValidate?"enabled":"disabled");
        printf("Running hybrid MPI + OpenMP + CUDA simulation\nComputation time: %.0f ms\nPerformance: %.3f MCellUpdates/s\n",seconds*1000.0,(static_cast<double>(nx)*ny*nz*iterations)/seconds/1e6);
        if(printResults) print_results(result,"Concentration");
        if(doValidate) { printf("Validating result...\n"); const bool valid = validate(result); printf("Validation: %s\n",valid?"PASSED":"FAILED"); if(!valid) MPI_Abort(MPI_COMM_WORLD,1); }
    }
    cudaFree(dCold); cudaFree(dCnew); cudaFree(dMu); cudaStreamDestroy(stream); MPI_Comm_free(&activeComm); MPI_Finalize(); return 0;
}
