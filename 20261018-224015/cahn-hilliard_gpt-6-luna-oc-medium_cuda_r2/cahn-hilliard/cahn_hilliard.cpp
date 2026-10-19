#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); std::exit(1); } } while (0)

__device__ __forceinline__ double laplacian(const double* __restrict__ a, size_t x, size_t y, size_t z,
                                             size_t nx, size_t ny, size_t nz) {
    const size_t i = z * nx * ny + y * nx + x;
    const size_t xp = x + (x + 1 < nx), xm = x - (x > 0);
    const size_t yp = y + (y + 1 < ny), ym = y - (y > 0);
    const size_t zp = z + (z + 1 < nz), zm = z - (z > 0);
    const double c = a[i];
    return (a[z*nx*ny+y*nx+xp] + a[z*nx*ny+y*nx+xm] - 2.0*c)
         + (a[z*nx*ny+yp*nx+x] + a[z*nx*ny+ym*nx+x] - 2.0*c)
         + (a[zp*nx*ny+y*nx+x] + a[zm*nx*ny+y*nx+x] - 2.0*c);
}

__global__ void initialize(double* c, size_t nx, size_t ny, size_t nz, size_t vol) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vol) return;
    c[i] = -1.0 + 2.0 * (((((i + 1) * size_t(1299709)) % vol) / double(vol)));
}

__global__ void chemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                  size_t nx, size_t ny, size_t nz) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t vol = nx * ny * nz;
    if (i >= vol) return;
    const size_t x = i % nx, y = (i / nx) % ny, z = i / (nx * ny);
    const double v = c[i];
    // Preserve the original expression and its parameter values.
    mu[i] = 4.5 * ((v + 1.0) * (-2.0/9.0) + (v - 1.0) * (-2.0/9.0) - 2.0*v*(2.0/9.0))
          + 3.0*v + v*v*v - 0.5 * laplacian(c, x, y, z, nx, ny, nz);
}

__global__ void updateField(double* __restrict__ out, const double* __restrict__ old,
                            const double* __restrict__ mu, size_t nx, size_t ny, size_t nz) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t vol = nx * ny * nz;
    if (i >= vol) return;
    const size_t x = i % nx, y = (i / nx) % ny, z = i / (nx * ny);
    out[i] = old[i] + 0.01 * laplacian(mu, x, y, z, nx, ny, nz);
}

bool validateResult(const std::vector<double>& c) {
    for (double v : c) if (!std::isfinite(v)) { std::printf("Validation failed: found NaN or Inf value\n"); return false; }
    const auto bounds = std::minmax_element(c.begin(), c.end());
    std::printf("Concentration range: [%.6f, %.6f]\n", *bounds.first, *bounds.second);
    if (*bounds.second > 10.0 || *bounds.first < -10.0) { std::printf("Validation failed: values out of expected range\n"); return false; }
    return true;
}

void printUsage(const char* p) {
    std::printf("Usage: %s [options]\nOptions:\n  -x <num>     Grid size in X dimension (default: 64)\n  -y <num>     Grid size in Y dimension (default: same as X)\n  -z <num>     Grid size in Z dimension (default: same as X)\n  -i <num>     Number of time steps (default: 20)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    size_t nx=64, ny=0, nz=0; int iterations=20; bool validate=false, printResults=false;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i+1<argc) nx=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-y") && i+1<argc) ny=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-z") && i+1<argc) nz=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i+1<argc) iterations=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate=true;
        else if (!std::strcmp(argv[i], "-r")) printResults=true;
        else if (!std::strcmp(argv[i], "-h")) { printUsage(argv[0]); return 0; }
        else { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    if (!ny) ny=nx; if (!nz) nz=nx;
    if (!nx || !ny || !nz || iterations < 0 || nx > SIZE_MAX/ny || nx*ny > SIZE_MAX/nz) { std::fprintf(stderr,"Invalid grid dimensions or iteration count\n"); return 1; }
    const size_t volume=nx*ny*nz;
    std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled");
    double *a=nullptr,*b=nullptr,*mu=nullptr;
    CUDA_CHECK(cudaMalloc(&a, volume*sizeof(double))); CUDA_CHECK(cudaMalloc(&b, volume*sizeof(double))); CUDA_CHECK(cudaMalloc(&mu, volume*sizeof(double)));
    constexpr int threads=256; const unsigned blocks=static_cast<unsigned>((volume+threads-1)/threads);
    std::printf("Initializing concentration field...\n"); initialize<<<blocks,threads>>>(a,nx,ny,nz,volume); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("Running Cahn-Hilliard simulation...\n");
    auto start=std::chrono::high_resolution_clock::now();
    for (int t=0;t<iterations;++t) {
        chemicalPotential<<<blocks,threads>>>(a,mu,nx,ny,nz);
        updateField<<<blocks,threads>>>(b,a,mu,nx,ny,nz);
        std::swap(a,b);
    }
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    auto end=std::chrono::high_resolution_clock::now();
    const auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::printf("Computation time: %ld ms\n", static_cast<long>(ms));
    const double mcups=ms ? double(volume)*iterations/(ms/1000.0)/1e6 : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n",mcups);
    std::vector<double> result(volume); CUDA_CHECK(cudaMemcpy(result.data(),a,volume*sizeof(double),cudaMemcpyDeviceToHost));
    if (printResults) print_results(result,"Concentration");
    bool valid=true; if (validate) { std::printf("Validating result...\n"); valid=validateResult(result); std::printf("Validation: %s\n",valid?"PASSED":"FAILED"); }
    CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(mu)); return validate && !valid ? 1 : 0;
}
