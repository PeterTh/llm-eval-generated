#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

void reportCudaError(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d CUDA error in %s: %s\n", rank, what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

inline void checkCuda(cudaError_t err, const char* what) {
    reportCudaError(err, what);
}

struct Decomposition {
    int rank = 0;
    int nprocs = 1;
    int py = 1;
    int pz = 1;
    int ty = 0;
    int tz = 0;
    size_t nx = 0;
    size_t ny = 0;
    size_t nz = 0;
    size_t y0 = 0;
    size_t z0 = 0;
    int lny = 0;
    int lnz = 0;
    int pny = 0;
    int pnz = 0;
    size_t cells = 0;
    size_t alloc = 0;
};

void partitionBlock(size_t n, int parts, int coord, size_t& start, int& length) {
    if (static_cast<size_t>(parts) <= n) {
        start = (n * static_cast<size_t>(coord)) / static_cast<size_t>(parts);
        const size_t end = (n * static_cast<size_t>(coord + 1)) / static_cast<size_t>(parts);
        length = static_cast<int>(end - start);
    } else {
        if (static_cast<size_t>(coord) < n) {
            start = static_cast<size_t>(coord);
            length = 1;
        } else {
            start = n;
            length = 0;
        }
    }
}

void chooseDecomposition(Decomposition& d, size_t nx, size_t ny, size_t nz) {
    d.nx = nx;
    d.ny = ny;
    d.nz = nz;

    int best_py = 1;
    int best_pz = d.nprocs;
    bool best_full = false;
    double best_cost = std::numeric_limits<double>::infinity();

    for (int py = 1; py <= d.nprocs; ++py) {
        if (d.nprocs % py != 0) continue;
        const int pz = d.nprocs / py;
        const bool full = (static_cast<size_t>(py) <= ny) && (static_cast<size_t>(pz) <= nz);
        const double avg_y = static_cast<double>(ny) / static_cast<double>(py);
        const double avg_z = static_cast<double>(nz) / static_cast<double>(pz);
        const double cost = 1.0 / (avg_y > 0.0 ? avg_y : 1.0) +
                            1.0 / (avg_z > 0.0 ? avg_z : 1.0);
        if ((full && !best_full) || (full == best_full && cost < best_cost)) {
            best_full = full;
            best_cost = cost;
            best_py = py;
            best_pz = pz;
        }
    }

    d.py = best_py;
    d.pz = best_pz;
    d.ty = d.rank / d.pz;
    d.tz = d.rank % d.pz;

    partitionBlock(ny, d.py, d.ty, d.y0, d.lny);
    partitionBlock(nz, d.pz, d.tz, d.z0, d.lnz);

    d.pny = d.lny + 2;
    d.pnz = d.lnz + 2;
    d.cells = nx * static_cast<size_t>(d.lny) * static_cast<size_t>(d.lnz);
    d.alloc = nx * static_cast<size_t>(d.pny) * static_cast<size_t>(d.pnz);
}

__device__ __host__ __forceinline__ long long localIndex(int x, int y, int z, int nx, int pny) {
    return static_cast<long long>(x) +
           static_cast<long long>(nx) *
               (static_cast<long long>(y) + static_cast<long long>(pny) * static_cast<long long>(z));
}

__device__ __forceinline__ double laplacian7(const double* __restrict__ f, int x, int iy, int iz,
                                             int nx, int pny) {
    const long long base = localIndex(x, iy, iz, nx, pny);
    const int oxp = (x < nx - 1) ? 1 : 0;
    const int oxn = (x > 0) ? -1 : 0;
    const long long sy = static_cast<long long>(nx);
    const long long sz = static_cast<long long>(nx) * static_cast<long long>(pny);
    const double c0 = f[base];
    const double cxx = (f[base + oxp] + f[base + oxn] - 2.0 * c0);
    const double cyy = (f[base + sy] + f[base - sy] - 2.0 * c0);
    const double czz = (f[base + sz] + f[base - sz] - 2.0 * c0);
    return cxx + cyy + czz;
}

__global__ void computeChemicalPotentialKernel(const double* __restrict__ c,
                                               double* __restrict__ mu,
                                               int nx, int pny, int lny, int lnz,
                                               int y_off, int z_off, int mode,
                                               double gamma, double e_AA, double e_BB, double e_AB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = y_off + blockIdx.y * blockDim.y + threadIdx.y;
    const int z = z_off + blockIdx.z;
    if (x >= nx || y >= lny || z >= lnz) return;
    if (mode == 2 && y > 0 && y + 1 < lny && z > 0 && z + 1 < lnz) return;

    const int iy = y + 1;
    const int iz = z + 1;
    const long long idx = localIndex(x, iy, iz, nx, pny);
    const double cv = c[idx];

    const double t1 = __fma_rn(cv + 1.0, e_AA, (cv - 1.0) * e_BB);
    const double t2 = __fma_rn(-2.0 * cv, e_AB, t1);
    const double t3 = __fma_rn(t2, 4.5, 3.0 * cv);
    const double t4 = __fma_rn(cv * cv, cv, t3);
    const double lap = laplacian7(c, x, iy, iz, nx, pny);

    mu[idx] = __fma_rn(-gamma, lap, t4);
}

__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew,
                                         const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         int nx, int pny, int lny, int lnz,
                                         int y_off, int z_off, int mode,
                                         double coef) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = y_off + blockIdx.y * blockDim.y + threadIdx.y;
    const int z = z_off + blockIdx.z;
    if (x >= nx || y >= lny || z >= lnz) return;
    if (mode == 2 && y > 0 && y + 1 < lny && z > 0 && z + 1 < lnz) return;

    const int iy = y + 1;
    const int iz = z + 1;
    const long long idx = localIndex(x, iy, iz, nx, pny);
    const double lap = laplacian7(mu, x, iy, iz, nx, pny);
    cnew[idx] = __fma_rn(lap, coef, cold[idx]);
}

struct FaceBuffers {
    double* send_ym = nullptr;
    double* recv_ym = nullptr;
    double* send_yp = nullptr;
    double* recv_yp = nullptr;
    double* send_zm = nullptr;
    double* recv_zm = nullptr;
    double* send_zp = nullptr;
    double* recv_zp = nullptr;
    size_t y_face = 1;
    size_t z_face = 1;
};

void allocFaceBuffers(FaceBuffers& fb, size_t nx, int lny, int lnz) {
    fb.y_face = nx * static_cast<size_t>(lnz > 0 ? lnz : 1);
    fb.z_face = nx * static_cast<size_t>(lny > 0 ? lny : 1);
    const size_t y_bytes = fb.y_face * sizeof(double);
    const size_t z_bytes = fb.z_face * sizeof(double);
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&fb.send_ym), y_bytes), "cudaMallocHost");
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&fb.recv_ym), y_bytes), "cudaMallocHost");
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&fb.send_yp), y_bytes), "cudaMallocHost");
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&fb.recv_yp), y_bytes), "cudaMallocHost");
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&fb.send_zm), z_bytes), "cudaMallocHost");
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&fb.recv_zm), z_bytes), "cudaMallocHost");
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&fb.send_zp), z_bytes), "cudaMallocHost");
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&fb.recv_zp), z_bytes), "cudaMallocHost");
}

void freeFaceBuffers(FaceBuffers& fb) {
    if (fb.send_ym) cudaFreeHost(fb.send_ym);
    if (fb.recv_ym) cudaFreeHost(fb.recv_ym);
    if (fb.send_yp) cudaFreeHost(fb.send_yp);
    if (fb.recv_yp) cudaFreeHost(fb.recv_yp);
    if (fb.send_zm) cudaFreeHost(fb.send_zm);
    if (fb.recv_zm) cudaFreeHost(fb.recv_zm);
    if (fb.send_zp) cudaFreeHost(fb.send_zp);
    if (fb.recv_zp) cudaFreeHost(fb.recv_zp);
}

void copy3DAsync(void* dst, size_t dst_pitch, size_t dst_ysize, size_t dst_x, size_t dst_y, size_t dst_z,
                 const void* src, size_t src_pitch, size_t src_ysize, size_t src_x, size_t src_y, size_t src_z,
                 size_t width_bytes, size_t height, size_t depth, cudaMemcpyKind kind, cudaStream_t stream) {
    cudaMemcpy3DParms p{};
    p.srcPtr = make_cudaPitchedPtr(const_cast<void*>(src), src_pitch,
                                   width_bytes / sizeof(double), src_ysize);
    p.dstPtr = make_cudaPitchedPtr(dst, dst_pitch, width_bytes / sizeof(double), dst_ysize);
    p.srcPos = make_cudaPos(src_x, src_y, src_z);
    p.dstPos = make_cudaPos(dst_x, dst_y, dst_z);
    p.extent = make_cudaExtent(width_bytes, height, depth);
    p.kind = kind;
    checkCuda(cudaMemcpy3DAsync(&p, stream), "cudaMemcpy3DAsync");
}

struct ExchangeState {
    MPI_Request req[8];
    int nreq = 0;
};

void startExchange(double* field, const Decomposition& d, FaceBuffers& fb,
                   cudaStream_t stream, ExchangeState& st) {
    st.nreq = 0;
    if (d.cells == 0) return;

    const size_t row_bytes = d.nx * sizeof(double);
    const size_t dev_pitch = row_bytes;
    const size_t dev_ysize = static_cast<size_t>(d.pny);
    const bool has_ym = d.y0 > 0;
    const bool has_yp = d.y0 + static_cast<size_t>(d.lny) < d.ny;
    const bool has_zm = d.z0 > 0;
    const bool has_zp = d.z0 + static_cast<size_t>(d.lnz) < d.nz;

    if (has_ym) {
        copy3DAsync(fb.send_ym, row_bytes, 1, 0, 0, 0,
                    field, dev_pitch, dev_ysize, 0, 1, 1,
                    row_bytes, 1, static_cast<size_t>(d.lnz),
                    cudaMemcpyDeviceToHost, stream);
    } else if (d.lny > 0) {
        copy3DAsync(field, dev_pitch, dev_ysize, 0, 0, 1,
                    field, dev_pitch, dev_ysize, 0, 1, 1,
                    row_bytes, 1, static_cast<size_t>(d.lnz),
                    cudaMemcpyDeviceToDevice, stream);
    }

    if (has_yp) {
        copy3DAsync(fb.send_yp, row_bytes, 1, 0, 0, 0,
                    field, dev_pitch, dev_ysize, 0, d.lny, 1,
                    row_bytes, 1, static_cast<size_t>(d.lnz),
                    cudaMemcpyDeviceToHost, stream);
    } else if (d.lny > 0) {
        copy3DAsync(field, dev_pitch, dev_ysize, 0, d.lny + 1, 1,
                    field, dev_pitch, dev_ysize, 0, d.lny, 1,
                    row_bytes, 1, static_cast<size_t>(d.lnz),
                    cudaMemcpyDeviceToDevice, stream);
    }

    if (has_zm) {
        copy3DAsync(fb.send_zm, row_bytes, static_cast<size_t>(d.lny), 0, 0, 0,
                    field, dev_pitch, dev_ysize, 0, 1, 1,
                    row_bytes, static_cast<size_t>(d.lny), 1,
                    cudaMemcpyDeviceToHost, stream);
    } else if (d.lnz > 0) {
        copy3DAsync(field, dev_pitch, dev_ysize, 0, 1, 0,
                    field, dev_pitch, dev_ysize, 0, 1, 1,
                    row_bytes, static_cast<size_t>(d.lny), 1,
                    cudaMemcpyDeviceToDevice, stream);
    }

    if (has_zp) {
        copy3DAsync(fb.send_zp, row_bytes, static_cast<size_t>(d.lny), 0, 0, 0,
                    field, dev_pitch, dev_ysize, 0, 1, d.lnz,
                    row_bytes, static_cast<size_t>(d.lny), 1,
                    cudaMemcpyDeviceToHost, stream);
    } else if (d.lnz > 0) {
        copy3DAsync(field, dev_pitch, dev_ysize, 0, 1, d.lnz + 1,
                    field, dev_pitch, dev_ysize, 0, 1, d.lnz,
                    row_bytes, static_cast<size_t>(d.lny), 1,
                    cudaMemcpyDeviceToDevice, stream);
    }

    const int neighbor_count = (has_ym ? 1 : 0) + (has_yp ? 1 : 0) +
                               (has_zm ? 1 : 0) + (has_zp ? 1 : 0);
    if (neighbor_count == 0) return;

    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    const int cnt_y = static_cast<int>(fb.y_face);
    const int cnt_z = static_cast<int>(fb.z_face);
    const int tag_y = 1;
    const int tag_z = 2;

    if (has_ym) {
        MPI_Irecv(fb.recv_ym, cnt_y, MPI_DOUBLE, d.rank - d.pz, tag_y, MPI_COMM_WORLD, &st.req[st.nreq++]);
    }
    if (has_yp) {
        MPI_Irecv(fb.recv_yp, cnt_y, MPI_DOUBLE, d.rank + d.pz, tag_y, MPI_COMM_WORLD, &st.req[st.nreq++]);
    }
    if (has_zm) {
        MPI_Irecv(fb.recv_zm, cnt_z, MPI_DOUBLE, d.rank - 1, tag_z, MPI_COMM_WORLD, &st.req[st.nreq++]);
    }
    if (has_zp) {
        MPI_Irecv(fb.recv_zp, cnt_z, MPI_DOUBLE, d.rank + 1, tag_z, MPI_COMM_WORLD, &st.req[st.nreq++]);
    }

    if (has_ym) {
        MPI_Isend(fb.send_ym, cnt_y, MPI_DOUBLE, d.rank - d.pz, tag_y, MPI_COMM_WORLD, &st.req[st.nreq++]);
    }
    if (has_yp) {
        MPI_Isend(fb.send_yp, cnt_y, MPI_DOUBLE, d.rank + d.pz, tag_y, MPI_COMM_WORLD, &st.req[st.nreq++]);
    }
    if (has_zm) {
        MPI_Isend(fb.send_zm, cnt_z, MPI_DOUBLE, d.rank - 1, tag_z, MPI_COMM_WORLD, &st.req[st.nreq++]);
    }
    if (has_zp) {
        MPI_Isend(fb.send_zp, cnt_z, MPI_DOUBLE, d.rank + 1, tag_z, MPI_COMM_WORLD, &st.req[st.nreq++]);
    }
}

void finishExchange(double* field, const Decomposition& d, FaceBuffers& fb,
                    cudaStream_t stream, ExchangeState& st) {
    if (d.cells == 0) return;

    if (st.nreq > 0) {
        MPI_Waitall(st.nreq, st.req, MPI_STATUSES_IGNORE);
        st.nreq = 0;
    }

    const size_t row_bytes = d.nx * sizeof(double);
    const size_t dev_pitch = row_bytes;
    const size_t dev_ysize = static_cast<size_t>(d.pny);
    const bool has_ym = d.y0 > 0;
    const bool has_yp = d.y0 + static_cast<size_t>(d.lny) < d.ny;
    const bool has_zm = d.z0 > 0;
    const bool has_zp = d.z0 + static_cast<size_t>(d.lnz) < d.nz;

    if (has_ym) {
        copy3DAsync(field, dev_pitch, dev_ysize, 0, 0, 1,
                    fb.recv_ym, row_bytes, 1, 0, 0, 0,
                    row_bytes, 1, static_cast<size_t>(d.lnz),
                    cudaMemcpyHostToDevice, stream);
    }
    if (has_yp) {
        copy3DAsync(field, dev_pitch, dev_ysize, 0, d.lny + 1, 1,
                    fb.recv_yp, row_bytes, 1, 0, 0, 0,
                    row_bytes, 1, static_cast<size_t>(d.lnz),
                    cudaMemcpyHostToDevice, stream);
    }
    if (has_zm) {
        copy3DAsync(field, dev_pitch, dev_ysize, 0, 1, 0,
                    fb.recv_zm, row_bytes, static_cast<size_t>(d.lny), 0, 0, 0,
                    row_bytes, static_cast<size_t>(d.lny), 1,
                    cudaMemcpyHostToDevice, stream);
    }
    if (has_zp) {
        copy3DAsync(field, dev_pitch, dev_ysize, 0, 1, d.lnz + 1,
                    fb.recv_zp, row_bytes, static_cast<size_t>(d.lny), 0, 0, 0,
                    row_bytes, static_cast<size_t>(d.lny), 1,
                    cudaMemcpyHostToDevice, stream);
    }
}

void initializeConcentration(double* d_cold, const Decomposition& d, cudaStream_t stream) {
    if (d.cells == 0) return;

    std::vector<double> h(d.cells);
    const size_t vol = d.nx * d.ny * d.nz;

#pragma omp parallel for collapse(2) schedule(static)
    for (int lz = 0; lz < d.lnz; ++lz) {
        for (int ly = 0; ly < d.lny; ++ly) {
            const size_t gz = d.z0 + static_cast<size_t>(lz);
            const size_t gy = d.y0 + static_cast<size_t>(ly);
            double* row = h.data() + (static_cast<size_t>(lz) * static_cast<size_t>(d.lny) +
                                      static_cast<size_t>(ly)) * d.nx;
            for (size_t x = 0; x < d.nx; ++x) {
                const size_t linear_id = gz * (d.nx * d.ny) + gy * d.nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                row[x] = std::fma(pseudo, 2.0, -1.0);
            }
        }
    }

    copy3DAsync(d_cold, d.nx * sizeof(double), static_cast<size_t>(d.pny), 0, 1, 1,
                h.data(), d.nx * sizeof(double), static_cast<size_t>(d.lny), 0, 0, 0,
                d.nx * sizeof(double), static_cast<size_t>(d.lny), static_cast<size_t>(d.lnz),
                cudaMemcpyHostToDevice, stream);
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

void gatherResults(const double* d_cold, const Decomposition& d,
                   std::vector<double>& global, cudaStream_t stream) {
    const int rank = d.rank;
    const int nprocs = d.nprocs;
    const size_t nx = d.nx;
    const size_t ny = d.ny;
    const size_t nz = d.nz;

    std::vector<double> local(d.cells > 0 ? d.cells : 1);
    if (d.cells > 0) {
        copy3DAsync(local.data(), nx * sizeof(double), static_cast<size_t>(d.lny), 0, 0, 0,
                    d_cold, nx * sizeof(double), static_cast<size_t>(d.pny), 0, 1, 1,
                    nx * sizeof(double), static_cast<size_t>(d.lny), static_cast<size_t>(d.lnz),
                    cudaMemcpyDeviceToHost, stream);
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }

    std::vector<int> counts(nprocs);
    std::vector<int> displs(nprocs);
    long long total_ll = 0;
    for (int r = 0; r < nprocs; ++r) {
        const int ty = r / d.pz;
        const int tz = r % d.pz;
        size_t y0 = 0, z0 = 0;
        int lny = 0, lnz = 0;
        partitionBlock(ny, d.py, ty, y0, lny);
        partitionBlock(nz, d.pz, tz, z0, lnz);
        counts[r] = static_cast<int>(nx * static_cast<size_t>(lny) * static_cast<size_t>(lnz));
        displs[r] = static_cast<int>(total_ll);
        total_ll += counts[r];
    }

    std::vector<double> gathered;
    if (rank == 0) {
        gathered.resize(static_cast<size_t>(total_ll));
    }

    MPI_Gatherv(local.data(), counts[rank], MPI_DOUBLE,
                rank == 0 ? gathered.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0) return;

    global.assign(nx * ny * nz, 0.0);

    std::vector<int> blockY(ny);
    std::vector<int> localY(ny);
    std::vector<int> blockZ(nz);
    std::vector<int> localZ(nz);

    for (int ty = 0; ty < d.py; ++ty) {
        size_t y0 = 0;
        int lny = 0;
        partitionBlock(ny, d.py, ty, y0, lny);
        for (int i = 0; i < lny; ++i) {
            blockY[y0 + static_cast<size_t>(i)] = ty;
            localY[y0 + static_cast<size_t>(i)] = i;
        }
    }
    for (int tz = 0; tz < d.pz; ++tz) {
        size_t z0 = 0;
        int lnz = 0;
        partitionBlock(nz, d.pz, tz, z0, lnz);
        for (int i = 0; i < lnz; ++i) {
            blockZ[z0 + static_cast<size_t>(i)] = tz;
            localZ[z0 + static_cast<size_t>(i)] = i;
        }
    }

    std::vector<int> lny_by_ty(d.py);
    for (int ty = 0; ty < d.py; ++ty) {
        size_t y0 = 0;
        int lny = 0;
        partitionBlock(ny, d.py, ty, y0, lny);
        lny_by_ty[ty] = lny;
    }

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t gz = 0; gz < nz; ++gz) {
        for (size_t gy = 0; gy < ny; ++gy) {
            const int ty = blockY[gy];
            const int tz = blockZ[gz];
            const int r = ty * d.pz + tz;
            const size_t src_off = static_cast<size_t>(displs[r]) +
                                   (static_cast<size_t>(localZ[gz]) *
                                        static_cast<size_t>(lny_by_ty[ty]) +
                                    static_cast<size_t>(localY[gy])) *
                                       nx;
            std::copy_n(gathered.data() + src_off, nx,
                        global.data() + (gz * ny + gy) * nx);
        }
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    int local_rank = rank;
    MPI_Comm shm_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
    if (shm_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(shm_comm, &local_rank);
    }

    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA device found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(local_rank % device_count), "cudaSetDevice");
    if (shm_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&shm_comm);
    }

    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    Decomposition dom;
    dom.rank = rank;
    dom.nprocs = nprocs;
    chooseDecomposition(dom, nx, ny, nz);

    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    const double coef = dt * D;
    (void)dx;
    (void)dy;
    (void)dz;

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_cold), dom.alloc * sizeof(double)), "cudaMalloc");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_cnew), dom.alloc * sizeof(double)), "cudaMalloc");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_mu), dom.alloc * sizeof(double)), "cudaMalloc");

    FaceBuffers fb;
    allocFaceBuffers(fb, dom.nx, dom.lny, dom.lnz);

    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(d_cold, dom, stream);

    const int nx_i = static_cast<int>(dom.nx);
    const dim3 block(32, std::min(8, dom.lny > 0 ? dom.lny : 1), 1);
    const dim3 grid((nx_i + block.x - 1) / block.x,
                    (dom.lny + static_cast<int>(block.y) - 1) / static_cast<int>(block.y),
                    dom.lnz > 0 ? dom.lnz : 1);

    const int inner_y = dom.lny - 2;
    const int inner_z = dom.lnz - 2;
    const bool has_inner = inner_y > 0 && inner_z > 0;
    const dim3 inner_block(32, std::min(8, inner_y > 0 ? inner_y : 1), 1);
    const dim3 inner_grid((nx_i + inner_block.x - 1) / inner_block.x,
                          (inner_y + static_cast<int>(inner_block.y) - 1) /
                              static_cast<int>(inner_block.y),
                          inner_z > 0 ? inner_z : 1);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    ExchangeState exchange;
    for (int t = 0; t < iterations; ++t) {
        startExchange(d_cold, dom, fb, stream, exchange);

        if (has_inner) {
            computeChemicalPotentialKernel<<<inner_grid, inner_block, 0, stream>>>(
                d_cold, d_mu, nx_i, dom.pny, dom.lny, dom.lnz,
                1, 1, 1, gamma, e_AA, e_BB, e_AB);
            checkCuda(cudaGetLastError(), "computeChemicalPotentialKernel");
        }

        finishExchange(d_cold, dom, fb, stream, exchange);

        if (dom.cells > 0) {
            computeChemicalPotentialKernel<<<grid, block, 0, stream>>>(
                d_cold, d_mu, nx_i, dom.pny, dom.lny, dom.lnz,
                0, 0, 2, gamma, e_AA, e_BB, e_AB);
            checkCuda(cudaGetLastError(), "computeChemicalPotentialKernel");
        }

        startExchange(d_mu, dom, fb, stream, exchange);

        if (has_inner) {
            cahnHilliardUpdateKernel<<<inner_grid, inner_block, 0, stream>>>(
                d_cnew, d_cold, d_mu, nx_i, dom.pny, dom.lny, dom.lnz,
                1, 1, 1, coef);
            checkCuda(cudaGetLastError(), "cahnHilliardUpdateKernel");
        }

        finishExchange(d_mu, dom, fb, stream, exchange);

        if (dom.cells > 0) {
            cahnHilliardUpdateKernel<<<grid, block, 0, stream>>>(
                d_cnew, d_cold, d_mu, nx_i, dom.pny, dom.lny, dom.lnz,
                0, 0, 2, coef);
            checkCuda(cudaGetLastError(), "cahnHilliardUpdateKernel");
        }

        std::swap(d_cold, d_cnew);
    }

    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    const long local_duration_ms = static_cast<long>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    const double elapsed = static_cast<double>(max_duration_ms) / 1000.0;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);

        const double gridSize = static_cast<double>(nx) * static_cast<double>(ny) * static_cast<double>(nz);
        const double cellUpdates = gridSize * static_cast<double>(iterations);
        const double mcups = cellUpdates / (elapsed > 0.0 ? elapsed : 1.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<double> global;
    if (printResults || validate) {
        gatherResults(d_cold, dom, global, stream);
    }

    int exit_code = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(global, "Concentration");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(global, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    int global_exit = 0;
    MPI_Allreduce(&exit_code, &global_exit, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    freeFaceBuffers(fb);
    cudaFree(d_cold);
    cudaFree(d_cnew);
    cudaFree(d_mu);
    cudaStreamDestroy(stream);

    MPI_Finalize();
    return global_exit;
}
