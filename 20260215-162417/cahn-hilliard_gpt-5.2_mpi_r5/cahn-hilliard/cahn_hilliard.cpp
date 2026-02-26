#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Global 3D index calculation (original layout)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Decomp3D {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;

    int dims[3] = {1, 1, 1};
    int coords[3] = {0, 0, 0};

    int nbr_xm = MPI_PROC_NULL;
    int nbr_xp = MPI_PROC_NULL;
    int nbr_ym = MPI_PROC_NULL;
    int nbr_yp = MPI_PROC_NULL;
    int nbr_zm = MPI_PROC_NULL;
    int nbr_zp = MPI_PROC_NULL;

    size_t x0 = 0, y0 = 0, z0 = 0;
    size_t lx = 0, ly = 0, lz = 0;
};

inline void decompose1d(const size_t n, const int p, const int coord, size_t& start, size_t& nloc) {
    const size_t base = n / static_cast<size_t>(p);
    const size_t rem = n % static_cast<size_t>(p);
    nloc = base + (static_cast<size_t>(coord) < rem ? 1 : 0);
    start = static_cast<size_t>(coord) * base + std::min(static_cast<size_t>(coord), rem);
}

struct HaloBuffers {
    std::vector<double> sxm, sxp, rxm, rxp;
    std::vector<double> sym, syp, rym, ryp;
    std::vector<double> szm, szp, rzm, rzp;

    explicit HaloBuffers(const size_t lx, const size_t ly, const size_t lz)
        : sxm(ly * lz), sxp(ly * lz), rxm(ly * lz), rxp(ly * lz),
          sym(lx * lz), syp(lx * lz), rym(lx * lz), ryp(lx * lz),
          szm(lx * ly), szp(lx * ly), rzm(lx * ly), rzp(lx * ly) {}
};

struct Exchange {
    std::array<MPI_Request, 12> req{};
};

inline size_t idxL(const size_t x, const size_t y, const size_t z, const size_t lxp2, const size_t lyp2) noexcept {
    return (z * lyp2 + y) * lxp2 + x;
}

static inline Exchange start_halo_exchange(const double* a,
                                          const Decomp3D& d,
                                          const size_t lxp2,
                                          const size_t lyp2,
                                          const size_t lx,
                                          const size_t ly,
                                          const size_t lz,
                                          HaloBuffers& b) {
    static constexpr int TAG_XP = 10, TAG_XM = 11;
    static constexpr int TAG_YP = 20, TAG_YM = 21;
    static constexpr int TAG_ZP = 30, TAG_ZM = 31;

    Exchange ex;
    ex.req.fill(MPI_REQUEST_NULL);
    int r = 0;

    // Pack X faces (x=1 and x=lx) -> buffers are indexed by (z,y)
    if (d.nbr_xm != MPI_PROC_NULL || d.nbr_xp != MPI_PROC_NULL) {
        for (size_t z = 1; z <= lz; ++z) {
            const size_t zo = (z - 1) * ly;
            for (size_t y = 1; y <= ly; ++y) {
                const size_t o = zo + (y - 1);
                b.sxm[o] = a[idxL(1, y, z, lxp2, lyp2)];
                b.sxp[o] = a[idxL(lx, y, z, lxp2, lyp2)];
            }
        }
    }

    // Pack Y faces (y=1 and y=ly) -> buffers are indexed by (z,x)
    if (d.nbr_ym != MPI_PROC_NULL || d.nbr_yp != MPI_PROC_NULL) {
        for (size_t z = 1; z <= lz; ++z) {
            const size_t zo = (z - 1) * lx;
            for (size_t x = 1; x <= lx; ++x) {
                const size_t o = zo + (x - 1);
                b.sym[o] = a[idxL(x, 1, z, lxp2, lyp2)];
                b.syp[o] = a[idxL(x, ly, z, lxp2, lyp2)];
            }
        }
    }

    // Pack Z faces (z=1 and z=lz) -> buffers are indexed by (y,x)
    if (d.nbr_zm != MPI_PROC_NULL || d.nbr_zp != MPI_PROC_NULL) {
        for (size_t y = 1; y <= ly; ++y) {
            const size_t yo = (y - 1) * lx;
            for (size_t x = 1; x <= lx; ++x) {
                const size_t o = yo + (x - 1);
                b.szm[o] = a[idxL(x, y, 1, lxp2, lyp2)];
                b.szp[o] = a[idxL(x, y, lz, lxp2, lyp2)];
            }
        }
    }

    const int cnt_x = static_cast<int>(ly * lz);
    const int cnt_y = static_cast<int>(lx * lz);
    const int cnt_z = static_cast<int>(lx * ly);

    if (d.nbr_xm != MPI_PROC_NULL) MPI_Irecv(b.rxm.data(), cnt_x, MPI_DOUBLE, d.nbr_xm, TAG_XP, d.comm, &ex.req[r++]);
    if (d.nbr_xp != MPI_PROC_NULL) MPI_Irecv(b.rxp.data(), cnt_x, MPI_DOUBLE, d.nbr_xp, TAG_XM, d.comm, &ex.req[r++]);
    if (d.nbr_ym != MPI_PROC_NULL) MPI_Irecv(b.rym.data(), cnt_y, MPI_DOUBLE, d.nbr_ym, TAG_YP, d.comm, &ex.req[r++]);
    if (d.nbr_yp != MPI_PROC_NULL) MPI_Irecv(b.ryp.data(), cnt_y, MPI_DOUBLE, d.nbr_yp, TAG_YM, d.comm, &ex.req[r++]);
    if (d.nbr_zm != MPI_PROC_NULL) MPI_Irecv(b.rzm.data(), cnt_z, MPI_DOUBLE, d.nbr_zm, TAG_ZP, d.comm, &ex.req[r++]);
    if (d.nbr_zp != MPI_PROC_NULL) MPI_Irecv(b.rzp.data(), cnt_z, MPI_DOUBLE, d.nbr_zp, TAG_ZM, d.comm, &ex.req[r++]);

    if (d.nbr_xp != MPI_PROC_NULL) MPI_Isend(b.sxp.data(), cnt_x, MPI_DOUBLE, d.nbr_xp, TAG_XP, d.comm, &ex.req[r++]);
    if (d.nbr_xm != MPI_PROC_NULL) MPI_Isend(b.sxm.data(), cnt_x, MPI_DOUBLE, d.nbr_xm, TAG_XM, d.comm, &ex.req[r++]);
    if (d.nbr_yp != MPI_PROC_NULL) MPI_Isend(b.syp.data(), cnt_y, MPI_DOUBLE, d.nbr_yp, TAG_YP, d.comm, &ex.req[r++]);
    if (d.nbr_ym != MPI_PROC_NULL) MPI_Isend(b.sym.data(), cnt_y, MPI_DOUBLE, d.nbr_ym, TAG_YM, d.comm, &ex.req[r++]);
    if (d.nbr_zp != MPI_PROC_NULL) MPI_Isend(b.szp.data(), cnt_z, MPI_DOUBLE, d.nbr_zp, TAG_ZP, d.comm, &ex.req[r++]);
    if (d.nbr_zm != MPI_PROC_NULL) MPI_Isend(b.szm.data(), cnt_z, MPI_DOUBLE, d.nbr_zm, TAG_ZM, d.comm, &ex.req[r++]);

    return ex;
}

static inline void finish_halo_exchange(double* a,
                                       const Decomp3D& d,
                                       const size_t lxp2,
                                       const size_t lyp2,
                                       const size_t lx,
                                       const size_t ly,
                                       const size_t lz,
                                       HaloBuffers& b,
                                       Exchange& ex) {
    MPI_Waitall(static_cast<int>(ex.req.size()), ex.req.data(), MPI_STATUSES_IGNORE);

    // Unpack X halos
    if (d.nbr_xm != MPI_PROC_NULL) {
        for (size_t z = 1; z <= lz; ++z) {
            const size_t zo = (z - 1) * ly;
            for (size_t y = 1; y <= ly; ++y) {
                a[idxL(0, y, z, lxp2, lyp2)] = b.rxm[zo + (y - 1)];
            }
        }
    } else {
        for (size_t z = 1; z <= lz; ++z) {
            for (size_t y = 1; y <= ly; ++y) {
                a[idxL(0, y, z, lxp2, lyp2)] = a[idxL(1, y, z, lxp2, lyp2)];
            }
        }
    }

    if (d.nbr_xp != MPI_PROC_NULL) {
        for (size_t z = 1; z <= lz; ++z) {
            const size_t zo = (z - 1) * ly;
            for (size_t y = 1; y <= ly; ++y) {
                a[idxL(lx + 1, y, z, lxp2, lyp2)] = b.rxp[zo + (y - 1)];
            }
        }
    } else {
        for (size_t z = 1; z <= lz; ++z) {
            for (size_t y = 1; y <= ly; ++y) {
                a[idxL(lx + 1, y, z, lxp2, lyp2)] = a[idxL(lx, y, z, lxp2, lyp2)];
            }
        }
    }

    // Unpack Y halos
    if (d.nbr_ym != MPI_PROC_NULL) {
        for (size_t z = 1; z <= lz; ++z) {
            const size_t zo = (z - 1) * lx;
            for (size_t x = 1; x <= lx; ++x) {
                a[idxL(x, 0, z, lxp2, lyp2)] = b.rym[zo + (x - 1)];
            }
        }
    } else {
        for (size_t z = 1; z <= lz; ++z) {
            for (size_t x = 1; x <= lx; ++x) {
                a[idxL(x, 0, z, lxp2, lyp2)] = a[idxL(x, 1, z, lxp2, lyp2)];
            }
        }
    }

    if (d.nbr_yp != MPI_PROC_NULL) {
        for (size_t z = 1; z <= lz; ++z) {
            const size_t zo = (z - 1) * lx;
            for (size_t x = 1; x <= lx; ++x) {
                a[idxL(x, ly + 1, z, lxp2, lyp2)] = b.ryp[zo + (x - 1)];
            }
        }
    } else {
        for (size_t z = 1; z <= lz; ++z) {
            for (size_t x = 1; x <= lx; ++x) {
                a[idxL(x, ly + 1, z, lxp2, lyp2)] = a[idxL(x, ly, z, lxp2, lyp2)];
            }
        }
    }

    // Unpack Z halos
    if (d.nbr_zm != MPI_PROC_NULL) {
        for (size_t y = 1; y <= ly; ++y) {
            const size_t yo = (y - 1) * lx;
            for (size_t x = 1; x <= lx; ++x) {
                a[idxL(x, y, 0, lxp2, lyp2)] = b.rzm[yo + (x - 1)];
            }
        }
    } else {
        for (size_t y = 1; y <= ly; ++y) {
            for (size_t x = 1; x <= lx; ++x) {
                a[idxL(x, y, 0, lxp2, lyp2)] = a[idxL(x, y, 1, lxp2, lyp2)];
            }
        }
    }

    if (d.nbr_zp != MPI_PROC_NULL) {
        for (size_t y = 1; y <= ly; ++y) {
            const size_t yo = (y - 1) * lx;
            for (size_t x = 1; x <= lx; ++x) {
                a[idxL(x, y, lz + 1, lxp2, lyp2)] = b.rzp[yo + (x - 1)];
            }
        }
    } else {
        for (size_t y = 1; y <= ly; ++y) {
            for (size_t x = 1; x <= lx; ++x) {
                a[idxL(x, y, lz + 1, lxp2, lyp2)] = a[idxL(x, y, lz, lxp2, lyp2)];
            }
        }
    }

    // Clamp edges/corners in ghosts where two/three directions are outside (only needed on global boundaries).
    // This keeps semantics identical to the original clamped stencil.
    if (d.nbr_xm == MPI_PROC_NULL) {
        for (size_t z = 0; z <= lz + 1; ++z) {
            for (size_t y = 0; y <= ly + 1; ++y) {
                a[idxL(0, y, z, lxp2, lyp2)] = a[idxL(1, y, z, lxp2, lyp2)];
            }
        }
    }
    if (d.nbr_xp == MPI_PROC_NULL) {
        for (size_t z = 0; z <= lz + 1; ++z) {
            for (size_t y = 0; y <= ly + 1; ++y) {
                a[idxL(lx + 1, y, z, lxp2, lyp2)] = a[idxL(lx, y, z, lxp2, lyp2)];
            }
        }
    }
    if (d.nbr_ym == MPI_PROC_NULL) {
        for (size_t z = 0; z <= lz + 1; ++z) {
            for (size_t x = 0; x <= lx + 1; ++x) {
                a[idxL(x, 0, z, lxp2, lyp2)] = a[idxL(x, 1, z, lxp2, lyp2)];
            }
        }
    }
    if (d.nbr_yp == MPI_PROC_NULL) {
        for (size_t z = 0; z <= lz + 1; ++z) {
            for (size_t x = 0; x <= lx + 1; ++x) {
                a[idxL(x, ly + 1, z, lxp2, lyp2)] = a[idxL(x, ly, z, lxp2, lyp2)];
            }
        }
    }
    if (d.nbr_zm == MPI_PROC_NULL) {
        for (size_t y = 0; y <= ly + 1; ++y) {
            for (size_t x = 0; x <= lx + 1; ++x) {
                a[idxL(x, y, 0, lxp2, lyp2)] = a[idxL(x, y, 1, lxp2, lyp2)];
            }
        }
    }
    if (d.nbr_zp == MPI_PROC_NULL) {
        for (size_t y = 0; y <= ly + 1; ++y) {
            for (size_t x = 0; x <= lx + 1; ++x) {
                a[idxL(x, y, lz + 1, lxp2, lyp2)] = a[idxL(x, y, lz, lxp2, lyp2)];
            }
        }
    }
}

static inline double laplacian_local(const double* a,
                                    const size_t x,
                                    const size_t y,
                                    const size_t z,
                                    const size_t lxp2,
                                    const size_t lyp2,
                                    const double invdx2,
                                    const double invdy2,
                                    const double invdz2) {
    const size_t c = idxL(x, y, z, lxp2, lyp2);
    const double v = a[c];
    const double cxx = (a[idxL(x + 1, y, z, lxp2, lyp2)] + a[idxL(x - 1, y, z, lxp2, lyp2)] - 2.0 * v) * invdx2;
    const double cyy = (a[idxL(x, y + 1, z, lxp2, lyp2)] + a[idxL(x, y - 1, z, lxp2, lyp2)] - 2.0 * v) * invdy2;
    const double czz = (a[idxL(x, y, z + 1, lxp2, lyp2)] + a[idxL(x, y, z - 1, lxp2, lyp2)] - 2.0 * v) * invdz2;
    return cxx + cyy + czz;
}

static inline void compute_mu_region(const double* cold,
                                    double* mu,
                                    const size_t lxp2,
                                    const size_t lyp2,
                                    const size_t x1,
                                    const size_t x2,
                                    const size_t y1,
                                    const size_t y2,
                                    const size_t z1,
                                    const size_t z2,
                                    const double invdx2,
                                    const double invdy2,
                                    const double invdz2,
                                    const double gamma,
                                    const double e_AA,
                                    const double e_BB,
                                    const double e_AB) {
    for (size_t z = z1; z <= z2; ++z) {
        for (size_t y = y1; y <= y2; ++y) {
            for (size_t x = x1; x <= x2; ++x) {
                const size_t i = idxL(x, y, z, lxp2, lyp2);
                const double cv = cold[i];
                mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv -
                        gamma * laplacian_local(cold, x, y, z, lxp2, lyp2, invdx2, invdy2, invdz2);
            }
        }
    }
}

static inline void update_region(double* cnew,
                                const double* cold,
                                const double* mu,
                                const size_t lxp2,
                                const size_t lyp2,
                                const size_t x1,
                                const size_t x2,
                                const size_t y1,
                                const size_t y2,
                                const size_t z1,
                                const size_t z2,
                                const double invdx2,
                                const double invdy2,
                                const double invdz2,
                                const double D,
                                const double dt) {
    const double dtD = dt * D;
    for (size_t z = z1; z <= z2; ++z) {
        for (size_t y = y1; y <= y2; ++y) {
            for (size_t x = x1; x <= x2; ++x) {
                const size_t i = idxL(x, y, z, lxp2, lyp2);
                cnew[i] = cold[i] + dtD * laplacian_local(mu, x, y, z, lxp2, lyp2, invdx2, invdy2, invdz2);
            }
        }
    }
}

static inline void initialize_concentration_local(double* cold,
                                                  const size_t lxp2,
                                                  const size_t lyp2,
                                                  const size_t nx,
                                                  const size_t ny,
                                                  const size_t nz,
                                                  const Decomp3D& d) {
    const size_t vol = nx * ny * nz;
    for (size_t z = 1; z <= d.lz; ++z) {
        const size_t gz = d.z0 + (z - 1);
        for (size_t y = 1; y <= d.ly; ++y) {
            const size_t gy = d.y0 + (y - 1);
            for (size_t x = 1; x <= d.lx; ++x) {
                const size_t gx = d.x0 + (x - 1);
                const size_t linear_id = gz * (nx * ny) + gy * nx + gx;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                cold[idxL(x, y, z, lxp2, lyp2)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

static bool validateResultMPI(const double* cold,
                             const size_t lxp2,
                             const size_t lyp2,
                             const Decomp3D& d) {
    int local_bad = 0;
    double local_min = std::numeric_limits<double>::infinity();
    double local_max = -std::numeric_limits<double>::infinity();

    for (size_t z = 1; z <= d.lz; ++z) {
        for (size_t y = 1; y <= d.ly; ++y) {
            for (size_t x = 1; x <= d.lx; ++x) {
                const double v = cold[idxL(x, y, z, lxp2, lyp2)];
                if (std::isnan(v) || std::isinf(v)) local_bad = 1;
                local_min = std::min(local_min, v);
                local_max = std::max(local_max, v);
            }
        }
    }

    int global_bad = 0;
    MPI_Allreduce(&local_bad, &global_bad, 1, MPI_INT, MPI_MAX, d.comm);

    double gmin = 0.0, gmax = 0.0;
    MPI_Allreduce(&local_min, &gmin, 1, MPI_DOUBLE, MPI_MIN, d.comm);
    MPI_Allreduce(&local_max, &gmax, 1, MPI_DOUBLE, MPI_MAX, d.comm);

    if (d.rank == 0) {
        if (global_bad) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        printf("Concentration range: [%.6f, %.6f]\n", gmin, gmax);
        if (gmax > 10.0 || gmin < -10.0) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    }

    int ok = (!global_bad) && (gmax <= 10.0) && (gmin >= -10.0);
    MPI_Bcast(&ok, 1, MPI_INT, 0, d.comm);
    return ok != 0;
}

static void gather_and_print_results(const double* cold,
                                    const size_t lxp2,
                                    const size_t lyp2,
                                    const size_t nx,
                                    const size_t ny,
                                    const size_t nz,
                                    const Decomp3D& d) {
    const int comm_size = d.size;

    // Pack local interior
    const int send_count = static_cast<int>(d.lx * d.ly * d.lz);
    std::vector<double> sendbuf(static_cast<size_t>(send_count));
    {
        size_t o = 0;
        for (size_t z = 1; z <= d.lz; ++z) {
            for (size_t y = 1; y <= d.ly; ++y) {
                for (size_t x = 1; x <= d.lx; ++x) {
                    sendbuf[o++] = cold[idxL(x, y, z, lxp2, lyp2)];
                }
            }
        }
    }

    // Gather subdomain descriptors: {x0,y0,z0,lx,ly,lz} as 6 unsigned long long (portable for size_t)
    unsigned long long local_desc[6] = {
        static_cast<unsigned long long>(d.x0),
        static_cast<unsigned long long>(d.y0),
        static_cast<unsigned long long>(d.z0),
        static_cast<unsigned long long>(d.lx),
        static_cast<unsigned long long>(d.ly),
        static_cast<unsigned long long>(d.lz),
    };

    std::vector<unsigned long long> all_desc;
    unsigned long long* all_desc_ptr = nullptr;
    if (d.rank == 0) {
        all_desc.resize(static_cast<size_t>(6 * comm_size));
        all_desc_ptr = all_desc.data();
    }

    MPI_Gather(local_desc, 6, MPI_UNSIGNED_LONG_LONG,
               all_desc_ptr, 6, MPI_UNSIGNED_LONG_LONG,
               0, d.comm);

    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<double> recvbuf;

    if (d.rank == 0) {
        counts.resize(static_cast<size_t>(comm_size));
        displs.resize(static_cast<size_t>(comm_size));
        int disp = 0;
        for (int r = 0; r < comm_size; ++r) {
            const size_t lx = static_cast<size_t>(all_desc[static_cast<size_t>(6 * r + 3)]);
            const size_t ly = static_cast<size_t>(all_desc[static_cast<size_t>(6 * r + 4)]);
            const size_t lz = static_cast<size_t>(all_desc[static_cast<size_t>(6 * r + 5)]);
            const size_t c = lx * ly * lz;
            counts[static_cast<size_t>(r)] = static_cast<int>(c);
            displs[static_cast<size_t>(r)] = disp;
            disp += counts[static_cast<size_t>(r)];
        }
        recvbuf.resize(static_cast<size_t>(disp));
    }

    double* recv_ptr = (d.rank == 0) ? recvbuf.data() : nullptr;
    int* counts_ptr = (d.rank == 0) ? counts.data() : nullptr;
    int* displs_ptr = (d.rank == 0) ? displs.data() : nullptr;

    MPI_Gatherv(sendbuf.data(), send_count, MPI_DOUBLE,
                recv_ptr, counts_ptr, displs_ptr, MPI_DOUBLE,
                0, d.comm);

    if (d.rank == 0) {
        std::vector<double> global(nx * ny * nz);
        for (int r = 0; r < comm_size; ++r) {
            const size_t x0 = static_cast<size_t>(all_desc[static_cast<size_t>(6 * r + 0)]);
            const size_t y0 = static_cast<size_t>(all_desc[static_cast<size_t>(6 * r + 1)]);
            const size_t z0 = static_cast<size_t>(all_desc[static_cast<size_t>(6 * r + 2)]);
            const size_t lx = static_cast<size_t>(all_desc[static_cast<size_t>(6 * r + 3)]);
            const size_t ly = static_cast<size_t>(all_desc[static_cast<size_t>(6 * r + 4)]);
            const size_t lz = static_cast<size_t>(all_desc[static_cast<size_t>(6 * r + 5)]);

            const double* src = recvbuf.data() + displs[static_cast<size_t>(r)];
            size_t o = 0;
            for (size_t zz = 0; zz < lz; ++zz) {
                for (size_t yy = 0; yy < ly; ++yy) {
                    for (size_t xx = 0; xx < lx; ++xx) {
                        global[idx3(x0 + xx, y0 + yy, z0 + zz, nx, ny)] = src[o++];
                    }
                }
            }
        }
        print_results(global, "Concentration");
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    Decomp3D d;
    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.size);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (d.rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (d.rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // Create 3D Cartesian decomposition
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(d.size, 3, dims);
    d.dims[0] = dims[0];
    d.dims[1] = dims[1];
    d.dims[2] = dims[2];

    int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, d.dims, periods, 1, &d.comm);
    if (d.comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 1;
    }

    MPI_Comm_rank(d.comm, &d.rank);
    MPI_Comm_size(d.comm, &d.size);
    MPI_Cart_coords(d.comm, d.rank, 3, d.coords);

    MPI_Cart_shift(d.comm, 0, 1, &d.nbr_xm, &d.nbr_xp);
    MPI_Cart_shift(d.comm, 1, 1, &d.nbr_ym, &d.nbr_yp);
    MPI_Cart_shift(d.comm, 2, 1, &d.nbr_zm, &d.nbr_zp);

    decompose1d(nx, d.dims[0], d.coords[0], d.x0, d.lx);
    decompose1d(ny, d.dims[1], d.coords[1], d.y0, d.ly);
    decompose1d(nz, d.dims[2], d.coords[2], d.z0, d.lz);

    if (d.lx == 0 || d.ly == 0 || d.lz == 0) {
        if (d.rank == 0) printf("Error: MPI decomposition produced empty subdomains; use fewer ranks or larger grid.\n");
        MPI_Abort(d.comm, 1);
    }

    if (d.rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (decomp %d x %d x %d)\n", d.size, d.dims[0], d.dims[1], d.dims[2]);
    }

    // Physical parameters (same as original)
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);

    const size_t lxp2 = d.lx + 2;
    const size_t lyp2 = d.ly + 2;
    const size_t lzp2 = d.lz + 2;
    const size_t localSize = lxp2 * lyp2 * lzp2;

    std::vector<double> cold(localSize);
    std::vector<double> cnew(localSize);
    std::vector<double> mu(localSize);

    if (d.rank == 0) printf("Initializing concentration field...\n");
    initialize_concentration_local(cold.data(), lxp2, lyp2, nx, ny, nz, d);

    HaloBuffers bufs(d.lx, d.ly, d.lz);

    const size_t max_cnt = static_cast<size_t>(std::numeric_limits<int>::max());
    if (d.ly * d.lz > max_cnt || d.lx * d.lz > max_cnt || d.lx * d.ly > max_cnt) {
        if (d.rank == 0) printf("Error: halo face size exceeds MPI INT count limit.\n");
        MPI_Abort(d.comm, 1);
    }

    if (d.rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(d.comm);
    const double t0 = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Exchange halos for cold
        Exchange ex_c = start_halo_exchange(cold.data(), d, lxp2, lyp2, d.lx, d.ly, d.lz, bufs);

        // Deep interior (can be computed before halos arrive)
        if (d.lx >= 3 && d.ly >= 3 && d.lz >= 3) {
            compute_mu_region(cold.data(), mu.data(), lxp2, lyp2,
                              2, d.lx - 1, 2, d.ly - 1, 2, d.lz - 1,
                              invdx2, invdy2, invdz2, gamma, e_AA, e_BB, e_AB);
        }

        finish_halo_exchange(cold.data(), d, lxp2, lyp2, d.lx, d.ly, d.lz, bufs, ex_c);

        // Boundary layers (need halos)
        for (size_t z = 1; z <= d.lz; ++z) {
            for (size_t y = 1; y <= d.ly; ++y) {
                for (size_t x = 1; x <= d.lx; ++x) {
                    if (d.lx >= 3 && d.ly >= 3 && d.lz >= 3 && x > 1 && x < d.lx && y > 1 && y < d.ly && z > 1 && z < d.lz) continue;
                    const size_t i = idxL(x, y, z, lxp2, lyp2);
                    const double cv = cold[i];
                    mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv -
                            gamma * laplacian_local(cold.data(), x, y, z, lxp2, lyp2, invdx2, invdy2, invdz2);
                }
            }
        }

        // Exchange halos for mu
        Exchange ex_m = start_halo_exchange(mu.data(), d, lxp2, lyp2, d.lx, d.ly, d.lz, bufs);

        // Deep interior update
        if (d.lx >= 3 && d.ly >= 3 && d.lz >= 3) {
            update_region(cnew.data(), cold.data(), mu.data(), lxp2, lyp2,
                          2, d.lx - 1, 2, d.ly - 1, 2, d.lz - 1,
                          invdx2, invdy2, invdz2, D, dt);
        }

        finish_halo_exchange(mu.data(), d, lxp2, lyp2, d.lx, d.ly, d.lz, bufs, ex_m);

        // Boundary layers update
        for (size_t z = 1; z <= d.lz; ++z) {
            for (size_t y = 1; y <= d.ly; ++y) {
                for (size_t x = 1; x <= d.lx; ++x) {
                    if (d.lx >= 3 && d.ly >= 3 && d.lz >= 3 && x > 1 && x < d.lx && y > 1 && y < d.ly && z > 1 && z < d.lz) continue;
                    const size_t i = idxL(x, y, z, lxp2, lyp2);
                    cnew[i] = cold[i] + (dt * D) * laplacian_local(mu.data(), x, y, z, lxp2, lyp2, invdx2, invdy2, invdz2);
                }
            }
        }

        std::swap(cold, cnew);
    }

    MPI_Barrier(d.comm);
    const double t1 = MPI_Wtime();
    const double local_time = t1 - t0;

    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, d.comm);

    if (d.rank == 0) {
        const double time_s = (max_time > 0.0) ? max_time : 1e-12;
        const double cellUpdates = static_cast<double>(nx) * static_cast<double>(ny) * static_cast<double>(nz) * static_cast<double>(iterations);
        const double mcups = cellUpdates / time_s / 1e6;
        printf("Computation time: %.3f ms\n", time_s * 1e3);
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        gather_and_print_results(cold.data(), lxp2, lyp2, nx, ny, nz, d);
    }

    if (validate) {
        if (d.rank == 0) printf("Validating result...\n");
        const bool ok = validateResultMPI(cold.data(), lxp2, lyp2, d);
        if (d.rank == 0) printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
        MPI_Comm_free(&d.comm);
        MPI_Finalize();
        return ok ? 0 : 1;
    }

    MPI_Comm_free(&d.comm);
    MPI_Finalize();
    return 0;
}
