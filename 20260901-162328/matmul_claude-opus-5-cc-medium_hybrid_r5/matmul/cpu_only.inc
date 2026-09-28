static constexpr size_t KC = 256; // K block
static constexpr size_t NC = 256; // N block
static constexpr size_t MR = 6;   // micro kernel rows
static constexpr size_t NR = 8;   // micro kernel columns (2 AVX2 vectors)

// C[mr x NR] += A[mr x kc] * B[kc x NR]
template <size_t MRT>
static inline void microKernel(const double* __restrict__ A, const double* __restrict__ B,
                               double* __restrict__ C, const size_t ld, const size_t kc) {
    double c[MRT][NR];
    for (size_t i = 0; i < MRT; ++i) {
        for (size_t j = 0; j < NR; ++j) {
            c[i][j] = C[i * ld + j];
        }
    }
    for (size_t k = 0; k < kc; ++k) {
        const double* __restrict__ b = B + k * ld;
        for (size_t i = 0; i < MRT; ++i) {
            const double a = A[i * ld + k];
            for (size_t j = 0; j < NR; ++j) {
                c[i][j] += a * b[j];
            }
        }
    }
    for (size_t i = 0; i < MRT; ++i) {
        for (size_t j = 0; j < NR; ++j) {
            C[i * ld + j] = c[i][j];
        }
    }
}

// Scalar edge handling for the column remainder (N % NR != 0)
static inline void edgeKernel(const double* __restrict__ A, const double* __restrict__ B,
                              double* __restrict__ C, const size_t ld, const size_t kc,
                              const size_t mr, const size_t nr) {
    for (size_t i = 0; i < mr; ++i) {
        for (size_t k = 0; k < kc; ++k) {
            const double a = A[i * ld + k];
            const double* __restrict__ b = B + k * ld;
            for (size_t j = 0; j < nr; ++j) {
                C[i * ld + j] += a * b[j];
            }
        }
    }
}

// C[rows x N] = A[rows x N] * B[N x N], A/C point at the first row of the panel.
static void cpuPanel(const double* __restrict__ A, const double* __restrict__ B,
                     double* __restrict__ C, const size_t N, const size_t rows) {
    for (size_t i = 0; i < rows; ++i) {
        memset(C + i * N, 0, N * sizeof(double));
    }

    for (size_t n0 = 0; n0 < N; n0 += NC) {
        const size_t nc = std::min(NC, N - n0);
        for (size_t k0 = 0; k0 < N; k0 += KC) {
            const size_t kc = std::min(KC, N - k0);
            for (size_t i0 = 0; i0 < rows; i0 += MR) {
                const size_t mr = std::min(MR, rows - i0);
                const double* Ap = A + i0 * N + k0;
                const double* Bp = B + k0 * N + n0;
                double* Cp = C + i0 * N + n0;
                size_t j = 0;
                for (; j + NR <= nc; j += NR) {
                    if (mr == MR) {
                        microKernel<MR>(Ap, Bp + j, Cp + j, N, kc);
                    } else {
                        for (size_t i = 0; i < mr; ++i) {
                            microKernel<1>(Ap + i * N, Bp + j, Cp + i * N + j, N, kc);
                        }
                    }
                }
                if (j < nc) {
                    edgeKernel(Ap, Bp + j, Cp + j, N, kc, mr, nc - j);
                }
            }
        }
    }
}

