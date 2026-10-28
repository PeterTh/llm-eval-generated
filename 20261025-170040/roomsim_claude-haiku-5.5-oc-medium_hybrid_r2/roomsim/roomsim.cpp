/**
 * Room Response Simulation Benchmark (hybrid MPI + OpenMP + CUDA)
 * 
 * This is a simplified implementation of room impulse response simulation
 * using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization:
 *   - MPI:    the N x N form-factor / time-delay matrices and the radiosity
 *             update are partitioned by receiver row across ranks. Every rank
 *             keeps the full radiosity history, which is re-synchronized with
 *             MPI_Allgatherv once per timestep.
 *   - OpenMP: host-side work (time delays, visibility sample counts, random
 *             sample generation, distance correlation) runs on OpenMP threads.
 *             Each thread owns a contiguous block of rows and a Mersenne
 *             Twister stream positioned exactly at that block's offset in the
 *             original single-stream sequence (see Mt19937Jump).
 *   - CUDA:   ray/octree visibility tests and form-factor accumulation run on
 *             the GPU (one thread per visible-candidate pair), and the wave
 *             propagation runs as one kernel per timestep.
 *
 * The random number stream, the floating-point evaluation order and the
 * output format are preserved from the sequential formulation, so results are
 * identical to the original program.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

constexpr size_t kStagingBudgetBytes = size_t(128) << 20;  // Host staging budget per pipeline slot
constexpr int kBlock = 128;               // CUDA threads per block
constexpr int kTraversalStack = 64;       // Octree traversal stack capacity on the device (needs 7*depth+2)

// ============================================================================
// Error handling
// ============================================================================

[[noreturn]] void fatal(const char* what) {
    fprintf(stderr, "roomsim fatal: %s\n", what);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error '%s' at %s:%d: %s\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__, #call);           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                           \
        }                                                                           \
    } while (0)

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    constexpr Vec3() : x(0), y(0), z(0) {}
    constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }

    val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    val_t squaredNorm() const { return x * x + y * y + z * z; }
    val_t norm() const { return std::sqrt(squaredNorm()); }
    Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    Vec3 center() const { return (a + b + c) / 3.0f; }
    Vec3 normal() const { return _normal; }

    val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction)
// ============================================================================

// Separating axis test for triangle-box overlap
bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
    // Translate triangle to box center
    Vec3 v0 = tri.a - boxCenter;
    Vec3 v1 = tri.b - boxCenter;
    Vec3 v2 = tri.c - boxCenter;

    // Triangle edges
    Vec3 e0 = v1 - v0;
    Vec3 e1 = v2 - v1;
    Vec3 e2 = v0 - v2;

    // Test box normals (AABB axes)
    auto minMax3 = [](val_t a, val_t b, val_t c) {
        return std::make_pair(std::min({a, b, c}), std::max({a, b, c}));
    };

    auto [minX, maxX] = minMax3(v0.x, v1.x, v2.x);
    if (minX > boxHalfSize.x || maxX < -boxHalfSize.x) return false;

    auto [minY, maxY] = minMax3(v0.y, v1.y, v2.y);
    if (minY > boxHalfSize.y || maxY < -boxHalfSize.y) return false;

    auto [minZ, maxZ] = minMax3(v0.z, v1.z, v2.z);
    if (minZ > boxHalfSize.z || maxZ < -boxHalfSize.z) return false;

    // Test triangle normal
    Vec3 triNormal = e0.cross(e1);
    val_t d = triNormal.dot(v0);
    val_t r = boxHalfSize.x * std::abs(triNormal.x) +
              boxHalfSize.y * std::abs(triNormal.y) +
              boxHalfSize.z * std::abs(triNormal.z);
    if (std::abs(d) > r) return false;

    // Test 9 edge cross products
    auto testAxis = [&](const Vec3& axis) {
        val_t p0 = axis.dot(v0);
        val_t p1 = axis.dot(v1);
        val_t p2 = axis.dot(v2);
        val_t r = boxHalfSize.x * std::abs(axis.x) +
                  boxHalfSize.y * std::abs(axis.y) +
                  boxHalfSize.z * std::abs(axis.z);
        auto [minP, maxP] = minMax3(p0, p1, p2);
        return !(minP > r || maxP < -r);
    };

    Vec3 axes[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
    Vec3 edges[3] = {e0, e1, e2};
    for (const auto& axis : axes) {
        for (const auto& edge : edges) {
            Vec3 crossAxis = axis.cross(edge);
            if (crossAxis.squaredNorm() > EPSILON) {
                if (!testAxis(crossAxis)) return false;
            }
        }
    }

    return true;
}

// ============================================================================
// Octree for Spatial Acceleration (built on the host, traversed on the GPU)
// ============================================================================

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;

class Octree {
public:
    Vec3 minBound, maxBound;
    Vec3 halfExtent, center;
    std::unique_ptr<Octree> children[8];
    std::vector<size_t> triangleIndices;  // Indices into the global triangle list
    const std::vector<Triangle>* allTriangles;  // Pointer to all triangles

    Octree() : allTriangles(nullptr) {}

    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if (triangles.empty()) return;

        // Compute bounding box
        minBound = triangles[0].a;
        maxBound = triangles[0].a;
        for (const auto& tri : triangles) {
            for (const auto* v : {&tri.a, &tri.b, &tri.c}) {
                minBound.x = std::min(minBound.x, v->x);
                minBound.y = std::min(minBound.y, v->y);
                minBound.z = std::min(minBound.z, v->z);
                maxBound.x = std::max(maxBound.x, v->x);
                maxBound.y = std::max(maxBound.y, v->y);
                maxBound.z = std::max(maxBound.z, v->z);
            }
        }

        // Collect all indices
        std::vector<size_t> allIndices(triangles.size());
        for (size_t i = 0; i < triangles.size(); ++i) allIndices[i] = i;

        buildNode(allIndices, minBound, maxBound);
    }

private:
    void buildNode(const std::vector<size_t>& indices, const Vec3& nodeMin, const Vec3& nodeMax) {
        minBound = nodeMin;
        maxBound = nodeMax;
        halfExtent = (maxBound - minBound) * 0.5f;
        center = (minBound + maxBound) * 0.5f;

        // If few enough triangles or too small, make this a leaf
        if (indices.size() <= MAX_OCTREE_TRIS ||
            (maxBound - minBound).norm() < MAX_OCTREE_LEAF_SIZE) {
            triangleIndices = indices;
            return;
        }

        // Subdivide into 8 children
        Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];

        for (size_t idx : indices) {
            const Triangle& tri = (*allTriangles)[idx];

            // Check which children this triangle overlaps
            for (int i = 0; i < 8; ++i) {
                Vec3 childCenter = center;
                childCenter.x += (i & 1) ? childHalfSize.x : -childHalfSize.x;
                childCenter.y += (i & 2) ? childHalfSize.y : -childHalfSize.y;
                childCenter.z += (i & 4) ? childHalfSize.z : -childHalfSize.z;

                if (triangleBoxOverlap(childCenter, childHalfSize, tri)) {
                    childIndices[i].push_back(idx);
                }
            }
        }

        // Check if we can't split further (all triangles in one child)
        bool canSplit = false;
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty() && childIndices[i].size() < indices.size()) {
                canSplit = true;
                break;
            }
        }

        if (!canSplit) {
            triangleIndices = indices;
            return;
        }

        // Build children
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty()) {
                Vec3 childMin = center;
                Vec3 childMax = center;
                childMin.x = (i & 1) ? center.x : minBound.x;
                childMax.x = (i & 1) ? maxBound.x : center.x;
                childMin.y = (i & 2) ? center.y : minBound.y;
                childMax.y = (i & 2) ? maxBound.y : center.y;
                childMin.z = (i & 4) ? center.z : minBound.z;
                childMax.z = (i & 4) ? maxBound.z : center.z;

                children[i] = std::make_unique<Octree>();
                children[i]->allTriangles = allTriangles;
                children[i]->buildNode(childIndices[i], childMin, childMax);
            }
        }
    }
};

// ============================================================================
// Mesh Generation: Icosphere
// ============================================================================

// Generate an icosphere by subdividing an icosahedron
class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
        // Initial icosahedron vertices
        const val_t t = (1.0f + std::sqrt(5.0f)) / 2.0f;

        std::vector<Vec3> vertices = {
            Vec3(-1,  t,  0).normalized() * radius,
            Vec3( 1,  t,  0).normalized() * radius,
            Vec3(-1, -t,  0).normalized() * radius,
            Vec3( 1, -t,  0).normalized() * radius,
            Vec3( 0, -1,  t).normalized() * radius,
            Vec3( 0,  1,  t).normalized() * radius,
            Vec3( 0, -1, -t).normalized() * radius,
            Vec3( 0,  1, -t).normalized() * radius,
            Vec3( t,  0, -1).normalized() * radius,
            Vec3( t,  0,  1).normalized() * radius,
            Vec3(-t,  0, -1).normalized() * radius,
            Vec3(-t,  0,  1).normalized() * radius
        };

        // Initial icosahedron faces (20 triangles)
        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

        // Subdivide
        for (int i = 0; i < subdivisions; ++i) {
            std::vector<std::array<idx_t, 3>> newFaces;
            std::map<std::pair<idx_t, idx_t>, idx_t> midpointCache;

            auto getMidpoint = [&](idx_t i1, idx_t i2) -> idx_t {
                auto key = std::make_pair(std::min(i1, i2), std::max(i1, i2));
                auto it = midpointCache.find(key);
                if (it != midpointCache.end()) return it->second;

                Vec3 mid = (vertices[i1] + vertices[i2]) / 2.0f;
                mid = mid.normalized() * radius;
                idx_t idx = static_cast<idx_t>(vertices.size());
                vertices.push_back(mid);
                midpointCache[key] = idx;
                return idx;
            };

            for (const auto& face : faces) {
                idx_t a = getMidpoint(face[0], face[1]);
                idx_t b = getMidpoint(face[1], face[2]);
                idx_t c = getMidpoint(face[2], face[0]);

                newFaces.push_back({face[0], a, c});
                newFaces.push_back({face[1], b, a});
                newFaces.push_back({face[2], c, b});
                newFaces.push_back({a, b, c});
            }
            faces = std::move(newFaces);
        }

        // Build triangles (normals pointing inward for a "room")
        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            // Reverse winding to make normals point inward
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Random Number Stream Positioning (jump-ahead for MT19937)
// ============================================================================
//
// The sequential formulation draws every random number from one
// std::mt19937(42) stream, consumed in (receiver i, emitter j, ray) order.
// To distribute rows over ranks and threads while reproducing that exact
// stream, each worker is placed at its starting offset with a jump-ahead:
// the MT19937 output sequence is a linear recurrence over GF(2) whose
// characteristic polynomial Q (degree 19937) is recovered at run time with
// Berlekamp-Massey. Then x^k mod Q gives the output window at offset k as a
// linear combination of the first outputs, from which the generator state
// is reconstructed exactly.
class Mt19937Jump {
public:
    Mt19937Jump() {
        constexpr size_t N = 41024;  // Must exceed 2 * degree + 624 (degree = 19937)
        O_.resize(N);
        {
            std::mt19937 e(42);
            for (auto& o : O_) o = e();
        }

        // Berlekamp-Massey over GF(2) on bit 0 of the output sequence.
        // R is the reversed bit sequence so the discrepancy loop is contiguous.
        std::vector<uint8_t> s(N), R(N);
        for (size_t t = 0; t < N; ++t) s[t] = static_cast<uint8_t>(O_[t] & 1u);
        for (size_t k = 0; k < N; ++k) R[k] = s[N - 1 - k];

        std::vector<uint8_t> C(N + 1, 0), B(N + 1, 0), Tmp;
        C[0] = 1;
        B[0] = 1;
        size_t L = 0;
        long m = -1;
        for (size_t n = 0; n < N; ++n) {
            const uint8_t* rb = R.data() + (N - 1 - n);  // rb[i] == s[n - i]
            uint8_t d = s[n];
            for (size_t i = 1; i <= L; ++i) d ^= C[i] & rb[i];
            if (d) {
                const bool grow = 2 * L <= n;
                if (grow) Tmp = C;
                const size_t sh = static_cast<size_t>(static_cast<long>(n) - m);
                for (size_t i = 0; i + sh <= N; ++i) C[i + sh] ^= B[i];
                if (grow) {
                    L = n + 1 - L;
                    B.swap(Tmp);
                    m = static_cast<long>(n);
                }
            }
        }
        if (2 * L > N || L + 624 > N) fatal("MT19937 jump: recurrence degree out of range");

        L_ = static_cast<int>(L);
        W_ = static_cast<int>((L + 1 + 63) / 64 + 1);
        Qlow_.assign(static_cast<size_t>(W_), 0);
        // x^L == sum_{j<L} Qlow_j x^j (mod Q)
        for (size_t j = 0; j < L; ++j)
            if (C[L - j]) Qlow_[j >> 6] |= uint64_t(1) << (j & 63);
    }

    // Returns an engine whose next output is output #k of the mt19937(42) stream.
    std::mt19937 engineAt(uint64_t k) const {
        std::vector<uint64_t> r(static_cast<size_t>(W_), 0);
        r[0] = 1;  // x^0
        int top = 63;
        while (top > 0 && !((k >> top) & 1u)) --top;
        for (int b = top; b >= 0; --b) {
            r = mulmod(r, r);
            if ((k >> b) & 1u) r = mulx(r);
        }
        // Output window at offset k: sum over set coefficients a_i of window(i).
        std::vector<uint32_t> win(624, 0);
        for (int i = 0; i < L_; ++i) {
            if ((r[i >> 6] >> (i & 63)) & 1u) {
                for (int w = 0; w < 624; ++w) win[w] ^= O_[static_cast<size_t>(i) + w];
            }
        }
        // Reconstruct the generator state (untempered window, read index 0).
        std::ostringstream os;
        for (int w = 0; w < 624; ++w) os << untemper(win[w]) << ' ';
        os << 0;
        std::istringstream is(os.str());
        std::mt19937 engine;
        is >> engine;
        return engine;
    }

    // Number of engine outputs consumed by one uniform_real_distribution<val_t>(0,1) draw.
    static int outputsPerDraw() {
        std::mt19937 a(42), b(42);
        std::uniform_real_distribution<val_t> dist(0.0f, 1.0f);
        dist(a);
        for (int c = 1; c <= 8; ++c) {
            b();
            if (a == b) return c;
        }
        fatal("could not determine RNG outputs per draw");
    }

private:
    static uint32_t untemper(uint32_t y) {
        y ^= y >> 18;
        y ^= (y << 15) & 0xefc60000u;
        uint32_t x = y;
        for (int k = 0; k < 5; ++k) x = y ^ ((x << 7) & 0x9d2c5680u);
        y = x;
        y ^= y >> 11;
        y ^= y >> 22;
        return y;
    }

    // dst ^= src << shift (bit-level), over the given word counts
    static void xorShift(uint64_t* dst, int dstWords, const uint64_t* src, int srcWords, int shift) {
        const int ws = shift >> 6, bs = shift & 63;
        for (int k = 0; k < srcWords; ++k) {
            const uint64_t v = src[k];
            if (!v) continue;
            const int di = k + ws;
            if (di < dstWords) dst[di] ^= v << bs;
            if (bs && di + 1 < dstWords) dst[di + 1] ^= v >> (64 - bs);
        }
    }

    // (a * b) mod Q, with a and b of degree < L
    std::vector<uint64_t> mulmod(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) const {
        const int pw = 2 * W_ + 2;
        std::vector<uint64_t> prod(static_cast<size_t>(pw), 0);
        for (int i = 0; i < L_; ++i) {
            if ((a[i >> 6] >> (i & 63)) & 1u) xorShift(prod.data(), pw, b.data(), W_, i);
        }
        for (int t = 2 * L_ - 2; t >= L_; --t) {
            if ((prod[t >> 6] >> (t & 63)) & 1u) {
                prod[t >> 6] ^= uint64_t(1) << (t & 63);
                xorShift(prod.data(), pw, Qlow_.data(), W_, t - L_);
            }
        }
        prod.resize(static_cast<size_t>(W_));
        return prod;
    }

    // (r * x) mod Q, with r of degree < L
    std::vector<uint64_t> mulx(std::vector<uint64_t> r) const {
        uint64_t carry = 0;
        for (auto& w : r) {
            const uint64_t nc = w >> 63;
            w = (w << 1) | carry;
            carry = nc;
        }
        if ((r[L_ >> 6] >> (L_ & 63)) & 1u) {
            r[L_ >> 6] ^= uint64_t(1) << (L_ & 63);
            for (int k = 0; k < W_; ++k) r[k] ^= Qlow_[k];
        }
        return r;
    }

    std::vector<uint32_t> O_;     // First outputs of mt19937(42)
    std::vector<uint64_t> Qlow_;  // Reduction polynomial, low part
    int L_ = 0;                   // Degree of the recurrence
    int W_ = 0;                   // Words per polynomial
};

// ============================================================================
// Device data layout
// ============================================================================

struct DevVec {
    float x, y, z;
};

struct DevTri {
    DevVec a, b, c, n;  // n: unit normal (precomputed on the host, identical bits)
};

struct DevNode {
    DevVec center, half;
    int32_t child[8];          // -1 when absent
    uint32_t leafBegin;        // Range into the flattened leaf index list
    uint32_t leafCount;        // > 0 marks a leaf node
};

struct DevPair {
    uint32_t i;       // Receiver triangle
    uint32_t j;       // Emitter triangle
    uint32_t drawOff; // Offset of this pair's 64 random samples in the draw buffer
};

// ============================================================================
// Device geometry helpers (same operation order as the host Vec3 code)
// ============================================================================

__device__ __forceinline__ DevVec vadd(DevVec a, DevVec b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ __forceinline__ DevVec vsub(DevVec a, DevVec b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ __forceinline__ DevVec vscale(DevVec a, float s) { return {a.x * s, a.y * s, a.z * s}; }
__device__ __forceinline__ DevVec vdivs(DevVec a, float s) { return {a.x / s, a.y / s, a.z / s}; }
__device__ __forceinline__ DevVec vneg(DevVec a) { return {-a.x, -a.y, -a.z}; }
__device__ __forceinline__ float vdot(DevVec a, DevVec b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ DevVec vcross(DevVec a, DevVec b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__device__ __forceinline__ float vsqnorm(DevVec a) { return a.x * a.x + a.y * a.y + a.z * a.z; }
__device__ __forceinline__ float vnorm(DevVec a) { return sqrtf(vsqnorm(a)); }

// Point from barycentric sample (u, v) inside a triangle, as in randomPointInTriangle
__device__ __forceinline__ DevVec samplePoint(const DevTri& t, float u, float v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    DevVec ab = vsub(t.b, t.a);
    DevVec ac = vsub(t.c, t.a);
    return vadd(vadd(t.a, vscale(ab, u)), vscale(ac, v));
}

// Mirrors Octree::rayIntersectsBox
__device__ __forceinline__ bool boxHit(const DevNode& nd, DevVec p1, DevVec p2) {
    const float EPS = 1e-6f;
    DevVec d = vscale(vsub(p2, p1), 0.5f);
    DevVec c = vsub(vadd(p1, d), nd.center);
    DevVec ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > nd.half.x + ad.x) return false;
    if (fabsf(c.y) > nd.half.y + ad.y) return false;
    if (fabsf(c.z) > nd.half.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > nd.half.y * ad.z + nd.half.z * ad.y + EPS) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > nd.half.z * ad.x + nd.half.x * ad.z + EPS) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > nd.half.x * ad.y + nd.half.y * ad.x + EPS) return false;
    return true;
}

// Mirrors rayTriangleIntersect (Moller-Trumbore); returns a large value on miss
__device__ __forceinline__ float rayTriIntersect(DevVec orig, DevVec dir, DevVec v0, DevVec v1, DevVec v2) {
    const float EPS = 1e-6f;
    DevVec e1 = vsub(v1, v0);
    DevVec e2 = vsub(v2, v0);
    DevVec pvec = vcross(dir, e2);
    float det = vdot(e1, pvec);

    if (fabsf(det) < EPS) return 3.402823466e38f;

    float invDet = 1.0f / det;
    DevVec tvec = vsub(orig, v0);
    float u = vdot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 3.402823466e38f;

    DevVec qvec = vcross(tvec, e1);
    float v = vdot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 3.402823466e38f;

    return vdot(e2, qvec) * invDet;
}

// Mirrors isRayBlocked + Octree::applyToTris (true if any non-endpoint triangle blocks)
__device__ bool rayBlocked(DevVec from, DevVec to, uint32_t srcIdx, uint32_t dstIdx,
                           const DevTri* tris, const DevNode* nodes, const uint32_t* leaf) {
    const float EPS = 1e-6f;
    DevVec dir = vsub(to, from);
    float rayLen = vnorm(dir);
    if (rayLen < EPS) return true;
    DevVec dirN = vdivs(dir, rayLen);

    int stack[kTraversalStack];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const DevNode& nd = nodes[stack[--sp]];
        if (nd.leafCount > 0) {
            for (uint32_t k = 0; k < nd.leafCount; ++k) {
                const uint32_t idx = leaf[nd.leafBegin + k];
                if (idx == srcIdx || idx == dstIdx) continue;
                const DevTri& tr = tris[idx];
                float dist = rayTriIntersect(from, dirN, tr.a, tr.b, tr.c);
                if (dist > EPS && dist < rayLen - EPS) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                const int32_t ch = nd.child[c];
                if (ch >= 0 && boxHit(nodes[ch], from, to)) stack[sp++] = ch;
            }
        }
    }
    return false;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// Form factor Kij for each (receiver i, emitter j) pair in the list.
// Accumulates NUM_RAYS samples in the same order as the sequential computeKij.
// Output layout: kijT[j * L + (i - r0)] (emitter-major for coalesced access).
__global__ void formFactorKernel(const DevPair* pairs, uint32_t numPairs, const float* draws,
                                 const DevTri* tris, const DevNode* nodes, const uint32_t* leaf,
                                 uint32_t r0, uint32_t L, float* kijT, unsigned long long* nzCount) {
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= numPairs) return;

    const DevPair pr = pairs[p];
    const float* d = draws + pr.drawOff;
    const DevTri& ti = tris[pr.i];
    const DevTri& tj = tris[pr.j];

    float kij = 0.0f;
    for (int r = 0; r < NUM_RAYS; ++r) {
        DevVec pI = samplePoint(ti, d[4 * r + 0], d[4 * r + 1]);
        DevVec pJ = samplePoint(tj, d[4 * r + 2], d[4 * r + 3]);

        if (rayBlocked(pI, pJ, pr.i, pr.j, tris, nodes, leaf)) continue;

        DevVec v = vsub(pJ, pI);
        float distSqr = vsqnorm(v);
        if (distSqr < 1e-6f) continue;

        // cosPhi(v, normalI) and cosPhi(-v, normalJ), clamped at zero
        float vNormI = vnorm(v);
        float cosI = 0.0f;
        if (vNormI > 1e-6f) {
            float t = vdot(v, ti.n) / vNormI;
            cosI = (0.0f < t) ? t : 0.0f;
        }
        DevVec nv = vneg(v);
        float vNormJ = vnorm(nv);
        float cosJ = 0.0f;
        if (vNormJ > 1e-6f) {
            float t = vdot(nv, tj.n) / vNormJ;
            cosJ = (0.0f < t) ? t : 0.0f;
        }

        if (cosI <= 0.0f || cosJ <= 0.0f) continue;

        kij += (cosI * cosJ) / (PI * distSqr);
    }

    kij = kij * INV_NUM_RAYS;
    kijT[static_cast<size_t>(pr.j) * L + (pr.i - r0)] = kij;
    if (kij > EPSILON) atomicAdd(nzCount, 1ULL);
}

// Radiosity update for timestep t, for the receivers owned by this rank.
// Each thread accumulates over emitters j in ascending order (as in the original).
__global__ void radiosityStepKernel(int t, int n, int L, int r0, int sourceIdx, int timeOff, float rho,
                                    const float* kijT, const int* tauT, const float* areas,
                                    float* radB) {
    const int il = blockIdx.x * blockDim.x + threadIdx.x;
    if (il >= L) return;
    const int i = r0 + il;

    float sumB = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;

        const int tau = tauT[static_cast<size_t>(j) * L + il];
        // Skip if wave hasn't yet propagated from j to i
        if (t < tau) continue;

        const float kij = kijT[static_cast<size_t>(j) * L + il];
        if (kij <= 0.0f) continue;

        const float radJ = radB[static_cast<size_t>(t - tau) * n + j];
        if (radJ <= 0.0f) continue;

        const float p = kij * areas[j];
        const float m = (ONE < p) ? ONE : p;
        sumB += m * radJ;
    }

    const float emit = (t < timeOff && i == sourceIdx) ? 1.0f : 0.0f;
    radB[static_cast<size_t>(t) * n + i] = rho * sumB + emit;
}

// ============================================================================
// Simulation State (distributed)
// ============================================================================

struct SimulationState {
    int rank = 0;
    int nprocs = 1;
    size_t numTriangles = 0;   // n
    size_t numTimesteps = 0;   // T
    size_t sourceIndex = 0;
    val_t reflectivity = 0;

    std::vector<Triangle> triangles;   // Full mesh (replicated)
    std::vector<val_t> areas;          // Area of each triangle (replicated)
    std::vector<size_t> rowStart;      // Receiver rows owned by rank k: [rowStart[k], rowStart[k+1])
    size_t r0 = 0, L = 0;              // This rank's receiver rows

    Octree octree;                     // Host octree (flattened to the device below)
    std::vector<DevNode> flatNodes;
    std::vector<uint32_t> flatLeaf;
    int maxDepth = 0;

    // Host copies of the radiosity history (replicated on every rank)
    std::vector<val_t> radB;           // T x n
    std::vector<val_t> distances;      // n (replicated after gather)
    uint64_t nonZeroKij = 0;

    // Device buffers
    DevTri* dTris = nullptr;
    DevNode* dNodes = nullptr;
    uint32_t* dLeaf = nullptr;
    float* dAreas = nullptr;
    float* dKijT = nullptr;            // n x L (emitter-major)
    int* dTauT = nullptr;             // n x L (emitter-major)
    float* dRadB = nullptr;            // T x n
    unsigned long long* dNz = nullptr;
};

int32_t flattenNode(const Octree& node, int depth, std::vector<DevNode>& nodes,
                    std::vector<uint32_t>& leaf, int& maxDepth) {
    const int32_t id = static_cast<int32_t>(nodes.size());
    nodes.emplace_back();
    maxDepth = std::max(maxDepth, depth);

    DevNode dn{};
    dn.center = {node.center.x, node.center.y, node.center.z};
    dn.half = {node.halfExtent.x, node.halfExtent.y, node.halfExtent.z};
    for (auto& c : dn.child) c = -1;
    if (!node.triangleIndices.empty()) {
        dn.leafBegin = static_cast<uint32_t>(leaf.size());
        dn.leafCount = static_cast<uint32_t>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) leaf.push_back(static_cast<uint32_t>(idx));
    } else {
        for (int c = 0; c < 8; ++c) {
            if (node.children[c]) {
                dn.child[c] = flattenNode(*node.children[c], depth + 1, nodes, leaf, maxDepth);
            }
        }
    }
    nodes[static_cast<size_t>(id)] = dn;
    return id;
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& st, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh (deterministic, identical on every rank)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    st.triangles = std::move(mesh.triangles);
    st.numTriangles = st.triangles.size();
    st.numTimesteps = timesteps;
    st.sourceIndex = sourceIdx % st.numTriangles;
    st.reflectivity = reflectivity;

    if (st.rank == 0) printf("Generated icosphere mesh with %zu triangles\n", st.numTriangles);

    // Build octree for spatial acceleration, then flatten for the device
    if (st.rank == 0) printf("Building octree...\n");
    st.octree.build(st.triangles);
    st.flatNodes.clear();
    st.flatLeaf.clear();
    st.maxDepth = 0;
    flattenNode(st.octree, 0, st.flatNodes, st.flatLeaf, st.maxDepth);
    if (7 * st.maxDepth + 2 > kTraversalStack) fatal("octree too deep for traversal stack");

    // Initialize areas
    st.areas.resize(st.numTriangles);
    for (size_t i = 0; i < st.numTriangles; ++i) {
        st.areas[i] = st.triangles[i].area();
    }

    // Row partition: rank k owns receiver rows [rowStart[k], rowStart[k+1])
    st.rowStart.resize(static_cast<size_t>(st.nprocs) + 1);
    for (int k = 0; k <= st.nprocs; ++k) {
        st.rowStart[static_cast<size_t>(k)] = st.numTriangles * static_cast<size_t>(k) / static_cast<size_t>(st.nprocs);
    }
    st.r0 = st.rowStart[static_cast<size_t>(st.rank)];
    st.L = st.rowStart[static_cast<size_t>(st.rank) + 1] - st.r0;

    // Radiosity history, replicated on every rank (emission is applied inside the timestep kernel)
    st.radB.assign(timesteps * st.numTriangles, ZERO);
}

// ============================================================================
// Device Setup
// ============================================================================

void setupDevice(SimulationState& st) {
    const size_t n = st.numTriangles, L = st.L, T = st.numTimesteps;

    std::vector<DevTri> tris(n);
    for (size_t i = 0; i < n; ++i) {
        const Triangle& t = st.triangles[i];
        tris[i].a = {t.a.x, t.a.y, t.a.z};
        tris[i].b = {t.b.x, t.b.y, t.b.z};
        tris[i].c = {t.c.x, t.c.y, t.c.z};
        tris[i].n = {t.normal().x, t.normal().y, t.normal().z};
    }

    CUDA_CHECK(cudaMalloc(&st.dTris, n * sizeof(DevTri)));
    CUDA_CHECK(cudaMemcpy(st.dTris, tris.data(), n * sizeof(DevTri), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&st.dNodes, st.flatNodes.size() * sizeof(DevNode)));
    CUDA_CHECK(cudaMemcpy(st.dNodes, st.flatNodes.data(), st.flatNodes.size() * sizeof(DevNode),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&st.dLeaf, std::max<size_t>(1, st.flatLeaf.size()) * sizeof(uint32_t)));
    if (!st.flatLeaf.empty())
        CUDA_CHECK(cudaMemcpy(st.dLeaf, st.flatLeaf.data(), st.flatLeaf.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&st.dAreas, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(st.dAreas, st.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const size_t nL = std::max<size_t>(1, n * L);
    CUDA_CHECK(cudaMalloc(&st.dKijT, nL * sizeof(float)));
    CUDA_CHECK(cudaMemset(st.dKijT, 0, nL * sizeof(float)));  // Culled pairs keep Kij = 0
    CUDA_CHECK(cudaMalloc(&st.dTauT, nL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&st.dRadB, std::max<size_t>(1, T * n) * sizeof(float)));
    CUDA_CHECK(cudaMemset(st.dRadB, 0, std::max<size_t>(1, T * n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&st.dNz, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(st.dNz, 0, sizeof(unsigned long long)));
}

void freeDevice(SimulationState& st) {
    cudaFree(st.dTris);
    cudaFree(st.dNodes);
    cudaFree(st.dLeaf);
    cudaFree(st.dAreas);
    cudaFree(st.dKijT);
    cudaFree(st.dTauT);
    cudaFree(st.dRadB);
    cudaFree(st.dNz);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Time delay between triangle centers, in timesteps
int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

void computeTimeDelays(SimulationState& st) {
    if (st.rank == 0) printf("Computing time delays (Tau)...\n");

    const size_t n = st.numTriangles, L = st.L;
    std::vector<int> tauT(std::max<size_t>(1, n * L), 0);

    #pragma omp parallel for schedule(static)
    for (size_t il = 0; il < L; ++il) {
        const size_t i = st.r0 + il;
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            tauT[j * L + il] = computeTau(st.triangles[i], st.triangles[j]);
        }
    }

    if (L > 0) {
        CUDA_CHECK(cudaMemcpy(st.dTauT, tauT.data(), n * L * sizeof(int), cudaMemcpyHostToDevice));
    }
}

void computeFormFactors(SimulationState& st, const Mt19937Jump& rngJump) {
    const size_t n = st.numTriangles, L = st.L;
    const bool root = st.rank == 0;
    if (root) printf("Computing form factors (Kij)...\n");

    // 1. Number of non-culled emitters per local receiver row (identical culling to computeKij)
    std::vector<uint32_t> localCounts(std::max<size_t>(1, L), 0);
    #pragma omp parallel for schedule(static)
    for (size_t il = 0; il < L; ++il) {
        const size_t i = st.r0 + il;
        const Vec3 ni = st.triangles[i].normal();
        uint32_t c = 0;
        for (size_t j = 0; j < n; ++j) {
            if (j != i && !(ni.dot(st.triangles[j].normal()) > 0.99f)) ++c;
        }
        localCounts[il] = c;
    }

    // Gather all row counts to every rank
    std::vector<uint32_t> countsAll(n, 0);
    {
        std::vector<int> rc(static_cast<size_t>(st.nprocs)), dp(static_cast<size_t>(st.nprocs));
        for (int k = 0; k < st.nprocs; ++k) {
            const size_t a = st.rowStart[static_cast<size_t>(k)], b = st.rowStart[static_cast<size_t>(k) + 1];
            rc[static_cast<size_t>(k)] = static_cast<int>(b - a);
            dp[static_cast<size_t>(k)] = static_cast<int>(a);
        }
        MPI_Allgatherv(localCounts.data(), static_cast<int>(L), MPI_UNSIGNED, countsAll.data(),
                       rc.data(), dp.data(), MPI_UNSIGNED, MPI_COMM_WORLD);
    }

    // Each sample consumes 64 generator draws (16 rays x 4 uniform values)
    std::vector<uint64_t> drawPrefix(n + 1, 0);
    for (size_t i = 0; i < n; ++i) drawPrefix[i + 1] = drawPrefix[i] + 64ull * countsAll[i];

    const int outPerDraw = Mt19937Jump::outputsPerDraw();

    // 2. RNG lanes: contiguous row blocks inside this rank, each with its own positioned engine.
    //    A step processes one row per lane, so the lane count is bounded by the host thread
    //    count and by the staging budget of one pipeline slot.
    size_t maxRowPairs = 1;
    for (size_t i = 0; i < n; ++i) maxRowPairs = std::max<size_t>(maxRowPairs, countsAll[i]);
    const size_t bytesPerRow = maxRowPairs * (sizeof(DevPair) + 64 * sizeof(float));
    const size_t budgetRows = std::max<size_t>(1, kStagingBudgetBytes / bytesPerRow);
    const size_t laneCap = std::min({static_cast<size_t>(omp_get_max_threads()), budgetRows, std::max<size_t>(L, 1)});
    const int nl = static_cast<int>(std::max<size_t>(1, laneCap));
    std::vector<size_t> lb(static_cast<size_t>(nl)), le(static_cast<size_t>(nl));
    for (int k = 0; k < nl; ++k) {
        lb[static_cast<size_t>(k)] = L * static_cast<size_t>(k) / static_cast<size_t>(nl);
        le[static_cast<size_t>(k)] = L * static_cast<size_t>(k + 1) / static_cast<size_t>(nl);
    }

    std::vector<std::mt19937> engines(static_cast<size_t>(nl));
    #pragma omp parallel for schedule(static, 1)
    for (int k = 0; k < nl; ++k) {
        const size_t ku = static_cast<size_t>(k);
        if (lb[ku] < le[ku]) {
            const uint64_t off = drawPrefix[st.r0 + lb[ku]] * static_cast<uint64_t>(outPerDraw);
            engines[ku] = rngJump.engineAt(off);
        }
    }

    // Every rank must run the same number of steps (they share collectives), even if it owns fewer rows
    size_t localSteps = 0;
    for (int k = 0; k < nl; ++k) localSteps = std::max(localSteps, le[static_cast<size_t>(k)] - lb[static_cast<size_t>(k)]);
    unsigned long long stepsLocal = localSteps, stepsGlobal = 0;
    MPI_Allreduce(&stepsLocal, &stepsGlobal, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    const size_t numSteps = static_cast<size_t>(stepsGlobal);

    size_t maxPairs = 1;
    for (size_t s = 0; s < numSteps; ++s) {
        size_t total = 0;
        for (int k = 0; k < nl; ++k) {
            const size_t ku = static_cast<size_t>(k);
            if (lb[ku] + s < le[ku]) total += countsAll[st.r0 + lb[ku] + s];
        }
        maxPairs = std::max(maxPairs, total);
    }

    // Two-slot pipeline: host sample generation for step s+1, the host-to-device copy of
    // that step, and the kernel of step s all overlap.
    DevPair* hPairs[2] = {nullptr, nullptr};
    float* hDraws[2] = {nullptr, nullptr};
    DevPair* dPairs[2] = {nullptr, nullptr};
    float* dDraws[2] = {nullptr, nullptr};
    cudaStream_t copyStream, computeStream;
    cudaEvent_t copyDone[2], kernelDone[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMallocHost(&hPairs[b], maxPairs * sizeof(DevPair)));
        CUDA_CHECK(cudaMallocHost(&hDraws[b], maxPairs * 64 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dPairs[b], maxPairs * sizeof(DevPair)));
        CUDA_CHECK(cudaMalloc(&dDraws[b], maxPairs * 64 * sizeof(float)));
        CUDA_CHECK(cudaEventCreateWithFlags(&copyDone[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&kernelDone[b], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaStreamCreate(&copyStream));
    CUDA_CHECK(cudaStreamCreate(&computeStream));

    // Generates the samples for step s into staging buffer `buf`, in the order of the original loops.
    auto generate = [&](size_t s, int buf) -> size_t {
        std::vector<size_t> base(static_cast<size_t>(nl), 0), cnt(static_cast<size_t>(nl), 0);
        size_t total = 0;
        for (int k = 0; k < nl; ++k) {
            const size_t ku = static_cast<size_t>(k);
            if (lb[ku] + s < le[ku]) {
                cnt[ku] = countsAll[st.r0 + lb[ku] + s];
                base[ku] = total;
                total += cnt[ku];
            }
        }
        #pragma omp parallel for schedule(static, 1)
        for (int k = 0; k < nl; ++k) {
            const size_t ku = static_cast<size_t>(k);
            if (lb[ku] + s >= le[ku]) continue;
            const size_t gi = st.r0 + lb[ku] + s;
            const Vec3 ni = st.triangles[gi].normal();
            std::uniform_real_distribution<val_t> dist(0.0f, 1.0f);
            std::mt19937& eng = engines[ku];
            size_t pos = base[ku];
            for (size_t j = 0; j < n; ++j) {
                if (j == gi) continue;
                if (ni.dot(st.triangles[j].normal()) > 0.99f) continue;
                DevPair& pr = hPairs[buf][pos];
                pr.i = static_cast<uint32_t>(gi);
                pr.j = static_cast<uint32_t>(j);
                pr.drawOff = static_cast<uint32_t>(pos * 64);
                float* dr = hDraws[buf] + pos * 64;
                for (int q = 0; q < 64; ++q) dr[q] = dist(eng);  // uI, vI, uJ, vJ per ray
                ++pos;
            }
        }
        return total;
    };

    size_t globalDone = 0;
    size_t nextMark = 100;
    size_t nPairs[2] = {0, 0};
    auto uploadSlot = [&](int b, size_t np) {
        if (np > 0) {
            CUDA_CHECK(cudaMemcpyAsync(dPairs[b], hPairs[b], np * sizeof(DevPair), cudaMemcpyHostToDevice, copyStream));
            CUDA_CHECK(cudaMemcpyAsync(dDraws[b], hDraws[b], np * 64 * sizeof(float), cudaMemcpyHostToDevice, copyStream));
        }
        CUDA_CHECK(cudaEventRecord(copyDone[b], copyStream));
    };
    if (numSteps > 0) {
        nPairs[0] = generate(0, 0);
        uploadSlot(0, nPairs[0]);
    }

    for (size_t s = 0; s < numSteps; ++s) {
        const int cur = static_cast<int>(s & 1u);
        const int nxt = 1 - cur;
        const size_t np = nPairs[cur];
        size_t activeRows = 0;
        for (int k = 0; k < nl; ++k) {
            const size_t ku = static_cast<size_t>(k);
            if (lb[ku] + s < le[ku]) ++activeRows;
        }

        CUDA_CHECK(cudaStreamWaitEvent(computeStream, copyDone[cur], 0));
        if (np > 0) {
            const unsigned grid = static_cast<unsigned>((np + kBlock - 1) / kBlock);
            formFactorKernel<<<grid, kBlock, 0, computeStream>>>(dPairs[cur], static_cast<uint32_t>(np), dDraws[cur],
                                                                 st.dTris, st.dNodes, st.dLeaf,
                                                                 static_cast<uint32_t>(st.r0), static_cast<uint32_t>(L),
                                                                 st.dKijT, st.dNz);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(kernelDone[cur], computeStream));

        // Stage step s+1 while the GPU runs step s. Slot nxt is reusable once kernel s-1 is done.
        if (s + 1 < numSteps) {
            if (s >= 1) CUDA_CHECK(cudaEventSynchronize(kernelDone[nxt]));
            nPairs[nxt] = generate(s + 1, nxt);
            CUDA_CHECK(cudaStreamWaitEvent(copyStream, kernelDone[nxt], 0));
            uploadSlot(nxt, nPairs[nxt]);
        }

        CUDA_CHECK(cudaStreamSynchronize(computeStream));

        unsigned long long activeLocal = activeRows, activeGlobal = 0;
        MPI_Allreduce(&activeLocal, &activeGlobal, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        globalDone += static_cast<size_t>(activeGlobal);
        if (root && (globalDone >= nextMark || globalDone == n)) {
            printf("  Progress: %zu/%zu triangles\n", std::min(globalDone, n), n);
            while (nextMark <= globalDone) nextMark += 100;
        }
    }

    CUDA_CHECK(cudaStreamDestroy(copyStream));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    for (int b = 0; b < 2; ++b) {
        cudaEventDestroy(copyDone[b]);
        cudaEventDestroy(kernelDone[b]);
        cudaFree(dPairs[b]);
        cudaFree(dDraws[b]);
        cudaFreeHost(hPairs[b]);
        cudaFreeHost(hDraws[b]);
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& st) {
    if (st.rank == 0) printf("Running wave propagation simulation...\n");

    const size_t n = st.numTriangles, T = st.numTimesteps, L = st.L;
    const int timeOff = static_cast<int>(T / 2);

    std::vector<int> recvCounts(static_cast<size_t>(st.nprocs)), recvDispls(static_cast<size_t>(st.nprocs));
    for (int k = 0; k < st.nprocs; ++k) {
        const size_t a = st.rowStart[static_cast<size_t>(k)], b = st.rowStart[static_cast<size_t>(k) + 1];
        recvCounts[static_cast<size_t>(k)] = static_cast<int>(b - a);
        recvDispls[static_cast<size_t>(k)] = static_cast<int>(a);
    }

    for (size_t t = 0; t < T; ++t) {
        if (L > 0) {
            const unsigned grid = static_cast<unsigned>((L + kBlock - 1) / kBlock);
            radiosityStepKernel<<<grid, kBlock>>>(static_cast<int>(t), static_cast<int>(n), static_cast<int>(L),
                                                 static_cast<int>(st.r0), static_cast<int>(st.sourceIndex), timeOff,
                                                 st.reflectivity, st.dKijT, st.dTauT, st.dAreas, st.dRadB);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            // Pull this rank's slice of the new radiosity row
            CUDA_CHECK(cudaMemcpy(st.radB.data() + t * n + st.r0, st.dRadB + t * n + st.r0,
                                  L * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // Share all slices so every rank holds the complete radiosity row t
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, st.radB.data() + t * n, recvCounts.data(),
                       recvDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(st.dRadB + t * n, st.radB.data() + t * n, n * sizeof(float), cudaMemcpyHostToDevice));

        if (st.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& st) {
    if (st.rank == 0) printf("Computing distances via cross-correlation...\n");

    const size_t n = st.numTriangles, T = st.numTimesteps, L = st.L;
    std::vector<val_t> local(std::max<size_t>(1, L), ZERO);

    #pragma omp parallel for schedule(dynamic, 8)
    for (size_t il = 0; il < L; ++il) {
        const size_t i = st.r0 + il;
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;
            for (size_t tt = t; tt < T; ++tt) {
                val_t pB = st.radB[tt * n + i];
                val_t pS = st.radB[(tt - t) * n + st.sourceIndex];
                sum += pS * pB;
            }
            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }
        local[il] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    st.distances.assign(n, ZERO);
    std::vector<int> rc(static_cast<size_t>(st.nprocs)), dp(static_cast<size_t>(st.nprocs));
    for (int k = 0; k < st.nprocs; ++k) {
        const size_t a = st.rowStart[static_cast<size_t>(k)], b = st.rowStart[static_cast<size_t>(k) + 1];
        rc[static_cast<size_t>(k)] = static_cast<int>(b - a);
        dp[static_cast<size_t>(k)] = static_cast<int>(a);
    }
    MPI_Allgatherv(local.data(), static_cast<int>(L), MPI_FLOAT, st.distances.data(), rc.data(), dp.data(),
                   MPI_FLOAT, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& st) {
    printf("\nValidation:\n");

    const size_t n = st.numTriangles;

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < n; ++i) {
        val_t d = st.distances[i];
        if (d < 0) {
            allNonNegative = false;
            printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(n));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, n);

    // Check source distance is zero or very small
    val_t srcDist = st.distances[st.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    int receivedEnergy = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t t = 0; t < st.numTimesteps; ++t) {
            if (st.radB[t * n + i] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, n);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    const uint64_t nonZeroKij = st.nonZeroKij;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           static_cast<unsigned long long>(nonZeroKij), n * n,
           100.0f * static_cast<val_t>(nonZeroKij) / static_cast<val_t>(n * n));

    if (nonZeroKij == 0) {
        printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    printf("  Validation: PASSED\n");
    return true;
}

// ============================================================================
// Hash for Verification
// ============================================================================

uint64_t computeHash(const std::vector<val_t>& distances) {
    uint64_t hash = 0;
    for (size_t i = 0; i < distances.size(); ++i) {
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&distances[i]);
        hash ^= (static_cast<uint64_t>(*ptr) + i) * 0x9e3779b97f4a7c15ULL;
    }
    return hash;
}

// ============================================================================
// Helper: Compute subdivision level from target triangle count
// ============================================================================

int getSubdivisionsForTriangleCount(int targetTriangles) {
    // Icosphere: 20 triangles initially, 4x per subdivision
    // subdivisions: 0->20, 1->80, 2->320, 3->1280, 4->5120, 5->20480
    int subdivisions = 0;
    int triangles = 20;
    while (triangles < targetTriangles && subdivisions < 6) {
        subdivisions++;
        triangles *= 4;
    }
    return subdivisions;
}

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Target number of triangles (default: 320)\n");
    printf("               Actual count will be rounded to nearest icosphere level:\n");
    printf("               20, 80, 320, 1280, 5120, 20480\n");
    printf("  -t <num>     Number of timesteps (default: 50)\n");
    printf("  -s <num>     Source triangle index (default: 0)\n");
    printf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    printf("  -v           Enable validation\n");
    printf("  -o           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    SimulationState st;
    MPI_Comm_rank(MPI_COMM_WORLD, &st.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &st.nprocs);
    const bool root = st.rank == 0;

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            targetTriangles = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            timesteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sourceIdx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            reflectivity = static_cast<val_t>(atof(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-o") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // One GPU per rank, round-robin over the devices on the node
    MPI_Comm shmComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, st.rank, MPI_INFO_NULL, &shmComm);
    int localRank = 0;
    MPI_Comm_rank(shmComm, &localRank);
    MPI_Comm_free(&shmComm);
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices <= 0) fatal("no CUDA devices available");
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    const int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (root) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Initialize
    initializeSimulation(st, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);
    setupDevice(st);

    if (root) printf("\n");

    const Mt19937Jump rngJump;

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(st);
    computeFormFactors(st, rngJump);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (root) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(st);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (root) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(st);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    // Gather visibility statistics
    {
        unsigned long long localNz = 0;
        CUDA_CHECK(cudaMemcpy(&localNz, st.dNz, sizeof(localNz), cudaMemcpyDeviceToHost));
        unsigned long long globalNz = 0;
        MPI_Allreduce(&localNz, &globalNz, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        st.nonZeroKij = globalNz;
    }

    if (root) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");

        // Total time
        long totalTime = preDuration + simDuration + distDuration;
        printf("Total computation time: %ld ms\n", totalTime);

        // Performance metrics
        size_t n = st.numTriangles;
        size_t t = st.numTimesteps;
        double kijOps = static_cast<double>(n * n);
        double simOps = static_cast<double>(n * n * t);
        double distOps = static_cast<double>(n * t * t);

        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        // Memory usage
        size_t memKij = n * n * sizeof(val_t);
        size_t memTau = n * n * sizeof(int);
        size_t memRad = 2 * t * n * sizeof(val_t);
        size_t totalMem = memKij + memTau + memRad;
        printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));
    }

    // Hash
    uint64_t hash = computeHash(st.distances);
    if (root) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // Print results for external validation
    if (root && printResults) {
        // Convert distances to double for output
        std::vector<double> distData(st.distances.begin(), st.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (root) {
            if (!validateResults(st)) exitCode = 1;
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    freeDevice(st);
    MPI_Finalize();
    return exitCode;
}
