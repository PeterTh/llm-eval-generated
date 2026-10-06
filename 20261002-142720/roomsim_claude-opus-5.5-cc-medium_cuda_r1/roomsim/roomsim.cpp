/**
 * Room Response Simulation Benchmark
 * 
 * This is a CUDA-parallel implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
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
#include <thread>
#include <vector>

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
// Octree for Spatial Acceleration
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

public:
    // Check if a ray intersects this node's bounding box
    bool rayIntersectsBox(const Vec3& p1, const Vec3& p2) const {
        Vec3 d = (p2 - p1) * 0.5f;
        Vec3 c = p1 + d - center;
        Vec3 ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};

        if (std::abs(c.x) > halfExtent.x + ad.x) return false;
        if (std::abs(c.y) > halfExtent.y + ad.y) return false;
        if (std::abs(c.z) > halfExtent.z + ad.z) return false;

        if (std::abs(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
        if (std::abs(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
        if (std::abs(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

        return true;
    }

    // Apply a function to all triangles potentially intersecting the ray
    // Returns true if the function returns true for any triangle
    template<typename Func>
    bool applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const {
        // If leaf node, check triangles directly
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

        // Otherwise, descend to children
        for (int i = 0; i < 8; ++i) {
            if (children[i] && children[i]->rayIntersectsBox(p1, p2)) {
                if (children[i]->applyToTris(p1, p2, func)) return true;
            }
        }
        return false;
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
// CUDA Error Handling
// ============================================================================

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err_));                        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

template <typename T>
static T* deviceUpload(const std::vector<T>& v) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(1, v.size()) * sizeof(T)));
    if (!v.empty()) CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// ============================================================================
// Random Number Generation (MT19937 with GPU jump-ahead)
// ============================================================================
//
// The reference uses one std::mt19937(42) stream consumed sequentially by
// computeKij(): every non-culled (i, j) pair draws exactly 64 numbers
// (16 rays x 2 points x 2 barycentric coordinates); culled pairs draw none.
// The stream offset of each pair is therefore 64 * (its rank among the
// non-culled pairs in row-major order). Each GPU block owns a contiguous
// range of ranks and starts from the exact MT19937 state at its offset,
// obtained by polynomial jump-ahead over GF(2). This reproduces the
// sequential random stream bit-for-bit.

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr int MT_DEG = 19937;                     // degree of the characteristic polynomial
constexpr int MT_POLY_WORDS = (MT_DEG + 63) / 64; // 312
constexpr uint32_t MT_MATRIX_A = 0x9908b0dfu;
constexpr uint32_t MT_UPPER = 0x80000000u;
constexpr uint32_t MT_LOWER = 0x7fffffffu;
constexpr int MT_SEQ_LEN = MT_DEG + MT_N - 1;     // words needed to evaluate a jump

__host__ __device__ inline uint32_t mtNext(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t y = (a & MT_UPPER) | (b & MT_LOWER);
    return c ^ (y >> 1) ^ ((b & 1u) ? MT_MATRIX_A : 0u);
}

__host__ __device__ inline uint32_t mtTemper(uint32_t y) {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

// std::uniform_real_distribution<float>(0, 1) on mt19937 (libstdc++ generate_canonical)
__device__ inline float mtToUnitFloat(uint32_t x) {
    float r = __uint2float_rn(x) * 2.3283064365386963e-10f;  // * 2^-32 (exact)
    return r >= 1.0f ? 0.99999994f : r;
}

static void hostTwist(uint32_t* mt) {
    for (int i = 0; i < MT_N; ++i)
        mt[i] = mtNext(mt[i], mt[(i + 1) % MT_N], mt[(i + MT_M) % MT_N]);
}

// Exponents k < 19937 with non-zero coefficient in the characteristic
// polynomial of MT19937 (x^19937 + ... + 1, 135 terms). Derived with
// Berlekamp-Massey from the generator's bit sequence.
static const int MT_CHARPOLY_TERMS[] = {
    0, 1189, 1416, 1585, 1643, 1870, 2493, 2773, 3000, 3227, 3454, 3681,
    3908, 4135, 4362, 4753, 5661, 6337, 6569, 7129, 7477, 7525, 7583, 7752,
    7979, 8206, 9505, 9901, 9969, 10128, 10693, 10761, 10920, 11089, 11147, 11157,
    11215, 11321, 11374, 11384, 11485, 11611, 11712, 11717, 11838, 11881, 11944, 11997,
    12277, 12335, 12393, 12504, 12509, 12620, 12673, 12731, 12736, 12789, 12905, 12958,
    12963, 13137, 13185, 13190, 13243, 13301, 13412, 13528, 13533, 13639, 13697, 13760,
    13813, 13866, 14093, 14151, 14209, 14320, 14325, 14436, 14547, 14552, 14605, 14721,
    14774, 14779, 14953, 15001, 15006, 15059, 15117, 15228, 15344, 15349, 15455, 15513,
    15576, 15629, 15682, 15909, 15967, 16025, 16136, 16141, 16252, 16363, 16368, 16421,
    16537, 16590, 16595, 16817, 16822, 16875, 16933, 17044, 17160, 17271, 17329, 17445,
    17498, 17725, 17783, 17841, 17952, 18068, 18179, 18237, 18406, 18633, 18691, 18860,
    19087, 19314,
};

// GF(2) polynomial arithmetic modulo the MT19937 characteristic polynomial
class MTJumpPoly {
public:
    std::vector<int> lowTerms;  // exponents < MT_DEG with non-zero coefficient

    MTJumpPoly() : lowTerms(std::begin(MT_CHARPOLY_TERMS), std::end(MT_CHARPOLY_TERMS)) {}

    using Poly = std::vector<uint64_t>;  // MT_POLY_WORDS words, degree < MT_DEG

    static bool getBit(const Poly& p, int k) { return (p[k >> 6] >> (k & 63)) & 1ULL; }

    // XOR w << shift into p (shift >= 0)
    static void xorShifted(Poly& p, uint64_t w, int shift) {
        int q = shift >> 6, b = shift & 63;
        p[q] ^= w << b;
        if (b) p[q + 1] ^= w >> (64 - b);
    }

    // Reduce bits [MT_DEG, topBit] word by word, top down. Every low term is
    // at most x^(MT_DEG-623), so a word only feeds lower, unprocessed words.
    void reduce(Poly& p, int topBit) const {
        for (int q = topBit >> 6; q >= 0 && q * 64 + 63 >= MT_DEG; --q) {
            int lowBit = std::max(q * 64, MT_DEG);
            uint64_t w = p[q] >> (lowBit - q * 64);
            if (!w) continue;
            p[q] &= (lowBit - q * 64) ? ((1ULL << (lowBit - q * 64)) - 1) : 0ULL;
            for (int e : lowTerms) xorShifted(p, w, lowBit - MT_DEG + e);
        }
    }

    Poly square(const Poly& a) const {
        Poly r(2 * MT_POLY_WORDS + 1, 0);
        for (int q = 0; q < MT_POLY_WORDS; ++q) {
            uint64_t w = a[q];
            for (int half = 0; half < 2; ++half) {
                uint64_t x = (w >> (32 * half)) & 0xffffffffULL;
                x = (x | (x << 16)) & 0x0000ffff0000ffffULL;
                x = (x | (x << 8)) & 0x00ff00ff00ff00ffULL;
                x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0fULL;
                x = (x | (x << 2)) & 0x3333333333333333ULL;
                x = (x | (x << 1)) & 0x5555555555555555ULL;
                r[2 * q + half] = x;
            }
        }
        reduce(r, 2 * MT_DEG - 2);
        r.resize(MT_POLY_WORDS);
        return r;
    }

    Poly mulX(const Poly& a) const {
        Poly r(MT_POLY_WORDS + 1, 0);
        for (int q = 0; q < MT_POLY_WORDS; ++q) {
            r[q] |= a[q] << 1;
            r[q + 1] |= a[q] >> 63;
        }
        reduce(r, MT_DEG);
        r.resize(MT_POLY_WORDS);
        return r;
    }

    // x^e mod charpoly
    Poly powX(uint64_t e) const {
        Poly r(MT_POLY_WORDS, 0);
        r[0] = 1;
        for (int b = 63; b >= 0; --b) {
            r = square(r);
            if ((e >> b) & 1ULL) r = mulX(r);
        }
        return r;
    }
};

// Jump kernel: dst ^= Q(T) * src, where T is the MT state transition and Q a
// polynomial of degree < 19937, given by the sorted positions of its non-zero
// coefficients. The state window W_n = (x_n .. x_{n+623}) satisfies
// T^i W_n = W_{n+i}, so (Q(T) W)[w] = XOR_{i : q_i = 1} x_{n+i+w}.
// Target t = (group g, index k): src state segBase[g] + k * elemStride,
// dst state segBase[g] + (k + dstOffset) * elemStride (zero-initialized).
// The coefficient list is split over `splits` blocks per target.
constexpr int JUMP_THREADS = MT_N / 3;  // each thread owns 3 state words

__global__ void __launch_bounds__(JUMP_THREADS)
mtJumpKernel(uint32_t* states, const int* __restrict__ segBase, int count, int dstOffset, int elemStride,
             const uint16_t* __restrict__ terms, int numTerms, int splits) {
    extern __shared__ uint32_t seq[];  // up to MT_SEQ_LEN words
    const int part = blockIdx.x % splits;
    const int target = blockIdx.x / splits;
    const int g = target / count;
    const int k = target % count;
    const int t0 = static_cast<int>(static_cast<int64_t>(numTerms) * part / splits);
    const int t1 = static_cast<int>(static_cast<int64_t>(numTerms) * (part + 1) / splits);
    if (t0 >= t1) return;
    const int seqEnd = terms[t1 - 1] + MT_N;
    const uint32_t* src = states + static_cast<size_t>(segBase[g] + k * elemStride) * MT_N;
    uint32_t* dst = states + static_cast<size_t>(segBase[g] + (k + dstOffset) * elemStride) * MT_N;
    for (int i = threadIdx.x; i < MT_N; i += blockDim.x) seq[i] = src[i];
    __syncthreads();
    // Extend the sequence: x_{n+624} depends on x_n, x_{n+1}, x_{n+397}
    for (int base = MT_N; base < seqEnd; base += MT_N - MT_M) {
        for (int t = threadIdx.x; t < MT_N - MT_M; t += blockDim.x) {
            int idx = base + t;
            if (idx < seqEnd)
                seq[idx] = mtNext(seq[idx - MT_N], seq[idx - MT_N + 1], seq[idx - (MT_N - MT_M)]);
        }
        __syncthreads();
    }
    const int w = threadIdx.x;
    uint32_t acc0 = 0, acc1 = 0, acc2 = 0;
    #pragma unroll 4
    for (int t = t0; t < t1; ++t) {
        const uint32_t* p = seq + __ldg(terms + t) + w;
        acc0 ^= p[0];
        acc1 ^= p[JUMP_THREADS];
        acc2 ^= p[2 * JUMP_THREADS];
    }
    if (acc0) atomicXor(dst + w, acc0);
    if (acc1) atomicXor(dst + w + JUMP_THREADS, acc1);
    if (acc2) atomicXor(dst + w + 2 * JUMP_THREADS, acc2);
}

// Jump polynomials as concatenated coefficient position lists
struct JumpTerms {
    std::vector<uint16_t> terms;
    std::vector<int> offset;  // per poly, plus end
};

static JumpTerms buildJumpTerms(const std::vector<MTJumpPoly::Poly>& polys) {
    JumpTerms jt;
    for (const auto& p : polys) {
        jt.offset.push_back(static_cast<int>(jt.terms.size()));
        for (int q = 0; q < MT_POLY_WORDS; ++q)
            for (uint64_t bits = p[q]; bits; bits &= bits - 1)
                jt.terms.push_back(static_cast<uint16_t>(q * 64 + __builtin_ctzll(bits)));
    }
    jt.offset.push_back(static_cast<int>(jt.terms.size()));
    return jt;
}

// Doubling: for groups starting at element dGroupBase[g] (in units of
// elemStride), fill elements 1 .. total-1 from element 0, where poly
// firstPoly + r (terms in dTerms) jumps by 2^r elements. Targets must be zero.
static void mtFillByDoubling(uint32_t* dStates, const int* dGroupBase, int groups, int total, int elemStride,
                             const uint16_t* dTerms, const JumpTerms& jt, int firstPoly, int numSMs,
                             cudaStream_t stream) {
    const size_t shmem = MT_SEQ_LEN * sizeof(uint32_t);  // opt-in limit set at initialization
    for (int r = 0, step = 1; step < total; ++r, step *= 2) {
        int count = std::min(step, total - step);
        int targets = count * groups;
        int splits = std::max(1, std::min(32, 2 * numSMs / targets));
        int p = firstPoly + r;
        mtJumpKernel<<<targets * splits, JUMP_THREADS, shmem, stream>>>(
            dStates, dGroupBase, count, step, elemStride, dTerms + jt.offset[p],
            jt.offset[p + 1] - jt.offset[p], splits);
        CUDA_CHECK(cudaGetLastError());
    }
}

// In-place MT19937 twist of a 624-word window in shared memory
__device__ inline void mtTwistShared(uint32_t* mt) {
    const int tid = threadIdx.x;
    // Three dependency phases: [0,227), [227,454), [454,624)
    #pragma unroll
    for (int phase = 0; phase < 3; ++phase) {
        int lo = phase * (MT_N - MT_M);
        int hi = phase == 2 ? MT_N : lo + (MT_N - MT_M);
        int i = lo + tid;
        uint32_t v = 0;
        bool active = i < hi;
        if (active) {
            int i1 = i + 1 == MT_N ? 0 : i + 1;
            int im = i + MT_M >= MT_N ? i + MT_M - MT_N : i + MT_M;
            v = mtNext(mt[i], mt[i1], mt[im]);
        }
        __syncthreads();
        if (active) mt[i] = v;
        __syncthreads();
    }
}

// ============================================================================
// Device Geometry (operation order matches the reference build, including
// the fused multiply-adds emitted by the host compiler)
// ============================================================================

struct DVec { float x, y, z; };

__device__ inline DVec dsub(DVec a, DVec b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ inline float ddot(DVec a, DVec b) {
    return __fmaf_rn(a.z, b.z, __fmaf_rn(a.x, b.x, a.y * b.y));
}
__device__ inline DVec dcross(DVec a, DVec b) {
    return {__fmaf_rn(a.y, b.z, -(a.z * b.y)),
            __fmaf_rn(a.z, b.x, -(a.x * b.z)),
            __fmaf_rn(a.x, b.y, -(a.y * b.x))};
}
__device__ inline DVec ldVec(const float* p) { return {p[0], p[1], p[2]}; }

// Per-triangle data: a, b-a, c-a, normal (12 floats)
constexpr int TRI_STRIDE = 12;

// Octree node: box = (cx, cy, cz, hx), info = (hy, hz, first, count), where
// count > 0 is the number of children (stored contiguously from `first`) and
// count < 0 marks a leaf with -count triangle entries starting at `first`.
struct __align__(16) GpuNode {
    float4 box;
    float4 info;
};

// Leaf triangle entry: v0.xyz, idx | e1.xyz, pad | e2.xyz, pad
struct GpuLeafTri {
    float4 v0i;
    float4 e1;
    float4 e2;
};

constexpr int OCT_STACK = 64;   // per-warp packet stack entries
constexpr int OCT_MAX_DEPTH = (OCT_STACK - 1) / 7;

// Segment/box overlap test of Octree::rayIntersectsBox, evaluated without
// early exits (identical result, no divergence)
__device__ inline bool rayIntersectsNode(float4 box, float4 info, DVec mid, DVec d, DVec ad) {
    const float hx = box.w, hy = info.x, hz = info.y;
    DVec c = {mid.x - box.x, mid.y - box.y, mid.z - box.z};
    bool ok = !(fabsf(c.x) > hx + ad.x);
    ok &= !(fabsf(c.y) > hy + ad.y);
    ok &= !(fabsf(c.z) > hz + ad.z);
    ok &= !(fabsf(__fmaf_rn(d.y, c.z, -(d.z * c.y))) > __fmaf_rn(hy, ad.z, hz * ad.y) + EPSILON);
    ok &= !(fabsf(__fmaf_rn(d.z, c.x, -(d.x * c.z))) > __fmaf_rn(hz, ad.x, hx * ad.z) + EPSILON);
    ok &= !(fabsf(__fmaf_rn(d.x, c.y, -(d.y * c.x))) > __fmaf_rn(hx, ad.y, hy * ad.x) + EPSILON);
    return ok;
}

// Moller-Trumbore test of rayTriangleIntersect combined with the blocking
// criterion of isRayBlocked, evaluated without early exits
__device__ inline bool rayHitsTriangle(const GpuLeafTri& t, DVec orig, DVec dir, float rayLen) {
    DVec v0 = {t.v0i.x, t.v0i.y, t.v0i.z};
    DVec e1 = {t.e1.x, t.e1.y, t.e1.z};
    DVec e2 = {t.e2.x, t.e2.y, t.e2.z};
    DVec pvec = dcross(dir, e2);
    float det = ddot(e1, pvec);
    bool ok = !(fabsf(det) < EPSILON);
    float invDet = __frcp_rn(det);  // == 1.0f / det (correctly rounded)
    DVec tvec = dsub(orig, v0);
    float u = ddot(tvec, pvec) * invDet;
    ok &= !(u < 0.0f || u > 1.0f);
    DVec qvec = dcross(tvec, e1);
    float v = ddot(dir, qvec) * invDet;
    ok &= !(v < 0.0f || u + v > 1.0f);
    float dist = ddot(e2, qvec) * invDet;
    return ok && dist > EPSILON && dist < rayLen - EPSILON;
}

// Warp-cooperative (packet) visibility test. Every lane carries its own ray;
// the warp walks the octree with a shared stack of (node, lane mask) entries,
// where a lane is in the mask only if its own ray passed every box test on the
// path. Each ray therefore visits exactly the same leaves and triangles as the
// recursive reference traversal, so the result (blocked or not) is identical.
constexpr int LEAF_COMPACT_FACTOR = 2;  // compact a leaf when it saves at least half the steps
struct PacketEntry { int node; unsigned mask; };

__device__ bool isRayBlockedPacket(bool valid, DVec from, DVec to, DVec dirNorm, float rayLen,
                                   const GpuNode* __restrict__ nodes,
                                   const GpuLeafTri* __restrict__ leafTris,
                                   int srcIdx, int dstIdx, PacketEntry* stack, int* rankLane,
                                   unsigned* sharedBlocked) {
    const int lane = threadIdx.x & 31;
    const unsigned laneBit = 1u << lane;
    DVec d = {(to.x - from.x) * 0.5f, (to.y - from.y) * 0.5f, (to.z - from.z) * 0.5f};
    DVec ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    DVec mid = {from.x + d.x, from.y + d.y, from.z + d.z};

    bool blocked = false;
    unsigned done = 0;  // lanes whose ray is already known to be blocked
    unsigned active = __ballot_sync(0xffffffffu, valid);
    if (!active) return false;
    if (lane == 0) stack[0] = {0, active};  // root is entered without a box test
    int sp = 1;
    __syncwarp();
    while (sp > 0) {
        --sp;
        PacketEntry e = stack[sp];
        __syncwarp();
        unsigned mask = e.mask & ~done;
        if (!mask) continue;
        const bool mine = (mask & laneBit) != 0;
        const float4 info = nodes[e.node].info;
        const int first = __float_as_int(info.z);
        const int count = __float_as_int(info.w);
        if (count < 0 && LEAF_COMPACT_FACTOR * ((__popc(mask) * -count + 31) / 32) < -count) {
            // Sparse mask: spread the (ray, triangle) tests densely over the lanes
            const unsigned tests = static_cast<unsigned>(__popc(mask) * -count);
            const unsigned inv = 0xffffffffu / static_cast<unsigned>(-count) + 1u;  // slot / count
            if (mine) rankLane[__popc(mask & (laneBit - 1))] = lane;
            if (lane == 0) *sharedBlocked = 0;
            __syncwarp();
            for (unsigned base = 0; base < tests; base += 32) {
                const unsigned slot = base + lane;
                const bool ok = slot < tests;
                const unsigned r = __umulhi(slot, inv);
                const int k = static_cast<int>(slot - r * static_cast<unsigned>(-count));
                const int src = rankLane[ok ? r : 0];
                DVec rFrom = {__shfl_sync(0xffffffffu, from.x, src), __shfl_sync(0xffffffffu, from.y, src),
                              __shfl_sync(0xffffffffu, from.z, src)};
                DVec rDir = {__shfl_sync(0xffffffffu, dirNorm.x, src), __shfl_sync(0xffffffffu, dirNorm.y, src),
                             __shfl_sync(0xffffffffu, dirNorm.z, src)};
                const float rLen = __shfl_sync(0xffffffffu, rayLen, src);
                const int rSrc = __shfl_sync(0xffffffffu, srcIdx, src);
                const int rDst = __shfl_sync(0xffffffffu, dstIdx, src);
                if (ok) {
                    const GpuLeafTri t = leafTris[first + k];
                    int idx = __float_as_int(t.v0i.w);
                    if (idx != rSrc && idx != rDst && rayHitsTriangle(t, rFrom, rDir, rLen))
                        atomicOr(sharedBlocked, 1u << src);
                }
            }
            __syncwarp();
            blocked |= ((*sharedBlocked >> lane) & 1u) != 0;
            done = __ballot_sync(0xffffffffu, blocked);
            __syncwarp();
        } else if (count < 0) {
            for (int k = 0; k < -count; ++k) {
                const GpuLeafTri t = leafTris[first + k];
                int idx = __float_as_int(t.v0i.w);
                bool test = mine && !blocked && idx != srcIdx && idx != dstIdx;
                bool hit = rayHitsTriangle(t, from, dirNorm, rayLen);
                blocked |= test && hit;
            }
            done = __ballot_sync(0xffffffffu, blocked);
        } else {
            for (int c = 0; c < count; ++c) {
                const GpuNode child = nodes[first + c];
                bool hit = mine && rayIntersectsNode(child.box, child.info, mid, d, ad);
                unsigned m = __ballot_sync(0xffffffffu, hit);
                if (m) {
                    if (lane == 0) stack[sp] = {first + c, m};
                    ++sp;
                }
            }
            __syncwarp();
        }
    }
    return blocked;
}

__device__ inline bool pairActive(const float* __restrict__ tris, uint32_t i, uint32_t j) {
    if (i == j) return false;
    DVec ni = ldVec(tris + i * TRI_STRIDE + 9);
    DVec nj = ldVec(tris + j * TRI_STRIDE + 9);
    return !(ddot(ni, nj) > 0.99f);
}

// Linear index i*n + j of the non-culled pair with the given rank
// (row via binary search over the row offsets, column via warp ballots).
// Warp-collective; rank must be < total number of non-culled pairs.
__device__ uint64_t locatePairOfRank(const float* __restrict__ tris, const uint64_t* __restrict__ rowOffset,
                                     uint32_t n, uint64_t rank) {
    const int lane = threadIdx.x & 31;
    uint32_t lo = 0, hi = n;  // largest row with rowOffset[row] <= rank
    while (hi - lo > 1) {
        uint32_t mid = (lo + hi) / 2;
        if (rowOffset[mid] <= rank) lo = mid; else hi = mid;
    }
    const uint32_t row = lo;
    const uint64_t local = rank - rowOffset[row];
    uint64_t seen = 0;
    for (uint32_t j0 = 0;; j0 += 32) {
        uint32_t j = j0 + lane;
        bool ok = j < n && pairActive(tris, row, j);
        unsigned mask = __ballot_sync(0xffffffffu, ok);
        uint32_t cnt = __popc(mask);
        if (seen + cnt > local) {
            unsigned target = static_cast<unsigned>(local - seen);
            unsigned before = __popc(mask & ((1u << lane) - 1u));
            unsigned hit = __ballot_sync(0xffffffffu, ok && before == target);
            return static_cast<uint64_t>(row) * n + j0 + (__ffs(hit) - 1);
        }
        seen += cnt;
    }
}

// One warp per requested rank (ranks >= totalActive map to n*n)
__global__ void locatePairsKernel(const float* __restrict__ tris, const uint64_t* __restrict__ rowOffset,
                                  uint32_t n, uint64_t totalActive, const uint64_t* __restrict__ ranks,
                                  int count, uint64_t* __restrict__ linear) {
    int k = blockIdx.x;
    if (k >= count) return;
    uint64_t r = ranks[k];
    uint64_t lin = r < totalActive ? locatePairOfRank(tris, rowOffset, n, r) : static_cast<uint64_t>(n) * n;
    if (threadIdx.x == 0) linear[k] = lin;
}

// Ray contribution to Kij (0 when the ray is skipped). Warp-collective: all
// 32 lanes must call it; lanes with valid == false only help the traversal.
__device__ float rayContribution(bool valid, const float* __restrict__ tris,
                                 const GpuNode* __restrict__ nodes,
                                 const GpuLeafTri* __restrict__ leafTris,
                                 int i, int j, float4 r, PacketEntry* stack, int* rankLane,
                                 unsigned* sharedBlocked) {
    const float* ti = tris + i * TRI_STRIDE;
    const float* tj = tris + j * TRI_STRIDE;

    float u = r.x, v = r.y;
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    DVec pI = {__fmaf_rn(v, ti[6], __fmaf_rn(u, ti[3], ti[0])),
               __fmaf_rn(v, ti[7], __fmaf_rn(u, ti[4], ti[1])),
               __fmaf_rn(v, ti[8], __fmaf_rn(u, ti[5], ti[2]))};
    u = r.z; v = r.w;
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    DVec pJ = {__fmaf_rn(v, tj[6], __fmaf_rn(u, tj[3], tj[0])),
               __fmaf_rn(v, tj[7], __fmaf_rn(u, tj[4], tj[1])),
               __fmaf_rn(v, tj[8], __fmaf_rn(u, tj[5], tj[2]))};

    DVec dir = dsub(pJ, pI);
    float distSqr = ddot(dir, dir);
    float rayLen = sqrtf(distSqr);
    valid = valid && !(rayLen < EPSILON);  // degenerate rays count as blocked
    DVec dirNorm = {dir.x / rayLen, dir.y / rayLen, dir.z / rayLen};
    bool blocked = isRayBlockedPacket(valid, pI, pJ, dirNorm, rayLen, nodes, leafTris, i, j, stack,
                                      rankLane, sharedBlocked);
    if (!valid || blocked) return 0.0f;

    if (distSqr < EPSILON) return 0.0f;
    float cosPhiI = ZERO, cosPhiJ = ZERO;
    if (rayLen > EPSILON) {
        float ci = ddot(dir, ldVec(ti + 9)) / rayLen;
        cosPhiI = ZERO < ci ? ci : ZERO;
        DVec ndir = {-dir.x, -dir.y, -dir.z};
        float cj = ddot(ndir, ldVec(tj + 9)) / rayLen;
        cosPhiJ = ZERO < cj ? cj : ZERO;
    }
    if (cosPhiI <= ZERO || cosPhiJ <= ZERO) return 0.0f;
    return (cosPhiI * cosPhiJ) / (PI * distSqr);
}

// ============================================================================
// Form Factor Kernel
// ============================================================================

constexpr int FF_TWISTS = 8;                                 // MT blocks per chunk
constexpr int FF_NUMS = FF_TWISTS * MT_N;                    // 4992 random numbers
constexpr int FF_RAYS = FF_NUMS / 4;                         // 1248 rays
constexpr int FF_PAIRS = FF_RAYS / NUM_RAYS;                 // 78 pairs
constexpr int FF_THREADS = 416;                              // 13 warps
constexpr int FF_MIN_BLOCKS = 2;  // resident blocks per SM
static_assert(FF_PAIRS * NUM_RAYS * 4 == FF_NUMS, "chunk must align with MT blocks");
static_assert(FF_THREADS >= MT_N - MT_M, "twist needs at least 227 threads");
static_assert(FF_THREADS % 32 == 0 && FF_THREADS >= FF_PAIRS, "whole warps; one thread per pair");

__global__ void __launch_bounds__(FF_THREADS, FF_MIN_BLOCKS)
formFactorKernel(const float* __restrict__ tris, const GpuNode* __restrict__ nodes,
                 const GpuLeafTri* __restrict__ leafTris, const float* __restrict__ areas,
                 const uint64_t* __restrict__ rowOffset, const uint32_t* __restrict__ mtStates,
                 uint32_t n, uint64_t totalActive, uint64_t chunksPerBlock, int blockBase,
                 float* __restrict__ coefRow,
                 unsigned long long* __restrict__ nonZeroCount) {
    __shared__ uint32_t mt[MT_N];
    __shared__ __align__(16) float rnd[FF_NUMS];
    __shared__ uint32_t pairI[FF_PAIRS], pairJ[FF_PAIRS];
    __shared__ unsigned int blockNonZero;
    __shared__ int nextPacket;
    __shared__ PacketEntry packetStacks[FF_THREADS / 32][OCT_STACK];
    __shared__ int rankLanes[FF_THREADS / 32][32];
    __shared__ unsigned warpBlocked[FF_THREADS / 32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    PacketEntry* warpStack = packetStacks[tid >> 5];
    const uint64_t nn = static_cast<uint64_t>(n) * n;
    const int block = blockBase + static_cast<int>(blockIdx.x);
    const uint64_t rankBegin = static_cast<uint64_t>(block) * chunksPerBlock * FF_PAIRS;
    if (rankBegin >= totalActive) return;
    const uint64_t rankLimit = rankBegin + chunksPerBlock * FF_PAIRS;
    const uint64_t rankEnd = rankLimit < totalActive ? rankLimit : totalActive;

    for (int k = tid; k < MT_N; k += blockDim.x)
        mt[k] = mtStates[static_cast<size_t>(block) * MT_N + k];
    if (tid == 0) { blockNonZero = 0; nextPacket = 0; }

    // Warp 0 locates the first pair
    uint64_t cursor = 0;  // linear index i*n + j of the next candidate pair
    if (tid < 32) cursor = locatePairOfRank(tris, rowOffset, n, rankBegin);

    bool fresh = true;  // the loaded window already holds the next 624 outputs
    for (uint64_t rank = rankBegin; rank < rankEnd; rank += FF_PAIRS) {
        const int npairs = rankEnd - rank < FF_PAIRS ? static_cast<int>(rankEnd - rank) : FF_PAIRS;

        // Random numbers for this chunk (always whole MT blocks)
        for (int g = 0; g < FF_TWISTS; ++g) {
            __syncthreads();
            if (!fresh) mtTwistShared(mt);
            fresh = false;
            for (int k = tid; k < MT_N; k += blockDim.x)
                rnd[g * MT_N + k] = mtToUnitFloat(mtTemper(mt[k]));
        }

        // Pairs for this chunk
        if (tid < 32) {
            int got = 0;
            while (got < npairs) {
                uint64_t cand = cursor + lane;
                bool ok = false;
                uint32_t ci = 0, cj = 0;
                if (cand < nn) {
                    ci = static_cast<uint32_t>(cand / n);
                    cj = static_cast<uint32_t>(cand - static_cast<uint64_t>(ci) * n);
                    ok = pairActive(tris, ci, cj);
                }
                unsigned mask = __ballot_sync(0xffffffffu, ok);
                int pos = got + __popc(mask & ((1u << lane) - 1u));
                if (ok && pos < npairs) { pairI[pos] = ci; pairJ[pos] = cj; }
                int total = __popc(mask);
                if (got + total >= npairs) {
                    unsigned last = __ballot_sync(0xffffffffu, ok && pos == npairs - 1);
                    cursor = cursor + (__ffs(last) - 1) + 1;
                    got = npairs;
                } else {
                    got += total;
                    cursor += 32;
                }
            }
        }
        __syncthreads();

        // Rays: warps fetch packets (32 consecutive rays = 2 pairs) dynamically
        const int nrays = npairs * NUM_RAYS;
        const float4* rnd4 = reinterpret_cast<const float4*>(rnd);
        for (;;) {
            int packet = 0;
            if (lane == 0) packet = atomicAdd(&nextPacket, 1);
            packet = __shfl_sync(0xffffffffu, packet, 0);
            const int k = packet * 32 + lane;
            if (packet * 32 >= nrays) break;
            bool valid = k < nrays;
            int p = valid ? k / NUM_RAYS : 0;
            float c = rayContribution(valid, tris, nodes, leafTris, static_cast<int>(pairI[p]),
                                      static_cast<int>(pairJ[p]), rnd4[valid ? k : 0], warpStack,
                                      rankLanes[tid >> 5], &warpBlocked[tid >> 5]);
            if (valid) rnd[4 * k] = c;  // ray k's numbers are consumed; reuse the slot
        }
        __syncthreads();
        if (tid == 0) nextPacket = 0;  // next use is behind at least one more barrier

        // Sequential (ray-ordered) accumulation per pair
        if (tid < npairs) {
            float kij = ZERO;
            #pragma unroll
            for (int r = 0; r < NUM_RAYS; ++r) kij += rnd[4 * (tid * NUM_RAYS + r)];
            kij = kij * INV_NUM_RAYS;
            uint32_t i = pairI[tid], j = pairJ[tid];
            float w = 0.0f;
            if (kij > ZERO) {
                float ka = kij * areas[j];
                w = ka > ONE ? ONE : ka;
            }
            coefRow[static_cast<size_t>(i) * n + j] = w;
            if (kij > EPSILON) atomicAdd(&blockNonZero, 1u);
        }
    }
    __syncthreads();
    if (tid == 0 && blockNonZero) atomicAdd(nonZeroCount, static_cast<unsigned long long>(blockNonZero));
}

__global__ void rowCountKernel(const float* __restrict__ tris, uint32_t n, uint32_t* __restrict__ rowCount) {
    __shared__ uint32_t warpSums[32];
    uint32_t i = blockIdx.x;
    uint32_t cnt = 0;
    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) cnt += pairActive(tris, i, j) ? 1u : 0u;
    for (int off = 16; off > 0; off >>= 1) cnt += __shfl_down_sync(0xffffffffu, cnt, off);
    if ((threadIdx.x & 31) == 0) warpSums[threadIdx.x >> 5] = cnt;
    __syncthreads();
    if (threadIdx.x < 32) {
        uint32_t v = threadIdx.x < (blockDim.x + 31) / 32 ? warpSums[threadIdx.x] : 0;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (threadIdx.x == 0) rowCount[i] = v;
    }
}

// ============================================================================
// Tau (time delay) Kernel
// ============================================================================

// tauT[j * n + i] = tau(i, j); stored transposed for coalesced simulation reads.
// The room has radius 10, so delays are at most ceil(20 / WAVE_SPEED) = 40 and
// fit in 8 bits; a larger value would raise *overflow.
__global__ void tauKernel(const float* __restrict__ centers, uint32_t n, uint8_t* __restrict__ tauT,
                          int* __restrict__ overflow) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y;
    if (i >= n) return;
    int tau = 0;
    if (i != j) {
        DVec d = dsub(ldVec(centers + 3 * i), ldVec(centers + 3 * j));
        float dist = sqrtf(ddot(d, d));
        tau = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
        if (tau > 255) *overflow = 1;
    }
    tauT[static_cast<size_t>(j) * n + i] = static_cast<uint8_t>(tau);
}

// ============================================================================
// Simulation Kernel (one timestep; sequential j-order sum per receiver)
// ============================================================================

// Block = SIM_RECV receivers. All threads load and multiply tiles of
// coefficients (coalesced along the receiver index); warp 0 then accumulates
// the products of each receiver strictly in source order j = 0, 1, ..., which
// reproduces the reference summation exactly (skipped terms contribute +0).
constexpr int SIM_RECV = 32;
constexpr int SIM_TILE = 128;
constexpr int SIM_THREADS = 256;

// Per tile (SIM_TILE sources x SIM_RECV receivers): smallest and largest delay
// among the entries with a non-zero coefficient (255/0 if there are none)
__global__ void tileDelayRangeKernel(const float* __restrict__ coefT, const uint8_t* __restrict__ tauT,
                                     uint32_t n, uint8_t* __restrict__ tileMin, uint8_t* __restrict__ tileMax) {
    __shared__ int smin, smax;
    if (threadIdx.x == 0) { smin = 255; smax = 0; }
    __syncthreads();
    const uint32_t i0 = blockIdx.x * SIM_RECV, j0 = blockIdx.y * SIM_TILE;
    int lmin = 255, lmax = 0;
    for (int idx = threadIdx.x; idx < SIM_TILE * SIM_RECV; idx += blockDim.x) {
        uint32_t j = j0 + idx / SIM_RECV, i = i0 + idx % SIM_RECV;
        if (j < n && i < n) {
            size_t o = static_cast<size_t>(j) * n + i;
            if (coefT[o] > ZERO) {
                int tau = tauT[o];
                lmin = min(lmin, tau);
                lmax = max(lmax, tau);
            }
        }
    }
    atomicMin(&smin, lmin);
    atomicMax(&smax, lmax);
    __syncthreads();
    if (threadIdx.x == 0) {
        size_t tile = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
        tileMin[tile] = static_cast<uint8_t>(smin);
        tileMax[tile] = static_cast<uint8_t>(smax);
    }
}

__global__ void __launch_bounds__(SIM_THREADS)
simulationStepKernel(const float* __restrict__ coefT, const uint8_t* __restrict__ tauT,
                     const uint8_t* __restrict__ tileMin, const uint8_t* __restrict__ tileMax,
                     const float* __restrict__ rho, const float* __restrict__ radE,
                     float* __restrict__ radB, uint32_t n, int t) {
    __shared__ float prod[2][SIM_TILE][SIM_RECV];
    constexpr int PER_THREAD = SIM_TILE * SIM_RECV / SIM_THREADS;
    const int tid = threadIdx.x;
    const uint32_t i0 = blockIdx.x * SIM_RECV;
    float sumB = ZERO;
    int buf = 0;
    for (uint32_t j0 = 0, tile = blockIdx.x; j0 < n; j0 += SIM_TILE, tile += gridDim.x) {
        // Wave has not reached any receiver of this tile yet: all terms are +0
        if (t < tileMin[tile]) continue;
        const bool gate = t < tileMax[tile];  // otherwise every delay has elapsed
        float (*P)[SIM_RECV] = prod[buf];
        buf ^= 1;
        int tauv[PER_THREAD];
        float wv[PER_THREAD];
        #pragma unroll
        for (int e = 0; e < PER_THREAD; ++e) {
            int idx = tid + e * SIM_THREADS;
            uint32_t j = j0 + idx / SIM_RECV, i = i0 + idx % SIM_RECV;
            bool in = j < n && i < n;
            size_t o = static_cast<size_t>(j) * n + i;
            tauv[e] = in ? tauT[o] : 255;
            wv[e] = in && !gate ? coefT[o] : ZERO;
        }
        if (gate) {
            #pragma unroll
            for (int e = 0; e < PER_THREAD; ++e) {
                int idx = tid + e * SIM_THREADS;
                uint32_t j = j0 + idx / SIM_RECV, i = i0 + idx % SIM_RECV;
                if (t >= tauv[e]) wv[e] = coefT[static_cast<size_t>(j) * n + i];
            }
        }
        #pragma unroll
        for (int e = 0; e < PER_THREAD; ++e) {
            int idx = tid + e * SIM_THREADS;
            uint32_t j = j0 + idx / SIM_RECV;
            float p = ZERO;
            if (t >= tauv[e] && wv[e] > ZERO) p = wv[e] * radB[static_cast<size_t>(t - tauv[e]) * n + j];
            P[idx / SIM_RECV][idx % SIM_RECV] = p;
        }
        __syncthreads();
        if (tid < SIM_RECV) {
            #pragma unroll 16
            for (int jj = 0; jj < SIM_TILE; ++jj) sumB += P[jj][tid];
        }
    }
    uint32_t i = i0 + tid;
    if (tid < SIM_RECV && i < n) {
        size_t o = static_cast<size_t>(t) * n + i;
        radB[o] = __fmaf_rn(rho[i], sumB, radE[o]);
    }
}

// ============================================================================
// Distance Kernel (cross-correlation)
// ============================================================================

__global__ void distanceKernel(const float* __restrict__ radB, uint32_t n, int timesteps,
                               uint32_t src, float* __restrict__ distances) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float maxCorr = ZERO;
    int bestT = 0;
    for (int t = 0; t < timesteps; ++t) {
        float sum = ZERO;
        for (int tt = t; tt < timesteps; ++tt) {
            float pB = radB[static_cast<size_t>(tt) * n + i];
            float pS = radB[static_cast<size_t>(tt - t) * n + src];
            sum += pS * pB;
        }
        if (sum > maxCorr) { maxCorr = sum; bestT = t; }
    }
    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
}

// ============================================================================
// Transpose Kernel (row-major coefficients -> receiver-contiguous layout)
// ============================================================================

__global__ void transposeKernel(const float* __restrict__ in, float* __restrict__ out, uint32_t n) {
    __shared__ float tile[32][33];
    uint32_t x = blockIdx.x * 32 + threadIdx.x;
    uint32_t y0 = blockIdx.y * 32;
    for (uint32_t r = threadIdx.y; r < 32; r += blockDim.y) {
        uint32_t y = y0 + r;
        if (x < n && y < n) tile[r][threadIdx.x] = in[static_cast<size_t>(y) * n + x];
    }
    __syncthreads();
    uint32_t ox = y0 + threadIdx.x;
    for (uint32_t r = threadIdx.y; r < 32; r += blockDim.y) {
        uint32_t oy = blockIdx.x * 32 + r;
        if (ox < n && oy < n) out[static_cast<size_t>(oy) * n + ox] = tile[threadIdx.x][r];
    }
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

// Form factor work partition
constexpr int FF_WAVES = 2;                  // single GPU: blocks per resident block slot
constexpr int FF_SEGMENTS_PER_DEVICE = 8;    // interleaved block ranges per GPU (load balance)
constexpr int FF_MIN_CHUNKS_PER_BLOCK = 16;  // every block needs its own jumped generator state

// Blocks per segment: a power of two covering one wave per GPU (two waves on a
// single GPU, which has no segment interleaving)
static int ffSegmentBlocks(int slots, int numDevices) {
    int blocks = 1;
    while (blocks < (numDevices > 1 ? slots : FF_WAVES * slots)) blocks *= 2;
    return blocks;
}

// Pinned host staging for streaming coefficients from GPU d > 0 to GPU 0
constexpr size_t STAGING_FLOATS = size_t(4) << 20;  // 16 MB per buffer
constexpr int STAGING_BUFFERS = 2;

// Per-GPU data for the form factor computation
struct DeviceContext {
    int device = 0;
    int numSMs = 1;
    float* dTris = nullptr;         // a, b-a, c-a, normal per triangle
    float* dAreas = nullptr;
    GpuNode* dNodes = nullptr;
    GpuLeafTri* dLeafTris = nullptr;
    float* dCoefRow = nullptr;      // min(Kij * area_j, 1), row-major (N x N)
    cudaStream_t computeStreams[2] = {};  // alternating form factor segments
    cudaStream_t copyStream = nullptr;    // GPU d: device -> staging
    cudaStream_t uploadStream = nullptr;  // created on GPU 0: staging -> GPU 0
    float* staging[STAGING_BUFFERS] = {};
    cudaEvent_t stagingFilled[STAGING_BUFFERS] = {};   // recorded on GPU d
    cudaEvent_t stagingDrained[STAGING_BUFFERS] = {};  // recorded on GPU 0
    // Per-run inputs of the form factor kernel
    uint64_t* dRowOffset = nullptr;     // N + 1 pair-rank offsets
    uint32_t* dStates = nullptr;        // generator start state of every block
    uint16_t* dJumpTerms = nullptr;     // jump polynomial coefficient positions
    int* dGroupBase = nullptr;          // first block of each own segment (+ slot for 0)
    unsigned long long* dNonZero = nullptr;
    cudaEvent_t ready = nullptr;
    cudaEvent_t segDone[FF_SEGMENTS_PER_DEVICE] = {};
};

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source
    unsigned long long nonZeroKij = 0;  // Form factors > EPSILON

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // GPUs sharing the form factor work; devices[0] also runs everything else
    std::vector<DeviceContext> devices;
    int ffSlots = 1;                // resident form factor blocks per GPU (smallest GPU)
    int ffMaxBlocks = 1;            // upper bound of the form factor block count
    float* dCenters = nullptr;
    float* dRho = nullptr;
    float* dRadE = nullptr;
    float* dRadB = nullptr;
    float* dDistances = nullptr;
    float* dCoefT = nullptr;        // min(Kij * area_j, 1), transposed (N x N)
    uint8_t* dTauT = nullptr;       // time delays, transposed (N x N)
    uint8_t* dTileMin = nullptr;    // per simulation tile: delay range of non-zero coefficients
    uint8_t* dTileMax = nullptr;
    int* dTauOverflow = nullptr;    // set if a delay does not fit in 8 bits

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

static int octreeDepth(const Octree& o) {
    int depth = 0;
    for (const auto& c : o.children)
        if (c) depth = std::max(depth, octreeDepth(*c));
    return depth + 1;
}

static void flattenOctree(const Octree& root, std::vector<GpuNode>& nodes,
                          std::vector<GpuLeafTri>& leafTris) {
    if (octreeDepth(root) > OCT_MAX_DEPTH) {
        fprintf(stderr, "Octree too deep for the GPU traversal stack\n");
        exit(1);
    }
    std::vector<const Octree*> order = {&root};
    nodes.assign(1, GpuNode{});
    for (size_t k = 0; k < order.size(); ++k) {
        const Octree* o = order[k];
        int first = 0, count = 0;
        if (!o->triangleIndices.empty()) {
            first = static_cast<int>(leafTris.size());
            count = -static_cast<int>(o->triangleIndices.size());
            for (size_t idx : o->triangleIndices) {
                const Triangle& t = (*o->allTriangles)[idx];
                Vec3 e1 = t.b - t.a, e2 = t.c - t.a;
                GpuLeafTri lt;
                lt.v0i = make_float4(t.a.x, t.a.y, t.a.z, __builtin_bit_cast(float, static_cast<int>(idx)));
                lt.e1 = make_float4(e1.x, e1.y, e1.z, 0.0f);
                lt.e2 = make_float4(e2.x, e2.y, e2.z, 0.0f);
                leafTris.push_back(lt);
            }
        } else {
            first = static_cast<int>(order.size());
            for (int c = 0; c < 8; ++c) {
                if (o->children[c]) { order.push_back(o->children[c].get()); ++count; }
            }
            nodes.resize(order.size());
        }
        GpuNode& g = nodes[k];
        g.box = make_float4(o->center.x, o->center.y, o->center.z, o->halfExtent.x);
        g.info = make_float4(o->halfExtent.y, o->halfExtent.z, __builtin_bit_cast(float, first),
                             __builtin_bit_cast(float, count));
    }
}

// Meshes below this size do not benefit from spreading the work over GPUs
constexpr size_t MULTI_GPU_MIN_TRIANGLES = 1280;

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Device data
    const size_t n = state.numTriangles;
    std::vector<float> tris(n * TRI_STRIDE), centers(n * 3);
    for (size_t i = 0; i < n; ++i) {
        const Triangle& t = state.triangles[i];
        Vec3 ab = t.b - t.a, ac = t.c - t.a, nr = t.normal(), c = t.center();
        float* p = &tris[i * TRI_STRIDE];
        p[0] = t.a.x; p[1] = t.a.y; p[2] = t.a.z;
        p[3] = ab.x;  p[4] = ab.y;  p[5] = ab.z;
        p[6] = ac.x;  p[7] = ac.y;  p[8] = ac.z;
        p[9] = nr.x;  p[10] = nr.y; p[11] = nr.z;
        centers[3 * i] = c.x; centers[3 * i + 1] = c.y; centers[3 * i + 2] = c.z;
    }
    std::vector<GpuNode> nodes;
    std::vector<GpuLeafTri> leafTris;
    flattenOctree(state.octree, nodes, leafTris);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        fprintf(stderr, "No CUDA device available\n");
        exit(1);
    }
    int useDevices = n >= MULTI_GPU_MIN_TRIANGLES ? deviceCount : 1;
    int blocksPerSM = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, formFactorKernel, FF_THREADS, 0));
    state.ffSlots = std::numeric_limits<int>::max();
    for (int d = 0; d < useDevices; ++d) {
        int sms = 1;
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, d));
        state.ffSlots = std::min(state.ffSlots, std::max(1, blocksPerSM) * sms);
    }
    state.ffMaxBlocks = useDevices * (useDevices > 1 ? FF_SEGMENTS_PER_DEVICE : 1) *
                        ffSegmentBlocks(state.ffSlots, useDevices);
    int maxRounds = 0;
    while ((1 << maxRounds) < state.ffMaxBlocks) ++maxRounds;
    for (int d = 0; d < useDevices; ++d) {
        DeviceContext ctx;
        ctx.device = d;
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaFree(0));
        CUDA_CHECK(cudaDeviceGetAttribute(&ctx.numSMs, cudaDevAttrMultiProcessorCount, d));
        ctx.dTris = deviceUpload(tris);
        ctx.dAreas = deviceUpload(state.areas);
        ctx.dNodes = deviceUpload(nodes);
        ctx.dLeafTris = deviceUpload(leafTris);
        CUDA_CHECK(cudaFuncSetAttribute(mtJumpKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(MT_SEQ_LEN * sizeof(uint32_t))));
        CUDA_CHECK(cudaMalloc(&ctx.dCoefRow, n * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx.dRowOffset, (n + 1) * sizeof(uint64_t)));
        // Also scratch for the row counts and segment ranks before the states are built
        const size_t statesBytes = std::max(static_cast<size_t>(state.ffMaxBlocks) * MT_N * sizeof(uint32_t),
                                            n * sizeof(uint32_t) + 4 * (FF_SEGMENTS_PER_DEVICE * deviceCount + 1) *
                                                                       sizeof(uint64_t));
        CUDA_CHECK(cudaMalloc(&ctx.dStates, statesBytes));
        CUDA_CHECK(cudaMalloc(&ctx.dJumpTerms, std::max(1, maxRounds) * static_cast<size_t>(MT_DEG) * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&ctx.dGroupBase, (FF_SEGMENTS_PER_DEVICE + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ctx.dNonZero, sizeof(unsigned long long)));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.ready, cudaEventDisableTiming));
        for (auto& e : ctx.segDone) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        for (auto& st : ctx.computeStreams) CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        if (d > 0) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.copyStream, cudaStreamNonBlocking));
            for (int b = 0; b < STAGING_BUFFERS; ++b) {
                CUDA_CHECK(cudaHostAlloc(&ctx.staging[b], STAGING_FLOATS * sizeof(float), cudaHostAllocPortable));
                CUDA_CHECK(cudaEventCreateWithFlags(&ctx.stagingFilled[b], cudaEventDisableTiming));
            }
        }
        state.devices.push_back(ctx);
    }
    CUDA_CHECK(cudaSetDevice(0));
    for (int d = 1; d < useDevices; ++d) {
        DeviceContext& ctx = state.devices[d];
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.uploadStream, cudaStreamNonBlocking));
        for (int b = 0; b < STAGING_BUFFERS; ++b)
            CUDA_CHECK(cudaEventCreateWithFlags(&ctx.stagingDrained[b], cudaEventDisableTiming));
    }

    CUDA_CHECK(cudaSetDevice(0));
    state.dCenters = deviceUpload(centers);
    state.dRho = deviceUpload(state.rho);
    state.dRadE = deviceUpload(state.radE);
    state.dRadB = deviceUpload(state.radB);
    state.dDistances = deviceUpload(state.distances);
    CUDA_CHECK(cudaMalloc(&state.dCoefT, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dTauT, n * n * sizeof(uint8_t)));
    const size_t simTiles = ((n + SIM_RECV - 1) / SIM_RECV) * ((n + SIM_TILE - 1) / SIM_TILE);
    CUDA_CHECK(cudaMalloc(&state.dTileMin, simTiles));
    CUDA_CHECK(cudaMalloc(&state.dTileMax, simTiles));
    CUDA_CHECK(cudaMalloc(&state.dTauOverflow, sizeof(int)));
    CUDA_CHECK(cudaMemset(state.dTauOverflow, 0, sizeof(int)));
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const uint32_t n = static_cast<uint32_t>(state.numTriangles);
    const size_t nn = static_cast<size_t>(n) * n;
    const int numDevices = static_cast<int>(state.devices.size());
    DeviceContext& dev0 = state.devices[0];
    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t st0 = dev0.computeStreams[0];

    // Rank of every non-culled pair (row offsets of the RNG stream)
    uint32_t* dRowCount = reinterpret_cast<uint32_t*>(dev0.dStates);  // scratch
    rowCountKernel<<<n, 256, 0, st0>>>(dev0.dTris, n, dRowCount);
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint32_t> rowCount(n);
    CUDA_CHECK(cudaMemcpyAsync(rowCount.data(), dRowCount, n * sizeof(uint32_t), cudaMemcpyDeviceToHost, st0));
    CUDA_CHECK(cudaStreamSynchronize(st0));
    std::vector<uint64_t> rowOffset(n + 1, 0);
    for (uint32_t i = 0; i < n; ++i) rowOffset[i + 1] = rowOffset[i] + rowCount[i];
    const uint64_t totalActive = rowOffset[n];

    // Work partition: each block owns a contiguous range of chunks. Blocks are
    // grouped into equal segments (a power-of-two number of blocks each) that
    // are dealt round-robin to the GPUs; fewer/smaller segments for small
    // problems, since every block needs its own jumped generator state.
    const uint64_t totalChunks = (totalActive + FF_PAIRS - 1) / FF_PAIRS;
    int segBlocks = ffSegmentBlocks(state.ffSlots, numDevices);
    int segsPerDevice = numDevices > 1 ? FF_SEGMENTS_PER_DEVICE : 1;
    auto tooFine = [&]() {
        return static_cast<uint64_t>(numDevices) * segsPerDevice * segBlocks * FF_MIN_CHUNKS_PER_BLOCK > totalChunks;
    };
    while (segsPerDevice > 1 && tooFine()) segsPerDevice /= 2;
    while (segBlocks > 32 && tooFine()) segBlocks /= 2;
    const int numSegments = numDevices * segsPerDevice;
    const int numBlocks = numSegments * segBlocks;
    const uint64_t chunksPerBlock =
        std::max<uint64_t>(1, (totalChunks + numBlocks - 1) / static_cast<uint64_t>(numBlocks));
    const size_t statesBytes = static_cast<size_t>(numBlocks) * MT_N * sizeof(uint32_t);

    // First linear pair index of every segment (GPU 0, same predicate as the kernel)
    CUDA_CHECK(cudaMemcpyAsync(dev0.dRowOffset, rowOffset.data(), (n + 1) * sizeof(uint64_t),
                               cudaMemcpyHostToDevice, st0));
    std::vector<uint64_t> segLinear(numSegments + 1, nn);
    {
        std::vector<uint64_t> ranks(numSegments + 1);
        for (int g = 0; g <= numSegments; ++g)
            ranks[g] = static_cast<uint64_t>(g) * segBlocks * chunksPerBlock * FF_PAIRS;
        uint64_t* dRanks = reinterpret_cast<uint64_t*>(dev0.dStates);  // scratch
        uint64_t* dLinear = dRanks + ranks.size();
        CUDA_CHECK(cudaMemcpyAsync(dRanks, ranks.data(), ranks.size() * sizeof(uint64_t),
                                   cudaMemcpyHostToDevice, st0));
        locatePairsKernel<<<numSegments + 1, 32, 0, st0>>>(dev0.dTris, dev0.dRowOffset, n, totalActive, dRanks,
                                                           numSegments + 1, dLinear);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(segLinear.data(), dLinear, ranks.size() * sizeof(uint64_t),
                                   cudaMemcpyDeviceToHost, st0));
        CUDA_CHECK(cudaStreamSynchronize(st0));
        segLinear[0] = 0;
        segLinear[numSegments] = nn;
    }

    // MT19937(42) after the first twist: window holding raw outputs 0..623
    std::vector<uint32_t> base(MT_N);
    base[0] = 42u;
    for (int k = 1; k < MT_N; ++k) base[k] = 1812433253u * (base[k - 1] ^ (base[k - 1] >> 30)) + k;
    hostTwist(base.data());

    // Jump polynomials: poly r advances the generator by 2^r blocks
    // (block b starts at stream offset b * chunksPerBlock * FF_NUMS)
    int rounds = 0;
    while ((1 << rounds) < numBlocks) ++rounds;
    JumpTerms jt;
    {
        MTJumpPoly jp;
        std::vector<MTJumpPoly::Poly> polys;
        if (rounds > 0) polys.push_back(jp.powX(chunksPerBlock * FF_NUMS));
        for (int r = 1; r < rounds; ++r) polys.push_back(jp.square(polys.back()));
        jt = buildJumpTerms(polys);
    }
    int segRounds = 0;
    while ((1 << segRounds) < segBlocks) ++segRounds;

    // Segment start states on GPU 0 (doubling over segments)
    CUDA_CHECK(cudaMemsetAsync(dev0.dStates, 0, statesBytes, st0));
    CUDA_CHECK(cudaMemcpyAsync(dev0.dStates, base.data(), MT_N * sizeof(uint32_t), cudaMemcpyHostToDevice, st0));
    CUDA_CHECK(cudaMemcpyAsync(dev0.dJumpTerms, jt.terms.data(), jt.terms.size() * sizeof(uint16_t),
                               cudaMemcpyHostToDevice, st0));
    const int zero = 0;
    CUDA_CHECK(cudaMemcpyAsync(dev0.dGroupBase + FF_SEGMENTS_PER_DEVICE, &zero, sizeof(int),
                               cudaMemcpyHostToDevice, st0));
    mtFillByDoubling(dev0.dStates, dev0.dGroupBase + FF_SEGMENTS_PER_DEVICE, 1, numSegments, segBlocks,
                     dev0.dJumpTerms, jt, segRounds, dev0.numSMs, st0);
    std::vector<uint32_t> segStates;
    if (numDevices > 1) {
        segStates.resize(static_cast<size_t>(numSegments) * MT_N);
        CUDA_CHECK(cudaMemcpy2DAsync(segStates.data(), MT_N * sizeof(uint32_t), dev0.dStates,
                                     static_cast<size_t>(segBlocks) * MT_N * sizeof(uint32_t),
                                     MT_N * sizeof(uint32_t), numSegments, cudaMemcpyDeviceToHost, st0));
        CUDA_CHECK(cudaStreamSynchronize(st0));
    }

    // Every GPU is driven by its own host thread: inputs, block start states of
    // its segments (doubling within segments), segment launches, and (d > 0)
    // streaming of the finished segments to GPU 0 through pinned staging.
    auto runDevice = [&](int d) {
        DeviceContext& ctx = state.devices[d];
        CUDA_CHECK(cudaSetDevice(ctx.device));
        cudaStream_t st = ctx.computeStreams[0];
        if (d > 0) {
            CUDA_CHECK(cudaMemcpyAsync(ctx.dRowOffset, rowOffset.data(), (n + 1) * sizeof(uint64_t),
                                       cudaMemcpyHostToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(ctx.dJumpTerms, jt.terms.data(), jt.terms.size() * sizeof(uint16_t),
                                       cudaMemcpyHostToDevice, st));
            CUDA_CHECK(cudaMemsetAsync(ctx.dStates, 0, statesBytes, st));
            for (int g = d; g < numSegments; g += numDevices)
                CUDA_CHECK(cudaMemcpyAsync(ctx.dStates + static_cast<size_t>(g) * segBlocks * MT_N,
                                           segStates.data() + static_cast<size_t>(g) * MT_N,
                                           MT_N * sizeof(uint32_t), cudaMemcpyHostToDevice, st));
        }
        CUDA_CHECK(cudaMemsetAsync(ctx.dCoefRow, 0, nn * sizeof(float), st));
        CUDA_CHECK(cudaMemsetAsync(ctx.dNonZero, 0, sizeof(unsigned long long), st));
        std::vector<int> own;
        for (int g = d; g < numSegments; g += numDevices) own.push_back(g * segBlocks);
        CUDA_CHECK(cudaMemcpyAsync(ctx.dGroupBase, own.data(), own.size() * sizeof(int),
                                   cudaMemcpyHostToDevice, st));
        mtFillByDoubling(ctx.dStates, ctx.dGroupBase, static_cast<int>(own.size()), segBlocks, 1,
                         ctx.dJumpTerms, jt, 0, ctx.numSMs, st);
        CUDA_CHECK(cudaEventRecord(ctx.ready, st));
        CUDA_CHECK(cudaStreamWaitEvent(ctx.computeStreams[1], ctx.ready, 0));

        for (int k = 0; k < segsPerDevice; ++k) {
            const int g = d + k * numDevices;
            cudaStream_t ks = ctx.computeStreams[k % 2];
            formFactorKernel<<<segBlocks, FF_THREADS, 0, ks>>>(
                ctx.dTris, ctx.dNodes, ctx.dLeafTris, ctx.dAreas, ctx.dRowOffset, ctx.dStates, n,
                totalActive, chunksPerBlock, g * segBlocks, ctx.dCoefRow, ctx.dNonZero);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(ctx.segDone[k], ks));
        }
        // The host arrays above are only read by asynchronous copies
        CUDA_CHECK(cudaStreamSynchronize(st));
        if (d == 0) return;

        int nextBuffer = 0;
        bool bufferUsed[STAGING_BUFFERS] = {};
        for (int k = 0; k < segsPerDevice; ++k) {
            const int g = d + k * numDevices;
            CUDA_CHECK(cudaSetDevice(ctx.device));
            CUDA_CHECK(cudaStreamWaitEvent(ctx.copyStream, ctx.segDone[k], 0));
            for (uint64_t off = segLinear[g]; off < segLinear[g + 1]; off += STAGING_FLOATS) {
                const size_t count = std::min<uint64_t>(STAGING_FLOATS, segLinear[g + 1] - off);
                const int b = nextBuffer;
                nextBuffer = (b + 1) % STAGING_BUFFERS;
                CUDA_CHECK(cudaSetDevice(ctx.device));
                if (bufferUsed[b]) CUDA_CHECK(cudaStreamWaitEvent(ctx.copyStream, ctx.stagingDrained[b], 0));
                CUDA_CHECK(cudaMemcpyAsync(ctx.staging[b], ctx.dCoefRow + off, count * sizeof(float),
                                           cudaMemcpyDeviceToHost, ctx.copyStream));
                CUDA_CHECK(cudaEventRecord(ctx.stagingFilled[b], ctx.copyStream));
                CUDA_CHECK(cudaSetDevice(0));
                CUDA_CHECK(cudaStreamWaitEvent(ctx.uploadStream, ctx.stagingFilled[b], 0));
                CUDA_CHECK(cudaMemcpyAsync(dev0.dCoefRow + off, ctx.staging[b], count * sizeof(float),
                                           cudaMemcpyHostToDevice, ctx.uploadStream));
                CUDA_CHECK(cudaEventRecord(ctx.stagingDrained[b], ctx.uploadStream));
                bufferUsed[b] = true;
            }
        }
    };
    {
        std::vector<std::thread> threads;
        for (int d = 1; d < numDevices; ++d) threads.emplace_back(runDevice, d);
        runDevice(0);
        for (auto& t : threads) t.join();
    }

    // Wait for everything
    state.nonZeroKij = 0;
    for (int d = 0; d < numDevices; ++d) {
        DeviceContext& ctx = state.devices[d];
        CUDA_CHECK(cudaSetDevice(ctx.device));
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned long long cnt = 0;
        CUDA_CHECK(cudaMemcpy(&cnt, ctx.dNonZero, sizeof(cnt), cudaMemcpyDeviceToHost));
        state.nonZeroKij += cnt;
    }

    // Receiver-contiguous layout for the simulation
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceSynchronize());
    dim3 tgrid((n + 31) / 32, (n + 31) / 32);
    transposeKernel<<<tgrid, dim3(32, 8)>>>(dev0.dCoefRow, state.dCoefT, n);
    CUDA_CHECK(cudaGetLastError());
    dim3 rgrid((n + SIM_RECV - 1) / SIM_RECV, (n + SIM_TILE - 1) / SIM_TILE);
    tileDelayRangeKernel<<<rgrid, 256>>>(state.dCoefT, state.dTauT, n, state.dTileMin, state.dTileMax);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (size_t i = 0; i < state.numTriangles; ++i) {
        if ((i + 1) % 100 == 0 || i + 1 == state.numTriangles) {
            printf("  Progress: %zu/%zu triangles\n", i + 1, state.numTriangles);
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const uint32_t n = static_cast<uint32_t>(state.numTriangles);
    CUDA_CHECK(cudaSetDevice(0));
    dim3 grid((n + 255) / 256, n);
    tauKernel<<<grid, 256>>>(state.dCenters, n, state.dTauT, state.dTauOverflow);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const uint32_t n = static_cast<uint32_t>(state.numTriangles);
    CUDA_CHECK(cudaSetDevice(0));
    int tauOverflow = 0;
    CUDA_CHECK(cudaMemcpy(&tauOverflow, state.dTauOverflow, sizeof(int), cudaMemcpyDeviceToHost));
    if (tauOverflow) {
        fprintf(stderr, "Time delay exceeds the 8-bit delay storage\n");
        exit(1);
    }
    const int blocks = (n + SIM_RECV - 1) / SIM_RECV;

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        simulationStepKernel<<<blocks, SIM_THREADS>>>(state.dCoefT, state.dTauT, state.dTileMin, state.dTileMax,
                                                      state.dRho, state.dRadE, state.dRadB, n, static_cast<int>(t));
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dRadB, state.radB.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const uint32_t n = static_cast<uint32_t>(state.numTriangles);
    distanceKernel<<<(n + 127) / 128, 128>>>(state.dRadB, n, static_cast<int>(state.numTimesteps),
                                             static_cast<uint32_t>(state.sourceIndex), state.dDistances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.dDistances, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t d = state.distances[i];
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
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    int receivedEnergy = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    int nonZeroKij = static_cast<int>(state.nonZeroKij);
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           nonZeroKij, state.numTriangles * state.numTriangles,
           100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));

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

uint64_t computeHash(const SimulationState& state) {
    uint64_t hash = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&state.distances[i]);
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
    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    printf("Room Response Simulation Benchmark\n");
    printf("===================================\n");
    printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    printf("Distance computation time: %ld ms\n", distDuration);
    printf("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    printf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
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

    // Hash
    uint64_t hash = computeHash(state);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    // Print results for external validation
    if (printResults) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    if (validate) {
        if (!validateResults(state)) {
            return 1;
        }
    }

    return 0;
}
