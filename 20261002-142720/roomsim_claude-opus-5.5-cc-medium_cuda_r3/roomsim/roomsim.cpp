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
#include <cfloat>
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

#define HD __host__ __device__ inline

// a*b + c*d and a*b - c*d. On the device the fused multiply-adds are spelled out
// in the order the host compiler contracts these expressions (c*d rounded first).
HD val_t mulAdd(val_t a, val_t b, val_t c, val_t d) {
#ifdef __CUDA_ARCH__
    return fmaf(a, b, c * d);
#else
    return a * b + c * d;
#endif
}

HD val_t mulSub(val_t a, val_t b, val_t c, val_t d) {
#ifdef __CUDA_ARCH__
    return fmaf(a, b, -(c * d));
#else
    return a * b - c * d;
#endif
}

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    HD constexpr Vec3() : x(0), y(0), z(0) {}
    HD constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    HD explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    HD Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    HD Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    HD Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    HD Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    HD Vec3 operator-() const { return {-x, -y, -z}; }

    HD val_t dot(const Vec3& o) const {
#ifdef __CUDA_ARCH__
        return fmaf(z, o.z, fmaf(x, o.x, y * o.y));
#else
        return x * o.x + y * o.y + z * o.z;
#endif
    }
    HD Vec3 cross(const Vec3& o) const {
        return {mulSub(y, o.z, z, o.y), mulSub(z, o.x, x, o.z), mulSub(x, o.y, y, o.x)};
    }

    HD val_t squaredNorm() const { return dot(*this); }
    HD val_t norm() const { return sqrtf(squaredNorm()); }
    HD Vec3 normalized() const {
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
// CUDA Helpers
// ============================================================================

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

template <typename T>
T* deviceAlloc(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, std::max<size_t>(count, 1) * sizeof(T)));
    return ptr;
}

template <typename T>
T* deviceUpload(const std::vector<T>& host) {
    T* ptr = deviceAlloc<T>(host.size());
    if (!host.empty()) {
        CUDA_CHECK(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    return ptr;
}

constexpr unsigned FULL_MASK = 0xffffffffu;

// ============================================================================
// Random Number Generation (exact reproduction of the sequential stream)
// ============================================================================
//
// The reference draws every random number from a single std::mt19937(42) via
// std::uniform_real_distribution<float>(0, 1). Each non-culled pair (i, j),
// visited in row-major order, consumes exactly 4 * NUM_RAYS numbers (u, v for
// the point on triangle i, then u, v for the point on triangle j, per ray).
//
// To reproduce that stream in parallel, it is cut into chunks of MT_CHUNK
// outputs. The generator state at every chunk start is obtained on the GPU via
// polynomial jump-ahead: A^J = (t^J mod phi)(A), where phi is the characteristic
// polynomial of the MT19937 transition. Each chunk is then generated
// independently by one thread block.

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr int MT_DIFF = MT_N - MT_M;                  // x[n] = x[n-227] ^ twist(x[n-624], x[n-623])
constexpr uint32_t MT_MATRIX_A = 0x9908b0dfu;
constexpr uint32_t MT_SEED = 42;
constexpr int MT_DEG = 19937;                         // degree of the characteristic polynomial
constexpr int MT_POLY_WORDS = (MT_DEG + 31) / 32;     // 32-bit words of a reduced polynomial
constexpr int MT_JUMP_SEQ = MT_DEG + MT_N;            // sequence length needed for one jump
constexpr int MT_JUMP_THREADS = 1024;
constexpr int MT_LOG_CHUNK = 18;
constexpr uint64_t MT_CHUNK = 1ull << MT_LOG_CHUNK;   // outputs per chunk
constexpr int MT_GEN_THREADS = 256;
constexpr int MT_RING = 1024;                         // ring buffer (>= MT_N + MT_DIFF, power of 2)
constexpr int VALS_PER_PAIR = 4 * NUM_RAYS;
constexpr int CHUNKS_PER_BATCH = 128;                 // 2^25 random numbers (128 MB) per batch
constexpr int MT_MAX_JUMP_POLYS = 24;                 // supports up to 2^24 chunks
static_assert(MT_CHUNK % VALS_PER_PAIR == 0, "chunks must hold whole pairs");

__host__ __device__ inline uint32_t mtTwist(uint32_t a, uint32_t b) {
    uint32_t y = (a & 0x80000000u) | (b & 0x7fffffffu);
    return (y >> 1) ^ ((y & 1u) ? MT_MATRIX_A : 0u);
}

__host__ __device__ inline uint32_t mtTemper(uint32_t y) {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

// libstdc++ generate_canonical<float, 24>(mt19937): float(x) / 2^32, clamped below 1
__host__ __device__ inline float mtToUniform(uint32_t y) {
#ifdef __CUDA_ARCH__
    float r = __uint2float_rn(y) * (1.0f / 4294967296.0f);
#else
    float r = static_cast<float>(y) * (1.0f / 4294967296.0f);
#endif
    return r >= 1.0f ? 0x1.fffffep-1f : r;
}

// Seeded state x[0..623] (std::mt19937 seeding)
static std::vector<uint32_t> mtSeedState(uint32_t seed) {
    std::vector<uint32_t> x(MT_N);
    x[0] = seed;
    for (uint32_t i = 1; i < MT_N; ++i) {
        x[i] = 1812433253u * (x[i - 1] ^ (x[i - 1] >> 30)) + i;
    }
    return x;
}

// Verifies that seeding, recurrence, tempering and float conversion above
// reproduce std::uniform_real_distribution<float>(0, 1) over std::mt19937.
static void mtVerifyStream() {
    constexpr int count = 4 * MT_N;
    std::vector<uint32_t> x = mtSeedState(MT_SEED);
    x.resize(MT_N + count);
    std::mt19937 gen(MT_SEED);
    std::uniform_real_distribution<val_t> dist(0.0f, 1.0f);
    for (int n = MT_N; n < MT_N + count; ++n) {
        x[n] = x[n - MT_DIFF] ^ mtTwist(x[n - MT_N], x[n - MT_N + 1]);
        const float expected = dist(gen);
        const float actual = mtToUniform(mtTemper(x[n]));
        if (std::memcmp(&expected, &actual, sizeof(float)) != 0) {
            fprintf(stderr, "Random number stream reproduction check failed\n");
            std::exit(EXIT_FAILURE);
        }
    }
}

// Polynomials over GF(2), little-endian bit order in 64-bit words
using Gf2Poly = std::vector<uint64_t>;

// Exponents of the characteristic polynomial phi of the MT19937 transition
// (135 terms, obtained with Berlekamp-Massey on the output sequence).
constexpr int MT_CHAR_POLY_TERMS[] = {
    0, 1189, 1416, 1585, 1643, 1870, 2493, 2773, 3000, 3227, 3454, 3681, 3908, 4135, 4362, 4753,
    5661, 6337, 6569, 7129, 7477, 7525, 7583, 7752, 7979, 8206, 9505, 9901, 9969, 10128, 10693,
    10761, 10920, 11089, 11147, 11157, 11215, 11321, 11374, 11384, 11485, 11611, 11712, 11717,
    11838, 11881, 11944, 11997, 12277, 12335, 12393, 12504, 12509, 12620, 12673, 12731, 12736,
    12789, 12905, 12958, 12963, 13137, 13185, 13190, 13243, 13301, 13412, 13528, 13533, 13639,
    13697, 13760, 13813, 13866, 14093, 14151, 14209, 14320, 14325, 14436, 14547, 14552, 14605,
    14721, 14774, 14779, 14953, 15001, 15006, 15059, 15117, 15228, 15344, 15349, 15455, 15513,
    15576, 15629, 15682, 15909, 15967, 16025, 16136, 16141, 16252, 16363, 16368, 16421, 16537,
    16590, 16595, 16817, 16822, 16875, 16933, 17044, 17160, 17271, 17329, 17445, 17498, 17725,
    17783, 17841, 17952, 18068, 18179, 18237, 18406, 18633, 18691, 18860, 19087, 19314, 19937
};

// phi(t) as a bit polynomial; verifies that phi annihilates the output sequence
// of std::mt19937, i.e. XOR over the terms e of y[n + e] = 0.
static Gf2Poly mtCharacteristicPolynomial() {
    Gf2Poly phi(MT_DEG / 64 + 1, 0);
    for (int e : MT_CHAR_POLY_TERMS) phi[e >> 6] |= 1ull << (e & 63);

    constexpr int checks = 64;
    std::vector<uint32_t> y(MT_DEG + checks);
    std::mt19937 gen(MT_SEED);
    for (auto& v : y) v = gen();
    for (int n = 0; n < checks; ++n) {
        uint32_t acc = 0;
        for (int e : MT_CHAR_POLY_TERMS) acc ^= y[n + e];
        if (acc != 0) {
            fprintf(stderr, "MT19937 characteristic polynomial check failed\n");
            std::exit(EXIT_FAILURE);
        }
    }
    return phi;
}

// Arithmetic modulo phi (degree MT_DEG). phi is sparse (135 terms for
// MT19937), so reduction folds blocks of high bits onto the lower terms.
class Gf2Modulus {
    static constexpr int PW = (MT_DEG + 63) / 64;   // words of a reduced polynomial
    static constexpr int MAX_BLOCK = 640;            // bits folded at once
    static constexpr int BLOCK_WORDS = MAX_BLOCK / 64 + 1;
    std::vector<int> terms;                          // exponents e < MT_DEG with phi_e = 1
    int blockBits;                                   // <= MT_DEG - max(terms)

    static uint64_t spread32(uint64_t x) {
        x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
        x = (x | (x << 8)) & 0x00FF00FF00FF00FFull;
        x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0Full;
        x = (x | (x << 2)) & 0x3333333333333333ull;
        x = (x | (x << 1)) & 0x5555555555555555ull;
        return x;
    }

public:
    explicit Gf2Modulus(const Gf2Poly& phi) {
        for (int k = 0; k < MT_DEG; ++k) {
            if ((phi[k >> 6] >> (k & 63)) & 1) terms.push_back(k);
        }
        const int gap = MT_DEG - (terms.empty() ? 0 : terms.back());
        blockBits = std::min(gap, MAX_BLOCK);
    }

    // a^2 mod phi
    Gf2Poly square(const Gf2Poly& a) const {
        Gf2Poly r(2 * PW + BLOCK_WORDS + 2, 0);
        for (int w = 0; w < PW; ++w) {
            r[2 * w] = spread32(a[w] & 0xffffffffull);
            r[2 * w + 1] = spread32(a[w] >> 32);
        }
        // Fold bits [k0, kTop] (top-down): t^(MT_DEG + s) = sum_e t^(e + s). With
        // blockBits <= MT_DEG - max(e), the folded bits land strictly below k0.
        int topWord = 2 * PW - 1;
        while (topWord >= 0 && r[topWord] == 0) --topWord;
        uint64_t h[BLOCK_WORDS + 1];
        for (int kTop = std::min(2 * MT_DEG - 2, 64 * topWord + 63); kTop >= MT_DEG;) {
            const int k0 = std::max(MT_DEG, kTop - blockBits + 1);
            const int len = kTop - k0 + 1;
            const int hw = (len + 63) / 64;
            for (int i = 0; i < hw; ++i) {
                const int pos = k0 + 64 * i;
                const int w = pos >> 6, o = pos & 63;
                h[i] = (r[w] >> o) | (o ? r[w + 1] << (64 - o) : 0);
            }
            if (len & 63) h[hw - 1] &= (1ull << (len & 63)) - 1;
            r[k0 >> 6] &= (1ull << (k0 & 63)) - 1;
            for (int w = (k0 >> 6) + 1; w <= (kTop >> 6); ++w) r[w] = 0;

            for (int e : terms) {
                const int shift = k0 - MT_DEG + e;
                const int ws = shift >> 6, bs = shift & 63;
                uint64_t* dst = r.data() + ws;
                if (bs == 0) {
                    for (int i = 0; i < hw; ++i) dst[i] ^= h[i];
                } else {
                    for (int i = 0; i < hw; ++i) {
                        dst[i] ^= h[i] << bs;
                        dst[i + 1] ^= h[i] >> (64 - bs);
                    }
                }
            }
            kTop = k0 - 1;
        }
        r.resize(PW);
        return r;
    }

    // a * t mod phi
    Gf2Poly mulT(const Gf2Poly& a) const {
        Gf2Poly r(PW, 0);
        for (int w = PW - 1; w > 0; --w) r[w] = (a[w] << 1) | (a[w - 1] >> 63);
        r[0] = a[0] << 1;
        if ((r[MT_DEG >> 6] >> (MT_DEG & 63)) & 1) {
            r[MT_DEG >> 6] ^= 1ull << (MT_DEG & 63);
            for (int e : terms) r[e >> 6] ^= 1ull << (e & 63);
        }
        return r;
    }

    // t^e mod phi
    Gf2Poly powT(unsigned long long e) const {
        Gf2Poly r = monomial(0);
        for (int b = e ? 63 - __builtin_clzll(e) : -1; b >= 0; --b) {
            r = square(r);
            if ((e >> b) & 1) r = mulT(r);
        }
        return r;
    }

    static Gf2Poly monomial(int k) {
        Gf2Poly r(PW, 0);
        r[k >> 6] = 1ull << (k & 63);
        return r;
    }
};

// Successive jump polynomials P_b = t^(MT_CHUNK * 2^b) mod phi, b = 0, 1, ...
class MtJumpPolynomials {
    Gf2Modulus mod;
    Gf2Poly p;
    int index = 0;

    static void store(const Gf2Poly& poly, uint32_t* dst) {
        for (int w = 0; w < MT_POLY_WORDS; ++w) {
            dst[w] = static_cast<uint32_t>(poly[w >> 1] >> ((w & 1) * 32));
        }
    }

public:
    explicit MtJumpPolynomials(const Gf2Poly& phi) : mod(phi), p(Gf2Modulus::monomial(1)) {
        for (int k = 0; k < MT_LOG_CHUNK; ++k) p = mod.square(p);
    }

    // Writes the next polynomial as MT_POLY_WORDS 32-bit words
    void next(uint32_t* dst) {
        if (index++ > 0) p = mod.square(p);
        store(p, dst);
    }

    // Writes t^(MT_CHUNK * chunks) mod phi (jump by a number of chunks)
    void chunkJump(unsigned long long chunks, uint32_t* dst) const {
        store(mod.powT(chunks * MT_CHUNK), dst);
    }
};

// One jump per block: states[q + half] = g(A) states[q] for the jump polynomial
// g (half = 0 jumps states[0] in place).
// Element p of g(A) x is the XOR over set bits k of g of x[k + p]; each warp
// handles a slice of the polynomial for all 624 outputs.
constexpr int MT_OUT_PER_LANE = (MT_N + 31) / 32;
constexpr size_t MT_JUMP_SMEM = (MT_JUMP_SEQ + MT_POLY_WORDS + MT_N) * sizeof(uint32_t);

__global__ void __launch_bounds__(MT_JUMP_THREADS)
mtJumpKernel(uint32_t* __restrict__ states, const uint32_t* __restrict__ poly,
             uint32_t half, uint32_t numChunks) {
    extern __shared__ uint32_t smem[];
    uint32_t* seq = smem;                         // x[0 .. MT_JUMP_SEQ)
    uint32_t* pw = seq + MT_JUMP_SEQ;             // polynomial bits
    uint32_t* result = pw + MT_POLY_WORDS;        // jumped state
    const uint32_t q = blockIdx.x;
    const uint32_t dst = q + half;
    if (dst >= numChunks) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;

    const uint32_t* src = states + static_cast<size_t>(q) * MT_N;
    for (int k = tid; k < MT_N; k += blockDim.x) {
        seq[k] = src[k];
        result[k] = 0;
    }
    for (int k = tid; k < MT_POLY_WORDS; k += blockDim.x) pw[k] = poly[k];
    __syncthreads();

    for (int base = MT_N; base < MT_JUMP_SEQ; base += MT_DIFF) {
        int n = base + tid;
        if (tid < MT_DIFF && n < MT_JUMP_SEQ) {
            seq[n] = seq[n - MT_DIFF] ^ mtTwist(seq[n - MT_N], seq[n - MT_N + 1]);
        }
        __syncthreads();
    }

    uint32_t acc[MT_OUT_PER_LANE];
#pragma unroll
    for (int m = 0; m < MT_OUT_PER_LANE; ++m) acc[m] = 0;

    const int numWarps = blockDim.x / 32;
    const int w1 = (warp + 1) * MT_POLY_WORDS / numWarps;
    for (int w = warp * MT_POLY_WORDS / numWarps; w < w1; ++w) {
        uint32_t bits = pw[w];
        while (bits) {
            const int b = __ffs(bits) - 1;
            bits &= bits - 1;
            const uint32_t* s = seq + w * 32 + b + lane;
#pragma unroll
            for (int m = 0; m < MT_OUT_PER_LANE; ++m) {
                if (lane + 32 * m < MT_N) acc[m] ^= s[32 * m];
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MT_OUT_PER_LANE; ++m) {
        if (lane + 32 * m < MT_N) atomicXor(&result[lane + 32 * m], acc[m]);
    }
    __syncthreads();
    for (int k = tid; k < MT_N; k += blockDim.x) states[static_cast<size_t>(dst) * MT_N + k] = result[k];
}

// One chunk per block: writes MT_CHUNK uniform floats of chunk (chunk0 + blockIdx.x)
__global__ void __launch_bounds__(MT_GEN_THREADS)
mtGenerateKernel(const uint32_t* __restrict__ states, uint32_t chunk0, float* __restrict__ out) {
    __shared__ uint32_t ring[MT_RING];
    const int tid = threadIdx.x;
    const uint32_t* src = states + static_cast<size_t>(chunk0 + blockIdx.x) * MT_N;
    for (int k = tid; k < MT_N; k += blockDim.x) ring[k] = src[k];
    __syncthreads();

    float* o = out + static_cast<size_t>(blockIdx.x) * MT_CHUNK;
    for (uint32_t base = MT_N; base < MT_N + MT_CHUNK; base += MT_DIFF) {
        uint32_t n = base + tid;
        if (tid < MT_DIFF && n < MT_N + MT_CHUNK) {
            uint32_t v = ring[(n - MT_DIFF) & (MT_RING - 1)] ^
                         mtTwist(ring[(n - MT_N) & (MT_RING - 1)], ring[(n - MT_N + 1) & (MT_RING - 1)]);
            ring[n & (MT_RING - 1)] = v;
            o[n - MT_N] = mtToUniform(mtTemper(v));
        }
        __syncthreads();
    }
}

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

// Edges e1 = v1 - v0 and e2 = v2 - v0 are precomputed per triangle
HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                              const Vec3& v0, const Vec3& e1, const Vec3& e2) {
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Flattened Octree (GPU traversal)
// ============================================================================
//
// Node k occupies nodes[2k] = (center.xyz, halfExtent.x) and
// nodes[2k+1] = (halfExtent.yz, start, count). Leaves have count > 0 triangles
// starting at leafTris[3 * start]; inner nodes have -count children stored
// contiguously starting at node index start (in child order 0..7).

static float intBitsToFloat(int v) {
    float f;
    std::memcpy(&f, &v, sizeof(f));
    return f;
}

struct FlatOctree {
    std::vector<float4> nodes;
    std::vector<float4> leafTris;   // a (w = triangle index), b - a, c - a per entry
    int depth = 0;                  // levels below the root
};

static void flattenOctreeNode(const Octree& node, size_t slot, int level, FlatOctree& flat,
                              const std::vector<Triangle>& triangles) {
    flat.depth = std::max(flat.depth, level);
    int start = 0, count = 0;
    if (!node.triangleIndices.empty()) {
        start = static_cast<int>(flat.leafTris.size() / 3);
        count = static_cast<int>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) {
            const Triangle& t = triangles[idx];
            float4 a = make_float4(t.a.x, t.a.y, t.a.z, 0.0f);
            a.w = intBitsToFloat(static_cast<int>(idx));
            const Vec3 e1 = t.b - t.a;
            const Vec3 e2 = t.c - t.a;
            flat.leafTris.push_back(a);
            flat.leafTris.push_back(make_float4(e1.x, e1.y, e1.z, 0.0f));
            flat.leafTris.push_back(make_float4(e2.x, e2.y, e2.z, 0.0f));
        }
    } else {
        start = static_cast<int>(flat.nodes.size() / 2);
        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) ++count;
        }
        flat.nodes.resize(flat.nodes.size() + 2 * count);
        int k = 0;
        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) {
                flattenOctreeNode(*node.children[i], start + k, level + 1, flat, triangles);
                ++k;
            }
        }
        count = -count;
    }
    flat.nodes[2 * slot] = make_float4(node.center.x, node.center.y, node.center.z, node.halfExtent.x);
    flat.nodes[2 * slot + 1] = make_float4(node.halfExtent.y, node.halfExtent.z,
                                           intBitsToFloat(start), intBitsToFloat(count));
}

static FlatOctree flattenOctree(const Octree& root, const std::vector<Triangle>& triangles) {
    FlatOctree flat;
    flat.nodes.resize(2);
    flattenOctreeNode(root, 0, 0, flat, triangles);
    return flat;
}

// Same test as Octree::rayIntersectsBox; mid = p1 + d, ad = |d|
__device__ inline bool rayIntersectsBoxGPU(const Vec3& d, const Vec3& mid, const Vec3& ad,
                                           const float4 n0, const float4 n1) {
    const Vec3 h(n0.w, n1.x, n1.y);
    const Vec3 c = mid - Vec3(n0.x, n0.y, n0.z);

    if (fabsf(c.x) > h.x + ad.x) return false;
    if (fabsf(c.y) > h.y + ad.y) return false;
    if (fabsf(c.z) > h.z + ad.z) return false;

    if (fabsf(mulSub(d.y, c.z, d.z, c.y)) > mulAdd(h.y, ad.z, h.z, ad.y) + EPSILON) return false;
    if (fabsf(mulSub(d.z, c.x, d.x, c.z)) > mulAdd(h.z, ad.x, h.x, ad.z) + EPSILON) return false;
    if (fabsf(mulSub(d.x, c.y, d.y, c.x)) > mulAdd(h.x, ad.y, h.y, ad.x) + EPSILON) return false;

    return true;
}

// Traversal stack entries; a depth-first walk holds at most 7 * depth + 1
constexpr int OCTREE_STACK = 128;

struct RayQuery {
    Vec3 from;
    Vec3 dirNorm;
    val_t maxDist;      // rayLen - EPSILON
    Vec3 d, mid, ad;    // segment half-vector, midpoint and |d| for the box tests
    int srcTriIdx, dstTriIdx;
};

// Warp-cooperative (packet) version of isRayBlocked: every lane traces its own
// ray, but the warp walks the octree together using a shared stack of
// (node, lane mask) entries. A lane takes part in a node only if all box tests
// on the path to it succeeded for its own ray, so each lane tests exactly the
// triangles the reference traversal would test for that ray.
// Must be called by all 32 lanes of the warp; 'active' lanes need a test.
__device__ bool isRayBlockedWarp(bool active, const RayQuery& q,
                                 const float4* __restrict__ nodes,
                                 const float4* __restrict__ leafTris,
                                 int2* __restrict__ stack) {
    const int lane = threadIdx.x & 31;
    const uint32_t laneBit = 1u << lane;
    bool blocked = false;

    uint32_t rootMask = __ballot_sync(FULL_MASK, active);
    if (rootMask == 0) return false;
    int sp = 0;
    if (lane == 0) stack[0] = make_int2(0, static_cast<int>(rootMask));
    sp = 1;
    __syncwarp();

    while (sp > 0) {
        --sp;
        const int2 entry = stack[sp];
        __syncwarp();
        const int ni = entry.x;
        const uint32_t m = static_cast<uint32_t>(entry.y) & __ballot_sync(FULL_MASK, !blocked);
        if (m == 0) continue;
        const bool mine = (m & laneBit) != 0;
        const float4 n1 = __ldg(&nodes[2 * ni + 1]);
        const int start = __float_as_int(n1.z);
        const int count = __float_as_int(n1.w);
        if (count > 0) {
            const float4* t = leafTris + 3 * start;
            for (int k = 0; k < count; ++k, t += 3) {
                const float4 a = __ldg(t);
                const int idx = __float_as_int(a.w);
                if (!mine || blocked || idx == q.srcTriIdx || idx == q.dstTriIdx) continue;
                const float4 e1 = __ldg(t + 1);
                const float4 e2 = __ldg(t + 2);
                val_t dist = rayTriangleIntersect(q.from, q.dirNorm, Vec3(a.x, a.y, a.z),
                                                  Vec3(e1.x, e1.y, e1.z), Vec3(e2.x, e2.y, e2.z));
                if (dist > EPSILON && dist < q.maxDist) blocked = true;
            }
        } else {
            for (int k = 0; k < -count; ++k) {
                const int ci = start + k;
                const float4 c0 = __ldg(&nodes[2 * ci]);
                const float4 c1 = __ldg(&nodes[2 * ci + 1]);
                const bool hit = mine && rayIntersectsBoxGPU(q.d, q.mid, q.ad, c0, c1);
                const uint32_t cm = __ballot_sync(FULL_MASK, hit);
                if (cm != 0) {
                    if (lane == 0) stack[sp] = make_int2(ci, static_cast<int>(cm));
                    ++sp;
                }
            }
            __syncwarp();
        }
    }
    return blocked;
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
HD val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// Generate a point inside a triangle from barycentric coordinates (u, v)
HD Vec3 pointInTriangle(const Vec3& a, const Vec3& b, const Vec3& c, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = b - a;
    Vec3 ac = c - a;
    return a + ab * u + ac * v;
}

struct DeviceTriangles {
    const float4* a;
    const float4* b;
    const float4* c;
    const float4* n;      // normal
    const float* area;
};

__device__ inline Vec3 loadVec3(const float4* p, int i) {
    float4 v = __ldg(p + i);
    return Vec3(v.x, v.y, v.z);
}

// 16 consecutive lanes evaluate the 16 rays of one pair (i, j); lane 0 of the
// group accumulates them in ray order. Writes w_ij = min(K_ij * A_j, 1) (the
// weight used by the simulation) per pair, in pair order.
constexpr int FF_THREADS = 256;
static_assert(NUM_RAYS == 16, "one half-warp per pair");

__global__ void __launch_bounds__(FF_THREADS)
formFactorKernel(const uint2* __restrict__ pairs, uint32_t numPairs,
                 const float4* __restrict__ rnd, DeviceTriangles tris,
                 const float4* __restrict__ nodes, const float4* __restrict__ leafTris,
                 float* __restrict__ wPair, unsigned long long* __restrict__ nonZeroCount) {
    const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pl = gtid / NUM_RAYS;
    const int r = gtid % NUM_RAYS;
    const bool valid = pl < numPairs;

    __shared__ int2 stacks[FF_THREADS / 32][OCTREE_STACK];

    val_t contrib = ZERO;
    uint32_t i = 0, j = 0;
    bool trace = false;
    RayQuery q;
    if (valid) {
        const uint2 pr = __ldg(&pairs[pl]);
        i = pr.x;
        j = pr.y;
        const float4 rr = __ldg(&rnd[static_cast<size_t>(pl) * NUM_RAYS + r]);
        const Vec3 aI = loadVec3(tris.a, i), bI = loadVec3(tris.b, i), cI = loadVec3(tris.c, i);
        const Vec3 aJ = loadVec3(tris.a, j), bJ = loadVec3(tris.b, j), cJ = loadVec3(tris.c, j);
        const Vec3 pI = pointInTriangle(aI, bI, cI, rr.x, rr.y);
        const Vec3 pJ = pointInTriangle(aJ, bJ, cJ, rr.z, rr.w);

        // Unblocked contribution (the visibility test is only needed if non-zero)
        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr >= EPSILON) {
            val_t cosPhiI = cosPhi(v, loadVec3(tris.n, i));
            val_t cosPhiJ = cosPhi(-v, loadVec3(tris.n, j));
            if (cosPhiI > ZERO && cosPhiJ > ZERO) {
                contrib = (cosPhiI * cosPhiJ) / (PI * distSqr);
                // Ray setup as in isRayBlocked / Octree::rayIntersectsBox
                val_t rayLen = v.norm();
                if (rayLen < EPSILON) {
                    contrib = ZERO;
                } else {
                    trace = true;
                    q.from = pI;
                    q.dirNorm = v / rayLen;
                    q.maxDist = rayLen - EPSILON;
                    q.d = v * 0.5f;
                    q.mid = pI + q.d;
                    q.ad = Vec3(fabsf(q.d.x), fabsf(q.d.y), fabsf(q.d.z));
                    q.srcTriIdx = static_cast<int>(i);
                    q.dstTriIdx = static_cast<int>(j);
                }
            }
        }
    }
    if (isRayBlockedWarp(trace, q, nodes, leafTris, stacks[threadIdx.x / 32])) contrib = ZERO;

    // Sum the rays in their original order
    val_t kij = ZERO;
#pragma unroll
    for (int k = 0; k < NUM_RAYS; ++k) {
        kij += __shfl_sync(FULL_MASK, contrib, k, NUM_RAYS);
    }

    __shared__ uint32_t blockNonZero;
    if (threadIdx.x == 0) blockNonZero = 0;
    __syncthreads();
    bool nonZero = false;
    if (valid && r == 0) {
        kij *= INV_NUM_RAYS;
        nonZero = kij > EPSILON;
        val_t w = ZERO;
        if (kij > ZERO) w = fminf(kij * __ldg(&tris.area[j]), ONE);
        wPair[pl] = w;
    }
    const uint32_t nzBits = __ballot_sync(FULL_MASK, nonZero);
    if ((threadIdx.x & 31) == 0 && nzBits) atomicAdd(&blockNonZero, static_cast<uint32_t>(__popc(nzBits)));
    __syncthreads();
    if (threadIdx.x == 0 && blockNonZero) atomicAdd(nonZeroCount, static_cast<unsigned long long>(blockNonZero));
}

// Bit j of row i is set if the pair (i, j) is not culled (i != j and the
// triangles do not face the same direction); counts[i] = set bits in row i.
constexpr int MASK_THREADS = 256;

__global__ void __launch_bounds__(MASK_THREADS)
cullMaskKernel(const float4* __restrict__ normals, uint32_t N, uint32_t maskWords,
               uint32_t* __restrict__ mask, uint32_t* __restrict__ counts) {
    const uint32_t i = blockIdx.x;
    const uint32_t j = blockIdx.y * blockDim.x + threadIdx.x;
    bool active = false;
    if (j < N && j != i) {
        active = !(loadVec3(normals, i).dot(loadVec3(normals, j)) > 0.99f);
    }
    const uint32_t bits = __ballot_sync(FULL_MASK, active);
    const uint32_t w = j / 32;
    if ((threadIdx.x & 31) == 0 && w < maskWords) {
        mask[static_cast<size_t>(i) * maskWords + w] = bits;
        if (counts && bits) atomicAdd(&counts[i], static_cast<uint32_t>(__popc(bits)));
    }
}

// Enumerates the non-culled pairs (i, j) of row i (one block per row) together
// with their global pair index pos, calling op(i, j, pos) for each.
constexpr int ROW_THREADS = 256;

template <typename Op>
__device__ void forEachPairInRow(const uint32_t* __restrict__ mask, uint32_t maskWords,
                                 uint32_t i, unsigned long long rowStart, Op op) {
    __shared__ uint32_t warpSums[ROW_THREADS / 32];
    const uint32_t* m = mask + static_cast<size_t>(i) * maskWords;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    unsigned long long running = rowStart;

    for (uint32_t w0 = 0; w0 < maskWords; w0 += ROW_THREADS) {
        const uint32_t w = w0 + tid;
        uint32_t bits = w < maskWords ? m[w] : 0u;
        const uint32_t cnt = __popc(bits);
        uint32_t incl = cnt;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            uint32_t y = __shfl_up_sync(FULL_MASK, incl, o);
            if (lane >= o) incl += y;
        }
        if (lane == 31) warpSums[wid] = incl;
        __syncthreads();
        uint32_t warpOff = 0, total = 0;
        for (int k = 0; k < ROW_THREADS / 32; ++k) {
            uint32_t s = warpSums[k];
            if (k < wid) warpOff += s;
            total += s;
        }
        unsigned long long pos = running + warpOff + incl - cnt;
        while (bits) {
            int b = __ffs(bits) - 1;
            bits &= bits - 1;
            op(i, w * 32 + b, pos);
            ++pos;
        }
        running += total;
        __syncthreads();
    }
}

// Lists the pairs with global pair index in [p0, p1)
__global__ void __launch_bounds__(ROW_THREADS)
compactPairsKernel(const uint32_t* __restrict__ mask, uint32_t maskWords,
                   const unsigned long long* __restrict__ rowStart, uint32_t row0,
                   unsigned long long p0, unsigned long long p1, uint2* __restrict__ pairs) {
    const uint32_t i = row0 + blockIdx.x;
    forEachPairInRow(mask, maskWords, i, rowStart[i], [&](uint32_t pi, uint32_t pj, unsigned long long pos) {
        if (pos >= p0 && pos < p1) pairs[pos - p0] = make_uint2(pi, pj);
    });
}

// Row-major weight block of rows [row0, row0 + gridDim.x):
// w[(i - row0) * N + j] = wPair[pos(i, j) - pairBase], zero for culled pairs.
__global__ void __launch_bounds__(ROW_THREADS)
expandWeightsKernel(const uint32_t* __restrict__ mask, uint32_t maskWords,
                    const unsigned long long* __restrict__ rowStart, uint32_t row0,
                    const float* __restrict__ wPair, unsigned long long pairBase,
                    float* __restrict__ w, uint32_t N) {
    __shared__ uint32_t warpSums[ROW_THREADS / 32];
    const uint32_t i = row0 + blockIdx.x;
    const uint32_t* m = mask + static_cast<size_t>(i) * maskWords;
    float* row = w + static_cast<size_t>(blockIdx.x) * N;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    unsigned long long running = rowStart[i] - pairBase;

    for (uint32_t j0 = 0; j0 < N; j0 += ROW_THREADS) {
        const uint32_t j = j0 + tid;
        const bool set = j < N && ((m[j >> 5] >> (j & 31)) & 1u);
        const uint32_t bits = __ballot_sync(FULL_MASK, set);
        if (lane == 0) warpSums[wid] = __popc(bits);
        __syncthreads();
        uint32_t warpOff = 0, total = 0;
        for (int k = 0; k < ROW_THREADS / 32; ++k) {
            const uint32_t c = warpSums[k];
            if (k < wid) warpOff += c;
            total += c;
        }
        if (j < N) {
            const uint32_t before = warpOff + __popc(bits & ((1u << lane) - 1u));
            row[j] = set ? wPair[running + before] : ZERO;
        }
        running += total;
        __syncthreads();
    }
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

HD int computeTau(const Vec3& centerI, const Vec3& centerJ) {
    val_t dist = (centerI - centerJ).norm();
    return static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// Row-major delay block of rows [row0, row0 + gridDim.x). With the room radius
// of 10 units, tau <= ceil(20 / WAVE_SPEED) = 40 fits in 8 bits.
__global__ void tauKernel(const float4* __restrict__ centers, uint8_t* __restrict__ tau,
                          uint32_t N, uint32_t row0) {
    const uint32_t j = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t i = row0 + blockIdx.x;
    if (j >= N) return;
    int tauij = 0;
    if (i != j) tauij = computeTau(loadVec3(centers, i), loadVec3(centers, j));
    tau[static_cast<size_t>(blockIdx.x) * N + j] = static_cast<uint8_t>(tauij);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

// Number of random numbers / pairs processed per batch on one GPU
constexpr unsigned long long BATCH_VALS = CHUNKS_PER_BATCH * MT_CHUNK;
constexpr unsigned long long BATCH_PAIRS = BATCH_VALS / VALS_PER_PAIR;
constexpr unsigned long long CHUNK_PAIRS = MT_CHUNK / VALS_PER_PAIR;
// Each additional GPU must receive at least this many batches of form factor work
constexpr unsigned long long MIN_BATCHES_PER_GPU = 2;

// Per-GPU state. Every GPU owns a contiguous block of receiving triangles
// (rows i of the N x N matrices): it computes the form factors of (about) these
// rows, stores their weights and delays row-major, and simulates them. The
// radiosity history radB is replicated and all-gathered after every timestep.
struct DeviceContext {
    int device = 0;
    uint32_t rowBegin = 0, rowEnd = 0;      // owned rows [rowBegin, rowEnd)
    // Geometry
    float4* triA = nullptr;
    float4* triB = nullptr;
    float4* triC = nullptr;
    float4* triN = nullptr;
    float4* centers = nullptr;
    float* areas = nullptr;
    float4* octNodes = nullptr;
    float4* octLeafTris = nullptr;
    // Owned rows of the simulation matrices and the replicated radiosity
    float* w = nullptr;                     // min(Kij * A_j, 1), row-major (rows x N)
    uint8_t* tau = nullptr;                 // time delays, row-major (rows x N)
    float* radB = nullptr;                  // reflected radiosity (T x N)
    // Form factor workspace
    cudaStream_t genStream = nullptr, rayStream = nullptr;
    cudaEvent_t genDone[2] = {}, rayDone[2] = {}, stepDone = nullptr;
    uint32_t* mask = nullptr;               // culling mask (N x maskWords)
    unsigned long long* rowStart = nullptr; // first pair index of each row (N + 1)
    uint32_t* polys = nullptr;              // jump polynomials
    uint32_t* states = nullptr;             // generator state at each chunk start
    float* rnd[2] = {};                     // uniform random numbers of a batch
    uint2* pairs[2] = {};                   // (i, j) of the pairs of a batch
    float* wPair = nullptr;                 // weights of the computed / owned pairs
    unsigned long long* nonZero = nullptr;
};

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    val_t rho;                      // Reflectivity (0.0 to 1.0), identical for all triangles
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source
    unsigned long long nonZeroKij = 0;

    Octree octree;                  // Spatial acceleration structure
    std::vector<DeviceContext> ctx; // ctx[0] is the primary GPU
    uint32_t* counts = nullptr;     // non-culled pairs per row (primary GPU)
    float* corr = nullptr;          // cross-correlation T x N (primary GPU)
    float* devDistances = nullptr;  // (primary GPU)
    float* hostRadB = nullptr;      // pinned T x N radiosity (multi-GPU all-gather)

    size_t sourceIndex;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

// Upper bounds of the form factor work (no pair culled)
struct FormFactorBounds {
    unsigned long long pairs, chunks, batches;
    explicit FormFactorBounds(size_t n)
        : pairs(static_cast<unsigned long long>(n) * (n - 1)),
          chunks((pairs * VALS_PER_PAIR + MT_CHUNK - 1) / MT_CHUNK),
          batches((chunks + CHUNKS_PER_BATCH - 1) / CHUNKS_PER_BATCH) {}
};

static int selectDeviceCount(const FormFactorBounds& bounds) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        fprintf(stderr, "No CUDA device available\n");
        std::exit(EXIT_FAILURE);
    }
    const unsigned long long useful = std::max<unsigned long long>(1, bounds.batches / MIN_BATCHES_PER_GPU);
    return static_cast<int>(std::min<unsigned long long>(deviceCount, useful));
}

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

    const size_t n = state.numTriangles;

    // Initialize areas
    state.areas.resize(n);
    for (size_t i = 0; i < n; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho = reflectivity;

    state.radB.resize(timesteps * n, ZERO);
    state.distances.resize(n, ZERO);

    // Host-side geometry in GPU layout
    std::vector<float4> a(n), b(n), c(n), nrm(n), centers(n);
    for (size_t i = 0; i < n; ++i) {
        const Triangle& t = state.triangles[i];
        Vec3 ct = t.center();
        a[i] = make_float4(t.a.x, t.a.y, t.a.z, 0.0f);
        b[i] = make_float4(t.b.x, t.b.y, t.b.z, 0.0f);
        c[i] = make_float4(t.c.x, t.c.y, t.c.z, 0.0f);
        nrm[i] = make_float4(t._normal.x, t._normal.y, t._normal.z, 0.0f);
        centers[i] = make_float4(ct.x, ct.y, ct.z, 0.0f);
    }
    FlatOctree flat = flattenOctree(state.octree, state.triangles);
    if (7 * flat.depth + 1 > OCTREE_STACK) {
        fprintf(stderr, "Octree too deep for the traversal stack\n");
        std::exit(EXIT_FAILURE);
    }

    // Create the GPU contexts concurrently, upload the geometry and allocate the
    // per-GPU matrices and form factor workspace (sized for no culled pairs)
    const FormFactorBounds bounds(n);
    const int numDevices = selectDeviceCount(bounds);
    int maxRounds = 0;
    while ((1ull << maxRounds) < bounds.chunks) ++maxRounds;
    if (maxRounds > MT_MAX_JUMP_POLYS) {
        fprintf(stderr, "Too many random numbers required\n");
        std::exit(EXIT_FAILURE);
    }
    const uint32_t maskWords = (static_cast<uint32_t>(n) + 31) / 32;

    state.ctx.resize(numDevices);
    std::vector<std::thread> workers;
    for (int g = 0; g < numDevices; ++g) {
        workers.emplace_back([&, g]() {
            DeviceContext& cx = state.ctx[g];
            cx.device = g;
            cx.rowBegin = static_cast<uint32_t>(n * g / numDevices);
            cx.rowEnd = static_cast<uint32_t>(n * (g + 1) / numDevices);
            const size_t rows = cx.rowEnd - cx.rowBegin;
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaFree(nullptr));
            cx.triA = deviceUpload(a);
            cx.triB = deviceUpload(b);
            cx.triC = deviceUpload(c);
            cx.triN = deviceUpload(nrm);
            cx.centers = deviceUpload(centers);
            cx.areas = deviceUpload(state.areas);
            cx.octNodes = deviceUpload(flat.nodes);
            cx.octLeafTris = deviceUpload(flat.leafTris);
            cx.w = deviceAlloc<float>(rows * n);
            cx.tau = deviceAlloc<uint8_t>(rows * n);
            cx.radB = deviceAlloc<float>(timesteps * n);

            CUDA_CHECK(cudaStreamCreateWithFlags(&cx.genStream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamCreateWithFlags(&cx.rayStream, cudaStreamNonBlocking));
            for (int k = 0; k < 2; ++k) {
                CUDA_CHECK(cudaEventCreateWithFlags(&cx.genDone[k], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&cx.rayDone[k], cudaEventDisableTiming));
            }
            CUDA_CHECK(cudaEventCreateWithFlags(&cx.stepDone, cudaEventDisableTiming));
            CUDA_CHECK(cudaFuncSetAttribute(mtJumpKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(MT_JUMP_SMEM)));

            // The computed pairs start at most one chunk before the owned pairs
            // and end before them; the owned pairs number at most rows * (N - 1).
            const unsigned long long chunks = std::min<unsigned long long>(
                bounds.chunks, (rows * (n - 1) * VALS_PER_PAIR + MT_CHUNK - 1) / MT_CHUNK + 1);
            cx.mask = deviceAlloc<uint32_t>(n * maskWords);
            cx.rowStart = deviceAlloc<unsigned long long>(n + 1);
            cx.polys = deviceAlloc<uint32_t>(static_cast<size_t>(maxRounds + 1) * MT_POLY_WORDS);
            cx.states = deviceAlloc<uint32_t>(chunks * MT_N);
            for (int k = 0; k < 2; ++k) {
                cx.rnd[k] = deviceAlloc<float>(std::min(BATCH_VALS, chunks * MT_CHUNK));
                cx.pairs[k] = deviceAlloc<uint2>(std::min(BATCH_PAIRS, chunks * CHUNK_PAIRS));
            }
            cx.wPair = deviceAlloc<float>(rows * (n - 1) + CHUNK_PAIRS);
            cx.nonZero = deviceAlloc<unsigned long long>(1);
            CUDA_CHECK(cudaDeviceSynchronize());
        });
    }
    for (auto& w : workers) w.join();

    CUDA_CHECK(cudaSetDevice(0));
    if (numDevices > 1) {
        CUDA_CHECK(cudaMallocHost(&state.hostRadB, std::max<size_t>(timesteps * n, 1) * sizeof(float)));
    }
    state.counts = deviceAlloc<uint32_t>(n);
    state.corr = deviceAlloc<float>(timesteps * n);
    state.devDistances = deviceAlloc<float>(n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void freeSimulation(SimulationState& state) {
    for (DeviceContext& cx : state.ctx) {
        CUDA_CHECK(cudaSetDevice(cx.device));
        CUDA_CHECK(cudaDeviceSynchronize());
        for (void* p : {static_cast<void*>(cx.triA), static_cast<void*>(cx.triB),
                        static_cast<void*>(cx.triC), static_cast<void*>(cx.triN),
                        static_cast<void*>(cx.centers), static_cast<void*>(cx.areas),
                        static_cast<void*>(cx.octNodes), static_cast<void*>(cx.octLeafTris),
                        static_cast<void*>(cx.w), static_cast<void*>(cx.tau), static_cast<void*>(cx.radB),
                        static_cast<void*>(cx.mask), static_cast<void*>(cx.rowStart),
                        static_cast<void*>(cx.polys), static_cast<void*>(cx.states),
                        static_cast<void*>(cx.rnd[0]), static_cast<void*>(cx.rnd[1]),
                        static_cast<void*>(cx.pairs[0]), static_cast<void*>(cx.pairs[1]),
                        static_cast<void*>(cx.wPair), static_cast<void*>(cx.nonZero)}) {
            CUDA_CHECK(cudaFree(p));
        }
        for (int k = 0; k < 2; ++k) {
            CUDA_CHECK(cudaEventDestroy(cx.genDone[k]));
            CUDA_CHECK(cudaEventDestroy(cx.rayDone[k]));
        }
        CUDA_CHECK(cudaEventDestroy(cx.stepDone));
        CUDA_CHECK(cudaStreamDestroy(cx.genStream));
        CUDA_CHECK(cudaStreamDestroy(cx.rayStream));
    }
    state.ctx.clear();
    CUDA_CHECK(cudaSetDevice(0));
    if (state.hostRadB) CUDA_CHECK(cudaFreeHost(state.hostRadB));
    state.hostRadB = nullptr;
    for (void* p : {static_cast<void*>(state.counts), static_cast<void*>(state.corr),
                    static_cast<void*>(state.devDistances)}) {
        CUDA_CHECK(cudaFree(p));
    }
    state.counts = nullptr;
    state.corr = nullptr;
    state.devDistances = nullptr;
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Shared inputs of the per-GPU form factor work
struct FormFactorJob {
    uint32_t N = 0, maskWords = 0;
    const std::vector<unsigned long long>* rowStart = nullptr;
    const Gf2Poly* phi = nullptr;
};

// Form factor work of one GPU: the random-number chunks [chunkBegin, chunkEnd).
// Its pairs start with the chunk containing the first owned pair, so the
// weights are stored in wPair from pair index pairBase = chunkBegin * CHUNK_PAIRS.
struct FormFactorRange {
    uint32_t chunkBegin = 0, chunkEnd = 0;
    unsigned long long pairBase = 0;
    unsigned long long nonZero = 0;          // result: number of K_ij > EPSILON
};

// Runs the form factor computation of one GPU's range: generator states via
// jump-ahead, then a double-buffered pipeline of (random numbers, pair list)
// generation and ray tracing.
static void runFormFactorRange(DeviceContext& cx, FormFactorRange& range, const FormFactorJob& job) {
    const int dev = cx.device;
    const uint32_t N = job.N;
    const std::vector<unsigned long long>& rowStart = *job.rowStart;
    const unsigned long long totalPairs = rowStart[N];
    CUDA_CHECK(cudaSetDevice(dev));

    // Culling mask (already computed on the primary GPU) and row offsets
    if (dev != 0) {
        cullMaskKernel<<<dim3(N, (N + MASK_THREADS - 1) / MASK_THREADS), MASK_THREADS, 0, cx.genStream>>>(
            cx.triN, N, job.maskWords, cx.mask, nullptr);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpyAsync(cx.rowStart, rowStart.data(), rowStart.size() * sizeof(unsigned long long),
                               cudaMemcpyHostToDevice, cx.genStream));
    CUDA_CHECK(cudaMemsetAsync(cx.nonZero, 0, sizeof(unsigned long long), cx.genStream));
    if (range.chunkBegin == range.chunkEnd) {
        CUDA_CHECK(cudaStreamSynchronize(cx.genStream));
        return;
    }

    // Generator states at the chunk starts of the range (states[k] for chunk
    // chunkBegin + k): the seed state is jumped to chunkBegin, then the range is
    // filled by doubling jumps (round b uses P_b = t^(MT_CHUNK * 2^b)).
    const uint32_t len = range.chunkEnd - range.chunkBegin;
    int lenRounds = 0;
    while ((1u << lenRounds) < len) ++lenRounds;
    std::vector<uint32_t> hostPolys(static_cast<size_t>(lenRounds + 1) * MT_POLY_WORDS);
    MtJumpPolynomials jumpPolys(*job.phi);
    for (int b = 0; b < lenRounds; ++b) jumpPolys.next(hostPolys.data() + static_cast<size_t>(b) * MT_POLY_WORDS);
    uint32_t* offsetPoly = hostPolys.data() + static_cast<size_t>(lenRounds) * MT_POLY_WORDS;
    if (range.chunkBegin > 0) jumpPolys.chunkJump(range.chunkBegin, offsetPoly);
    CUDA_CHECK(cudaMemcpyAsync(cx.polys, hostPolys.data(), hostPolys.size() * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, cx.genStream));
    const std::vector<uint32_t> seed = mtSeedState(MT_SEED);
    CUDA_CHECK(cudaMemcpyAsync(cx.states, seed.data(), MT_N * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, cx.genStream));
    if (range.chunkBegin > 0) {
        mtJumpKernel<<<1, MT_JUMP_THREADS, MT_JUMP_SMEM, cx.genStream>>>(
            cx.states, cx.polys + static_cast<size_t>(lenRounds) * MT_POLY_WORDS, 0, 1);
        CUDA_CHECK(cudaGetLastError());
    }
    for (int b = 0; b < lenRounds; ++b) {
        const uint32_t half = 1u << b;
        mtJumpKernel<<<std::min(half, len - half), MT_JUMP_THREADS, MT_JUMP_SMEM, cx.genStream>>>(
            cx.states, cx.polys + static_cast<size_t>(b) * MT_POLY_WORDS, half, len);
        CUDA_CHECK(cudaGetLastError());
    }

    DeviceTriangles tris{cx.triA, cx.triB, cx.triC, cx.triN, cx.areas};
    uint32_t batch = 0;
    for (uint32_t c0 = range.chunkBegin; c0 < range.chunkEnd; c0 += CHUNKS_PER_BATCH, ++batch) {
        const int buf = batch & 1;
        const uint32_t nc = std::min<uint32_t>(CHUNKS_PER_BATCH, range.chunkEnd - c0);
        const unsigned long long p0 = static_cast<unsigned long long>(c0) * CHUNK_PAIRS;
        const unsigned long long p1 = std::min(totalPairs, p0 + nc * CHUNK_PAIRS);
        const uint32_t numPairs = static_cast<uint32_t>(p1 - p0);

        if (batch >= 2) CUDA_CHECK(cudaStreamWaitEvent(cx.genStream, cx.rayDone[buf], 0));
        mtGenerateKernel<<<nc, MT_GEN_THREADS, 0, cx.genStream>>>(cx.states, c0 - range.chunkBegin, cx.rnd[buf]);
        CUDA_CHECK(cudaGetLastError());

        // Rows overlapping [p0, p1)
        const uint32_t rowLo = static_cast<uint32_t>(
            std::upper_bound(rowStart.begin(), rowStart.end(), p0) - rowStart.begin() - 1);
        const uint32_t rowHi = static_cast<uint32_t>(
            std::lower_bound(rowStart.begin(), rowStart.end(), p1) - rowStart.begin());
        compactPairsKernel<<<rowHi - rowLo, ROW_THREADS, 0, cx.genStream>>>(
            cx.mask, job.maskWords, cx.rowStart, rowLo, p0, p1, cx.pairs[buf]);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(cx.genDone[buf], cx.genStream));

        CUDA_CHECK(cudaStreamWaitEvent(cx.rayStream, cx.genDone[buf], 0));
        const size_t threads = static_cast<size_t>(numPairs) * NUM_RAYS;
        const uint32_t blocks = static_cast<uint32_t>((threads + FF_THREADS - 1) / FF_THREADS);
        formFactorKernel<<<blocks, FF_THREADS, 0, cx.rayStream>>>(
            cx.pairs[buf], numPairs, reinterpret_cast<const float4*>(cx.rnd[buf]), tris,
            cx.octNodes, cx.octLeafTris, cx.wPair + (p0 - range.pairBase), cx.nonZero);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(cx.rayDone[buf], cx.rayStream));
    }

    CUDA_CHECK(cudaMemcpyAsync(&range.nonZero, cx.nonZero, sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost, cx.rayStream));
    CUDA_CHECK(cudaStreamSynchronize(cx.rayStream));
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const size_t n = state.numTriangles;
    const uint32_t N = static_cast<uint32_t>(n);
    DeviceContext& primary = state.ctx[0];
    CUDA_CHECK(cudaSetDevice(0));

    // Pairs culled for facing the same direction consume no random numbers
    const uint32_t maskWords = (N + 31) / 32;
    CUDA_CHECK(cudaMemsetAsync(state.counts, 0, n * sizeof(uint32_t), primary.genStream));
    cullMaskKernel<<<dim3(N, (N + MASK_THREADS - 1) / MASK_THREADS), MASK_THREADS, 0, primary.genStream>>>(
        primary.triN, N, maskWords, primary.mask, state.counts);
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint32_t> counts(n);
    CUDA_CHECK(cudaMemcpyAsync(counts.data(), state.counts, n * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                               primary.genStream));

    // Characteristic polynomial of the generator (host, overlapping the mask kernel)
    mtVerifyStream();
    const Gf2Poly phi = mtCharacteristicPolynomial();

    CUDA_CHECK(cudaStreamSynchronize(primary.genStream));
    std::vector<unsigned long long> rowStart(n + 1, 0);
    for (size_t i = 0; i < n; ++i) rowStart[i + 1] = rowStart[i] + counts[i];
    const unsigned long long totalPairs = rowStart[n];
    const uint32_t numChunks = static_cast<uint32_t>((totalPairs * VALS_PER_PAIR + MT_CHUNK - 1) / MT_CHUNK);

    // GPU g computes the chunks from the one holding its first owned pair up to
    // the one holding the next GPU's first owned pair
    const uint32_t numDevices = static_cast<uint32_t>(state.ctx.size());
    std::vector<FormFactorRange> ranges(numDevices);
    for (uint32_t g = 0; g < numDevices; ++g) {
        const DeviceContext& cx = state.ctx[g];
        FormFactorRange& r = ranges[g];
        r.chunkBegin = static_cast<uint32_t>(rowStart[cx.rowBegin] / CHUNK_PAIRS);
        r.chunkEnd = (g + 1 < numDevices) ? static_cast<uint32_t>(rowStart[cx.rowEnd] / CHUNK_PAIRS) : numChunks;
        r.chunkEnd = std::max(r.chunkEnd, r.chunkBegin);
        r.pairBase = static_cast<unsigned long long>(r.chunkBegin) * CHUNK_PAIRS;
    }

    FormFactorJob job;
    job.N = N;
    job.maskWords = maskWords;
    job.rowStart = &rowStart;
    job.phi = &phi;
    std::vector<std::thread> threads;
    for (uint32_t g = 0; g < numDevices; ++g) {
        threads.emplace_back(runFormFactorRange, std::ref(state.ctx[g]), std::ref(ranges[g]), std::cref(job));
    }
    for (auto& t : threads) t.join();
    state.nonZeroKij = 0;
    for (const FormFactorRange& r : ranges) state.nonZeroKij += r.nonZero;

    // Owned pairs of GPU g beyond its computed range [pairBase, chunkEnd * CHUNK_PAIRS)
    // lie in chunk chunkEnd, which another GPU computed; fetch them from there.
    for (uint32_t g = 0; g < numDevices; ++g) {
        const unsigned long long computedEnd = std::min(totalPairs,
            static_cast<unsigned long long>(ranges[g].chunkEnd) * CHUNK_PAIRS);
        const unsigned long long from = std::max(rowStart[state.ctx[g].rowBegin], computedEnd);
        const unsigned long long to = rowStart[state.ctx[g].rowEnd];
        if (to <= from) continue;
        uint32_t h = 0;
        while (!(ranges[h].chunkBegin <= ranges[g].chunkEnd && ranges[g].chunkEnd < ranges[h].chunkEnd)) ++h;
        std::vector<float> part(to - from);
        CUDA_CHECK(cudaSetDevice(state.ctx[h].device));
        CUDA_CHECK(cudaMemcpy(part.data(), state.ctx[h].wPair + (from - ranges[h].pairBase),
                              part.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaSetDevice(state.ctx[g].device));
        CUDA_CHECK(cudaMemcpy(state.ctx[g].wPair + (from - ranges[g].pairBase), part.data(),
                              part.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Row-major weights of the owned rows
    for (uint32_t g = 0; g < numDevices; ++g) {
        DeviceContext& cx = state.ctx[g];
        if (cx.rowEnd == cx.rowBegin) continue;
        CUDA_CHECK(cudaSetDevice(cx.device));
        expandWeightsKernel<<<cx.rowEnd - cx.rowBegin, ROW_THREADS, 0, cx.rayStream>>>(
            cx.mask, maskWords, cx.rowStart, cx.rowBegin, cx.wPair, ranges[g].pairBase, cx.w, N);
        CUDA_CHECK(cudaGetLastError());
    }
    for (DeviceContext& cx : state.ctx) {
        CUDA_CHECK(cudaSetDevice(cx.device));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaSetDevice(0));

    for (size_t i = 0; i < n; ++i) {
        if ((i + 1) % 100 == 0 || i + 1 == n) {
            printf("  Progress: %zu/%zu triangles\n", i + 1, n);
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const uint32_t N = static_cast<uint32_t>(state.numTriangles);
    for (DeviceContext& cx : state.ctx) {
        if (cx.rowEnd == cx.rowBegin) continue;
        CUDA_CHECK(cudaSetDevice(cx.device));
        dim3 block(256);
        dim3 grid(cx.rowEnd - cx.rowBegin, (N + block.x - 1) / block.x);
        tauKernel<<<grid, block, 0, cx.rayStream>>>(cx.centers, cx.tau, N, cx.rowBegin);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaSetDevice(0));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

constexpr int SIM_ROWS = 16;                     // receiving triangles per block (<= 32)
constexpr int SIM_LOAD_WARPS = 8;
constexpr int SIM_THREADS = 32 * (SIM_LOAD_WARPS + 1);
constexpr int SIM_TILE = 64;                     // source triangles per tile
constexpr int SIM_ROWS_PER_WARP = SIM_ROWS / SIM_LOAD_WARPS;

// Each block handles SIM_ROWS receiving triangles i (rows of the row-major
// w / tau blocks). Loader warps stage the terms of the next tile of source
// triangles j in shared memory (weight and source radiosity, the latter set to
// zero for terms the reference skips, which then add exactly +0) while warp 0
// accumulates the current tile sequentially in the reference order.
__global__ void __launch_bounds__(SIM_THREADS)
simulationStepKernel(const float* __restrict__ w, const uint8_t* __restrict__ tau,
                     float* __restrict__ radB, uint32_t N, uint32_t row0, uint32_t rows,
                     int t, val_t rho, val_t emission, uint32_t sourceIndex) {
    __shared__ float sW[2][SIM_TILE][SIM_ROWS + 1];
    __shared__ float sR[2][SIM_TILE][SIM_ROWS + 1];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const uint32_t rowBase = blockIdx.x * SIM_ROWS;
    const uint32_t numTiles = (N + SIM_TILE - 1) / SIM_TILE;

    auto loadTile = [&](uint32_t tile) {
        const int buf = tile & 1;
        const uint32_t j0 = tile * SIM_TILE;
        constexpr int PER_LANE = SIM_ROWS_PER_WARP * (SIM_TILE / 32);
        val_t wv[PER_LANE];
        int tv[PER_LANE];
#pragma unroll
        for (int k = 0; k < PER_LANE; ++k) {
            const uint32_t r = rowBase + (warp - 1) * SIM_ROWS_PER_WARP + k / (SIM_TILE / 32);
            const uint32_t j = j0 + (k % (SIM_TILE / 32)) * 32 + lane;
            wv[k] = ZERO;
            tv[k] = 0;
            if (r < rows && j < N) {
                const size_t idx = static_cast<size_t>(r) * N + j;
                wv[k] = __ldg(&w[idx]);
                tv[k] = __ldg(&tau[idx]);
            }
        }
#pragma unroll
        for (int k = 0; k < PER_LANE; ++k) {
            const int rr = (warp - 1) * SIM_ROWS_PER_WARP + k / (SIM_TILE / 32);
            const int jj = (k % (SIM_TILE / 32)) * 32 + lane;
            val_t radJ = ZERO;
            if (t >= tv[k] && wv[k] > ZERO) {
                const val_t r = radB[static_cast<size_t>(t - tv[k]) * N + j0 + jj];
                if (r > ZERO) radJ = r;
            }
            sW[buf][jj][rr] = wv[k];
            sR[buf][jj][rr] = radJ;
        }
    };

    val_t sumB = ZERO;
    if (warp > 0) loadTile(0);
    __syncthreads();
    for (uint32_t tile = 0; tile < numTiles; ++tile) {
        if (warp > 0) {
            if (tile + 1 < numTiles) loadTile(tile + 1);
        } else if (lane < SIM_ROWS) {
            const int buf = tile & 1;
            const uint32_t j0 = tile * SIM_TILE;
            if (j0 + SIM_TILE <= N) {
#pragma unroll 16
                for (int jj = 0; jj < SIM_TILE; ++jj) sumB += sW[buf][jj][lane] * sR[buf][jj][lane];
            } else {
                for (int jj = 0; jj < static_cast<int>(N - j0); ++jj) sumB += sW[buf][jj][lane] * sR[buf][jj][lane];
            }
        }
        __syncthreads();
    }
    const uint32_t r = rowBase + lane;
    if (warp == 0 && lane < SIM_ROWS && r < rows) {
        const uint32_t i = row0 + r;
        const val_t radE = (i == sourceIndex) ? emission : ZERO;
        radB[static_cast<size_t>(t) * N + i] = rho * sumB + radE;
    }
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const uint32_t N = static_cast<uint32_t>(state.numTriangles);
    const size_t T = state.numTimesteps;
    const size_t timeOff = T / 2;   // source active for the first half of timesteps
    const size_t numDevices = state.ctx.size();

    for (DeviceContext& cx : state.ctx) {
        CUDA_CHECK(cudaSetDevice(cx.device));
        CUDA_CHECK(cudaMemsetAsync(cx.radB, 0, T * N * sizeof(float), cx.rayStream));
    }
    for (size_t t = 0; t < T; ++t) {
        const val_t emission = t < timeOff ? 1.0f : ZERO;
        // Owned rows of timestep t on every GPU, copied to the host
        for (DeviceContext& cx : state.ctx) {
            const uint32_t rows = cx.rowEnd - cx.rowBegin;
            CUDA_CHECK(cudaSetDevice(cx.device));
            if (rows > 0) {
                simulationStepKernel<<<(rows + SIM_ROWS - 1) / SIM_ROWS, SIM_THREADS, 0, cx.rayStream>>>(
                    cx.w, cx.tau, cx.radB, N, cx.rowBegin, rows, static_cast<int>(t), state.rho, emission,
                    static_cast<uint32_t>(state.sourceIndex));
                CUDA_CHECK(cudaGetLastError());
                if (numDevices > 1) {
                    const size_t off = t * N + cx.rowBegin;
                    CUDA_CHECK(cudaMemcpyAsync(state.hostRadB + off, cx.radB + off, rows * sizeof(float),
                                               cudaMemcpyDeviceToHost, cx.rayStream));
                }
            }
            CUDA_CHECK(cudaEventRecord(cx.stepDone, cx.rayStream));
        }
        // All-gather of timestep t
        if (numDevices > 1) {
            for (DeviceContext& cx : state.ctx) {
                CUDA_CHECK(cudaSetDevice(cx.device));
                for (const DeviceContext& other : state.ctx) {
                    const uint32_t rows = other.rowEnd - other.rowBegin;
                    if (&other == &cx || rows == 0) continue;
                    CUDA_CHECK(cudaStreamWaitEvent(cx.rayStream, other.stepDone, 0));
                    const size_t off = t * N + other.rowBegin;
                    CUDA_CHECK(cudaMemcpyAsync(cx.radB + off, state.hostRadB + off, rows * sizeof(float),
                                               cudaMemcpyHostToDevice, cx.rayStream));
                }
            }
        }

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
    for (DeviceContext& cx : state.ctx) {
        CUDA_CHECK(cudaSetDevice(cx.device));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaSetDevice(0));
    if (T > 0) {
        if (numDevices > 1) {
            std::copy(state.hostRadB, state.hostRadB + T * N, state.radB.begin());
        } else {
            CUDA_CHECK(cudaMemcpy(state.radB.data(), state.ctx[0].radB, T * N * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

// corr[lag * N + i] = sum_{tt >= lag} radB[tt - lag][source] * radB[tt][i]
__global__ void correlationKernel(const float* __restrict__ radB, float* __restrict__ corr,
                                  uint32_t N, uint32_t T, uint32_t sourceIndex) {
    const uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t lag = blockIdx.x;
    if (i >= N) return;
    val_t sum = ZERO;
    for (uint32_t tt = lag; tt < T; ++tt) {
        val_t pB = __ldg(&radB[static_cast<size_t>(tt) * N + i]);
        val_t pS = __ldg(&radB[static_cast<size_t>(tt - lag) * N + sourceIndex]);
        sum += pS * pB;
    }
    corr[static_cast<size_t>(lag) * N + i] = sum;
}

__global__ void bestLagKernel(const float* __restrict__ corr, float* __restrict__ distances,
                              uint32_t N, uint32_t T) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    val_t maxCorr = ZERO;
    int bestT = 0;
    for (uint32_t t = 0; t < T; ++t) {
        val_t sum = corr[static_cast<size_t>(t) * N + i];
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }
    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const uint32_t N = static_cast<uint32_t>(state.numTriangles);
    const uint32_t T = static_cast<uint32_t>(state.numTimesteps);
    CUDA_CHECK(cudaSetDevice(0));
    const float* radB = state.ctx[0].radB;   // complete after the all-gathers

    const uint32_t threads = 128;
    if (T > 0) {
        dim3 grid(T, (N + threads - 1) / threads);
        correlationKernel<<<grid, threads>>>(radB, state.corr, N, T,
                                             static_cast<uint32_t>(state.sourceIndex));
        CUDA_CHECK(cudaGetLastError());
    }
    bestLagKernel<<<(N + threads - 1) / threads, threads>>>(state.corr, state.devDistances, N, T);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.devDistances, N * sizeof(float), cudaMemcpyDeviceToHost));
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
    bool ok = !validate || validateResults(state);
    freeSimulation(state);
    return ok ? 0 : 1;
}
