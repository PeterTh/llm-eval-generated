/**
 * Room Response Simulation Benchmark (CUDA)
 * 
 * Simulates room impulse response using radiosity-based wave propagation. It
 * models how sound/light waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * GPU parallelization:
 * - Form factors: one thread per (receiver, emitter) pair. The random sample
 *   stream is consumed in exactly the same order as the sequential reference
 *   (row-major over non-culled pairs), so the generated samples are identical.
 *   The host produces the stream in chunks that are double-buffered against the
 *   GPU, so generation overlaps with kernel execution.
 * - Visibility: the octree is flattened and traversed on the device.
 * - Wave propagation: one thread per receiver triangle for each timestep. Each
 *   timestep only reads radiosity from strictly earlier timesteps (tau >= 1 for
 *   distinct triangles), so the timestep loop is sequential while the receivers
 *   are fully parallel. The per-receiver summation order is preserved.
 * - Cross-correlation: one thread per triangle.
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
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// Error handling
// ============================================================================

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__,         \
                    __LINE__, cudaGetErrorString(err__), #call);                \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

[[noreturn]] static void fatal(const char* msg) {
    fprintf(stderr, "Fatal: %s\n", msg);
    std::exit(1);
}

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

constexpr int DRAWS_PER_PAIR = NUM_RAYS * 4;   // 2 points per ray, 2 random numbers each
constexpr uint32_t SKIP_PAIR = 0xFFFFFFFFu;    // Marks pairs that consume no random numbers
constexpr size_t MAX_PAIRS_PER_CHUNK = size_t(1) << 19;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int TRAVERSAL_STACK = 128;
constexpr int MAX_OCTREE_DEPTH = 15;           // Stack bound: 7 * depth + 8 <= TRAVERSAL_STACK

// ============================================================================
// Vector and Triangle Types (host)
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
// Octree for Spatial Acceleration (built on host, traversed on device)
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
// Random Number Generation
// ============================================================================

// Mersenne Twister (MT19937) producing the same sequence as std::mt19937 and the
// same floats as std::uniform_real_distribution<float>(0, 1). The loops are
// written so that the compiler can vectorize the twist and tempering steps.
class RandomGenerator {
    static constexpr size_t MT_N = 624;
    static constexpr size_t MT_M = 397;
    uint32_t mt[MT_N];
    size_t idx;

    void twist() {
        constexpr uint32_t UPPER = 0x80000000u;
        constexpr uint32_t LOWER = 0x7fffffffu;
        constexpr uint32_t MATRIX_A = 0x9908b0dfu;
        size_t i = 0;
        for (; i < MT_N - MT_M; ++i) {
            uint32_t y = (mt[i] & UPPER) | (mt[i + 1] & LOWER);
            mt[i] = mt[i + MT_M] ^ (y >> 1) ^ ((0u - (y & 1u)) & MATRIX_A);
        }
        for (; i < MT_N - 1; ++i) {
            uint32_t y = (mt[i] & UPPER) | (mt[i + 1] & LOWER);
            mt[i] = mt[i - (MT_N - MT_M)] ^ (y >> 1) ^ ((0u - (y & 1u)) & MATRIX_A);
        }
        uint32_t y = (mt[MT_N - 1] & UPPER) | (mt[0] & LOWER);
        mt[MT_N - 1] = mt[MT_M - 1] ^ (y >> 1) ^ ((0u - (y & 1u)) & MATRIX_A);
    }

public:
    explicit RandomGenerator(uint32_t seed = 42) : idx(MT_N) {
        mt[0] = seed;
        for (size_t i = 1; i < MT_N; ++i) {
            mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
        }
    }

    // Writes the next `count` values of the stream into `out`.
    void fill(val_t* out, size_t count) {
        while (count > 0) {
            if (idx == MT_N) {
                twist();
                idx = 0;
            }
            const size_t m = std::min(count, MT_N - idx);
            const uint32_t* __restrict__ src = mt + idx;
            for (size_t k = 0; k < m; ++k) {
                uint32_t y = src[k];
                y ^= y >> 11;
                y ^= (y << 7) & 0x9d2c5680u;
                y ^= (y << 15) & 0xefc60000u;
                y ^= y >> 18;
                // Same conversion as generate_canonical<float, 24>: value / 2^32, clamped below 1
                const val_t f = static_cast<val_t>(y) * 0x1p-32f;
                out[k] = (f >= 1.0f) ? 0x1.fffffep-1f : f;
            }
            idx += m;
            out += m;
            count -= m;
        }
    }
};

// ============================================================================
// Device data layout
// ============================================================================

struct DVec3 {
    float x, y, z;
};

// Triangle as seen by the device. `n` is the precomputed normal from the host.
struct DTri {
    DVec3 a, b, c, n;
};

// Flattened octree node. child[k] == -1 means no child. Leaves have triCount > 0.
struct DNode {
    DVec3 center;
    DVec3 half;
    int32_t child[8];
    uint32_t triBegin;
    uint32_t triCount;
};

static DVec3 toDVec3(const Vec3& v) { return DVec3{v.x, v.y, v.z}; }

__host__ __device__ __forceinline__ DVec3 dadd(DVec3 a, DVec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__host__ __device__ __forceinline__ DVec3 dsub(DVec3 a, DVec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__host__ __device__ __forceinline__ DVec3 dmul(DVec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
__host__ __device__ __forceinline__ DVec3 ddiv(DVec3 a, float s) { return {a.x / s, a.y / s, a.z / s}; }
__host__ __device__ __forceinline__ DVec3 dneg(DVec3 a) { return {-a.x, -a.y, -a.z}; }
__host__ __device__ __forceinline__ float ddot(DVec3 a, DVec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ __forceinline__ DVec3 dcross(DVec3 a, DVec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__host__ __device__ __forceinline__ float dsqnorm(DVec3 a) { return a.x * a.x + a.y * a.y + a.z * a.z; }
__host__ __device__ __forceinline__ float dnorm(DVec3 a) { return sqrtf(dsqnorm(a)); }

// Point inside a triangle from barycentric-style samples (same expression order as host)
__device__ __forceinline__ DVec3 dPointInTriangle(const DTri& t, float u, float v) {
    DVec3 ab = dsub(t.b, t.a);
    DVec3 ac = dsub(t.c, t.a);
    return dadd(dadd(t.a, dmul(ab, u)), dmul(ac, v));
}

// ============================================================================
// Device geometry: ray-box and ray-triangle tests
// ============================================================================

__device__ __forceinline__ bool dRayIntersectsBox(const DNode& nd, DVec3 p1, DVec3 p2) {
    DVec3 d = dmul(dsub(p2, p1), 0.5f);
    DVec3 c = dsub(dadd(p1, d), nd.center);
    DVec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const DVec3& he = nd.half;

    if (fabsf(c.x) > he.x + ad.x) return false;
    if (fabsf(c.y) > he.y + ad.y) return false;
    if (fabsf(c.z) > he.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > he.y * ad.z + he.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > he.z * ad.x + he.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > he.x * ad.y + he.y * ad.x + EPSILON) return false;

    return true;
}

// Moller-Trumbore ray-triangle intersection. Returns FLT_MAX on miss.
__device__ __forceinline__ float dRayTriangleIntersect(DVec3 orig, DVec3 dir,
                                                       DVec3 v0, DVec3 v1, DVec3 v2) {
    DVec3 e1 = dsub(v1, v0);
    DVec3 e2 = dsub(v2, v0);
    DVec3 pvec = dcross(dir, e2);
    float det = ddot(e1, pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    float invDet = 1.0f / det;
    DVec3 tvec = dsub(orig, v0);
    float u = ddot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    DVec3 qvec = dcross(tvec, e1);
    float v = ddot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return ddot(e2, qvec) * invDet;
}

// Check if the segment from -> to is blocked by any triangle other than src/dst.
// Octree traversal: the result is a pure "any blocker" predicate, so traversal
// order does not affect the outcome.
__device__ bool dIsRayBlocked(DVec3 from, DVec3 to,
                              const DTri* __restrict__ tris,
                              const DNode* __restrict__ nodes,
                              const uint32_t* __restrict__ triList,
                              uint32_t srcIdx, uint32_t dstIdx) {
    DVec3 dir = dsub(to, from);
    float rayLen = dnorm(dir);
    if (rayLen < EPSILON) return true;
    DVec3 dirNorm = ddiv(dir, rayLen);

    int stack[TRAVERSAL_STACK];
    int sp = 0;
    stack[sp++] = 0;  // Root (its box is not tested, matching the host traversal)

    while (sp > 0) {
        const DNode& nd = nodes[stack[--sp]];
        if (nd.triCount > 0) {
            for (uint32_t q = 0; q < nd.triCount; ++q) {
                const uint32_t idx = triList[nd.triBegin + q];
                if (idx == srcIdx || idx == dstIdx) continue;
                const DTri& t = tris[idx];
                float dist = dRayTriangleIntersect(from, dirNorm, t.a, t.b, t.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                const int ch = nd.child[c];
                if (ch >= 0 && dRayIntersectsBox(nodes[ch], from, to)) {
                    stack[sp++] = ch;
                }
            }
        }
    }
    return false;
}

__device__ __forceinline__ float dCosPhi(DVec3 v, DVec3 normal) {
    float vNorm = dnorm(v);
    if (vNorm <= EPSILON) return ZERO;
    float c = ddot(v, normal) / vNorm;
    return (ZERO < c) ? c : ZERO;  // matches std::max(ZERO, c)
}

// ============================================================================
// Simulation state (host side)
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix); filled for validation only
    std::vector<val_t> kij;         // Form factors (N x N, row-major); filled for validation only
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Host-side build of the spatial acceleration structure

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// Staging resources for the chunked form factor pipeline (double-buffered)
struct ChunkStaging {
    size_t rowsPerChunk = 0;
    size_t maxPairs = 0;
    size_t maxDraws = 0;
    float* hostDraws[2] = {nullptr, nullptr};       // pinned
    uint32_t* hostOffs[2] = {nullptr, nullptr};     // pinned
    cudaStream_t streams[2] = {nullptr, nullptr};
    cudaEvent_t done[2] = {nullptr, nullptr};
};

// Device-side buffers
struct DeviceState {
    ChunkStaging stage;
    DTri* tris = nullptr;
    DVec3* centers = nullptr;
    float* areas = nullptr;
    float* rho = nullptr;
    float* radE = nullptr;
    float* radB = nullptr;
    float* kij = nullptr;       // N x N row-major, kij[i*N + j]: receiver i, emitter j
    int* tauT = nullptr;        // N x N transposed, tauT[j*N + i]
    float* wT = nullptr;        // N x N transposed, wT[j*N + i] = min(kij[i][j] * area[j], 1) or 0
    float* distances = nullptr;
    DNode* nodes = nullptr;
    uint32_t* triList = nullptr;
    // Double-buffered per-chunk random samples and draw offsets
    float* chunkDraws[2] = {nullptr, nullptr};
    uint32_t* chunkOffs[2] = {nullptr, nullptr};
};

template <typename T>
static T* deviceAlloc(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), std::max<size_t>(count, 1) * sizeof(T)));
    return p;
}

// ============================================================================
// Octree flattening
// ============================================================================

struct FlatOctree {
    std::vector<DNode> nodes;
    std::vector<uint32_t> triList;
    int maxDepth = 0;
};

static int flattenOctree(const Octree& node, FlatOctree& flat, int depth) {
    const int index = static_cast<int>(flat.nodes.size());
    flat.nodes.push_back(DNode{});
    flat.maxDepth = std::max(flat.maxDepth, depth);

    DNode dn{};
    dn.center = toDVec3(node.center);
    dn.half = toDVec3(node.halfExtent);
    for (int c = 0; c < 8; ++c) dn.child[c] = -1;

    if (!node.triangleIndices.empty()) {
        dn.triBegin = static_cast<uint32_t>(flat.triList.size());
        for (size_t idx : node.triangleIndices) {
            flat.triList.push_back(static_cast<uint32_t>(idx));
        }
        dn.triCount = static_cast<uint32_t>(node.triangleIndices.size());
    } else {
        for (int c = 0; c < 8; ++c) {
            if (node.children[c]) {
                dn.child[c] = flattenOctree(*node.children[c], flat, depth + 1);
            }
        }
    }

    flat.nodes[index] = dn;
    return index;
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, DeviceState& dev, int subdivisions,
                          size_t timesteps, size_t sourceIdx, val_t reflectivity) {
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
    state.rho.assign(state.numTriangles, reflectivity);

    // Source emission (active for first half of timesteps)
    state.radE.assign(timesteps * state.numTriangles, ZERO);
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // ---- Device allocation and upload ----
    const size_t n = state.numTriangles;
    const size_t T = timesteps;

    FlatOctree flat;
    flattenOctree(state.octree, flat, 0);
    if (flat.maxDepth > MAX_OCTREE_DEPTH) fatal("octree deeper than device traversal stack supports");

    std::vector<DTri> hostTris(n);
    std::vector<DVec3> hostCenters(n);
    for (size_t i = 0; i < n; ++i) {
        const Triangle& t = state.triangles[i];
        hostTris[i] = DTri{toDVec3(t.a), toDVec3(t.b), toDVec3(t.c), toDVec3(t.normal())};
        hostCenters[i] = toDVec3(t.center());
    }

    dev.tris = deviceAlloc<DTri>(n);
    dev.centers = deviceAlloc<DVec3>(n);
    dev.areas = deviceAlloc<float>(n);
    dev.rho = deviceAlloc<float>(n);
    dev.radE = deviceAlloc<float>(T * n);
    dev.radB = deviceAlloc<float>(T * n);
    dev.kij = deviceAlloc<float>(n * n);
    dev.tauT = deviceAlloc<int>(n * n);
    dev.wT = deviceAlloc<float>(n * n);
    dev.distances = deviceAlloc<float>(n);
    dev.nodes = deviceAlloc<DNode>(flat.nodes.size());
    dev.triList = deviceAlloc<uint32_t>(flat.triList.size());

    CUDA_CHECK(cudaMemcpy(dev.tris, hostTris.data(), n * sizeof(DTri), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.centers, hostCenters.data(), n * sizeof(DVec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.areas, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.rho, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.radE, state.radE.data(), T * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dev.radB, 0, T * n * sizeof(float)));
    CUDA_CHECK(cudaMemset(dev.kij, 0, n * n * sizeof(float)));
    CUDA_CHECK(cudaMemset(dev.tauT, 0, n * n * sizeof(int)));
    CUDA_CHECK(cudaMemset(dev.wT, 0, n * n * sizeof(float)));
    CUDA_CHECK(cudaMemset(dev.distances, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev.nodes, flat.nodes.data(), flat.nodes.size() * sizeof(DNode), cudaMemcpyHostToDevice));
    if (!flat.triList.empty()) {
        CUDA_CHECK(cudaMemcpy(dev.triList, flat.triList.data(), flat.triList.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // Host-side result buffers are only materialized when validation needs them
    state.distances.assign(n, ZERO);

    // ---- Chunked form factor pipeline resources ----
    ChunkStaging& st = dev.stage;
    // Rows per chunk: bounded so that staging buffers stay modest in size
    st.rowsPerChunk = std::max<size_t>(1, std::min(n, MAX_PAIRS_PER_CHUNK / n));
    st.maxPairs = st.rowsPerChunk * n;
    st.maxDraws = st.maxPairs * DRAWS_PER_PAIR;
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&st.hostDraws[b]), st.maxDraws * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&st.hostOffs[b]), st.maxPairs * sizeof(uint32_t)));
        dev.chunkDraws[b] = deviceAlloc<float>(st.maxDraws);
        dev.chunkOffs[b] = deviceAlloc<uint32_t>(st.maxPairs);
        CUDA_CHECK(cudaStreamCreateWithFlags(&st.streams[b], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&st.done[b], cudaEventDisableTiming));
    }
}

void releaseChunkStaging(DeviceState& dev) {
    ChunkStaging& st = dev.stage;
    for (int b = 0; b < 2; ++b) {
        if (st.done[b]) cudaEventDestroy(st.done[b]);
        if (st.streams[b]) cudaStreamDestroy(st.streams[b]);
        if (st.hostDraws[b]) cudaFreeHost(st.hostDraws[b]);
        if (st.hostOffs[b]) cudaFreeHost(st.hostOffs[b]);
        if (dev.chunkDraws[b]) cudaFree(dev.chunkDraws[b]);
        if (dev.chunkOffs[b]) cudaFree(dev.chunkOffs[b]);
    }
    st = ChunkStaging{};
    dev.chunkDraws[0] = dev.chunkDraws[1] = nullptr;
    dev.chunkOffs[0] = dev.chunkOffs[1] = nullptr;
}

// ============================================================================
// Form factor kernels
// ============================================================================

// One thread per (receiver i, emitter j) pair in a chunk of rows.
// offs[k] is the offset into `draws` of this pair's 64 random numbers, or SKIP_PAIR.
__global__ void formFactorKernel(size_t n, size_t i0, size_t numPairs,
                                 const uint32_t* __restrict__ offs,
                                 const float* __restrict__ draws,
                                 const DTri* __restrict__ tris,
                                 const DNode* __restrict__ nodes,
                                 const uint32_t* __restrict__ triList,
                                 float* __restrict__ kij) {
    const size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= numPairs) return;

    const size_t i = i0 + k / n;
    const size_t j = k % n;
    float* out = kij + i * n + j;

    const uint32_t off = offs[k];
    if (off == SKIP_PAIR) {
        *out = ZERO;
        return;
    }

    const DTri ti = tris[i];
    const DTri tj = tris[j];
    const float* R = draws + off;

    float kacc = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        float u = R[0];
        float v = R[1];
        if (u + v > ONE) {
            u = ONE - u;
            v = ONE - v;
        }
        const DVec3 pI = dPointInTriangle(ti, u, v);

        u = R[2];
        v = R[3];
        if (u + v > ONE) {
            u = ONE - u;
            v = ONE - v;
        }
        const DVec3 pJ = dPointInTriangle(tj, u, v);
        R += 4;

        if (dIsRayBlocked(pI, pJ, tris, nodes, triList,
                          static_cast<uint32_t>(i), static_cast<uint32_t>(j))) continue;

        const DVec3 vec = dsub(pJ, pI);
        const float distSqr = dsqnorm(vec);
        if (distSqr < EPSILON) continue;

        const float cosI = dCosPhi(vec, ti.n);
        const float cosJ = dCosPhi(dneg(vec), tj.n);
        if (cosI <= ZERO || cosJ <= ZERO) continue;

        kacc += (cosI * cosJ) / (PI * distSqr);
    }

    *out = kacc * INV_NUM_RAYS;
}

// Time delay for all pairs, stored transposed: tauT[j*n + i]
__global__ void timeDelayKernel(size_t n, const DVec3* __restrict__ centers, int* __restrict__ tauT) {
    const size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= n * n) return;
    const size_t i = k / n;
    const size_t j = k % n;

    int tau = 0;
    if (i != j) {
        const float dist = dnorm(dsub(centers[i], centers[j]));
        tau = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
    }
    tauT[j * n + i] = tau;
}

// Propagation weights, stored transposed (indexed by emitter j, then receiver i):
// wT[j*n + i] = min(kij[i][j] * area[j], 1) when kij > 0, else 0.
__global__ void weightKernel(size_t n, const float* __restrict__ kij,
                             const float* __restrict__ areas, float* __restrict__ wT) {
    const size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= n * n) return;
    const size_t i = k / n;
    const size_t j = k % n;

    float w = ZERO;
    if (i != j) {
        const float kv = kij[i * n + j];
        if (kv > ZERO) {
            const float p = kv * areas[j];
            w = (ONE < p) ? ONE : p;  // matches std::min(p, ONE)
        }
    }
    wT[j * n + i] = w;
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state, DeviceState& dev) {
    printf("Computing time delays (Tau)...\n");
    const size_t n = state.numTriangles;
    const size_t total = n * n;
    const unsigned blocks = static_cast<unsigned>((total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    timeDelayKernel<<<blocks, THREADS_PER_BLOCK>>>(n, dev.centers, dev.tauT);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeFormFactors(SimulationState& state, DeviceState& dev) {
    printf("Computing form factors (Kij)...\n");
    const size_t n = state.numTriangles;
    RandomGenerator rng(42);

    ChunkStaging& st = dev.stage;
    const size_t rowsPerChunk = st.rowsPerChunk;
    bool inFlight[2] = {false, false};

    size_t chunk = 0;
    for (size_t i0 = 0; i0 < n; i0 += rowsPerChunk, ++chunk) {
        const size_t i1 = std::min(n, i0 + rowsPerChunk);
        const size_t pairs = (i1 - i0) * n;
        const int b = static_cast<int>(chunk & 1);

        // Wait until the previous kernel that used this staging buffer has finished
        if (inFlight[b]) {
            CUDA_CHECK(cudaEventSynchronize(st.done[b]));
            inFlight[b] = false;
        }

        // Host: assign draw offsets in the exact order of the sequential reference.
        // Pairs with i == j or culled by the normal test consume no draws.
        uint32_t* offs = st.hostOffs[b];
        float* draws = st.hostDraws[b];
        size_t cursor = 0;
        for (size_t i = i0; i < i1; ++i) {
            const Triangle& triI = state.triangles[i];
            for (size_t j = 0; j < n; ++j) {
                const size_t k = (i - i0) * n + j;
                if (i == j || triI.normal().dot(state.triangles[j].normal()) > 0.99f) {
                    offs[k] = SKIP_PAIR;
                    continue;
                }
                offs[k] = static_cast<uint32_t>(cursor);
                cursor += DRAWS_PER_PAIR;
            }
        }
        // The stream is consumed contiguously in that same order
        rng.fill(draws, cursor);

        if (cursor > 0) {
            CUDA_CHECK(cudaMemcpyAsync(dev.chunkDraws[b], draws, cursor * sizeof(float),
                                       cudaMemcpyHostToDevice, st.streams[b]));
        }
        CUDA_CHECK(cudaMemcpyAsync(dev.chunkOffs[b], offs, pairs * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, st.streams[b]));

        const unsigned blocks = static_cast<unsigned>((pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        formFactorKernel<<<blocks, THREADS_PER_BLOCK, 0, st.streams[b]>>>(
            n, i0, pairs, dev.chunkOffs[b], dev.chunkDraws[b],
            dev.tris, dev.nodes, dev.triList, dev.kij);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(st.done[b], st.streams[b]));
        inFlight[b] = true;

        for (size_t i = i0; i < i1; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == n) {
                printf("  Progress: %zu/%zu triangles\n", i + 1, n);
            }
        }
    }

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaStreamSynchronize(st.streams[b]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Derive propagation weights and transposed layout from the form factors
    const size_t total = n * n;
    const unsigned blocks = static_cast<unsigned>((total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    weightKernel<<<blocks, THREADS_PER_BLOCK>>>(n, dev.kij, dev.areas, dev.wT);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// One receiver per thread for a single timestep t. Reads only rows < t.
__global__ void simulationStepKernel(size_t n, size_t t,
                                     const float* __restrict__ wT,
                                     const int* __restrict__ tauT,
                                     const float* __restrict__ rho,
                                     const float* __restrict__ radE,
                                     float* __restrict__ radB) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sumB = ZERO;
    for (size_t j = 0; j < n; ++j) {
        if (j == i) continue;

        const size_t p = j * n + i;
        const int tauij = tauT[p];

        // Skip if wave hasn't yet propagated from j to i
        if (static_cast<int>(t) < tauij) continue;

        const float w = wT[p];
        if (w <= ZERO) continue;

        // Radiosity from the source triangle at the time emission occurred
        const size_t srcTime = t - static_cast<size_t>(tauij);
        const float radJ = radB[srcTime * n + j];
        if (radJ <= ZERO) continue;

        sumB += w * radJ;
    }

    radB[t * n + i] = rho[i] * sumB + radE[t * n + i];
}

void runSimulation(SimulationState& state, DeviceState& dev) {
    printf("Running wave propagation simulation...\n");
    const size_t n = state.numTriangles;
    const size_t T = state.numTimesteps;
    const unsigned blocks = static_cast<unsigned>((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    for (size_t t = 0; t < T; ++t) {
        simulationStepKernel<<<blocks, THREADS_PER_BLOCK>>>(
            n, t, dev.wT, dev.tauT, dev.rho, dev.radE, dev.radB);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

__global__ void distanceKernel(size_t n, size_t T, size_t src,
                               const float* __restrict__ radB,
                               float* __restrict__ distances) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float maxCorr = ZERO;
    int bestT = 0;

    // Discrete cross-correlation to find time delay
    for (size_t t = 0; t < T; ++t) {
        float sum = ZERO;
        for (size_t tt = t; tt < T; ++tt) {
            const float pB = radB[tt * n + i];
            const float pS = radB[(tt - t) * n + src];
            sum += pS * pB;
        }

        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }

    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
}

void computeDistances(SimulationState& state, DeviceState& dev) {
    printf("Computing distances via cross-correlation...\n");
    const size_t n = state.numTriangles;
    const size_t T = state.numTimesteps;
    const unsigned blocks = static_cast<unsigned>((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    distanceKernel<<<blocks, THREADS_PER_BLOCK>>>(n, T, state.sourceIndex, dev.radB, dev.distances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), dev.distances, n * sizeof(float), cudaMemcpyDeviceToHost));
}

// Copy full matrices back to the host, only needed for validation
void downloadForValidation(SimulationState& state, const DeviceState& dev) {
    const size_t n = state.numTriangles;
    const size_t T = state.numTimesteps;

    state.kij.assign(n * n, ZERO);
    CUDA_CHECK(cudaMemcpy(state.kij.data(), dev.kij, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    state.radB.assign(T * n, ZERO);
    if (T * n > 0) {
        CUDA_CHECK(cudaMemcpy(state.radB.data(), dev.radB, T * n * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

void freeDeviceState(DeviceState& dev) {
    cudaFree(dev.tris);
    cudaFree(dev.centers);
    cudaFree(dev.areas);
    cudaFree(dev.rho);
    cudaFree(dev.radE);
    cudaFree(dev.radB);
    cudaFree(dev.kij);
    cudaFree(dev.tauT);
    cudaFree(dev.wT);
    cudaFree(dev.distances);
    cudaFree(dev.nodes);
    cudaFree(dev.triList);
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
    int nonZeroKij = 0;
    for (size_t i = 0; i < state.numTriangles * state.numTriangles; ++i) {
        if (state.kij[i] > EPSILON) nonZeroKij++;
    }
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
    DeviceState dev;
    initializeSimulation(state, dev, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state, dev);
    computeFormFactors(state, dev);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, dev);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state, dev);

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
        downloadForValidation(state, dev);
        if (!validateResults(state)) {
            releaseChunkStaging(dev);
            freeDeviceState(dev);
            return 1;
        }
    }

    releaseChunkStaging(dev);
    freeDeviceState(dev);
    return 0;
}
