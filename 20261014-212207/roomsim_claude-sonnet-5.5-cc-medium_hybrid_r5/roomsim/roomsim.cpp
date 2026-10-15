/**
 * Room Response Simulation Benchmark
 * 
 * This is a simplified implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization: MPI distributes the matrix rows (cyclically) over ranks, one GPU
 * per rank runs the CUDA kernels (ray tracing for form factors, radiosity update,
 * cross-correlation), and OpenMP is used for host-side work (octree construction,
 * data preparation, random stream snapshotting overlapped with GPU work).
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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

#pragma omp parallel
#pragma omp single
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
#pragma omp task firstprivate(i) shared(childIndices) if (indices.size() > 256)
                {
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
#pragma omp taskwait
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
// Parallel Environment (MPI + OpenMP + CUDA)
// ============================================================================

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

static int g_rank = 0;
static int g_size = 1;

#define ROOT_PRINTF(...)                      \
    do {                                      \
        if (g_rank == 0) printf(__VA_ARGS__); \
    } while (0)

// Each rank owns the rows i with i % size == rank (cyclic distribution).
// The random stream of the sequential algorithm is reproduced exactly: every
// non-culled (i,j) pair consumes exactly 4*NUM_RAYS draws of the mt19937
// stream, so the stream offset of every row is known in advance. The host
// skips through the generator to obtain a state snapshot per row, and the GPU
// regenerates the stream of each row from its snapshot.

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr int DRAWS_PER_PAIR = 4 * NUM_RAYS;
constexpr int GEN_THREADS = 224;     // <= MT_N - MT_M so one twist round has no intra-round dependencies
constexpr int KIJ_THREADS = 128;
constexpr int STATE_WORDS = MT_N + 1;  // state + position
constexpr float CULL_THRESHOLD = 0.99f;
constexpr int STACK_SIZE = 96;

// ============================================================================
// Host-side Mersenne Twister with fast skip (state snapshots)
// ============================================================================

struct HostMT {
    uint32_t s[MT_N];
    uint32_t pos;

    explicit HostMT(uint32_t seed) {
        s[0] = seed;
        for (uint32_t i = 1; i < MT_N; ++i) s[i] = 1812433253u * (s[i - 1] ^ (s[i - 1] >> 30)) + i;
        pos = MT_N;
    }

    static inline uint32_t mix(uint32_t a, uint32_t b) {
        uint32_t y = (a & 0x80000000u) | (b & 0x7fffffffu);
        return (y >> 1) ^ ((0u - (y & 1u)) & 0x9908b0dfu);
    }

    void twist() {
        uint32_t* x = s;
        for (int k = 0; k < MT_N - MT_M; ++k) x[k] = x[k + MT_M] ^ mix(x[k], x[k + 1]);
        for (int k = MT_N - MT_M; k < MT_N - 1; ++k) x[k] = x[k - (MT_N - MT_M)] ^ mix(x[k], x[k + 1]);
        x[MT_N - 1] = x[MT_M - 1] ^ mix(x[MT_N - 1], x[0]);
    }

    void skip(uint64_t count) {
        while (count > 0) {
            if (pos == MT_N) {
                twist();
                pos = 0;
            }
            uint64_t m = std::min<uint64_t>(count, MT_N - pos);
            pos += static_cast<uint32_t>(m);
            count -= m;
        }
    }
};

// ============================================================================
// Device data structures
// ============================================================================

struct GNode {
    float cx, cy, cz, hx, hy, hz;
    int child[8];
    int leafStart, leafCount;
};

struct GTri {  // vertex a, edges e1, e2 (the index rides in the w component of a)
    float4 a, e1, e2;
};

struct DeviceGeometry {
    float4* a;
    float4* b;
    float4* c;
    float4* normal;
    float4* center;
    float* area;
    GNode* nodes;
    GTri* leafTris;
    int numTriangles;
    int rootIsLeaf;
};

// ============================================================================
// Device functions
// ============================================================================

__device__ __forceinline__ float3 f3(const float4& v) { return make_float3(v.x, v.y, v.z); }
__device__ __forceinline__ float3 sub3(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 add3(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 mul3(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 cross3(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// Culling decision; evaluated with explicit roundings so that every kernel agrees.
__device__ __forceinline__ bool culledPair(const float4& ni, const float4& nj) {
    float d = __fadd_rn(__fadd_rn(__fmul_rn(ni.x, nj.x), __fmul_rn(ni.y, nj.y)), __fmul_rn(ni.z, nj.z));
    return d > CULL_THRESHOLD;
}

__device__ __forceinline__ float mtFloat(uint32_t y) {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    float f = __uint2float_rn(y) * (1.0f / 4294967296.0f);
    return f >= 1.0f ? 0.99999994f : f;
}

__device__ __forceinline__ bool rayIntersectsBox(const GNode& nd, float3 p1, float3 p2) {
    float3 d = mul3(sub3(p2, p1), 0.5f);
    float3 c = make_float3(p1.x + d.x - nd.cx, p1.y + d.y - nd.cy, p1.z + d.z - nd.cz);
    float adx = fabsf(d.x), ady = fabsf(d.y), adz = fabsf(d.z);

    if (fabsf(c.x) > nd.hx + adx) return false;
    if (fabsf(c.y) > nd.hy + ady) return false;
    if (fabsf(c.z) > nd.hz + adz) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > nd.hy * adz + nd.hz * ady + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > nd.hz * adx + nd.hx * adz + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > nd.hx * ady + nd.hy * adx + EPSILON) return false;
    return true;
}

// Returns true if any triangle in the leaf blocks the ray
__device__ __forceinline__ bool leafBlocks(const GNode& nd, const GTri* leafTris, float3 from, float3 dirNorm,
                                           float rayLen, int src, int dst) {
    for (int k = 0; k < nd.leafCount; ++k) {
        const GTri t = leafTris[nd.leafStart + k];
        int idx = __float_as_int(t.a.w);
        if (idx == src || idx == dst) continue;

        float3 e1 = f3(t.e1), e2 = f3(t.e2);
        float3 pvec = cross3(dirNorm, e2);
        float det = dot3(e1, pvec);
        if (fabsf(det) < EPSILON) continue;
        float invDet = 1.0f / det;
        float3 tvec = sub3(from, f3(t.a));
        float u = dot3(tvec, pvec) * invDet;
        if (u < 0.0f || u > 1.0f) continue;
        float3 qvec = cross3(tvec, e1);
        float v = dot3(dirNorm, qvec) * invDet;
        if (v < 0.0f || u + v > 1.0f) continue;
        float dist = dot3(e2, qvec) * invDet;
        if (dist > EPSILON && dist < rayLen - EPSILON) return true;
    }
    return false;
}

__device__ bool isRayBlocked(const DeviceGeometry& g, float3 from, float3 to, int src, int dst) {
    float3 dir = sub3(to, from);
    float rayLen = sqrtf(dot3(dir, dir));
    if (rayLen < EPSILON) return true;
    float3 dirNorm = make_float3(dir.x / rayLen, dir.y / rayLen, dir.z / rayLen);

    if (g.rootIsLeaf) return leafBlocks(g.nodes[0], g.leafTris, from, dirNorm, rayLen, src, dst);

    int stack[STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const GNode nd = g.nodes[stack[--sp]];
        if (nd.leafCount > 0) {
            if (leafBlocks(nd, g.leafTris, from, dirNorm, rayLen, src, dst)) return true;
            continue;
        }
        for (int c = 0; c < 8; ++c) {
            int ch = nd.child[c];
            if (ch >= 0 && rayIntersectsBox(g.nodes[ch], from, to)) {
                if (sp < STACK_SIZE) stack[sp++] = ch;
            }
        }
    }
    return false;
}

__device__ __forceinline__ float3 samplePoint(float3 a, float3 ab, float3 ac, float u, float v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    return add3(add3(a, mul3(ab, u)), mul3(ac, v));
}

__device__ __forceinline__ float cosPhiDev(float3 v, float3 n) {
    float vNorm = sqrtf(dot3(v, v));
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, dot3(v, n) / vNorm);
}

// ============================================================================
// Kernels
// ============================================================================

// Per row: number of non-culled pairs, and the exclusive prefix per column chunk.
__global__ void countPairsKernel(const float4* __restrict__ normals, int n, int nChunks,
                                 int* __restrict__ rowCnt, int* __restrict__ chunkOff) {
    int i = blockIdx.x;
    float4 ni = normals[i];
    int running = 0;
    for (int c = 0; c < nChunks; ++c) {
        int j = c * KIJ_THREADS + threadIdx.x;
        bool flag = j < n && j != i && !culledPair(ni, normals[j]);
        int cnt = __syncthreads_count(flag);
        if (threadIdx.x == 0) chunkOff[(size_t)i * nChunks + c] = running;
        running += cnt;
    }
    if (threadIdx.x == 0) rowCnt[i] = running;
}

// Regenerates the mt19937 stream of each row from its state snapshot.
__global__ void __launch_bounds__(GEN_THREADS)
generateStreamKernel(const uint32_t* __restrict__ states, const int* __restrict__ rowCnt, int rank, int nranks,
                     int rowBase, float* __restrict__ out, size_t slotStride) {
    __shared__ uint32_t s[MT_N];
    const int tid = threadIdx.x;
    const int li = rowBase + blockIdx.x;
    const int i = rank + li * nranks;
    const uint64_t total = (uint64_t)DRAWS_PER_PAIR * (uint64_t)rowCnt[i];
    const uint32_t* st = states + (size_t)blockIdx.x * STATE_WORDS;
    float* o = out + (size_t)blockIdx.x * slotStride;

    for (int k = tid; k < MT_N; k += GEN_THREADS) s[k] = st[k];
    const uint32_t pos = st[MT_N];
    __syncthreads();

    uint64_t emitted = 0;
    // Remaining words of the current block
    for (uint32_t q = pos + tid; q < MT_N; q += GEN_THREADS) {
        uint64_t e = q - pos;
        if (e < total) o[e] = mtFloat(s[q]);
    }
    emitted = MT_N - pos;

    while (emitted < total) {
#pragma unroll
        for (int g = 0; g < 3; ++g) {
            int k = g * GEN_THREADS + tid;
            bool active = k < MT_N;
            uint32_t a = 0, b = 0, c = 0;
            if (active) {
                a = s[k];
                b = s[k + 1 < MT_N ? k + 1 : 0];
                c = s[k + MT_M < MT_N ? k + MT_M : k + MT_M - MT_N];
            }
            __syncthreads();
            if (active) {
                uint32_t y = (a & 0x80000000u) | (b & 0x7fffffffu);
                uint32_t nv = c ^ (y >> 1) ^ ((0u - (y & 1u)) & 0x9908b0dfu);
                s[k] = nv;
                uint64_t e = emitted + k;
                if (e < total) o[e] = mtFloat(nv);
            }
            __syncthreads();
        }
        emitted += MT_N;
    }
}

// Form factors for a batch of rows (one thread per (row, column) pair).
__global__ void __launch_bounds__(KIJ_THREADS)
kijKernel(DeviceGeometry g, const float* __restrict__ stream, size_t slotStride, const int* __restrict__ chunkOff,
          int nChunks, int rank, int nranks, int rowBase, int mPad, float* __restrict__ WT,
          unsigned long long* __restrict__ nonZeroCount) {
    __shared__ int warpCnt[KIJ_THREADS / 32];
    const int n = g.numTriangles;
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int li = rowBase + blockIdx.y;
    const int i = rank + li * nranks;
    const int j = blockIdx.x * KIJ_THREADS + tid;

    const float4 ni = g.normal[i];
    bool flag = j < n && j != i;
    float4 nj = make_float4(0, 0, 0, 0);
    if (flag) {
        nj = g.normal[j];
        flag = !culledPair(ni, nj);
    }

    // Block-wide exclusive scan of flags -> index of this pair inside the chunk
    unsigned ballot = __ballot_sync(0xffffffffu, flag);
    if (lane == 0) warpCnt[warp] = __popc(ballot);
    __syncthreads();
    int off = __popc(ballot & ((1u << lane) - 1u));
    for (int w = 0; w < warp; ++w) off += warpCnt[w];

    float kij = ZERO;
    if (flag) {
        const float4* rnd = reinterpret_cast<const float4*>(
            stream + (size_t)blockIdx.y * slotStride +
            (size_t)DRAWS_PER_PAIR * (size_t)(chunkOff[(size_t)i * nChunks + blockIdx.x] + off));

        const float3 ai = f3(g.a[i]);
        const float3 abI = sub3(f3(g.b[i]), ai);
        const float3 acI = sub3(f3(g.c[i]), ai);
        const float3 aj = f3(g.a[j]);
        const float3 abJ = sub3(f3(g.b[j]), aj);
        const float3 acJ = sub3(f3(g.c[j]), aj);
        const float3 nI = f3(ni), nJ = f3(nj);

        for (int r = 0; r < NUM_RAYS; ++r) {
            float4 u = rnd[r];
            float3 pI = samplePoint(ai, abI, acI, u.x, u.y);
            float3 pJ = samplePoint(aj, abJ, acJ, u.z, u.w);

            if (isRayBlocked(g, pI, pJ, i, j)) continue;

            float3 v = sub3(pJ, pI);
            float distSqr = dot3(v, v);
            if (distSqr < EPSILON) continue;

            float cosPhiI = cosPhiDev(v, nI);
            float cosPhiJ = cosPhiDev(make_float3(-v.x, -v.y, -v.z), nJ);
            if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

            kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
        }
        kij *= INV_NUM_RAYS;
    }

    if (j < n) {
        // Weight actually used by the simulation: min(kij * area_j, 1); zero if kij <= 0
        float w = kij > ZERO ? fminf(kij * g.area[j], ONE) : ZERO;
        if (w != ZERO) WT[(size_t)j * mPad + li] = w;
    }
    unsigned nz = __ballot_sync(0xffffffffu, j < n && kij > EPSILON);
    if (lane == 0 && nz) atomicAdd(nonZeroCount, (unsigned long long)__popc(nz));
}

__global__ void tauKernel(const float4* __restrict__ center, int n, int m, int mPad, int rank, int nranks,
                          int* __restrict__ TT) {
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    if (li >= m) return;
    int i = rank + li * nranks;
    float4 ci = center[i], cj = center[j];
    float dx = ci.x - cj.x, dy = ci.y - cj.y, dz = ci.z - cj.z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    TT[(size_t)j * mPad + li] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// One timestep of the radiosity update for the rows owned by this rank. The sum over j
// is kept strictly sequential (same association order as the sequential code).
__global__ void simStepKernel(int t, int n, int m, int mPad, int rank, int nranks, const int* __restrict__ TT,
                              const float* __restrict__ WT, const float* __restrict__ radB, float rho,
                              int sourceIdx, int timeOff, float* __restrict__ out) {
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= m) return;
    int i = rank + li * nranks;
    float sumB = ZERO;
#pragma unroll 8
    for (int j = 0; j < n; ++j) {
        size_t o = (size_t)j * mPad + li;
        int tau = TT[o];
        if (t < tau) continue;
        float w = WT[o];
        if (w <= ZERO) continue;
        float radJ = radB[(size_t)(t - tau) * n + j];
        if (radJ <= ZERO) continue;
        sumB = fmaf(w, radJ, sumB);
    }
    float radE = (i == sourceIdx && t < timeOff) ? 1.0f : 0.0f;
    out[li] = fmaf(rho, sumB, radE);
}

__global__ void transposeKernel(const float* __restrict__ radB, int n, int T, int m, int rank, int nranks,
                                int sourceIdx, float* __restrict__ radBT, float* __restrict__ srcCol) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int li = blockIdx.y;
    if (t >= T) return;
    int i = rank + li * nranks;
    radBT[(size_t)li * T + t] = radB[(size_t)t * n + i];
    if (li == 0) srcCol[t] = radB[(size_t)t * n + sourceIdx];
}

// Cross-correlation: one block per owned row, threads over lags.
__global__ void distanceKernel(const float* __restrict__ radBT, const float* __restrict__ srcCol, int T,
                               float* __restrict__ out) {
    __shared__ float bestVal[256];
    __shared__ int bestT[256];
    const int li = blockIdx.x;
    const float* pB = radBT + (size_t)li * T;

    float maxCorr = ZERO;
    int best = 0;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        float sum = ZERO;
        for (int tt = t; tt < T; ++tt) sum = fmaf(srcCol[tt - t], pB[tt], sum);
        if (sum > maxCorr) {
            maxCorr = sum;
            best = t;
        }
    }
    bestVal[threadIdx.x] = maxCorr;
    bestT[threadIdx.x] = best;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float v2 = bestVal[threadIdx.x + s];
            int t2 = bestT[threadIdx.x + s];
            if (v2 > bestVal[threadIdx.x] || (v2 == bestVal[threadIdx.x] && t2 < bestT[threadIdx.x])) {
                bestVal[threadIdx.x] = v2;
                bestT[threadIdx.x] = t2;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[li] = WAVE_SPEED * static_cast<val_t>(bestT[0]);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> distances;   // Computed distances from source (all rows, on every rank)
    std::vector<val_t> radB;        // Reflected radiosity (T x N), only gathered when validating
    uint64_t nonZeroKij = 0;
    val_t reflectivity = 0;

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // Parallel decomposition
    size_t localRows = 0;           // rows owned by this rank
    size_t mPad = 0;                // max rows owned by any rank
    DeviceGeometry geo{};
    float* d_radB = nullptr;        // T x N, replicated on every device
    float* d_WT = nullptr;          // [j][li] weights min(kij*area_j, 1)
    int* d_TT = nullptr;            // [j][li] time delays

    size_t rowsOf(int r) const { return numTriangles > (size_t)r ? (numTriangles - r + g_size - 1) / g_size : 0; }
};

// ============================================================================
// Octree flattening and device upload
// ============================================================================

static int flattenOctree(const Octree& node, std::vector<GNode>& nodes, std::vector<GTri>& leafTris,
                         const std::vector<Triangle>& tris) {
    int id = static_cast<int>(nodes.size());
    nodes.emplace_back();
    GNode gn{};
    gn.cx = node.center.x; gn.cy = node.center.y; gn.cz = node.center.z;
    gn.hx = node.halfExtent.x; gn.hy = node.halfExtent.y; gn.hz = node.halfExtent.z;
    for (int c = 0; c < 8; ++c) gn.child[c] = -1;
    gn.leafStart = 0;
    gn.leafCount = 0;
    if (!node.triangleIndices.empty()) {
        gn.leafStart = static_cast<int>(leafTris.size());
        gn.leafCount = static_cast<int>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) {
            const Triangle& t = tris[idx];
            Vec3 e1 = t.b - t.a, e2 = t.c - t.a;
            GTri gt;
            gt.a = make_float4(t.a.x, t.a.y, t.a.z, 0.0f);
            int ii = static_cast<int>(idx);
            memcpy(&gt.a.w, &ii, sizeof(int));
            gt.e1 = make_float4(e1.x, e1.y, e1.z, 0.0f);
            gt.e2 = make_float4(e2.x, e2.y, e2.z, 0.0f);
            leafTris.push_back(gt);
        }
    } else {
        for (int c = 0; c < 8; ++c) {
            if (node.children[c]) gn.child[c] = flattenOctree(*node.children[c], nodes, leafTris, tris);
        }
    }
    nodes[id] = gn;
    return id;
}

template <typename T>
static T* deviceCopy(const std::vector<T>& v) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

static void uploadGeometry(SimulationState& state) {
    const size_t n = state.numTriangles;
    std::vector<float4> a(n), b(n), c(n), nrm(n), cen(n);
    std::vector<float> area(n);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const Triangle& t = state.triangles[i];
        Vec3 ctr = t.center();
        Vec3 nn = t.normal();
        a[i] = make_float4(t.a.x, t.a.y, t.a.z, 0);
        b[i] = make_float4(t.b.x, t.b.y, t.b.z, 0);
        c[i] = make_float4(t.c.x, t.c.y, t.c.z, 0);
        nrm[i] = make_float4(nn.x, nn.y, nn.z, 0);
        cen[i] = make_float4(ctr.x, ctr.y, ctr.z, 0);
        area[i] = t.area();
    }

    std::vector<GNode> nodes;
    std::vector<GTri> leafTris;
    flattenOctree(state.octree, nodes, leafTris, state.triangles);

    DeviceGeometry& g = state.geo;
    g.a = deviceCopy(a);
    g.b = deviceCopy(b);
    g.c = deviceCopy(c);
    g.normal = deviceCopy(nrm);
    g.center = deviceCopy(cen);
    g.area = deviceCopy(area);
    g.nodes = deviceCopy(nodes);
    g.leafTris = deviceCopy(leafTris);
    g.numTriangles = static_cast<int>(n);
    g.rootIsLeaf = nodes[0].leafCount > 0 ? 1 : 0;
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;

    ROOT_PRINTF("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    ROOT_PRINTF("Building octree...\n");
    state.octree.build(state.triangles);

    state.distances.assign(state.numTriangles, ZERO);

    state.localRows = state.rowsOf(g_rank);
    state.mPad = state.rowsOf(0);

    uploadGeometry(state);

    const size_t n = state.numTriangles;
    CUDA_CHECK(cudaMalloc(&state.d_radB, std::max<size_t>(timesteps * n, 1) * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, std::max<size_t>(timesteps * n, 1) * sizeof(float)));
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    ROOT_PRINTF("Computing form factors (Kij)...\n");

    const int n = static_cast<int>(state.numTriangles);
    const int m = static_cast<int>(state.localRows);
    const int mPad = static_cast<int>(state.mPad);
    const int nChunks = (n + KIJ_THREADS - 1) / KIJ_THREADS;

    int *d_rowCnt, *d_chunkOff;
    CUDA_CHECK(cudaMalloc(&d_rowCnt, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_chunkOff, (size_t)n * nChunks * sizeof(int)));
    countPairsKernel<<<n, KIJ_THREADS>>>(state.geo.normal, n, nChunks, d_rowCnt, d_chunkOff);
    CUDA_CHECK(cudaGetLastError());
    std::vector<int> rowCnt(n);
    CUDA_CHECK(cudaMemcpy(rowCnt.data(), d_rowCnt, n * sizeof(int), cudaMemcpyDeviceToHost));
    int maxCnt = 1;
    for (int c : rowCnt) maxCnt = std::max(maxCnt, c);

    CUDA_CHECK(cudaMalloc(&state.d_WT, (size_t)n * mPad * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.d_WT, 0, (size_t)n * mPad * sizeof(float)));

    // Rows are processed in batches; two buffers allow overlapping stream generation and tracing.
    const size_t slotStride = (size_t)DRAWS_PER_PAIR * maxCnt;
    size_t batch = std::max<size_t>(1, (size_t)(1u << 30) / (slotStride * sizeof(float)));
    batch = std::min<size_t>(std::min<size_t>(batch, 1024), std::max(1, m));

    float* d_stream[2];
    uint32_t* d_states[2];
    cudaStream_t streams[2];
    unsigned long long* d_nz;
    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaMalloc(&d_stream[s], batch * slotStride * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_states[s], batch * STATE_WORDS * sizeof(uint32_t)));
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }
    CUDA_CHECK(cudaMalloc(&d_nz, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_nz, 0, sizeof(unsigned long long)));

    uint32_t* h_states = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_states, std::max<size_t>(m, 1) * STATE_WORDS * sizeof(uint32_t)));
    std::atomic<int> ready(0);

    // Producer: walks the sequential random stream and snapshots the generator at each owned row.
    auto produce = [&]() {
        HostMT mt(42);
        int q = 0;
        for (int i = 0; i < n && q < m; ++i) {
            if (i % g_size == g_rank) {
                uint32_t* dst = h_states + (size_t)q * STATE_WORDS;
                memcpy(dst, mt.s, sizeof(mt.s));
                dst[MT_N] = mt.pos;
                ++q;
                ready.store(q, std::memory_order_release);
            }
            mt.skip((uint64_t)DRAWS_PER_PAIR * (uint64_t)rowCnt[i]);
        }
    };

    auto consume = [&]() {
        size_t k = 0;
        for (size_t b0 = 0; b0 < (size_t)m; b0 += batch, ++k) {
            int slot = k & 1;
            CUDA_CHECK(cudaStreamSynchronize(streams[slot]));
            size_t bb = std::min<size_t>(batch, m - b0);
            while (ready.load(std::memory_order_acquire) < (int)(b0 + bb)) std::this_thread::yield();

            CUDA_CHECK(cudaMemcpyAsync(d_states[slot], h_states + b0 * STATE_WORDS,
                                       bb * STATE_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice,
                                       streams[slot]));
            generateStreamKernel<<<(unsigned)bb, GEN_THREADS, 0, streams[slot]>>>(
                d_states[slot], d_rowCnt, g_rank, g_size, (int)b0, d_stream[slot], slotStride);
            kijKernel<<<dim3(nChunks, (unsigned)bb), KIJ_THREADS, 0, streams[slot]>>>(
                state.geo, d_stream[slot], slotStride, d_chunkOff, nChunks, g_rank, g_size, (int)b0, mPad,
                state.d_WT, d_nz);
            CUDA_CHECK(cudaGetLastError());
            if (g_rank == 0) {
                printf("  Progress: %zu/%zu triangles\n", std::min<size_t>((b0 + bb) * g_size, n), (size_t)n);
            }
        }
        for (int s = 0; s < 2; ++s) CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    };

    if (m > 0) {
#pragma omp parallel num_threads(2)
        {
            if (omp_get_num_threads() == 1) {
                produce();
                consume();
            } else if (omp_get_thread_num() == 1) {
                produce();
            } else {
                consume();
            }
        }
    }

    unsigned long long nzLocal = 0, nzTotal = 0;
    CUDA_CHECK(cudaMemcpy(&nzLocal, d_nz, sizeof(nzLocal), cudaMemcpyDeviceToHost));
    MPI_Allreduce(&nzLocal, &nzTotal, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    state.nonZeroKij = nzTotal;

    for (int s = 0; s < 2; ++s) {
        cudaFree(d_stream[s]);
        cudaFree(d_states[s]);
        cudaStreamDestroy(streams[s]);
    }
    cudaFree(d_nz);
    cudaFree(d_rowCnt);
    cudaFree(d_chunkOff);
    cudaFreeHost(h_states);
}

void computeTimeDelays(SimulationState& state) {
    ROOT_PRINTF("Computing time delays (Tau)...\n");
    const int n = static_cast<int>(state.numTriangles);
    const int m = static_cast<int>(state.localRows);
    CUDA_CHECK(cudaMalloc(&state.d_TT, (size_t)n * state.mPad * sizeof(int)));
    CUDA_CHECK(cudaMemset(state.d_TT, 0, (size_t)n * state.mPad * sizeof(int)));
    if (m > 0) {
        tauKernel<<<dim3((m + 127) / 128, n), 128>>>(state.geo.center, n, m, (int)state.mPad, g_rank, g_size,
                                                     state.d_TT);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// Gathers a per-rank vector (rows owned cyclically) into global row order on every rank.
static void allgatherRows(const std::vector<float>& local, std::vector<float>& gathered,
                          std::vector<float>& global, size_t n, size_t mPad) {
    MPI_Allgather(local.data(), (int)mPad, MPI_FLOAT, gathered.data(), (int)mPad, MPI_FLOAT, MPI_COMM_WORLD);
    for (size_t i = 0; i < n; ++i) global[i] = gathered[(i % g_size) * mPad + i / g_size];
}

void runSimulation(SimulationState& state) {
    ROOT_PRINTF("Running wave propagation simulation...\n");

    const int n = static_cast<int>(state.numTriangles);
    const int m = static_cast<int>(state.localRows);
    const size_t mPad = state.mPad;
    const int T = static_cast<int>(state.numTimesteps);
    const int timeOff = T / 2;

    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, std::max<size_t>(mPad, 1) * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, std::max<size_t>(mPad, 1) * sizeof(float)));
    std::vector<float> hLocal(mPad, 0.0f), hGathered(mPad * g_size), hRow(n);

    for (int t = 0; t < T; ++t) {
        if (m > 0) {
            simStepKernel<<<(m + 31) / 32, 32>>>(t, n, m, (int)mPad, g_rank, g_size, state.d_TT, state.d_WT,
                                                  state.d_radB, state.reflectivity, (int)state.sourceIndex,
                                                  timeOff, d_out);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(hLocal.data(), d_out, m * sizeof(float), cudaMemcpyDeviceToHost));
        }
        allgatherRows(hLocal, hGathered, hRow, n, mPad);
        CUDA_CHECK(cudaMemcpy(state.d_radB + (size_t)t * n, hRow.data(), n * sizeof(float),
                              cudaMemcpyHostToDevice));

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            ROOT_PRINTF("  Timestep %d/%d\n", t + 1, T);
        }
    }
    cudaFree(d_out);
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    ROOT_PRINTF("Computing distances via cross-correlation...\n");

    const int n = static_cast<int>(state.numTriangles);
    const int m = static_cast<int>(state.localRows);
    const size_t mPad = state.mPad;
    const int T = static_cast<int>(state.numTimesteps);

    std::vector<float> hLocal(mPad, 0.0f), hGathered(mPad * g_size);
    if (m > 0 && T > 0) {
        float *d_radBT, *d_src, *d_dist;
        CUDA_CHECK(cudaMalloc(&d_radBT, (size_t)m * T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_src, (size_t)T * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dist, (size_t)m * sizeof(float)));
        transposeKernel<<<dim3((T + 127) / 128, m), 128>>>(state.d_radB, n, T, m, g_rank, g_size,
                                                           (int)state.sourceIndex, d_radBT, d_src);
        CUDA_CHECK(cudaGetLastError());
        distanceKernel<<<m, 128>>>(d_radBT, d_src, T, d_dist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hLocal.data(), d_dist, m * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_radBT);
        cudaFree(d_src);
        cudaFree(d_dist);
    }
    allgatherRows(hLocal, hGathered, state.distances, n, mPad);
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
#pragma omp parallel for reduction(+ : receivedEnergy) schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[t * state.numTriangles + i] > EPSILON) {
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

static void selectDevice() {
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int localRank = 0;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_free(&local);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    selectDevice();

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
            if (g_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (timesteps < 0) timesteps = 0;

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    ROOT_PRINTF("Room Response Simulation Benchmark\n");
    ROOT_PRINTF("===================================\n");
    ROOT_PRINTF("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    ROOT_PRINTF("Timesteps: %d\n", timesteps);
    ROOT_PRINTF("Source triangle: %d\n", sourceIdx);
    ROOT_PRINTF("Reflectivity: %.2f\n", reflectivity);
    ROOT_PRINTF("Validation: %s\n", validate ? "enabled" : "disabled");
    ROOT_PRINTF("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    ROOT_PRINTF("\n");

    auto now = [] {
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
        return std::chrono::high_resolution_clock::now();
    };

    // Precomputation
    auto startPre = now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    ROOT_PRINTF("Precomputation time: %ld ms\n", preDuration);
    ROOT_PRINTF("\n");

    // Simulation
    auto startSim = now();

    runSimulation(state);

    auto endSim = now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    ROOT_PRINTF("Simulation time: %ld ms\n", simDuration);
    ROOT_PRINTF("\n");

    // Distance computation
    auto startDist = now();

    computeDistances(state);

    auto endDist = now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    ROOT_PRINTF("Distance computation time: %ld ms\n", distDuration);
    ROOT_PRINTF("\n");

    // Total time
    long totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startPre).count();
    MPI_Allreduce(MPI_IN_PLACE, &totalTime, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    ROOT_PRINTF("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    ROOT_PRINTF("\nPerformance:\n");
    ROOT_PRINTF("  Triangles: %zu\n", n);
    ROOT_PRINTF("  Timesteps: %zu\n", t);
    ROOT_PRINTF("  Form factor computations: %.2e\n", kijOps);
    ROOT_PRINTF("  Simulation operations: %.2e\n", simOps);
    ROOT_PRINTF("  Distance computations: %.2e\n", distOps);
    ROOT_PRINTF("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Memory usage
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    ROOT_PRINTF("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    ROOT_PRINTF("  Result hash: %016lX\n", hash);
    ROOT_PRINTF("\n");

    // Print results for external validation
    if (printResults && g_rank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int status = 0;
    if (validate) {
        if (g_rank == 0) {
            state.radB.resize(t * n);
            CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB, t * n * sizeof(float), cudaMemcpyDeviceToHost));
            if (!validateResults(state)) status = 1;
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    cudaFree(state.d_radB);
    cudaFree(state.d_WT);
    cudaFree(state.d_TT);
    MPI_Finalize();
    return status;
}
