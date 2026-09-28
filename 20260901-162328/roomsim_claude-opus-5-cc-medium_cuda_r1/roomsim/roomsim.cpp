/**
 * Room Response Simulation Benchmark
 * 
 * This is a simplified, purely sequential implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * This version is parallelized with CUDA: all four compute phases (time delays,
 * form factors, wave propagation and cross-correlation) run on the GPU.  The
 * mesh and the octree are still built on the host and then uploaded in a flat,
 * GPU friendly layout.
 *
 * The random number stream of the sequential reference implementation is
 * reproduced bit-exactly: every non-culled triangle pair consumes exactly
 * 4 * NUM_RAYS values from a single std::mt19937 sequence, so each pair's
 * offset into that stream can be computed in parallel with a per-row prefix
 * sum.  The raw stream is produced on the host in chunks (which is cheap
 * compared to the ray casting work) and overlapped with the GPU computation
 * through double buffering.
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
#include <vector>

#include <cfloat>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// CUDA Error Handling
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(_err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

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
// GPU Data Layout
// ============================================================================

// Triangle vertices, padded to 16 bytes per vertex for wide aligned loads
struct __align__(16) GTri {
    float4 a, b, c;
};

// Flattened octree node, 32 bytes = exactly one cache sector, fetched with two
// vector loads.  lo = (center, first), hi = (halfExtent, count) where the two
// integers are bit-cast into the w components:
//   count > 0 : leaf, `first` is the offset into the leaf triangle array
//   count < 0 : inner node with -count children stored consecutively at `first`
struct __align__(16) GNode {
    float4 lo;
    float4 hi;
};

// A triangle as referenced by an octree leaf.  Leaf references are stored
// contiguously (duplicated between leaves) so that the innermost loop reads
// straight through memory, with the Möller-Trumbore edges precomputed and the
// triangle index bit-cast into v0.w.
struct __align__(16) GLeafTri {
    float4 v0;
    float4 e1;
    float4 e2;
};

// Number of random values consumed by one (non-culled) triangle pair:
// two random points per ray, two uniform values per point.
constexpr int RANDS_PER_PAIR = 4 * NUM_RAYS;

// Everything the propagation step needs about the pair (i <- j), packed into a
// single 8 byte, coalescable load and indexed transposed (j * N + i):
//   weight = min(Kij * area_j, 1)  (<= 0 means "no contribution")
//   tau    = propagation delay in timesteps
struct PairWeight {
    float weight;
    int tau;
};

// ============================================================================
// Device Vector Helpers
// ============================================================================

__device__ __forceinline__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float3 operator/(float3 a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}
__device__ __forceinline__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}
__device__ __forceinline__ float dev_dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float3 dev_cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float dev_sqNorm(float3 a) { return dev_dot(a, a); }
__device__ __forceinline__ float dev_norm(float3 a) { return sqrtf(dev_sqNorm(a)); }
__device__ __forceinline__ float3 xyz(const float4& v) { return make_float3(v.x, v.y, v.z); }

// ============================================================================
// Device Random Number Generation
// ============================================================================

// MT19937 tempering.  The host only produces the raw state words (which is the
// strictly sequential part); tempering is a per-word function and is applied
// here, on the GPU.
__device__ __forceinline__ uint32_t dev_temper(uint32_t x) {
    x ^= x >> 11;
    x ^= (x << 7) & 0x9d2c5680u;
    x ^= (x << 15) & 0xefc60000u;
    x ^= x >> 18;
    return x;
}

// Reproduce std::uniform_real_distribution<float>(0,1) on top of the raw
// std::mt19937 output word (libstdc++ generate_canonical<float, 24>).
__device__ __forceinline__ float dev_canonical(uint32_t raw) {
    const uint32_t x = dev_temper(raw);
    float r = __uint2float_rn(x) * 2.3283064365386963e-10f;  // x / 2^32
    return r >= 1.0f ? 0.99999994039535522f : r;             // nextafter(1, 0)
}

// Random point inside a triangle using barycentric coordinates
__device__ __forceinline__ float3 dev_randomPointInTriangle(float3 a, float3 b, float3 c,
                                                            uint32_t r0, uint32_t r1) {
    float u = dev_canonical(r0);
    float v = dev_canonical(r1);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    float3 ab = b - a;
    float3 ac = c - a;
    return a + ab * u + ac * v;
}

// ============================================================================
// Device Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

__device__ __forceinline__ float dev_rayTriangleIntersect(float3 orig, float3 dir,
                                                          float3 v0, float3 e1, float3 e2) {
    float3 pvec = dev_cross(dir, e2);
    float det = dev_dot(e1, pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    float invDet = 1.0f / det;
    float3 tvec = orig - v0;
    float u = dev_dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    float3 qvec = dev_cross(tvec, e1);
    float v = dev_dot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return dev_dot(e2, qvec) * invDet;
}

// ============================================================================
// Device Visibility Testing (Octree-accelerated)
// ============================================================================

// The ray dependent quantities (half vector `d`, its absolute value `ad` and
// the segment midpoint `mid` = p1 + d) are hoisted out of the traversal.
__device__ __forceinline__ bool dev_rayIntersectsBox(const GNode& node, float3 mid,
                                                     float3 d, float3 ad) {
    float3 c = mid - xyz(node.lo);
    const float3 h = xyz(node.hi);

    if (fabsf(c.x) > h.x + ad.x) return false;
    if (fabsf(c.y) > h.y + ad.y) return false;
    if (fabsf(c.z) > h.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;

    return true;
}

// Maximum octree depth supported by the traversal stack (verified on the host)
constexpr int MAX_OCTREE_DEPTH = 12;

// Check if a ray between two triangles is blocked by any other triangle.
// Iterative octree traversal, equivalent to Octree::applyToTris().
__device__ bool dev_isRayBlocked(float3 from, float3 to,
                                 const GNode* __restrict__ nodes,
                                 const GLeafTri* __restrict__ leafTris,
                                 int srcTriIdx, int dstTriIdx) {
    float3 dir = to - from;
    float rayLen = dev_norm(dir);
    if (rayLen < EPSILON) return true;
    float3 dirNorm = dir / rayLen;

    const float3 halfVec = dir * 0.5f;
    const float3 mid = from + halfVec;
    const float3 absHalf = make_float3(fabsf(halfVec.x), fabsf(halfVec.y), fabsf(halfVec.z));

    int stack[7 * MAX_OCTREE_DEPTH];
    int sp = 0;
    stack[sp++] = 0;  // root is entered without a box test (as in the reference)

    while (sp > 0) {
        const GNode node = nodes[stack[--sp]];
        const int first = __float_as_int(node.lo.w);
        const int count = __float_as_int(node.hi.w);

        if (count > 0) {
            for (int k = 0; k < count; ++k) {
                const GLeafTri tri = leafTris[first + k];
                const int idx = __float_as_int(tri.v0.w);
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                float dist = dev_rayTriangleIntersect(from, dirNorm, xyz(tri.v0), xyz(tri.e1), xyz(tri.e2));
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;  // Ray is blocked
                }
            }
        } else {
            for (int k = 0; k < -count; ++k) {
                const int c = first + k;
                if (dev_rayIntersectsBox(nodes[c], mid, halfVec, absHalf)) {
                    stack[sp++] = c;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Kernels: Geometry Precomputation
// ============================================================================

__global__ void centersKernel(const GTri* __restrict__ tris, int n, float3* __restrict__ centers) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const GTri t = tris[i];
    centers[i] = (xyz(t.a) + xyz(t.b) + xyz(t.c)) / 3.0f;
}

// Tau (time delay) matrix, stored row-major and transposed
__global__ void tauKernel(const float3* __restrict__ centers, int n,
                          int* __restrict__ tau, int* __restrict__ minTau) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    if (j >= n) return;
    if (i == j) {
        tau[static_cast<size_t>(i) * n + j] = 0;
        return;
    }
    float dist = dev_norm(centers[i] - centers[j]);
    int t = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
    tau[static_cast<size_t>(i) * n + j] = t;
    if (t < 1) atomicMin(minTau, t);
}

// Per-row prefix sum of the random values consumed by each pair.  Pairs that
// are culled (same facing direction) or on the diagonal consume nothing, which
// matches the sequential reference stream exactly.
__global__ void randOffsetKernel(const float3* __restrict__ normals, int n,
                                 uint32_t* __restrict__ relOff,
                                 uint32_t* __restrict__ rowCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float3 ni = normals[i];
    uint32_t acc = 0;
    const size_t row = static_cast<size_t>(i) * n;
    for (int j = 0; j < n; ++j) {
        relOff[row + j] = acc;
        if (i != j && dev_dot(ni, normals[j]) <= 0.99f) acc += RANDS_PER_PAIR;
    }
    rowCount[i] = acc;
}

// ============================================================================
// Kernel: Form Factors (Kij)
// ============================================================================

#ifndef KIJ_BLOCK
#define KIJ_BLOCK 128
#endif
#ifndef KIJ_MIN_BLOCKS
#define KIJ_MIN_BLOCKS 10
#endif

// One thread per triangle pair; the NUM_RAYS samples of a pair are accumulated
// sequentially so that the result is bit-identical to the reference.
__global__ __launch_bounds__(KIJ_BLOCK, KIJ_MIN_BLOCKS)
void kijKernel(const GTri* __restrict__ tris, const float3* __restrict__ normals,
               const float3* __restrict__ centers, const float* __restrict__ areas,
               const GNode* __restrict__ nodes, const GLeafTri* __restrict__ leafTris,
               const uint32_t* __restrict__ rands, const uint32_t* __restrict__ relOff,
               const unsigned long long* __restrict__ rowBase,
               unsigned long long chunkBase, int rowFirst, int n,
               float* __restrict__ kij, PairWeight* __restrict__ weights) {
    // grid.x spans the emitters of a row, grid.y the receivers of the chunk
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int i = rowFirst + static_cast<int>(blockIdx.y);
    if (i == j) return;

    const float3 nI = normals[i];
    const float3 nJ = normals[j];
    // Cull triangles facing the same direction
    if (dev_dot(nI, nJ) > 0.99f) return;

    const GTri triI = tris[i];
    const GTri triJ = tris[j];
    const float3 aI = xyz(triI.a), bI = xyz(triI.b), cI = xyz(triI.c);
    const float3 aJ = xyz(triJ.a), bJ = xyz(triJ.b), cJ = xyz(triJ.c);

    const uint32_t* w = rands + (rowBase[i] - chunkBase + relOff[static_cast<size_t>(i) * n + j]);

    float kacc = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        const uint4 q = *reinterpret_cast<const uint4*>(w + 4 * r);

        float3 pI = dev_randomPointInTriangle(aI, bI, cI, q.x, q.y);
        float3 pJ = dev_randomPointInTriangle(aJ, bJ, cJ, q.z, q.w);

        if (dev_isRayBlocked(pI, pJ, nodes, leafTris, i, j)) continue;

        float3 v = pJ - pI;
        float distSqr = dev_sqNorm(v);
        if (distSqr < EPSILON) continue;

        // cosPhi(v, normal): max(0, dot(v, n) / |v|)
        float vNorm = dev_norm(v);
        if (vNorm <= EPSILON) continue;
        float cosPhiI = fmaxf(ZERO, dev_dot(v, nI) / vNorm);
        float cosPhiJ = fmaxf(ZERO, dev_dot(-v, nJ) / vNorm);

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kacc += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    const float value = kacc * INV_NUM_RAYS;
    kij[static_cast<size_t>(i) * n + j] = value;

    // Fuse the per-pair weight of the propagation step (identical expression,
    // hoisted out of the timestep loop) and the delay into one record
    PairWeight pw;
    pw.weight = fminf(value * areas[j], ONE);
    pw.tau = static_cast<int>(ceilf(dev_norm(centers[i] - centers[j]) * INV_WAVE_SPEED));
    weights[static_cast<size_t>(j) * n + i] = pw;
}

// ============================================================================
// Kernel: Wave Propagation (one timestep)
// ============================================================================

// One thread per receiver triangle.  The reduction over the emitters keeps the
// sequential order of the reference, and the transposed weight matrix makes the
// accesses of a warp fully coalesced.
// The loop is branch free: a skipped emitter contributes fmaf(0, radJ, sum),
// which leaves the accumulator bit-identical while allowing the loads of
// several iterations to be in flight at once.  Because all off-diagonal delays
// are >= 1, only strictly older radiosity rows are read.
__global__ void simKernel(const PairWeight* __restrict__ weights, const float* __restrict__ rho,
                          const float* __restrict__ radE, float* __restrict__ radB,
                          int n, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sumB = ZERO;
    #pragma unroll 8
    for (int j = 0; j < n; ++j) {
        const PairWeight e = weights[static_cast<size_t>(j) * n + i];
        const int srcTime = t - e.tau;
        const bool ok = (e.weight > ZERO) & (srcTime >= 0);
        const float radJ = ok ? radB[static_cast<size_t>(srcTime) * n + j] : ZERO;
        sumB = fmaf((ok & (radJ > ZERO)) ? e.weight : ZERO, radJ, sumB);
    }

    radB[static_cast<size_t>(t) * n + i] = fmaf(rho[i], sumB, radE[static_cast<size_t>(t) * n + i]);
}

// ============================================================================
// Kernels: Distance Computation (Cross-Correlation)
// ============================================================================

// One thread per (triangle, lag) pair; the sum over time keeps sequential order
__global__ void corrKernel(const float* __restrict__ radB, int n, int numT, int sourceIndex,
                           float* __restrict__ corr) {
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<long long>(n) * numT) return;
    const int i = static_cast<int>(tid % n);
    const int t = static_cast<int>(tid / n);

    float sum = ZERO;
    for (int tt = t; tt < numT; ++tt) {
        float pB = radB[static_cast<size_t>(tt) * n + i];
        float pS = radB[static_cast<size_t>(tt - t) * n + sourceIndex];
        sum = fmaf(pS, pB, sum);
    }
    corr[static_cast<size_t>(t) * n + i] = sum;
}

__global__ void argmaxKernel(const float* __restrict__ corr, int n, int numT,
                             float* __restrict__ distances) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float maxCorr = ZERO;
    int bestT = 0;
    for (int t = 0; t < numT; ++t) {
        float sum = corr[static_cast<size_t>(t) * n + i];
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }
    distances[i] = WAVE_SPEED * static_cast<float>(bestT);
}

// ============================================================================
// Host-side MT19937 State Generator
// ============================================================================

// Produces the raw (untempered) state words of std::mt19937 in stream order.
// The twist of one 624 word block is split into ranges without intra-range
// dependencies, which lets the compiler vectorize it; the tempering step is
// applied on the GPU.  The resulting value stream is identical to that of
// std::mt19937 seeded with the same value.
class FastMT19937 {
public:
    explicit FastMT19937(uint32_t seed) : pos_(N) {
        state_[0] = seed;
        for (uint32_t i = 1; i < N; ++i) {
            state_[i] = 1812433253u * (state_[i - 1] ^ (state_[i - 1] >> 30)) + i;
        }
    }

    // Append `count` raw words to `out`
    void generate(uint32_t* out, size_t count) {
        // Drain what is left of the current block
        if (pos_ < N) {
            const size_t take = std::min(count, static_cast<size_t>(N - pos_));
            std::memcpy(out, state_ + pos_, take * sizeof(uint32_t));
            pos_ += static_cast<uint32_t>(take);
            out += take;
            count -= take;
        }

        // Bulk: twist straight into the destination buffer
        const uint32_t* prev = state_;
        while (count >= N) {
            twist(prev, out);
            prev = out;
            out += N;
            count -= N;
        }
        if (prev != state_) std::memcpy(state_, prev, N * sizeof(uint32_t));

        // Tail: keep the remainder of the last block for the next call
        if (count > 0) {
            uint32_t block[N];
            twist(state_, block);
            std::memcpy(state_, block, N * sizeof(uint32_t));
            std::memcpy(out, block, count * sizeof(uint32_t));
            pos_ = static_cast<uint32_t>(count);
        } else {
            pos_ = N;
        }
    }

private:
    static constexpr uint32_t N = 624;
    static constexpr uint32_t M = 397;

    static inline uint32_t twiddle(uint32_t x, uint32_t y) {
        const uint32_t v = (x & 0x80000000u) | (y & 0x7fffffffu);
        return (v >> 1) ^ (static_cast<uint32_t>(-static_cast<int32_t>(y & 1u)) & 0x9908b0dfu);
    }

    static void twist(const uint32_t* __restrict old, uint32_t* __restrict neu) {
        for (uint32_t i = 0; i < N - M; ++i)          // 0 .. 226, reads old only
            neu[i] = old[i + M] ^ twiddle(old[i], old[i + 1]);
        for (uint32_t i = N - M; i < 2 * (N - M); ++i)  // 227 .. 453
            neu[i] = neu[i - (N - M)] ^ twiddle(old[i], old[i + 1]);
        for (uint32_t i = 2 * (N - M); i < N - 1; ++i)  // 454 .. 622
            neu[i] = neu[i - (N - M)] ^ twiddle(old[i], old[i + 1]);
        neu[N - 1] = neu[M - 1] ^ twiddle(old[N - 1], neu[0]);
    }

    alignas(64) uint32_t state_[N];
    uint32_t pos_;
};

// ============================================================================
// GPU Resources
// ============================================================================

struct GpuData {
    GTri* tris = nullptr;
    float3* normals = nullptr;
    float3* centers = nullptr;
    GNode* nodes = nullptr;
    GLeafTri* leafTris = nullptr;

    float* kij = nullptr;    // N x N, row-major
    int* tau = nullptr;      // N x N, row-major
    PairWeight* weights = nullptr;  // N x N, transposed (weight + delay)
    float* areas = nullptr;
    float* rho = nullptr;
    float* radE = nullptr;
    float* radB = nullptr;
    float* corr = nullptr;
    float* distances = nullptr;

    uint32_t* relOff = nullptr;              // per-pair offset within its row
    unsigned long long* rowBase = nullptr;   // absolute stream offset of each row
    uint32_t* randBuf[2] = {nullptr, nullptr};
    uint32_t* hostRand[2] = {nullptr, nullptr};
    cudaEvent_t copyDone[2] = {nullptr, nullptr};
    cudaStream_t stream = nullptr;      // main stream
    cudaStream_t chunkStream[2] = {nullptr, nullptr};  // one per staging buffer

    std::vector<unsigned long long> rowBaseHost;
    size_t chunkCap = 0;
};

// Flatten the host octree into the GPU layout: breadth first, with the
// children of a node stored consecutively and the triangles of a leaf inlined.
static void flattenOctree(const Octree& root, const std::vector<Triangle>& triangles,
                          std::vector<GNode>& nodes, std::vector<GLeafTri>& leafTris,
                          int& maxDepth) {
    struct Item {
        const Octree* node;
        int slot;
        int depth;
    };

    auto asFloat = [](int v) {
        float f;
        std::memcpy(&f, &v, sizeof(f));
        return f;
    };

    nodes.assign(1, GNode{});
    std::vector<Item> queue{{&root, 0, 1}};
    maxDepth = 0;

    for (size_t q = 0; q < queue.size(); ++q) {
        const Item item = queue[q];
        const Octree& node = *item.node;
        maxDepth = std::max(maxDepth, item.depth);

        int first = 0;
        int count = 0;

        if (!node.triangleIndices.empty()) {
            first = static_cast<int>(leafTris.size());
            count = static_cast<int>(node.triangleIndices.size());
            for (size_t idx : node.triangleIndices) {
                const Triangle& tr = triangles[idx];
                const Vec3 e1 = tr.b - tr.a;
                const Vec3 e2 = tr.c - tr.a;
                GLeafTri lt;
                lt.v0 = make_float4(tr.a.x, tr.a.y, tr.a.z, asFloat(static_cast<int>(idx)));
                lt.e1 = make_float4(e1.x, e1.y, e1.z, 0.0f);
                lt.e2 = make_float4(e2.x, e2.y, e2.z, 0.0f);
                leafTris.push_back(lt);
            }
        } else {
            int numChildren = 0;
            for (int k = 0; k < 8; ++k) numChildren += node.children[k] ? 1 : 0;
            first = static_cast<int>(nodes.size());
            count = -numChildren;
            nodes.resize(nodes.size() + numChildren);
            int slot = first;
            for (int k = 0; k < 8; ++k) {
                if (node.children[k]) {
                    queue.push_back({node.children[k].get(), slot++, item.depth + 1});
                }
            }
        }

        GNode g;
        g.lo = make_float4(node.center.x, node.center.y, node.center.z, asFloat(first));
        g.hi = make_float4(node.halfExtent.x, node.halfExtent.y, node.halfExtent.z, asFloat(count));
        nodes[item.slot] = g;
    }
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major)
    std::vector<int> tau;           // Time delays (N x N matrix, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure
    GpuData gpu;                    // Device mirrors of the above

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

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

    // Initialize matrices (the N x N matrices and the radiosity history live on
    // the device; host copies are only materialized for validation)
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // ---- Upload geometry, octree and simulation buffers to the GPU ----
    printf("Uploading data to GPU...\n");
    GpuData& g = state.gpu;
    const size_t n = state.numTriangles;
    const size_t nn = n * n;
    const size_t tn = timesteps * n;

    std::vector<GTri> hTris(n);
    std::vector<float3> hNormals(n);
    for (size_t i = 0; i < n; ++i) {
        const Triangle& tr = state.triangles[i];
        hTris[i].a = make_float4(tr.a.x, tr.a.y, tr.a.z, 0.0f);
        hTris[i].b = make_float4(tr.b.x, tr.b.y, tr.b.z, 0.0f);
        hTris[i].c = make_float4(tr.c.x, tr.c.y, tr.c.z, 0.0f);
        hNormals[i] = make_float3(tr._normal.x, tr._normal.y, tr._normal.z);
    }

    std::vector<GNode> hNodes;
    std::vector<GLeafTri> hLeafTris;
    int maxDepth = 0;
    flattenOctree(state.octree, state.triangles, hNodes, hLeafTris, maxDepth);
    if (maxDepth > MAX_OCTREE_DEPTH) {
        fprintf(stderr, "Octree depth %d exceeds device traversal stack (%d)\n",
                maxDepth, MAX_OCTREE_DEPTH);
        exit(EXIT_FAILURE);
    }
    printf("Octree: %zu nodes, %zu triangle references, depth %d\n",
           hNodes.size(), hLeafTris.size(), maxDepth);

    CUDA_CHECK(cudaMalloc(&g.tris, n * sizeof(GTri)));
    CUDA_CHECK(cudaMalloc(&g.normals, n * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&g.centers, n * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&g.nodes, hNodes.size() * sizeof(GNode)));
    CUDA_CHECK(cudaMalloc(&g.leafTris, hLeafTris.size() * sizeof(GLeafTri)));
    CUDA_CHECK(cudaMalloc(&g.kij, nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.tau, nn * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.weights, nn * sizeof(PairWeight)));
    CUDA_CHECK(cudaMalloc(&g.relOff, nn * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&g.rowBase, (n + 1) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&g.areas, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.rho, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.radE, tn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.radB, tn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.corr, tn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.distances, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(g.tris, hTris.data(), n * sizeof(GTri), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.normals, hNormals.data(), n * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.nodes, hNodes.data(), hNodes.size() * sizeof(GNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.leafTris, hLeafTris.data(), hLeafTris.size() * sizeof(GLeafTri),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.areas, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.rho, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.radE, state.radE.data(), tn * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(g.radB, 0, tn * sizeof(float)));
    CUDA_CHECK(cudaMemset(g.kij, 0, nn * sizeof(float)));
    CUDA_CHECK(cudaMemset(g.weights, 0, nn * sizeof(PairWeight)));

    // Staging buffers for the random number stream (double buffered)
    // Large enough to keep the GPU busy, but never larger than the whole
    // stream, and always big enough to hold a full row
    const size_t maxRands = static_cast<size_t>(RANDS_PER_PAIR) * n * n;
    g.chunkCap = std::max<size_t>(std::min<size_t>(size_t(1) << 24, maxRands),
                                  static_cast<size_t>(RANDS_PER_PAIR) * n);
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&g.randBuf[b], g.chunkCap * sizeof(uint32_t)));
        CUDA_CHECK(cudaHostAlloc(&g.hostRand[b], g.chunkCap * sizeof(uint32_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaEventCreate(&g.copyDone[b]));
    }
    CUDA_CHECK(cudaStreamCreate(&g.stream));
    CUDA_CHECK(cudaStreamCreate(&g.chunkStream[0]));
    CUDA_CHECK(cudaStreamCreate(&g.chunkStream[1]));

    centersKernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(g.tris, static_cast<int>(n), g.centers);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");

    GpuData& g = state.gpu;
    const int n = static_cast<int>(state.numTriangles);

    // Determine how many random values every pair consumes and where in the
    // global std::mt19937 stream its slice starts.
    uint32_t* dRowCount = nullptr;
    CUDA_CHECK(cudaMalloc(&dRowCount, static_cast<size_t>(n) * sizeof(uint32_t)));
    randOffsetKernel<<<(n + 127) / 128, 128>>>(g.normals, n, g.relOff, dRowCount);
    std::vector<uint32_t> hRowCount(n);
    CUDA_CHECK(cudaMemcpy(hRowCount.data(), dRowCount, static_cast<size_t>(n) * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dRowCount));

    g.rowBaseHost.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) g.rowBaseHost[i + 1] = g.rowBaseHost[i] + hRowCount[i];
    CUDA_CHECK(cudaMemcpy(g.rowBase, g.rowBaseHost.data(),
                          static_cast<size_t>(n + 1) * sizeof(unsigned long long),
                          cudaMemcpyHostToDevice));

    // The random stream is generated on the host (it is a strictly sequential
    // recurrence) in chunks of whole rows, and overlapped with the GPU work of
    // the previous chunk via double buffering.
    FastMT19937 rng(42);
    int buf = 0;
    int row = 0;
    while (row < n) {
        int rowEnd = row + 1;
        while (rowEnd < n &&
               g.rowBaseHost[rowEnd + 1] - g.rowBaseHost[row] <= g.chunkCap) {
            ++rowEnd;
        }
        const unsigned long long base = g.rowBaseHost[row];
        const size_t count = static_cast<size_t>(g.rowBaseHost[rowEnd] - base);

        // Wait until the previous transfer out of this buffer has completed
        CUDA_CHECK(cudaEventSynchronize(g.copyDone[buf]));
        uint32_t* h = g.hostRand[buf];
        rng.generate(h, count);

        CUDA_CHECK(cudaMemcpyAsync(g.randBuf[buf], h, count * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, g.chunkStream[buf]));
        CUDA_CHECK(cudaEventRecord(g.copyDone[buf], g.chunkStream[buf]));

        const dim3 block(KIJ_BLOCK, 1, 1);
        const dim3 grid((n + KIJ_BLOCK - 1) / KIJ_BLOCK, static_cast<unsigned>(rowEnd - row), 1);
        kijKernel<<<grid, block, 0, g.chunkStream[buf]>>>(
            g.tris, g.normals, g.centers, g.areas, g.nodes, g.leafTris, g.randBuf[buf], g.relOff,
            g.rowBase, base, row, n, g.kij, g.weights);
        CUDA_CHECK(cudaGetLastError());

        for (int i = row; i < rowEnd; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == n) {
                printf("  Progress: %d/%d triangles\n", i + 1, n);
            }
        }

        row = rowEnd;
        buf ^= 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(g.chunkStream[0]));
    CUDA_CHECK(cudaStreamSynchronize(g.chunkStream[1]));

}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    GpuData& g = state.gpu;
    const int n = static_cast<int>(state.numTriangles);

    int* dMinTau = nullptr;
    int hMinTau = std::numeric_limits<int>::max();
    CUDA_CHECK(cudaMalloc(&dMinTau, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dMinTau, &hMinTau, sizeof(int), cudaMemcpyHostToDevice));

    dim3 block(256, 1, 1);
    dim3 grid((n + block.x - 1) / block.x, n, 1);
    tauKernel<<<grid, block>>>(g.centers, n, g.tau, dMinTau);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(&hMinTau, dMinTau, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dMinTau));

    // The per-timestep kernel relies on all off-diagonal delays being >= 1, so
    // that a timestep only reads strictly older radiosity values.
    if (n > 1 && hMinTau < 1) {
        fprintf(stderr, "Unsupported mesh: zero propagation delay between distinct triangles\n");
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    GpuData& g = state.gpu;
    const int n = static_cast<int>(state.numTriangles);
    const int block = 128;
    const unsigned grid = static_cast<unsigned>((n + block - 1) / block);

    // Timesteps are sequentially dependent; the triangles within a timestep are
    // independent and are mapped one per thread.
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        simKernel<<<grid, block, 0, g.stream>>>(g.weights, g.rho, g.radE, g.radB,
                                                n, static_cast<int>(t));

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(g.stream));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    GpuData& g = state.gpu;
    const int n = static_cast<int>(state.numTriangles);
    const int numT = static_cast<int>(state.numTimesteps);

    if (numT == 0) {
        CUDA_CHECK(cudaMemset(g.distances, 0, static_cast<size_t>(n) * sizeof(float)));
        std::fill(state.distances.begin(), state.distances.end(), ZERO);
        return;
    }

    // One thread per (triangle, lag) pair for the correlation, then a scan over
    // the lags picks the first maximum exactly as the sequential version does.
    const long long total = static_cast<long long>(n) * numT;
    const int block = 256;
    corrKernel<<<static_cast<unsigned>((total + block - 1) / block), block, 0, g.stream>>>(
        g.radB, n, numT, static_cast<int>(state.sourceIndex), g.corr);
    argmaxKernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, g.stream>>>(
        g.corr, n, numT, g.distances);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(state.distances.data(), g.distances, static_cast<size_t>(n) * sizeof(float),
                               cudaMemcpyDeviceToHost, g.stream));
    CUDA_CHECK(cudaStreamSynchronize(g.stream));
}

// ============================================================================
// Copy the device-side matrices back for validation
// ============================================================================

void fetchDeviceState(SimulationState& state) {
    GpuData& g = state.gpu;
    const size_t n = state.numTriangles;
    state.kij.resize(n * n);
    state.tau.resize(n * n);
    state.radB.resize(state.numTimesteps * n);
    CUDA_CHECK(cudaMemcpy(state.kij.data(), g.kij, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.tau.data(), g.tau, n * n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.radB.data(), g.radB, state.numTimesteps * n * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

void releaseDeviceState(SimulationState& state) {
    GpuData& g = state.gpu;
    void* buffers[] = {g.tris, g.normals, g.centers, g.nodes, g.leafTris, g.kij, g.tau,
                       g.weights, g.areas, g.rho, g.radE, g.radB, g.corr, g.distances,
                       g.relOff, g.rowBase, g.randBuf[0], g.randBuf[1]};
    for (void* p : buffers) cudaFree(p);
    for (int b = 0; b < 2; ++b) {
        cudaFreeHost(g.hostRand[b]);
        cudaEventDestroy(g.copyDone[b]);
        cudaStreamDestroy(g.chunkStream[b]);
    }
    cudaStreamDestroy(g.stream);
    g = GpuData{};
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

    // Initialize the CUDA context up front so that its one-time cost is not
    // attributed to any of the timed phases
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    CUDA_CHECK(cudaFree(nullptr));

    printf("Room Response Simulation Benchmark\n");
    printf("===================================\n");
    printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("GPU: %s (%d SMs, CC %d.%d)\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);
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
        fetchDeviceState(state);
        if (!validateResults(state)) {
            releaseDeviceState(state);
            return 1;
        }
    }

    releaseDeviceState(state);
    return 0;
}
