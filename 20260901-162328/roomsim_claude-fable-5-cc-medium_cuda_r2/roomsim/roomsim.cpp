/**
 * Room Response Simulation Benchmark (CUDA-parallel version)
 *
 * GPU-parallel implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * Parallelization strategy:
 *  - Form factors: one GPU thread per triangle pair; the octree is flattened
 *    into linear arrays and traversed with an explicit stack on the device.
 *    The Monte Carlo samples are drawn on the host from the same std::mt19937
 *    stream (in the same order) as the sequential code and streamed to the
 *    GPU in double-buffered chunks so RNG generation overlaps GPU compute.
 *  - Tau: one GPU thread per triangle pair.
 *  - Wave propagation: the timestep loop is an inherent recurrence; within a
 *    timestep, one thread block per receiving triangle reduces over sources.
 *  - Distances: one thread block per triangle; threads share the correlation
 *    lags and an argmax reduction reproduces the sequential tie-breaking.
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,   \
                    __LINE__, cudaGetErrorString(err__));                      \
            exit(1);                                                           \
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

// Uniform random draws consumed per (non-culled) triangle pair:
// NUM_RAYS rays x 2 sample points x 2 barycentric coordinates
constexpr int DRAWS_PER_PAIR = NUM_RAYS * 4;

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    __host__ __device__ explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    __host__ __device__ val_t squaredNorm() const { return x * x + y * y + z * z; }
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
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

// Plain triangle layout used on the device (vertices + precomputed normal)
struct DevTri {
    Vec3 a, b, c, n;
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
// Octree for Spatial Acceleration (built on the host, flattened for the GPU)
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

// Flattened octree node for GPU traversal. A node is a leaf iff triCount > 0
// (matches the sequential applyToTris logic which treats nodes with a
// non-empty triangle list as leaves).
struct DevNode {
    Vec3 center, halfExtent;
    int child[8];
    int triStart, triCount;
};

static int flattenOctree(const Octree& node, std::vector<DevNode>& nodes,
                         std::vector<int>& triIdx) {
    int id = static_cast<int>(nodes.size());
    nodes.emplace_back();

    DevNode dn;
    dn.center = node.center;
    dn.halfExtent = node.halfExtent;
    for (int i = 0; i < 8; ++i) dn.child[i] = -1;
    dn.triStart = static_cast<int>(triIdx.size());
    dn.triCount = static_cast<int>(node.triangleIndices.size());

    for (size_t t : node.triangleIndices) triIdx.push_back(static_cast<int>(t));

    if (node.triangleIndices.empty()) {
        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) {
                dn.child[i] = flattenOctree(*node.children[i], nodes, triIdx);
            }
        }
    }

    nodes[id] = dn;
    return id;
}

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

// Drop-in replacement for std::mt19937 + std::uniform_real_distribution<float>
// (0, 1). MT19937 is fully specified by the standard and the float mapping
// replicates libstdc++'s generate_canonical (x / 2^32, clamped below 1.0), so
// the produced stream is bit-identical to the original sequential code while
// being considerably faster to sample.
class RandomGenerator {
    static constexpr int N = 624, M = 397;
    uint32_t mt[N];
    val_t out[N];
    int idx;

    static uint32_t mix(uint32_t a, uint32_t b) {
        uint32_t y = (a & 0x80000000u) | (b & 0x7fffffffu);
        return (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    }

    // Advance the state by one full period and temper/convert the whole
    // block to floats at once (both loops auto-vectorize).
    void refill() {
        for (int i = 0; i < N - M; ++i) mt[i] = mt[i + M] ^ mix(mt[i], mt[i + 1]);
        for (int i = N - M; i < N - 1; ++i) mt[i] = mt[i + M - N] ^ mix(mt[i], mt[i + 1]);
        mt[N - 1] = mt[M - 1] ^ mix(mt[N - 1], mt[0]);

        for (int i = 0; i < N; ++i) {
            uint32_t y = mt[i];
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680u;
            y ^= (y << 15) & 0xefc60000u;
            y ^= y >> 18;
            val_t f = static_cast<val_t>(y) * 0x1p-32f;
            out[i] = f >= 1.0f ? 0.99999994f : f;
        }
        idx = 0;
    }

public:
    explicit RandomGenerator(uint32_t seed = 42) {
        mt[0] = seed;
        for (int i = 1; i < N; ++i)
            mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
        idx = N;
    }

    val_t rand() {
        if (idx == N) refill();
        return out[idx++];
    }

    // Bulk draw: equivalent to `count` consecutive rand() calls
    void randBlock(val_t* dst, int count) {
        while (count > 0) {
            if (idx == N) refill();
            int take = std::min(count, N - idx);
            std::memcpy(dst, out + idx, take * sizeof(val_t));
            idx += take;
            dst += take;
            count -= take;
        }
    }
};

// ============================================================================
// Device Geometry Routines
// ============================================================================

// Generate a random point inside a triangle using barycentric coordinates
// (u, v are the two uniform draws consumed by the sequential version)
__device__ inline Vec3 pointInTriangle(const DevTri& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// Ray-Triangle Intersection (Möller-Trumbore algorithm)
__device__ inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                             const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
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

// Segment vs axis-aligned box overlap (same test as Octree::rayIntersectsBox)
__device__ inline bool rayIntersectsBox(const Vec3& p1, const Vec3& p2,
                                        const Vec3& center, const Vec3& he) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > he.x + ad.x) return false;
    if (fabsf(c.y) > he.y + ad.y) return false;
    if (fabsf(c.z) > he.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > he.y * ad.z + he.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > he.z * ad.x + he.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > he.x * ad.y + he.y * ad.x + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle.
// Octree-accelerated: explicit-stack traversal of the flattened tree.
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const DevNode* nodes, const int* triIdx,
                             const DevTri* tris, int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[96];
    int sp = 0;
    stack[sp++] = 0;  // root is processed unconditionally, as in the original

    while (sp > 0) {
        const DevNode& n = nodes[stack[--sp]];

        if (n.triCount > 0) {
            // Leaf: test triangles directly
            for (int k = 0; k < n.triCount; ++k) {
                int idx = triIdx[n.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const DevTri tri = tris[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;  // Ray is blocked
                }
            }
        } else {
            // Internal: descend to children whose box the segment overlaps
            for (int i = 0; i < 8; ++i) {
                int ci = n.child[i];
                if (ci >= 0) {
                    const DevNode& cn = nodes[ci];
                    if (rayIntersectsBox(from, to, cn.center, cn.halfExtent)) {
                        stack[sp++] = ci;
                    }
                }
            }
        }
    }
    return false;
}

// Compute the cosine of angle between vector and triangle normal
__device__ inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// ============================================================================
// GPU Kernels
// ============================================================================

// Form factor Kij between triangle i (receiver) and triangle j (emitter).
// One thread per (i, j) pair within the current row chunk. The uniform draws
// (DRAWS_PER_PAIR per active pair) come from the host-side mt19937 stream so
// the Monte Carlo estimate is identical to the sequential implementation.
__global__ void kijKernel(size_t pairBase, size_t pairCount, int numTriangles,
                          const val_t* __restrict__ uniforms,
                          const uint8_t* __restrict__ active,
                          const DevTri* __restrict__ tris,
                          const DevNode* __restrict__ nodes,
                          const int* __restrict__ triIdx,
                          val_t* __restrict__ kijOut) {
    size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= pairCount) return;

    size_t pairGlobal = pairBase + p;
    int i = static_cast<int>(pairGlobal / numTriangles);
    int j = static_cast<int>(pairGlobal % numTriangles);

    // Diagonal or normal-culled pair: form factor is zero, no draws consumed
    if (!active[p]) {
        kijOut[pairGlobal] = ZERO;
        return;
    }

    const DevTri triI = tris[i];
    const DevTri triJ = tris[j];
    const val_t* u = uniforms + p * DRAWS_PER_PAIR;

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = pointInTriangle(triI, u[r * 4 + 0], u[r * 4 + 1]);
        Vec3 pJ = pointInTriangle(triJ, u[r * 4 + 2], u[r * 4 + 3]);

        if (isRayBlocked(pI, pJ, nodes, triIdx, tris, i, j)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.n);
        val_t cosPhiJ = cosPhi(-v, triJ.n);

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kijOut[pairGlobal] = kij * INV_NUM_RAYS;
}

// Tau (time delay) between all triangle pairs; one thread per pair
__global__ void tauKernel(int numTriangles, const DevTri* __restrict__ tris,
                          int* __restrict__ tauOut) {
    size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(numTriangles) * numTriangles;
    if (p >= total) return;

    int i = static_cast<int>(p / numTriangles);
    int j = static_cast<int>(p % numTriangles);
    if (i == j) {
        tauOut[p] = 0;
        return;
    }

    const DevTri& ti = tris[i];
    const DevTri& tj = tris[j];
    Vec3 ci = (ti.a + ti.b + ti.c) / 3.0f;
    Vec3 cj = (tj.a + tj.b + tj.c) / 3.0f;
    val_t dist = (ci - cj).norm();
    tauOut[p] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// One wave-propagation timestep: block per receiving triangle i, threads
// cooperatively reduce the contributions of all source triangles j.
constexpr int SIM_BLOCK = 256;

__global__ void simStepKernel(int t, int numTriangles,
                              const val_t* __restrict__ kij,
                              const int* __restrict__ tau,
                              const val_t* __restrict__ areas,
                              const val_t* __restrict__ rho,
                              const val_t* __restrict__ radE,
                              val_t* __restrict__ radB) {
    int i = blockIdx.x;
    size_t rowOff = static_cast<size_t>(i) * numTriangles;

    val_t sumB = ZERO;
    for (int j = threadIdx.x; j < numTriangles; j += SIM_BLOCK) {
        if (i == j) continue;

        int tauij = tau[rowOff + j];

        // Skip if wave hasn't yet propagated from j to i
        if (t < tauij) continue;

        val_t k = kij[rowOff + j];
        if (k <= ZERO) continue;

        // Get radiosity from source triangle at time when emission occurred
        size_t srcTime = static_cast<size_t>(t - tauij);
        val_t radJ = radB[srcTime * numTriangles + j];
        if (radJ <= ZERO) continue;

        // Accumulate contribution: form factor * area * source radiosity
        sumB += fminf(k * areas[j], ONE) * radJ;
    }

    __shared__ val_t sh[SIM_BLOCK];
    sh[threadIdx.x] = sumB;
    __syncthreads();
    for (int s = SIM_BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        size_t idx = static_cast<size_t>(t) * numTriangles + i;
        // Update radiosity: reflection + emission
        radB[idx] = rho[i] * sh[0] + radE[idx];
    }
}

// Distance estimation via discrete cross-correlation. Block per triangle i;
// threads split the correlation lags, then an argmax reduction picks the peak
// (ties resolved toward the smallest lag, matching the sequential scan).
constexpr int DIST_BLOCK = 128;

__global__ void distKernel(int numTriangles, int numTimesteps, int sourceIndex,
                           const val_t* __restrict__ radB,
                           val_t* __restrict__ distances) {
    int i = blockIdx.x;

    val_t maxCorr = ZERO;
    int bestT = 0;

    for (int t = threadIdx.x; t < numTimesteps; t += DIST_BLOCK) {
        val_t sum = ZERO;
        for (int tt = t; tt < numTimesteps; ++tt) {
            val_t pB = radB[static_cast<size_t>(tt) * numTriangles + i];
            val_t pS = radB[static_cast<size_t>(tt - t) * numTriangles + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    __shared__ val_t shV[DIST_BLOCK];
    __shared__ int shT[DIST_BLOCK];
    shV[threadIdx.x] = maxCorr;
    shT[threadIdx.x] = bestT;
    __syncthreads();
    for (int s = DIST_BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            val_t vo = shV[threadIdx.x + s];
            int to = shT[threadIdx.x + s];
            if (vo > shV[threadIdx.x] ||
                (vo == shV[threadIdx.x] && to < shT[threadIdx.x])) {
                shV[threadIdx.x] = vo;
                shT[threadIdx.x] = to;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        distances[i] = WAVE_SPEED * static_cast<val_t>(shT[0]);
    }
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct GpuState {
    DevTri* tris = nullptr;
    DevNode* nodes = nullptr;
    int* triIdx = nullptr;
    val_t* areas = nullptr;
    val_t* rho = nullptr;
    val_t* kij = nullptr;
    int* tau = nullptr;
    val_t* radE = nullptr;
    val_t* radB = nullptr;
    val_t* distances = nullptr;

    void release() {
        cudaFree(tris); cudaFree(nodes); cudaFree(triIdx);
        cudaFree(areas); cudaFree(rho); cudaFree(kij); cudaFree(tau);
        cudaFree(radE); cudaFree(radB); cudaFree(distances);
        *this = GpuState{};
    }
};

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure
    GpuState gpu;                   // Device-side buffers

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

    // Initialize matrices
    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Upload geometry, octree and per-triangle data to the GPU
    printf("Uploading data to GPU...\n");
    const size_t n = state.numTriangles;

    std::vector<DevTri> devTris(n);
    for (size_t i = 0; i < n; ++i) {
        devTris[i] = {state.triangles[i].a, state.triangles[i].b,
                      state.triangles[i].c, state.triangles[i].normal()};
    }

    std::vector<DevNode> nodes;
    std::vector<int> triIdx;
    flattenOctree(state.octree, nodes, triIdx);

    GpuState& g = state.gpu;
    CUDA_CHECK(cudaMalloc(&g.tris, n * sizeof(DevTri)));
    CUDA_CHECK(cudaMalloc(&g.nodes, nodes.size() * sizeof(DevNode)));
    CUDA_CHECK(cudaMalloc(&g.triIdx, std::max<size_t>(1, triIdx.size()) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.areas, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&g.rho, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&g.kij, n * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&g.tau, n * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.radE, timesteps * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&g.radB, timesteps * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&g.distances, n * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(g.tris, devTris.data(), n * sizeof(DevTri), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.nodes, nodes.data(), nodes.size() * sizeof(DevNode), cudaMemcpyHostToDevice));
    if (!triIdx.empty()) {
        CUDA_CHECK(cudaMemcpy(g.triIdx, triIdx.data(), triIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(g.areas, state.areas.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.rho, state.rho.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.radE, state.radE.data(), timesteps * n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(g.radB, 0, timesteps * n * sizeof(val_t)));
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    RandomGenerator rng(42);

    const size_t n = state.numTriangles;
    const size_t totalPairs = n * n;

    // Chunk the pair space by rows: the host draws the Monte Carlo samples
    // from the sequential mt19937 stream for one chunk while the GPU
    // processes the previous one (double-buffered, two streams).
    size_t rowsPerChunk = std::max<size_t>(1, (1u << 20) / n);
    rowsPerChunk = std::min(rowsPerChunk, n);
    const size_t maxPairs = rowsPerChunk * n;

    val_t* hUniforms[2];
    uint8_t* hActive[2];
    val_t* dUniforms[2];
    uint8_t* dActive[2];
    cudaStream_t streams[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMallocHost(&hUniforms[b], maxPairs * DRAWS_PER_PAIR * sizeof(val_t)));
        CUDA_CHECK(cudaMallocHost(&hActive[b], maxPairs * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&dUniforms[b], maxPairs * DRAWS_PER_PAIR * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&dActive[b], maxPairs * sizeof(uint8_t)));
        CUDA_CHECK(cudaStreamCreate(&streams[b]));
    }

    // Draw the samples for one chunk of rows, in the exact order the
    // sequential code consumes them (row-major, skipping the diagonal and
    // pairs culled by the facing-direction test, which draw nothing).
    auto fillChunk = [&](size_t rowStart, size_t rowCount, val_t* uni, uint8_t* act) {
        size_t p = 0;
        for (size_t i = rowStart; i < rowStart + rowCount; ++i) {
            const Vec3 ni = state.triangles[i].normal();
            for (size_t j = 0; j < n; ++j, ++p) {
                if (i == j || ni.dot(state.triangles[j].normal()) > 0.99f) {
                    act[p] = 0;
                    continue;
                }
                act[p] = 1;
                rng.randBlock(uni + p * DRAWS_PER_PAIR, DRAWS_PER_PAIR);
            }
        }
    };

    const size_t numChunks = (n + rowsPerChunk - 1) / rowsPerChunk;
    size_t chunkRows[2] = {0, 0};

    // Prime the pipeline with the first chunk
    fillChunk(0, std::min(rowsPerChunk, n), hUniforms[0], hActive[0]);
    chunkRows[0] = std::min(rowsPerChunk, n);

    for (size_t c = 0; c < numChunks; ++c) {
        const int buf = static_cast<int>(c & 1);
        const size_t rowStart = c * rowsPerChunk;
        const size_t rows = chunkRows[buf];
        const size_t pairCount = rows * n;

        CUDA_CHECK(cudaMemcpyAsync(dUniforms[buf], hUniforms[buf],
                                   pairCount * DRAWS_PER_PAIR * sizeof(val_t),
                                   cudaMemcpyHostToDevice, streams[buf]));
        CUDA_CHECK(cudaMemcpyAsync(dActive[buf], hActive[buf],
                                   pairCount * sizeof(uint8_t),
                                   cudaMemcpyHostToDevice, streams[buf]));

        const int threads = 128;
        const size_t blocks = (pairCount + threads - 1) / threads;
        kijKernel<<<static_cast<unsigned>(blocks), threads, 0, streams[buf]>>>(
            rowStart * n, pairCount, static_cast<int>(n),
            dUniforms[buf], dActive[buf],
            state.gpu.tris, state.gpu.nodes, state.gpu.triIdx, state.gpu.kij);
        CUDA_CHECK(cudaGetLastError());

        // Prepare the next chunk on the other buffer while the GPU works
        if (c + 1 < numChunks) {
            const int nb = static_cast<int>((c + 1) & 1);
            // Make sure the previous transfer from that buffer has finished
            CUDA_CHECK(cudaStreamSynchronize(streams[nb]));
            const size_t nextStart = (c + 1) * rowsPerChunk;
            const size_t nextRows = std::min(rowsPerChunk, n - nextStart);
            fillChunk(nextStart, nextRows, hUniforms[nb], hActive[nb]);
            chunkRows[nb] = nextRows;
        }

        if ((rowStart + rows) % 100 < rows || rowStart + rows == n) {
            printf("  Progress: %zu/%zu triangles\n", rowStart + rows, n);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the form factor matrix back for validation/inspection
    CUDA_CHECK(cudaMemcpy(state.kij.data(), state.gpu.kij,
                          totalPairs * sizeof(val_t), cudaMemcpyDeviceToHost));

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFreeHost(hUniforms[b]));
        CUDA_CHECK(cudaFreeHost(hActive[b]));
        CUDA_CHECK(cudaFree(dUniforms[b]));
        CUDA_CHECK(cudaFree(dActive[b]));
        CUDA_CHECK(cudaStreamDestroy(streams[b]));
    }
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    const size_t n = state.numTriangles;
    const size_t total = n * n;
    const int threads = 256;
    const size_t blocks = (total + threads - 1) / threads;
    tauKernel<<<static_cast<unsigned>(blocks), threads>>>(
        static_cast<int>(n), state.gpu.tris, state.gpu.tau);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    const size_t n = state.numTriangles;

    // Timesteps form a recurrence (radB[t] depends on radB[t - tau]), so the
    // time loop stays on the host; each step is fully parallel over triangles.
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        simStepKernel<<<static_cast<unsigned>(n), SIM_BLOCK>>>(
            static_cast<int>(t), static_cast<int>(n),
            state.gpu.kij, state.gpu.tau, state.gpu.areas, state.gpu.rho,
            state.gpu.radE, state.gpu.radB);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the radiosity history back for validation/inspection
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.gpu.radB,
                          state.numTimesteps * n * sizeof(val_t),
                          cudaMemcpyDeviceToHost));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    const size_t n = state.numTriangles;
    distKernel<<<static_cast<unsigned>(n), DIST_BLOCK>>>(
        static_cast<int>(n), static_cast<int>(state.numTimesteps),
        static_cast<int>(state.sourceIndex), state.gpu.radB, state.gpu.distances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.gpu.distances,
                          n * sizeof(val_t), cudaMemcpyDeviceToHost));
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

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));  // Initialize CUDA context up front

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
            state.gpu.release();
            return 1;
        }
    }

    state.gpu.release();
    return 0;
}
