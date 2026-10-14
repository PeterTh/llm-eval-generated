/**
 * Room Response Simulation Benchmark
 *
 * This is a CUDA-parallel implementation of room impulse response simulation
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
 * - Form factors: one GPU thread per visible (receiver, emitter) pair. The random
 *   stream is the same single mt19937 sequence as the sequential code, so the host
 *   generates the draws in the original (i, j) order and the GPU consumes them.
 *   Generation of the next chunk overlaps with GPU evaluation of the previous one.
 * - Time delays: one thread per (i, j) pair.
 * - Wave propagation: timesteps are sequential (Tau >= 1 for i != j), but all
 *   receivers of a timestep are computed in parallel, one thread per receiver.
 * - Distances: one thread per triangle.
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
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define HD __host__ __device__

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                 \
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
constexpr int RANDS_PER_PAIR = 4 * NUM_RAYS;  // Per ray: u,v for emitter point and u,v for receiver point
constexpr val_t EPSILON = 1e-6f;
constexpr val_t NO_HIT = FLT_MAX;

constexpr int BLOCK_SIZE = 128;
constexpr int TRAVERSAL_STACK = 128;      // Octree traversal stack capacity (device)

// Scalar helpers with std::min / std::max semantics, usable on host and device
HD inline val_t vmin(val_t a, val_t b) { return b < a ? b : a; }
HD inline val_t vmax(val_t a, val_t b) { return a < b ? b : a; }

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

    HD val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    HD Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    HD val_t squaredNorm() const { return x * x + y * y + z * z; }
    HD val_t norm() const { return sqrtf(squaredNorm()); }
    HD Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    HD bool operator==(const Vec3& o) const {
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

// Device-side triangle: vertices and normal (the only triangle data the kernels need)
struct DevTriangle {
    Vec3 a, b, c;
    Vec3 n;
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
// Octree for Spatial Acceleration (built on host, flattened for the device)
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

// Flat, device-friendly octree node. Leaves have triCount > 0; inner nodes have triCount == 0.
struct DevNode {
    Vec3 center;
    Vec3 halfExtent;
    int32_t child[8];   // -1 for empty children
    uint32_t triBegin;  // Offset into the leaf triangle index array
    uint32_t triCount;
};

struct FlatOctree {
    std::vector<DevNode> nodes;      // nodes[0] is the root
    std::vector<uint32_t> leafTris;  // Triangle indices of all leaves, concatenated
    int maxDepth = 0;
};

int flattenOctreeNode(const Octree& node, FlatOctree& flat, int depth) {
    int id = static_cast<int>(flat.nodes.size());
    flat.nodes.emplace_back();
    flat.maxDepth = std::max(flat.maxDepth, depth);

    DevNode dn{};
    dn.center = node.center;
    dn.halfExtent = node.halfExtent;
    for (int c = 0; c < 8; ++c) {
        dn.child[c] = -1;
        if (node.children[c]) {
            dn.child[c] = flattenOctreeNode(*node.children[c], flat, depth + 1);
        }
    }
    dn.triBegin = static_cast<uint32_t>(flat.leafTris.size());
    for (size_t idx : node.triangleIndices) {
        flat.leafTris.push_back(static_cast<uint32_t>(idx));
    }
    dn.triCount = static_cast<uint32_t>(node.triangleIndices.size());

    flat.nodes[id] = dn;
    return id;
}

FlatOctree flattenOctree(const Octree& root) {
    FlatOctree flat;
    flattenOctreeNode(root, flat, 0);
    return flat;
}

// Test whether the segment p1-p2 intersects a node's bounding box
HD inline bool boxRayHit(const Vec3& center, const Vec3& halfExtent, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

    return true;
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

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
};

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

HD inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                     const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return NO_HIT;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return NO_HIT;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return NO_HIT;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated, device)
// ============================================================================

struct DeviceGeometry {
    const DevTriangle* tris;
    const DevNode* nodes;
    const uint32_t* leafTris;
};

// Check if a ray between two triangles is blocked by any other triangle.
// The result is a logical OR over the candidate triangles, so the traversal order
// does not affect the answer; it stops at the first blocking triangle.
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to, const DeviceGeometry& geo,
                             size_t srcTriIdx, size_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[TRAVERSAL_STACK];
    int sp = 0;
    stack[sp++] = 0;  // Root (its box is not tested, matching the sequential traversal)

    while (sp > 0) {
        const DevNode& node = geo.nodes[stack[--sp]];
        if (node.triCount > 0) {
            for (uint32_t k = 0; k < node.triCount; ++k) {
                size_t idx = geo.leafTris[node.triBegin + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const DevTriangle& tri = geo.tris[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;  // Ray is blocked
                }
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int ch = node.child[c];
                if (ch >= 0 && boxRayHit(geo.nodes[ch].center, geo.nodes[ch].halfExtent, from, to)) {
                    stack[sp++] = ch;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij, device)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
HD inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return vmax(ZERO, v.dot(normal) / vNorm);
}

// Random point in a triangle from two uniform draws (same arithmetic as the sequential code)
__device__ inline Vec3 samplePoint(const DevTriangle& t, const val_t* uv) {
    val_t u = uv[0];
    val_t v = uv[1];
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// One thread per (receiver i, emitter j) pair. The pair's draws are the
// RANDS_PER_PAIR values starting at draws[k * RANDS_PER_PAIR], where k is the
// pair's position in the chunk, which matches the sequential RNG consumption order.
// kijT is stored transposed: kijT[j * N + i] = Kij.
__global__ void formFactorKernel(const uint2* __restrict__ pairs, const val_t* __restrict__ draws,
                                 size_t numPairs, DeviceGeometry geo, val_t* kijT, size_t N) {
    size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (k >= numPairs) return;

    const uint2 pr = pairs[k];
    const size_t i = pr.x;
    const size_t j = pr.y;
    const DevTriangle& triI = geo.tris[i];
    const DevTriangle& triJ = geo.tris[j];
    const val_t* rnd = draws + k * RANDS_PER_PAIR;

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = samplePoint(triI, rnd);
        Vec3 pJ = samplePoint(triJ, rnd + 2);
        rnd += 4;

        if (isRayBlocked(pI, pJ, geo, i, j)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.n);
        val_t cosPhiJ = cosPhi(-v, triJ.n);

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kijT[j * N + i] = kij * INV_NUM_RAYS;
}

// ============================================================================
// Tau (time delay) Computation (device)
// ============================================================================

// One thread per matrix entry. tauT is stored transposed: tauT[j * N + i] = Tau_ij
__global__ void timeDelayKernel(const Vec3* __restrict__ centers, int* tauT, size_t N) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (idx >= N * N) return;

    size_t j = idx / N;
    size_t i = idx - j * N;
    if (i == j) {
        tauT[idx] = 0;
        return;
    }

    val_t dist = (centers[i] - centers[j]).norm();
    tauT[idx] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;
    size_t sourceIndex;

    std::vector<Triangle> triangles;  // Host geometry (normals are used for culling)
    std::vector<val_t> areas;         // Area of each triangle
    std::vector<val_t> rho;           // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radE;          // Emission radiosity (T x N matrix), host copy used for init
    std::vector<val_t> radB;          // Reflected radiosity (T x N), copied back for validation
    std::vector<val_t> distances;     // Computed distances from source

    Octree octree;                    // Host-side spatial structure (flattened to dNodes)

    // Device-side data
    DevTriangle* dTriangles = nullptr;
    DevNode* dNodes = nullptr;
    uint32_t* dLeafTris = nullptr;
    Vec3* dCenters = nullptr;
    val_t* dAreas = nullptr;
    val_t* dRho = nullptr;
    val_t* dRadE = nullptr;
    val_t* dRadB = nullptr;           // T x N
    val_t* dKijT = nullptr;           // N x N, transposed: [j * N + i]
    int* dTauT = nullptr;             // N x N, transposed: [j * N + i]
    val_t* dDistances = nullptr;

    DeviceGeometry geometry() const { return {dTriangles, dNodes, dLeafTris}; }
};

template <typename T>
T* deviceUpload(const std::vector<T>& host) {
    T* dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, host.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return dev;
}

template <typename T>
T* deviceZeros(size_t count) {
    T* dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, count * sizeof(T)));
    CUDA_CHECK(cudaMemset(dev, 0, count * sizeof(T)));
    return dev;
}

inline unsigned int blocksFor(size_t n) {
    return static_cast<unsigned int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
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

    const size_t N = state.numTriangles;
    const size_t T = timesteps;

    printf("Generated icosphere mesh with %zu triangles\n", N);

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);
    FlatOctree flat = flattenOctree(state.octree);
    if (flat.maxDepth * 8 + 8 > TRAVERSAL_STACK) {
        fprintf(stderr, "Octree too deep for device traversal stack (depth %d)\n", flat.maxDepth);
        exit(1);
    }

    // Initialize areas
    state.areas.resize(N);
    for (size_t i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.assign(N, reflectivity);

    // Set source emission (active for first half of timesteps)
    state.radE.assign(T * N, ZERO);
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[t * N + state.sourceIndex] = 1.0f;
    }

    // Upload geometry and data to the device
    std::vector<DevTriangle> devTris(N);
    std::vector<Vec3> centers(N);
    for (size_t i = 0; i < N; ++i) {
        const Triangle& tri = state.triangles[i];
        devTris[i] = {tri.a, tri.b, tri.c, tri.normal()};
        centers[i] = tri.center();
    }

    state.dTriangles = deviceUpload(devTris);
    state.dNodes = deviceUpload(flat.nodes);
    state.dLeafTris = deviceUpload(flat.leafTris);
    state.dCenters = deviceUpload(centers);
    state.dAreas = deviceUpload(state.areas);
    state.dRho = deviceUpload(state.rho);
    state.dRadE = deviceUpload(state.radE);
    state.dRadB = deviceZeros<val_t>(T * N);
    state.dKijT = deviceZeros<val_t>(N * N);
    state.dTauT = deviceZeros<int>(N * N);
    state.dDistances = deviceZeros<val_t>(N);
}

void releaseDevice(SimulationState& state) {
    CUDA_CHECK(cudaFree(state.dTriangles));
    CUDA_CHECK(cudaFree(state.dNodes));
    CUDA_CHECK(cudaFree(state.dLeafTris));
    CUDA_CHECK(cudaFree(state.dCenters));
    CUDA_CHECK(cudaFree(state.dAreas));
    CUDA_CHECK(cudaFree(state.dRho));
    CUDA_CHECK(cudaFree(state.dRadE));
    CUDA_CHECK(cudaFree(state.dRadB));
    CUDA_CHECK(cudaFree(state.dKijT));
    CUDA_CHECK(cudaFree(state.dTauT));
    CUDA_CHECK(cudaFree(state.dDistances));
}

// ============================================================================
// Precomputation Phase
// ============================================================================

static void reportProgress(size_t from, size_t to, size_t total) {
    for (size_t done = from + 1; done <= to; ++done) {
        if (done % 100 == 0 || done == total) {
            printf("  Progress: %zu/%zu triangles\n", done, total);
        }
    }
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");

    const size_t N = state.numTriangles;
    const DeviceGeometry geo = state.geometry();

    // The sequential reference draws from one RNG stream, in (i, j) order, and only
    // for pairs that are not culled. The host reproduces exactly that stream in chunks
    // of whole receiver rows; the device evaluates each chunk. Two buffers let the host
    // generate chunk c+1 while the device works on chunk c.
    RandomGenerator rng(42);

    constexpr int NUM_BUFFERS = 2;
    const size_t capacity = std::max(static_cast<size_t>(1) << 18, N);  // Pairs per chunk

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    uint2* hPairs[NUM_BUFFERS];
    val_t* hDraws[NUM_BUFFERS];
    uint2* dPairs[NUM_BUFFERS];
    val_t* dDraws[NUM_BUFFERS];
    cudaEvent_t done[NUM_BUFFERS];
    size_t chunkEnd[NUM_BUFFERS] = {0, 0};
    bool busy[NUM_BUFFERS] = {false, false};

    for (int b = 0; b < NUM_BUFFERS; ++b) {
        CUDA_CHECK(cudaMallocHost(&hPairs[b], capacity * sizeof(uint2)));
        CUDA_CHECK(cudaMallocHost(&hDraws[b], capacity * RANDS_PER_PAIR * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&dPairs[b], capacity * sizeof(uint2)));
        CUDA_CHECK(cudaMalloc(&dDraws[b], capacity * RANDS_PER_PAIR * sizeof(val_t)));
        CUDA_CHECK(cudaEventCreateWithFlags(&done[b], cudaEventDisableTiming));
    }

    size_t reported = 0;
    size_t rowBegin = 0;
    int b = 0;
    while (rowBegin < N) {
        // Wait until this buffer's previous chunk has been uploaded and evaluated
        if (busy[b]) {
            CUDA_CHECK(cudaEventSynchronize(done[b]));
            reportProgress(reported, chunkEnd[b], N);
            reported = chunkEnd[b];
        }

        // Generate draws for whole rows, in the sequential order
        size_t rowEnd = rowBegin;
        size_t numPairs = 0;
        while (rowEnd < N && numPairs + N - 1 <= capacity) {
            const Vec3 nI = state.triangles[rowEnd].normal();
            for (size_t j = 0; j < N; ++j) {
                if (j == rowEnd) continue;
                // Cull triangles facing the same direction (no draws consumed)
                if (nI.dot(state.triangles[j].normal()) > 0.99f) continue;

                hPairs[b][numPairs] = make_uint2(static_cast<unsigned int>(rowEnd),
                                                 static_cast<unsigned int>(j));
                val_t* out = hDraws[b] + numPairs * RANDS_PER_PAIR;
                for (int r = 0; r < RANDS_PER_PAIR; ++r) {
                    out[r] = rng.rand();
                }
                ++numPairs;
            }
            ++rowEnd;
        }

        CUDA_CHECK(cudaMemcpyAsync(dPairs[b], hPairs[b], numPairs * sizeof(uint2),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dDraws[b], hDraws[b], numPairs * RANDS_PER_PAIR * sizeof(val_t),
                                   cudaMemcpyHostToDevice, stream));
        if (numPairs > 0) {
            formFactorKernel<<<blocksFor(numPairs), BLOCK_SIZE, 0, stream>>>(
                dPairs[b], dDraws[b], numPairs, geo, state.dKijT, N);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(done[b], stream));

        busy[b] = true;
        chunkEnd[b] = rowEnd;
        rowBegin = rowEnd;
        b ^= 1;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    reportProgress(reported, N, N);

    for (int k = 0; k < NUM_BUFFERS; ++k) {
        CUDA_CHECK(cudaFreeHost(hPairs[k]));
        CUDA_CHECK(cudaFreeHost(hDraws[k]));
        CUDA_CHECK(cudaFree(dPairs[k]));
        CUDA_CHECK(cudaFree(dDraws[k]));
        CUDA_CHECK(cudaEventDestroy(done[k]));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    const size_t N = state.numTriangles;
    timeDelayKernel<<<blocksFor(N * N), BLOCK_SIZE>>>(state.dCenters, state.dTauT, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// Computes radB[t][i] for all receivers i. Tau_ij >= 1 for i != j, so every read
// is from an earlier timestep and receivers within a timestep are independent.
// The sum over j keeps the sequential order, so results match the reference exactly.
__global__ void waveStepKernel(size_t t, size_t N, const val_t* __restrict__ kijT,
                               const int* __restrict__ tauT, const val_t* __restrict__ areas,
                               const val_t* __restrict__ rho, const val_t* __restrict__ radE,
                               val_t* radB) {
    size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= N) return;

    val_t sumB = ZERO;

    for (size_t j = 0; j < N; ++j) {
        if (i == j) continue;

        int tauij = tauT[j * N + i];

        // Skip if wave hasn't yet propagated from j to i
        if (static_cast<int>(t) < tauij) continue;

        val_t kij = kijT[j * N + i];
        if (kij <= ZERO) continue;

        // Get radiosity from source triangle at time when emission occurred
        size_t srcTime = t - static_cast<size_t>(tauij);
        val_t radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        // Accumulate contribution: form factor * area * source radiosity
        sumB += vmin(kij * areas[j], ONE) * radJ;
    }

    // Update radiosity: reflection + emission
    radB[t * N + i] = rho[i] * sumB + radE[t * N + i];
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    for (size_t t = 0; t < T; ++t) {
        waveStepKernel<<<blocksFor(N), BLOCK_SIZE>>>(t, N, state.dKijT, state.dTauT, state.dAreas,
                                                      state.dRho, state.dRadE, state.dRadB);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

// One thread per triangle. Each thread runs the same sequential correlation loop as
// the reference, so the argmax (first maximum wins) is identical.
__global__ void distanceKernel(size_t N, size_t T, size_t sourceIndex, const val_t* __restrict__ radB,
                               val_t* distances) {
    size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= N) return;

    val_t maxCorr = ZERO;
    int bestT = 0;

    // Discrete cross-correlation to find time delay
    for (size_t t = 0; t < T; ++t) {
        val_t sum = ZERO;

        for (size_t tt = t; tt < T; ++tt) {
            val_t pB = radB[tt * N + i];
            val_t pS = radB[(tt - t) * N + sourceIndex];
            sum += pS * pB;
        }

        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }

    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    const size_t N = state.numTriangles;
    distanceKernel<<<blocksFor(N), BLOCK_SIZE>>>(N, state.numTimesteps, state.sourceIndex,
                                                  state.dRadB, state.dDistances);
    CUDA_CHECK(cudaGetLastError());

    state.distances.resize(N);
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.dDistances, N * sizeof(val_t),
                          cudaMemcpyDeviceToHost));
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, const std::vector<val_t>& kij) {
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
    int nonZeroKij = 0;
    for (size_t i = 0; i < state.numTriangles * state.numTriangles; ++i) {
        if (kij[i] > EPSILON) nonZeroKij++;
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
        // Bring back the matrices used by the validation checks
        state.radB.resize(t * n);
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dRadB, t * n * sizeof(val_t), cudaMemcpyDeviceToHost));
        std::vector<val_t> kij(n * n);
        CUDA_CHECK(cudaMemcpy(kij.data(), state.dKijT, n * n * sizeof(val_t), cudaMemcpyDeviceToHost));

        if (!validateResults(state, kij)) {
            releaseDevice(state);
            return 1;
        }
    }

    releaseDevice(state);
    return 0;
}
