/**
 * Room Response Simulation Benchmark
 *
 * This is a CUDA implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 * Mesh generation, octree construction and random number generation stay on the
 * host; the numerical phases run on the GPU and produce the same results as the
 * sequential formulation (the random sequence is consumed in the same order).
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

// ============================================================================
// Types and Constants
// ============================================================================

#define HOST_DEVICE __host__ __device__

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

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

constexpr size_t DRAWS_PER_PAIR = 4 * NUM_RAYS;         // (u,v) for each of the two points per ray
constexpr size_t MAX_CHUNK_DRAWS = size_t(1) << 24;     // Host staging budget per form-factor chunk (floats)
constexpr unsigned BLOCK_SIZE = 256;
constexpr int STACK_CAPACITY = 64;                      // Octree traversal stack (depth-limited, see initializeSimulation)

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    HOST_DEVICE constexpr Vec3() : x(0), y(0), z(0) {}
    HOST_DEVICE constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    HOST_DEVICE explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    HOST_DEVICE Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    HOST_DEVICE Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    HOST_DEVICE Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    HOST_DEVICE Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    HOST_DEVICE Vec3 operator-() const { return {-x, -y, -z}; }

    HOST_DEVICE val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    HOST_DEVICE Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    HOST_DEVICE val_t squaredNorm() const { return x * x + y * y + z * z; }
    HOST_DEVICE val_t norm() const { return sqrtf(squaredNorm()); }
    HOST_DEVICE Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    HOST_DEVICE bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    HOST_DEVICE Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    HOST_DEVICE Vec3 center() const { return (a + b + c) / 3.0f; }
    HOST_DEVICE Vec3 normal() const { return _normal; }

    HOST_DEVICE val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    HOST_DEVICE bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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
};

// Octree node in a flat, pointer-free layout that device code can traverse.
// A node is a leaf iff triCount > 0 (same rule as the host octree).
struct DevNode {
    Vec3 center;
    Vec3 halfExtent;
    int32_t child[8];     // Index into the node array, -1 when absent
    uint32_t triBegin;    // Range into the flat triangle index array (leaves)
    uint32_t triCount;
};

// Flattens the host octree depth-first; the root ends up at index 0.
int32_t flattenOctree(const Octree& node, std::vector<DevNode>& nodes, std::vector<uint32_t>& triIdx,
                      int depth, int& maxDepth) {
    const int32_t id = static_cast<int32_t>(nodes.size());
    nodes.emplace_back();
    maxDepth = std::max(maxDepth, depth);

    DevNode dev;
    dev.center = node.center;
    dev.halfExtent = node.halfExtent;
    dev.triBegin = static_cast<uint32_t>(triIdx.size());
    dev.triCount = static_cast<uint32_t>(node.triangleIndices.size());
    for (size_t idx : node.triangleIndices) triIdx.push_back(static_cast<uint32_t>(idx));
    for (int c = 0; c < 8; ++c) {
        dev.child[c] = node.children[c] ? flattenOctree(*node.children[c], nodes, triIdx, depth + 1, maxDepth) : -1;
    }
    nodes[id] = dev;
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

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
};

// ============================================================================
// Device Geometry: Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

__device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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

// ============================================================================
// Device Geometry: Random Points, Visibility Testing (Octree-accelerated)
// ============================================================================

// Point in a triangle from two uniform samples (same mapping as the original sequential code)
__device__ Vec3 pointInTriangle(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// Read-only view of the scene used by the kernels
struct DeviceScene {
    const Triangle* tris;
    const DevNode* nodes;
    const uint32_t* triIdx;
};

__device__ bool rayIntersectsBox(const DevNode& box, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - box.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3& h = box.halfExtent;

    if (fabsf(c.x) > h.x + ad.x) return false;
    if (fabsf(c.y) > h.y + ad.y) return false;
    if (fabsf(c.z) > h.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle.
// Iterative traversal of the flattened octree; only the boolean result is needed,
// so visiting order does not affect it.
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to, const DeviceScene& scene,
                             size_t srcTriIdx, size_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int32_t stack[STACK_CAPACITY];
    int sp = 0;
    stack[sp++] = 0;  // root

    while (sp > 0) {
        const DevNode& node = scene.nodes[stack[--sp]];
        if (node.triCount > 0) {
            for (uint32_t k = 0; k < node.triCount; ++k) {
                size_t idx = scene.triIdx[node.triBegin + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const Triangle& tri = scene.tris[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;  // Ray is blocked
                }
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int32_t child = node.child[c];
                if (child >= 0 && rayIntersectsBox(scene.nodes[child], from, to)) {
                    stack[sp++] = child;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Device: Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__device__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// Form factor Kij between triangle i (receiver) and triangle j (emitter).
// `draws` holds the DRAWS_PER_PAIR uniform samples for this pair, in the order
// the sequential code would have drawn them.
__device__ val_t computeKij(size_t idxI, size_t idxJ, const DeviceScene& scene, const val_t* draws) {
    const Triangle& triI = scene.tris[idxI];
    const Triangle& triJ = scene.tris[idxJ];

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        const val_t* d = draws + 4 * r;
        Vec3 pI = pointInTriangle(triI, d[0], d[1]);
        Vec3 pJ = pointInTriangle(triJ, d[2], d[3]);

        if (isRayBlocked(pI, pJ, scene, idxI, idxJ)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.normal());
        val_t cosPhiJ = cosPhi(-v, triJ.normal());

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    return kij * INV_NUM_RAYS;
}

// ============================================================================
// Kernels
// ============================================================================

// Time delays, stored transposed: tauT[j * N + i] = tau(i, j)
__global__ void computeTimeDelaysKernel(const Triangle* tris, size_t N, int* tauT) {
    size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= N * N) return;

    size_t i = p / N;
    size_t j = p % N;
    if (i == j) {
        tauT[j * N + i] = 0;  // Never used: the form factor on the diagonal is zero
        return;
    }
    Vec3 d = tris[i].center() - tris[j].center();
    tauT[j * N + i] = static_cast<int>(ceilf(d.norm() * INV_WAVE_SPEED));
}

// Form factors for a chunk of receiver rows [rowBegin, rowBegin + numPairs / N).
// Output is transposed: kijT[j * N + i] = Kij.
__global__ void formFactorKernel(DeviceScene scene, const val_t* draws, const uint8_t* active,
                                 size_t N, size_t rowBegin, size_t numPairs, val_t* kijT) {
    size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= numPairs) return;

    size_t i = rowBegin + p / N;
    size_t j = p % N;

    val_t kij = ZERO;
    if (active[p]) {
        kij = computeKij(i, j, scene, draws + p * DRAWS_PER_PAIR);
    }
    kijT[j * N + i] = kij;
}

// One wave-propagation timestep for all receivers. Each receiver accumulates its
// emitters in ascending j, the same order as the sequential code.
// Emission at tau >= 1 always reads earlier timesteps, so receivers are independent.
__global__ void simulateStepKernel(size_t t, size_t N, const val_t* kijT, const int* tauT,
                                   const val_t* areas, const val_t* rho, const val_t* radE,
                                   val_t* radB) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    val_t sumB = ZERO;

    for (size_t j = 0; j < N; ++j) {
        // Diagonal entries have zero form factor, so they are skipped here
        int tauij = tauT[j * N + i];

        // Skip if wave hasn't yet propagated from j to i
        if (static_cast<int>(t) < tauij) continue;

        val_t kij = kijT[j * N + i];
        if (kij <= ZERO) continue;

        // Radiosity from the source emission time
        size_t srcTime = t - static_cast<size_t>(tauij);
        val_t radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        // Accumulate contribution: form factor * area * source radiosity
        sumB += fminf(kij * areas[j], ONE) * radJ;
    }

    // Update radiosity: reflection + emission
    radB[t * N + i] = rho[i] * sumB + radE[t * N + i];
}

// Correlation sums: corr[i * T + t] = sum_{tt >= t} radB[tt][src] * radB[tt][i] (shifted)
__global__ void correlationKernel(size_t N, size_t T, size_t sourceIndex, const val_t* radB, val_t* corr) {
    size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= N * T) return;

    size_t i = p / T;
    size_t t = p % T;

    val_t sum = ZERO;
    for (size_t tt = t; tt < T; ++tt) {
        val_t pB = radB[tt * N + i];
        val_t pS = radB[(tt - t) * N + sourceIndex];
        sum += pS * pB;
    }
    corr[p] = sum;
}

// Pick the first lag with the largest correlation (strictly greater, as in the sequential code)
__global__ void argmaxKernel(size_t N, size_t T, const val_t* corr, val_t* distances) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    val_t maxCorr = ZERO;
    int bestT = 0;
    for (size_t t = 0; t < T; ++t) {
        val_t sum = corr[i * T + t];
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }

    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
}

// ============================================================================
// Device Memory Helpers
// ============================================================================

struct CudaFreeDeleter {
    void operator()(void* ptr) const { cudaFree(ptr); }
};

struct CudaFreeHostDeleter {
    void operator()(void* ptr) const { cudaFreeHost(ptr); }
};

template <typename T>
using DeviceArray = std::unique_ptr<T[], CudaFreeDeleter>;

template <typename T>
using PinnedArray = std::unique_ptr<T[], CudaFreeHostDeleter>;

// Allocations are at least one element so that pointers are always valid
template <typename T>
DeviceArray<T> allocDevice(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, std::max<size_t>(count, 1) * sizeof(T)));
    return DeviceArray<T>(ptr);
}

template <typename T>
PinnedArray<T> allocPinned(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, std::max<size_t>(count, 1) * sizeof(T)));
    return PinnedArray<T>(ptr);
}

template <typename T>
void uploadTo(DeviceArray<T>& dst, const std::vector<T>& src) {
    if (!src.empty()) {
        CUDA_CHECK(cudaMemcpy(dst.get(), src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
}

template <typename T>
void downloadFrom(const DeviceArray<T>& src, std::vector<T>& dst, size_t count) {
    dst.resize(count);
    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(dst.data(), src.get(), count * sizeof(T), cudaMemcpyDeviceToHost));
    }
}

inline unsigned blocksFor(size_t n) {
    return static_cast<unsigned>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct DeviceData {
    DeviceArray<Triangle> triangles;
    DeviceArray<DevNode> nodes;
    DeviceArray<uint32_t> triIdx;
    DeviceArray<val_t> areas;       // Area of each triangle
    DeviceArray<val_t> rho;         // Reflectivity (0.0 to 1.0)
    DeviceArray<val_t> radE;        // Emission radiosity (T x N matrix)
    DeviceArray<val_t> radB;        // Reflected radiosity (T x N matrix)
    DeviceArray<val_t> kijT;        // Form factors (N x N, transposed: [j * N + i])
    DeviceArray<int> tauT;          // Time delays (N x N, transposed: [j * N + i])
    DeviceArray<val_t> corr;        // Correlation sums (N x T)
    DeviceArray<val_t> distances;   // Computed distances from source
};

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> kij;         // Host copy of form factors, only filled for validation
    std::vector<val_t> radB;        // Host copy of radiosity, only filled for validation
    std::vector<val_t> distances;   // Host copy of computed distances

    Octree octree;                  // Spatial acceleration structure (host build)

    size_t sourceIndex;

    DeviceData gpu;

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

    const size_t N = state.numTriangles;
    const size_t T = timesteps;

    printf("Generated icosphere mesh with %zu triangles\n", N);

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    std::vector<DevNode> nodes;
    std::vector<uint32_t> triIdx;
    int maxDepth = 0;
    flattenOctree(state.octree, nodes, triIdx, 0, maxDepth);
    if (static_cast<size_t>(7 * maxDepth + 1) > static_cast<size_t>(STACK_CAPACITY)) {
        fprintf(stderr, "Octree depth %d exceeds traversal stack capacity\n", maxDepth);
        std::exit(EXIT_FAILURE);
    }

    // Initialize areas
    std::vector<val_t> areas(N);
    for (size_t i = 0; i < N; ++i) {
        areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    std::vector<val_t> rho(N, reflectivity);

    // Set source emission (active for first half of timesteps)
    std::vector<val_t> radE(T * N, ZERO);
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Upload geometry and simulation data
    DeviceData& gpu = state.gpu;
    gpu.triangles = allocDevice<Triangle>(N);
    uploadTo(gpu.triangles, state.triangles);
    gpu.nodes = allocDevice<DevNode>(nodes.size());
    uploadTo(gpu.nodes, nodes);
    gpu.triIdx = allocDevice<uint32_t>(triIdx.size());
    uploadTo(gpu.triIdx, triIdx);
    gpu.areas = allocDevice<val_t>(N);
    uploadTo(gpu.areas, areas);
    gpu.rho = allocDevice<val_t>(N);
    uploadTo(gpu.rho, rho);
    gpu.radE = allocDevice<val_t>(T * N);
    uploadTo(gpu.radE, radE);
    gpu.radB = allocDevice<val_t>(T * N);
    CUDA_CHECK(cudaMemset(gpu.radB.get(), 0, T * N * sizeof(val_t)));
    gpu.kijT = allocDevice<val_t>(N * N);
    gpu.tauT = allocDevice<int>(N * N);
    gpu.corr = allocDevice<val_t>(N * T);
    gpu.distances = allocDevice<val_t>(N);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    const size_t N = state.numTriangles;
    computeTimeDelaysKernel<<<blocksFor(N * N), BLOCK_SIZE>>>(
        state.gpu.triangles.get(), N, state.gpu.tauT.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");

    const size_t N = state.numTriangles;
    RandomGenerator rng(42);

    // The random sequence is consumed in the original order: row-major over pairs,
    // 64 draws per pair that is not culled. It is generated on the host in row chunks
    // and each chunk is processed on the GPU while the next one is being generated.
    const size_t rowsPerChunk = std::max<size_t>(1, std::min(N, MAX_CHUNK_DRAWS / (N * DRAWS_PER_PAIR)));
    const size_t maxPairs = rowsPerChunk * N;

    constexpr size_t NUM_STAGES = 2;
    PinnedArray<val_t> hDraws[NUM_STAGES];
    PinnedArray<uint8_t> hActive[NUM_STAGES];
    DeviceArray<val_t> dDraws[NUM_STAGES];
    DeviceArray<uint8_t> dActive[NUM_STAGES];
    cudaStream_t streams[NUM_STAGES];
    cudaEvent_t done[NUM_STAGES];

    for (size_t s = 0; s < NUM_STAGES; ++s) {
        hDraws[s] = allocPinned<val_t>(maxPairs * DRAWS_PER_PAIR);
        hActive[s] = allocPinned<uint8_t>(maxPairs);
        dDraws[s] = allocDevice<val_t>(maxPairs * DRAWS_PER_PAIR);
        dActive[s] = allocDevice<uint8_t>(maxPairs);
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaEventCreate(&done[s]));
        CUDA_CHECK(cudaEventRecord(done[s], streams[s]));
    }

    size_t chunk = 0;
    for (size_t r0 = 0; r0 < N; r0 += rowsPerChunk, ++chunk) {
        const size_t r1 = std::min(N, r0 + rowsPerChunk);
        const size_t numPairs = (r1 - r0) * N;
        const size_t s = chunk % NUM_STAGES;

        // Wait until the GPU is done with this staging buffer
        CUDA_CHECK(cudaEventSynchronize(done[s]));

        val_t* draws = hDraws[s].get();
        uint8_t* active = hActive[s].get();
        for (size_t p = 0; p < numPairs; ++p) {
            const size_t i = r0 + p / N;
            const size_t j = p % N;
            // Self pairs and culled pairs consume no random numbers
            const bool isActive = i != j &&
                !(state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f);
            active[p] = isActive;
            if (isActive) {
                for (size_t k = 0; k < DRAWS_PER_PAIR; ++k) {
                    draws[p * DRAWS_PER_PAIR + k] = rng.rand();
                }
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(dActive[s].get(), active, numPairs * sizeof(uint8_t),
                                   cudaMemcpyHostToDevice, streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(dDraws[s].get(), draws, numPairs * DRAWS_PER_PAIR * sizeof(val_t),
                                   cudaMemcpyHostToDevice, streams[s]));

        DeviceScene scene{state.gpu.triangles.get(), state.gpu.nodes.get(), state.gpu.triIdx.get()};
        formFactorKernel<<<blocksFor(numPairs), BLOCK_SIZE, 0, streams[s]>>>(
            scene, dDraws[s].get(), dActive[s].get(), N, r0, numPairs, state.gpu.kijT.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(done[s], streams[s]));

        printf("  Progress: %zu/%zu triangles\n", r1, N);
    }

    for (size_t s = 0; s < NUM_STAGES; ++s) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        CUDA_CHECK(cudaEventDestroy(done[s]));
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    DeviceData& gpu = state.gpu;

    for (size_t t = 0; t < T; ++t) {
        simulateStepKernel<<<blocksFor(N), BLOCK_SIZE>>>(
            t, N, gpu.kijT.get(), gpu.tauT.get(), gpu.areas.get(), gpu.rho.get(),
            gpu.radE.get(), gpu.radB.get());
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

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    DeviceData& gpu = state.gpu;

    if (N * T > 0) {
        correlationKernel<<<blocksFor(N * T), BLOCK_SIZE>>>(
            N, T, state.sourceIndex, gpu.radB.get(), gpu.corr.get());
        CUDA_CHECK(cudaGetLastError());
    }

    argmaxKernel<<<blocksFor(N), BLOCK_SIZE>>>(N, T, gpu.corr.get(), gpu.distances.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Copy results back to the host. Matrices are only needed for validation.
void downloadResults(SimulationState& state, bool includeMatrices) {
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    DeviceData& gpu = state.gpu;

    downloadFrom(gpu.distances, state.distances, N);
    if (includeMatrices) {
        downloadFrom(gpu.kijT, state.kij, N * N);
        downloadFrom(gpu.radB, state.radB, T * N);
    }
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

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("Room Response Simulation Benchmark (CUDA)\n");
    printf("===================================\n");
    printf("Device: %s\n", prop.name);
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

    downloadResults(state, validate);

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
