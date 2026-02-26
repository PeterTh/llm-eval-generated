/**
 * Room Response Simulation Benchmark - CUDA GPU Implementation
 * 
 * This is a CUDA-parallelized implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 * GPU parallelism is used for form factor computation, time delay computation,
 * wave propagation simulation, and distance computation.
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
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

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
// Vector and Triangle Types (host + device)
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    __host__ __device__ explicit Vec3(val_t v) : x(v), y(v), z(v) {}

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

    __host__ __device__ Triangle() : a(), b(), c(), _normal() {}
    __host__ __device__ Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    __host__ __device__ val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    __host__ __device__ bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

// ============================================================================
// Flat Octree for GPU Traversal
// ============================================================================

struct FlatOctreeNode {
    float cx, cy, cz;    // center
    float hx, hy, hz;    // half extent
    int children[8];      // child indices, -1 if empty
    int triStart;         // start index in flat triangle index array
    int triCount;         // number of triangles (>0 for leaf nodes)
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
// Octree for Spatial Acceleration (host only, used for building)
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
// Octree Flattening (for GPU)
// ============================================================================

static int countOctreeNodes(const Octree* node) {
    if (!node) return 0;
    int count = 1;
    if (node->triangleIndices.empty()) {
        for (int i = 0; i < 8; i++) {
            if (node->children[i]) {
                count += countOctreeNodes(node->children[i].get());
            }
        }
    }
    return count;
}

static int flattenOctreeHelper(const Octree* node,
                               std::vector<FlatOctreeNode>& nodes,
                               std::vector<int>& triIndices) {
    int myIdx = (int)nodes.size();
    nodes.push_back(FlatOctreeNode{});

    // Safe to write here since we reserved enough capacity (no reallocation)
    nodes[myIdx].cx = node->center.x;
    nodes[myIdx].cy = node->center.y;
    nodes[myIdx].cz = node->center.z;
    nodes[myIdx].hx = node->halfExtent.x;
    nodes[myIdx].hy = node->halfExtent.y;
    nodes[myIdx].hz = node->halfExtent.z;

    for (int i = 0; i < 8; i++) nodes[myIdx].children[i] = -1;

    if (!node->triangleIndices.empty()) {
        // Leaf node
        nodes[myIdx].triStart = (int)triIndices.size();
        nodes[myIdx].triCount = (int)node->triangleIndices.size();
        for (size_t idx : node->triangleIndices) {
            triIndices.push_back((int)idx);
        }
    } else {
        // Internal node
        nodes[myIdx].triStart = 0;
        nodes[myIdx].triCount = 0;

        int childIdx[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        for (int i = 0; i < 8; i++) {
            if (node->children[i]) {
                childIdx[i] = flattenOctreeHelper(node->children[i].get(), nodes, triIndices);
            }
        }
        for (int i = 0; i < 8; i++) {
            nodes[myIdx].children[i] = childIdx[i];
        }
    }

    return myIdx;
}

static void flattenOctree(const Octree& octree,
                           std::vector<FlatOctreeNode>& nodes,
                           std::vector<int>& triIndices) {
    int totalNodes = countOctreeNodes(&octree);
    nodes.reserve(totalNodes);
    triIndices.reserve(totalNodes * 4);
    flattenOctreeHelper(&octree, nodes, triIndices);
}

// ============================================================================
// GPU Device Functions
// ============================================================================

__device__ uint32_t gpu_hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    return x;
}

__device__ float gpu_rand(uint32_t& state) {
    state = state * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    word = (word >> 22u) ^ word;
    return (float)(word & 0x00FFFFFFu) / (float)0x01000000u;
}

__device__ bool gpu_rayIntersectsBox(
    const Vec3& p1, const Vec3& p2,
    float cx, float cy, float cz,
    float hx, float hy, float hz)
{
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - Vec3(cx, cy, cz);
    float adx = fabsf(d.x), ady = fabsf(d.y), adz = fabsf(d.z);

    if (fabsf(c.x) > hx + adx) return false;
    if (fabsf(c.y) > hy + ady) return false;
    if (fabsf(c.z) > hz + adz) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > hy * adz + hz * ady + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > hz * adx + hx * adz + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > hx * ady + hy * adx + EPSILON) return false;

    return true;
}

__device__ bool gpu_isRayBlocked(
    const Vec3& from, const Vec3& to,
    const FlatOctreeNode* nodes, const int* triIndices,
    const Triangle* triangles,
    int srcTriIdx, int dstTriIdx)
{
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    // Stack-based octree traversal
    int stack[64];
    int stackTop = 0;
    stack[stackTop++] = 0; // root node

    while (stackTop > 0) {
        int nodeIdx = stack[--stackTop];
        const FlatOctreeNode& node = nodes[nodeIdx];

        if (node.triCount > 0) {
            // Leaf node - test triangles
            for (int t = 0; t < node.triCount; t++) {
                int triIdx = triIndices[node.triStart + t];
                if (triIdx == srcTriIdx || triIdx == dstTriIdx) continue;

                const Triangle& tri = triangles[triIdx];

                // Moller-Trumbore intersection
                Vec3 e1 = tri.b - tri.a;
                Vec3 e2 = tri.c - tri.a;
                Vec3 pvec = dirNorm.cross(e2);
                val_t det = e1.dot(pvec);
                if (fabsf(det) < EPSILON) continue;

                val_t invDet = 1.0f / det;
                Vec3 tvec = from - tri.a;
                val_t u = tvec.dot(pvec) * invDet;
                if (u < 0.0f || u > 1.0f) continue;

                Vec3 qvec = tvec.cross(e1);
                val_t v = dirNorm.dot(qvec) * invDet;
                if (v < 0.0f || u + v > 1.0f) continue;

                val_t dist = e2.dot(qvec) * invDet;
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;
                }
            }
        } else {
            // Internal node - push children that intersect ray
            for (int i = 7; i >= 0; i--) {
                int childIdx = node.children[i];
                if (childIdx >= 0) {
                    const FlatOctreeNode& child = nodes[childIdx];
                    if (gpu_rayIntersectsBox(from, to,
                            child.cx, child.cy, child.cz,
                            child.hx, child.hy, child.hz)) {
                        if (stackTop < 64) {
                            stack[stackTop++] = childIdx;
                        }
                    }
                }
            }
        }
    }

    return false;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void computeTimeDelaysKernel(
    int* tau,
    const Triangle* triangles,
    int numTriangles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPairs = numTriangles * numTriangles;
    if (idx >= totalPairs) return;

    int i = idx / numTriangles;
    int j = idx % numTriangles;

    if (i == j) {
        tau[idx] = 0;
        return;
    }

    Vec3 ci = triangles[i].center();
    Vec3 cj = triangles[j].center();
    val_t dist = (ci - cj).norm();
    tau[idx] = (int)ceilf(dist * INV_WAVE_SPEED);
}

__global__ void computeFormFactorsKernel(
    val_t* kij,
    const Triangle* triangles,
    const FlatOctreeNode* octreeNodes,
    const int* octreeTriIndices,
    int numTriangles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPairs = numTriangles * numTriangles;
    if (idx >= totalPairs) return;

    int i = idx / numTriangles;
    int j = idx % numTriangles;

    if (i == j) {
        kij[idx] = ZERO;
        return;
    }

    const Triangle& triI = triangles[i];
    const Triangle& triJ = triangles[j];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) {
        kij[idx] = ZERO;
        return;
    }

    // Per-thread deterministic RNG seeded by (i, j)
    uint32_t rngState = gpu_hash((uint32_t)i * 1000003u + (uint32_t)j * 999983u + 42u);

    val_t kijVal = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        // Random point in triangle I
        float u1 = gpu_rand(rngState);
        float v1 = gpu_rand(rngState);
        if (u1 + v1 > 1.0f) { u1 = 1.0f - u1; v1 = 1.0f - v1; }
        Vec3 pI = triI.a + (triI.b - triI.a) * u1 + (triI.c - triI.a) * v1;

        // Random point in triangle J
        float u2 = gpu_rand(rngState);
        float v2 = gpu_rand(rngState);
        if (u2 + v2 > 1.0f) { u2 = 1.0f - u2; v2 = 1.0f - v2; }
        Vec3 pJ = triJ.a + (triJ.b - triJ.a) * u2 + (triJ.c - triJ.a) * v2;

        if (gpu_isRayBlocked(pI, pJ, octreeNodes, octreeTriIndices, triangles, i, j)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t vNorm = sqrtf(distSqr);
        val_t cosPhiI = fmaxf(ZERO, v.dot(triI.normal()) / vNorm);
        val_t cosPhiJ = fmaxf(ZERO, (-v).dot(triJ.normal()) / vNorm);

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kijVal += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kij[idx] = kijVal * INV_NUM_RAYS;
}

__global__ void simulationStepKernel(
    val_t* radB,
    const val_t* radE,
    const val_t* kij,
    const int* tau,
    const val_t* areas,
    const val_t* rho,
    int numTriangles,
    int t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numTriangles) return;

    val_t sumB = ZERO;

    for (int j = 0; j < numTriangles; ++j) {
        if (i == j) continue;

        int tauij = tau[i * numTriangles + j];
        if (t < tauij) continue;

        val_t k = kij[i * numTriangles + j];
        if (k <= ZERO) continue;

        int srcTime = t - tauij;
        val_t radJ = radB[srcTime * numTriangles + j];
        if (radJ <= ZERO) continue;

        sumB += fminf(k * areas[j], ONE) * radJ;
    }

    radB[t * numTriangles + i] = rho[i] * sumB + radE[t * numTriangles + i];
}

__global__ void computeDistancesKernel(
    val_t* distances,
    const val_t* radB,
    int numTriangles,
    int numTimesteps,
    int sourceIndex)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numTriangles) return;

    val_t maxCorr = ZERO;
    int bestT = 0;

    for (int t = 0; t < numTimesteps; ++t) {
        val_t sum = ZERO;

        for (int tt = t; tt < numTimesteps; ++tt) {
            val_t pB = radB[tt * numTriangles + i];
            val_t pS = radB[(tt - t) * numTriangles + sourceIndex];
            sum += pS * pB;
        }

        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distances[i] = WAVE_SPEED * (val_t)bestT;
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

    // Flat octree for GPU traversal
    std::vector<FlatOctreeNode> flatOctreeNodes;
    std::vector<int> flatOctreeTriIndices;

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

    // Flatten octree for GPU traversal
    printf("Flattening octree for GPU...\n");
    flattenOctree(state.octree, state.flatOctreeNodes, state.flatOctreeTriIndices);
    printf("  Flat octree: %zu nodes, %zu triangle indices\n",
           state.flatOctreeNodes.size(), state.flatOctreeTriIndices.size());

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.tau.resize(state.numTriangles * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// ============================================================================
// GPU Computation Phases
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij) on GPU...\n");

    int N = (int)state.numTriangles;
    int totalPairs = N * N;

    Triangle* d_triangles;
    val_t* d_kij;
    FlatOctreeNode* d_octreeNodes;
    int* d_octreeTriIndices;

    CUDA_CHECK(cudaMalloc(&d_triangles, N * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&d_kij, (size_t)totalPairs * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_octreeNodes, state.flatOctreeNodes.size() * sizeof(FlatOctreeNode)));
    CUDA_CHECK(cudaMalloc(&d_octreeTriIndices, state.flatOctreeTriIndices.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_triangles, state.triangles.data(), N * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_octreeNodes, state.flatOctreeNodes.data(),
                          state.flatOctreeNodes.size() * sizeof(FlatOctreeNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_octreeTriIndices, state.flatOctreeTriIndices.data(),
                          state.flatOctreeTriIndices.size() * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (totalPairs + blockSize - 1) / blockSize;
    computeFormFactorsKernel<<<numBlocks, blockSize>>>(
        d_kij, d_triangles, d_octreeNodes, d_octreeTriIndices, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.kij.data(), d_kij, (size_t)totalPairs * sizeof(val_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_triangles));
    CUDA_CHECK(cudaFree(d_kij));
    CUDA_CHECK(cudaFree(d_octreeNodes));
    CUDA_CHECK(cudaFree(d_octreeTriIndices));
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau) on GPU...\n");

    int N = (int)state.numTriangles;
    int totalPairs = N * N;

    Triangle* d_triangles;
    int* d_tau;

    CUDA_CHECK(cudaMalloc(&d_triangles, N * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&d_tau, (size_t)totalPairs * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_triangles, state.triangles.data(), N * sizeof(Triangle), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (totalPairs + blockSize - 1) / blockSize;
    computeTimeDelaysKernel<<<numBlocks, blockSize>>>(d_tau, d_triangles, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.tau.data(), d_tau, (size_t)totalPairs * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_triangles));
    CUDA_CHECK(cudaFree(d_tau));
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation on GPU...\n");

    int N = (int)state.numTriangles;
    int T = (int)state.numTimesteps;

    val_t *d_radB, *d_radE, *d_kij, *d_areas, *d_rho;
    int *d_tau;

    CUDA_CHECK(cudaMalloc(&d_radB, (size_t)T * N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_radE, (size_t)T * N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_kij, (size_t)N * N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_tau, (size_t)N * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_areas, N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_rho, N * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(d_radB, state.radB.data(), (size_t)T * N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radE, state.radE.data(), (size_t)T * N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kij, state.kij.data(), (size_t)N * N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tau, state.tau.data(), (size_t)N * N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_areas, state.areas.data(), N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rho, state.rho.data(), N * sizeof(val_t), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    for (int t = 0; t < T; ++t) {
        simulationStepKernel<<<numBlocks, blockSize>>>(
            d_radB, d_radE, d_kij, d_tau, d_areas, d_rho, N, t);

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.radB.data(), d_radB, (size_t)T * N * sizeof(val_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_radB));
    CUDA_CHECK(cudaFree(d_radE));
    CUDA_CHECK(cudaFree(d_kij));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_areas));
    CUDA_CHECK(cudaFree(d_rho));
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation on GPU...\n");

    int N = (int)state.numTriangles;
    int T = (int)state.numTimesteps;

    val_t *d_distances, *d_radB;

    CUDA_CHECK(cudaMalloc(&d_distances, N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_radB, (size_t)T * N * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(d_radB, state.radB.data(), (size_t)T * N * sizeof(val_t), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    computeDistancesKernel<<<numBlocks, blockSize>>>(
        d_distances, d_radB, N, T, (int)state.sourceIndex);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), d_distances, N * sizeof(val_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_radB));
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

    printf("Room Response Simulation Benchmark (CUDA)\n");
    printf("==========================================\n");
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
