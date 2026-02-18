/**
 * Room Response Simulation Benchmark - CUDA Parallelized
 *
 * GPU-parallelized implementation of room impulse response simulation using
 * radiosity-based wave propagation. All compute-intensive phases run on GPU:
 *
 * 1. Form factors (Kij) - one GPU thread per triangle pair
 * 2. Time delays (Tau) - one GPU thread per triangle pair
 * 3. Wave propagation - one GPU thread per triangle per timestep
 * 4. Distance estimation - one GPU thread per triangle
 *
 * Mesh generation and octree construction remain on CPU.
 * The octree is flattened into arrays for GPU traversal.
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

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
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
constexpr val_t WAVE_SPEED = 0.5f;
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;
constexpr val_t NO_HIT = 1e30f;

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
// Flat Octree Node for GPU Traversal
// ============================================================================

struct FlatOctreeNode {
    Vec3 halfExtent, center;
    int children[8];
    int triIdxStart;
    int triIdxCount;
};

// ============================================================================
// Triangle-Box Overlap Test (host only, for octree construction)
// ============================================================================

bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
    Vec3 v0 = tri.a - boxCenter;
    Vec3 v1 = tri.b - boxCenter;
    Vec3 v2 = tri.c - boxCenter;

    Vec3 e0 = v1 - v0;
    Vec3 e1 = v2 - v1;
    Vec3 e2 = v0 - v2;

    auto minMax3 = [](val_t a, val_t b, val_t c) {
        return std::make_pair(std::min({a, b, c}), std::max({a, b, c}));
    };

    auto [minX, maxX] = minMax3(v0.x, v1.x, v2.x);
    if (minX > boxHalfSize.x || maxX < -boxHalfSize.x) return false;

    auto [minY, maxY] = minMax3(v0.y, v1.y, v2.y);
    if (minY > boxHalfSize.y || maxY < -boxHalfSize.y) return false;

    auto [minZ, maxZ] = minMax3(v0.z, v1.z, v2.z);
    if (minZ > boxHalfSize.z || maxZ < -boxHalfSize.z) return false;

    Vec3 triNormal = e0.cross(e1);
    val_t d = triNormal.dot(v0);
    val_t r = boxHalfSize.x * std::abs(triNormal.x) +
              boxHalfSize.y * std::abs(triNormal.y) +
              boxHalfSize.z * std::abs(triNormal.z);
    if (std::abs(d) > r) return false;

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
// Octree for Spatial Acceleration (host only)
// ============================================================================

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;

class Octree {
public:
    Vec3 minBound, maxBound;
    Vec3 halfExtent, center;
    std::unique_ptr<Octree> children[8];
    std::vector<size_t> triangleIndices;
    const std::vector<Triangle>* allTriangles;

    Octree() : allTriangles(nullptr) {}

    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if (triangles.empty()) return;

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

        if (indices.size() <= MAX_OCTREE_TRIS ||
            (maxBound - minBound).norm() < MAX_OCTREE_LEAF_SIZE) {
            triangleIndices = indices;
            return;
        }

        Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];

        for (size_t idx : indices) {
            const Triangle& tri = (*allTriangles)[idx];

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
// Flatten Octree for GPU
// ============================================================================

void flattenOctree(const Octree& root,
                   std::vector<FlatOctreeNode>& nodes,
                   std::vector<int>& triIndices) {
    nodes.clear();
    triIndices.clear();

    struct Entry {
        const Octree* node;
        int flatIdx;
    };

    FlatOctreeNode rootNode;
    rootNode.halfExtent = root.halfExtent;
    rootNode.center = root.center;
    for (int i = 0; i < 8; i++) rootNode.children[i] = -1;
    rootNode.triIdxStart = 0;
    rootNode.triIdxCount = 0;
    nodes.push_back(rootNode);

    std::vector<Entry> stack;
    stack.push_back({&root, 0});

    while (!stack.empty()) {
        Entry entry = stack.back();
        stack.pop_back();

        const Octree* oct = entry.node;
        int flatIdx = entry.flatIdx;

        if (!oct->triangleIndices.empty()) {
            nodes[flatIdx].triIdxStart = (int)triIndices.size();
            nodes[flatIdx].triIdxCount = (int)oct->triangleIndices.size();
            for (size_t idx : oct->triangleIndices) {
                triIndices.push_back((int)idx);
            }
            continue;
        }

        for (int i = 0; i < 8; i++) {
            if (oct->children[i]) {
                int childFlatIdx = (int)nodes.size();
                nodes[flatIdx].children[i] = childFlatIdx;

                FlatOctreeNode childNode;
                childNode.halfExtent = oct->children[i]->halfExtent;
                childNode.center = oct->children[i]->center;
                for (int j = 0; j < 8; j++) childNode.children[j] = -1;
                childNode.triIdxStart = 0;
                childNode.triIdxCount = 0;
                nodes.push_back(childNode);

                stack.push_back({oct->children[i].get(), childFlatIdx});
            }
        }
    }
}

// ============================================================================
// Mesh Generation: Icosphere (host only)
// ============================================================================

class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
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

        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

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

        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Device Random Number Generator (xorshift32)
// ============================================================================

struct DeviceRNG {
    unsigned int state;

    __device__ DeviceRNG(unsigned int seed) {
        // Wang hash for seed mixing
        seed = (seed ^ 61u) ^ (seed >> 16u);
        seed *= 9u;
        seed ^= seed >> 4u;
        seed *= 0x27d4eb2du;
        seed ^= seed >> 15u;
        state = seed != 0 ? seed : 1u;
    }

    __device__ float rand() {
        state ^= state << 13u;
        state ^= state >> 17u;
        state ^= state << 5u;
        return (state & 0xFFFFFFu) / (float)0x1000000u;
    }
};

// ============================================================================
// Device Helper Functions
// ============================================================================

__device__ Vec3 randomPointInTriangle_d(const Triangle& t, DeviceRNG& rng) {
    val_t u = rng.rand();
    val_t v = rng.rand();
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    return t.a + (t.b - t.a) * u + (t.c - t.a) * v;
}

__device__ val_t rayTriangleIntersect_d(const Vec3& orig, const Vec3& dir,
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

__device__ bool rayIntersectsBox_d(const Vec3& p1, const Vec3& p2,
                                    const Vec3& halfExtent, const Vec3& center) {
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

__device__ bool isRayBlocked_d(const Vec3& from, const Vec3& to,
                                const FlatOctreeNode* nodes,
                                const int* triIndices,
                                const Triangle* triangles,
                                int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        const FlatOctreeNode& node = nodes[nodeIdx];

        if (!rayIntersectsBox_d(from, to, node.halfExtent, node.center)) continue;

        if (node.triIdxCount > 0) {
            for (int i = 0; i < node.triIdxCount; ++i) {
                int idx = triIndices[node.triIdxStart + i];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                val_t dist = rayTriangleIntersect_d(from, dirNorm,
                    triangles[idx].a, triangles[idx].b, triangles[idx].c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                if (node.children[c] >= 0) {
                    stack[stackPtr++] = node.children[c];
                }
            }
        }
    }

    return false;
}

__device__ val_t cosPhi_d(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void computeFormFactorsKernel(
    val_t* kij, const Triangle* triangles,
    const FlatOctreeNode* octreeNodes, const int* octreeTriIndices,
    int numTriangles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles * numTriangles) return;

    int i = idx / numTriangles;
    int j = idx % numTriangles;

    if (i == j) { kij[idx] = ZERO; return; }

    const Triangle& triI = triangles[i];
    const Triangle& triJ = triangles[j];

    if (triI._normal.dot(triJ._normal) > 0.99f) { kij[idx] = ZERO; return; }

    DeviceRNG rng(idx + 42u);
    val_t result = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle_d(triI, rng);
        Vec3 pJ = randomPointInTriangle_d(triJ, rng);

        if (isRayBlocked_d(pI, pJ, octreeNodes, octreeTriIndices, triangles, i, j)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi_d(v, triI._normal);
        val_t cosPhiJ = cosPhi_d(-v, triJ._normal);

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        result += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kij[idx] = result * INV_NUM_RAYS;
}

__global__ void computeTimeDelaysKernel(
    int* tau, const Triangle* triangles, int numTriangles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles * numTriangles) return;

    int i = idx / numTriangles;
    int j = idx % numTriangles;

    if (i == j) { tau[idx] = 0; return; }

    Vec3 ci = triangles[i].center();
    Vec3 cj = triangles[j].center();
    val_t dist = (ci - cj).norm();
    tau[idx] = (int)ceilf(dist * INV_WAVE_SPEED);
}

__global__ void runSimulationStepKernel(
    val_t* radB, const val_t* radE, const val_t* kij, const int* tau,
    const val_t* areas, const val_t* rho,
    int numTriangles, int t)
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
    val_t* distances, const val_t* radB,
    int numTriangles, int numTimesteps, int sourceIndex)
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
    std::vector<val_t> areas;
    std::vector<val_t> rho;
    std::vector<val_t> kij;
    std::vector<int> tau;
    std::vector<val_t> radE;
    std::vector<val_t> radB;
    std::vector<val_t> distances;

    Octree octree;

    std::vector<FlatOctreeNode> flatOctreeNodes;
    std::vector<int> flatOctreeTriIndices;

    // Device pointers
    Triangle* d_triangles;
    val_t* d_areas;
    val_t* d_rho;
    val_t* d_kij;
    int* d_tau;
    val_t* d_radE;
    val_t* d_radB;
    val_t* d_distances;
    FlatOctreeNode* d_octreeNodes;
    int* d_octreeTriIndices;

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    printf("Building octree...\n");
    state.octree.build(state.triangles);

    printf("Flattening octree for GPU...\n");
    flattenOctree(state.octree, state.flatOctreeNodes, state.flatOctreeTriIndices);
    printf("Flat octree: %zu nodes, %zu triangle indices\n",
           state.flatOctreeNodes.size(), state.flatOctreeTriIndices.size());

    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    state.rho.resize(state.numTriangles, reflectivity);

    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.tau.resize(state.numTriangles * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Allocate GPU memory
    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;

    CUDA_CHECK(cudaMalloc(&state.d_triangles, N * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&state.d_areas, N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_rho, N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_kij, N * N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_tau, N * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_radE, T * N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_radB, T * N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_distances, N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_octreeNodes, state.flatOctreeNodes.size() * sizeof(FlatOctreeNode)));
    CUDA_CHECK(cudaMalloc(&state.d_octreeTriIndices, state.flatOctreeTriIndices.size() * sizeof(int)));

    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(state.d_triangles, state.triangles.data(), N * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_radE, state.radE.data(), T * N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_radB, state.radB.data(), T * N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_octreeNodes, state.flatOctreeNodes.data(),
                          state.flatOctreeNodes.size() * sizeof(FlatOctreeNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_octreeTriIndices, state.flatOctreeTriIndices.data(),
                          state.flatOctreeTriIndices.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(state.d_kij, 0, N * N * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.d_tau, 0, N * N * sizeof(int)));
    CUDA_CHECK(cudaMemset(state.d_distances, 0, N * sizeof(val_t)));
}

// ============================================================================
// Precomputation Phase (GPU)
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij) on GPU...\n");
    int N = (int)state.numTriangles;
    int total = N * N;
    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;

    computeFormFactorsKernel<<<numBlocks, blockSize>>>(
        state.d_kij, state.d_triangles,
        state.d_octreeNodes, state.d_octreeTriIndices, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau) on GPU...\n");
    int N = (int)state.numTriangles;
    int total = N * N;
    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;

    computeTimeDelaysKernel<<<numBlocks, blockSize>>>(
        state.d_tau, state.d_triangles, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (GPU)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation on GPU...\n");
    int N = (int)state.numTriangles;
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    for (int t = 0; t < (int)state.numTimesteps; ++t) {
        runSimulationStepKernel<<<numBlocks, blockSize>>>(
            state.d_radB, state.d_radE, state.d_kij, state.d_tau,
            state.d_areas, state.d_rho, N, t);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Completed %zu timesteps\n", state.numTimesteps);
}

// ============================================================================
// Distance Computation (GPU)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation on GPU...\n");
    int N = (int)state.numTriangles;
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    computeDistancesKernel<<<numBlocks, blockSize>>>(
        state.d_distances, state.d_radB,
        N, (int)state.numTimesteps, (int)state.sourceIndex);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Copy Results and Free GPU Memory
// ============================================================================

void copyResultsBack(SimulationState& state) {
    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;

    CUDA_CHECK(cudaMemcpy(state.kij.data(), state.d_kij, N * N * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.tau.data(), state.d_tau, N * N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB, T * N * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.d_distances, N * sizeof(val_t), cudaMemcpyDeviceToHost));
}

void freeDeviceMemory(SimulationState& state) {
    cudaFree(state.d_triangles);
    cudaFree(state.d_areas);
    cudaFree(state.d_rho);
    cudaFree(state.d_kij);
    cudaFree(state.d_tau);
    cudaFree(state.d_radE);
    cudaFree(state.d_radB);
    cudaFree(state.d_distances);
    cudaFree(state.d_octreeNodes);
    cudaFree(state.d_octreeTriIndices);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");

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

    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

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

    // Copy results back from GPU
    copyResultsBack(state);
    freeDeviceMemory(state);

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
