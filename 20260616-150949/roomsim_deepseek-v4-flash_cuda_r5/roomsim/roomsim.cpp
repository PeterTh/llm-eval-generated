/**
 * Room Response Simulation Benchmark (CUDA Parallelized)
 * 
 * CUDA-accelerated implementation of room impulse response simulation
 * using radiosity-based wave propagation. Computes:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 */

#include <cuda_runtime.h>

#include <algorithm>
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

#include "../common/results_output.hpp"

// ============================================================================
// CUDA Error Checking Macro
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t _err = call; \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================================
// Types and Constants (accessible from both host and device)
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

#define VAL_PI 3.14159265358979323846f
#define VAL_ZERO 0.0f
#define VAL_ONE 1.0f
#define VAL_WAVE_SPEED 0.5f
#define VAL_INV_WAVE_SPEED (1.0f / 0.5f)
#define VAL_NUM_RAYS 16
#define VAL_INV_NUM_RAYS (1.0f / 16.0f)
#define VAL_EPSILON 1e-6f

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
        return n > VAL_EPSILON ? *this / n : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < VAL_EPSILON && fabsf(y - o.y) < VAL_EPSILON && fabsf(z - o.z) < VAL_EPSILON;
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
// GPU Data Structures
// ============================================================================

// Flattened octree node for GPU
struct GPUOctreeNode {
    float cx, cy, cz;          // center
    float hx, hy, hz;          // half extent
    int children[8];            // child node indices (-1 = none)
    int triStart;               // start index in leaf triangle index array
    int triCount;               // triangle count (0 = internal node)
};

// ============================================================================
// CUDA Device Functions
// ============================================================================

// XOR-shift RNG for GPU (per-thread state)
__device__ inline uint32_t xorshift32(uint32_t& state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

__device__ inline float randFloatGPU(uint32_t& state) {
    // Convert to [0, 1)
    return (float)xorshift32(state) * 2.3283064365386963e-10f;
}

__device__ Vec3 randomPointInTriangleGPU(const Triangle& t, uint32_t& rngState) {
    float u = randFloatGPU(rngState);
    float v = randFloatGPU(rngState);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    return t.a + (t.b - t.a) * u + (t.c - t.a) * v;
}

__device__ inline float cosPhiGPU(const Vec3& v, const Vec3& normal) {
    float vNorm = v.norm();
    if (vNorm <= VAL_EPSILON) return 0.0f;
    return fmaxf(0.0f, v.dot(normal) / vNorm);
}

// Ray-box intersection for flattened octree on GPU
__device__ bool rayIntersectsBoxGPU(const Vec3& p1, const Vec3& p2,
                                    const GPUOctreeNode& node) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - Vec3(node.cx, node.cy, node.cz);
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    Vec3 halfExt = {node.hx, node.hy, node.hz};

    if (fabsf(c.x) > halfExt.x + ad.x) return false;
    if (fabsf(c.y) > halfExt.y + ad.y) return false;
    if (fabsf(c.z) > halfExt.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > halfExt.y * ad.z + halfExt.z * ad.y + VAL_EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > halfExt.z * ad.x + halfExt.x * ad.z + VAL_EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > halfExt.x * ad.y + halfExt.y * ad.x + VAL_EPSILON) return false;

    return true;
}

// Ray-triangle intersection (Möller-Trumbore) on GPU
__device__ float rayTriangleIntersectGPU(const Vec3& orig, const Vec3& dir,
                                          const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    float det = e1.dot(pvec);

    if (fabsf(det) < VAL_EPSILON) return 3.40282347e+38f;

    float invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    float u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 3.40282347e+38f;

    Vec3 qvec = tvec.cross(e1);
    float v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 3.40282347e+38f;

    return e2.dot(qvec) * invDet;
}

// Octree ray traversal on GPU (stack-based DFS)
__device__ bool isRayBlockedGPU(const Vec3& from, const Vec3& to,
                                 const GPUOctreeNode* nodes,
                                 const int* triIndices,
                                 const Triangle* triangles,
                                 int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    float rayLen = sqrtf(dir.squaredNorm());
    if (rayLen < VAL_EPSILON) return true;
    Vec3 dirNorm = dir * (1.0f / rayLen);

    int stack[64];
    int sp = 0;
    stack[sp++] = 0; // root node

    while (sp > 0) {
        int nodeIdx = stack[--sp];
        const GPUOctreeNode& node = nodes[nodeIdx];

        if (!rayIntersectsBoxGPU(from, to, node)) continue;

        if (node.triCount > 0) {
            // Leaf: check triangles
            for (int k = 0; k < node.triCount; ++k) {
                int idx = triIndices[node.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const Triangle& tri = triangles[idx];
                float dist = rayTriangleIntersectGPU(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > VAL_EPSILON && dist < rayLen - VAL_EPSILON) return true;
            }
        } else {
            // Internal: push children in reverse order (for correct 0..7 traversal)
            for (int c = 7; c >= 0; --c) {
                if (node.children[c] >= 0) {
                    stack[sp++] = node.children[c];
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
    const Triangle* triangles, int* tau, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int i = idx / N;
    int j = idx % N;
    if (i == j) return;

    Vec3 ci = (triangles[i].a + triangles[i].b + triangles[i].c) * (1.0f / 3.0f);
    Vec3 cj = (triangles[j].a + triangles[j].b + triangles[j].c) * (1.0f / 3.0f);
    float dist = sqrtf((ci - cj).squaredNorm());
    tau[idx] = (int)ceilf(dist * VAL_INV_WAVE_SPEED);
}

__global__ void computeFormFactorsKernel(
    const Triangle* triangles,
    const GPUOctreeNode* octreeNodes,
    const int* triIndices,
    float* kij,
    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int i = idx / N;
    int j = idx % N;
    if (i == j) { kij[idx] = 0.0f; return; }

    const Triangle& triI = triangles[i];
    const Triangle& triJ = triangles[j];

    // Cull triangles facing the same direction
    if (triI._normal.dot(triJ._normal) > 0.99f) { kij[idx] = 0.0f; return; }

    // Per-thread RNG seeded uniquely per (i,j)
    uint32_t rngState = (uint32_t)(i * 2654435761U + j * 2246822519U + 1U);

    float kijVal = 0.0f;
    for (int r = 0; r < VAL_NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangleGPU(triI, rngState);
        Vec3 pJ = randomPointInTriangleGPU(triJ, rngState);

        if (isRayBlockedGPU(pI, pJ, octreeNodes, triIndices, triangles, i, j)) continue;

        Vec3 v = pJ - pI;
        float distSqr = v.squaredNorm();
        if (distSqr < VAL_EPSILON) continue;

        float cI = cosPhiGPU(v, triI._normal);
        float cJ = cosPhiGPU(-v, triJ._normal);
        if (cI <= 0.0f || cJ <= 0.0f) continue;

        kijVal += (cI * cJ) / (VAL_PI * distSqr);
    }

    kij[idx] = kijVal * VAL_INV_NUM_RAYS;
}

__global__ void simulationStepKernel(
    const float* kij,
    const int* tau,
    const float* areas,
    const float* rho,
    const float* radE,
    float* radB,
    int N, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sumB = 0.0f;
    for (int j = 0; j < N; ++j) {
        if (i == j) continue;

        int tauij = tau[i * N + j];
        if (t < tauij) continue;

        float kijVal = kij[i * N + j];
        if (kijVal <= 0.0f) continue;

        int srcTime = t - tauij;
        float radJ = radB[srcTime * N + j];
        if (radJ <= 0.0f) continue;

        sumB += fminf(kijVal * areas[j], 1.0f) * radJ;
    }

    radB[t * N + i] = rho[i] * sumB + radE[t * N + i];
}

__global__ void computeDistancesKernel(
    const float* radB,
    float* distances,
    int N, int T, int sourceIdx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float maxCorr = 0.0f;
    int bestT = 0;

    for (int t = 0; t < T; ++t) {
        float sum = 0.0f;
        for (int tt = t; tt < T; ++tt) {
            float pB = radB[tt * N + i];
            float pS = radB[(tt - t) * N + sourceIdx];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distances[i] = VAL_WAVE_SPEED * (float)bestT;
}

// ============================================================================
// Triangle-Box Overlap Test (CPU only, for octree construction)
// ============================================================================

static bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
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
    val_t r = boxHalfSize.x * fabsf(triNormal.x) +
              boxHalfSize.y * fabsf(triNormal.y) +
              boxHalfSize.z * fabsf(triNormal.z);
    if (fabsf(d) > r) return false;

    auto testAxis = [&](const Vec3& axis) -> bool {
        val_t p0 = axis.dot(v0);
        val_t p1 = axis.dot(v1);
        val_t p2 = axis.dot(v2);
        val_t rr = boxHalfSize.x * fabsf(axis.x) +
                   boxHalfSize.y * fabsf(axis.y) +
                   boxHalfSize.z * fabsf(axis.z);
        auto [minP, maxP] = minMax3(p0, p1, p2);
        return !(minP > rr || maxP < -rr);
    };

    Vec3 axes[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
    Vec3 edges[3] = {e0, e1, e2};
    for (const auto& axis : axes) {
        for (const auto& edge : edges) {
            Vec3 crossAxis = axis.cross(edge);
            if (crossAxis.squaredNorm() > VAL_EPSILON) {
                if (!testAxis(crossAxis)) return false;
            }
        }
    }
    return true;
}

// ============================================================================
// Octree for Spatial Acceleration (CPU)
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
                minBound.x = fminf(minBound.x, v->x);
                minBound.y = fminf(minBound.y, v->y);
                minBound.z = fminf(minBound.z, v->z);
                maxBound.x = fmaxf(maxBound.x, v->x);
                maxBound.y = fmaxf(maxBound.y, v->y);
                maxBound.z = fmaxf(maxBound.z, v->z);
            }
        }
        std::vector<size_t> allIndices(triangles.size());
        for (size_t i = 0; i < triangles.size(); ++i) allIndices[i] = i;
        buildNode(allIndices, minBound, maxBound);
    }

    // Flatten octree into node array for GPU transfer
    // Returns index of this node in the flat array
    int flatten(std::vector<GPUOctreeNode>& flatNodes,
                std::vector<int>& flatTriIndices) const {
        int idx = (int)flatNodes.size();
        GPUOctreeNode node;
        node.cx = center.x; node.cy = center.y; node.cz = center.z;
        node.hx = halfExtent.x; node.hy = halfExtent.y; node.hz = halfExtent.z;

        if (!triangleIndices.empty()) {
            node.triStart = (int)flatTriIndices.size();
            node.triCount = (int)triangleIndices.size();
            for (int i = 0; i < 8; ++i) node.children[i] = -1;
            for (size_t ti : triangleIndices) flatTriIndices.push_back((int)ti);
        } else {
            node.triStart = -1;
            node.triCount = 0;
            for (int i = 0; i < 8; ++i) {
                if (children[i]) {
                    node.children[i] = children[i]->flatten(flatNodes, flatTriIndices);
                } else {
                    node.children[i] = -1;
                }
            }
        }
        flatNodes.push_back(node);
        return idx;
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
        if (!canSplit) { triangleIndices = indices; return; }
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
// Mesh Generation: Icosphere (CPU)
// ============================================================================

struct Face3 { idx_t v[3]; };

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
        Face3 initFaces[20] = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };
        std::vector<Face3> faces(initFaces, initFaces + 20);
        for (int i = 0; i < subdivisions; ++i) {
            std::vector<Face3> newFaces;
            std::map<std::pair<idx_t, idx_t>, idx_t> midpointCache;
            auto getMidpoint = [&](idx_t i1, idx_t i2) -> idx_t {
                auto key = std::make_pair(std::min(i1, i2), std::max(i1, i2));
                auto it = midpointCache.find(key);
                if (it != midpointCache.end()) return it->second;
                Vec3 mid = (vertices[i1] + vertices[i2]) / 2.0f;
                mid = mid.normalized() * radius;
                idx_t id = static_cast<idx_t>(vertices.size());
                vertices.push_back(mid);
                midpointCache[key] = id;
                return id;
            };
            for (const auto& face : faces) {
                idx_t a = getMidpoint(face.v[0], face.v[1]);
                idx_t b = getMidpoint(face.v[1], face.v[2]);
                idx_t c = getMidpoint(face.v[2], face.v[0]);
                Face3 f1 = {face.v[0], a, c};
                Face3 f2 = {face.v[1], b, a};
                Face3 f3 = {face.v[2], c, b};
                Face3 f4 = {a, b, c};
                newFaces.push_back(f1);
                newFaces.push_back(f2);
                newFaces.push_back(f3);
                newFaces.push_back(f4);
            }
            faces = std::move(newFaces);
        }
        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            triangles.emplace_back(vertices[face.v[2]], vertices[face.v[1]], vertices[face.v[0]]);
        }
    }
};

// ============================================================================
// Simulation State (with GPU pointers)
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    // CPU data
    std::vector<Triangle> triangles;
    std::vector<val_t> areas;
    std::vector<val_t> rho;
    std::vector<val_t> kij;
    std::vector<int> tau;
    std::vector<val_t> radE;
    std::vector<val_t> radB;
    std::vector<val_t> distances;

    Octree octree;

    size_t sourceIndex;

    // GPU data
    Triangle* d_triangles;
    val_t* d_areas;
    val_t* d_rho;
    val_t* d_kij;
    int* d_tau;
    val_t* d_radE;
    val_t* d_radB;
    val_t* d_distances;
    GPUOctreeNode* d_octreeNodes;
    int* d_triIndices;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// GPU Memory Management
// ============================================================================

static void allocateGPUMemory(SimulationState& state,
                               const std::vector<GPUOctreeNode>& flatNodes,
                               const std::vector<int>& flatTriIndices) {
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

    CUDA_CHECK(cudaMemcpy(state.d_triangles, state.triangles.data(),
                          N * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(),
                          N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(),
                          N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_radE, state.radE.data(),
                          T * N * sizeof(val_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(state.d_kij, 0, N * N * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.d_tau, 0, N * N * sizeof(int)));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, T * N * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.d_distances, 0, N * sizeof(val_t)));

    CUDA_CHECK(cudaMalloc(&state.d_octreeNodes, flatNodes.size() * sizeof(GPUOctreeNode)));
    CUDA_CHECK(cudaMemcpy(state.d_octreeNodes, flatNodes.data(),
                          flatNodes.size() * sizeof(GPUOctreeNode), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_triIndices, flatTriIndices.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(state.d_triIndices, flatTriIndices.data(),
                          flatTriIndices.size() * sizeof(int), cudaMemcpyHostToDevice));
}

static void freeGPUMemory(SimulationState& state) {
    CUDA_CHECK(cudaFree(state.d_triangles));
    CUDA_CHECK(cudaFree(state.d_areas));
    CUDA_CHECK(cudaFree(state.d_rho));
    CUDA_CHECK(cudaFree(state.d_kij));
    CUDA_CHECK(cudaFree(state.d_tau));
    CUDA_CHECK(cudaFree(state.d_radE));
    CUDA_CHECK(cudaFree(state.d_radB));
    CUDA_CHECK(cudaFree(state.d_distances));
    CUDA_CHECK(cudaFree(state.d_octreeNodes));
    CUDA_CHECK(cudaFree(state.d_triIndices));
}

// ============================================================================
// Initialization (CPU mesh + octree, GPU memory allocation)
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

    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    state.rho.resize(state.numTriangles, reflectivity);

    state.kij.resize(state.numTriangles * state.numTriangles, 0.0f);
    state.tau.resize(state.numTriangles * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, 0.0f);
    state.radB.resize(timesteps * state.numTriangles, 0.0f);
    state.distances.resize(state.numTriangles, 0.0f);

    size_t timeOff = timesteps / 2;
    for (size_t t = 0; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Flatten octree for GPU
    std::vector<GPUOctreeNode> flatNodes;
    std::vector<int> flatTriIndices;
    state.octree.flatten(flatNodes, flatTriIndices);
    printf("  Octree nodes: %zu, triangle indices: %zu\n", flatNodes.size(), flatTriIndices.size());

    // Allocate and copy to GPU
    allocateGPUMemory(state, flatNodes, flatTriIndices);
    printf("  GPU memory allocated\n");
}

// ============================================================================
// GPU Launcher Functions
// ============================================================================

void computeTimeDelaysGPU(SimulationState& state) {
    int N = (int)state.numTriangles;
    long long totalThreads = (long long)N * N;
    int blockSize = 256;
    int gridSize = (int)((totalThreads + blockSize - 1) / blockSize);
    computeTimeDelaysKernel<<<gridSize, blockSize>>>(state.d_triangles, state.d_tau, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back for CPU use
    CUDA_CHECK(cudaMemcpy(state.tau.data(), state.d_tau,
                          (size_t)N * N * sizeof(int), cudaMemcpyDeviceToHost));
}

void computeFormFactorsGPU(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    int N = (int)state.numTriangles;
    long long totalThreads = (long long)N * N;
    int blockSize = 256;
    int gridSize = (int)((totalThreads + blockSize - 1) / blockSize);

    computeFormFactorsKernel<<<gridSize, blockSize>>>(
        state.d_triangles, state.d_octreeNodes, state.d_triIndices,
        state.d_kij, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("  Form factor computation complete\n");
}

void runSimulationGPU(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    int N = (int)state.numTriangles;
    int T = (int)state.numTimesteps;
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    for (int t = 0; t < T; ++t) {
        simulationStepKernel<<<gridSize, blockSize>>>(
            state.d_kij, state.d_tau, state.d_areas, state.d_rho,
            state.d_radE, state.d_radB, N, t);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeDistancesGPU(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    int N = (int)state.numTriangles;
    int T = (int)state.numTimesteps;
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    computeDistancesKernel<<<gridSize, blockSize>>>(
        state.d_radB, state.d_distances, N, T, (int)state.sourceIndex);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");

    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = 0.0f;
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
        minDist = fminf(minDist, d);
        maxDist = fmaxf(maxDist, d);
        sumDist += d;
        if (d > VAL_EPSILON) nonZeroCount++;
    }

    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > VAL_WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    int receivedEnergy = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > VAL_EPSILON) {
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
        if (state.kij[i] > VAL_EPSILON) nonZeroKij++;
    }
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           nonZeroKij, state.numTriangles * state.numTriangles,
           100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));
    if (nonZeroKij == 0) {
        printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) return false;
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

    // Select fastest GPU device
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable device found\n");
        return 1;
    }
    int bestDevice = 0;
    int bestSMs = 0;
    for (int d = 0; d < deviceCount; ++d) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, d));
        if (prop.multiProcessorCount > bestSMs) {
            bestSMs = prop.multiProcessorCount;
            bestDevice = d;
        }
    }
    CUDA_CHECK(cudaSetDevice(bestDevice));
    cudaDeviceProp devProp;
    CUDA_CHECK(cudaGetDeviceProperties(&devProp, bestDevice));
    printf("Using GPU: %s (compute %d.%d, %d SMs, %lld MB VRAM)\n",
           devProp.name, devProp.major, devProp.minor,
           devProp.multiProcessorCount,
           (long long)(devProp.totalGlobalMem / (1024 * 1024)));

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    printf("Room Response Simulation Benchmark\n");
    printf("===================================\n");
    printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");

    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);
    printf("\n");

    // Precomputation phase
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelaysGPU(state);
    computeFormFactorsGPU(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();
    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation phase
    auto startSim = std::chrono::high_resolution_clock::now();
    runSimulationGPU(state);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();
    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation phase
    auto startDist = std::chrono::high_resolution_clock::now();
    computeDistancesGPU(state);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();
    printf("Distance computation time: %ld ms\n", distDuration);
    printf("\n");

    // Copy results back from GPU
    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.d_distances,
                          N * sizeof(val_t), cudaMemcpyDeviceToHost));
    if (validate || printResults) {
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB,
                              T * N * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.kij.data(), state.d_kij,
                              N * N * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

    long totalTime = preDuration + simDuration + distDuration;
    printf("Total computation time: %ld ms\n", totalTime);

    double kijOps = static_cast<double>(N * N);
    double simOps = static_cast<double>(N * N * T);
    double distOps = static_cast<double>(N * T * T);

    printf("\nPerformance:\n");
    printf("  Triangles: %zu\n", N);
    printf("  Timesteps: %zu\n", T);
    printf("  Form factor computations: %.2e\n", kijOps);
    printf("  Simulation operations: %.2e\n", simOps);
    printf("  Distance computations: %.2e\n", distOps);
    printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / N);

    size_t memKij = N * N * sizeof(val_t);
    size_t memTau = N * N * sizeof(int);
    size_t memRad = 2 * T * N * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    uint64_t hash = computeHash(state);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");

    if (printResults) {
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    if (validate) {
        if (!validateResults(state)) {
            freeGPUMemory(state);
            return 1;
        }
    }

    freeGPUMemory(state);
    return 0;
}
