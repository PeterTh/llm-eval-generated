/**
 * Room Response Simulation Benchmark
 *
 * Hybrid MPI + OpenMP + CUDA implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization strategy:
 *  - MPI partitions the triangle set (receiver rows) across ranks/nodes, each
 *    rank driving one GPU.
 *  - CUDA kernels perform the O(N^2), O(N^2*T) and O(N*T^2) numerical kernels
 *    (form factors, wave propagation, cross-correlation) on the GPU owned by
 *    each rank.
 *  - OpenMP parallelizes the remaining host-side CPU work (mesh generation,
 *    octree bounding-box reduction, area computation, validation/hash
 *    reductions).
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
#include <map>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// ============================================================================
// Error-checking helpers
// ============================================================================

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
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
// Vector and Triangle Types (usable from both host and device code)
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

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction, host-side build only)
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
// Flattened Octree representation (uploaded to and traversed on the GPU)
// ============================================================================

struct FlatOctreeNode {
    Vec3 center, halfExtent;
    int child[8];
    int triStart;  // offset into flat triangle-index array, -1 if internal node
    int triCount;
};

// ============================================================================
// Octree for Spatial Acceleration (built on host, flattened for GPU use)
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

    // Flatten the pointer-based octree into arrays suitable for GPU upload.
    // Returns the index of the root node in `nodes`.
    int flatten(std::vector<FlatOctreeNode>& nodes, std::vector<uint32_t>& triIdx) const {
        return flattenRec(nodes, triIdx);
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

    int flattenRec(std::vector<FlatOctreeNode>& nodes, std::vector<uint32_t>& triIdx) const {
        int childIdx[8];
        for (int i = 0; i < 8; ++i) {
            childIdx[i] = children[i] ? children[i]->flattenRec(nodes, triIdx) : -1;
        }

        FlatOctreeNode node;
        node.center = center;
        node.halfExtent = halfExtent;
        for (int i = 0; i < 8; ++i) node.child[i] = childIdx[i];

        if (!triangleIndices.empty()) {
            node.triStart = static_cast<int>(triIdx.size());
            node.triCount = static_cast<int>(triangleIndices.size());
            triIdx.insert(triIdx.end(), triangleIndices.begin(), triangleIndices.end());
        } else {
            node.triStart = -1;
            node.triCount = 0;
        }

        nodes.push_back(node);
        return static_cast<int>(nodes.size()) - 1;
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

        // Subdivide (inherently sequential: shared vertex midpoint cache)
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

        // Build triangles (normals pointing inward for a "room").
        // Independent per-face, so parallelize with OpenMP.
        triangles.resize(faces.size());
        #pragma omp parallel for schedule(static)
        for (size_t f = 0; f < faces.size(); ++f) {
            const auto& face = faces[f];
            // Reverse winding to make normals point inward
            triangles[f] = Triangle(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

__host__ __device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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
// Form Factor Support
// ============================================================================

__host__ __device__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

__host__ __device__ int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// ============================================================================
// GPU Octree Traversal (flattened, iterative)
// ============================================================================

__device__ bool rayIntersectsBoxFlat(const FlatOctreeNode& node, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > node.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > node.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > node.halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

__device__ bool isRayBlockedDevice(const Vec3& from, const Vec3& to,
                                    const FlatOctreeNode* nodes, const uint32_t* triIdx,
                                    const Triangle* tris, int rootIdx,
                                    uint32_t srcTriIdx, uint32_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    constexpr int MAX_STACK = 128;
    int stack[MAX_STACK];
    int sp = 0;
    stack[sp++] = rootIdx;

    while (sp > 0) {
        int ni = stack[--sp];
        const FlatOctreeNode& node = nodes[ni];
        if (!rayIntersectsBoxFlat(node, from, to)) continue;

        if (node.triCount > 0) {
            for (int k = 0; k < node.triCount; ++k) {
                uint32_t idx = triIdx[node.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const Triangle& tri = tris[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                if (node.child[c] >= 0 && sp < MAX_STACK) stack[sp++] = node.child[c];
            }
        }
    }
    return false;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// Form factor Kij between owned rows [rowStart, rowStart+rowCount) and all N columns.
__global__ void computeKijKernel(const Triangle* tris, const FlatOctreeNode* nodes,
                                  const uint32_t* triIdx, int rootIdx,
                                  size_t numTriangles, size_t rowStart, size_t rowCount,
                                  val_t* kijOut, uint64_t seed) {
    size_t li = blockIdx.x;
    size_t j = static_cast<size_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (li >= rowCount || j >= numTriangles) return;

    size_t i = rowStart + li;
    size_t outIdx = li * numTriangles + j;

    if (i == j) {
        kijOut[outIdx] = ZERO;
        return;
    }

    const Triangle& triI = tris[i];
    const Triangle& triJ = tris[j];

    if (triI.normal().dot(triJ.normal()) > 0.99f) {
        kijOut[outIdx] = ZERO;
        return;
    }

    curandStatePhilox4_32_10_t rngState;
    curand_init(seed, static_cast<uint64_t>(i) * numTriangles + j, 0, &rngState);

    val_t kij = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        val_t u1 = curand_uniform(&rngState);
        val_t v1 = curand_uniform(&rngState);
        if (u1 + v1 > 1.0f) { u1 = 1.0f - u1; v1 = 1.0f - v1; }
        Vec3 pI = triI.a + (triI.b - triI.a) * u1 + (triI.c - triI.a) * v1;

        val_t u2 = curand_uniform(&rngState);
        val_t v2 = curand_uniform(&rngState);
        if (u2 + v2 > 1.0f) { u2 = 1.0f - u2; v2 = 1.0f - v2; }
        Vec3 pJ = triJ.a + (triJ.b - triJ.a) * u2 + (triJ.c - triJ.a) * v2;

        if (isRayBlockedDevice(pI, pJ, nodes, triIdx, tris, rootIdx,
                                static_cast<uint32_t>(i), static_cast<uint32_t>(j))) {
            continue;
        }

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.normal());
        val_t cosPhiJ = cosPhi(-v, triJ.normal());
        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kijOut[outIdx] = kij * INV_NUM_RAYS;
}

__global__ void computeTauKernel(const Triangle* tris, size_t numTriangles,
                                  size_t rowStart, size_t rowCount, int* tauOut) {
    size_t li = blockIdx.x;
    size_t j = static_cast<size_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (li >= rowCount || j >= numTriangles) return;

    size_t i = rowStart + li;
    size_t outIdx = li * numTriangles + j;

    if (i == j) {
        tauOut[outIdx] = 0;
        return;
    }
    tauOut[outIdx] = computeTau(tris[i], tris[j]);
}

// One CUDA block per owned row; threads cooperatively reduce over all j.
__global__ void simStepKernel(const val_t* kijLocal, const int* tauLocal, const val_t* areas,
                               const val_t* rho, const val_t* radE, const val_t* radBFull,
                               val_t* radBLocalOut, size_t numTriangles, size_t rowStart,
                               size_t rowCount, size_t t) {
    extern __shared__ val_t sdata[];
    size_t li = blockIdx.x;
    if (li >= rowCount) return;
    size_t i = rowStart + li;

    val_t partial = ZERO;
    for (size_t j = threadIdx.x; j < numTriangles; j += blockDim.x) {
        if (j == i) continue;

        int tauij = tauLocal[li * numTriangles + j];
        if (static_cast<long long>(t) < tauij) continue;

        val_t kij = kijLocal[li * numTriangles + j];
        if (kij <= ZERO) continue;

        size_t srcTime = t - static_cast<size_t>(tauij);
        val_t radJ = radBFull[srcTime * numTriangles + j];
        if (radJ <= ZERO) continue;

        partial += fminf(kij * areas[j], ONE) * radJ;
    }

    sdata[threadIdx.x] = partial;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        radBLocalOut[li] = rho[i] * sdata[0] + radE[t * numTriangles + i];
    }
}

// One CUDA block per owned row; threads cooperatively scan over candidate delays t.
__global__ void distanceKernel(const val_t* radBFull, size_t numTriangles, size_t numTimesteps,
                                size_t rowStart, size_t rowCount, size_t sourceIndex,
                                val_t* distLocalOut) {
    extern __shared__ unsigned char sharedRaw[];
    val_t* shCorr = reinterpret_cast<val_t*>(sharedRaw);
    int* shBest = reinterpret_cast<int*>(shCorr + blockDim.x);

    size_t li = blockIdx.x;
    if (li >= rowCount) return;
    size_t i = rowStart + li;

    val_t bestCorr = ZERO;
    int bestT = 0;

    for (size_t t = threadIdx.x; t < numTimesteps; t += blockDim.x) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < numTimesteps; ++tt) {
            val_t pB = radBFull[tt * numTriangles + i];
            val_t pS = radBFull[(tt - t) * numTriangles + sourceIndex];
            sum += pS * pB;
        }
        if (sum > bestCorr) {
            bestCorr = sum;
            bestT = static_cast<int>(t);
        }
    }

    shCorr[threadIdx.x] = bestCorr;
    shBest[threadIdx.x] = bestT;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shCorr[threadIdx.x + s] > shCorr[threadIdx.x]) {
                shCorr[threadIdx.x] = shCorr[threadIdx.x + s];
                shBest[threadIdx.x] = shBest[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        distLocalOut[li] = WAVE_SPEED * static_cast<val_t>(shBest[0]);
    }
}

__global__ void countNonZeroKernel(const val_t* data, size_t n, unsigned long long* counter) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned long long local = 0;
    for (size_t i = idx; i < n; i += stride) {
        if (data[i] > EPSILON) local++;
    }
    if (local) atomicAdd(counter, local);
}

__global__ void receivedEnergyKernel(const val_t* radB, size_t numTriangles, size_t numTimesteps,
                                      int* flags) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numTriangles) return;
    int got = 0;
    for (size_t t = 0; t < numTimesteps; ++t) {
        if (radB[t * numTriangles + i] > EPSILON) { got = 1; break; }
    }
    flags[i] = got;
}

// ============================================================================
// MPI row partitioning
// ============================================================================

static void computeRange(size_t n, int rank, int size, size_t& start, size_t& count) {
    size_t base = n / static_cast<size_t>(size);
    size_t rem = n % static_cast<size_t>(size);
    if (static_cast<size_t>(rank) < rem) {
        count = base + 1;
        start = static_cast<size_t>(rank) * (base + 1);
    } else {
        count = base;
        start = rem * (base + 1) + (static_cast<size_t>(rank) - rem) * base;
    }
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix), replicated
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix), replicated
    std::vector<val_t> distances;   // Computed distances from source (full, rank 0 only valid)

    size_t sourceIndex = 0;

    // MPI partitioning: this rank owns rows [rowStart, rowStart+rowCount)
    int mpiRank = 0;
    int mpiSize = 1;
    size_t rowStart = 0;
    size_t rowCount = 0;
    std::vector<int> recvCounts;  // per-rank row counts (for Allgatherv)
    std::vector<int> displs;      // per-rank row displacements

    // Device buffers
    Triangle* d_triangles = nullptr;
    FlatOctreeNode* d_octreeNodes = nullptr;
    uint32_t* d_octreeTriIdx = nullptr;
    int octreeRoot = 0;

    val_t* d_areas = nullptr;
    val_t* d_rho = nullptr;
    val_t* d_radE = nullptr;
    val_t* d_radB = nullptr;  // full T x N, replicated on every rank's GPU

    val_t* d_kijLocal = nullptr;  // rowCount x N
    int* d_tauLocal = nullptr;    // rowCount x N

    val_t* d_rowScratch = nullptr;   // rowCount (per-timestep local radB output)
    val_t* d_distLocal = nullptr;    // rowCount

    size_t idx2dLocal(size_t li, size_t j) const { return li * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    MPI_Comm_rank(MPI_COMM_WORLD, &state.mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &state.mpiSize);

    // Assign one GPU per rank (round-robin over local node's visible devices)
    MPI_Comm shmComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, state.mpiRank, MPI_INFO_NULL, &shmComm);
    int localRank = 0;
    MPI_Comm_rank(shmComm, &localRank);
    MPI_Comm_free(&shmComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        fprintf(stderr, "No CUDA devices visible to rank %d\n", state.mpiRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // Generate mesh on rank 0, broadcast to all ranks
    if (state.mpiRank == 0) {
        IcosphereMesh mesh(subdivisions, 10.0f);
        state.triangles = std::move(mesh.triangles);
    }

    uint64_t numTri64 = state.mpiRank == 0 ? static_cast<uint64_t>(state.triangles.size()) : 0;
    MPI_Bcast(&numTri64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    state.numTriangles = static_cast<size_t>(numTri64);

    if (state.mpiRank != 0) state.triangles.resize(state.numTriangles);
    MPI_Bcast(state.triangles.data(), static_cast<int>(state.numTriangles * sizeof(Triangle)),
              MPI_BYTE, 0, MPI_COMM_WORLD);

    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    if (state.mpiRank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("Building octree...\n");
    }

    // Build octree on rank 0, flatten, and broadcast to all ranks
    std::vector<FlatOctreeNode> flatNodes;
    std::vector<uint32_t> flatTriIdx;
    int rootIdx = 0;
    if (state.mpiRank == 0) {
        Octree octree;
        octree.build(state.triangles);
        rootIdx = octree.flatten(flatNodes, flatTriIdx);
    }

    MPI_Bcast(&rootIdx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    uint64_t numNodes64 = state.mpiRank == 0 ? static_cast<uint64_t>(flatNodes.size()) : 0;
    uint64_t numTriIdx64 = state.mpiRank == 0 ? static_cast<uint64_t>(flatTriIdx.size()) : 0;
    MPI_Bcast(&numNodes64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numTriIdx64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    if (state.mpiRank != 0) {
        flatNodes.resize(static_cast<size_t>(numNodes64));
        flatTriIdx.resize(static_cast<size_t>(numTriIdx64));
    }
    MPI_Bcast(flatNodes.data(), static_cast<int>(flatNodes.size() * sizeof(FlatOctreeNode)),
              MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(flatTriIdx.data(), static_cast<int>(flatTriIdx.size() * sizeof(uint32_t)),
              MPI_BYTE, 0, MPI_COMM_WORLD);
    state.octreeRoot = rootIdx;

    // Areas (independent per triangle -> OpenMP)
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    state.rho.assign(state.numTriangles, reflectivity);

    // Row partition for this rank
    computeRange(state.numTriangles, state.mpiRank, state.mpiSize, state.rowStart, state.rowCount);

    state.recvCounts.resize(state.mpiSize);
    state.displs.resize(state.mpiSize);
    for (int r = 0; r < state.mpiSize; ++r) {
        size_t rs, rc;
        computeRange(state.numTriangles, r, state.mpiSize, rs, rc);
        state.recvCounts[r] = static_cast<int>(rc);
        state.displs[r] = static_cast<int>(rs);
    }

    state.radE.assign(timesteps * state.numTriangles, ZERO);
    state.radB.assign(timesteps * state.numTriangles, ZERO);
    state.distances.assign(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps) - identical on all ranks
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Upload static data to the GPU
    size_t N = state.numTriangles;
    CUDA_CHECK(cudaMalloc(&state.d_triangles, N * sizeof(Triangle)));
    CUDA_CHECK(cudaMemcpy(state.d_triangles, state.triangles.data(), N * sizeof(Triangle),
                           cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_octreeNodes, flatNodes.size() * sizeof(FlatOctreeNode)));
    CUDA_CHECK(cudaMemcpy(state.d_octreeNodes, flatNodes.data(),
                           flatNodes.size() * sizeof(FlatOctreeNode), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_octreeTriIdx, flatTriIdx.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(state.d_octreeTriIdx, flatTriIdx.data(),
                           flatTriIdx.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_areas, N * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), N * sizeof(val_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_rho, N * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), N * sizeof(val_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_radE, timesteps * N * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(state.d_radE, state.radE.data(), timesteps * N * sizeof(val_t),
                           cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_radB, timesteps * N * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, timesteps * N * sizeof(val_t)));

    if (state.rowCount > 0) {
        CUDA_CHECK(cudaMalloc(&state.d_kijLocal, state.rowCount * N * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&state.d_tauLocal, state.rowCount * N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&state.d_rowScratch, state.rowCount * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&state.d_distLocal, state.rowCount * sizeof(val_t)));
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing time delays (Tau)...\n");
    if (state.rowCount == 0) return;

    dim3 block(256);
    dim3 grid(static_cast<unsigned>(state.rowCount),
              static_cast<unsigned>((state.numTriangles + block.x - 1) / block.x));

    computeTauKernel<<<grid, block>>>(state.d_triangles, state.numTriangles,
                                       state.rowStart, state.rowCount, state.d_tauLocal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeFormFactors(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing form factors (Kij)...\n");
    if (state.rowCount == 0) return;

    dim3 block(256);
    dim3 grid(static_cast<unsigned>(state.rowCount),
              static_cast<unsigned>((state.numTriangles + block.x - 1) / block.x));

    computeKijKernel<<<grid, block>>>(state.d_triangles, state.d_octreeNodes, state.d_octreeTriIdx,
                                       state.octreeRoot, state.numTriangles, state.rowStart,
                                       state.rowCount, state.d_kijLocal, /*seed=*/42ULL);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.mpiRank == 0) printf("Running wave propagation simulation...\n");

    size_t N = state.numTriangles;
    std::vector<val_t> hostLocalRow(state.rowCount);
    std::vector<val_t> hostFullRow(N);

    const int block = 256;
    size_t shmem = block * sizeof(val_t);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (state.rowCount > 0) {
            dim3 grid(static_cast<unsigned>(state.rowCount));
            simStepKernel<<<grid, block, shmem>>>(state.d_kijLocal, state.d_tauLocal, state.d_areas,
                                                   state.d_rho, state.d_radE, state.d_radB,
                                                   state.d_rowScratch, N, state.rowStart,
                                                   state.rowCount, t);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(hostLocalRow.data(), state.d_rowScratch,
                                   state.rowCount * sizeof(val_t), cudaMemcpyDeviceToHost));
        }

        MPI_Allgatherv(state.rowCount > 0 ? hostLocalRow.data() : nullptr,
                        static_cast<int>(state.rowCount), MPI_FLOAT,
                        hostFullRow.data(), state.recvCounts.data(), state.displs.data(),
                        MPI_FLOAT, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(state.d_radB + t * N, hostFullRow.data(), N * sizeof(val_t),
                               cudaMemcpyHostToDevice));
        std::copy(hostFullRow.begin(), hostFullRow.end(), state.radB.begin() + t * N);

        if (state.mpiRank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing distances via cross-correlation...\n");

    size_t N = state.numTriangles;
    if (state.rowCount > 0) {
        const int block = static_cast<int>(std::min<size_t>(256, state.numTimesteps > 0 ? state.numTimesteps : 1));
        size_t shmem = block * (sizeof(val_t) + sizeof(int));
        dim3 grid(static_cast<unsigned>(state.rowCount));

        distanceKernel<<<grid, block, shmem>>>(state.d_radB, N, state.numTimesteps, state.rowStart,
                                                state.rowCount, state.sourceIndex, state.d_distLocal);
        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<val_t> hostLocalDist(state.rowCount);
    if (state.rowCount > 0) {
        CUDA_CHECK(cudaMemcpy(hostLocalDist.data(), state.d_distLocal, state.rowCount * sizeof(val_t),
                               cudaMemcpyDeviceToHost));
    }

    MPI_Allgatherv(state.rowCount > 0 ? hostLocalDist.data() : nullptr,
                    static_cast<int>(state.rowCount), MPI_FLOAT,
                    state.distances.data(), state.recvCounts.data(), state.displs.data(),
                    MPI_FLOAT, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(SimulationState& state) {
    // distances is fully replicated (Allgatherv); nonZeroKij must be reduced across ranks.
    unsigned long long localNonZeroKij = 0;
    if (state.rowCount > 0) {
        unsigned long long* d_counter = nullptr;
        CUDA_CHECK(cudaMalloc(&d_counter, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(unsigned long long)));
        size_t total = state.rowCount * state.numTriangles;
        int block = 256;
        int grid = static_cast<int>(std::min<size_t>(4096, (total + block - 1) / block));
        countNonZeroKernel<<<grid, block>>>(state.d_kijLocal, total, d_counter);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&localNonZeroKij, d_counter, sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_counter));
    }

    unsigned long long globalNonZeroKij = 0;
    MPI_Reduce(&localNonZeroKij, &globalNonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (state.mpiRank != 0) return true;  // only rank 0 reports/validates

    printf("\nValidation:\n");

    bool allNonNegative = true;
    val_t minDist = FLT_MAX;
    val_t maxDist = -FLT_MAX;
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    #pragma omp parallel for reduction(min:minDist) reduction(max:maxDist) reduction(+:sumDist) \
                              reduction(+:nonZeroCount)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t d = state.distances[i];
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

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
    }

    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation using rank 0's own replicated d_radB copy
    size_t N = state.numTriangles;
    int* d_flags = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flags, N * sizeof(int)));
    int block = 256;
    int grid = static_cast<int>((N + block - 1) / block);
    receivedEnergyKernel<<<grid, block>>>(state.d_radB, N, state.numTimesteps, d_flags);
    CUDA_CHECK(cudaGetLastError());
    std::vector<int> flags(N);
    CUDA_CHECK(cudaMemcpy(flags.data(), d_flags, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_flags));

    int receivedEnergy = 0;
    #pragma omp parallel for reduction(+:receivedEnergy)
    for (size_t i = 0; i < N; ++i) receivedEnergy += flags[i];

    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n", globalNonZeroKij,
           state.numTriangles * state.numTriangles,
           100.0 * static_cast<double>(globalNonZeroKij) /
               static_cast<double>(state.numTriangles * state.numTriangles));

    if (globalNonZeroKij == 0) {
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
    #pragma omp parallel for reduction(^:hash)
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
// Cleanup
// ============================================================================

void freeDeviceBuffers(SimulationState& state) {
    cudaFree(state.d_triangles);
    cudaFree(state.d_octreeNodes);
    cudaFree(state.d_octreeTriIdx);
    cudaFree(state.d_areas);
    cudaFree(state.d_rho);
    cudaFree(state.d_radE);
    cudaFree(state.d_radB);
    cudaFree(state.d_kijLocal);
    cudaFree(state.d_tauLocal);
    cudaFree(state.d_rowScratch);
    cudaFree(state.d_distLocal);
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
    MPI_Init(&argc, &argv);
    int mpiRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (argv identical across ranks under mpirun)
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
            if (mpiRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (mpiRank == 0) {
        int mpiSize = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", mpiSize, omp_get_max_threads());
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (mpiRank == 0) printf("\n");

    MPI_Barrier(MPI_COMM_WORLD);

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (mpiRank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (mpiRank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    int result = 0;

    if (mpiRank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");

        long totalTime = preDuration + simDuration + distDuration;
        printf("Total computation time: %ld ms\n", totalTime);

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

        size_t memKij = n * n * sizeof(val_t);
        size_t memTau = n * n * sizeof(int);
        size_t memRad = 2 * t * n * sizeof(val_t);
        size_t totalMem = memKij + memTau + memRad;
        printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

        uint64_t hash = computeHash(state);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }
    }

    if (validate) {
        bool ok = validateResults(state);
        if (mpiRank == 0 && !ok) result = 1;
    }

    freeDeviceBuffers(state);
    MPI_Finalize();
    return result;
}
