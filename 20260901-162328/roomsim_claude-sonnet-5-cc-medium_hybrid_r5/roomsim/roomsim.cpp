/**
 * Room Response Simulation Benchmark
 *
 * This is a hybrid MPI + OpenMP + CUDA implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization strategy
 * -------------------------
 * - MPI splits the N triangles into contiguous row blocks across ranks. Each
 *   rank is responsible for computing Kij/Tau rows and radiosity updates for
 *   its own block of triangles, and owns one GPU (rank % numGPUs).
 * - CUDA performs the heavy, embarrassingly-parallel numerical kernels: form
 *   factor (visibility + Monte-Carlo integration against a flattened octree),
 *   time-delay computation, the per-timestep radiosity update, and the
 *   cross-correlation distance search.
 * - OpenMP parallelizes host-side preprocessing that runs identically and
 *   redundantly on every rank (mesh generation, octree construction support
 *   loops, and octree flattening) since this is deterministic and cheap
 *   relative to the GPU kernels, avoiding a broadcast of the mesh.
 * - Per-timestep radiosity results are exchanged with MPI_Allgatherv so every
 *   rank has a consistent, fully up to date radiosity history before moving
 *   to the next timestep (a true sequential dependency of the algorithm).
 *
 * This file is compiled as CUDA source (see CMakeLists.txt) so that host and
 * device code can live side by side without introducing additional files.
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

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                              \
    } while (0)

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
    __host__ __device__ val_t norm() const { return std::sqrt(squaredNorm()); }
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
// Triangle-Box Overlap Test (for octree construction, host-only)
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

        // Compute bounding box (parallel reduction over all triangle vertices)
        Vec3 mn = triangles[0].a;
        Vec3 mx = triangles[0].a;
        #pragma omp parallel
        {
            Vec3 lmn = mn, lmx = mx;
            #pragma omp for nowait
            for (size_t k = 0; k < triangles.size(); ++k) {
                const Triangle& tri = triangles[k];
                for (const auto* v : {&tri.a, &tri.b, &tri.c}) {
                    lmn.x = std::min(lmn.x, v->x); lmx.x = std::max(lmx.x, v->x);
                    lmn.y = std::min(lmn.y, v->y); lmx.y = std::max(lmx.y, v->y);
                    lmn.z = std::min(lmn.z, v->z); lmx.z = std::max(lmx.z, v->z);
                }
            }
            #pragma omp critical
            {
                mn.x = std::min(mn.x, lmn.x); mx.x = std::max(mx.x, lmx.x);
                mn.y = std::min(mn.y, lmn.y); mx.y = std::max(mx.y, lmx.y);
                mn.z = std::min(mn.z, lmn.z); mx.z = std::max(mx.z, lmx.z);
            }
        }
        minBound = mn;
        maxBound = mx;

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
// Flattened Octree (array-based, GPU-friendly)
// ============================================================================

struct FlatOctreeNode {
    Vec3 center, halfExtent;
    int children[8];
    int triStart;  // offset into the flat triangle-index array, -1 if internal node
    int triCount;  // -1 if internal node
};

static int flattenOctreeNode(const Octree& node, std::vector<FlatOctreeNode>& nodes,
                              std::vector<idx_t>& triIndices) {
    FlatOctreeNode fn{};
    fn.center = node.center;
    fn.halfExtent = node.halfExtent;
    for (int i = 0; i < 8; ++i) fn.children[i] = -1;

    if (!node.triangleIndices.empty()) {
        fn.triStart = static_cast<int>(triIndices.size());
        fn.triCount = static_cast<int>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) triIndices.push_back(static_cast<idx_t>(idx));
    } else {
        fn.triStart = -1;
        fn.triCount = -1;
    }

    int myIndex = static_cast<int>(nodes.size());
    nodes.push_back(fn);

    if (fn.triCount < 0) {
        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) {
                int childIdx = flattenOctreeNode(*node.children[i], nodes, triIndices);
                nodes[myIndex].children[i] = childIdx;
            }
        }
    }
    return myIndex;
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

        // Subdivide (sequential: midpoint cache creates a data dependency
        // within each level, but this is cheap relative to the rest of the
        // pipeline)
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
        // Independent per-face, so parallelize across host threads.
        triangles.resize(faces.size());
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < faces.size(); ++k) {
            const auto& face = faces[k];
            // Reverse winding to make normals point inward
            triangles[k] = Triangle(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Geometry helper functions (shared between host validation and device kernels)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__host__ __device__ inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t c = v.dot(normal) / vNorm;
    return c > ZERO ? c : ZERO;
}

// Ray-Triangle intersection (Möller-Trumbore algorithm)
__host__ __device__ inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                                        const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return std::numeric_limits<val_t>::max();

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return std::numeric_limits<val_t>::max();

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return std::numeric_limits<val_t>::max();

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// CUDA device kernels
// ============================================================================

__device__ inline bool rayIntersectsBoxDevice(const Vec3& p1, const Vec3& p2,
                                               const Vec3& center, const Vec3& halfExtent) {
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

// Iterative (stack-based) traversal of the flattened octree to test ray occlusion
__device__ inline bool isRayBlockedDevice(const Vec3& from, const Vec3& to,
                                           idx_t srcIdx, idx_t dstIdx,
                                           const FlatOctreeNode* __restrict__ nodes,
                                           const idx_t* __restrict__ triIdxArr,
                                           const Triangle* __restrict__ tris) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[128];
    int sp = 0;
    stack[sp++] = 0;  // root node index

    while (sp > 0) {
        int ni = stack[--sp];
        const FlatOctreeNode& node = nodes[ni];
        if (!rayIntersectsBoxDevice(from, to, node.center, node.halfExtent)) continue;

        if (node.triCount >= 0) {
            for (int k = 0; k < node.triCount; ++k) {
                idx_t idx = triIdxArr[node.triStart + k];
                if (idx == srcIdx || idx == dstIdx) continue;
                const Triangle& tri = tris[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                if (node.children[c] >= 0) stack[sp++] = node.children[c];
            }
        }
    }
    return false;
}

__device__ inline Vec3 randomPointInTriangleDevice(const Triangle& t, curandStatePhilox4_32_10_t& state) {
    val_t u = curand_uniform(&state);
    val_t v = curand_uniform(&state);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

__device__ val_t computeKijDevice(idx_t idxI, idx_t idxJ,
                                   const Triangle* __restrict__ tris,
                                   const FlatOctreeNode* __restrict__ nodes,
                                   const idx_t* __restrict__ triIdxArr,
                                   curandStatePhilox4_32_10_t& rngState) {
    const Triangle& triI = tris[idxI];
    const Triangle& triJ = tris[idxJ];

    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangleDevice(triI, rngState);
        Vec3 pJ = randomPointInTriangleDevice(triJ, rngState);

        if (isRayBlockedDevice(pI, pJ, idxI, idxJ, nodes, triIdxArr, tris)) continue;

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

// One thread per (i,j) pair in this rank's row block: computes both Kij and Tau.
__global__ void computeFormFactorsAndTauKernel(idx_t rowStart, idx_t rowCount, idx_t N,
                                                const Triangle* __restrict__ tris,
                                                const FlatOctreeNode* __restrict__ nodes,
                                                const idx_t* __restrict__ triIdxArr,
                                                val_t* __restrict__ kijLocal,
                                                int* __restrict__ tauLocal,
                                                uint64_t seed) {
    size_t total = static_cast<size_t>(rowCount) * static_cast<size_t>(N);
    for (size_t linear = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         linear < total; linear += static_cast<size_t>(blockDim.x) * gridDim.x) {
        idx_t iLocal = static_cast<idx_t>(linear / N);
        idx_t j = static_cast<idx_t>(linear % N);
        idx_t i = rowStart + iLocal;

        if (i == j) {
            kijLocal[linear] = ZERO;
            tauLocal[linear] = 0;
            continue;
        }

        val_t dist = (tris[i].center() - tris[j].center()).norm();
        tauLocal[linear] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));

        curandStatePhilox4_32_10_t state;
        curand_init(seed, static_cast<uint64_t>(i) * N + j, 0, &state);
        kijLocal[linear] = computeKijDevice(i, j, tris, nodes, triIdxArr, state);
    }
}

// One block per local row i, threads reduce contributions over j.
__global__ void simulationStepKernel(idx_t rowStart, idx_t rowCount, idx_t N, size_t t,
                                      const val_t* __restrict__ kijLocal,
                                      const int* __restrict__ tauLocal,
                                      const val_t* __restrict__ areas,
                                      const val_t* __restrict__ rho,
                                      const val_t* __restrict__ radE_row_t,
                                      const val_t* __restrict__ radB_full,
                                      val_t* __restrict__ radB_row_local_out) {
    extern __shared__ val_t sdata[];
    idx_t iLocal = blockIdx.x;
    if (iLocal >= rowCount) return;
    idx_t i = rowStart + iLocal;

    val_t partial = ZERO;
    for (idx_t j = threadIdx.x; j < N; j += blockDim.x) {
        if (j == i) continue;

        size_t li = static_cast<size_t>(iLocal) * N + j;
        int tauij = tauLocal[li];
        if (static_cast<long long>(t) < tauij) continue;

        val_t kij = kijLocal[li];
        if (kij <= ZERO) continue;

        size_t srcTime = t - static_cast<size_t>(tauij);
        val_t radJ = radB_full[srcTime * N + j];
        if (radJ <= ZERO) continue;

        val_t contrib = kij * areas[j];
        contrib = contrib < ONE ? contrib : ONE;
        partial += contrib * radJ;
    }

    sdata[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        val_t sumB = sdata[0];
        radB_row_local_out[iLocal] = rho[i] * sumB + radE_row_t[i];
    }
}

// One block per local row i; sequential scan over t (matches original semantics),
// with the inner tt reduction parallelized across threads.
__global__ void computeDistancesKernel(idx_t rowStart, idx_t rowCount, idx_t N, size_t T,
                                        idx_t sourceIndex,
                                        const val_t* __restrict__ radB_full,
                                        val_t* __restrict__ distancesLocalOut) {
    extern __shared__ val_t sdata[];
    idx_t iLocal = blockIdx.x;
    if (iLocal >= rowCount) return;
    idx_t i = rowStart + iLocal;

    val_t maxCorr = ZERO;
    int bestT = 0;

    for (size_t t = 0; t < T; ++t) {
        val_t partial = ZERO;
        for (size_t tt = t + threadIdx.x; tt < T; tt += blockDim.x) {
            val_t pB = radB_full[tt * N + i];
            val_t pS = radB_full[(tt - t) * N + sourceIndex];
            partial += pS * pB;
        }
        sdata[threadIdx.x] = partial;
        __syncthreads();
        for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0 && sdata[0] > maxCorr) {
            maxCorr = sdata[0];
            bestT = static_cast<int>(t);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        distancesLocalOut[iLocal] = WAVE_SPEED * static_cast<val_t>(bestT);
    }
}

__global__ void countNonZeroKernel(const val_t* __restrict__ data, size_t count,
                                    unsigned long long* __restrict__ counter) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    unsigned long long local = 0;
    for (size_t k = idx; k < count; k += stride) {
        if (data[k] > EPSILON) local++;
    }
    if (local > 0) atomicAdd(counter, local);
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
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source (full, gathered on rank 0)

    Octree octree;                  // Spatial acceleration structure (host)
    std::vector<FlatOctreeNode> flatNodes;
    std::vector<idx_t> flatTriIndices;

    size_t sourceIndex = 0;

    // MPI row-block partition of the N triangles owned by this rank.
    size_t rowStart = 0;
    size_t rowCount = 0;
};

// ============================================================================
// MPI row-block partition helpers
// ============================================================================

static void computePartition(size_t N, int rank, int size, size_t& rowStart, size_t& rowCount) {
    size_t base = N / static_cast<size_t>(size);
    size_t rem = N % static_cast<size_t>(size);
    rowStart = static_cast<size_t>(rank) * base + std::min<size_t>(static_cast<size_t>(rank), rem);
    rowCount = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
}

static void computeAllCounts(size_t N, int size, std::vector<int>& counts, std::vector<int>& displs) {
    counts.resize(size);
    displs.resize(size);
    for (int r = 0; r < size; ++r) {
        size_t rs, rc;
        computePartition(N, r, size, rs, rc);
        counts[r] = static_cast<int>(rc);
        displs[r] = static_cast<int>(rs);
    }
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, int rank, int size) {
    // Generate mesh (deterministic, redundantly computed on every rank to avoid
    // a broadcast of the geometry)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    computePartition(state.numTriangles, rank, size, state.rowStart, state.rowCount);

    if (rank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("Building octree...\n");
    }

    // Build octree for spatial acceleration (host), then flatten into
    // array form so it can be uploaded to and traversed on the GPU.
    state.octree.build(state.triangles);
    state.flatNodes.reserve(state.numTriangles * 2);
    state.flatTriIndices.reserve(state.numTriangles);
    flattenOctreeNode(state.octree, state.flatNodes, state.flatTriIndices);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize radiosity matrices and distances
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[t * state.numTriangles + state.sourceIndex] = 1.0f;
    }
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, const std::vector<val_t>& radBFull,
                     unsigned long long nonZeroKij, size_t totalKijEntries) {
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
            if (radBFull[t * state.numTriangles + i] > EPSILON) {
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

    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           nonZeroKij, totalKijEntries,
           100.0f * static_cast<double>(nonZeroKij) / static_cast<double>(totalKijEntries));

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
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int myDevice = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(myDevice));

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical argv on every rank, so all
    // ranks reach the same branch deterministically)
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Hybrid MPI + OpenMP + CUDA (ranks: %d, GPUs visible per rank: %d)\n", size, deviceCount);
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Initialize (mesh + octree, replicated per rank; row-block partition computed)
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity, rank, size);

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const size_t rowStart = state.rowStart;
    const size_t rowCount = state.rowCount;

    std::vector<int> rowCounts, rowDispls;
    computeAllCounts(N, size, rowCounts, rowDispls);

    if (rank == 0) printf("\n");
    MPI_Barrier(MPI_COMM_WORLD);

    // ------------------------------------------------------------------
    // Upload geometry / octree to the GPU (replicated across ranks)
    // ------------------------------------------------------------------
    Triangle* d_triangles = nullptr;
    FlatOctreeNode* d_nodes = nullptr;
    idx_t* d_triIndices = nullptr;
    val_t* d_areas = nullptr;
    val_t* d_rho = nullptr;
    val_t* d_radE = nullptr;
    val_t* d_radB = nullptr;  // full T x N history, replicated on every rank
    val_t* d_kijLocal = nullptr;
    int* d_tauLocal = nullptr;
    val_t* d_rowLocalOut = nullptr;
    val_t* d_distancesLocal = nullptr;
    unsigned long long* d_nonZeroCounter = nullptr;

    CUDA_CHECK(cudaMalloc(&d_triangles, N * sizeof(Triangle)));
    CUDA_CHECK(cudaMemcpy(d_triangles, state.triangles.data(), N * sizeof(Triangle), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_nodes, state.flatNodes.size() * sizeof(FlatOctreeNode)));
    CUDA_CHECK(cudaMemcpy(d_nodes, state.flatNodes.data(),
                          state.flatNodes.size() * sizeof(FlatOctreeNode), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_triIndices, std::max<size_t>(1, state.flatTriIndices.size()) * sizeof(idx_t)));
    if (!state.flatTriIndices.empty()) {
        CUDA_CHECK(cudaMemcpy(d_triIndices, state.flatTriIndices.data(),
                              state.flatTriIndices.size() * sizeof(idx_t), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&d_areas, N * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(d_areas, state.areas.data(), N * sizeof(val_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_rho, N * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(d_rho, state.rho.data(), N * sizeof(val_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_radE, T * N * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(d_radE, state.radE.data(), T * N * sizeof(val_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_radB, T * N * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(d_radB, 0, T * N * sizeof(val_t)));

    size_t localMatrixElems = std::max<size_t>(1, rowCount * N);
    CUDA_CHECK(cudaMalloc(&d_kijLocal, localMatrixElems * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_tauLocal, localMatrixElems * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rowLocalOut, std::max<size_t>(1, rowCount) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_distancesLocal, std::max<size_t>(1, rowCount) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d_nonZeroCounter, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_nonZeroCounter, 0, sizeof(unsigned long long)));

    // ------------------------------------------------------------------
    // Precomputation phase: Form factors (Kij) + time delays (Tau) on GPU
    // ------------------------------------------------------------------
    if (rank == 0) printf("Computing time delays (Tau) and form factors (Kij)...\n");

    auto startPre = std::chrono::high_resolution_clock::now();

    if (rowCount > 0) {
        const int threadsPerBlock = 256;
        size_t totalPairs = rowCount * N;
        int blocks = static_cast<int>(std::min<size_t>(65535, (totalPairs + threadsPerBlock - 1) / threadsPerBlock));
        blocks = std::max(blocks, 1);
        computeFormFactorsAndTauKernel<<<blocks, threadsPerBlock>>>(
            static_cast<idx_t>(rowStart), static_cast<idx_t>(rowCount), static_cast<idx_t>(N),
            d_triangles, d_nodes, d_triIndices, d_kijLocal, d_tauLocal, /*seed=*/42ULL);
        CUDA_CHECK(cudaGetLastError());

        int cblocks = static_cast<int>(std::min<size_t>(4096, (localMatrixElems + threadsPerBlock - 1) / threadsPerBlock));
        cblocks = std::max(cblocks, 1);
        countNonZeroKernel<<<cblocks, threadsPerBlock>>>(d_kijLocal, rowCount * N, d_nonZeroCounter);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // ------------------------------------------------------------------
    // Simulation phase (wave propagation), sequential over timesteps
    // ------------------------------------------------------------------
    if (rank == 0) printf("Running wave propagation simulation...\n");

    std::vector<val_t> localSendBuf(std::max<size_t>(1, rowCount));
    std::vector<val_t> fullRowBuf(N);

    auto startSim = std::chrono::high_resolution_clock::now();

    const int simThreads = 256;
    for (size_t t = 0; t < T; ++t) {
        if (rowCount > 0) {
            simulationStepKernel<<<static_cast<int>(rowCount), simThreads, simThreads * sizeof(val_t)>>>(
                static_cast<idx_t>(rowStart), static_cast<idx_t>(rowCount), static_cast<idx_t>(N), t,
                d_kijLocal, d_tauLocal, d_areas, d_rho,
                d_radE + t * N, d_radB, d_rowLocalOut);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(localSendBuf.data(), d_rowLocalOut, rowCount * sizeof(val_t), cudaMemcpyDeviceToHost));
        }

        MPI_Allgatherv(rowCount > 0 ? localSendBuf.data() : nullptr, static_cast<int>(rowCount), MPI_FLOAT,
                       fullRowBuf.data(), rowCounts.data(), rowDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(d_radB + t * N, fullRowBuf.data(), N * sizeof(val_t), cudaMemcpyHostToDevice));

        if (rank == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // ------------------------------------------------------------------
    // Distance computation (cross-correlation) on GPU
    // ------------------------------------------------------------------
    if (rank == 0) printf("Computing distances via cross-correlation...\n");

    auto startDist = std::chrono::high_resolution_clock::now();

    if (rowCount > 0) {
        const int distThreads = 256;
        computeDistancesKernel<<<static_cast<int>(rowCount), distThreads, distThreads * sizeof(val_t)>>>(
            static_cast<idx_t>(rowStart), static_cast<idx_t>(rowCount), static_cast<idx_t>(N), T,
            static_cast<idx_t>(state.sourceIndex), d_radB, d_distancesLocal);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<val_t> distLocal(std::max<size_t>(1, rowCount));
    if (rowCount > 0) {
        CUDA_CHECK(cudaMemcpy(distLocal.data(), d_distancesLocal, rowCount * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    MPI_Gatherv(rowCount > 0 ? distLocal.data() : nullptr, static_cast<int>(rowCount), MPI_FLOAT,
               state.distances.data(), rowCounts.data(), rowDispls.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    unsigned long long nonZeroKijLocal = 0, nonZeroKijTotal = 0;
    CUDA_CHECK(cudaMemcpy(&nonZeroKijLocal, d_nonZeroCounter, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    MPI_Reduce(&nonZeroKijLocal, &nonZeroKijTotal, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    int result = 0;

    if (rank == 0) {
        // Total time
        long totalTime = preDuration + simDuration + distDuration;
        printf("Total computation time: %ld ms\n", totalTime);

        // Performance metrics
        size_t n = N;
        size_t t = T;
        double kijOps = static_cast<double>(n) * static_cast<double>(n);
        double simOps = static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(t);
        double distOps = static_cast<double>(n) * static_cast<double>(t) * static_cast<double>(t);

        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        // Memory usage (full replicated matrices, as in the original single-node accounting)
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

        // Validation (needs the full radiosity history; gather it here)
        if (validate) {
            std::vector<val_t> radBFull(T * N);
            CUDA_CHECK(cudaMemcpy(radBFull.data(), d_radB, T * N * sizeof(val_t), cudaMemcpyDeviceToHost));
            if (!validateResults(state, radBFull, nonZeroKijTotal, n * n)) {
                result = 1;
            }
        }
    }

    cudaFree(d_triangles);
    cudaFree(d_nodes);
    cudaFree(d_triIndices);
    cudaFree(d_areas);
    cudaFree(d_rho);
    cudaFree(d_radE);
    cudaFree(d_radB);
    cudaFree(d_kijLocal);
    cudaFree(d_tauLocal);
    cudaFree(d_rowLocalOut);
    cudaFree(d_distancesLocal);
    cudaFree(d_nonZeroCounter);

    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
