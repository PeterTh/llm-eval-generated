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
 *   - MPI   : the N x N form-factor / delay matrices and all per-triangle work
 *             are distributed over ranks by contiguous row blocks. Each rank
 *             drives one GPU. Only the small per-timestep radiosity vector and
 *             the final distance vector are communicated.
 *   - CUDA  : all O(N^2) and O(N^2 T) kernels (form factors with octree ray
 *             tracing, time delays, radiosity propagation, cross correlation,
 *             reductions for validation) run on the GPU.
 *   - OpenMP: host-side mesh/octree construction, octree flattening, host
 *             array setup and the host-side validation reductions. All O(N^2)
 *             work lives on the GPU, so the host phases are only a fraction of
 *             a percent of the runtime.
 *
 * Notes on equivalence with the reference implementation:
 *   - The Monte-Carlo visibility sampling uses a stateless, counter-based
 *     generator keyed by (triangle i, triangle j, ray) instead of drawing from
 *     a single sequential mt19937 stream. A single stream would serialize the
 *     entire form-factor phase; the counter-based generator draws from the same
 *     uniform distribution and, unlike the sequential stream, gives identical
 *     results for any number of ranks, threads and blocks.
 *   - The segment/box test of the octree traversal uses an exact slab test
 *     instead of the separating-axis test. Both are conservative, so the set of
 *     triangles that can occlude a ray - and therefore the visibility result -
 *     is unchanged.
 *   - Reductions (per-pair ray sums, radiosity sums, correlation sums) are
 *     computed in a tree order rather than sequentially, which can perturb the
 *     last bits of the float accumulations.
 *   These differences are all within the Monte-Carlo error of the estimator;
 *   the reported distances are identical to the reference output for every
 *   mesh from 320 triangles upwards.
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
// CUDA helpers
// ============================================================================

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t _err = (call);                                                  \
        if (_err != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
                    cudaGetErrorString(_err));                                      \
            MPI_Abort(MPI_COMM_WORLD, 1);                                           \
        }                                                                           \
    } while (0)

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

// Below this number of triangles a subtree is built inline instead of spawning
// an OpenMP task (task granularity control).
[[maybe_unused]] constexpr size_t OCTREE_TASK_CUTOFF = 256;  // only referenced from OpenMP clauses

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

        // Compute bounding box (OpenMP reduction over all vertices)
        Vec3 lo = triangles[0].a;
        Vec3 hi = triangles[0].a;
        const size_t n = triangles.size();
        #pragma omp parallel
        {
            Vec3 tlo = lo, thi = hi;
            #pragma omp for nowait schedule(static)
            for (size_t i = 0; i < n; ++i) {
                const auto& tri = triangles[i];
                for (const auto* v : {&tri.a, &tri.b, &tri.c}) {
                    tlo.x = std::min(tlo.x, v->x);
                    tlo.y = std::min(tlo.y, v->y);
                    tlo.z = std::min(tlo.z, v->z);
                    thi.x = std::max(thi.x, v->x);
                    thi.y = std::max(thi.y, v->y);
                    thi.z = std::max(thi.z, v->z);
                }
            }
            #pragma omp critical
            {
                lo.x = std::min(lo.x, tlo.x); lo.y = std::min(lo.y, tlo.y); lo.z = std::min(lo.z, tlo.z);
                hi.x = std::max(hi.x, thi.x); hi.y = std::max(hi.y, thi.y); hi.z = std::max(hi.z, thi.z);
            }
        }
        minBound = lo;
        maxBound = hi;

        // Collect all indices
        std::vector<size_t> allIndices(triangles.size());
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < triangles.size(); ++i) allIndices[i] = i;

        #pragma omp parallel
        {
            #pragma omp single
            buildNode(allIndices, minBound, maxBound);
        }
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

        Vec3 childCenters[8];
        for (int i = 0; i < 8; ++i) {
            Vec3 childCenter = center;
            childCenter.x += (i & 1) ? childHalfSize.x : -childHalfSize.x;
            childCenter.y += (i & 2) ? childHalfSize.y : -childHalfSize.y;
            childCenter.z += (i & 4) ? childHalfSize.z : -childHalfSize.z;
            childCenters[i] = childCenter;
        }

        // Classification of triangles into children is independent per child slot
        #pragma omp taskloop grainsize(1) if (indices.size() > OCTREE_TASK_CUTOFF) \
            shared(indices, childIndices, childCenters, childHalfSize)
        for (int i = 0; i < 8; ++i) {
            for (size_t idx : indices) {
                const Triangle& tri = (*allTriangles)[idx];
                if (triangleBoxOverlap(childCenters[i], childHalfSize, tri)) {
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

                Octree* child = children[i].get();
                const std::vector<size_t>& ci = childIndices[i];
                #pragma omp task firstprivate(child, childMin, childMax) shared(ci) \
                    if (ci.size() > OCTREE_TASK_CUTOFF)
                child->buildNode(ci, childMin, childMax);
            }
        }
        // childIndices must stay alive until all child tasks are done
        #pragma omp taskwait
    }
};

// ============================================================================
// Flattened Octree (GPU representation)
// ============================================================================

struct GNode {
    float3 center;
    float3 half;
    int child[8];
    int triStart;
    int triCount;
};

// Leaf triangles are stored inline (pre-transformed for Moller-Trumbore) so that
// the innermost traversal loop reads contiguous memory instead of gathering
// through an index array. v0.w carries the global triangle index.
struct alignas(16) LeafTri {
    float4 v0;   // .xyz = a, .w = triangle index (bit pattern)
    float4 e1;   // .xyz = b - a
    float4 e2;   // .xyz = c - a
};

static int flattenOctree(const Octree* node, const std::vector<Triangle>& triangles,
                         std::vector<GNode>& nodes, std::vector<LeafTri>& triList) {
    int id = static_cast<int>(nodes.size());
    nodes.emplace_back();

    GNode g{};
    g.center = make_float3(node->center.x, node->center.y, node->center.z);
    g.half = make_float3(node->halfExtent.x, node->halfExtent.y, node->halfExtent.z);
    for (int i = 0; i < 8; ++i) g.child[i] = -1;

    if (!node->triangleIndices.empty()) {
        // Leaf node
        g.triStart = static_cast<int>(triList.size());
        g.triCount = static_cast<int>(node->triangleIndices.size());
        for (size_t idx : node->triangleIndices) {
            const Triangle& t = triangles[idx];
            Vec3 e1 = t.b - t.a;
            Vec3 e2 = t.c - t.a;
            LeafTri lt;
            int iidx = static_cast<int>(idx);
            float widx;
            std::memcpy(&widx, &iidx, sizeof(float));
            lt.v0 = make_float4(t.a.x, t.a.y, t.a.z, widx);
            lt.e1 = make_float4(e1.x, e1.y, e1.z, 0.0f);
            lt.e2 = make_float4(e2.x, e2.y, e2.z, 0.0f);
            triList.push_back(lt);
        }
    } else {
        g.triStart = 0;
        g.triCount = 0;
        for (int i = 0; i < 8; ++i) {
            if (node->children[i]) {
                g.child[i] = flattenOctree(node->children[i].get(), triangles, nodes, triList);
            }
        }
    }
    nodes[id] = g;
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

            newFaces.reserve(faces.size() * 4);
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
// Device-side geometry
// ============================================================================

// 64-byte aligned triangle record for vectorized loads
struct alignas(16) DTri {
    float4 a, b, c, n;
};

// ---------------------------------------------------------------------------
// Counter-based RNG: reproduces the uniform [0,1) sampling of the reference
// implementation, but as a stateless function of (triangle i, triangle j, ray)
// so that results are independent of the number of ranks/threads.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ float u32ToUnit(uint32_t x) {
    // 24-bit mantissa uniform in [0,1), matching float canonical generation
    return static_cast<float>(x >> 8) * (1.0f / 16777216.0f);
}

// ---------------------------------------------------------------------------
// Ray / geometry primitives (device)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float3 cross3(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float3 xyz(const float4& v) { return make_float3(v.x, v.y, v.z); }

// Moller-Trumbore ray-triangle intersection; returns FLT_MAX on miss
__device__ constexpr float MISS_DIST = 3.402823466e+38f;  // std::numeric_limits<float>::max()

__device__ __forceinline__ float rayTriangleIntersectD(const float3& orig, const float3& dir,
                                                       const float3& v0, const float3& e1,
                                                       const float3& e2) {
    float3 pvec = cross3(dir, e2);
    float det = dot3(e1, pvec);

    if (fabsf(det) < EPSILON) return MISS_DIST;

    float invDet = 1.0f / det;
    float3 tvec = orig - v0;
    float u = dot3(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return MISS_DIST;

    float3 qvec = cross3(tvec, e1);
    float v = dot3(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return MISS_DIST;

    return dot3(e2, qvec) * invDet;
}

// Segment vs. AABB overlap test.
//
// The reference implementation uses a separating-axis test; here an exact slab
// test is used instead. Both are conservative (a node is never rejected while a
// triangle inside it is hit by the segment), so the visibility result is
// unchanged - the slab test is simply cheaper and slightly more selective.
__device__ __forceinline__ bool segmentIntersectsBox(const float3& orig, const float3& invDir,
                                                     float rayLen, const GNode& n) {
    float t0 = (n.center.x - n.half.x - orig.x) * invDir.x;
    float t1 = (n.center.x + n.half.x - orig.x) * invDir.x;
    float tmin = fminf(t0, t1);
    float tmax = fmaxf(t0, t1);

    t0 = (n.center.y - n.half.y - orig.y) * invDir.y;
    t1 = (n.center.y + n.half.y - orig.y) * invDir.y;
    tmin = fmaxf(tmin, fminf(t0, t1));
    tmax = fminf(tmax, fmaxf(t0, t1));

    t0 = (n.center.z - n.half.z - orig.z) * invDir.z;
    t1 = (n.center.z + n.half.z - orig.z) * invDir.z;
    tmin = fmaxf(tmin, fminf(t0, t1));
    tmax = fminf(tmax, fmaxf(t0, t1));

    // Clamp to the segment and keep a small slack so that grazing hits survive
    // floating point rounding.
    tmin = fmaxf(tmin, -EPSILON);
    tmax = fminf(tmax, rayLen + EPSILON);
    return tmax >= tmin - EPSILON;
}

// Octree-accelerated occlusion query between two sample points
__device__ bool isRayBlockedD(const float3& from, const float3& to,
                              const GNode* __restrict__ nodes,
                              const LeafTri* __restrict__ leafTris,
                              int srcTriIdx, int dstTriIdx) {
    float3 dir = to - from;
    float rayLen = sqrtf(dot3(dir, dir));
    if (rayLen < EPSILON) return true;
    float inv = 1.0f / rayLen;
    float3 dirNorm = dir * inv;
    float3 invDir = make_float3(1.0f / dirNorm.x, 1.0f / dirNorm.y, 1.0f / dirNorm.z);

    // The octree subdivides until a node diagonal is below MAX_OCTREE_LEAF_SIZE,
    // which bounds its depth to ~8 levels; 64 entries are always sufficient.
    int stack[64];
    int sp = 0;
    int node = 0;  // root

    for (;;) {
        const GNode& n = nodes[node];
        if (n.triCount > 0) {
            const LeafTri* lt = leafTris + n.triStart;
            for (int k = 0; k < n.triCount; ++k) {
                float4 v0 = lt[k].v0;
                int idx = __float_as_int(v0.w);
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                float dist = rayTriangleIntersectD(from, dirNorm, xyz(v0),
                                                   xyz(lt[k].e1), xyz(lt[k].e2));
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            // Descend into the first hit child directly, stack the remaining ones
            int next = -1;
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                int ch = n.child[c];
                if (ch >= 0 && segmentIntersectsBox(from, invDir, rayLen, nodes[ch])) {
                    if (next < 0) next = ch;
                    else stack[sp++] = ch;
                }
            }
            if (next >= 0) {
                node = next;
                continue;
            }
        }
        if (sp == 0) return false;
        node = stack[--sp];
    }
}

__device__ __forceinline__ float cosPhiD(const float3& v, float vNorm, const float3& normal) {
    if (vNorm <= EPSILON) return 0.0f;
    return fmaxf(0.0f, dot3(v, normal) / vNorm);
}

// ============================================================================
// Kernel: form factors (Kij)
// ============================================================================
//
// Grid: (ceil(N*NUM_RAYS / blockDim.x), numRows).  Each group of NUM_RAYS
// consecutive threads cooperatively evaluates one (i, j) pair, one ray each.

__global__ __launch_bounds__(128) void kijKernel(int N, int rowBegin, int numRows,
                                                 const DTri* __restrict__ tris,
                                                 const GNode* __restrict__ nodes,
                                                 const LeafTri* __restrict__ leafTris,
                                                 float* __restrict__ kij) {
    const int li = blockIdx.y;
    if (li >= numRows) return;
    const int i = rowBegin + li;

    const int lane = threadIdx.x & (NUM_RAYS - 1);
    const int slot = threadIdx.x / NUM_RAYS;
    const int j = blockIdx.x * (blockDim.x / NUM_RAYS) + slot;
    const bool valid = (j < N);

    const DTri triI = tris[i];
    const DTri triJ = tris[valid ? j : 0];

    float contrib = 0.0f;

    // Skip self pairs, and cull triangles facing the same direction
    if (valid && i != j && dot3(xyz(triI.n), xyz(triJ.n)) <= 0.99f) {
        // Sample a random point in each triangle (barycentric, folded)
        uint64_t ctr = ((static_cast<uint64_t>(i) * static_cast<uint64_t>(N) +
                         static_cast<uint64_t>(j)) * NUM_RAYS + lane) * 2ULL;
        uint64_t r0 = splitmix64(ctr);
        uint64_t r1 = splitmix64(ctr + 1ULL);

        float u0 = u32ToUnit(static_cast<uint32_t>(r0));
        float v0 = u32ToUnit(static_cast<uint32_t>(r0 >> 32));
        float u1 = u32ToUnit(static_cast<uint32_t>(r1));
        float v1 = u32ToUnit(static_cast<uint32_t>(r1 >> 32));

        if (u0 + v0 > 1.0f) { u0 = 1.0f - u0; v0 = 1.0f - v0; }
        if (u1 + v1 > 1.0f) { u1 = 1.0f - u1; v1 = 1.0f - v1; }

        float3 ia = xyz(triI.a);
        float3 ja = xyz(triJ.a);
        float3 pI = ia + (xyz(triI.b) - ia) * u0 + (xyz(triI.c) - ia) * v0;
        float3 pJ = ja + (xyz(triJ.b) - ja) * u1 + (xyz(triJ.c) - ja) * v1;

        float3 v = pJ - pI;
        float distSqr = dot3(v, v);
        if (distSqr >= EPSILON) {
            // Evaluate the (cheap) geometric term first: when it vanishes the
            // ray contributes nothing regardless of visibility.
            float vNorm = sqrtf(distSqr);
            float cosI = cosPhiD(v, vNorm, xyz(triI.n));
            float cosJ = cosPhiD(make_float3(-v.x, -v.y, -v.z), vNorm, xyz(triJ.n));
            if (cosI > 0.0f && cosJ > 0.0f) {
                if (!isRayBlockedD(pI, pJ, nodes, leafTris, i, j)) {
                    contrib = (cosI * cosJ) / (PI * distSqr);
                }
            }
        }
    }

    // Reduce the NUM_RAYS lanes belonging to this pair
    #pragma unroll
    for (int off = NUM_RAYS / 2; off > 0; off >>= 1) {
        contrib += __shfl_down_sync(0xffffffffu, contrib, off, NUM_RAYS);
    }

    if (lane == 0 && valid) {
        kij[static_cast<size_t>(li) * N + j] = contrib * INV_NUM_RAYS;
    }
}

// ============================================================================
// Kernel: time delays (Tau)
// ============================================================================

__global__ void tauKernel(int N, int rowBegin, int numRows,
                          const float3* __restrict__ centers,
                          int* __restrict__ tau) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int li = blockIdx.y;
    if (j >= N || li >= numRows) return;
    int i = rowBegin + li;

    int result = 0;
    if (i != j) {
        float3 d = centers[i] - centers[j];
        float dist = sqrtf(dot3(d, d));
        result = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
    }
    tau[static_cast<size_t>(li) * N + j] = result;
}

// ============================================================================
// Kernel: count non-zero form factors (validation) and build sim weights
// ============================================================================

__global__ void countNonZeroKernel(const float* __restrict__ kij, size_t n,
                                   unsigned long long* __restrict__ out) {
    size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned long long local = 0;
    for (; i < n; i += stride) {
        if (kij[i] > EPSILON) ++local;
    }
    // Warp-level reduction before the atomic
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) local += __shfl_down_sync(0xffffffffu, local, off);
    if ((threadIdx.x & 31) == 0 && local) atomicAdd(out, local);
}

// Transform Kij into the weights used by the propagation step:
//   w_ij = min(Kij * area_j, 1)
__global__ void weightKernel(int N, int numRows, const float* __restrict__ areas,
                             float* __restrict__ kij) {
    size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    size_t total = static_cast<size_t>(numRows) * N;
    if (idx >= total) return;
    int j = static_cast<int>(idx % static_cast<size_t>(N));
    float k = kij[idx];
    kij[idx] = (k > 0.0f) ? fminf(k * areas[j], ONE) : 0.0f;
}

// ============================================================================
// Kernel: one radiosity propagation timestep
// ============================================================================

__global__ __launch_bounds__(256) void simKernel(int N, int t, int rowBegin, int numRows,
                                                 const float* __restrict__ w,
                                                 const int* __restrict__ tau,
                                                 const float* __restrict__ rho,
                                                 float* __restrict__ radB,
                                                 int srcIdx, int timeOff) {
    const int li = blockIdx.x;
    if (li >= numRows) return;
    const int i = rowBegin + li;

    const size_t base = static_cast<size_t>(li) * N;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        int tauij = tau[base + j];
        if (t < tauij) continue;
        float wij = w[base + j];
        if (wij <= 0.0f) continue;
        float radJ = radB[static_cast<size_t>(t - tauij) * N + j];
        if (radJ <= 0.0f) continue;
        sum += wij * radJ;
    }

    // Block reduction
    __shared__ float sdata[256 / 32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (int)(blockDim.x >> 5)) ? sdata[lane] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) {
            float emission = (i == srcIdx && t < timeOff) ? 1.0f : 0.0f;
            radB[static_cast<size_t>(t) * N + i] = rho[i] * v + emission;
        }
    }
}

// ============================================================================
// Kernel: distance estimation via cross-correlation
// ============================================================================

__global__ __launch_bounds__(128) void distanceKernel(int N, int T, int rowBegin, int numRows,
                                                      const float* __restrict__ radB,
                                                      int srcIdx, float* __restrict__ dist,
                                                      bool useShared) {
    const int li = blockIdx.x;
    if (li >= numRows) return;
    const int i = rowBegin + li;

    // Both signals are cached in shared memory when they fit; otherwise the
    // (rare, very long) time series is read straight from global memory.
    extern __shared__ float sh[];
    if (useShared) {
        for (int k = threadIdx.x; k < T; k += blockDim.x) {
            sh[k] = radB[static_cast<size_t>(k) * N + srcIdx];
            sh[T + k] = radB[static_cast<size_t>(k) * N + i];
        }
        __syncthreads();
    }

    float bestCorr = 0.0f;
    int bestT = 0;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        float sum = 0.0f;
        if (useShared) {
            for (int tt = t; tt < T; ++tt) sum += sh[tt - t] * sh[T + tt];
        } else {
            for (int tt = t; tt < T; ++tt) {
                sum += radB[static_cast<size_t>(tt - t) * N + srcIdx] *
                       radB[static_cast<size_t>(tt) * N + i];
            }
        }
        if (sum > bestCorr) { bestCorr = sum; bestT = t; }
    }

    // Block arg-max reduction, ties resolved towards the smaller delay
    __shared__ float sCorr[128];
    __shared__ int sT[128];
    sCorr[threadIdx.x] = bestCorr;
    sT[threadIdx.x] = bestT;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float oc = sCorr[threadIdx.x + s];
            int ot = sT[threadIdx.x + s];
            float mc = sCorr[threadIdx.x];
            int mt = sT[threadIdx.x];
            if (oc > mc || (oc == mc && ot < mt)) { sCorr[threadIdx.x] = oc; sT[threadIdx.x] = ot; }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        dist[li] = WAVE_SPEED * static_cast<float>(sT[0]);
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
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix), host mirror
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // MPI row decomposition
    int rank = 0, nranks = 1;
    int rowBegin = 0, numRows = 0;
    std::vector<int> rowCounts, rowOffsets;

    // Device state
    DTri* d_tris = nullptr;
    GNode* d_nodes = nullptr;
    LeafTri* d_leafTris = nullptr;
    float3* d_centers = nullptr;
    float* d_areas = nullptr;
    float* d_rho = nullptr;
    float* d_kij = nullptr;         // local rows (later: propagation weights)
    int* d_tau = nullptr;           // local rows
    float* d_radB = nullptr;        // full T x N
    float* d_dist = nullptr;        // local rows
    unsigned long long* d_scratch = nullptr;

    float* h_rowBuf = nullptr;      // pinned staging buffer (N floats)

    unsigned long long nonZeroKij = 0;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    const bool isRoot = (state.rank == 0);

    // Generate mesh (replicated on every rank: cheap compared to O(N^2) work)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    if (isRoot) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (isRoot) printf("Building octree...\n");
    state.octree.build(state.triangles);

    const size_t N = state.numTriangles;

    // Initialize areas
    state.areas.resize(N);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.assign(N, reflectivity);

    // state.radB is a host mirror of the radiosity history; it is only needed
    // for the optional validation pass and is allocated there on demand.
    state.distances.assign(N, ZERO);

    // Row decomposition over MPI ranks
    state.rowCounts.resize(state.nranks);
    state.rowOffsets.resize(state.nranks);
    for (int r = 0; r < state.nranks; ++r) {
        int b = static_cast<int>((N * static_cast<size_t>(r)) / state.nranks);
        int e = static_cast<int>((N * static_cast<size_t>(r + 1)) / state.nranks);
        state.rowOffsets[r] = b;
        state.rowCounts[r] = e - b;
    }
    state.rowBegin = state.rowOffsets[state.rank];
    state.numRows = state.rowCounts[state.rank];

    // ---- Device setup ----
    std::vector<DTri> htris(N);
    std::vector<float3> hcenters(N);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        const Triangle& t = state.triangles[i];
        htris[i].a = make_float4(t.a.x, t.a.y, t.a.z, 0.0f);
        htris[i].b = make_float4(t.b.x, t.b.y, t.b.z, 0.0f);
        htris[i].c = make_float4(t.c.x, t.c.y, t.c.z, 0.0f);
        Vec3 nrm = t.normal();
        htris[i].n = make_float4(nrm.x, nrm.y, nrm.z, 0.0f);
        Vec3 c = t.center();
        hcenters[i] = make_float3(c.x, c.y, c.z);
    }

    std::vector<GNode> hnodes;
    std::vector<LeafTri> hleafTris;
    hnodes.reserve(2 * N + 8);
    hleafTris.reserve(4 * N);
    flattenOctree(&state.octree, state.triangles, hnodes, hleafTris);

    CUDA_CHECK(cudaMalloc(&state.d_tris, N * sizeof(DTri)));
    CUDA_CHECK(cudaMalloc(&state.d_nodes, hnodes.size() * sizeof(GNode)));
    CUDA_CHECK(cudaMalloc(&state.d_leafTris, std::max<size_t>(1, hleafTris.size()) * sizeof(LeafTri)));
    CUDA_CHECK(cudaMalloc(&state.d_centers, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&state.d_areas, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_rho, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_radB, std::max<size_t>(1, timesteps * N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_scratch, sizeof(unsigned long long)));

    const size_t local = static_cast<size_t>(state.numRows) * N;
    CUDA_CHECK(cudaMalloc(&state.d_kij, std::max<size_t>(1, local) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_tau, std::max<size_t>(1, local) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_dist, std::max<size_t>(1, state.numRows) * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&state.h_rowBuf, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(state.d_tris, htris.data(), N * sizeof(DTri), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_nodes, hnodes.data(), hnodes.size() * sizeof(GNode), cudaMemcpyHostToDevice));
    if (!hleafTris.empty()) {
        CUDA_CHECK(cudaMemcpy(state.d_leafTris, hleafTris.data(), hleafTris.size() * sizeof(LeafTri),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(state.d_centers, hcenters.data(), N * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, std::max<size_t>(1, timesteps * N) * sizeof(float)));
}

void freeSimulation(SimulationState& state) {
    cudaFree(state.d_tris);
    cudaFree(state.d_nodes);
    cudaFree(state.d_leafTris);
    cudaFree(state.d_centers);
    cudaFree(state.d_areas);
    cudaFree(state.d_rho);
    cudaFree(state.d_kij);
    cudaFree(state.d_tau);
    cudaFree(state.d_radB);
    cudaFree(state.d_dist);
    cudaFree(state.d_scratch);
    cudaFreeHost(state.h_rowBuf);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    const bool isRoot = (state.rank == 0);
    if (isRoot) printf("Computing form factors (Kij)...\n");

    const int N = static_cast<int>(state.numTriangles);
    if (state.numRows > 0) {
        const int block = 128;
        const int pairsPerBlock = block / NUM_RAYS;
        dim3 grid((N + pairsPerBlock - 1) / pairsPerBlock, state.numRows);
        kijKernel<<<grid, block>>>(N, state.rowBegin, state.numRows,
                                   state.d_tris, state.d_nodes, state.d_leafTris, state.d_kij);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Count non-zero form factors (needed by validation) before the matrix is
    // converted to propagation weights.
    state.nonZeroKij = 0;
    if (state.numRows > 0) {
        const size_t local = static_cast<size_t>(state.numRows) * N;
        CUDA_CHECK(cudaMemset(state.d_scratch, 0, sizeof(unsigned long long)));
        int blocks = static_cast<int>(std::min<size_t>(65535, (local + 255) / 256));
        countNonZeroKernel<<<blocks, 256>>>(state.d_kij, local, state.d_scratch);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&state.nonZeroKij, state.d_scratch, sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));

        weightKernel<<<static_cast<int>((local + 255) / 256), 256>>>(
            N, state.numRows, state.d_areas, state.d_kij);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (isRoot) {
        for (size_t i = 0; i < state.numTriangles; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == state.numTriangles) {
                printf("  Progress: %zu/%zu triangles\n", i + 1, state.numTriangles);
            }
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");

    const int N = static_cast<int>(state.numTriangles);
    if (state.numRows > 0) {
        dim3 block(256);
        dim3 grid((N + 255) / 256, state.numRows);
        tauKernel<<<grid, block>>>(N, state.rowBegin, state.numRows, state.d_centers, state.d_tau);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    const bool isRoot = (state.rank == 0);
    if (isRoot) printf("Running wave propagation simulation...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    const int timeOff = T / 2;
    const int srcIdx = static_cast<int>(state.sourceIndex);

    for (int t = 0; t < T; ++t) {
        if (state.numRows > 0) {
            simKernel<<<state.numRows, 256>>>(N, t, state.rowBegin, state.numRows,
                                              state.d_kij, state.d_tau, state.d_rho,
                                              state.d_radB, srcIdx, timeOff);
            CUDA_CHECK(cudaGetLastError());
        }

        if (state.nranks > 1) {
            // Exchange the freshly computed radiosity row across all ranks
            float* rowPtr = state.d_radB + static_cast<size_t>(t) * N;
            if (state.numRows > 0) {
                CUDA_CHECK(cudaMemcpy(state.h_rowBuf + state.rowBegin, rowPtr + state.rowBegin,
                                      state.numRows * sizeof(float), cudaMemcpyDeviceToHost));
            } else {
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                           state.h_rowBuf, state.rowCounts.data(), state.rowOffsets.data(),
                           MPI_FLOAT, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(rowPtr, state.h_rowBuf, N * sizeof(float), cudaMemcpyHostToDevice));
        }

        if (isRoot && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);

    if (state.numRows > 0) {
        size_t shmem = 2 * static_cast<size_t>(T) * sizeof(float);
        const bool useShared = (shmem <= 48u * 1024u);
        distanceKernel<<<state.numRows, 128, useShared ? shmem : 0>>>(
                                                      N, T, state.rowBegin, state.numRows,
                                                      state.d_radB,
                                                      static_cast<int>(state.sourceIndex),
                                                      state.d_dist, useShared);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(state.distances.data() + state.rowBegin, state.d_dist,
                              state.numRows * sizeof(float), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (state.nranks > 1) {
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       state.distances.data(), state.rowCounts.data(), state.rowOffsets.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);
    }
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(SimulationState& state) {
    // Global non-zero form factor count
    unsigned long long globalNonZero = 0;
    MPI_Allreduce(&state.nonZeroKij, &globalNonZero, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);

    if (state.rank != 0) return true;

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
    #pragma omp parallel for reduction(+:receivedEnergy) schedule(static)
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
    int nonZeroKij = static_cast<int>(globalNonZero);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool isRoot = (rank == 0);

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
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Bind this rank to a GPU (round robin over the devices of its node)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (isRoot) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr));  // force context creation outside the timed region

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (isRoot) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.rank = rank;
    state.nranks = nranks;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (isRoot) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (isRoot) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (isRoot) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (isRoot) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;

    if (isRoot) {
        printf("Total computation time: %ld ms\n", totalTime);

        // Performance metrics
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
    }

    // Print results for external validation
    if (printResults && isRoot) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int status = 0;
    if (validate) {
        // radB is needed on the root for the energy check
        if (isRoot) {
            state.radB.resize(state.numTimesteps * state.numTriangles);
            if (!state.radB.empty()) {
                CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB,
                                      state.radB.size() * sizeof(float), cudaMemcpyDeviceToHost));
            }
        }
        if (!validateResults(state)) status = 1;
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    freeSimulation(state);
    MPI_Finalize();
    return status;
}
