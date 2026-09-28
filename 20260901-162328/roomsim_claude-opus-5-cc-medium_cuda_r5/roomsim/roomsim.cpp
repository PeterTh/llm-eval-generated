/**
 * Room Response Simulation Benchmark -- CUDA parallel implementation
 *
 * This is a CUDA-parallel implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization notes
 * ---------------------
 * All three compute phases run on the GPU:
 *
 *  - Form factors: one 16-lane group per triangle pair, one lane per visibility
 *    ray, traversing a flattened (array based) octree. The reference code draws
 *    its random samples from a single sequential std::mt19937 stream, so the
 *    exact stream is reproduced on the device: pairs that are culled by the
 *    normal test consume no random numbers, therefore an exclusive prefix sum
 *    over the cull mask yields each pair's rank and hence its offset (64 draws
 *    per pair) into the global stream. The stream itself is produced by a
 *    block-parallel MT19937 kernel (the twist is split into three dependency
 *    phases) into a double buffered window, generated on a second CUDA stream
 *    so that it overlaps with the form factor kernel.
 *
 *  - Wave propagation: one warp per receiver triangle per timestep. The warp
 *    evaluates 32 emitters at a time with fully coalesced row loads, then folds
 *    their contributions into the running sum lane by lane.
 *
 *  - Cross correlation: one block per triangle, one thread per lag, with the
 *    radiosity columns staged in shared memory and a first-wins argmax
 *    reduction that reproduces the sequential tie-breaking rule.
 *
 * Results are bit-identical to the sequential reference. Floating point
 * summation orders are preserved everywhere (lane by lane folds instead of tree
 * reductions), and the multiply-add fusions the reference compiler performs are
 * spelled out explicitly so the device rounds at exactly the same points.
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
// CUDA helpers
// ============================================================================

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t _err = (call);                                                             \
        if (_err != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_err), __FILE__,    \
                    __LINE__);                                                                 \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
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

// Random draws consumed by one non-culled triangle pair (2 points * 2 draws * NUM_RAYS)
constexpr int DRAWS_PER_PAIR = 4 * NUM_RAYS;

// ============================================================================
// Vector and Triangle Types
// ============================================================================

// The reference build (g++ -O3 -march=native, i.e. -ffp-contract=fast) fuses
// the multiply-add chains of the vector maths into FMAs in a fixed order:
// "a*b + c*d" keeps c*d rounded and fuses a*b, and "a*b - c*d" becomes an
// fmsub. Spelling those fusions out explicitly keeps the device arithmetic
// bit-identical to the sequential reference instead of relying on whatever the
// device compiler happens to contract.
__host__ __device__ inline val_t fmaAdd2(val_t a, val_t b, val_t c, val_t d) {
    return fmaf(a, b, c * d);  // a*b + c*d
}
__host__ __device__ inline val_t fmaAdd3(val_t a, val_t b, val_t c, val_t d, val_t e, val_t f) {
    return fmaf(e, f, fmaAdd2(a, b, c, d));  // a*b + c*d + e*f
}
__host__ __device__ inline val_t fmaSub2(val_t a, val_t b, val_t c, val_t d) {
    return fmaf(a, b, -(c * d));  // a*b - c*d
}

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

    __host__ __device__ val_t dot(const Vec3& o) const {
        return fmaAdd3(x, o.x, y, o.y, z, o.z);
    }
    // Left in its original form: the reference compiler contracts this one
    // differently per call site, and the value only reaches the device through
    // host computed normals (identical code) plus the ray/triangle test, whose
    // outcome is a robust boolean.
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    __host__ __device__ val_t squaredNorm() const { return fmaAdd3(x, x, y, y, z, z); }
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
    __host__ __device__ Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    __host__ __device__ val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    __host__ __device__ bool operator==(const Triangle& o) const {
        return a == o.a && b == o.b && c == o.c;
    }
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

    // The ray/box test and the traversal itself live on the device, see
    // nodeRayIntersects() and isRayBlockedDev() below.
};

// ============================================================================
// Flattened (device side) octree
// ============================================================================

// Depth of the explicit traversal stack. A node expansion pops one entry and
// pushes up to eight, so the requirement is bounded by 1 + 7 * treeDepth; the
// actual requirement is computed while flattening and checked against this.
constexpr int MAX_TRAVERSAL_STACK = 64;

// A node is a leaf iff triCount > 0; the reference traversal descends into
// children only for nodes that hold no triangles of their own.
struct GNode {
    Vec3 center;
    Vec3 halfExtent;
    int child[8];
    int triStart;
    int triCount;
};

static int flattenOctree(const Octree* node, std::vector<GNode>& nodes, std::vector<int>& triIdx) {
    const int id = static_cast<int>(nodes.size());
    nodes.emplace_back();

    GNode n;
    n.center = node->center;
    n.halfExtent = node->halfExtent;
    for (int i = 0; i < 8; ++i) n.child[i] = -1;
    n.triStart = 0;
    n.triCount = 0;

    if (!node->triangleIndices.empty()) {
        n.triStart = static_cast<int>(triIdx.size());
        n.triCount = static_cast<int>(node->triangleIndices.size());
        for (size_t idx : node->triangleIndices) triIdx.push_back(static_cast<int>(idx));
    } else {
        for (int i = 0; i < 8; ++i) {
            if (node->children[i]) {
                n.child[i] = flattenOctree(node->children[i].get(), nodes, triIdx);
            }
        }
    }

    nodes[id] = n;
    return id;
}

// Deepest stack occupancy a depth first traversal of the flattened tree can reach
static int maxTraversalStack(const std::vector<GNode>& nodes, int id, int stackOnEntry) {
    const GNode& n = nodes[id];
    if (n.triCount > 0) return stackOnEntry;

    int children = 0;
    for (int i = 0; i < 8; ++i) {
        if (n.child[i] >= 0) ++children;
    }
    // this node is popped, its children are pushed
    const int afterExpand = stackOnEntry - 1 + children;
    int worst = afterExpand;
    for (int i = 0; i < 8; ++i) {
        if (n.child[i] >= 0) {
            worst = std::max(worst, maxTraversalStack(nodes, n.child[i], afterExpand));
        }
    }
    return worst;
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
//
// The reference implementation draws all samples from one sequential
// std::mt19937 fed through std::uniform_real_distribution<float>(0,1).
// For libstdc++ that distribution reduces to std::generate_canonical<float,24>,
// which consumes exactly one 32 bit word and returns word * 2^-32 (clamped just
// below one). The device generator below reproduces that stream exactly.

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr uint32_t MT_MATRIX_A = 0x9908b0dfu;
constexpr uint32_t MT_UPPER_MASK = 0x80000000u;
constexpr uint32_t MT_LOWER_MASK = 0x7fffffffu;

// Seed the MT19937 state exactly like std::mt19937 does
static void mtSeedState(uint32_t seed, uint32_t* state) {
    state[0] = seed;
    for (int i = 1; i < MT_N; ++i) {
        state[i] = 1812433253u * (state[i - 1] ^ (state[i - 1] >> 30)) + static_cast<uint32_t>(i);
    }
}

__host__ __device__ inline uint32_t mtTemper(uint32_t x) {
    x ^= (x >> 11);
    x ^= (x << 7) & 0x9d2c5680u;
    x ^= (x << 15) & 0xefc60000u;
    x ^= (x >> 18);
    return x;
}

// std::generate_canonical<float, 24, mt19937>
__host__ __device__ inline float mtCanonical(uint32_t x) {
    float r = static_cast<float>(x) * (1.0f / 4294967296.0f);
    // The cast can round up to exactly 2^32, in which case libstdc++ clamps
    return r >= 1.0f ? 0x1.fffffep-1f : r;
}

// Generate `numTwists * 624` canonical floats starting from `state`.
// After `snapTwist` twists the state is written back so the next window can
// continue the very same stream.
__global__ void mtGenerateKernel(uint32_t* __restrict__ state, float* __restrict__ out,
                                 int numTwists, int snapTwist) {
    __shared__ uint32_t buf[2][MT_N];

    const int tid = static_cast<int>(threadIdx.x);
    if (tid < MT_N) buf[0][tid] = state[tid];
    __syncthreads();

    int cur = 0;
    for (int tw = 0; tw < numTwists; ++tw) {
        const uint32_t* __restrict__ c = buf[cur];
        uint32_t* __restrict__ n = buf[cur ^ 1];

        // Phase 1: indices [0, N-M) depend only on the previous state
        if (tid < MT_N - MT_M) {
            uint32_t y = (c[tid] & MT_UPPER_MASK) | (c[tid + 1] & MT_LOWER_MASK);
            n[tid] = c[tid + MT_M] ^ (y >> 1) ^ ((y & 1u) ? MT_MATRIX_A : 0u);
        }
        __syncthreads();

        // Phase 2: indices [N-M, 2*(N-M)) depend on phase 1 results
        if (tid >= MT_N - MT_M && tid < 2 * (MT_N - MT_M)) {
            uint32_t y = (c[tid] & MT_UPPER_MASK) | (c[tid + 1] & MT_LOWER_MASK);
            n[tid] = n[tid - (MT_N - MT_M)] ^ (y >> 1) ^ ((y & 1u) ? MT_MATRIX_A : 0u);
        }
        __syncthreads();

        // Phase 3: the remaining indices depend on phase 2 results
        if (tid >= 2 * (MT_N - MT_M) && tid < MT_N - 1) {
            uint32_t y = (c[tid] & MT_UPPER_MASK) | (c[tid + 1] & MT_LOWER_MASK);
            n[tid] = n[tid - (MT_N - MT_M)] ^ (y >> 1) ^ ((y & 1u) ? MT_MATRIX_A : 0u);
        } else if (tid == MT_N - 1) {
            uint32_t y = (c[MT_N - 1] & MT_UPPER_MASK) | (n[0] & MT_LOWER_MASK);
            n[MT_N - 1] = n[MT_M - 1] ^ (y >> 1) ^ ((y & 1u) ? MT_MATRIX_A : 0u);
        }
        __syncthreads();

        if (tid < MT_N) {
            out[static_cast<size_t>(tw) * MT_N + tid] = mtCanonical(mtTemper(n[tid]));
            if (tw == snapTwist - 1) state[tid] = n[tid];
        }
        cur ^= 1;
    }
}

// Generate a random point inside a triangle using barycentric coordinates
__host__ __device__ inline Vec3 pointInTriangle(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return {fmaf(ac.x, v, fmaf(ab.x, u, t.a.x)), fmaf(ac.y, v, fmaf(ab.y, u, t.a.y)),
            fmaf(ac.z, v, fmaf(ab.z, u, t.a.z))};
}

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

__host__ __device__ inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                                      const Vec3& v0, const Vec3& v1,
                                                      const Vec3& v2) {
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
// Visibility Testing (Octree-accelerated, device side)
// ============================================================================

__device__ inline bool nodeRayIntersects(const GNode& node, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3 h = node.halfExtent;

    if (fabsf(c.x) > h.x + ad.x) return false;
    if (fabsf(c.y) > h.y + ad.y) return false;
    if (fabsf(c.z) > h.z + ad.z) return false;

    if (fabsf(fmaSub2(d.y, c.z, d.z, c.y)) > fmaAdd2(h.y, ad.z, h.z, ad.y) + EPSILON) return false;
    if (fabsf(fmaSub2(d.z, c.x, d.x, c.z)) > fmaAdd2(h.z, ad.x, h.x, ad.z) + EPSILON) return false;
    if (fabsf(fmaSub2(d.x, c.y, d.y, c.x)) > fmaAdd2(h.x, ad.y, h.y, ad.x) + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle
__device__ inline bool isRayBlockedDev(const Vec3& from, const Vec3& to,
                                       const GNode* __restrict__ nodes,
                                       const int* __restrict__ nodeTris,
                                       const Triangle* __restrict__ triangles,
                                       int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[MAX_TRAVERSAL_STACK];
    int sp = 0;
    stack[sp++] = 0;  // the root is entered without a bounding box test

    while (sp > 0) {
        const GNode& node = nodes[stack[--sp]];

        if (node.triCount > 0) {
            const int end = node.triStart + node.triCount;
            for (int k = node.triStart; k < end; ++k) {
                const int idx = nodeTris[k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const Triangle& tri = triangles[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int i = 0; i < 8; ++i) {
                const int ci = node.child[i];
                if (ci >= 0 && nodeRayIntersects(nodes[ci], from, to)) stack[sp++] = ci;
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__host__ __device__ inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// Mark the pairs that actually consume random numbers (i.e. are not culled)
__global__ void cullMaskKernel(uint32_t n, const Triangle* __restrict__ triangles,
                               uint32_t* __restrict__ flags) {
    const size_t total = static_cast<size_t>(n) * n;
    for (size_t p = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; p < total;
         p += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const uint32_t i = static_cast<uint32_t>(p / n);
        const uint32_t j = static_cast<uint32_t>(p - static_cast<size_t>(i) * n);
        bool active = (i != j) && !(triangles[i].normal().dot(triangles[j].normal()) > 0.99f);
        flags[p] = active ? 1u : 0u;
    }
}

// One 16-lane group per triangle pair, one lane per visibility ray.
__global__ __launch_bounds__(256) void formFactorKernel(
    uint32_t n, size_t pairBegin, size_t pairEnd, const Triangle* __restrict__ triangles,
    const GNode* __restrict__ nodes, const int* __restrict__ nodeTris,
    const uint32_t* __restrict__ rank, const float* __restrict__ rnd, unsigned long long rndBase,
    float* __restrict__ kij) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int groupLane = lane & (NUM_RAYS - 1);
    const int groupBase = lane & ~(NUM_RAYS - 1);
    const unsigned groupMask = 0xffffu << groupBase;

    const size_t groupsPerBlock = blockDim.x / NUM_RAYS;
    const size_t stride = groupsPerBlock * gridDim.x;
    const size_t first = pairBegin + blockIdx.x * groupsPerBlock + threadIdx.x / NUM_RAYS;

    // All lanes of a group share the same pair, so the loop is group-uniform
    for (size_t p = first; p < pairEnd; p += stride) {
        const uint32_t i = static_cast<uint32_t>(p / n);
        const uint32_t j = static_cast<uint32_t>(p - static_cast<size_t>(i) * n);
        const Triangle triI = triangles[i];
        const Triangle triJ = triangles[j];

        if (i == j || triI.normal().dot(triJ.normal()) > 0.99f) continue;

        const float* r =
            rnd + (static_cast<unsigned long long>(rank[p]) * DRAWS_PER_PAIR - rndBase) +
            4 * groupLane;

        const Vec3 pI = pointInTriangle(triI, r[0], r[1]);
        const Vec3 pJ = pointInTriangle(triJ, r[2], r[3]);

        float contrib = ZERO;
        if (!isRayBlockedDev(pI, pJ, nodes, nodeTris, triangles, static_cast<int>(i),
                             static_cast<int>(j))) {
            const Vec3 v = pJ - pI;
            const val_t distSqr = v.squaredNorm();
            if (distSqr >= EPSILON) {
                const val_t cosPhiI = cosPhi(v, triI.normal());
                const val_t cosPhiJ = cosPhi(-v, triJ.normal());
                if (cosPhiI > ZERO && cosPhiJ > ZERO) {
                    contrib = (cosPhiI * cosPhiJ) / (PI * distSqr);
                }
            }
        }

        // Sequential (bit-exact) accumulation of the NUM_RAYS contributions
        float kijSum = ZERO;
#pragma unroll
        for (int q = 0; q < NUM_RAYS; ++q) {
            kijSum += __shfl_sync(groupMask, contrib, groupBase + q);
        }

        if (groupLane == 0) {
            kij[static_cast<size_t>(i) * n + j] = kijSum * INV_NUM_RAYS;
        }
    }
}

// ============================================================================
// Exclusive prefix sum (used to map triangle pairs onto the random stream)
// ============================================================================

constexpr int SCAN_BLOCK = 256;
constexpr int SCAN_ITEMS = 4;
constexpr int SCAN_TILE = SCAN_BLOCK * SCAN_ITEMS;

__global__ void scanTileKernel(uint32_t* __restrict__ data, uint32_t* __restrict__ blockSums,
                               size_t count) {
    __shared__ uint32_t s[SCAN_BLOCK];

    const size_t base = static_cast<size_t>(blockIdx.x) * SCAN_TILE + threadIdx.x * SCAN_ITEMS;
    uint32_t v[SCAN_ITEMS];
    uint32_t sum = 0;
#pragma unroll
    for (int k = 0; k < SCAN_ITEMS; ++k) {
        v[k] = (base + k < count) ? data[base + k] : 0u;
        sum += v[k];
    }

    s[threadIdx.x] = sum;
    __syncthreads();
    for (int off = 1; off < SCAN_BLOCK; off <<= 1) {
        const uint32_t t = (threadIdx.x >= static_cast<unsigned>(off)) ? s[threadIdx.x - off] : 0u;
        __syncthreads();
        s[threadIdx.x] += t;
        __syncthreads();
    }

    uint32_t run = s[threadIdx.x] - sum;  // exclusive prefix of this thread's items
#pragma unroll
    for (int k = 0; k < SCAN_ITEMS; ++k) {
        const uint32_t val = v[k];
        if (base + k < count) data[base + k] = run;
        run += val;
    }

    if (blockSums && threadIdx.x == SCAN_BLOCK - 1) blockSums[blockIdx.x] = s[SCAN_BLOCK - 1];
}

__global__ void scanAddKernel(uint32_t* __restrict__ data, const uint32_t* __restrict__ offsets,
                              size_t count) {
    const size_t base = static_cast<size_t>(blockIdx.x) * SCAN_TILE + threadIdx.x * SCAN_ITEMS;
    const uint32_t off = offsets[blockIdx.x];
#pragma unroll
    for (int k = 0; k < SCAN_ITEMS; ++k) {
        if (base + k < count) data[base + k] += off;
    }
}

// Scratch space needed by the multi level scan of `count` elements
static size_t scanScratchSize(size_t count) {
    size_t total = 0;
    while (count > SCAN_TILE) {
        count = (count + SCAN_TILE - 1) / SCAN_TILE;
        total += count;
    }
    return total;
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

__global__ void timeDelayKernel(uint32_t n, const Triangle* __restrict__ triangles,
                                int* __restrict__ tau) {
    const size_t total = static_cast<size_t>(n) * n;
    for (size_t p = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; p < total;
         p += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const uint32_t i = static_cast<uint32_t>(p / n);
        const uint32_t j = static_cast<uint32_t>(p - static_cast<size_t>(i) * n);
        if (i == j) {
            tau[p] = 0;
            continue;
        }
        const val_t dist = (triangles[i].center() - triangles[j].center()).norm();
        tau[p] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// One warp per receiver triangle: the 32 lanes evaluate 32 emitters in parallel
// (coalesced loads of the Kij / Tau rows) and their contributions are then
// folded into the running sum lane by lane, so the floating point summation
// order is bit-identical to the sequential reference.
__global__ void propagateKernel(uint32_t n, int t, const float* __restrict__ kij,
                                const int* __restrict__ tau, const float* __restrict__ areas,
                                const float* __restrict__ rho, const float* __restrict__ radE,
                                float* __restrict__ radB) {
    const uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    if (i >= n) return;

    const size_t rowBase = static_cast<size_t>(i) * n;
    val_t sumB = ZERO;

    for (uint32_t j0 = 0; j0 < n; j0 += 32) {
        const uint32_t j = j0 + lane;

        val_t contrib = ZERO;
        if (j < n && j != i) {
            const int tauij = tau[rowBase + j];
            const int srcTime = t - tauij;
            if (srcTime >= 0) {
                const val_t kv = kij[rowBase + j];
                if (kv > ZERO) {
                    const val_t radJ = radB[static_cast<size_t>(srcTime) * n + j];
                    if (radJ > ZERO) contrib = fminf(kv * areas[j], ONE) * radJ;
                }
            }
        }

#pragma unroll
        for (int q = 0; q < 32; ++q) {
            sumB += __shfl_sync(0xffffffffu, contrib, q);
        }
    }

    if (lane == 0) {
        radB[static_cast<size_t>(t) * n + i] = fmaf(rho[i], sumB, radE[static_cast<size_t>(t) * n + i]);
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

// One block per triangle, one thread per lag. Ties are resolved towards the
// smallest lag, reproducing the sequential "first strictly greater wins" rule.
__global__ void crossCorrelationKernel(uint32_t n, int numTimesteps,
                                       const float* __restrict__ radB, uint32_t sourceIndex,
                                       int useShared, float* __restrict__ distances) {
    extern __shared__ float sh[];
    const uint32_t i = blockIdx.x;
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);

    float* colI = sh;
    float* colS = sh + numTimesteps;

    if (useShared) {
        for (int tt = tid; tt < numTimesteps; tt += nthreads) {
            colI[tt] = radB[static_cast<size_t>(tt) * n + i];
            colS[tt] = radB[static_cast<size_t>(tt) * n + sourceIndex];
        }
        __syncthreads();
    }

    float bestVal = ZERO;
    int bestT = 0;

    // __fmul_rn / __fadd_rn keep the product rounded before the accumulation,
    // matching the reference (which the host compiler does not contract here)
    for (int t = tid; t < numTimesteps; t += nthreads) {
        val_t sum = ZERO;
        if (useShared) {
            for (int tt = t; tt < numTimesteps; ++tt) {
                sum = __fadd_rn(sum, __fmul_rn(colS[tt - t], colI[tt]));
            }
        } else {
            for (int tt = t; tt < numTimesteps; ++tt) {
                const val_t pB = radB[static_cast<size_t>(tt) * n + i];
                const val_t pS = radB[static_cast<size_t>(tt - t) * n + sourceIndex];
                sum = __fadd_rn(sum, __fmul_rn(pS, pB));
            }
        }
        if (sum > bestVal) {
            bestVal = sum;
            bestT = t;
        }
    }

    // Block-wide argmax, ties broken towards the smaller lag
    __shared__ float redVal[256];
    __shared__ int redIdx[256];
    redVal[tid] = bestVal;
    redIdx[tid] = bestT;
    __syncthreads();

    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float ov = redVal[tid + s];
            const int oi = redIdx[tid + s];
            if (ov > redVal[tid] || (ov == redVal[tid] && oi < redIdx[tid])) {
                redVal[tid] = ov;
                redIdx[tid] = oi;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        distances[i] = WAVE_SPEED * static_cast<val_t>(redVal[0] > ZERO ? redIdx[0] : 0);
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

    size_t sourceIndex;

    // Device state
    Triangle* d_triangles = nullptr;
    GNode* d_nodes = nullptr;
    int* d_nodeTris = nullptr;
    val_t* d_areas = nullptr;
    val_t* d_rho = nullptr;
    val_t* d_kij = nullptr;
    int* d_tau = nullptr;
    val_t* d_radE = nullptr;
    val_t* d_radB = nullptr;
    val_t* d_distances = nullptr;
    uint32_t* d_rank = nullptr;
    uint32_t* d_scanScratch = nullptr;
    uint32_t* d_mtState = nullptr;
    float* d_rnd[2] = {nullptr, nullptr};

    size_t chunkPairs = 0;          // Triangle pairs per random number window
    cudaStream_t streamCompute = nullptr;
    cudaStream_t streamRng = nullptr;

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

    // ---- Device side setup (mirrors the host allocations above) ----
    const size_t n = state.numTriangles;
    const size_t nn = n * n;

    std::vector<GNode> nodes;
    std::vector<int> nodeTris;
    flattenOctree(&state.octree, nodes, nodeTris);
    if (nodeTris.empty()) nodeTris.push_back(0);

    const int stackNeeded = maxTraversalStack(nodes, 0, 1);
    if (stackNeeded > MAX_TRAVERSAL_STACK) {
        fprintf(stderr, "Octree too deep for the device traversal stack (%d > %d)\n", stackNeeded,
                MAX_TRAVERSAL_STACK);
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamCreate(&state.streamCompute));
    CUDA_CHECK(cudaStreamCreate(&state.streamRng));

    CUDA_CHECK(cudaMalloc(&state.d_triangles, n * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&state.d_nodes, nodes.size() * sizeof(GNode)));
    CUDA_CHECK(cudaMalloc(&state.d_nodeTris, nodeTris.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_areas, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_rho, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_kij, nn * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_tau, nn * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_radE, timesteps * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_radB, timesteps * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_distances, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_rank, nn * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&state.d_mtState, MT_N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&state.d_scanScratch, (scanScratchSize(nn) + 1) * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(state.d_triangles, state.triangles.data(), n * sizeof(Triangle),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_nodes, nodes.data(), nodes.size() * sizeof(GNode),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_nodeTris, nodeTris.data(), nodeTris.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), n * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), n * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_radE, state.radE.data(), timesteps * n * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, timesteps * n * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.d_kij, 0, nn * sizeof(val_t)));

    // Random number window: large enough to keep the GPU busy, small enough to
    // overlap generation of the next window with the current form factor kernel
    const size_t maxChunk = 1u << 20;
    size_t chunk = std::max<size_t>(nn / 8, 1);
    chunk = std::min(chunk, maxChunk);
    chunk = std::min(chunk, nn);
    state.chunkPairs = chunk;

    const size_t rndWords = chunk * DRAWS_PER_PAIR + 2 * MT_N;
    CUDA_CHECK(cudaMalloc(&state.d_rnd[0], rndWords * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_rnd[1], rndWords * sizeof(float)));
}

void releaseSimulation(SimulationState& state) {
    cudaFree(state.d_triangles);
    cudaFree(state.d_nodes);
    cudaFree(state.d_nodeTris);
    cudaFree(state.d_areas);
    cudaFree(state.d_rho);
    cudaFree(state.d_kij);
    cudaFree(state.d_tau);
    cudaFree(state.d_radE);
    cudaFree(state.d_radB);
    cudaFree(state.d_distances);
    cudaFree(state.d_rank);
    cudaFree(state.d_scanScratch);
    cudaFree(state.d_mtState);
    cudaFree(state.d_rnd[0]);
    cudaFree(state.d_rnd[1]);
    if (state.streamCompute) cudaStreamDestroy(state.streamCompute);
    if (state.streamRng) cudaStreamDestroy(state.streamRng);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Exclusive prefix sum over the cull mask (multi level scan, in place)
static void exclusiveScan(uint32_t* d_data, size_t count, uint32_t* d_scratch,
                          cudaStream_t stream) {
    const size_t numBlocks = (count + SCAN_TILE - 1) / SCAN_TILE;
    if (numBlocks <= 1) {
        scanTileKernel<<<1, SCAN_BLOCK, 0, stream>>>(d_data, nullptr, count);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    uint32_t* d_sums = d_scratch;
    scanTileKernel<<<numBlocks, SCAN_BLOCK, 0, stream>>>(d_data, d_sums, count);
    CUDA_CHECK(cudaGetLastError());
    exclusiveScan(d_sums, numBlocks, d_scratch + numBlocks, stream);
    scanAddKernel<<<numBlocks, SCAN_BLOCK, 0, stream>>>(d_data, d_sums, count);
    CUDA_CHECK(cudaGetLastError());
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");

    const uint32_t n = static_cast<uint32_t>(state.numTriangles);
    const size_t nn = static_cast<size_t>(n) * n;
    cudaStream_t sc = state.streamCompute;
    cudaStream_t sr = state.streamRng;

    // 1) Which pairs consume random draws?
    {
        const int block = 256;
        const int grid = static_cast<int>(std::min<size_t>((nn + block - 1) / block, 65535));
        cullMaskKernel<<<grid, block, 0, sc>>>(n, state.d_triangles, state.d_rank);
    }

    // 2) Rank of every pair within the sequence of non-culled pairs
    uint32_t lastFlag = 0;
    CUDA_CHECK(cudaMemcpyAsync(&lastFlag, state.d_rank + (nn - 1), sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, sc));
    CUDA_CHECK(cudaStreamSynchronize(sc));
    exclusiveScan(state.d_rank, nn, state.d_scanScratch, sc);

    uint32_t lastRank = 0;
    CUDA_CHECK(cudaMemcpyAsync(&lastRank, state.d_rank + (nn - 1), sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, sc));
    CUDA_CHECK(cudaStreamSynchronize(sc));
    const unsigned long long activePairs = lastRank + lastFlag;

    // 3) Seed the device MT19937 exactly like RandomGenerator(42)
    {
        uint32_t hostState[MT_N];
        mtSeedState(42u, hostState);
        CUDA_CHECK(cudaMemcpyAsync(state.d_mtState, hostState, sizeof(hostState),
                                   cudaMemcpyHostToDevice, sr));
        CUDA_CHECK(cudaStreamSynchronize(sr));
    }

    // 4) Walk the pair space in windows, generating the matching slice of the
    //    random stream one window ahead on a separate stream.
    const size_t chunk = state.chunkPairs;
    const size_t numChunks = (nn + chunk - 1) / chunk;

    std::vector<uint32_t> chunkRank(numChunks + 1);
    for (size_t c = 0; c <= numChunks; ++c) {
        const size_t p = std::min(c * chunk, nn);
        if (p == nn) {
            chunkRank[c] = static_cast<uint32_t>(activePairs);
        } else {
            CUDA_CHECK(cudaMemcpyAsync(&chunkRank[c], state.d_rank + p, sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, sc));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(sc));

    std::vector<cudaEvent_t> genDone(numChunks), useDone(numChunks);
    for (size_t c = 0; c < numChunks; ++c) {
        CUDA_CHECK(cudaEventCreateWithFlags(&genDone[c], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&useDone[c], cudaEventDisableTiming));
    }

    // Word position (within the global stream) that the saved MT state refers to
    unsigned long long stateWord = 0;
    std::vector<unsigned long long> windowBase(numChunks);

    auto launchGeneration = [&](size_t c) {
        const unsigned long long endWord =
            static_cast<unsigned long long>(chunkRank[c + 1]) * DRAWS_PER_PAIR;
        windowBase[c] = stateWord;
        long long needed = static_cast<long long>(endWord - stateWord);
        const int numTwists = static_cast<int>((needed + MT_N - 1) / MT_N);
        const int snapTwist = static_cast<int>(needed / MT_N);
        if (numTwists > 0) {
            mtGenerateKernel<<<1, 640, 0, sr>>>(state.d_mtState, state.d_rnd[c & 1], numTwists,
                                                snapTwist);
        }
        stateWord += static_cast<unsigned long long>(snapTwist) * MT_N;
        CUDA_CHECK(cudaEventRecord(genDone[c], sr));
    };

    const int block = 256;
    const size_t groupsPerBlock = block / NUM_RAYS;

    launchGeneration(0);
    for (size_t c = 0; c < numChunks; ++c) {
        // Overlap: produce the next window while this one is consumed
        if (c + 1 < numChunks) {
            if (c >= 1) CUDA_CHECK(cudaStreamWaitEvent(sr, useDone[c - 1], 0));
            launchGeneration(c + 1);
        }

        const size_t pBegin = c * chunk;
        const size_t pEnd = std::min(pBegin + chunk, nn);
        const size_t groups = pEnd - pBegin;
        const int grid =
            static_cast<int>(std::min<size_t>((groups + groupsPerBlock - 1) / groupsPerBlock,
                                              1u << 20));

        CUDA_CHECK(cudaStreamWaitEvent(sc, genDone[c], 0));
        formFactorKernel<<<grid, block, 0, sc>>>(n, pBegin, pEnd, state.d_triangles, state.d_nodes,
                                                 state.d_nodeTris, state.d_rank,
                                                 state.d_rnd[c & 1], windowBase[c], state.d_kij);
        CUDA_CHECK(cudaEventRecord(useDone[c], sc));
    }

    CUDA_CHECK(cudaStreamSynchronize(sc));
    CUDA_CHECK(cudaStreamSynchronize(sr));
    CUDA_CHECK(cudaGetLastError());

    for (size_t c = 0; c < numChunks; ++c) {
        cudaEventDestroy(genDone[c]);
        cudaEventDestroy(useDone[c]);
    }

    for (size_t i = 0; i < state.numTriangles; ++i) {
        if ((i + 1) % 100 == 0 || i + 1 == state.numTriangles) {
            printf("  Progress: %zu/%zu triangles\n", i + 1, state.numTriangles);
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    const uint32_t n = static_cast<uint32_t>(state.numTriangles);
    const size_t nn = static_cast<size_t>(n) * n;
    const int block = 256;
    const int grid = static_cast<int>(std::min<size_t>((nn + block - 1) / block, 65535));
    timeDelayKernel<<<grid, block, 0, state.streamCompute>>>(n, state.d_triangles, state.d_tau);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(state.streamCompute));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    const uint32_t n = static_cast<uint32_t>(state.numTriangles);
    const int block = 128;
    const int grid = static_cast<int>((static_cast<size_t>(n) * 32 + block - 1) / block);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        propagateKernel<<<grid, block, 0, state.streamCompute>>>(
            n, static_cast<int>(t), state.d_kij, state.d_tau, state.d_areas, state.d_rho,
            state.d_radE, state.d_radB);

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(state.streamCompute));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    const uint32_t n = static_cast<uint32_t>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);

    int block = 256;
    while (block > 32 && block / 2 >= T) block /= 2;

    const size_t shBytes = 2 * static_cast<size_t>(T) * sizeof(float);
    const int useShared = shBytes <= 32768 ? 1 : 0;

    crossCorrelationKernel<<<n, block, useShared ? shBytes : 0, state.streamCompute>>>(
        n, T, state.d_radB, static_cast<uint32_t>(state.sourceIndex), useShared,
        state.d_distances);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(state.distances.data(), state.d_distances, n * sizeof(val_t),
                               cudaMemcpyDeviceToHost, state.streamCompute));
    CUDA_CHECK(cudaStreamSynchronize(state.streamCompute));
}

// Copy the bulk results back for validation (outside the timed regions)
void fetchDeviceResults(SimulationState& state) {
    const size_t n = state.numTriangles;
    CUDA_CHECK(cudaMemcpy(state.kij.data(), state.d_kij, n * n * sizeof(val_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB,
                          state.numTimesteps * n * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.tau.data(), state.d_tau, n * n * sizeof(int),
                          cudaMemcpyDeviceToHost));
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
        fetchDeviceResults(state);
        if (!validateResults(state)) {
            releaseSimulation(state);
            return 1;
        }
    }

    releaseSimulation(state);
    return 0;
}
