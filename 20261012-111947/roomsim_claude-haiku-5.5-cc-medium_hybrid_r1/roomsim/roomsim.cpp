/**
 * Room Response Simulation Benchmark
 *
 * This is a simplified implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization (hybrid MPI + OpenMP + CUDA):
 *  - MPI:    receiver rows of the N x N Kij/Tau matrices are block-distributed across
 *            ranks. Each timestep's radiosity row is exchanged with MPI_Allgatherv, and
 *            the distances are gathered the same way.
 *  - CUDA:   each rank drives one GPU. Ray casting for the form factors runs on the GPU
 *            against a flattened octree, and the radiosity update runs as a kernel.
 *  - OpenMP: host-side loops (time delays, sample-stream bookkeeping, cross-correlation).
 *
 * The Monte Carlo sample stream is the same single Mersenne Twister sequence as in the
 * sequential program. Culled pairs consume no samples, so each rank first advances the
 * generator to the position its first row reaches. The results therefore match the
 * sequential program.
 */

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdarg>
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

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

// Host and device annotation: the file is compiled by nvcc, and vector math runs on both sides
#define HD __host__ __device__

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;
constexpr int SAMPLES_PER_PAIR = NUM_RAYS * 4;  // Two points per ray, two uniforms per point
constexpr size_t TARGET_BATCH_BYTES = size_t(64) << 20;  // Host/device staging size per batch
constexpr int THREADS_PER_BLOCK = 128;
constexpr int NODE_STACK_SIZE = 128;      // Traversal stack for the octree on the device

HD inline val_t vsqrt(val_t x) { return sqrtf(x); }

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
    HD val_t norm() const { return vsqrt(squaredNorm()); }
    HD Vec3 normalized() const {
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

// Device-side triangle: vertices plus the precomputed (inward-facing) normal
struct DevTri {
    Vec3 a, b, c, n;
};

// Device-side octree node. Leaves have triCount > 0 and reference triIdx[triBegin, triBegin + triCount)
struct DevNode {
    Vec3 center, halfExtent;
    int32_t child[8];
    uint32_t triBegin, triCount;
};

// ============================================================================
// CUDA and MPI Helpers
// ============================================================================

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Rank identity (set in main)
static int gRank = 0;
static int gSize = 1;

// Print only from the root rank
__attribute__((format(printf, 1, 2))) void rootPrintf(const char* fmt, ...) {
    if (gRank != 0) return;
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

template <typename T>
struct CudaFreeDeleter {
    void operator()(T* p) const { cudaFree(p); }
};

template <typename T>
struct CudaFreeHostDeleter {
    void operator()(T* p) const { cudaFreeHost(p); }
};

template <typename T>
using DevBuf = std::unique_ptr<T, CudaFreeDeleter<T>>;

template <typename T>
using PinnedBuf = std::unique_ptr<T, CudaFreeHostDeleter<T>>;

template <typename T>
DevBuf<T> allocDevice(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)));
    return DevBuf<T>(p);
}

template <typename T>
PinnedBuf<T> allocPinned(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMallocHost(&p, std::max<size_t>(count, 1) * sizeof(T)));
    return PinnedBuf<T>(p);
}

template <typename T>
DevBuf<T> uploadDevice(const std::vector<T>& host) {
    DevBuf<T> dev = allocDevice<T>(host.size());
    if (!host.empty()) {
        CUDA_CHECK(cudaMemcpy(dev.get(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    return dev;
}

// Contiguous row block [begin, end) of an N-row problem owned by a rank
struct RowRange {
    size_t begin, end;
};

RowRange rowRangeOf(int rank, size_t N) {
    return {N * static_cast<size_t>(rank) / gSize, N * static_cast<size_t>(rank + 1) / gSize};
}

// Gathers every rank's row block of a length-N array in place across all ranks
template <typename T>
void allgatherRowBlocks(T* data, size_t N, MPI_Datatype type) {
    std::vector<int> counts(gSize);
    std::vector<int> displs(gSize);
    for (int r = 0; r < gSize; ++r) {
        RowRange rr = rowRangeOf(r, N);
        counts[r] = static_cast<int>(rr.end - rr.begin);
        displs[r] = static_cast<int>(rr.begin);
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, data, counts.data(), displs.data(), type,
                   MPI_COMM_WORLD);
}

// Binds this rank to a GPU, round-robin over the ranks that share the node
void selectDevice() {
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, gRank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
}

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

    // Flatten into index-based nodes for device traversal. Node 0 is the root.
    uint32_t flatten(std::vector<DevNode>& nodes, std::vector<uint32_t>& triIdx) const {
        const uint32_t self = static_cast<uint32_t>(nodes.size());
        nodes.emplace_back();

        DevNode nd{};
        nd.center = center;
        nd.halfExtent = halfExtent;
        std::fill(std::begin(nd.child), std::end(nd.child), -1);
        if (!triangleIndices.empty()) {
            nd.triBegin = static_cast<uint32_t>(triIdx.size());
            nd.triCount = static_cast<uint32_t>(triangleIndices.size());
            for (size_t idx : triangleIndices) triIdx.push_back(static_cast<uint32_t>(idx));
        } else {
            for (int i = 0; i < 8; ++i) {
                if (children[i]) nd.child[i] = static_cast<int32_t>(children[i]->flatten(nodes, triIdx));
            }
        }
        nodes[self] = nd;
        return self;
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
// Random Number Generation
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
    // Advance the stream by n engine outputs (each rand() consumes exactly one)
    void skip(unsigned long long n) { rng.discard(n); }
};

// ============================================================================
// Device Geometry: Visibility and Form Factors
// ============================================================================

// Point in triangle from two uniforms, identical to the sequential randomPointInTriangle
HD inline Vec3 samplePointInTriangle(const DevTri& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// Möller-Trumbore ray-triangle intersection
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

// Slab-style test of the segment p1-p2 against a node's bounding box
__device__ inline bool segmentHitsNode(const DevNode& nd, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - nd.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > nd.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > nd.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > nd.halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > nd.halfExtent.y * ad.z + nd.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > nd.halfExtent.z * ad.x + nd.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > nd.halfExtent.x * ad.y + nd.halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle (octree-accelerated)
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const DevNode* nodes, const uint32_t* triIdx, const DevTri* tris,
                             uint32_t srcTriIdx, uint32_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    // The root is always visited. Children are only pushed after their box test.
    uint32_t stack[NODE_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const DevNode& nd = nodes[stack[--sp]];
        if (nd.triCount > 0) {
            for (uint32_t k = 0; k < nd.triCount; ++k) {
                uint32_t idx = triIdx[nd.triBegin + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const DevTri& tri = tris[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;  // Ray is blocked
                }
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int32_t ch = nd.child[c];
                if (ch >= 0 && segmentHitsNode(nodes[ch], from, to)) {
                    stack[sp++] = static_cast<uint32_t>(ch);
                }
            }
        }
    }
    return false;
}

// Cosine of angle between vector and surface normal, clamped at zero
__device__ inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t d = v.dot(normal) / vNorm;
    return (ZERO < d) ? d : ZERO;
}

// ============================================================================
// Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;
    size_t sourceIndex;
    val_t reflectivity;

    // Receiver rows [rowBegin, rowEnd) of the Kij/Tau matrices owned by this rank
    size_t rowBegin;
    size_t rowEnd;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix), replicated on all ranks
    std::vector<val_t> distances;   // Computed distances from source, replicated on all ranks

    Octree octree;                  // Spatial acceleration structure (host build)

    // Device-resident data. Local matrices use layout [j * nLoc + il] for receiver row il
    DevBuf<DevTri> trisD;
    DevBuf<val_t> areasD;
    DevBuf<DevNode> nodesD;
    DevBuf<uint32_t> triIdxD;
    DevBuf<int> tauD;               // Time delays (local rows)
    DevBuf<val_t> kijD;             // Form factors (local rows)
    DevBuf<val_t> radBD;            // Reflected radiosity (T x N matrix)
    DevBuf<val_t> radED;            // Emission radiosity (T x N matrix)

    size_t localRows() const { return rowEnd - rowBegin; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    const size_t N = state.triangles.size();
    state.numTriangles = N;
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % N;
    state.reflectivity = reflectivity;

    RowRange rows = rowRangeOf(gRank, N);
    state.rowBegin = rows.begin;
    state.rowEnd = rows.end;
    const size_t nLoc = state.localRows();

    rootPrintf("Generated icosphere mesh with %zu triangles\n", N);

    // Build octree for spatial acceleration and flatten it for the device
    rootPrintf("Building octree...\n");
    state.octree.build(state.triangles);
    std::vector<DevNode> nodes;
    std::vector<uint32_t> triIdx;
    state.octree.flatten(nodes, triIdx);

    std::vector<DevTri> devTris(N);
    for (size_t i = 0; i < N; ++i) {
        const Triangle& t = state.triangles[i];
        devTris[i] = {t.a, t.b, t.c, t._normal};
    }

    // Initialize areas
    state.areas.resize(N);
    for (size_t i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Emission: source is active for first half of timesteps
    std::vector<val_t> radE(timesteps * N, ZERO);
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        radE[t * N + state.sourceIndex] = 1.0f;
    }

    state.radB.assign(timesteps * N, ZERO);
    state.distances.assign(N, ZERO);

    state.trisD = uploadDevice(devTris);
    state.areasD = uploadDevice(state.areas);
    state.nodesD = uploadDevice(nodes);
    state.triIdxD = uploadDevice(triIdx);
    state.radED = uploadDevice(radE);
    state.radBD = uploadDevice(state.radB);
    state.tauD = allocDevice<int>(nLoc * N);
    state.kijD = allocDevice<val_t>(nLoc * N);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    rootPrintf("Computing time delays (Tau)...\n");
    const size_t N = state.numTriangles;
    const size_t nLoc = state.localRows();

    std::vector<int> tau(nLoc * N, 0);
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < N; ++j) {
        for (size_t il = 0; il < nLoc; ++il) {
            size_t i = state.rowBegin + il;
            if (i == j) continue;
            float dist = (state.triangles[i].center() - state.triangles[j].center()).norm();
            tau[j * nLoc + il] = static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
        }
    }

    if (!tau.empty()) {
        CUDA_CHECK(cudaMemcpy(state.tauD.get(), tau.data(), tau.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
}

// True if the pair (receiver i, emitter j) is evaluated, i.e. not culled and not the same triangle.
// Only evaluated pairs consume Monte Carlo samples.
bool consumesSamples(const std::vector<Triangle>& tris, size_t i, size_t j) {
    return i != j && !(tris[i].normal().dot(tris[j].normal()) > 0.99f);
}

// Draws the samples for each evaluated pair of a row batch, in the sequential program's order
// (row-major, culled pairs consume nothing). Slots map each pair to its sample block, or -1.
size_t drawBatchSamples(RandomGenerator& rng, const std::vector<Triangle>& tris, size_t firstRow,
                        size_t rows, int* slots, val_t* samples) {
    const size_t N = tris.size();
    size_t drawn = 0;
    for (size_t bl = 0; bl < rows; ++bl) {
        const size_t i = firstRow + bl;
        for (size_t j = 0; j < N; ++j) {
            int& slot = slots[bl * N + j];
            if (!consumesSamples(tris, i, j)) {
                slot = -1;
                continue;
            }
            slot = static_cast<int>(drawn);
            val_t* block = samples + drawn * SAMPLES_PER_PAIR;
            for (int k = 0; k < SAMPLES_PER_PAIR; ++k) block[k] = rng.rand();
            ++drawn;
        }
    }
    return drawn;
}

// Form factor for one evaluated pair: the device version of the sequential computeKij
__device__ val_t computeKijDevice(const DevTri* tris, const DevNode* nodes, const uint32_t* triIdx,
                                  uint32_t i, uint32_t j, const val_t* smp) {
    const DevTri& triI = tris[i];
    const DevTri& triJ = tris[j];

    val_t kij = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        const val_t* q = smp + 4 * r;
        Vec3 pI = samplePointInTriangle(triI, q[0], q[1]);
        Vec3 pJ = samplePointInTriangle(triJ, q[2], q[3]);

        if (isRayBlocked(pI, pJ, nodes, triIdx, tris, i, j)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.n);
        val_t cosPhiJ = cosPhi(-v, triJ.n);

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    return kij * INV_NUM_RAYS;
}

// One thread per (batch row, emitter) pair. Writes into the local transposed Kij matrix.
__global__ void formFactorKernel(const DevTri* tris, const DevNode* nodes, const uint32_t* triIdx,
                                 const int* slots, const val_t* samples,
                                 size_t N, size_t firstRow, size_t firstLocal, size_t batchRows,
                                 size_t nLoc, val_t* kijT) {
    const size_t g = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (g >= batchRows * N) return;

    const size_t bl = g / N;
    const size_t j = g % N;
    const size_t i = firstRow + bl;
    const size_t il = firstLocal + bl;

    const int slot = slots[g];
    val_t kij = ZERO;
    if (slot >= 0) {
        kij = computeKijDevice(tris, nodes, triIdx, static_cast<uint32_t>(i), static_cast<uint32_t>(j),
                               samples + static_cast<size_t>(slot) * SAMPLES_PER_PAIR);
    }
    kijT[j * nLoc + il] = kij;
}

// Computes the form factors of the local rows, streaming the RNG samples in batches.
// Sample generation for batch b+1 overlaps the GPU work of batch b.
void computeFormFactors(SimulationState& state) {
    const size_t N = state.numTriangles;
    const size_t nLoc = state.localRows();
    rootPrintf("Computing form factors (Kij)...\n");

    // Every rank must start the shared sample stream at the position its first row reaches.
    // Culled pairs consume no samples, so the count per row is needed for all rows before it.
    std::vector<unsigned long long> samplesPerRow(N, 0);
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t il = 0; il < nLoc; ++il) {
        const size_t i = state.rowBegin + il;
        unsigned long long count = 0;
        for (size_t j = 0; j < N; ++j) {
            if (consumesSamples(state.triangles, i, j)) ++count;
        }
        samplesPerRow[i] = count;
    }
    allgatherRowBlocks(samplesPerRow.data(), N, MPI_UNSIGNED_LONG_LONG);

    unsigned long long samplesBefore = 0;
    for (size_t i = 0; i < state.rowBegin; ++i) samplesBefore += samplesPerRow[i];

    RandomGenerator rng(42);
    rng.skip(samplesBefore * SAMPLES_PER_PAIR);

    // Batch geometry: bounded host and device staging memory
    const size_t bytesPerRow = N * (SAMPLES_PER_PAIR * sizeof(val_t) + sizeof(int));
    const size_t rowsPerBatch = std::max<size_t>(1, std::min(nLoc, TARGET_BATCH_BYTES / bytesPerRow));
    const size_t numBatches = (nLoc + rowsPerBatch - 1) / rowsPerBatch;
    const size_t batchCap = rowsPerBatch * N;

    PinnedBuf<int> slotsH[2] = {allocPinned<int>(batchCap), allocPinned<int>(batchCap)};
    PinnedBuf<val_t> samplesH[2] = {allocPinned<val_t>(batchCap * SAMPLES_PER_PAIR),
                                    allocPinned<val_t>(batchCap * SAMPLES_PER_PAIR)};
    DevBuf<int> slotsD = allocDevice<int>(batchCap);
    DevBuf<val_t> samplesD = allocDevice<val_t>(batchCap * SAMPLES_PER_PAIR);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    size_t drawnCur = 0;
    if (numBatches > 0) {
        const size_t rows = std::min(rowsPerBatch, nLoc);
        drawnCur = drawBatchSamples(rng, state.triangles, state.rowBegin, rows, slotsH[0].get(), samplesH[0].get());
    }

    for (size_t b = 0; b < numBatches; ++b) {
        const size_t cur = b & 1;
        const size_t nxt = cur ^ 1;
        const size_t firstLocal = b * rowsPerBatch;
        const size_t rows = std::min(rowsPerBatch, nLoc - firstLocal);
        const size_t firstRow = state.rowBegin + firstLocal;

        CUDA_CHECK(cudaMemcpyAsync(slotsD.get(), slotsH[cur].get(), rows * N * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        if (drawnCur > 0) {
            CUDA_CHECK(cudaMemcpyAsync(samplesD.get(), samplesH[cur].get(),
                                       drawnCur * SAMPLES_PER_PAIR * sizeof(val_t),
                                       cudaMemcpyHostToDevice, stream));
        }

        const size_t work = rows * N;
        const int blocks = static_cast<int>((work + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        formFactorKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            state.trisD.get(), state.nodesD.get(), state.triIdxD.get(),
            slotsD.get(), samplesD.get(), N, firstRow, firstLocal, rows, nLoc, state.kijD.get());
        CUDA_CHECK(cudaGetLastError());

        // Host-side sampling for the next batch overlaps the kernel
        size_t drawnNxt = 0;
        if (b + 1 < numBatches) {
            const size_t nextRows = std::min(rowsPerBatch, nLoc - (b + 1) * rowsPerBatch);
            drawnNxt = drawBatchSamples(rng, state.triangles, state.rowBegin + (b + 1) * rowsPerBatch,
                                        nextRows, slotsH[nxt].get(), samplesH[nxt].get());
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        drawnCur = drawnNxt;

        rootPrintf("  Progress: %zu/%zu triangles\n", firstLocal + rows, nLoc);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// One timestep of radiosity for the local receiver rows. Thread per receiver row.
__global__ void simulationStepKernel(int t, size_t N, size_t nLoc, size_t rowBegin, val_t rho,
                                     const val_t* areas, const int* tau, const val_t* kijT,
                                     const val_t* radE, val_t* radB) {
    const size_t il = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (il >= nLoc) return;

    const size_t i = rowBegin + il;
    val_t sumB = ZERO;

    for (size_t j = 0; j < N; ++j) {
        if (i == j) continue;

        const size_t k = j * nLoc + il;
        const int tauij = tau[k];

        // Skip if wave hasn't yet propagated from j to i
        if (t < tauij) continue;

        const val_t kij = kijT[k];
        if (kij <= ZERO) continue;

        // Get radiosity from source triangle at time when emission occurred
        const size_t srcTime = static_cast<size_t>(t - tauij);
        const val_t radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        // Accumulate contribution: form factor * area * source radiosity
        val_t x = kij * areas[j];
        x = (ONE < x) ? ONE : x;
        sumB += x * radJ;
    }

    // Update radiosity: reflection + emission
    radB[static_cast<size_t>(t) * N + i] = rho * sumB + radE[static_cast<size_t>(t) * N + i];
}

// Counts the entries of the local Kij block that exceed EPSILON
__global__ void countNonZeroKernel(const val_t* kij, size_t n, unsigned long long* count) {
    unsigned long long local = 0;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t k = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; k < n; k += stride) {
        if (kij[k] > EPSILON) ++local;
    }
    atomicAdd(count, local);
}

// Propagates radiosity over the timesteps. Each step's row is exchanged across ranks.
void runSimulation(SimulationState& state) {
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const size_t nLoc = state.localRows();
    rootPrintf("Running wave propagation simulation...\n");

    for (size_t t = 0; t < T; ++t) {
        if (nLoc > 0) {
            const int blocks = static_cast<int>((nLoc + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            simulationStepKernel<<<blocks, THREADS_PER_BLOCK>>>(
                static_cast<int>(t), N, nLoc, state.rowBegin, state.reflectivity,
                state.areasD.get(), state.tauD.get(), state.kijD.get(),
                state.radED.get(), state.radBD.get());
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(state.radB.data() + t * N + state.rowBegin,
                                  state.radBD.get() + t * N + state.rowBegin,
                                  nLoc * sizeof(val_t), cudaMemcpyDeviceToHost));
        }

        allgatherRowBlocks(state.radB.data() + t * N, N, MPI_FLOAT);
        CUDA_CHECK(cudaMemcpy(state.radBD.get() + t * N, state.radB.data() + t * N,
                              N * sizeof(val_t), cudaMemcpyHostToDevice));

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            rootPrintf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    rootPrintf("Computing distances via cross-correlation...\n");
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const size_t nLoc = state.localRows();

    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t il = 0; il < nLoc; ++il) {
        const size_t i = state.rowBegin + il;
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < T; ++tt) {
                val_t pB = state.radB[tt * N + i];
                val_t pS = state.radB[(tt - t) * N + state.sourceIndex];
                sum += pS * pB;
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    allgatherRowBlocks(state.distances.data(), N, MPI_FLOAT);
}

// Number of Kij entries above EPSILON across all ranks
unsigned long long countNonZeroFormFactors(const SimulationState& state) {
    const size_t n = state.localRows() * state.numTriangles;
    DevBuf<unsigned long long> countD = allocDevice<unsigned long long>(1);
    CUDA_CHECK(cudaMemset(countD.get(), 0, sizeof(unsigned long long)));
    if (n > 0) {
        countNonZeroKernel<<<256, 256>>>(state.kijD.get(), n, countD.get());
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned long long local = 0;
    CUDA_CHECK(cudaMemcpy(&local, countD.get(), sizeof(local), cudaMemcpyDeviceToHost));

    MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return local;
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, unsigned long long nonZeroKijGlobal) {
    rootPrintf("\nValidation:\n");

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
            rootPrintf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            rootPrintf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    rootPrintf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    rootPrintf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    rootPrintf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        rootPrintf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
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

    rootPrintf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        rootPrintf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    const size_t kijCount = state.numTriangles * state.numTriangles;
    int nonZeroKij = static_cast<int>(nonZeroKijGlobal);
    rootPrintf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
               nonZeroKij, kijCount,
               100.0f * nonZeroKij / static_cast<val_t>(kijCount));

    if (nonZeroKij == 0) {
        rootPrintf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    rootPrintf("  Validation: PASSED\n");
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
    rootPrintf("Usage: %s [options]\n", progName);
    rootPrintf("Options:\n");
    rootPrintf("  -n <num>     Target number of triangles (default: 320)\n");
    rootPrintf("               Actual count will be rounded to nearest icosphere level:\n");
    rootPrintf("               20, 80, 320, 1280, 5120, 20480\n");
    rootPrintf("  -t <num>     Number of timesteps (default: 50)\n");
    rootPrintf("  -s <num>     Source triangle index (default: 0)\n");
    rootPrintf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    rootPrintf("  -v           Enable validation\n");
    rootPrintf("  -o           Print results for external validation\n");
    rootPrintf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &gRank);
    MPI_Comm_size(MPI_COMM_WORLD, &gSize);

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
            MPI_Finalize();
            return 0;
        } else {
            rootPrintf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    rootPrintf("Room Response Simulation Benchmark\n");
    rootPrintf("===================================\n");
    rootPrintf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    rootPrintf("Timesteps: %d\n", timesteps);
    rootPrintf("Source triangle: %d\n", sourceIdx);
    rootPrintf("Reflectivity: %.2f\n", reflectivity);
    rootPrintf("Validation: %s\n", validate ? "enabled" : "disabled");
    rootPrintf("\n");

    selectDevice();

    int exitCode = 0;
    {
        // Initialize
        SimulationState state;
        initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                             static_cast<size_t>(sourceIdx), reflectivity);

        rootPrintf("\n");

        // Precomputation
        MPI_Barrier(MPI_COMM_WORLD);
        auto startPre = std::chrono::high_resolution_clock::now();

        computeTimeDelays(state);
        computeFormFactors(state);

        MPI_Barrier(MPI_COMM_WORLD);
        auto endPre = std::chrono::high_resolution_clock::now();
        auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

        rootPrintf("Precomputation time: %ld ms\n", preDuration);
        rootPrintf("\n");

        // Simulation
        MPI_Barrier(MPI_COMM_WORLD);
        auto startSim = std::chrono::high_resolution_clock::now();

        runSimulation(state);

        MPI_Barrier(MPI_COMM_WORLD);
        auto endSim = std::chrono::high_resolution_clock::now();
        auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

        rootPrintf("Simulation time: %ld ms\n", simDuration);
        rootPrintf("\n");

        // Distance computation
        MPI_Barrier(MPI_COMM_WORLD);
        auto startDist = std::chrono::high_resolution_clock::now();

        computeDistances(state);

        MPI_Barrier(MPI_COMM_WORLD);
        auto endDist = std::chrono::high_resolution_clock::now();
        auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

        rootPrintf("Distance computation time: %ld ms\n", distDuration);
        rootPrintf("\n");

        // Total time
        long totalTime = preDuration + simDuration + distDuration;
        rootPrintf("Total computation time: %ld ms\n", totalTime);

        // Performance metrics
        size_t n = state.numTriangles;
        size_t t = state.numTimesteps;
        double kijOps = static_cast<double>(n * n);
        double simOps = static_cast<double>(n * n * t);
        double distOps = static_cast<double>(n * t * t);

        rootPrintf("\nPerformance:\n");
        rootPrintf("  Triangles: %zu\n", n);
        rootPrintf("  Timesteps: %zu\n", t);
        rootPrintf("  Form factor computations: %.2e\n", kijOps);
        rootPrintf("  Simulation operations: %.2e\n", simOps);
        rootPrintf("  Distance computations: %.2e\n", distOps);
        rootPrintf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        // Memory usage
        size_t memKij = n * n * sizeof(val_t);
        size_t memTau = n * n * sizeof(int);
        size_t memRad = 2 * t * n * sizeof(val_t);
        size_t totalMem = memKij + memTau + memRad;
        rootPrintf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

        // Hash
        uint64_t hash = computeHash(state);
        rootPrintf("  Result hash: %016lX\n", hash);
        rootPrintf("\n");

        // Print results for external validation
        if (printResults && gRank == 0) {
            // Convert distances to double for output
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }

        // Validation
        if (validate) {
            unsigned long long nonZeroKij = countNonZeroFormFactors(state);
            if (!validateResults(state, nonZeroKij)) {
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
