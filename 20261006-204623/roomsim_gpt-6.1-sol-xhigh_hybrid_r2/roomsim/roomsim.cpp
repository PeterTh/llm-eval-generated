/**
 * Room Response Simulation Benchmark
 * 
 * This is an MPI/OpenMP/CUDA implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
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
#include <cstdarg>
#include <climits>
#include <cfloat>
#include <stdexcept>
#include <string>
#include <mpi.h>
#ifdef ROOMSIM_MPI_EXT_CUDA_QUERY
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>
#include <vector>

#include "../common/results_output.hpp"

// MPI calls stay on the initializing thread (MPI_THREAD_FUNNELED).
static int worldRank = 0;
static int worldSize = 1;
static bool deviceMPI = false;

void mpiCheck(int error) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(error, message, &length);
    throw std::runtime_error(std::string(message, length));
}

void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void rootPrintf(const char* format, ...) {
    if (worldRank != 0) return;
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

struct ParallelRuntime {
    ParallelRuntime(int& argc, char**& argv) {
        int provided = 0;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        if (provided < MPI_THREAD_FUNNELED) {
            fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Honor OMP_NUM_THREADS; otherwise share the available CPUs among local ranks.
        MPI_Comm local;
        mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                    worldRank, MPI_INFO_NULL, &local));
        int localSize = 1;
        mpiCheck(MPI_Comm_size(local, &localSize));
        if (!std::getenv("OMP_NUM_THREADS"))
            omp_set_num_threads(std::max(1, std::min(16, omp_get_num_procs() / localSize)));
        mpiCheck(MPI_Comm_free(&local));
    }
    ~ParallelRuntime() { MPI_Finalize(); }

    void selectDevice() {
        MPI_Comm local;
        mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                    worldRank, MPI_INFO_NULL, &local));
        int localRank = 0;
        mpiCheck(MPI_Comm_rank(local, &localRank));
        mpiCheck(MPI_Comm_free(&local));
        int devices = 0;
        cudaCheck(cudaGetDeviceCount(&devices));
        if (!devices) throw std::runtime_error("roomsim requires a CUDA device on every rank");
        cudaCheck(cudaSetDevice(localRank % devices));
        cudaCheck(cudaFree(nullptr));
        int localSupport = 0, globalSupport = 0;
#ifdef ROOMSIM_MPI_CUDA_QUERY
        localSupport = MPIX_Query_cuda_support() != 0;
#endif
        mpiCheck(MPI_Allreduce(&localSupport, &globalSupport, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        deviceMPI = globalSupport != 0;
        rootPrintf("MPI ranks: %d, OpenMP threads per rank: %d, CUDA enabled\n",
                   worldSize, omp_get_max_threads());
    }
};

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    size_t count = 0;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data) cudaFree(data); }
    void resize(size_t n) {
        if (data) cudaCheck(cudaFree(data));
        data = nullptr;
        count = n;
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), std::max(n, size_t(1)) * sizeof(T)));
    }
    void upload(const T* src, size_t n) {
        if (n) cudaCheck(cudaMemcpy(data, src, n * sizeof(T), cudaMemcpyHostToDevice));
    }
};

template<class T> struct PinnedBuffer {
    T* data = nullptr;
    PinnedBuffer() = default;
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    ~PinnedBuffer() { if (data) cudaFreeHost(data); }
    void resize(size_t n) {
        if (data) cudaCheck(cudaFreeHost(data));
        data = nullptr;
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&data), std::max(n, size_t(1)) * sizeof(T)));
    }
};

#define HD __host__ __device__

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
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    HD Triangle() = default;
    HD Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    HD Vec3 center() const { return (a + b + c) / 3.0f; }
    HD Vec3 normal() const { return _normal; }

    HD val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    HD bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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
    void buildNode(const std::vector<size_t>& indices, const Vec3& nodeMin, const Vec3& nodeMax, int depth = 0) {
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
                #pragma omp task firstprivate(i, childMin, childMax, depth) shared(childIndices) if(indices.size() > 256 && depth < 3)
                {
                    children[i]->buildNode(childIndices[i], childMin, childMax, depth + 1);
                }
            }
        }
        #pragma omp taskwait
    }

public:
    // Check if a ray intersects this node's bounding box
    bool rayIntersectsBox(const Vec3& p1, const Vec3& p2) const {
        Vec3 d = (p2 - p1) * 0.5f;
        Vec3 c = p1 + d - center;
        Vec3 ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};

        if (std::abs(c.x) > halfExtent.x + ad.x) return false;
        if (std::abs(c.y) > halfExtent.y + ad.y) return false;
        if (std::abs(c.z) > halfExtent.z + ad.z) return false;

        if (std::abs(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
        if (std::abs(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
        if (std::abs(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

        return true;
    }

    // Apply a function to all triangles potentially intersecting the ray
    // Returns true if the function returns true for any triangle
    template<typename Func>
    bool applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const {
        // If leaf node, check triangles directly
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

        // Otherwise, descend to children
        for (int i = 0; i < 8; ++i) {
            if (children[i] && children[i]->rayIntersectsBox(p1, p2)) {
                if (children[i]->applyToTris(p1, p2, func)) return true;
            }
        }
        return false;
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
        triangles.resize(faces.size());
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < faces.size(); ++i) {
            const auto& face = faces[i];
            triangles[i] = Triangle(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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
// Room Response Simulation State
// ============================================================================

// Stackless, preorder octree. escape skips an entire rejected subtree.
struct FlatNode {
    Vec3 center, halfExtent;
    idx_t first, count, escape;
};

void flattenOctree(const Octree& tree, std::vector<FlatNode>& nodes,
                   std::vector<idx_t>& indices) {
    size_t n = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, static_cast<idx_t>(indices.size()),
                     static_cast<idx_t>(tree.triangleIndices.size()), 0});
    for (size_t i : tree.triangleIndices) indices.push_back(static_cast<idx_t>(i));
    for (const auto& child : tree.children)
        if (child) flattenOctree(*child, nodes, indices);
    nodes[n].escape = static_cast<idx_t>(nodes.size());
}

HD bool intersectsBox(const FlatNode& node, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    const Vec3& h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y ||
        fabsf(c.z) > h.z + ad.z) return false;
    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;
    return true;
}

__device__ bool isRayBlocked(const Vec3& from, const Vec3& to, const Triangle* triangles,
                             const FlatNode* nodes, const idx_t* indices,
                             idx_t src, idx_t dst) {
    Vec3 dir = to - from;
    val_t length = dir.norm();
    if (length < EPSILON) return true;
    dir = dir / length;
    idx_t node = 0;
    while (node < nodes[0].escape) {
        const FlatNode& current = nodes[node];
        // The original traversal always enters the root without a box test.
        if (node != 0 && !intersectsBox(current, from, to)) {
            node = current.escape;
            continue;
        }
        for (idx_t k = 0; k < current.count; ++k) {
            idx_t index = indices[current.first + k];
            if (index == src || index == dst) continue;
            const Triangle& tri = triangles[index];
            val_t distance = rayTriangleIntersect(from, dir, tri.a, tri.b, tri.c);
            if (distance > EPSILON && distance < length - EPSILON) return true;
        }
        ++node;
    }
    return false;
}

// Raw MT19937 checkpoints avoid changing the sequential benchmark's random
// stream. Each checkpoint spans 624 pairs * 64 draws, exactly 64 twists.
constexpr size_t RNG_WORDS = 624;
constexpr size_t RNG_PAIRS = 624;
constexpr size_t DRAWS_PER_PAIR = 4 * NUM_RAYS;
constexpr size_t BATCH_CHUNKS = 64;
constexpr size_t BATCH_PAIRS = BATCH_CHUNKS * RNG_PAIRS;

struct RawMT19937 {
    uint32_t words[RNG_WORDS];
    RawMT19937() {
        words[0] = 42;
        for (uint32_t i = 1; i < RNG_WORDS; ++i)
            words[i] = 1812433253U * (words[i - 1] ^ (words[i - 1] >> 30)) + i;
        twist();
    }
    void twist() {
        for (size_t i = 0; i < 227; ++i) {
            uint32_t y = (words[i] & 0x80000000U) | (words[i + 1] & 0x7fffffffU);
            words[i] = words[i + 397] ^ (y >> 1) ^ ((y & 1) * 0x9908b0dfU);
        }
        for (size_t i = 227; i < 623; ++i) {
            uint32_t y = (words[i] & 0x80000000U) | (words[i + 1] & 0x7fffffffU);
            words[i] = words[i - 227] ^ (y >> 1) ^ ((y & 1) * 0x9908b0dfU);
        }
        uint32_t y = (words[623] & 0x80000000U) | (words[0] & 0x7fffffffU);
        words[623] = words[396] ^ (y >> 1) ^ ((y & 1) * 0x9908b0dfU);
    }
    void advanceChunk() {
        for (size_t i = 0; i < DRAWS_PER_PAIR; ++i) twist();
    }
};

// Jumping the linear MT transition removes the global O(N^2) prefix scan at
// large rank counts. Berlekamp-Massey derives its 19937-degree characteristic
// polynomial once; binary polynomial exponentiation computes x^steps mod P.
// This preserves the original MT stream exactly without a new RNG dependency.
constexpr unsigned MT_DEGREE = 19937;
constexpr size_t MT_POLY_WORDS = MT_DEGREE / 64 + 1;
using MTPolynomial = std::array<uint64_t, MT_POLY_WORDS>;

void xorShifted(MTPolynomial& destination, const MTPolynomial& source, unsigned shift) {
    unsigned word = shift / 64, bit = shift % 64;
    for (size_t i = word; i < MT_POLY_WORDS; ++i) {
        uint64_t value = source[i - word] << bit;
        if (bit && i > word) value |= source[i - word - 1] >> (64 - bit);
        destination[i] ^= value;
    }
}

MTPolynomial mtCharacteristic() {
    MTPolynomial current{}, previous{}, recent{};
    current[0] = previous[0] = 1;
    unsigned degree = 0, shift = 1;
    RawMT19937 sequence;
    for (unsigned step = 0; step < 2 * MT_DEGREE; ++step) {
        uint64_t carry = sequence.words[0] & 1;
        for (size_t i = 0; i < MT_POLY_WORDS; ++i) {
            uint64_t nextCarry = recent[i] >> 63;
            recent[i] = (recent[i] << 1) | carry;
            carry = nextCarry;
        }
        unsigned discrepancy = 0;
        for (size_t i = 0; i <= degree / 64; ++i)
            discrepancy ^= static_cast<unsigned>(__builtin_parityll(current[i] & recent[i]));
        if (discrepancy) {
            MTPolynomial saved = current;
            xorShifted(current, previous, shift);
            if (2 * degree <= step) {
                degree = step + 1 - degree;
                previous = saved;
                shift = 1;
            } else ++shift;
        } else ++shift;
        sequence.twist();
    }
    if (degree != MT_DEGREE) throw std::runtime_error("MT19937 characteristic polynomial has unexpected degree");
    MTPolynomial characteristic{};
    for (unsigned i = 0; i <= MT_DEGREE; ++i) {
        if ((current[i / 64] >> (i % 64)) & 1) {
            unsigned reverse = MT_DEGREE - i;
            characteristic[reverse / 64] |= uint64_t(1) << (reverse % 64);
        }
    }
    return characteristic;
}

uint64_t interleaveBits(uint64_t value) {
    value = (value | (value << 16)) & 0x0000ffff0000ffffULL;
    value = (value | (value << 8)) & 0x00ff00ff00ff00ffULL;
    value = (value | (value << 4)) & 0x0f0f0f0f0f0f0f0fULL;
    value = (value | (value << 2)) & 0x3333333333333333ULL;
    return (value | (value << 1)) & 0x5555555555555555ULL;
}

MTPolynomial squareModulo(const MTPolynomial& polynomial, const MTPolynomial& characteristic) {
    std::array<uint64_t, 2 * MT_POLY_WORDS> squared{};
    for (size_t i = 0; i < MT_POLY_WORDS; ++i) {
        squared[2 * i] = interleaveBits(polynomial[i] & 0xffffffffU);
        squared[2 * i + 1] = interleaveBits(polynomial[i] >> 32);
    }
    for (int i = 2 * (MT_DEGREE - 1); i >= int(MT_DEGREE); --i) {
        if (!((squared[i / 64] >> (i % 64)) & 1)) continue;
        unsigned shift = i - MT_DEGREE, word = shift / 64, bit = shift % 64;
        for (size_t j = 0; j < MT_POLY_WORDS; ++j) {
            squared[j + word] ^= characteristic[j] << bit;
            if (bit) squared[j + word + 1] ^= characteristic[j] >> (64 - bit);
        }
    }
    MTPolynomial result;
    std::copy_n(squared.begin(), MT_POLY_WORDS, result.begin());
    return result;
}

void jumpMT(RawMT19937& rng, uint64_t steps, const MTPolynomial& characteristic) {
    if (!steps) return;
    MTPolynomial polynomial{};
    polynomial[0] = 1;
    int highest = 63 - __builtin_clzll(steps);
    for (int bit = highest; bit >= 0; --bit) {
        polynomial = squareModulo(polynomial, characteristic);
        if ((steps >> bit) & 1) {
            uint64_t carry = 0;
            for (auto& word : polynomial) {
                uint64_t nextCarry = word >> 63;
                word = (word << 1) | carry;
                carry = nextCarry;
            }
            if ((polynomial[MT_DEGREE / 64] >> (MT_DEGREE % 64)) & 1)
                for (size_t i = 0; i < MT_POLY_WORDS; ++i) polynomial[i] ^= characteristic[i];
        }
    }
    std::array<uint32_t, RNG_WORDS> jumped{};
    for (unsigned i = 0; i < MT_DEGREE; ++i) {
        if ((polynomial[i / 64] >> (i % 64)) & 1)
            for (size_t j = 0; j < RNG_WORDS; ++j) jumped[j] ^= rng.words[j];
        rng.twist();
    }
    std::copy(jumped.begin(), jumped.end(), rng.words);
}

__global__ void generateSamples(const uint32_t* checkpoints, val_t* samples) {
    __shared__ uint32_t states[2][RNG_WORDS];
    unsigned lane = threadIdx.x;
    uint32_t* current = states[0];
    uint32_t* next = states[1];
    for (unsigned i = lane; i < RNG_WORDS; i += blockDim.x)
        current[i] = checkpoints[size_t(blockIdx.x) * RNG_WORDS + i];
    __syncthreads();
    for (unsigned round = 0; round < DRAWS_PER_PAIR; ++round) {
        for (unsigned i = lane; i < RNG_WORDS; i += blockDim.x) {
            uint32_t y = current[i];
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680U;
            y ^= (y << 15) & 0xefc60000U;
            y ^= y >> 18;
            // Match libstdc++ uniform_real_distribution<float>, including its
            // upper-bound correction when conversion rounds UINT_MAX to 2^32.
            samples[(size_t(blockIdx.x) * DRAWS_PER_PAIR + round) * RNG_WORDS + i] =
                fminf(__uint2float_rn(y) * 0x1p-32f, 0x1.fffffep-1f);
        }
        if (round + 1 == DRAWS_PER_PAIR) break;
        if (lane < 227) {
            uint32_t y = (current[lane] & 0x80000000U) | (current[lane + 1] & 0x7fffffffU);
            next[lane] = current[lane + 397] ^ (y >> 1) ^ ((y & 1) * 0x9908b0dfU);
        }
        __syncthreads();
        if (lane < 227) {
            unsigned i = lane + 227;
            uint32_t y = (current[i] & 0x80000000U) | (current[i + 1] & 0x7fffffffU);
            next[i] = next[i - 227] ^ (y >> 1) ^ ((y & 1) * 0x9908b0dfU);
        }
        __syncthreads();
        if (lane < 169) {
            unsigned i = lane + 454;
            uint32_t y = (current[i] & 0x80000000U) | (current[i + 1] & 0x7fffffffU);
            next[i] = next[i - 227] ^ (y >> 1) ^ ((y & 1) * 0x9908b0dfU);
        }
        __syncthreads();
        if (lane == 0) {
            uint32_t y = (current[623] & 0x80000000U) | (next[0] & 0x7fffffffU);
            next[623] = next[396] ^ (y >> 1) ^ ((y & 1) * 0x9908b0dfU);
        }
        __syncthreads();
        uint32_t* temporary = current;
        current = next;
        next = temporary;
    }
}

struct Pair { idx_t receiver, emitter; };

__device__ Vec3 sampledPoint(const Triangle& tri, val_t u, val_t v) {
    if (u + v > ONE) { u = ONE - u; v = ONE - v; }
    return tri.a + (tri.b - tri.a) * u + (tri.c - tri.a) * v;
}

__global__ void formFactorKernel(const Triangle* triangles, const FlatNode* nodes,
                                 const idx_t* indices, const Pair* pairs,
                                 const val_t* samples, size_t sampleOffset,
                                 size_t pairCount, size_t firstRow, size_t rows,
                                 val_t* coefficients) {
    // A half warp traces a pair's 16 independent rays. Sum in original ray
    // order, rather than a tree reduction that changes floating-point results.
    unsigned lane = threadIdx.x & 31;
    unsigned ray = lane & 15;
    size_t p = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / NUM_RAYS;
    val_t contribution = ZERO;
    Pair pair{};
    if (p < pairCount) {
        pair = pairs[p];
        const Triangle& triI = triangles[pair.receiver];
        const Triangle& triJ = triangles[pair.emitter];
        const val_t* random = samples + (sampleOffset + p) * DRAWS_PER_PAIR + ray * 4;
        Vec3 pI = sampledPoint(triI, random[0], random[1]);
        Vec3 pJ = sampledPoint(triJ, random[2], random[3]);
        if (!isRayBlocked(pI, pJ, triangles, nodes, indices, pair.receiver, pair.emitter)) {
            Vec3 v = pJ - pI;
            val_t squaredDistance = v.squaredNorm();
            if (squaredDistance >= EPSILON) {
                val_t length = v.norm();
                val_t cosI = fmaxf(ZERO, v.dot(triI.normal()) / length);
                val_t cosJ = fmaxf(ZERO, (-v).dot(triJ.normal()) / length);
                if (cosI > ZERO && cosJ > ZERO)
                    contribution = (cosI * cosJ) / (PI * squaredDistance);
            }
        }
    }
    val_t total = ZERO;
    for (unsigned r = 0; r < NUM_RAYS; ++r) {
        val_t term = __shfl_sync(0xffffffffU, contribution, (lane & 16) + r);
        if (ray == 0) total += term;
    }
    if (ray == 0 && p < pairCount)
        coefficients[size_t(pair.emitter) * rows + pair.receiver - firstRow] = total * INV_NUM_RAYS;
}

__global__ void timeDelayKernel(const Triangle* triangles, size_t firstRow,
                                size_t rows, size_t n, int* delays, int* bounds) {
    __shared__ int minimum[256], maximum[256];
    unsigned lane = threadIdx.x;
    size_t k = size_t(blockIdx.x) * blockDim.x + lane;
    int delay = 0, positiveMinimum = INT_MAX;
    if (k < rows * n) {
        size_t i = firstRow + k % rows, j = k / rows;
        if (i != j) {
            delay = static_cast<int>(ceilf((triangles[i].center() - triangles[j].center()).norm() * INV_WAVE_SPEED));
            positiveMinimum = delay;
        }
        delays[k] = delay;
    }
    minimum[lane] = positiveMinimum;
    maximum[lane] = delay;
    __syncthreads();
    for (unsigned s = 128; s; s >>= 1) {
        if (lane < s) {
            minimum[lane] = min(minimum[lane], minimum[lane + s]);
            maximum[lane] = max(maximum[lane], maximum[lane + s]);
        }
        __syncthreads();
    }
    if (lane == 0) { atomicMin(bounds, minimum[0]); atomicMax(bounds + 1, maximum[0]); }
}

__global__ void prepareCoefficients(val_t* coefficients, const val_t* areas,
                                    size_t rows, size_t n, unsigned long long* statistics) {
    size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    bool nonzero = false;
    if (k < rows * n) {
        val_t value = coefficients[k];
        nonzero = value > EPSILON;
        coefficients[k] = fminf(value * areas[k / rows], ONE);
    }
    unsigned mask = __ballot_sync(0xffffffffU, nonzero);
    if ((threadIdx.x & 31) == 0 && mask) atomicAdd(statistics, static_cast<unsigned long long>(__popc(mask)));
}

// Eight adjacent receivers share each tile. Emitter lanes compute products in
// parallel; receiver lanes then sum them in increasing j order. This provides
// enough blocks to occupy the accelerator even for a small local row partition.
constexpr unsigned RECEIVER_TILE = 8;
constexpr unsigned EMITTER_TILE = 128;
constexpr unsigned EMITTER_LANES = 16;

__global__ void propagationKernel(const val_t* coefficients, const int* delays,
                                  const val_t* history, size_t historyFrames,
                                  size_t firstRow, size_t rows, size_t n, size_t timesteps,
                                  size_t source, val_t reflectivity, size_t firstTime,
                                  val_t* localHistory, val_t* output) {
    __shared__ val_t contributions[EMITTER_TILE][RECEIVER_TILE];
    unsigned receiverLane = threadIdx.x % RECEIVER_TILE;
    unsigned emitterLane = threadIdx.x / RECEIVER_TILE;
    size_t i = size_t(blockIdx.x) * RECEIVER_TILE + receiverLane;
    size_t t = firstTime + blockIdx.y;
    val_t sum = ZERO;
    for (size_t first = 0; first < n; first += EMITTER_TILE) {
        for (unsigned offset = emitterLane; offset < EMITTER_TILE; offset += EMITTER_LANES) {
            size_t j = first + offset;
            val_t product = ZERO;
            if (i < rows && j < n && firstRow + i != j) {
                size_t matrixIndex = j * rows + i;
                int delay = delays[matrixIndex];
                if (t >= static_cast<size_t>(delay)) {
                    val_t weight = coefficients[matrixIndex];
                    if (weight > ZERO) {
                        val_t rad = history[((t - delay) % historyFrames) * n + j];
                        if (!(rad <= ZERO)) product = weight * rad;
                    }
                }
            }
            contributions[offset][receiverLane] = product;
        }
        __syncthreads();
        if (emitterLane == 0) {
            unsigned count = static_cast<unsigned>(n - first < EMITTER_TILE ? n - first : EMITTER_TILE);
            for (unsigned j = 0; j < count; ++j) sum += contributions[j][receiverLane];
        }
        __syncthreads();
    }
    if (emitterLane == 0 && i < rows) {
        val_t emission = firstRow + i == source && t < timesteps / 2 ? ONE : ZERO;
        val_t value = reflectivity * sum + emission;
        output[size_t(blockIdx.y) * rows + i] = value;
        localHistory[i * timesteps + t] = value;
    }
}

__global__ void mergeHistory(const val_t* gathered, val_t* history, val_t* sourceHistory,
                             size_t n, size_t historyFrames, size_t source,
                             size_t firstTime, size_t batchSize, int ranks) {
    size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= n * batchSize) return;
    size_t j = k % n, b = k / n;
    size_t quotient = n / ranks, remainder = n % ranks;
    size_t slot = j < (quotient + 1) * remainder ? j / (quotient + 1) :
                  remainder + (j - (quotient + 1) * remainder) / quotient;
    size_t first = slot * quotient + (slot < remainder ? slot : remainder);
    size_t count = quotient + (slot < remainder);
    val_t value = gathered[first * batchSize + b * count + j - first];
    history[((firstTime + b) % historyFrames) * n + j] = value;
    if (j == source) sourceHistory[firstTime + b] = value;
}

__global__ void distanceKernel(const val_t* localHistory, const val_t* sourceHistory,
                               size_t rows, size_t timesteps, val_t* distances,
                               unsigned long long* statistics) {
    size_t i = blockIdx.x;
    if (i >= rows) return;
    unsigned lane = threadIdx.x;
    val_t maxCorrelation = ZERO;
    size_t bestTime = 0;
    const val_t* triangleHistory = localHistory + i * timesteps;
    for (size_t t = lane; t < timesteps; t += blockDim.x) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < timesteps; ++tt)
            sum += sourceHistory[tt - t] * triangleHistory[tt];
        if (sum > maxCorrelation || (sum == maxCorrelation && t < bestTime)) {
            maxCorrelation = sum;
            bestTime = t;
        }
    }
    __shared__ val_t correlations[256];
    __shared__ size_t times[256];
    correlations[lane] = maxCorrelation;
    times[lane] = bestTime;
    __syncthreads();
    for (unsigned stride = 128; stride; stride >>= 1) {
        if (lane < stride && (correlations[lane + stride] > correlations[lane] ||
            (correlations[lane + stride] == correlations[lane] && times[lane + stride] < times[lane]))) {
            correlations[lane] = correlations[lane + stride];
            times[lane] = times[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        distances[i] = WAVE_SPEED * static_cast<val_t>(times[0]);
        for (size_t t = 0; t < timesteps; ++t) {
            if (triangleHistory[t] > EPSILON) { atomicAdd(statistics + 1, 1ULL); break; }
        }
    }
}

struct SimulationState {
    size_t numTriangles = 0, numTimesteps = 0, sourceIndex = 0;
    size_t firstRow = 0, localRows = 0, historyFrames = 1, batchSize = 1;
    val_t reflectivity = 0.8f;
    int minimumDelay = 1, maximumDelay = 0;
    std::vector<Triangle> triangles;
    std::vector<val_t> areas, distances;
    std::vector<int> rowCounts, rowStarts;
    std::vector<uint64_t> activeOffsets;
    unsigned long long nonZeroKij = 0, receivedEnergy = 0;
    DeviceBuffer<Triangle> deviceTriangles;
    DeviceBuffer<FlatNode> nodes;
    DeviceBuffer<idx_t> indices;
    DeviceBuffer<val_t> deviceAreas, coefficients, history, localHistory, sourceHistory, deviceDistances;
    DeviceBuffer<int> delays, delayBounds;
    DeviceBuffer<unsigned long long> statistics;
};

unsigned blocksFor(size_t count) { return static_cast<unsigned>((count + 255) / 256); }

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;
    size_t n = state.numTriangles;
    state.rowCounts.resize(worldSize);
    state.rowStarts.resize(worldSize);
    // Rank zero owns the final partition: other ranks can begin processing
    // while it advances and distributes their starting RNG checkpoints.
    for (int rank = 0; rank < worldSize; ++rank) {
        size_t slot = (rank + worldSize - 1) % worldSize;
        state.rowCounts[rank] = static_cast<int>(n / worldSize + (slot < n % worldSize));
        state.rowStarts[rank] = static_cast<int>(slot * (n / worldSize) + std::min(slot, n % worldSize));
    }
    state.firstRow = state.rowStarts[worldRank];
    state.localRows = state.rowCounts[worldRank];
    rootPrintf("Generated icosphere mesh with %zu triangles\nBuilding octree...\n", n);
    std::vector<FlatNode> nodes;
    std::vector<idx_t> indices;
    {
        Octree tree;
        #pragma omp parallel
        {
            #pragma omp single
            tree.build(state.triangles);
        }
        flattenOctree(tree, nodes, indices);
    }
    state.areas.resize(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) state.areas[i] = state.triangles[i].area();
    state.activeOffsets.resize(state.localRows + 1, 0);
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < state.localRows; ++r) {
        size_t i = state.firstRow + r;
        uint64_t count = 0;
        for (size_t j = 0; j < n; ++j)
            if (i != j && !(state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f)) ++count;
        state.activeOffsets[r + 1] = count;
    }
    for (size_t r = 0; r < state.localRows; ++r) state.activeOffsets[r + 1] += state.activeOffsets[r];
    state.deviceTriangles.resize(n);
    state.deviceTriangles.upload(state.triangles.data(), n);
    state.nodes.resize(nodes.size());
    state.nodes.upload(nodes.data(), nodes.size());
    state.indices.resize(indices.size());
    state.indices.upload(indices.data(), indices.size());
    state.deviceAreas.resize(n);
    state.deviceAreas.upload(state.areas.data(), n);
    state.coefficients.resize(n * state.localRows);
    state.delays.resize(n * state.localRows);
    cudaCheck(cudaMemset(state.coefficients.data, 0, n * state.localRows * sizeof(val_t)));
    state.localHistory.resize(timesteps * state.localRows);
    state.sourceHistory.resize(timesteps);
    state.deviceDistances.resize(state.localRows);
    state.statistics.resize(2);
    cudaCheck(cudaMemset(state.statistics.data, 0, 2 * sizeof(unsigned long long)));
    state.delayBounds.resize(2);
    int bounds[2] = {INT_MAX, 0};
    state.delayBounds.upload(bounds, 2);
}

void computeTimeDelays(SimulationState& state) {
    rootPrintf("Computing time delays (Tau)...\n");
    size_t entries = state.localRows * state.numTriangles;
    if (entries) {
        timeDelayKernel<<<blocksFor(entries), 256>>>(state.deviceTriangles.data, state.firstRow,
            state.localRows, state.numTriangles, state.delays.data, state.delayBounds.data);
        cudaCheck(cudaGetLastError());
    }
    int bounds[2];
    cudaCheck(cudaMemcpy(bounds, state.delayBounds.data, sizeof(bounds), cudaMemcpyDeviceToHost));
    mpiCheck(MPI_Allreduce(bounds, &state.minimumDelay, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    mpiCheck(MPI_Allreduce(bounds + 1, &state.maximumDelay, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    if (state.minimumDelay <= 0 || state.minimumDelay == INT_MAX)
        throw std::runtime_error("non-diagonal propagation delays must be positive");
    state.historyFrames = std::max(size_t(1), std::min(state.numTimesteps, size_t(state.maximumDelay)));
    state.batchSize = std::max(size_t(1), std::min({size_t(state.minimumDelay), state.numTimesteps,
                                                  size_t(INT_MAX) / state.numTriangles}));
    state.history.resize(state.historyFrames * state.numTriangles);
    cudaCheck(cudaMemset(state.history.data, 0, state.history.count * sizeof(val_t)));
}

struct FactorBatch {
    PinnedBuffer<Pair> pairs;
    PinnedBuffer<uint32_t> checkpoints;
    DeviceBuffer<Pair> devicePairs;
    DeviceBuffer<uint32_t> deviceCheckpoints;
    DeviceBuffer<val_t> samples;
    cudaStream_t stream = nullptr;
    FactorBatch() {
        pairs.resize(BATCH_PAIRS);
        checkpoints.resize(BATCH_CHUNKS * RNG_WORDS);
        devicePairs.resize(BATCH_PAIRS);
        deviceCheckpoints.resize(BATCH_CHUNKS * RNG_WORDS);
        samples.resize(BATCH_CHUNKS * RNG_PAIRS * DRAWS_PER_PAIR);
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~FactorBatch() { if (stream) cudaStreamDestroy(stream); }
};

void computeFormFactors(SimulationState& state) {
    rootPrintf("Computing form factors (Kij)...\n");
    uint64_t localCount = state.activeOffsets.back();
    std::vector<uint64_t> counts(worldSize);
    mpiCheck(MPI_Allgather(&localCount, 1, MPI_UINT64_T, counts.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD));
    std::vector<uint64_t> starts(worldSize);
    uint64_t total = 0;
    for (int slot = 0; slot < worldSize; ++slot) {
        int rank = (slot + 1) % worldSize;
        starts[rank] = total;
        total += counts[rank];
    }
    RawMT19937 rng;
    uint64_t largestSkip = (starts[0] / RNG_PAIRS) * DRAWS_PER_PAIR;
    if (worldSize > 1 && largestSkip > 500000) {
        MTPolynomial characteristic{};
        if (worldRank == 0) characteristic = mtCharacteristic();
        mpiCheck(MPI_Bcast(characteristic.data(), static_cast<int>(MT_POLY_WORDS),
                           MPI_UINT64_T, 0, MPI_COMM_WORLD));
        jumpMT(rng, (starts[worldRank] / RNG_PAIRS) * DRAWS_PER_PAIR, characteristic);
    } else if (worldRank == 0) {
        uint64_t chunk = 0;
        for (int slot = 0; slot < worldSize; ++slot) {
            int rank = (slot + 1) % worldSize;
            uint64_t targetChunk = starts[rank] / RNG_PAIRS;
            while (chunk < targetChunk) { rng.advanceChunk(); ++chunk; }
            if (rank != 0)
                mpiCheck(MPI_Send(rng.words, RNG_WORDS, MPI_UINT32_T, rank, 7, MPI_COMM_WORLD));
        }
    } else {
        mpiCheck(MPI_Recv(rng.words, RNG_WORDS, MPI_UINT32_T, 0, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    if (localCount) {
        FactorBatch batches[2];
        uint64_t begin = 0;
        size_t offset = starts[worldRank] % RNG_PAIRS;
        unsigned turn = 0;
        while (begin < localCount) {
            FactorBatch& batch = batches[turn++ & 1];
            cudaCheck(cudaStreamSynchronize(batch.stream));
            size_t count = static_cast<size_t>(std::min<uint64_t>(BATCH_PAIRS - offset, localCount - begin));
            size_t chunks = (offset + count + RNG_PAIRS - 1) / RNG_PAIRS;
            size_t first = std::upper_bound(state.activeOffsets.begin(), state.activeOffsets.end(), begin)
                           - state.activeOffsets.begin() - 1;
            size_t last = std::lower_bound(state.activeOffsets.begin(), state.activeOffsets.end(), begin + count)
                          - state.activeOffsets.begin();
            #pragma omp parallel for schedule(static)
            for (size_t r = first; r < last; ++r) {
                size_t i = state.firstRow + r;
                uint64_t ordinal = state.activeOffsets[r];
                for (size_t j = 0; j < state.numTriangles; ++j) {
                    if (i == j || state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) continue;
                    if (ordinal >= begin && ordinal < begin + count)
                        batch.pairs.data[ordinal - begin] = {static_cast<idx_t>(i), static_cast<idx_t>(j)};
                    ++ordinal;
                }
            }
            for (size_t c = 0; c < chunks; ++c) {
                std::memcpy(batch.checkpoints.data + c * RNG_WORDS, rng.words, sizeof(rng.words));
                rng.advanceChunk();
            }
            cudaCheck(cudaMemcpyAsync(batch.devicePairs.data, batch.pairs.data, count * sizeof(Pair),
                                      cudaMemcpyHostToDevice, batch.stream));
            cudaCheck(cudaMemcpyAsync(batch.deviceCheckpoints.data, batch.checkpoints.data,
                                      chunks * RNG_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice, batch.stream));
            generateSamples<<<static_cast<unsigned>(chunks), 256, 0, batch.stream>>>(batch.deviceCheckpoints.data, batch.samples.data);
            cudaCheck(cudaGetLastError());
            formFactorKernel<<<static_cast<unsigned>((count + 15) / 16), 256, 0, batch.stream>>>(
                state.deviceTriangles.data, state.nodes.data, state.indices.data, batch.devicePairs.data,
                batch.samples.data, offset, count, state.firstRow, state.localRows, state.coefficients.data);
            cudaCheck(cudaGetLastError());
            begin += count;
            offset = 0;
        }
        for (auto& batch : batches) cudaCheck(cudaStreamSynchronize(batch.stream));
    }
    size_t entries = state.localRows * state.numTriangles;
    if (entries) {
        prepareCoefficients<<<blocksFor(entries), 256>>>(state.coefficients.data, state.deviceAreas.data,
                                                        state.localRows, state.numTriangles, state.statistics.data);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());
}

void runSimulation(SimulationState& state) {
    rootPrintf("Running wave propagation simulation...\n");
    DeviceBuffer<val_t> batchOutput, gathered;
    batchOutput.resize(state.batchSize * state.localRows);
    PinnedBuffer<val_t> send, receive;
    if (worldSize > 1) {
        if (!deviceMPI) {
            send.resize(state.batchSize * state.localRows);
            receive.resize(state.batchSize * state.numTriangles);
        }
        gathered.resize(state.batchSize * state.numTriangles);
    }
    std::vector<int> counts(worldSize), starts(worldSize);
    for (size_t t = 0; t < state.numTimesteps; t += state.batchSize) {
        size_t batch = std::min(state.batchSize, state.numTimesteps - t);
        size_t entries = batch * state.localRows;
        if (entries) {
            dim3 grid(static_cast<unsigned>((state.localRows + RECEIVER_TILE - 1) / RECEIVER_TILE),
                      static_cast<unsigned>(batch));
            propagationKernel<<<grid, RECEIVER_TILE * EMITTER_LANES>>>(state.coefficients.data, state.delays.data,
                state.history.data, state.historyFrames, state.firstRow, state.localRows, state.numTriangles,
                state.numTimesteps, state.sourceIndex, state.reflectivity, t,
                state.localHistory.data, batchOutput.data);
            cudaCheck(cudaGetLastError());
        }
        const val_t* merged = batchOutput.data;
        if (worldSize > 1) {
            for (int rank = 0; rank < worldSize; ++rank) {
                counts[rank] = static_cast<int>(batch * state.rowCounts[rank]);
                starts[rank] = static_cast<int>(batch * state.rowStarts[rank]);
            }
            if (deviceMPI) {
                // MPI must see completed CUDA writes, including the previous
                // merge on ranks with an empty local partition.
                cudaCheck(cudaDeviceSynchronize());
                mpiCheck(MPI_Allgatherv(batchOutput.data, static_cast<int>(entries), MPI_FLOAT,
                                       gathered.data, counts.data(), starts.data(), MPI_FLOAT, MPI_COMM_WORLD));
            } else {
                if (entries) cudaCheck(cudaMemcpy(send.data, batchOutput.data, entries * sizeof(val_t), cudaMemcpyDeviceToHost));
                mpiCheck(MPI_Allgatherv(send.data, static_cast<int>(entries), MPI_FLOAT, receive.data,
                                       counts.data(), starts.data(), MPI_FLOAT, MPI_COMM_WORLD));
                gathered.upload(receive.data, batch * state.numTriangles);
            }
            merged = gathered.data;
        }
        mergeHistory<<<blocksFor(batch * state.numTriangles), 256>>>(merged, state.history.data,
            state.sourceHistory.data, state.numTriangles, state.historyFrames, state.sourceIndex, t, batch, worldSize);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());
}

void computeDistances(SimulationState& state) {
    rootPrintf("Computing distances via cross-correlation...\n");
    if (state.localRows) {
        distanceKernel<<<static_cast<unsigned>(state.localRows), 256>>>(state.localHistory.data,
            state.sourceHistory.data, state.localRows, state.numTimesteps, state.deviceDistances.data, state.statistics.data);
        cudaCheck(cudaGetLastError());
    }
    std::vector<val_t> local(state.localRows);
    if (state.localRows) cudaCheck(cudaMemcpy(local.data(), state.deviceDistances.data,
                                             state.localRows * sizeof(val_t), cudaMemcpyDeviceToHost));
    if (worldRank == 0) state.distances.resize(state.numTriangles);
    mpiCheck(MPI_Gatherv(local.data(), static_cast<int>(state.localRows), MPI_FLOAT, state.distances.data(),
                         state.rowCounts.data(), state.rowStarts.data(), MPI_FLOAT, 0, MPI_COMM_WORLD));
    unsigned long long localStatistics[2], globalStatistics[2] = {};
    cudaCheck(cudaMemcpy(localStatistics, state.statistics.data, sizeof(localStatistics), cudaMemcpyDeviceToHost));
    mpiCheck(MPI_Reduce(localStatistics, globalStatistics, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD));
    state.nonZeroKij = globalStatistics[0];
    state.receivedEnergy = globalStatistics[1];
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
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

    rootPrintf("  Triangles receiving energy: %llu/%zu\n", state.receivedEnergy, state.numTriangles);
    if (state.receivedEnergy == 0) {
        rootPrintf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }
    rootPrintf("  Non-zero form factors: %llu/%zu (%.2f%%)\n", state.nonZeroKij,
               state.numTriangles * state.numTriangles,
               100.0 * state.nonZeroKij / (double(state.numTriangles) * state.numTriangles));
    if (state.nonZeroKij == 0) {
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
        uint32_t bits;
        std::memcpy(&bits, &state.distances[i], sizeof(bits));
        hash ^= (static_cast<uint64_t>(bits) + i) * 0x9e3779b97f4a7c15ULL;
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

int benchmarkMain(int argc, char** argv, ParallelRuntime& runtime) {
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
            rootPrintf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (timesteps < 0) {
        rootPrintf("Number of timesteps must be non-negative\n");
        return 1;
    }
    runtime.selectDevice();
    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    rootPrintf("Room Response Simulation Benchmark\n");
    rootPrintf("===================================\n");
    rootPrintf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    rootPrintf("Timesteps: %d\n", timesteps);
    rootPrintf("Source triangle: %d\n", sourceIdx);
    rootPrintf("Reflectivity: %.2f\n", reflectivity);
    rootPrintf("Validation: %s\n", validate ? "enabled" : "disabled");
    rootPrintf("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    rootPrintf("\n");

    // Precomputation
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();
    long preMillis = static_cast<long>(preDuration);
    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &preMillis, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD));
    preDuration = preMillis;

    rootPrintf("Precomputation time: %ld ms\n", preDuration);
    rootPrintf("\n");

    // Simulation
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();
    long simMillis = static_cast<long>(simDuration);
    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &simMillis, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD));
    simDuration = simMillis;

    rootPrintf("Simulation time: %ld ms\n", simDuration);
    rootPrintf("\n");

    // Distance computation
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();
    long distMillis = static_cast<long>(distDuration);
    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &distMillis, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD));
    distDuration = distMillis;

    rootPrintf("Distance computation time: %ld ms\n", distDuration);
    rootPrintf("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    rootPrintf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n) * n;
    double simOps = kijOps * t;
    double distOps = static_cast<double>(n) * t * t;

    rootPrintf("\nPerformance:\n");
    rootPrintf("  Triangles: %zu\n", n);
    rootPrintf("  Timesteps: %zu\n", t);
    rootPrintf("  Form factor computations: %.2e\n", kijOps);
    rootPrintf("  Simulation operations: %.2e\n", simOps);
    rootPrintf("  Distance computations: %.2e\n", distOps);
    rootPrintf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Persistent device storage; temporary form-factor batches are bounded.
    unsigned long long localMemory = state.localRows * n * (sizeof(val_t) + sizeof(int)) +
        (state.localRows * t + state.historyFrames * n + t + n + state.localRows) * sizeof(val_t) +
        n * sizeof(Triangle) + state.nodes.count * sizeof(FlatNode) + state.indices.count * sizeof(idx_t);
    unsigned long long maximumMemory = 0;
    mpiCheck(MPI_Reduce(&localMemory, &maximumMemory, 1, MPI_UNSIGNED_LONG_LONG,
                        MPI_MAX, 0, MPI_COMM_WORLD));
    rootPrintf("  Maximum persistent GPU memory per rank: %.2f MB\n", maximumMemory / (1024.0 * 1024.0));

    int result = 0;
    if (worldRank == 0) {
        uint64_t hash = computeHash(state);
        rootPrintf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));
        if (printResults) {
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }
        if (validate && !validateResults(state)) result = 1;
    }
    mpiCheck(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return result;
}

int main(int argc, char** argv) {
    ParallelRuntime runtime(argc, argv);
    try {
        return benchmarkMain(argc, argv, runtime);
    } catch (const std::exception& error) {
        fprintf(stderr, "roomsim rank %d: %s\n", worldRank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
