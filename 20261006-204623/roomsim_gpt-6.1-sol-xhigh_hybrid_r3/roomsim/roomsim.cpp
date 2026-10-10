/**
 * Room Response Simulation Benchmark
 *
 * This is a hybrid MPI/OpenMP/CUDA implementation of room impulse response
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
#include <exception>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <vector>
#include <stdexcept>

#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f; // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16; // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    __host__ __device__ explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    __host__ __device__ Vec3 operator+(const Vec3 &o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3 &o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ val_t dot(const Vec3 &o) const {
#ifdef __CUDA_ARCH__
        // Fix contraction order to match the original native CPU arithmetic.
        return fmaf(z, o.z, fmaf(x, o.x, y * o.y));
#else
        return x * o.x + y * o.y + z * o.z;
#endif
    }
    __host__ __device__ Vec3 cross(const Vec3 &o) const {
#ifdef __CUDA_ARCH__
        return {fmaf(y, o.z, -z * o.y), fmaf(z, o.x, -x * o.z), fmaf(x, o.y, -y * o.x)};
#else
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
#endif
    }

    __host__ __device__ val_t squaredNorm() const { return dot(*this); }
    __host__ __device__ val_t norm() const { return std::sqrt(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3 &o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    __host__ __device__ Triangle() = default;
    __host__ __device__ Triangle(const Vec3 &a, const Vec3 &b, const Vec3 &c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    __host__ __device__ val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    __host__ __device__ bool operator==(const Triangle &o) const { return a == o.a && b == o.b && c == o.c; }
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction)
// ============================================================================

// Separating axis test for triangle-box overlap
bool triangleBoxOverlap(const Vec3 &boxCenter, const Vec3 &boxHalfSize, const Triangle &tri) {
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
    if (minX > boxHalfSize.x || maxX < -boxHalfSize.x)
        return false;

    auto [minY, maxY] = minMax3(v0.y, v1.y, v2.y);
    if (minY > boxHalfSize.y || maxY < -boxHalfSize.y)
        return false;

    auto [minZ, maxZ] = minMax3(v0.z, v1.z, v2.z);
    if (minZ > boxHalfSize.z || maxZ < -boxHalfSize.z)
        return false;

    // Test triangle normal
    Vec3 triNormal = e0.cross(e1);
    val_t d = triNormal.dot(v0);
    val_t r = boxHalfSize.x * std::abs(triNormal.x) + boxHalfSize.y * std::abs(triNormal.y) +
              boxHalfSize.z * std::abs(triNormal.z);
    if (std::abs(d) > r)
        return false;

    // Test 9 edge cross products
    auto testAxis = [&](const Vec3 &axis) {
        val_t p0 = axis.dot(v0);
        val_t p1 = axis.dot(v1);
        val_t p2 = axis.dot(v2);
        val_t r = boxHalfSize.x * std::abs(axis.x) + boxHalfSize.y * std::abs(axis.y) +
                  boxHalfSize.z * std::abs(axis.z);
        auto [minP, maxP] = minMax3(p0, p1, p2);
        return !(minP > r || maxP < -r);
    };

    Vec3 axes[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    Vec3 edges[3] = {e0, e1, e2};
    for (const auto &axis : axes) {
        for (const auto &edge : edges) {
            Vec3 crossAxis = axis.cross(edge);
            if (crossAxis.squaredNorm() > EPSILON) {
                if (!testAxis(crossAxis))
                    return false;
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
    std::vector<size_t> triangleIndices;       // Indices into the global triangle list
    const std::vector<Triangle> *allTriangles; // Pointer to all triangles

    Octree() : allTriangles(nullptr) {}

    void build(const std::vector<Triangle> &triangles) {
        allTriangles = &triangles;
        if (triangles.empty())
            return;

        // Compute bounding box
        minBound = triangles[0].a;
        maxBound = triangles[0].a;
        for (const auto &tri : triangles) {
            for (const auto *v : {&tri.a, &tri.b, &tri.c}) {
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
        for (size_t i = 0; i < triangles.size(); ++i)
            allIndices[i] = i;

        buildNode(allIndices, minBound, maxBound);
    }

  private:
    void buildNode(const std::vector<size_t> &indices, const Vec3 &nodeMin, const Vec3 &nodeMax) {
        minBound = nodeMin;
        maxBound = nodeMax;
        halfExtent = (maxBound - minBound) * 0.5f;
        center = (minBound + maxBound) * 0.5f;

        // If few enough triangles or too small, make this a leaf
        if (indices.size() <= MAX_OCTREE_TRIS || (maxBound - minBound).norm() < MAX_OCTREE_LEAF_SIZE) {
            triangleIndices = indices;
            return;
        }

        // Subdivide into 8 children
        Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];

        for (size_t idx : indices) {
            const Triangle &tri = (*allTriangles)[idx];

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
#pragma omp task firstprivate(i, childMin, childMax) shared(childIndices) if (indices.size() > 256)
                children[i]->buildNode(childIndices[i], childMin, childMax);
            }
        }
#pragma omp taskwait
    }

  public:
    // Check if a ray intersects this node's bounding box
    bool rayIntersectsBox(const Vec3 &p1, const Vec3 &p2) const {
        Vec3 d = (p2 - p1) * 0.5f;
        Vec3 c = p1 + d - center;
        Vec3 ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};

        if (std::abs(c.x) > halfExtent.x + ad.x)
            return false;
        if (std::abs(c.y) > halfExtent.y + ad.y)
            return false;
        if (std::abs(c.z) > halfExtent.z + ad.z)
            return false;

        if (std::abs(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON)
            return false;
        if (std::abs(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON)
            return false;
        if (std::abs(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON)
            return false;

        return true;
    }

    // Apply a function to all triangles potentially intersecting the ray
    // Returns true if the function returns true for any triangle
    template <typename Func> bool applyToTris(const Vec3 &p1, const Vec3 &p2, Func &&func) const {
        // If leaf node, check triangles directly
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx]))
                    return true;
            }
            return false;
        }

        // Otherwise, descend to children
        for (int i = 0; i < 8; ++i) {
            if (children[i] && children[i]->rayIntersectsBox(p1, p2)) {
                if (children[i]->applyToTris(p1, p2, func))
                    return true;
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
            Vec3(-1, t, 0).normalized() * radius,  Vec3(1, t, 0).normalized() * radius,
            Vec3(-1, -t, 0).normalized() * radius, Vec3(1, -t, 0).normalized() * radius,
            Vec3(0, -1, t).normalized() * radius,  Vec3(0, 1, t).normalized() * radius,
            Vec3(0, -1, -t).normalized() * radius, Vec3(0, 1, -t).normalized() * radius,
            Vec3(t, 0, -1).normalized() * radius,  Vec3(t, 0, 1).normalized() * radius,
            Vec3(-t, 0, -1).normalized() * radius, Vec3(-t, 0, 1).normalized() * radius};

        // Initial icosahedron faces (20 triangles)
        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5},  {0, 5, 1},  {0, 1, 7},  {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
            {11, 10, 2}, {10, 7, 6}, {7, 1, 8},  {3, 9, 4},  {3, 4, 2},   {3, 2, 6}, {3, 6, 8},
            {3, 8, 9},   {4, 9, 5},  {2, 4, 11}, {6, 2, 10}, {8, 6, 7},   {9, 8, 1}};

        // Subdivide
        for (int i = 0; i < subdivisions; ++i) {
            std::vector<std::array<idx_t, 3>> newFaces;
            std::map<std::pair<idx_t, idx_t>, idx_t> midpointCache;

            auto getMidpoint = [&](idx_t i1, idx_t i2) -> idx_t {
                auto key = std::make_pair(std::min(i1, i2), std::max(i1, i2));
                auto it = midpointCache.find(key);
                if (it != midpointCache.end())
                    return it->second;

                Vec3 mid = (vertices[i1] + vertices[i2]) / 2.0f;
                mid = mid.normalized() * radius;
                idx_t idx = static_cast<idx_t>(vertices.size());
                vertices.push_back(mid);
                midpointCache[key] = idx;
                return idx;
            };

            for (const auto &face : faces) {
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
            const auto &face = faces[i];
            triangles[i] = Triangle(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Hybrid runtime and device geometry
// ============================================================================

void mpiCheck(int error) {
    if (error == MPI_SUCCESS)
        return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(error, message, &length);
    fprintf(stderr, "MPI: %.*s\n", length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
}

void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

struct ParallelRuntime {
    int rank = 0, ranks = 1;
    MPI_Comm localComm = MPI_COMM_NULL;
    ParallelRuntime(int &argc, char **&argv) {
        int provided = 0;
        mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
        mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
        mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
        if (provided < MPI_THREAD_FUNNELED) {
            fprintf(stderr, "MPI_THREAD_FUNNELED is required.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm));
        if (!std::getenv("OMP_NUM_THREADS")) {
            int localRanks = 1;
            mpiCheck(MPI_Comm_size(localComm, &localRanks));
            // Avoid every MPI rank creating a full-node OpenMP team by default.
            omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs() / localRanks)));
        }
    }
    void selectDevice() const {
        int localRank = 0, devices = 0;
        mpiCheck(MPI_Comm_rank(localComm, &localRank));
        cudaCheck(cudaGetDeviceCount(&devices));
        if (!devices)
            throw std::runtime_error("A CUDA accelerator is required on every rank");
        // Also supports schedulers that expose just one assigned GPU per process.
        cudaCheck(cudaSetDevice(localRank % devices));
    }
    ~ParallelRuntime() {
        MPI_Comm_free(&localComm);
        MPI_Finalize();
    }
};

template <class T> class DeviceArray {
    T *data_ = nullptr;

  public:
    DeviceArray() = default;
    DeviceArray(const DeviceArray &) = delete;
    DeviceArray &operator=(const DeviceArray &) = delete;
    void resize(size_t n) {
        if (data_)
            cudaCheck(cudaFree(data_));
        data_ = nullptr;
        if (n > SIZE_MAX / sizeof(T))
            throw std::runtime_error("Device allocation size overflow");
        if (n)
            cudaCheck(cudaMalloc(reinterpret_cast<void **>(&data_), n * sizeof(T)));
    }
    T *get() const { return data_; }
    ~DeviceArray() {
        if (data_)
            cudaFree(data_);
    }
};

template <class T> class PinnedArray {
    T *data_ = nullptr;

  public:
    PinnedArray() = default;
    PinnedArray(const PinnedArray &) = delete;
    PinnedArray &operator=(const PinnedArray &) = delete;
    void resize(size_t n) {
        if (data_)
            cudaCheck(cudaFreeHost(data_));
        data_ = nullptr;
        if (n > SIZE_MAX / sizeof(T))
            throw std::runtime_error("Host allocation size overflow");
        // Allocate one element even on ranks owning no receiver rows.
        cudaCheck(cudaMallocHost(reinterpret_cast<void **>(&data_), std::max(n, size_t(1)) * sizeof(T)));
    }
    T *get() const { return data_; }
    ~PinnedArray() {
        if (data_)
            cudaFreeHost(data_);
    }
};

struct FlatNode {
    Vec3 center, halfExtent;
    uint32_t begin, count, escape;
};

void flattenOctree(const Octree &tree, std::vector<FlatNode> &nodes, std::vector<idx_t> &indices) {
    const size_t here = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, static_cast<uint32_t>(indices.size()),
                     static_cast<uint32_t>(tree.triangleIndices.size()), 0});
    for (size_t i : tree.triangleIndices)
        indices.push_back(static_cast<idx_t>(i));
    for (const auto &child : tree.children) {
        if (child)
            flattenOctree(*child, nodes, indices);
    }
    if (nodes.size() > UINT32_MAX || indices.size() > UINT32_MAX)
        throw std::runtime_error("Octree exceeds device index range");
    nodes[here].escape = static_cast<uint32_t>(nodes.size());
}

__device__ bool intersectsBox(const Vec3 &p1, const Vec3 &p2, const FlatNode &node) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3 &h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y || fabsf(c.z) > h.z + ad.z)
        return false;
    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON)
        return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON)
        return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON)
        return false;
    return true;
}

__device__ val_t rayTriangleIntersect(const Vec3 &orig, const Vec3 &dir, const Triangle &tri) {
    Vec3 e1 = tri.b - tri.a, e2 = tri.c - tri.a;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);
    if (fabsf(det) < EPSILON)
        return FLT_MAX;
    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - tri.a;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < ZERO || u > ONE)
        return FLT_MAX;
    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < ZERO || u + v > ONE)
        return FLT_MAX;
    return e2.dot(qvec) * invDet;
}

__device__ bool isRayBlocked(const Vec3 &from, const Vec3 &to, const Triangle *triangles,
                             const FlatNode *nodes, const idx_t *indices, idx_t src, idx_t dst) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON)
        return true;
    Vec3 dirNorm = dir / rayLen;
    // Depth-first escape links preserve the original traversal with no per-ray stack.
    uint32_t node = 0;
    while (node < nodes[0].escape) {
        const FlatNode &n = nodes[node];
        if (node && !intersectsBox(from, to, n)) {
            node = n.escape;
            continue;
        }
        for (uint32_t k = 0; k < n.count; ++k) {
            idx_t i = indices[n.begin + k];
            if (i == src || i == dst)
                continue;
            val_t distance = rayTriangleIntersect(from, dirNorm, triangles[i]);
            if (distance > EPSILON && distance < rayLen - EPSILON)
                return true;
        }
        ++node;
    }
    return false;
}

// Exact MT19937 checkpoints: culling consumes no samples, and every surviving
// pair consumes 16 * 4 words. Rank/thread count never changes the sample stream.
struct MTState {
    uint32_t words[624];
    int position;
    explicit MTState(uint32_t seed = 42) : position(624) {
        words[0] = seed;
        for (uint32_t i = 1; i < 624; ++i)
            words[i] = 1812433253u * (words[i - 1] ^ (words[i - 1] >> 30)) + i;
    }
    void twist() {
        for (int begin = 0; begin < 623; begin += 227) {
            int end = std::min(begin + 227, 623);
// Within each range all sources precede or follow the whole range.
#pragma omp simd
            for (int k = begin; k < end; ++k) {
                uint32_t y = (words[k] & 0x80000000u) | (words[k + 1] & 0x7fffffffu);
                words[k] = words[k < 227 ? k + 397 : k - 227] ^ (y >> 1) ^ ((y & 1u) * 0x9908b0dfu);
            }
        }
        uint32_t y = (words[623] & 0x80000000u) | (words[0] & 0x7fffffffu);
        words[623] = words[396] ^ (y >> 1) ^ ((y & 1u) * 0x9908b0dfu);
        position = 0;
    }
    void discard(uint64_t count) {
        while (count > static_cast<uint64_t>(624 - position)) {
            count -= 624 - position;
            twist();
        }
        position += static_cast<int>(count);
    }
    void jump(uint64_t count) {
        // Characteristic polynomial of the MT19937 word recurrence (degree
        // 19937, 135 terms). Apply x^count modulo this polynomial to the
        // canonical window of 624 consecutive untempered stream words.
        static constexpr int degree = 19937;
        static constexpr int powers[] = {
            19937, 19314, 19087, 18860, 18691, 18633, 18406, 18237, 18179, 18068, 17952, 17841, 17783, 17725,
            17498, 17445, 17329, 17271, 17160, 17044, 16933, 16875, 16822, 16817, 16595, 16590, 16537, 16421,
            16368, 16363, 16252, 16141, 16136, 16025, 15967, 15909, 15682, 15629, 15576, 15513, 15455, 15349,
            15344, 15228, 15117, 15059, 15006, 15001, 14953, 14779, 14774, 14721, 14605, 14552, 14547, 14436,
            14325, 14320, 14209, 14151, 14093, 13866, 13813, 13760, 13697, 13639, 13533, 13528, 13412, 13301,
            13243, 13190, 13185, 13137, 12963, 12958, 12905, 12789, 12736, 12731, 12673, 12620, 12509, 12504,
            12393, 12335, 12277, 11997, 11944, 11881, 11838, 11717, 11712, 11611, 11485, 11384, 11374, 11321,
            11215, 11157, 11147, 11089, 10920, 10761, 10693, 10128, 9969,  9901,  9505,  8206,  7979,  7752,
            7583,  7525,  7477,  7129,  6569,  6337,  5661,  4753,  4362,  4135,  3908,  3681,  3454,  3227,
            3000,  2773,  2493,  1870,  1643,  1585,  1416,  1189,  0};
        std::array<uint64_t, 312> polynomial{};
        polynomial[0] = 1;
        int highest = 63;
        while (highest >= 0 && !(count & (uint64_t(1) << highest)))
            --highest;
        for (int bit = highest; bit >= 0; --bit) {
            std::array<uint64_t, 624> squared{};
            for (int i = 0; i < degree; ++i)
                if ((polynomial[i / 64] >> (i % 64)) & 1u)
                    squared[(2 * i) / 64] |= uint64_t(1) << ((2 * i) % 64);
            for (int i = 2 * degree - 2; i >= degree; --i) {
                if ((squared[i / 64] >> (i % 64)) & 1u) {
                    for (int power : powers) {
                        int k = i - degree + power;
                        squared[k / 64] ^= uint64_t(1) << (k % 64);
                    }
                }
            }
            std::copy_n(squared.begin(), polynomial.size(), polynomial.begin());
            if ((count >> bit) & 1u) {
                uint64_t carry = 0;
                for (auto &word : polynomial) {
                    uint64_t next = word >> 63;
                    word = (word << 1) | carry;
                    carry = next;
                }
                if ((polynomial[degree / 64] >> (degree % 64)) & 1u)
                    for (int power : powers)
                        polynomial[power / 64] ^= uint64_t(1) << (power % 64);
            }
        }
        // A complete twist projects the seeded state into the invariant
        // 19937-bit state space and supplies the first raw output window.
        twist();
        uint32_t result[624]{};
        int head = 0;
        for (int bit = 0; bit < degree; ++bit) {
            if ((polynomial[bit / 64] >> (bit % 64)) & 1u) {
                int first = 624 - head;
#pragma omp simd
                for (int k = 0; k < first; ++k)
                    result[k] ^= words[head + k];
#pragma omp simd
                for (int k = first; k < 624; ++k)
                    result[k] ^= words[k - first];
            }
            uint32_t y = (words[head] & 0x80000000u) | (words[(head + 1) % 624] & 0x7fffffffu);
            words[head] = words[(head + 397) % 624] ^ (y >> 1) ^ ((y & 1u) * 0x9908b0dfu);
            head = (head + 1) % 624;
        }
        std::copy_n(result, 624, words);
        position = 0;
    }
    void seek(uint64_t count) {
        // Only called on a freshly seeded generator. Short skips are cheaper
        // than polynomial evaluation; large skips have logarithmic cost.
        if (count < 64000000)
            discard(count);
        else
            jump(count);
    }
};

__device__ void twistMT(uint32_t *words) {
    for (int begin = 0; begin < 623; begin += 227) {
        int k = begin + threadIdx.x;
        int end = min(begin + 227, 623);
        uint32_t result[2];
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            int i = k + q * blockDim.x;
            if (i < end) {
                uint32_t y = (words[i] & 0x80000000u) | (words[i + 1] & 0x7fffffffu);
                result[q] = words[i < 227 ? i + 397 : i - 227] ^ (y >> 1) ^ ((y & 1u) * 0x9908b0dfu);
            }
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            int i = k + q * blockDim.x;
            if (i < end)
                words[i] = result[q];
        }
        __syncthreads();
    }
    if (!threadIdx.x) {
        uint32_t y = (words[623] & 0x80000000u) | (words[0] & 0x7fffffffu);
        words[623] = words[396] ^ (y >> 1) ^ ((y & 1u) * 0x9908b0dfu);
    }
    __syncthreads();
}

__device__ val_t uniformMT(uint32_t value) {
    value ^= value >> 11;
    value ^= (value << 7) & 0x9d2c5680u;
    value ^= (value << 15) & 0xefc60000u;
    value ^= value >> 18;
    // Match std::uniform_real_distribution<float>, including its upper clamp.
    val_t result = static_cast<val_t>(value) * (1.0f / 4294967296.0f);
    return result < ONE ? result : 0x1.fffffep-1f;
}

__device__ Vec3 sampleTriangle(const Triangle &tri, val_t u, val_t v) {
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    Vec3 ab = tri.b - tri.a, ac = tri.c - tri.a;
    return {fmaf(ac.x, v, fmaf(ab.x, u, tri.a.x)), fmaf(ac.y, v, fmaf(ab.y, u, tri.a.y)),
            fmaf(ac.z, v, fmaf(ab.z, u, tri.a.z))};
}

__device__ val_t cosPhi(const Vec3 &v, const Vec3 &normal) {
    val_t norm = v.norm();
    return norm <= EPSILON ? ZERO : fmaxf(ZERO, v.dot(normal) / norm);
}

struct Pair {
    idx_t i, j;
};
constexpr size_t PAIRS_PER_BLOCK = 64;
constexpr size_t PAIRS_PER_BATCH = 262144;

__global__ void formFactorKernel(const Triangle *triangles, const FlatNode *nodes, const idx_t *indices,
                                 const val_t *areas, const Pair *pairs, const MTState *checkpoints,
                                 size_t pairCount, size_t firstRow, size_t rows, val_t *coupling,
                                 const int *delays, unsigned long long *nonzero, int *minimumDelay) {
    __shared__ uint32_t words[624];
    __shared__ int position;
    __shared__ val_t samples[512], contributions[128];
    __shared__ unsigned counts[8];
    __shared__ int minimums[8];
    const MTState &checkpoint = checkpoints[blockIdx.x];
    for (int k = threadIdx.x; k < 624; k += blockDim.x)
        words[k] = checkpoint.words[k];
    if (!threadIdx.x)
        position = checkpoint.position;
    __syncthreads();
    unsigned count = 0;
    int minimum = INT_MAX;
    const int ray = threadIdx.x % NUM_RAYS, pairInGroup = threadIdx.x / NUM_RAYS;
    const size_t blockStart = size_t(blockIdx.x) * PAIRS_PER_BLOCK;
    for (size_t offset = 0; offset < PAIRS_PER_BLOCK && blockStart + offset < pairCount; offset += 8) {
        int active = static_cast<int>(min(size_t(8), pairCount - blockStart - offset));
        int required = active * NUM_RAYS * 4;
        int copied = 0;
        while (copied < required) {
            if (position == 624) {
                twistMT(words);
                if (!threadIdx.x)
                    position = 0;
                __syncthreads();
            }
            int chunk = min(required - copied, 624 - position);
            for (int k = threadIdx.x; k < chunk; k += blockDim.x)
                samples[copied + k] = uniformMT(words[position + k]);
            __syncthreads();
            if (!threadIdx.x)
                position += chunk;
            __syncthreads();
            copied += chunk;
        }
        val_t contribution = ZERO;
        Pair pair{};
        if (pairInGroup < active) {
            pair = pairs[blockStart + offset + pairInGroup];
            const Triangle &a = triangles[pair.i];
            const Triangle &b = triangles[pair.j];
            int s = pairInGroup * 64 + ray * 4;
            Vec3 pI = sampleTriangle(a, samples[s], samples[s + 1]);
            Vec3 pJ = sampleTriangle(b, samples[s + 2], samples[s + 3]);
            if (!isRayBlocked(pI, pJ, triangles, nodes, indices, pair.i, pair.j)) {
                Vec3 v = pJ - pI;
                val_t distanceSquared = v.squaredNorm();
                if (distanceSquared >= EPSILON) {
                    val_t ca = cosPhi(v, a.normal()), cb = cosPhi(-v, b.normal());
                    if (ca > ZERO && cb > ZERO)
                        contribution = (ca * cb) / (PI * distanceSquared);
                }
            }
        }
        contributions[threadIdx.x] = contribution;
        __syncthreads();
        if (!ray && pairInGroup < active) {
            val_t sum = ZERO;
            // Preserve the sequential ray sum rather than changing its rounding.
            for (int r = 0; r < NUM_RAYS; ++r)
                sum += contributions[threadIdx.x + r];
            val_t kij = sum * INV_NUM_RAYS;
            count += kij > EPSILON;
            size_t k = size_t(pair.j) * rows + pair.i - firstRow;
            coupling[k] = fminf(kij * areas[pair.j], ONE);
            if (coupling[k] > ZERO)
                minimum = min(minimum, delays[k]);
        }
        __syncthreads();
    }
    if (!ray) {
        counts[pairInGroup] = count;
        minimums[pairInGroup] = minimum;
    }
    __syncthreads();
    if (!threadIdx.x) {
        unsigned sum = 0;
        for (int k = 0; k < 8; ++k) {
            sum += counts[k];
            minimum = min(minimum, minimums[k]);
        }
        atomicAdd(nonzero, static_cast<unsigned long long>(sum));
        atomicMin(minimumDelay, minimum);
    }
}

__global__ void delayKernel(const Vec3 *centers, size_t n, size_t firstRow, size_t rows, int *delays,
                            int *minimum) {
    __shared__ int mins[256];
    int localMin = INT_MAX;
    for (size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x; k < n * rows;
         k += size_t(blockDim.x) * gridDim.x) {
        size_t i = firstRow + k % rows, j = k / rows;
        int delay = i == j ? 0 : static_cast<int>(ceilf((centers[i] - centers[j]).norm() * INV_WAVE_SPEED));
        delays[k] = delay;
        if (i != j)
            localMin = min(localMin, delay);
    }
    mins[threadIdx.x] = localMin;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride)
            mins[threadIdx.x] = min(mins[threadIdx.x], mins[threadIdx.x + stride]);
        __syncthreads();
    }
    if (!threadIdx.x)
        atomicMin(minimum, mins[0]);
}

__global__ void propagationKernel(const val_t *coupling, const int *delays, val_t *history,
                                  val_t *localHistory, val_t *batch, size_t n, size_t rows, size_t firstRow,
                                  size_t time, size_t steps, size_t timesteps, size_t source, val_t rho,
                                  bool singleRank) {
    size_t step = blockIdx.y;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < rows;
         i += size_t(blockDim.x) * gridDim.x) {
        size_t t = time + step;
        val_t sum = ZERO;
        // Adjacent threads handle adjacent receivers: column-major matrices
        // give coalesced loads without reordering the emitter summation.
        for (size_t j = 0; j < n; ++j) {
            size_t k = j * rows + i;
            int delay = delays[k];
            if (i + firstRow == j || t < static_cast<size_t>(delay))
                continue;
            val_t weight = coupling[k];
            if (weight <= ZERO)
                continue;
            val_t radiosity = history[(t - delay) * n + j];
            if (radiosity > ZERO)
                sum = fmaf(weight, radiosity, sum);
        }
        val_t emission = i + firstRow == source && t < timesteps / 2 ? ONE : ZERO;
        val_t result = fmaf(rho, sum, emission);
        localHistory[i * timesteps + t] = result;
        if (singleRank)
            history[t * n + i] = result;
        else
            batch[i * steps + step] = result;
    }
}

__global__ void unpackKernel(const val_t *batch, val_t *history, size_t n, size_t time, size_t steps) {
    for (size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x; k < n * steps;
         k += size_t(blockDim.x) * gridDim.x)
        history[(time + k % steps) * n + k / steps] = batch[k];
}

__global__ void distanceKernel(const val_t *localHistory, const val_t *history, size_t n, size_t rows,
                               size_t timesteps, size_t source, val_t *distances,
                               unsigned long long *received) {
    __shared__ val_t scores[128];
    __shared__ size_t lags[128];
    for (size_t i = blockIdx.x; i < rows; i += gridDim.x) {
        val_t best = ZERO;
        size_t lag = 0;
        for (size_t t = threadIdx.x; t < timesteps; t += blockDim.x) {
            val_t sum = ZERO;
            for (size_t tt = t; tt < timesteps; ++tt)
                sum = fmaf(history[(tt - t) * n + source], localHistory[i * timesteps + tt], sum);
            if (sum > best || (sum == best && t < lag)) {
                best = sum;
                lag = t;
            }
        }
        scores[threadIdx.x] = best;
        lags[threadIdx.x] = lag;
        __syncthreads();
        for (int stride = 64; stride; stride /= 2) {
            if (threadIdx.x < stride) {
                val_t other = scores[threadIdx.x + stride];
                size_t otherLag = lags[threadIdx.x + stride];
                if (other > scores[threadIdx.x] ||
                    (other == scores[threadIdx.x] && otherLag < lags[threadIdx.x])) {
                    scores[threadIdx.x] = other;
                    lags[threadIdx.x] = otherLag;
                }
            }
            __syncthreads();
        }
        if (!threadIdx.x) {
            distances[i] = WAVE_SPEED * static_cast<val_t>(lags[0]);
            for (size_t t = 0; t < timesteps; ++t) {
                if (localHistory[i * timesteps + t] > EPSILON) {
                    atomicAdd(received, 1ULL);
                    break;
                }
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0, numTimesteps = 0, sourceIndex = 0;
    size_t firstRow = 0, rows = 0;
    int rank = 0, ranks = 1, minimumDelay = 1;
    val_t reflectivity = 0.8f;
    unsigned long long nonZeroKij = 0, receivedEnergy = 0;
    std::vector<Triangle> triangles;
    std::vector<val_t> areas, distances;
    std::vector<int> rowCounts, rowStarts;
    std::vector<uint64_t> eligibleRows;
    Octree octree;
    DeviceArray<Triangle> dTriangles;
    DeviceArray<FlatNode> dNodes;
    DeviceArray<idx_t> dIndices;
    DeviceArray<Vec3> dCenters;
    DeviceArray<val_t> dAreas, dCoupling, dHistory, dLocalHistory, dBatch, dDistances;
    DeviceArray<int> dDelays, dMinimum;
    DeviceArray<unsigned long long> dNonzero, dReceived;
    cudaStream_t stream = nullptr;
    ~SimulationState() {
        if (stream)
            cudaStreamDestroy(stream);
    }
};

int gridSize(size_t n, int threads = 256) {
    return static_cast<int>(std::min(size_t(65535), (n + threads - 1) / threads));
}

void initializeSimulation(SimulationState &state, const ParallelRuntime &runtime, int subdivisions,
                          size_t timesteps, size_t source, val_t rho) {
    state.rank = runtime.rank;
    state.ranks = runtime.ranks;
    state.reflectivity = rho;
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    const size_t n = state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = source % n;
    state.rowCounts.resize(state.ranks);
    state.rowStarts.resize(state.ranks);
    for (int r = 0; r < state.ranks; ++r) {
        state.rowStarts[r] = static_cast<int>(n * r / state.ranks);
        state.rowCounts[r] = static_cast<int>(n * (r + 1) / state.ranks - state.rowStarts[r]);
    }
    state.firstRow = state.rowStarts[state.rank];
    state.rows = state.rowCounts[state.rank];
    if (state.rank == 0)
        printf("Generated icosphere mesh with %zu triangles\nBuilding octree...\n", n);
#pragma omp parallel
    {
#pragma omp single
        state.octree.build(state.triangles);
    }
    std::vector<FlatNode> nodes;
    std::vector<idx_t> indices;
    flattenOctree(state.octree, nodes, indices);
    std::vector<Vec3> centers(n);
    state.areas.resize(n);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        state.areas[i] = state.triangles[i].area();
        centers[i] = state.triangles[i].center();
    }
    state.eligibleRows.resize(n);
    std::vector<uint64_t> localEligible(state.rows);
#pragma omp parallel for schedule(static)
    for (size_t row = 0; row < state.rows; ++row) {
        size_t i = state.firstRow + row;
        uint64_t count = 0;
        for (size_t j = 0; j < n; ++j)
            count += i != j && !(state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f);
        localEligible[row] = count;
    }
    mpiCheck(MPI_Allgatherv(localEligible.data(), static_cast<int>(state.rows), MPI_UINT64_T,
                            state.eligibleRows.data(), state.rowCounts.data(), state.rowStarts.data(),
                            MPI_UINT64_T, MPI_COMM_WORLD));
    cudaCheck(cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking));
    state.dTriangles.resize(n);
    state.dCenters.resize(n);
    state.dAreas.resize(n);
    state.dNodes.resize(nodes.size());
    state.dIndices.resize(indices.size());
    state.dCoupling.resize(state.rows * n);
    state.dDelays.resize(state.rows * n);
    state.dHistory.resize(timesteps * n);
    state.dLocalHistory.resize(timesteps * state.rows);
    state.dDistances.resize(state.rows);
    state.dMinimum.resize(1);
    state.dNonzero.resize(1);
    state.dReceived.resize(1);
    cudaCheck(cudaMemcpyAsync(state.dTriangles.get(), state.triangles.data(), n * sizeof(Triangle),
                              cudaMemcpyHostToDevice, state.stream));
    cudaCheck(cudaMemcpyAsync(state.dCenters.get(), centers.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice,
                              state.stream));
    cudaCheck(cudaMemcpyAsync(state.dAreas.get(), state.areas.data(), n * sizeof(val_t),
                              cudaMemcpyHostToDevice, state.stream));
    cudaCheck(cudaMemcpyAsync(state.dNodes.get(), nodes.data(), nodes.size() * sizeof(FlatNode),
                              cudaMemcpyHostToDevice, state.stream));
    cudaCheck(cudaMemcpyAsync(state.dIndices.get(), indices.data(), indices.size() * sizeof(idx_t),
                              cudaMemcpyHostToDevice, state.stream));
    if (state.rows)
        cudaCheck(cudaMemsetAsync(state.dCoupling.get(), 0, state.rows * n * sizeof(val_t), state.stream));
    if (timesteps)
        cudaCheck(cudaMemsetAsync(state.dHistory.get(), 0, timesteps * n * sizeof(val_t), state.stream));
    cudaCheck(cudaMemsetAsync(state.dNonzero.get(), 0, sizeof(unsigned long long), state.stream));
    cudaCheck(cudaMemsetAsync(state.dReceived.get(), 0, sizeof(unsigned long long), state.stream));
    cudaCheck(cudaStreamSynchronize(state.stream));
    if (state.rank == 0)
        state.distances.resize(n);
}

void computeTimeDelays(SimulationState &state) {
    if (!state.rank)
        printf("Computing time delays (Tau)...\n");
    int minimum = INT_MAX;
    cudaCheck(
        cudaMemcpyAsync(state.dMinimum.get(), &minimum, sizeof(int), cudaMemcpyHostToDevice, state.stream));
    if (state.rows) {
        delayKernel<<<gridSize(state.rows * state.numTriangles), 256, 0, state.stream>>>(
            state.dCenters.get(), state.numTriangles, state.firstRow, state.rows, state.dDelays.get(),
            state.dMinimum.get());
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(
        cudaMemcpyAsync(&minimum, state.dMinimum.get(), sizeof(int), cudaMemcpyDeviceToHost, state.stream));
    cudaCheck(cudaStreamSynchronize(state.stream));
    mpiCheck(MPI_Allreduce(&minimum, &state.minimumDelay, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if (state.minimumDelay <= 0)
        throw std::runtime_error("Distinct mesh triangles must have a positive propagation delay");
}

void computeFormFactors(SimulationState &state) {
    if (!state.rank)
        printf("Computing form factors (Kij) on distributed CUDA accelerators...\n");
    uint64_t prefix = 0, total = 0;
    for (size_t i = 0; i < state.firstRow; ++i)
        prefix += state.eligibleRows[i];
    for (size_t i = state.firstRow; i < state.firstRow + state.rows; ++i)
        total += state.eligibleRows[i];
    int device = 0;
    cudaCheck(cudaGetDevice(&device));
    const int workers = static_cast<int>(std::min(
        {uint64_t(omp_get_max_threads()), uint64_t(8), std::max(uint64_t(1), (total + 65535) / 65536)}));
    std::exception_ptr failure;
    int minimum = INT_MAX;
    cudaCheck(cudaMemcpyAsync(state.dMinimum.get(), &minimum, sizeof(minimum), cudaMemcpyHostToDevice,
                              state.stream));
    cudaCheck(cudaStreamSynchronize(state.stream));
// Each worker owns a contiguous part of the original stream, two bounded
// staging buffers, and a CUDA stream. GPU tracing overlaps CPU preparation
// and transfers, without a serial random-number bottleneck on rank zero.
#pragma omp parallel num_threads(workers)
    {
        try {
            cudaCheck(cudaSetDevice(device));
            const uint64_t begin = total * omp_get_thread_num() / omp_get_num_threads();
            const uint64_t end = total * (omp_get_thread_num() + 1) / omp_get_num_threads();
            uint64_t remaining = end - begin;
            if (remaining) {
                MTState rng;
                rng.seek((prefix + begin) * NUM_RAYS * 4);
                size_t i = state.firstRow, j = 0;
                uint64_t skipped = 0;
                while (skipped + state.eligibleRows[i] <= begin)
                    skipped += state.eligibleRows[i++];
                while (skipped < begin) {
                    if (i != j && !(state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f))
                        ++skipped;
                    ++j;
                }
                const size_t capacity = static_cast<size_t>(std::min(
                    uint64_t(PAIRS_PER_BATCH / workers / PAIRS_PER_BLOCK * PAIRS_PER_BLOCK), remaining));
                PinnedArray<Pair> hostPairs[2];
                PinnedArray<MTState> hostCheckpoints[2];
                DeviceArray<Pair> devicePairs[2];
                DeviceArray<MTState> deviceCheckpoints[2];
                struct WorkerStream {
                    cudaStream_t value = nullptr;
                    ~WorkerStream() {
                        if (value)
                            cudaStreamDestroy(value);
                    }
                } stream;
                cudaCheck(cudaStreamCreateWithFlags(&stream.value, cudaStreamNonBlocking));
                cudaEvent_t ready[2];
                const size_t checkpointCapacity = (capacity + PAIRS_PER_BLOCK - 1) / PAIRS_PER_BLOCK;
                for (int b = 0; b < 2; ++b) {
                    hostPairs[b].resize(capacity);
                    hostCheckpoints[b].resize(checkpointCapacity);
                    devicePairs[b].resize(capacity);
                    deviceCheckpoints[b].resize(checkpointCapacity);
                    cudaCheck(cudaEventCreateWithFlags(&ready[b], cudaEventDisableTiming));
                    cudaCheck(cudaEventRecord(ready[b], stream.value));
                }
                size_t batchNumber = 0;
                while (remaining) {
                    int b = batchNumber++ % 2;
                    cudaCheck(cudaEventSynchronize(ready[b]));
                    size_t count = static_cast<size_t>(std::min(uint64_t(capacity), remaining));
                    for (size_t k = 0; k < count; ++k) {
                        while (i == j ||
                               state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) {
                            if (++j == state.numTriangles) {
                                ++i;
                                j = 0;
                            }
                        }
                        hostPairs[b].get()[k] = {static_cast<idx_t>(i), static_cast<idx_t>(j)};
                        if (++j == state.numTriangles) {
                            ++i;
                            j = 0;
                        }
                    }
                    const size_t blocks = (count + PAIRS_PER_BLOCK - 1) / PAIRS_PER_BLOCK;
                    for (size_t block = 0; block < blocks; ++block) {
                        hostCheckpoints[b].get()[block] = rng;
                        rng.discard(std::min(PAIRS_PER_BLOCK, count - block * PAIRS_PER_BLOCK) * NUM_RAYS *
                                    4);
                    }
                    cudaCheck(cudaMemcpyAsync(devicePairs[b].get(), hostPairs[b].get(), count * sizeof(Pair),
                                              cudaMemcpyHostToDevice, stream.value));
                    cudaCheck(cudaMemcpyAsync(deviceCheckpoints[b].get(), hostCheckpoints[b].get(),
                                              blocks * sizeof(MTState), cudaMemcpyHostToDevice,
                                              stream.value));
                    formFactorKernel<<<static_cast<unsigned>(blocks), 128, 0, stream.value>>>(
                        state.dTriangles.get(), state.dNodes.get(), state.dIndices.get(), state.dAreas.get(),
                        devicePairs[b].get(), deviceCheckpoints[b].get(), count, state.firstRow, state.rows,
                        state.dCoupling.get(), state.dDelays.get(), state.dNonzero.get(),
                        state.dMinimum.get());
                    cudaCheck(cudaGetLastError());
                    cudaCheck(cudaEventRecord(ready[b], stream.value));
                    remaining -= count;
                }
                cudaCheck(cudaStreamSynchronize(stream.value));
                for (auto event : ready)
                    cudaCheck(cudaEventDestroy(event));
            }
        } catch (...) {
#pragma omp critical
            {
                if (!failure)
                    failure = std::current_exception();
            }
        }
    }
    if (failure)
        std::rethrow_exception(failure);
    unsigned long long localNonzero = 0;
    cudaCheck(cudaMemcpy(&localNonzero, state.dNonzero.get(), sizeof(localNonzero), cudaMemcpyDeviceToHost));
    mpiCheck(
        MPI_Allreduce(&localNonzero, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));
    cudaCheck(cudaMemcpy(&minimum, state.dMinimum.get(), sizeof(minimum), cudaMemcpyDeviceToHost));
    // Culled/zero couplings impose no dependency. Their short geometric delays
    // must not limit temporal parallelism or cause needless MPI exchanges.
    mpiCheck(MPI_Allreduce(&minimum, &state.minimumDelay, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
}

void runSimulation(SimulationState &state) {
    if (!state.rank)
        printf("Running wave propagation simulation...\n");
    const size_t n = state.numTriangles, timesteps = state.numTimesteps;
    if (!timesteps)
        return;
    // Every read precedes this entire batch, so steps can run concurrently.
    const size_t batchLimit =
        std::min({size_t(state.minimumDelay), timesteps, size_t(INT_MAX) / n, size_t(65535)});
    PinnedArray<val_t> send, receive;
    if (state.ranks > 1) {
        send.resize(state.rows * batchLimit);
        receive.resize(n * batchLimit);
        state.dBatch.resize(n * batchLimit);
    }
    std::vector<int> counts(state.ranks), starts(state.ranks);
    for (size_t time = 0; time < timesteps; time += batchLimit) {
        size_t steps = std::min(batchLimit, timesteps - time);
        if (state.rows) {
            propagationKernel<<<dim3(gridSize(state.rows, 128), static_cast<unsigned>(steps)), 128, 0,
                                state.stream>>>(
                state.dCoupling.get(), state.dDelays.get(), state.dHistory.get(), state.dLocalHistory.get(),
                state.dBatch.get(), n, state.rows, state.firstRow, time, steps, timesteps, state.sourceIndex,
                state.reflectivity, state.ranks == 1);
            cudaCheck(cudaGetLastError());
        }
        if (state.ranks > 1) {
            if (state.rows)
                cudaCheck(cudaMemcpyAsync(send.get(), state.dBatch.get(), state.rows * steps * sizeof(val_t),
                                          cudaMemcpyDeviceToHost, state.stream));
            cudaCheck(cudaStreamSynchronize(state.stream));
            for (int r = 0; r < state.ranks; ++r) {
                counts[r] = static_cast<int>(state.rowCounts[r] * steps);
                starts[r] = static_cast<int>(state.rowStarts[r] * steps);
            }
            mpiCheck(MPI_Allgatherv(send.get(), counts[state.rank], MPI_FLOAT, receive.get(), counts.data(),
                                    starts.data(), MPI_FLOAT, MPI_COMM_WORLD));
            cudaCheck(cudaMemcpyAsync(state.dBatch.get(), receive.get(), n * steps * sizeof(val_t),
                                      cudaMemcpyHostToDevice, state.stream));
            unpackKernel<<<gridSize(n * steps), 256, 0, state.stream>>>(state.dBatch.get(),
                                                                        state.dHistory.get(), n, time, steps);
            cudaCheck(cudaGetLastError());
            // The receive buffer is reused by MPI at the next batch.
            cudaCheck(cudaStreamSynchronize(state.stream));
        }
        if (!state.rank && ((time + steps) % 10 == 0 || time + steps == timesteps))
            printf("  Timestep %zu/%zu\n", time + steps, timesteps);
    }
    cudaCheck(cudaStreamSynchronize(state.stream));
}

void computeDistances(SimulationState &state) {
    if (!state.rank)
        printf("Computing distances via cross-correlation...\n");
    if (state.rows) {
        distanceKernel<<<static_cast<unsigned>(std::min(state.rows, size_t(65535))), 128, 0, state.stream>>>(
            state.dLocalHistory.get(), state.dHistory.get(), state.numTriangles, state.rows,
            state.numTimesteps, state.sourceIndex, state.dDistances.get(), state.dReceived.get());
        cudaCheck(cudaGetLastError());
    }
    std::vector<val_t> localDistances(state.rows);
    if (state.rows)
        cudaCheck(cudaMemcpyAsync(localDistances.data(), state.dDistances.get(), state.rows * sizeof(val_t),
                                  cudaMemcpyDeviceToHost, state.stream));
    unsigned long long received = 0;
    cudaCheck(cudaMemcpyAsync(&received, state.dReceived.get(), sizeof(received), cudaMemcpyDeviceToHost,
                              state.stream));
    cudaCheck(cudaStreamSynchronize(state.stream));
    mpiCheck(MPI_Gatherv(localDistances.data(), static_cast<int>(state.rows), MPI_FLOAT,
                         state.distances.data(), state.rowCounts.data(), state.rowStarts.data(), MPI_FLOAT, 0,
                         MPI_COMM_WORLD));
    mpiCheck(
        MPI_Allreduce(&received, &state.receivedEnergy, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState &state) {
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
        if (d > EPSILON)
            nonZeroCount++;
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
    printf("  Triangles receiving energy: %llu/%zu\n", state.receivedEnergy, state.numTriangles);
    if (state.receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n", state.nonZeroKij,
           state.numTriangles * state.numTriangles,
           100.0f * state.nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));
    if (state.nonZeroKij == 0) {
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

uint64_t computeHash(const SimulationState &state) {
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

void printUsage(const char *progName) {
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

int main(int argc, char **argv) {
    ParallelRuntime runtime(argc, argv);
    try {
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
                if (!runtime.rank)
                    printUsage(argv[0]);
                return 0;
            } else {
                if (!runtime.rank) {
                    printf("Unknown option: %s\n", argv[i]);
                    printUsage(argv[0]);
                }
                return 1;
            }
        }

        if (timesteps < 0)
            throw std::runtime_error("Timesteps must be non-negative");
        runtime.selectDevice();
        int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
        if (!runtime.rank) {

            printf("Room Response Simulation Benchmark\n");
            printf("===================================\n");
            printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
            printf("Timesteps: %d\n", timesteps);
            printf("Source triangle: %d\n", sourceIdx);
            printf("Reflectivity: %.2f\n", reflectivity);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("\n");

            printf("MPI ranks: %d; OpenMP threads per rank: %d; CUDA enabled\n", runtime.ranks,
                   omp_get_max_threads());
        }
        // Initialize
        SimulationState state;
        initializeSimulation(state, runtime, subdivisions, static_cast<size_t>(timesteps),
                             static_cast<size_t>(sourceIdx), reflectivity);

        if (!runtime.rank)
            printf("\n");

        // Precomputation
        mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
        double startPre = MPI_Wtime();

        computeTimeDelays(state);
        computeFormFactors(state);

        mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
        long preDuration = static_cast<long>((MPI_Wtime() - startPre) * 1000);
        if (!runtime.rank)
            printf("Precomputation time: %ld ms\n\n", preDuration);

        // Simulation
        double startSim = MPI_Wtime();

        runSimulation(state);

        mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
        long simDuration = static_cast<long>((MPI_Wtime() - startSim) * 1000);
        if (!runtime.rank)
            printf("Simulation time: %ld ms\n\n", simDuration);

        // Distance computation
        double startDist = MPI_Wtime();

        computeDistances(state);

        mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
        long distDuration = static_cast<long>((MPI_Wtime() - startDist) * 1000);
        uint64_t matrixMemory = state.rows * state.numTriangles * (sizeof(val_t) + sizeof(int)) +
                                state.numTimesteps * (state.numTriangles + state.rows) * sizeof(val_t);
        uint64_t maximumMatrixMemory = 0;
        mpiCheck(
            MPI_Allreduce(&matrixMemory, &maximumMatrixMemory, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD));
        if (!runtime.rank) {
            printf("Distance computation time: %ld ms\n\n", distDuration);

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
            printf("  Matrix/history memory per rank (maximum): %.2f MB\n",
                   maximumMatrixMemory / (1024.0 * 1024.0));

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
        }
        int status = 0;
        if (!runtime.rank && validate && !validateResults(state))
            status = 1;
        mpiCheck(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
        return status;
    } catch (const std::exception &error) {
        fprintf(stderr, "Rank %d: %s\n", runtime.rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
