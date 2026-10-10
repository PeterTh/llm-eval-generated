/**
 * Room Response Simulation Benchmark
 * 
 * This is a distributed MPI/OpenMP/CUDA implementation of room impulse response
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
#include <climits>
#include <cfloat>
#include <stdexcept>
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#ifdef OMPI_MAJOR_VERSION
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>
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

#include "../common/results_output.hpp"

// MPI calls are confined to the main thread. Every rank owns a contiguous set
// of receivers and selects a GPU using its node-local rank.
static int processRank = 0;

void checkMPI(int error, const char* operation) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(error, message, &length);
    fprintf(stderr, "Rank %d: %s: %.*s\n", processRank, operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

void checkCUDA(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    fprintf(stderr, "Rank %d: %s: %s\n", processRank, operation, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define MPI_CALL(operation) checkMPI((operation), #operation)
#define CUDA_CALL(operation) checkCUDA((operation), #operation)

struct ParallelContext {
    int rank = 0, size = 1, localRank = 0, localSize = 1, device = 0;
    bool cudaAware = false;
    ParallelContext(int& argc, char**& argv) {
        int provided = 0;
        MPI_CALL(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
        MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        processRank = rank;
        MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
        MPI_CALL(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
        if (provided < MPI_THREAD_FUNNELED) {
            fprintf(stderr, "MPI_THREAD_FUNNELED is required.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Comm node;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
        MPI_CALL(MPI_Comm_rank(node, &localRank));
        MPI_CALL(MPI_Comm_size(node, &localSize));
        MPI_CALL(MPI_Comm_free(&node));
        if (!std::getenv("OMP_NUM_THREADS"))
            omp_set_num_threads(std::max(1, std::min(32, omp_get_max_threads() / localSize)));
        int devices = 0;
        CUDA_CALL(cudaGetDeviceCount(&devices));
        if (devices == 0) {
            fprintf(stderr, "Rank %d: a CUDA device is required.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        device = localRank % devices;
        CUDA_CALL(cudaSetDevice(device));
        CUDA_CALL(cudaFree(nullptr)); // Initialize the context outside measured phases.
        int support = 0;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        support = MPIX_Query_cuda_support();
#endif
        int allSupport = 0;
        MPI_CALL(MPI_Allreduce(&support, &allSupport, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        cudaAware = allSupport != 0;
    }
    ~ParallelContext() { MPI_Finalize(); }
};

template<class T> class DeviceBuffer {
public:
    T* data = nullptr;
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    void allocate(size_t count) {
        CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template<class T> class PinnedBuffer {
public:
    T* data = nullptr;
    explicit PinnedBuffer(size_t count) {
        CUDA_CALL(cudaMallocHost(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~PinnedBuffer() { cudaFreeHost(data); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

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

    __host__ __device__ bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    __host__ __device__ Triangle() = default;
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

                #pragma omp task shared(childIndices) firstprivate(i, childMin, childMax) if(indices.size() > 256)
                {
                    children[i] = std::make_unique<Octree>();
                    children[i]->allTriangles = allTriangles;
                    children[i]->buildNode(childIndices[i], childMin, childMax);
                }
            }
        }
        #pragma omp taskwait
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
// Random Number Generation
// ============================================================================

// Compact, portable checkpoints of the original MT19937 stream. Unlike
// per-pair reseeding, these preserve every sample and the conditional culling.
struct RandomState {
    uint32_t words[624];
    uint32_t position;
};
static_assert(sizeof(RandomState) == 625 * sizeof(uint32_t), "MT checkpoints must contain packed words");

class RandomGenerator {
    RandomState state;
    std::uniform_real_distribution<val_t> dist{0.0f, 1.0f};
    void twist() {
        #pragma omp simd
        for (int i = 0; i < 227; ++i) {
            uint32_t y = (state.words[i] & 0x80000000u) | (state.words[i + 1] & 0x7fffffffu);
            state.words[i] = state.words[i + 397] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        }
        #pragma omp simd simdlen(8)
        for (int i = 227; i < 623; ++i) {
            uint32_t y = (state.words[i] & 0x80000000u) | (state.words[i + 1] & 0x7fffffffu);
            state.words[i] = state.words[i - 227] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        }
        uint32_t y = (state.words[623] & 0x80000000u) | (state.words[0] & 0x7fffffffu);
        state.words[623] = state.words[396] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        state.position = 0;
    }
public:
    using result_type = uint32_t;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT32_MAX; }
    explicit RandomGenerator(uint32_t seed = 42) {
        state.words[0] = seed;
        for (uint32_t i = 1; i < 624; ++i)
            state.words[i] = 1812433253u * (state.words[i - 1] ^ (state.words[i - 1] >> 30)) + i;
        state.position = 624;
    }
    explicit RandomGenerator(const RandomState& checkpoint) : state(checkpoint) {}
    RandomState checkpoint() const { return state; }
    result_type operator()() {
        if (state.position == 624) twist();
        uint32_t y = state.words[state.position++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        return y ^ (y >> 18);
    }
    void discard(size_t count) {
        while (count) {
            if (state.position == 624) twist();
            size_t step = std::min(count, size_t(624 - state.position));
            state.position += static_cast<uint32_t>(step);
            count -= step;
        }
    }
    val_t rand() { return dist(*this); }
    void fill(val_t* output, size_t count) {
        while (count) {
            if (state.position == 624) twist();
            size_t step = std::min(count, size_t(624 - state.position));
            const uint32_t* input = state.words + state.position;
            // This is generate_canonical<float, 24> for a 32-bit MT engine,
            // including the standard distribution's rounding clamp at one.
            #pragma omp simd
            for (size_t k = 0; k < step; ++k) {
                uint32_t y = input[k];
                y ^= y >> 11;
                y ^= (y << 7) & 0x9d2c5680u;
                y ^= (y << 15) & 0xefc60000u;
                y ^= y >> 18;
                val_t value = static_cast<val_t>(y) * 0x1p-32f;
                output[k] = value < ONE ? value : 0x1.fffffep-1f;
            }
            output += step;
            state.position += static_cast<uint32_t>(step);
            count -= step;
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

    if (std::abs(det) < EPSILON) return FLT_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return e2.dot(qvec) * invDet;
}

// A preorder octree with escape links permits stackless traversal on the GPU.
struct FlatNode {
    Vec3 center, halfExtent;
    uint32_t first, count, escape;
};

uint32_t flattenOctree(const Octree& tree, std::vector<FlatNode>& nodes,
                       std::vector<uint32_t>& indices) {
    uint32_t current = static_cast<uint32_t>(nodes.size());
    nodes.push_back({tree.center, tree.halfExtent, static_cast<uint32_t>(indices.size()),
                     static_cast<uint32_t>(tree.triangleIndices.size()), 0});
    for (size_t i : tree.triangleIndices) indices.push_back(static_cast<uint32_t>(i));
    for (const auto& child : tree.children)
        if (child) flattenOctree(*child, nodes, indices);
    nodes[current].escape = static_cast<uint32_t>(nodes.size());
    return current;
}

__device__ bool rayBox(const Vec3& from, const Vec3& to, const FlatNode& node) {
    Vec3 d = (to - from) * 0.5f;
    Vec3 c = from + d - node.center;
    Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    if (fabsf(c.x) > node.halfExtent.x + ad.x ||
        fabsf(c.y) > node.halfExtent.y + ad.y ||
        fabsf(c.z) > node.halfExtent.z + ad.z) return false;
    if (fabsf(d.y * c.z - d.z * c.y) > node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;
    return true;
}

__device__ bool rayBlocked(const Vec3& from, const Vec3& to, const Triangle* triangles,
                           const FlatNode* nodes, const uint32_t* indices,
                           uint32_t source, uint32_t destination) {
    Vec3 dir = to - from;
    val_t length = dir.norm();
    if (length < EPSILON) return true;
    Vec3 normal = dir / length;
    uint32_t current = 0;
    while (current < nodes[0].escape) {
        const FlatNode& node = nodes[current];
        // The original implementation tests child boxes but not the root box.
        if (current && !rayBox(from, to, node)) {
            current = node.escape;
            continue;
        }
        for (uint32_t k = 0; k < node.count; ++k) {
            uint32_t index = indices[node.first + k];
            if (index == source || index == destination) continue;
            const Triangle& tri = triangles[index];
            val_t distance = rayTriangleIntersect(from, normal, tri.a, tri.b, tri.c);
            if (distance > EPSILON && distance < length - EPSILON) return true;
        }
        ++current;
    }
    return false;
}

__device__ Vec3 samplePoint(const Triangle& tri, val_t u, val_t v) {
    if (u + v > ONE) { u = ONE - u; v = ONE - v; }
    return tri.a + (tri.b - tri.a) * u + (tri.c - tri.a) * v;
}

// Sixteen lanes cooperate on each pair. Only the final sixteen-term sum is
// serial, preserving the original sample order and floating-point accumulation.
__global__ void formFactorKernel(const Triangle* triangles, const Vec3* centers,
                                 const FlatNode* nodes, const uint32_t* indices,
                                 const val_t* samples, val_t* kij, int* tau,
                                 size_t n, size_t localRows, size_t rowStart,
                                 size_t batchStart, size_t batchRows) {
    size_t thread = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t pair = thread / NUM_RAYS;
    int ray = threadIdx.x % NUM_RAYS;
    if (pair >= batchRows * n) return;
    size_t row = batchStart + pair / n;
    size_t i = rowStart + row, j = pair % n;
    size_t entry = j * localRows + row;
    if (ray == 0)
        tau[entry] = i == j ? 0 : static_cast<int>(ceilf((centers[i] - centers[j]).norm() * INV_WAVE_SPEED));
    const Triangle& triI = triangles[i];
    const Triangle& triJ = triangles[j];
    val_t contribution = ZERO;
    if (i != j && triI.normal().dot(triJ.normal()) <= 0.99f) {
        const val_t* sample = samples + (pair * NUM_RAYS + ray) * 4;
        Vec3 pI = samplePoint(triI, sample[0], sample[1]);
        Vec3 pJ = samplePoint(triJ, sample[2], sample[3]);
        if (!rayBlocked(pI, pJ, triangles, nodes, indices, static_cast<uint32_t>(i), static_cast<uint32_t>(j))) {
            Vec3 v = pJ - pI;
            val_t squared = v.squaredNorm();
            if (squared >= EPSILON) {
                val_t norm = v.norm();
                val_t cosI = norm <= EPSILON ? ZERO : fmaxf(ZERO, v.dot(triI.normal()) / norm);
                val_t cosJ = norm <= EPSILON ? ZERO : fmaxf(ZERO, (-v).dot(triJ.normal()) / norm);
                if (cosI > ZERO && cosJ > ZERO)
                    contribution = (cosI * cosJ) / (PI * squared);
            }
        }
    }
    unsigned mask = __activemask();
    val_t sum = ZERO;
    #pragma unroll
    for (int r = 0; r < NUM_RAYS; ++r) {
        val_t term = __shfl_sync(mask, contribution, r, NUM_RAYS);
        if (ray == 0) sum += term;
    }
    if (ray == 0) kij[entry] = sum * INV_NUM_RAYS;
}

__global__ void matrixStatsKernel(val_t* kij, const int* tau, const val_t* areas,
                                  size_t localRows, size_t entries,
                                  unsigned long long* nonzero, int* minDelay) {
    unsigned long long count = 0;
    int delay = INT_MAX;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < entries;
         i += size_t(blockDim.x) * gridDim.x) {
        if (kij[i] > EPSILON) ++count;
        if (kij[i] > ZERO) delay = min(delay, tau[i]);
        // Cache the clamped area-weighted coupling once for all timesteps.
        kij[i] = fminf(kij[i] * areas[i / localRows], ONE);
    }
    __shared__ unsigned long long counts[256];
    __shared__ int delays[256];
    counts[threadIdx.x] = count;
    delays[threadIdx.x] = delay;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            counts[threadIdx.x] += counts[threadIdx.x + stride];
            delays[threadIdx.x] = min(delays[threadIdx.x], delays[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(nonzero, counts[0]);
        atomicMin(minDelay, delays[0]);
    }
}

// Receiver-major matrix entries are transposed for coalesced accesses by
// neighboring receiver threads. Each receiver sums emitters in original order.
__global__ void propagationKernel(const val_t* kij, const int* tau,
                                  val_t* history, val_t* exchange, size_t n,
                                  size_t localRows, size_t rowStart, size_t timeStart,
                                  size_t batch, size_t timesteps, size_t source, val_t rho) {
    size_t item = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= localRows * batch) return;
    size_t row = item % localRows, offset = item / localRows;
    size_t i = rowStart + row, t = timeStart + offset;
    val_t sum = ZERO;
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        size_t entry = j * localRows + row;
        int delay = tau[entry];
        if (t < static_cast<size_t>(delay)) continue;
        val_t factor = kij[entry];
        if (factor <= ZERO) continue;
        val_t value = history[(t - static_cast<size_t>(delay)) * n + j];
        if (value <= ZERO) continue;
        sum += factor * value;
    }
    val_t emission = i == source && t < timesteps / 2 ? ONE : ZERO;
    val_t value = rho * sum + emission;
    history[t * n + i] = value;
    if (exchange) exchange[row * batch + offset] = value;
}

__global__ void unpackHistoryKernel(const val_t* exchange, val_t* history, size_t n,
                                    size_t timeStart, size_t batch) {
    size_t item = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item < n * batch) {
        size_t offset = item / n, triangle = item % n;
        history[(timeStart + offset) * n + triangle] = exchange[triangle * batch + offset];
    }
}

// Tile the local history so neighboring lag threads read consecutive values.
__global__ void transposeHistoryKernel(const val_t* history, val_t* localHistory,
                                       size_t n, size_t timesteps, size_t rowStart,
                                       size_t localRows) {
    __shared__ val_t tile[32][33];
    size_t row = size_t(blockIdx.x) * 32 + threadIdx.x;
    size_t t = size_t(blockIdx.y) * 32 + threadIdx.y;
    for (int k = 0; k < 32; k += 8)
        if (row < localRows && t + k < timesteps)
            tile[threadIdx.y + k][threadIdx.x] = history[(t + k) * n + rowStart + row];
    __syncthreads();
    row = size_t(blockIdx.x) * 32 + threadIdx.y;
    t = size_t(blockIdx.y) * 32 + threadIdx.x;
    for (int k = 0; k < 32; k += 8)
        if (row + k < localRows && t < timesteps)
            localHistory[(row + k) * timesteps + t] = tile[threadIdx.x][threadIdx.y + k];
}

// Parallelize correlation over receivers and lags while keeping each temporal
// dot product serial. The argmax uses the earliest lag for equal correlations.
__global__ void correlationKernel(const val_t* history, const val_t* localHistory, val_t* distances, size_t n,
                                  size_t timesteps, size_t localRows, size_t source) {
    size_t row = blockIdx.x;
    if (row >= localRows) return;
    val_t best = ZERO;
    size_t bestLag = 0;
    for (size_t lag = threadIdx.x; lag < timesteps; lag += blockDim.x) {
        val_t sum = ZERO;
        for (size_t t = lag; t < timesteps; ++t)
            sum += history[(t - lag) * n + source] * localHistory[row * timesteps + t];
        if (sum > best || (sum == best && lag < bestLag)) { best = sum; bestLag = lag; }
    }
    __shared__ val_t maxima[128];
    __shared__ size_t lags[128];
    maxima[threadIdx.x] = best;
    lags[threadIdx.x] = bestLag;
    __syncthreads();
    for (int stride = 64; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            val_t other = maxima[threadIdx.x + stride];
            size_t otherLag = lags[threadIdx.x + stride];
            if (other > maxima[threadIdx.x] || (other == maxima[threadIdx.x] && otherLag < lags[threadIdx.x])) {
                maxima[threadIdx.x] = other;
                lags[threadIdx.x] = otherLag;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) distances[row] = WAVE_SPEED * static_cast<val_t>(lags[0]);
}

struct GpuState {
    DeviceBuffer<Triangle> triangles;
    DeviceBuffer<Vec3> centers;
    DeviceBuffer<FlatNode> nodes;
    DeviceBuffer<uint32_t> indices;
    DeviceBuffer<val_t> areas, kij, history, distances;
    DeviceBuffer<int> tau;
};

struct SimulationState {
    size_t numTriangles = 0, numTimesteps = 0, sourceIndex = 0;
    size_t rowStart = 0, localRows = 0;
    int ranks = 1, minDelay = 1;
    bool cudaAware = false;
    val_t reflectivity = 0.8f;
    unsigned long long nonZeroKij = 0;
    std::vector<int> counts, displacements;
    std::vector<Triangle> triangles;
    std::vector<val_t> areas, radB, distances;
    std::unique_ptr<GpuState> gpu;
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, const ParallelContext& parallel) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;
    state.ranks = parallel.size;
    state.cudaAware = parallel.cudaAware;
    state.counts.resize(parallel.size);
    state.displacements.resize(parallel.size);
    for (int rank = 0; rank < parallel.size; ++rank) {
        size_t first = state.numTriangles * rank / parallel.size;
        size_t last = state.numTriangles * (rank + 1) / parallel.size;
        state.counts[rank] = static_cast<int>(last - first);
        state.displacements[rank] = static_cast<int>(first);
    }
    state.rowStart = state.displacements[parallel.rank];
    state.localRows = state.counts[parallel.rank];
    if (processRank == 0) printf("Generated icosphere mesh with %zu triangles\nBuilding octree...\n", state.numTriangles);
    Octree tree;
    #pragma omp parallel
    {
        #pragma omp single
        tree.build(state.triangles);
    }
    std::vector<FlatNode> nodes;
    std::vector<uint32_t> indices;
    flattenOctree(tree, nodes, indices);
    std::vector<Vec3> centers(state.numTriangles);
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
        centers[i] = state.triangles[i].center();
    }
    if (processRank == 0) state.distances.resize(state.numTriangles);
    state.gpu = std::make_unique<GpuState>();
    auto& gpu = *state.gpu;
    size_t n = state.numTriangles;
    gpu.triangles.allocate(n);
    gpu.centers.allocate(n);
    gpu.nodes.allocate(nodes.size());
    gpu.indices.allocate(indices.size());
    gpu.areas.allocate(n);
    gpu.kij.allocate(state.localRows * n);
    gpu.tau.allocate(state.localRows * n);
    gpu.history.allocate(timesteps * n);
    gpu.distances.allocate(state.localRows);
    CUDA_CALL(cudaMemcpy(gpu.triangles.data, state.triangles.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(gpu.centers.data, centers.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(gpu.nodes.data, nodes.data(), nodes.size() * sizeof(FlatNode), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(gpu.indices.data, indices.data(), indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(gpu.areas.data, state.areas.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemset(gpu.history.data, 0, timesteps * n * sizeof(val_t)));
}

void computeFormFactors(SimulationState& state) {
    if (processRank == 0) printf("Computing form factors (Kij) and time delays (Tau)...\n");
    size_t n = state.numTriangles;
    std::vector<RandomState> checkpoints;
    // Count consumed samples in parallel, then save the exact sequential stream
    // position once per row. Discarding avoids all tempering and float conversions.
    if (processRank == 0) {
        std::vector<size_t> draws(n, 0);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                if (i != j && state.triangles[i].normal().dot(state.triangles[j].normal()) <= 0.99f)
                    draws[i] += 4 * NUM_RAYS;
        checkpoints.resize(n);
        RandomGenerator rng(42);
        for (size_t i = 0; i < n; ++i) {
            checkpoints[i] = rng.checkpoint();
            if (i + 1 < n) rng.discard(draws[i]);
        }
    }
    std::vector<int> counts(state.ranks), displacements(state.ranks);
    for (int rank = 0; rank < state.ranks; ++rank) {
        counts[rank] = state.counts[rank] * 625;
        displacements[rank] = state.displacements[rank] * 625;
    }
    std::vector<RandomState> localCheckpoints(state.localRows);
    MPI_CALL(MPI_Scatterv(checkpoints.data(), counts.data(), displacements.data(), MPI_UINT32_T,
                         localCheckpoints.data(), counts[processRank], MPI_UINT32_T, 0, MPI_COMM_WORLD));
    // Bound staging memory independently of mesh size. Double buffering overlaps
    // OpenMP sampling and PCIe transfers with CUDA ray traversal.
    size_t samplesPerRow = n * NUM_RAYS * 4;
    size_t batchRows = std::max(size_t(1), std::min(state.localRows, (16 * 1024 * 1024) / (samplesPerRow * sizeof(val_t))));
    size_t secondSize = state.localRows > batchRows ? batchRows * samplesPerRow : 0;
    PinnedBuffer<val_t> host0(batchRows * samplesPerRow), host1(secondSize);
    DeviceBuffer<val_t> device0(batchRows * samplesPerRow), device1(secondSize);
    val_t* host[2] = {host0.data, host1.data};
    val_t* device[2] = {device0.data, device1.data};
    cudaStream_t streams[2];
    for (auto& stream : streams) CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    size_t batchIndex = 0;
    for (size_t start = 0; start < state.localRows; start += batchRows, ++batchIndex) {
        int slot = static_cast<int>(batchIndex % 2);
        CUDA_CALL(cudaStreamSynchronize(streams[slot]));
        size_t rows = std::min(batchRows, state.localRows - start);
        #pragma omp parallel for schedule(static)
        for (size_t row = 0; row < rows; ++row) {
            size_t i = state.rowStart + start + row;
            RandomGenerator rng(localCheckpoints[start + row]);
            val_t* output = host[slot] + row * samplesPerRow;
            for (size_t j = 0; j < n; ++j) {
                if (i == j || state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) continue;
                rng.fill(output + j * NUM_RAYS * 4, NUM_RAYS * 4);
            }
        }
        CUDA_CALL(cudaMemcpyAsync(device[slot], host[slot], rows * samplesPerRow * sizeof(val_t), cudaMemcpyHostToDevice, streams[slot]));
        size_t threads = rows * n * NUM_RAYS;
        formFactorKernel<<<static_cast<unsigned>((threads + 255) / 256), 256, 0, streams[slot]>>>(
            state.gpu->triangles.data, state.gpu->centers.data, state.gpu->nodes.data,
            state.gpu->indices.data, device[slot], state.gpu->kij.data, state.gpu->tau.data,
            n, state.localRows, state.rowStart, start, rows);
        CUDA_CALL(cudaGetLastError());
    }
    for (auto stream : streams) {
        CUDA_CALL(cudaStreamSynchronize(stream));
        CUDA_CALL(cudaStreamDestroy(stream));
    }
    DeviceBuffer<unsigned long long> nonzero(1);
    DeviceBuffer<int> minDelay(1);
    int delay = INT_MAX;
    unsigned long long count = 0;
    CUDA_CALL(cudaMemset(nonzero.data, 0, sizeof(count)));
    CUDA_CALL(cudaMemcpy(minDelay.data, &delay, sizeof(delay), cudaMemcpyHostToDevice));
    size_t entries = n * state.localRows;
    unsigned blocks = static_cast<unsigned>(std::max(size_t(1), std::min(size_t(1024), (entries + 255) / 256)));
    matrixStatsKernel<<<blocks, 256>>>(state.gpu->kij.data, state.gpu->tau.data,
        state.gpu->areas.data, state.localRows, entries, nonzero.data, minDelay.data);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaMemcpy(&count, nonzero.data, sizeof(count), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&delay, minDelay.data, sizeof(delay), cudaMemcpyDeviceToHost));
    MPI_CALL(MPI_Allreduce(&count, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));
    MPI_CALL(MPI_Allreduce(&delay, &state.minDelay, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if (state.minDelay <= 0) throw std::runtime_error("Zero-delay interactions cannot be advanced in parallel");
    if (state.minDelay == INT_MAX) state.minDelay = static_cast<int>(std::max(size_t(1), state.numTimesteps));
}

void runSimulation(SimulationState& state) {
    if (processRank == 0) printf("Running wave propagation simulation...\n");
    size_t n = state.numTriangles;
    size_t maxBatch = std::min(size_t(state.minDelay), std::max(size_t(1), state.numTimesteps));
    // CUDA-aware MPI exchanges GPU buffers directly. Otherwise pinned staging
    // supports ordinary MPI. Matrices and full histories remain on GPUs.
    DeviceBuffer<val_t> exchange(state.ranks > 1 ? n * maxBatch : 0);
    PinnedBuffer<val_t> send(state.ranks > 1 && !state.cudaAware ? state.localRows * maxBatch : 0);
    PinnedBuffer<val_t> receive(state.ranks > 1 && !state.cudaAware ? n * maxBatch : 0);
    std::vector<int> counts(state.ranks), displacements(state.ranks);
    for (size_t t = 0; t < state.numTimesteps; t += maxBatch) {
        size_t batch = std::min(maxBatch, state.numTimesteps - t);
        size_t items = batch * state.localRows;
        if (items) {
            propagationKernel<<<static_cast<unsigned>((items + 127) / 128), 128>>>(
                state.gpu->kij.data, state.gpu->tau.data, state.gpu->history.data, state.ranks > 1 ? exchange.data + state.rowStart * batch : nullptr,
                n, state.localRows, state.rowStart, t, batch, state.numTimesteps,
                state.sourceIndex, state.reflectivity);
            CUDA_CALL(cudaGetLastError());
        }
        if (state.ranks > 1) {
            for (int rank = 0; rank < state.ranks; ++rank) {
                counts[rank] = static_cast<int>(batch) * state.counts[rank];
                displacements[rank] = static_cast<int>(batch) * state.displacements[rank];
            }
            if (state.cudaAware) {
                CUDA_CALL(cudaDeviceSynchronize());
                MPI_CALL(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, exchange.data,
                                       counts.data(), displacements.data(), MPI_FLOAT, MPI_COMM_WORLD));
            } else {
                CUDA_CALL(cudaMemcpy(send.data, exchange.data + state.rowStart * batch,
                                     items * sizeof(val_t), cudaMemcpyDeviceToHost));
                MPI_CALL(MPI_Allgatherv(send.data, counts[processRank], MPI_FLOAT, receive.data,
                                       counts.data(), displacements.data(), MPI_FLOAT, MPI_COMM_WORLD));
                CUDA_CALL(cudaMemcpy(exchange.data, receive.data, n * batch * sizeof(val_t), cudaMemcpyHostToDevice));
            }
            unpackHistoryKernel<<<static_cast<unsigned>((n * batch + 255) / 256), 256>>>(
                exchange.data, state.gpu->history.data, n, t, batch);
            CUDA_CALL(cudaGetLastError());
        }
    }
    CUDA_CALL(cudaDeviceSynchronize());
}

void computeDistances(SimulationState& state, bool validate) {
    if (processRank == 0) printf("Computing distances via cross-correlation...\n");
    DeviceBuffer<val_t> localHistory(state.localRows * state.numTimesteps);
    if (state.localRows && state.numTimesteps) {
        dim3 grid(static_cast<unsigned>((state.localRows + 31) / 32),
                  static_cast<unsigned>((state.numTimesteps + 31) / 32));
        transposeHistoryKernel<<<grid, dim3(32, 8)>>>(state.gpu->history.data, localHistory.data,
            state.numTriangles, state.numTimesteps, state.rowStart, state.localRows);
        CUDA_CALL(cudaGetLastError());
    }
    if (state.localRows) {
        correlationKernel<<<static_cast<unsigned>(state.localRows), 128>>>(
            state.gpu->history.data, localHistory.data, state.gpu->distances.data, state.numTriangles,
            state.numTimesteps, state.localRows, state.sourceIndex);
        CUDA_CALL(cudaGetLastError());
    }
    std::vector<val_t> local(state.localRows);
    if (!local.empty())
        CUDA_CALL(cudaMemcpy(local.data(), state.gpu->distances.data, local.size() * sizeof(val_t), cudaMemcpyDeviceToHost));
    MPI_CALL(MPI_Gatherv(local.data(), state.counts[processRank], MPI_FLOAT, state.distances.data(),
                         state.counts.data(), state.displacements.data(), MPI_FLOAT, 0, MPI_COMM_WORLD));
    if (validate && processRank == 0) {
        state.radB.resize(state.numTimesteps * state.numTriangles);
        CUDA_CALL(cudaMemcpy(state.radB.data(), state.gpu->history.data, state.radB.size() * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
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
    unsigned long long nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           nonZeroKij, state.numTriangles * state.numTriangles,
           100.0 * static_cast<double>(nonZeroKij) / static_cast<double>(state.numTriangles * state.numTriangles));

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

long maximumDuration(long duration) {
    long maximum = 0;
    MPI_CALL(MPI_Allreduce(&duration, &maximum, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD));
    return maximum;
}

int benchmarkMain(int argc, char** argv, const ParallelContext& parallel) {
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
            if (processRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (processRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (timesteps < 0) throw std::runtime_error("Timesteps must be non-negative");
    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (processRank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        printf("MPI ranks: %d; OpenMP threads per rank: %d; CUDA device on rank 0: %d\n",
               parallel.size, omp_get_max_threads(), parallel.device);
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity, parallel);

    if (processRank == 0) printf("\n");

    // Precomputation
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    auto startPre = std::chrono::high_resolution_clock::now();

    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();
    preDuration = maximumDuration(static_cast<long>(preDuration));

    if (processRank == 0) printf("Precomputation time: %ld ms\n\n", preDuration);

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();
    simDuration = maximumDuration(static_cast<long>(simDuration));

    if (processRank == 0) printf("Simulation time: %ld ms\n\n", simDuration);

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state, validate);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();
    distDuration = maximumDuration(static_cast<long>(distDuration));

    if (processRank != 0) return 0;
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
    size_t maxRows = *std::max_element(state.counts.begin(), state.counts.end());
    size_t memKij = maxRows * n * sizeof(val_t);
    size_t memTau = maxRows * n * sizeof(int);
    size_t memRad = t * (n + maxRows) * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    printf("  Maximum matrix/history memory per rank: %.2f MB\n", totalMem / (1024.0 * 1024.0));

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

int main(int argc, char** argv) {
    ParallelContext parallel(argc, argv);
    try {
        return benchmarkMain(argc, argv, parallel);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", processRank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
