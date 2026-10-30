// QT Clustering Benchmark - CUDA Version
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

struct Point { double x, y; };

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        if (group_cnt > (N - count)) group_cnt = N - count;
        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * R;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            points[count] = {x, y};
            count++;
            group_cnt--;
        }
    }
}

__device__ inline double d_distance(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return sqrt(dx*dx + dy*dy);
}

__global__ void findClosestKernel(const double* px, const double* py,
    const unsigned char* clustered, const unsigned char* in_cluster,
    const int* members, int member_count,
    double threshold, int N, double* maxdists) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    if (clustered[c] || in_cluster[c]) { maxdists[c] = 1e300; return; }
    double md = 0.0;
    for (int i = 0; i < member_count; ++i) {
        int m = members[i];
        double d = d_distance(px[c], py[c], px[m], py[m]);
        if (d > md) md = d;
    }
    maxdists[c] = md;
}

int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    // device buffers
    double *d_px, *d_py, *d_max;
    unsigned char *d_cl, *d_in;
    int *d_mem;
    cudaMalloc(&d_px, point_count*sizeof(double));
    cudaMalloc(&d_py, point_count*sizeof(double));
    cudaMalloc(&d_cl, point_count*sizeof(unsigned char));
    cudaMalloc(&d_in, point_count*sizeof(unsigned char));
    cudaMalloc(&d_max, point_count*sizeof(double));
    cudaMalloc(&d_mem, cluster_members.size()*sizeof(int));

    std::vector<double> px(point_count), py(point_count);
    std::vector<unsigned char> cl(point_count), in(point_count);
    for (int i=0;i<point_count;i++){ px[i]=points[i].x; py[i]=points[i].y; cl[i]=clustered[i]?1:0; in[i]=in_cluster[i]?1:0; }
    cudaMemcpy(d_px, px.data(), point_count*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py.data(), point_count*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cl, cl.data(), point_count*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, in.data(), point_count*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mem, cluster_members.data(), cluster_members.size()*sizeof(int), cudaMemcpyHostToDevice);

    int threads=256;
    int blocks=(point_count+threads-1)/threads;
    findClosestKernel<<<blocks,threads>>>(d_px,d_py,d_cl,d_in,d_mem,(int)cluster_members.size(),threshold,point_count,d_max);
    cudaDeviceSynchronize();

    std::vector<double> maxd(point_count);
    cudaMemcpy(maxd.data(), d_max, point_count*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_px); cudaFree(d_py); cudaFree(d_cl); cudaFree(d_in); cudaFree(d_max); cudaFree(d_mem);

    int closest=-1;
    double min_diameter=std::numeric_limits<double>::max();
    for (int c=0;c<point_count;c++) {
        if (maxd[c] < threshold && maxd[c] < min_diameter) {
            min_diameter=maxd[c];
            closest=c;
        }
    }
    return closest;
}

int generateCandidateCluster(const int seed_point,
    const std::vector<bool>& clustered,
    const std::vector<Point>& points,
    const double threshold,
    const int point_count,
    std::vector<int>* cluster_members) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    in_cluster[seed_point]=true;
    members.push_back(seed_point);
    while ((int)members.size() < point_count) {
        const int closest = findClosestPoint(members, clustered, in_cluster, points, threshold, point_count);
        if (closest < 0) break;
        in_cluster[closest]=true;
        members.push_back(closest);
    }
    if (cluster_members) *cluster_members = members;
    return (int)members.size();
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int N = (int)points.size();
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    for (int i=0;i<N;i++) unclustered_indices.push_back(i);
    while (!unclustered_indices.empty()) {
        int max_cardinality=-1;
        int best_seed=-1;
        std::vector<int> best_cluster_members;
        for (size_t i=0;i<unclustered_indices.size();++i) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;
            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(seed, clustered, points, threshold, N, &candidate_members);
            if (cardinality > max_cardinality) {
                max_cardinality=cardinality;
                best_seed=seed;
                best_cluster_members=candidate_members;
            }
        }
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point=best_seed;
            cluster.members=best_cluster_members;
            clusters.push_back(cluster);
            for (size_t i=0;i<best_cluster_members.size();++i) clustered[best_cluster_members[i]]=true;
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                    [&clustered](int idx){ return clustered[idx]; }),
                unclustered_indices.end());
        } else break;
    }
    return clusters;
}

// validation, printUsage, main unchanged except include
bool validateClusters(const std::vector<Cluster>& clusters, const std::vector<Point>& points, const double threshold) {
    bool valid=true;
    printf("Validating clusters:\n");
    for (size_t c=0;c<clusters.size();++c) {
        const auto& cluster=clusters[c];
        double max_diameter=0.0;
        for (size_t i=0;i<cluster.members.size();++i)
            for (size_t j=i+1;j<cluster.members.size();++j) {
                double dx=points[cluster.members[i]].x-points[cluster.members[j]].x;
                double dy=points[cluster.members[i]].y-points[cluster.members[j]].y;
                max_diameter=std::max(max_diameter, std::sqrt(dx*dx+dy*dy));
            }
        if (c<10) printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c, cluster.members.size(), cluster.seed_point, max_diameter);
        if (max_diameter > threshold*1.001) { printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", c, max_diameter, threshold); valid=false; }
    }
    std::vector<int> membership(points.size(), -1);
    for (size_t c=0;c<clusters.size();++c)
        for (size_t i=0;i<clusters[c].members.size();++i) {
            int member=clusters[c].members[i];
            if (membership[member]>=0) { printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n", member, membership[member], c); valid=false; }
            membership[member]=(int)c;
        }
    int clustered_count=0;
    for (size_t i=0;i<membership.size();++i) if (membership[i]>=0) clustered_count++;
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(), clustered_count, points.size()-clustered_count);
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int num_points=1000;
    double threshold=2.0;
    bool validate=false;
    bool printResults=false;
    for (int i=1;i<argc;++i) {
        if (strcmp(argv[i],"-n")==0 && i+1<argc) num_points=atoi(argv[++i]);
        else if (strcmp(argv[i],"-t")==0 && i+1<argc) threshold=atof(argv[++i]);
        else if (strcmp(argv[i],"-v")==0) validate=true;
        else if (strcmp(argv[i],"-r")==0) printResults=true;
        else if (strcmp(argv[i],"-h")==0) { printUsage(argv[0]); return 0; }
        else { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    if (num_points<=0 || threshold<=0.0) { printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points, threshold); return 1; }
    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    auto cluster_start=std::chrono::high_resolution_clock::now();
    const std::vector<Cluster> clusters=qtClustering(points, threshold);
    auto cluster_end=std::chrono::high_resolution_clock::now();
    auto cluster_time=std::chrono::duration_cast<std::chrono::milliseconds>(cluster_end-cluster_start);
    printf("Clustering time: %ld ms\n", cluster_time.count());
    printf("Clusters found: %zu\n", clusters.size());
    int total_clustered=0, max_cluster_size=0;
    for (size_t i=0;i<clusters.size();++i){ int size=(int)clusters[i].members.size(); total_clustered+=size; max_cluster_size=std::max(max_cluster_size,size);}
    const double avg_cluster_size=clusters.empty()?0.0:(double)total_clustered/clusters.size();
    printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points, 100.0*total_clustered/num_points);
    printf("Average cluster size: %.2f\n", avg_cluster_size);
    printf("Maximum cluster size: %d\n", max_cluster_size);
    const double time_sec=cluster_time.count()/1000.0;
    const double clusters_per_sec=clusters.size()/time_sec;
    const double points_per_sec=num_points/time_sec;
    printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters_per_sec, points_per_sec);
    if (printResults) {
        std::vector<double> membershipData;
        membershipData.reserve(num_points);
        std::vector<int> membership(num_points,-1);
        for (size_t c=0;c<clusters.size();++c)
            for (size_t i=0;i<clusters[c].members.size();++i)
                membership[clusters[c].members[i]]=(int)c;
        for (int m:membership) membershipData.push_back((double)m);
        print_results(membershipData, "ClusterMembership");
    }
    if (validate) {
        const bool valid=validateClusters(clusters, points, threshold);
        if (valid) { printf("Validation: PASSED\n"); return 0; }
        else { printf("Validation: FAILED\n"); return 1; }
    }
    return 0;
}