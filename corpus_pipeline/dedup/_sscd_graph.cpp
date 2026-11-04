// SSCD graph arithmetic only. No asset/record deletion or transitive dedup.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

struct Edge { uint32_t a, b; float score; };
static_assert(sizeof(Edge) == 12, "Unexpected edge packing");
static uint32_t find_root(uint32_t x, uint32_t* parent) {
    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
}

extern "C" {
int graph_initialize(uint64_t n, uint32_t* parent, uint32_t* sizes, uint64_t* degree) {
    if (n >= UINT32_MAX) return 1;
    for (uint64_t i=0;i<n;++i) { parent[i]=uint32_t(i); sizes[i]=1; degree[i]=0; }
    return 0;
}
int graph_count(uint64_t n, uint64_t m, const Edge* edges, uint32_t* parent,
                uint32_t* sizes, uint64_t* degree, float threshold) {
    for (uint64_t i=0;i<m;++i) {
        const Edge& e=edges[i];
        if (e.a >= n || e.b >= n || e.a >= e.b || !std::isfinite(e.score) || e.score < threshold) return 2;
        ++degree[e.a]; ++degree[e.b];
        uint32_t a=find_root(e.a,parent), b=find_root(e.b,parent);
        if (a != b) { if (sizes[a]<sizes[b]) std::swap(a,b); parent[b]=a; sizes[a]+=sizes[b]; }
    }
    return 0;
}
int graph_components(uint64_t n, uint32_t* parent, const uint32_t* sizes,
                     const uint64_t* degree, uint32_t* components,
                     uint32_t* component_sizes, uint64_t* offsets,
                     uint64_t* summary, uint64_t* histogram) {
    std::fill(components,components+n,UINT32_MAX);
    std::fill(summary,summary+6,0); std::fill(histogram,histogram+32,0);
    for (uint64_t i=0;i<n;++i) { parent[i]=find_root(uint32_t(i),parent); components[parent[i]]=std::min(components[parent[i]],uint32_t(i)); }
    offsets[0]=0;
    for (uint64_t i=0;i<n;++i) {
        uint32_t root=parent[i]; components[i]=components[root]; component_sizes[i]=sizes[root];
        offsets[i+1]=offsets[i]+degree[i];
        if (degree[i]) ++summary[3]; else ++summary[4];
        summary[5]=std::max(summary[5],degree[i]);
        if (root == i) {
            ++summary[0]; if(sizes[i]>1) ++summary[1]; summary[2]=std::max(summary[2],uint64_t(sizes[i]));
            unsigned bin=0; uint32_t value=sizes[i]; while(value>1){value>>=1;++bin;} ++histogram[bin];
        }
    }
    return 0;
}
int graph_fill(uint64_t n, uint64_t m, const Edge* edges, const uint64_t* offsets,
               uint64_t* cursor, uint32_t* neighbors, float* scores) {
    for(uint64_t i=0;i<m;++i) {
        const Edge& e=edges[i];
        if(e.a>=n||e.b>=n||e.a>=e.b) return 3;
        uint64_t a=cursor[e.a]++, b=cursor[e.b]++;
        if(a>=offsets[e.a+1]||b>=offsets[e.b+1]) return 4;
        neighbors[a]=e.b; scores[a]=e.score; neighbors[b]=e.a; scores[b]=e.score;
    }
    return 0;
}
int graph_representatives(uint64_t n, const uint64_t* offsets, const uint32_t* neighbors,
                          const float* scores, const uint32_t* order, const uint8_t* eligible,
                          const uint8_t* protect, uint32_t* representative, float* direct_score,
                          uint32_t* group_size, uint64_t* summary, float threshold) {
    std::fill(representative,representative+n,UINT32_MAX);
    std::fill(direct_score,direct_score+n,std::numeric_limits<float>::quiet_NaN());
    std::fill(group_size,group_size+n,0); std::fill(summary,summary+5,0);
    // Protected vertices cannot be suppressed and do not suppress others.
    for(uint64_t i=0;i<n;++i) if(eligible[i]&&protect[i]){representative[i]=uint32_t(i);direct_score[i]=1.f;}
    for(uint64_t position=0;position<n;++position) {
        uint32_t v=order[position]; if(v>=n) return 5;
        if(!eligible[v]||protect[v]||representative[v]!=UINT32_MAX) continue;
        representative[v]=v; direct_score[v]=1.f;
        for(uint64_t k=offsets[v];k<offsets[v+1];++k) {
            uint32_t other=neighbors[k];
            if(other>=n||!std::isfinite(scores[k])||scores[k]<threshold) return 6;
            if(!eligible[other]||protect[other]) continue;
            if(representative[other]==UINT32_MAX){representative[other]=v;direct_score[other]=scores[k];}
            else if(representative[other]==v) direct_score[other]=std::max(direct_score[other],scores[k]);
        }
    }
    for(uint64_t i=0;i<n;++i) {
        if(!eligible[i]){++summary[3];continue;}
        uint32_t rep=representative[i];
        if(rep>=n||representative[rep]!=rep||!eligible[rep]) return 7;
        ++group_size[rep];++summary[0]; if(rep==i)++summary[1];else ++summary[2];
        if(protect[i])++summary[4];
    }
    return 0;
}
}
