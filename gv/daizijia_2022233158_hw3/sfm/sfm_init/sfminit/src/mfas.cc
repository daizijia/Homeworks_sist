#include <vector>
#include <set>
#include <map>
#include "mfas.h"

using std::pair;
using std::vector;
using std::map;
using std::set;

void reindex_problem(vector<Edge> &edges,
                     map<int, int> &reindexing_key) {
    // get the unique set of notes
    set<int> nodes;
    for (int i=0; i<edges.size(); ++i) {
        nodes.insert(edges[i].first);
        nodes.insert(edges[i].second);
    }

    // iterator through them and assign a new name to each vertex
    std::set<int>::const_iterator it;
    reindexing_key.clear();
    int i=0;
    for (it = nodes.begin(); it != nodes.end(); ++it) {
        reindexing_key[*it] = i;
        ++i;
    }

    // now renumber the edges
    for (int i=0; i<edges.size(); ++i) {
        edges[i].first  = reindexing_key[edges[i].first ];
        edges[i].second = reindexing_key[edges[i].second];
    }
}

void flip_neg_edges(vector<Edge> &edges,
                    vector<double> &weights) {

    // now renumber the edges
    for (int i=0; i<edges.size(); ++i) {
        if (weights[i] < 0.0) {
            /*
            TO Do: flip negative edges
            */
            std::swap(edges[i].first, edges[i].second);
            weights[i] *= -1;
        }
    }
}



void mfas_ratio(const std::vector<Edge> &edges,
                const std::vector<double> &weights,
                std::vector<int> &order) {
    /*
    TODO: implement Minimum Feedback Arc Set algorithm
    Input: egdes and weights
    Output: ordering
    */
    int n = -1;
    for (const auto& edge : edges) {
        n = std::max(n, std::max(edge.first, edge.second));
    }
    n++; 

    std::vector<double> win_deg(n, 0.0), wout_deg(n, 0.0);
    std::vector<bool> unchosen(n, true);
    std::vector<std::vector<std::pair<int, double>>> inbrs(n), onbrs(n);

    for (int ii = 0; ii < edges.size(); ++ii) {
        int i = edges[ii].first;
        int j = edges[ii].second;
        double w = weights[ii];

        win_deg[j] += w;
        wout_deg[i] += w;
        inbrs[j].emplace_back(i, w);
        onbrs[i].emplace_back(j, w);
    }

    while (order.size() < n) {
        int choice = -1;
        double max_score = 0.0;
        for (int i = 0; i < n; ++i) {
            if (unchosen[i]) {
                double score = win_deg[i] < 1e-8 ? std::numeric_limits<double>::infinity() :
                           (wout_deg[i] + 1) / (win_deg[i] + 1);
                if (score > max_score) {
                    max_score = score;
                    choice = i;
                }
            }
        }

        for (auto& in_neighbor : inbrs[choice]) {
            wout_deg[in_neighbor.first] -= in_neighbor.second;
        }
        for (auto& out_neighbor : onbrs[choice]) {
            win_deg[out_neighbor.first] -= out_neighbor.second;
        }

        order.push_back(choice);
        unchosen[choice] = false;
    }
}

void broken_weight(const std::vector<Edge>   &edges,
                   const std::vector<double> &weight,
                   const std::vector<int>    &order,
                         std::vector<double> &broken) {

    // clear the output vector
    int m = edges.size();
    broken.resize(m);
    broken.assign(broken.size(), 0.0);

    // find the number of nodes in this problem
    int n = -1;
    for (int i=0; i<m; ++i) {
        n = (edges[i].first  > n) ? edges[i].first  : n;
        n = (edges[i].second > n) ? edges[i].second : n;
    }
    n += 1; // 0 indexed

    // invert the permutation
    std::vector<int> inv_perm(n, 0.0);
    for (int i=0; i<n; ++i)
        inv_perm[order[i]] = i;

    /*
    TODO: find the broken edges and store it in vector "broken"
    */
    // find the broken edges
    for (int i = 0; i < m; ++i) {
        auto x0 = inv_perm[edges[i].first];
        auto x1 = inv_perm[edges[i].second];
    
        if ((x1 - x0) * weight[i] < 0) {
            broken[i] += std::abs(weight[i]);
        }
    }

}
