#include<bits/stdc++.h>
using namespace std;

typedef pair<int, int> pr; // first代表最小距离，second表示城市位置
const int inf = 0x3f3f3f3f;

/**
 * LC787. K 站中转内最便宜的航班
 * dijkstra 算法 单源最短路径
 * 
 * LC207. 课程表
 * 拓扑排序
 * 给定一个包含 nn 个节点的有向图 GG，我们给出它的节点编号的一种排列，如果满足：
 * 对于图 GG 中的任意一条有向边 (u, v)，u 在排列中都出现在 v 的前面。
 * 那么称该排列是图 GG 的 拓扑排序
*/

class LC787 {
private:
    int n;
    vector<vector<int>> dist;
    priority_queue<Node> pque;
    vector<vector<int>> graph;
public:
    struct Node
    {
        int min_cost; // min cost
        int loc;
        int level;
        Node(int mc, int lc, int ll) : min_cost(mc), loc(lc), level(ll){}
        bool operator < (const Node& a) const
        {
            return min_cost > a.min_cost;
        }
    };
    int Dijkstra(int src, int dst, int K)
    {
        dist.assign(K + 2, vector<int>(n, inf)); // dist[i][j] 经过 i 站到达 j 最便宜价格
        dist[0][src] = 0;
        pque.emplace(0, src, 0);
        while(!pque.empty())
        {
            Node p = pque.top(); pque.pop();
            int v = p.loc, cost = p.min_cost, level = p.level;
            if (v == dst)
            {
                return cost;
            }
            if (level >= K + 1 || dist[level][v] < cost)
            {
                continue;
            }
            for (int i = 0; i < n; ++i)
            {
                if (graph[v][i] < inf && dist[level + 1][i] > cost + graph[v][i])
                {
                    dist[level + 1][i] = cost + graph[v][i];
                    pque.emplace(dist[level + 1][i], i, level + 1);
                }
            }
        }
        return inf;
    }
    int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int K) {
        this -> n = n;
        graph.assign(n, vector<int>(n, inf));
        for (int i = 0; i < flights.size(); ++i)
        {
            int u = flights[i][0], v = flights[i][1], w = flights[i][2];
            graph[u][v] = w; 
        }
        int cost = Dijkstra(src, dst, K);
        if (cost != inf)
        {
            return cost;
        }
        else
        {
            return -1;
        }

    }
};


class LC207 {
private:
    vector<vector<int>> graph;
    vector<int> inDegrees;
public:
    /**
     * 拓扑排序判断图中是否有环
    */
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {

        inDegrees.assign(numCourses, 0);
        graph.resize(numCourses);
        // 建图
        for (int i = 0; i < prerequisites.size(); ++i)
        {
            int u = prerequisites[i][0], v = prerequisites[i][1];
            graph[u].push_back(v);
            inDegrees[v]++;
        }

        // 找环
        queue<int> que;
        for (int i = 0; i < numCourses; ++i)
        {
            if (inDegrees[i] == 0)
            {
                que.push(i);
            }
        }
        int cnt = 0;
        while (!que.empty())
        {
            int tmp = que.front();
            que.pop();
            cnt++;

            for (int i = 0; i < graph[tmp].size(); ++i)
            {
                int nt = graph[tmp][i];
                inDegrees[nt]--;
                if (inDegrees[nt] == 0)
                {
                    que.push(nt);
                }
            }
        }
        return cnt == numCourses;

    }
};