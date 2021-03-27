#include<bits/stdc++.h>
using namespace std;

typedef pair<int, int> pr; // first代表最小距离，second表示城市位置
const int inf = 0x3f3f3f3f;

/**
 * LC787. K 站中转内最便宜的航班 dijkstra 算法 单源最短路径
 * LC207. 课程表 拓扑排序
 * 1162. 地图分析 多源 BFS
 * 797. 所有可能的路径 DFS
 * 130. 被围绕的区域 并查集
 * 990. 等式方程的可满足性 并查集
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
     *  * 拓扑排序
        * 给定一个包含 nn 个节点的有向图 GG，我们给出它的节点编号的一种排列，如果满足：
        * 对于图 GG 中的任意一条有向边 (u, v)，u 在排列中都出现在 v 的前面。
        * 那么称该排列是图 GG 的 拓扑排序
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



class LC1162 {
public:
    /**
     * 多源 BFS
     * https://leetcode-cn.com/problems/as-far-from-land-as-possible/solution/zhen-liang-yan-sou-huan-neng-duo-yuan-kan-wan-miao/
    */
    int N;
    int dx[4] = {1, -1, 0, 0};
    int dy[4] = {0, 0, 1, -1};
    vector<vector<int>> dis;
    int maxDistance(vector<vector<int>>& grid) {
        N = grid.size();
        if (N == 0) return -1;
        dis.assign(N, vector<int>(N, 0));
        queue<pair<int, int>> que;
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                if (grid[i][j] == 1) que.emplace(i, j);
            }
        }
        int x = -1, y = -1;
        bool hasLand = false;
        while (!que.empty())
        {
            pair<int, int> tq = que.front(); que.pop();
            x = tq.first;
            y = tq.second;
            for (int i = 0; i < 4; ++i)
            {
                int nx = x + dx[i], ny = y + dy[i];
                if (nx >= 0 && nx < N && ny >= 0 && ny < N && grid[nx][ny] == 0)
                {
                    hasLand = true;
                    grid[nx][ny] = grid[x][y] + 1;
                    que.emplace(nx, ny);
                }
            }
        }

        if (x == -1 && y == -1) return -1;
        if (!hasLand) return -1;

        return grid[x][y] - 1;



    }
};

class LC797 {
public:
    vector<vector<int>> paths;
    vector<int> path;

    void dfs(int q, vector<vector<int>>& graph)
    {
        path.push_back(q);

        if (q == graph.size() - 1)
        {
            paths.push_back(path);
            return;
        }

        for (int i = 0; i < graph[q].size(); ++i)
        {
            dfs(graph[q][i], graph);
            path.pop_back();
        }
    }
    vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
        dfs(0, graph);
        return paths;
    }
};

/**
https://leetcode-cn.com/problems/surrounded-regions/solution/bfsdi-gui-dfsfei-di-gui-dfsbing-cha-ji-by-ac_pipe/

并查集的思想就是，同一个连通区域内的所有点的根节点是同一个。
将每个点映射成一个数字。先假设每个点的根节点就是他们自己，
然后我们以此输入连通的点对，然后将其中一个点的根节点赋成另一个节点的根节点，
这样这两个点所在连通区域又相互连通了。
并查集的主要操作有：
find(int m)：这是并查集的基本操作，查找 mm 的根节点。
isConnected(int m,int n)：判断 m，nm，n 两个点是否在一个连通区域。
union(int m,int n):合并 m，nm，n 两个点所在的连通区域。
*/

class UnionFind
{
private:
    vector<int> parents;
public:
    UnionFind(int totalNodes)
    {
        for (int i = 0; i < totalNodes; ++i)
        {
            parents.push_back(i);
        }
    }

    int find(int node)
    {
        while (parents[node] != node) 
        {
            // 当前节点的父节点 指向父节点的父节点.
            // 保证一个连通区域最终的parents只有一个.
            parents[node] = parents[parents[node]];
            node = parents[node];
        }
        return node;
    }
    bool isconnect(int m, int n)
    {
        return find(m) == find(n);
    }
    void uunion(int m, int n)
    {
        int root1 = find(m);
        int root2 = find(n);
        if (root1 != root2)
        {
            parents[root2] = root1;
        }
    }
};

class LC130 {
public:
    void solve(vector<vector<char>>& board) {
        if (board.size() == 0 || board[0].size() == 0) return;
        int m = board.size(), n = board[0].size();
        // 用一个虚拟节点 m * n, 边界上的O 的父节点都是这个虚拟节点
        UnionFind uf = UnionFind(m * n + 1);
        int dummyNode = m * n;
        int dx[4] = {1, -1, 0, 0};
        int dy[4] = {0, 0, 1, -1};
        
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (board[i][j] == 'O')
                {
                    if (i == 0 || i == m - 1 || j == 0 || j == n - 1)
                    {
                        uf.uunion(i * n + j, dummyNode);
                    }
                    else
                    {
                        for (int k = 0; k < 4; ++k)
                        {
                            int x = i + dx[k], y = j + dy[k];
                            if (x >= 0 && x < m && y >= 0 && y < n && board[x][y] == 'O')
                            {
                                uf.uunion(i * n +j, x * n + y);
                            }
                        }
                    }
                }
            }
        }

        for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
        {
            if (uf.isconnect(i * n + j, dummyNode))  // 和dummyNode 在一个连通区域的,那么就是O；
            {
                board[i][j] = 'O';
            }
            else
            {
                board[i][j] = 'X';
            }
        }
        
    }
};


/**
 * 并查集
 * 由a == b, b == c 可以知道 a == c。所以可将a,b,c视为一个类型。 
 */
class LC990 {
public:
    struct UnionFind
    {
        vector<int> parent;
        // 小写字母a-z对应parent是size=26
        // 初始化各自的parent为自己，即满足a==a.
        UnionFind()
        {
            parent.resize(26);
            for (int i = 0; i < 26; ++i) parent[i] = i;
        }
        // 递归找到当前类型的代表元
        int find(int idx)
        {
            if (parent[idx] == idx) return idx;
            else
            {
                parent[idx] = find(parent[idx]);
                return parent[idx];
            }
        }
        // 将idx1 和 idx2归为一类，这里默认将idx2这类归到idx1中去 
        void unite(int idx1, int idx2)
        {
            parent[find(idx1)] = find(idx2);
        }

    };
    bool equationsPossible(vector<string>& equations) {
        UnionFind uf;
        // 根据相等元素构造对应的并查集
        for (string &s:equations)
        {
            if (s[1] == '=')
            {
                uf.unite(s[0]-'a', s[3]-'a');
            }
        }
        // 并查集得到后，所有相等的元素归成了一个类。
        // 所以只需要判断不等元素是否在同一个类中
        // 如果有一个不等的两个元素在一个类中即返回false,否则返回true
        for (string &s:equations)
        {
            if (s[1] == '!')
            {
                int idx1 = uf.find(s[0]-'a'), idx2 = uf.find(s[3]-'a');
                if (idx1 == idx2) return false;
            }
        }
        return true;

    }
};