#include<bits/stdc++.h>
using namespace std;


/**
1000万条有重复的字符串，找出重复数前10的字符串
https://www.cnblogs.com/marginalman/p/4808888.html
*/

/**
6 2
sss
sss
ss
ss
s
sss
*/
struct node
{
    string s;
    int num;
    node(string _s, int _num) : s(_s), num(_num) {}
    bool operator < (const node& a) const
    {
        return num > a.num;
    }
};



int main()
{
    int N, k;
    priority_queue<node> pq;
    unordered_map<string, int> umap;
    string s;
    cin >> N >> k;
    for (int i = 0; i < N; ++i)
    {
        cin >> s;
        umap[s]++;
    }

    for (auto it = umap.begin(); it != umap.end(); ++it)
    {
        if (pq.size() < k)
        {
            node t(it -> first, it -> second);
            pq.push(t);
        }
        else
        {
            if (it -> second > pq.top().num)
            {
                node t(it -> first, it -> second);
                pq.pop();
                pq.push(t);
            }
        }
    }
    while (!pq.empty())
    {
        cout << pq.top().s << " ";
        pq.pop();
    }
    system("pause");
    return 0;
}

/**
 * LC407. 接雨水 II
 * 优先队列
*/
class LC407 {
public:
    int dirs[4][2] = {
        {0, 1}, {0, -1}, {-1, 0}, {1, 0}
    };
    struct Cell
    {
        int x;
        int y;
        int h;
        Cell(int _x, int _y, int _h) : x(_x), y(_y), h(_h) {}

        bool operator < (const Cell& a) const
        {
            return h > a.h;
        }
    };
    int trapRainWater(vector<vector<int>>& heightMap) {
        
        if (heightMap.size() == 0 || heightMap[0].size() == 0) return 0;
        int m = heightMap.size(), n = heightMap[0].size();
        vector<vector<int>> vis(m, vector<int>(n, 0));

        priority_queue<Cell> pque;
        // 边界初始化
        for (int i = 0; i < m; ++i)
        {
            pque.emplace(i, 0, heightMap[i][0]);
            pque.emplace(i, n - 1, heightMap[i][n - 1]);
            vis[i][0] = 1;
            vis[i][n - 1] = 1;
        }
        for (int j = 1; j < n - 1; ++j)
        {
            pque.emplace(0, j, heightMap[0][j]);
            pque.emplace(m - 1, j, heightMap[m - 1][j]);
            vis[0][j] = 1;
            vis[m - 1][j] = 1;
        }

        int ans = 0;
        while (!pque.empty())
        {
            Cell ce = pque.top();
            pque.pop();
            for (int i = 0; i < 4; ++i)
            {
                int nx = ce.x + dirs[i][0], ny = ce.y + dirs[i][1];
                if (nx >= 0 && ny >= 0 && nx < m && ny < n && vis[nx][ny] == 0)
                {
                    vis[nx][ny] = 1;
                    int th = max(heightMap[nx][ny], ce.h);
                    pque.emplace(nx, ny, th);
                    if (heightMap[nx][ny] < ce.h)
                    {
                        ans += ce.h - heightMap[nx][ny];
                    }
                }
            }
        }
        return ans;

    }
};