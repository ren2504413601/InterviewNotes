#include<bits/stdc++.h>
using namespace std;

/**
 * 215. 数组中的第K个最大元素
 * TopK
 * 407. 接雨水 II
 * 23. 合并K个升序链表
 * 295. 数据流的中位数
*/

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

class LC215_quickSort {
public:

    int partation(vector<int>& nums, int l, int r)
    {
        int x = nums[r];
        int i = l;
        for (int j = l; j < r; ++j)
        {
            if (x >= nums[j])
            {
                swap(nums[j], nums[i]);
                i++;
            }
        }
        swap(nums[i], nums[r]);
        return i;
    }
    int random_partation(vector<int>& nums, int l, int r)
    {
        int randIdx = rand() % (r - l + 1) + l;
        swap(nums[randIdx], nums[r]);
        return partation(nums, l, r);
    }
    int quickSort(vector<int>& nums, int l, int r, int idx)
    {
        // if (l > r) return;

        int p = partation(nums, l, r);
        if (p == idx) return nums[p];
        else if (p < idx)
        {
            return quickSort(nums, p+1, r, idx);
        }
        else
        {
            return quickSort(nums, l, p-1, idx);
        }
    }
    int findKthLargest(vector<int>& nums, int k) {
        
        srand(time(0));
        int n = nums.size();
        return quickSort(nums, 0, n-1, n-k);

        return nums[n-k];
    }
}; 

class LC215_heapsort {
public:

    void make_heap(vector<int>& nums, int heap_size)
    {
        for (int i = heap_size / 2; i >= 0; --i)
        {
            make_heap_fix(nums, i, heap_size);
        }
    }

    void make_heap_fix(vector<int>& nums, int cur_idx ,int heap_size)
    {
        int lchildIdx = cur_idx * 2 + 1, rchildIdx= cur_idx * 2 + 2;
        int largeIdx = cur_idx;
        if (lchildIdx < heap_size && nums[lchildIdx] > nums[largeIdx])
        {
            largeIdx = lchildIdx;
        }
        if (rchildIdx < heap_size && nums[rchildIdx] > nums[largeIdx])
        {
            largeIdx = rchildIdx;
        }

        if (largeIdx != cur_idx)
        {
            swap(nums[largeIdx], nums[cur_idx]);
            cur_idx = largeIdx;
            make_heap_fix(nums, cur_idx, heap_size);
        }
    }
    int findKthLargest(vector<int>& nums, int k) {
        int heapSize = nums.size();
        make_heap(nums, heapSize);
        for (int i = nums.size() - 1; i >= nums.size() - k + 1; --i) {
            swap(nums[0], nums[i]);
            --heapSize;
            make_heap_fix(nums, 0, heapSize);
        }
        return nums[0];
    }
};

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


 struct ListNode {
     int val;
     ListNode *next;
     ListNode() : val(0), next(nullptr) {}
     ListNode(int x) : val(x), next(nullptr) {}
     ListNode(int x, ListNode *next) : val(x), next(next) {}
 };

/**
 * 23. 合并K个升序链表
 * https://leetcode-cn.com/problems/merge-k-sorted-lists/solution/he-bing-kge-pai-xu-lian-biao-by-leetcode-solutio-2/
*/
class LC23 {
public:
    struct Status {
        int val;
        ListNode *ptr;
        bool operator < (const Status &rhs) const {
            return val > rhs.val;
        }
    };

    priority_queue <Status> q;

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        for (auto node: lists) {
            if (node) q.push({node->val, node});
        }
        ListNode head;
        ListNode *tail = &head;
        while (!q.empty()) {
            auto f = q.top(); q.pop();
            tail->next = f.ptr; 
            tail = tail->next;
            if (f.ptr->next) q.push({f.ptr->next->val, f.ptr->next});
        }
        return head.next;
    }
};

class LC295 {
public:
    priority_queue<int> lo; // max-heap
    priority_queue<int, vector<int>, greater<int>> hi; // min-heap
    /** initialize your data structure here. */
    LC295() {
        
    }
    
    void addNum(int num) {
        lo.push(num);

        hi.push(lo.top());
        lo.pop();

        if (lo.size() < hi.size())
        {
            lo.push(hi.top());
            hi.pop();
        }
    }
    
    double findMedian() {
        return lo.size() > hi.size() ? double(lo.top()) : double(lo.top() + hi.top()) / 2;
    }
};

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder* obj = new MedianFinder();
 * obj->addNum(num);
 * double param_2 = obj->findMedian();
 */