#include<bits/stdc++.h>
using namespace std;

/**
 * 1. 两数之和
 * nums[i] + nums[j] = target
 * 一次遍历 + Hash
 * 时间：O(n)
*/
class LC1 {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> umap;
        for (int i = 0; i < nums.size(); ++i)
        {
            if (umap.find(target-nums[i]) != umap.end())
            {
                return {umap[target-nums[i]], i};
            }
            umap[nums[i]] = i; 
        }
        return {-1, -1};
    }
};

/**
 * 15. 三数之和
 * a + b + c = 0
 * 注意：答案中不可以包含重复的三元组
 * 排序 + 双指针
 * 时间：O(n^2)
*/
class LC15 {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int n = nums.size();
        vector<vector<int>> ans;
        for (int i = 0; i < n; ++i)
        {
            if (i > 0 && nums[i] == nums[i-1]) continue; // 跳过重复数字
            int j = i + 1, k = n - 1;
            for (; j < n; ++j)
            {
                if (j > i + 1 && nums[j] == nums[j-1]) continue; // 跳过重复数字

                while (j < k && nums[i] + nums[j] + nums[k] > 0)
                {
                    --k;
                }
                if (j == k) break;

                if (nums[i] + nums[j] + nums[k] == 0)
                {
                    ans.push_back({nums[i], nums[j], nums[k]});
                }
            }
        }
        return ans;
    }
};

/**
 * 18. 四数之和
 * 类似三数之和
 * 时间：O(n^3)
*/
class LC18 {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> ans;
        sort(nums.begin(), nums.end());
        int n = nums.size();
        for (int i = 0; i < n; ++i)
        {
            if (i > 0 && nums[i] == nums[i-1]) continue; // 跳重
            for (int j = i + 1; j < n; ++j)
            {
                if (j > i + 1 && nums[j] == nums[j-1]) continue; 

                int k = j + 1, s = n - 1;
                for (; k < n; ++k)
                {
                    if (k > j + 1 && nums[k] == nums[k-1]) continue;

                    while (s > k && nums[i] + nums[j] + nums[k] + nums[s] > target)
                    {
                        --s;
                    }

                    if (s == k) break;

                    if (nums[i] + nums[j] + nums[k] + nums[s] == target)
                    {
                        ans.push_back({nums[i], nums[j], nums[k], nums[s]});
                    }
                }
            }
        }
        return ans;
    }
};

/**
 * 454. 四数相加 II
 * 给定四个包含整数的数组列表 A , B , C , D ,计算有多少个元组 (i, j, k, l) ，
 * 使得 A[i] + B[j] + C[k] + D[l] = 0
 * 两轮Hash计数， 类似于 两数之和 
*/
class LC454 {
public:
    int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D) {
        unordered_map<int, int> umap;
        for (int& a : A)
        {
            for (int& b : B)
            {
                umap[-a-b]++;
            }
        }
        int cnt = 0;
        for (int& c : C)
        {
            for (int& d : D)
            {
                if (umap.find(c+d) != umap.end()) cnt += umap[c+d];
            }
        }
        return cnt;
    }
};