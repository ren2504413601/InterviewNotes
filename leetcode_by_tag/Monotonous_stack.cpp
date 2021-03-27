#include<bits/stdc++.h>

using namespace std;

/**
 * 42. 接雨水
 * 84. 柱状图中最大的矩形
 * 496. 下一个更大元素 I
 * 503. 下一个更大元素 II
 * 739. 每日温度
*/
class LC42 {
public:
    /**
     * https://leetcode-cn.com/problems/trapping-rain-water/solution/dan-diao-zhan-jie-jue-jie-yu-shui-wen-ti-by-sweeti/
    */
    int trap(vector<int>& height) 
    {
        if (height.empty()) return 0;
        stack<int> stk;
        int ans = 0;
        for (int i = 0; i < height.size(); ++i)
        {
            while (!stk.empty() && height[stk.top()] < height[i])
            {
                int currIdx = stk.top();
                while (!stk.empty() && stk.top() == currIdx)
                {
                    stk.pop();
                }
                if (!stk.empty())
                {
                    ans += (i - stk.top() - 1) * (min(height[i], height[stk.top()]) - height[currIdx]);
                }                
            }
            stk.push(i);
        }
        return ans;
    }
};

/**
 * 84. 柱状图中最大的矩形
*/
class LC84 {
public:
    /**
     * 单调栈（单调不增） + 头尾哨兵
    */
    int largestRectangleArea(vector<int>& heights) 
    {
        int n = heights.size();
        vector<int> h(n + 2, 0);
        for (int i = 0; i < n; ++i)
        {
            h[i+1] = heights[i];
        }
        heights = h;
        n += 2;

        int area = 0;
        stack<int> stk;
        for (int i = 0; i < heights.size(); ++i)
        {
            while (!stk.empty() && heights[stk.top()] > heights[i])
            {
                int currIdx = stk.top();
                while (!stk.empty() && heights[currIdx] == heights[stk.top()])
                {
                    stk.pop();
                }
                if (!stk.empty())
                {
                    area = max(area, (i - 1 - stk.top()) * heights[currIdx]);
                }
            }
            stk.push(i);
        }
        return area;
    }
};

class LC496 {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        int len = nums2.size();
        unordered_map<int, int> numsToIdx;
        for (int i = 0; i < len; ++i)
        {
            numsToIdx[nums2[i]] = i;
        }
        // 单调栈 维护  Next Greater Number 结果存放在 vector 中
        stack<int> stk;
        vector<int> nextGreaterNumber(len, 0);
        // 从后向前遍历，因为栈结构是后进先出，所以是正着出栈
        for (int i = len-1; i >= 0; --i)
        {
            // 找到栈中第一个大于当前数组值的数
            while (!stk.empty() && stk.top() <= nums2[i])
            {
                stk.pop();
            }
            nextGreaterNumber[i] = stk.empty() ? -1 : stk.top();
            stk.push(nums2[i]);
        }
        vector<int> ans;
        for (int& val : nums1)
        {
            int tmpAns = nextGreaterNumber[numsToIdx[val]];
            ans.push_back(tmpAns);
        }
        return ans;

    }
};

class LC503 {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        int len = nums.size();
        vector<int> tmp_nums = nums;
        tmp_nums.insert(tmp_nums.end(), tmp_nums.begin(), tmp_nums.end());
        // 单调栈 生成下一个个大的元素 
        stack<int> stk;
        vector<int> nextGreaterNum(tmp_nums.size(), 0);
        for (int i = tmp_nums.size() - 1; i >= 0 ; --i)
        {
            while (!stk.empty() && stk.top() <= tmp_nums[i])
            {
                stk.pop();
            }
            nextGreaterNum[i] = stk.empty() ? -1 : stk.top();
            stk.push(tmp_nums[i]);
        }
        vector<int> ans;
        for (int i = 0; i < len; ++i)
        {
            ans.push_back(nextGreaterNum[i]);
        } 
        return ans;
    }
};


class LC739 {
public:
    /**
     * 单调栈
    */
    vector<int> dailyTemperatures(vector<int>& T) {
        int n = T.size();
        stack<int> stk;
        vector<int> ans(n, 0);
        for (int i = 0; i < n; ++i)
        {
            while (!stk.empty() && T[i] > T[stk.top()])
            {
                int preIdx = stk.top();
                stk.pop();
                ans[preIdx] = i - preIdx;
            }
            stk.push(i);
        }
        return ans;
    }
};