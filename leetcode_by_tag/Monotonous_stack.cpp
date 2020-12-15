#include<bits/stdc++.h>

using namespace std;

/**
 * 42. 接雨水
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