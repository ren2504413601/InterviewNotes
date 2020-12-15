#include<bits/stdc++.h>

using namespace std;

/**
 * 42. 接雨水
*/
class LC42 {
public:
    /**
     * 对于数组中的每个元素，
     * 找出下雨后水能达到的最高位置，
     * 等于两边最大高度的较小值减去当前高度的值
    */
    int trap(vector<int>& height) {
        int l = 0, r = height.size() - 1;
        int lmax = 0, rmax = 0, ans = 0;
        while (l < r)
        {
            if (height[l] < height[r])
            {
                height[l] > lmax ? lmax = height[l] : ans += (lmax - height[l]);
                ++l;
            }
            else
            {
                height[r] > rmax ? rmax = height[r] : ans += (rmax - height[r]);
                --r;
            }
        }
        return ans;
        
    }
};