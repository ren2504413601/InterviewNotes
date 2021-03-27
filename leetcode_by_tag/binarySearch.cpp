#include<bits/stdc++.h>
using namespace std;

/**
 * 704. 二分查找
 * 34. 在排序数组中查找元素的第一个和最后一个位置
 * 172. 阶乘后的零
 * 793. 阶乘函数后K个零
*/

class LC704 {
public:
    int search(vector<int>& nums, int target) {
        int l = 0, r = nums.size() - 1;
        while (l <= r)
        {
            int mid = (r - l) / 2 + l;
            if (nums[mid] == target) return mid;
            else if (nums[mid] > target)
            {
                r = mid - 1;
                
            }
            else 
            {
                l = mid + 1;
            }
        }
        return -1;

    }
};

class LC34 {
public:
    /**
     * https://labuladong.gitbook.io/algo/bi-du-wen-zhang/er-fen-cha-zhao-xiang-jie
    */
    vector<int> searchRange(vector<int>& nums, int target) {
        int l, r;
        vector<int> ans;
        l = 0;
        r = nums.size() - 1;

        // 寻找左边界
        while (l <= r)
        {
            int mid = (r - l) / 2 + l;
            if (nums[mid] >= target)  // 相等情况也继续向左缩小区间
            {
                r = mid - 1;
            }
            else
            {
                l = mid + 1;
            }
        }
        // 检查出界情况
        if (l >= nums.size() || nums[l] != target) ans.push_back(-1);
        else ans.push_back(l);

        // 寻找右边界
        l = 0;
        r = nums.size() - 1;
        while (l <= r)
        {
            int mid = (r - l) / 2 + l;
            if (nums[mid] <= target) // 相等情况继续向右缩小区间
            {
                l = mid + 1;
            }
            else
            {
                r = mid - 1;
            }
        }
        // 检查出界情况
        if (r < 0 || nums[r] != target) ans.push_back(-1);
        else ans.push_back(r);
        

        return ans;

    }
};


class LC172 {
public:
    int trailingZeroes(int n) {
        int res = 0;
        for (int d = n; d / 5 > 0; d = d / 5)
        {
            res += d / 5;
        }
        return res;
    }
};


class LC793 {
public:
    /**
     * K 给定后，末尾 0 的个数为 K 个的数要么是 5 个， 要么只能是 0.
     * https://leetcode-cn.com/problems/preimage-size-of-factorial-zeroes-function/solution/jie-cheng-han-shu-hou-kge-ling-by-leetcode/
    */
    long trailingZeroes(long n) 
    {
        long res = 0;
        for (long d = n; d / 5 > 0; d = d / 5)
        {
            res += d / 5;
        }
        return res;
    }
    int preimageSizeFZF(int K) 
    {
        long l = 0, r = 10 * long(K) + 1;
        while (l <= r)
        {
            long mid = (r - l) / 2 + l;
            if (trailingZeroes(mid) == K) return 5;
            else if (trailingZeroes(mid) > K)
            {
                r = mid - 1;
            }
            else
            {
                l = mid + 1;
            } 
        }
        return 0;
    }
};