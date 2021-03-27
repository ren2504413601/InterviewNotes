#include<bits/stdc++.h>
using namespace std;
typedef long long ll;

/**
 * 7. 整数反转 -- 注意整数越界问题
 * 43. 字符串相乘 
 * 172. 阶乘后的零
 * 793. 阶乘函数后 K 个零
 * 204. 计数质数 -- 计算 n 以内的质数个数
 * 50. Pow(x, n) -- 快速幂
 * 382. 链表随机节点 -- 蓄水池算法【随机算法】
*/

struct ListNode 
{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};


class LC7 {
public:
    int reverse(int x) {
        bool ispos = x >= 0 ? true : false;
        long val = ispos ? long(x) : -long(x);
        long ans = 0;
        while (val)
        {
            ans = ans * 10 + (val % 10);
            val /= 10;
        }
        if (ispos)
        {
            return ans > INT_MAX ? 0 : ans;
        }
        else
        {
            return -ans < INT_MIN ? 0 : -ans;
        }
    }
};

class LC43 {
public:
    /**
     * https://leetcode-cn.com/problems/multiply-strings/solution/zi-fu-chuan-xiang-cheng-by-leetcode-solution/
    */
    string multiply(string num1, string num2) {
        if (num1 == "0" || num2 == "0") {
            return "0";
        }
        int m = num1.size(), n = num2.size();
        auto ansArr = vector<int>(m + n);
        for (int i = m - 1; i >= 0; i--) {
            int x = num1.at(i) - '0';
            for (int j = n - 1; j >= 0; j--) {
                int y = num2.at(j) - '0';
                ansArr[i + j + 1] += x * y;
            }
        }
        for (int i = m + n - 1; i > 0; i--) {
            ansArr[i - 1] += ansArr[i] / 10;
            ansArr[i] %= 10;
        }
        int index = ansArr[0] == 0 ? 1 : 0;
        string ans;
        while (index < m + n) {
            ans.push_back(ansArr[index]);
            index++;
        }
        for (auto &c: ans) {
            c += '0';
        }
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

class LC204 {
public:
/**
 * https://labuladong.gitbook.io/algo/suan-fa-si-wei-xi-lie/shu-xue-yun-suan-ji-qiao/da-yin-su-shu
*/
    bool isPrime(int n)
    {
        for (int i = 2; i * i <= n; ++i)
        {
            if (n % i == 0) return false;
        }
        return true;
    }
    int countPrimes(int n) {
        vector<bool> primeArr(n, true);
        for (int i = 2; i < n; ++i)
        {
            if (primeArr[i])
            {
                for (int j = 2 * i; j < n; j += i) primeArr[j] = false;
            }
        }

        int cnt = 0;
        for (int i = 2; i < n; ++i)
        {
            if (primeArr[i]) cnt++;
        }
        return cnt;

    }
};

class LC50 {
public:
    double myPow(double x, int n) {
    long long N = n;
    if (N < 0)
    {
        x = 1./x;
        N = -N;
    }
    int p = 1;
    double ans = 1.0;
    while (N > 0)
    {
        if (N&p)
        {
            ans *= x;
        }
        N >>= 1;
        x *= x;
    }
    return ans;
    }
};

class LC382 {
public:
/**
 * 蓄水池算法
 * https://leetcode-cn.com/problems/linked-list-random-node/solution/zhong-ju-si-wei-by-li-zi-he/
*/
    LC382(ListNode* head): head(head) {

    }
    
    /** Returns a random node's value. */
    int getRandom() {
        int i=2;
        ListNode* cur = head->next;
        int val = head->val;
        while(cur != nullptr) {
            if(rand() % i + 1 == 1) val = cur->val; //第 i 节点以 1/i 概率替换当前值
            i++;
            cur = cur->next;
        }
        return val;
    }
private:
    ListNode* head;
};
