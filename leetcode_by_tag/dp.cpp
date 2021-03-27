#include<bits/stdc++.h>
using namespace std;

/** 
 * 509. 斐波那契数
 * dp[10] == 55
*/
class LC509 {
public:
    int fib(int N) {
        if (N < 2) return N;
        int dp[N+1];
        dp[0] = 0;
        dp[1] = 1;
        for (int i =2; i <= N; ++i)
        {
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[N];
    }
};
/**
 * 322. 零钱兑换
 * dp 数组的定义：当目标金额为 i 时，至少需要 dp[i] 枚硬币凑出
 */ 
class LC322 {
public:
    int coinChange(vector<int>& coins, int amount) {
        vector<int> dp(amount+1, amount+1);
        dp[0] = 0;
        for (int i = 0; i <= amount; ++i)
        {
            for (int j = 0; j < coins.size(); ++j)
            {
                if (i - coins[j] < 0) continue;
                dp[i] = min(dp[i], dp[i-coins[j]]+1);
            }
        }
        if (dp[amount] == amount+1) return -1;
        else return dp[amount];
    }
};

/**
 * lc300. 最长上升子序列
 * dp[i] 为考虑前 i 个元素，
 * 以第 i 个数字结尾的最长上升子序列的长度，
 * 注意 nums[i] 必须被选取
 * 状态转移： dp[i]=max(dp[j])+1,其中0≤j<i且num[j]<num[i]
 */
class LC300 {
public:
    int lengthOfLIS(vector<int>& nums) {
        int len = nums.size();
        if (len == 0) return 0;
        vector<int> dp(len, 1);
        for (int i = 1; i < len; ++i)
        {
            for (int j = i-1; j >= 0; --j)
            {
                if (nums[i] > nums[j])
                {
                    dp[i] = max(dp[i], 1+dp[j]);
                }
            }
        }
        return *max_element(dp.begin(), dp.end());
    }
};

/**
 * lc53. 最大子序和
 * dp[i] 表示考虑前 i 个元素（包含nums[i]），最大子序和
 * 这里题目要求子串是连续的，所以当前状态只会与迁移状态相关，这别与 lc300. 最长上升子序列
 * 所以 状态转移方程 ： dp[i] = max(dp[i-1]+nums[i], nums[i])
 */ 
class LC53 {
public:
    int maxSubArray(vector<int>& nums) {
        int len = nums.size();
        if (len == 0) return 0;
        vector<int> dp(len, 0);
        dp[0] = nums[0];
        for (int i = 1; i < len; ++i)
        {
            dp[i] = max(nums[i], dp[i-1]+nums[i]);
        }
        return *max_element(dp.begin(), dp.end());
    }
};

int main()
{
    LC509 lc509;
    LC322 lc322;
    // cout << lc509.fib(10);
    vector<int> vec322 = {1, 2, 5};
    cout << lc322.coinChange(vec322, 11);
    system("pause");
    return 0;
}



/**
 * 416. 分割等和子集
 * dp[i][j] 表示前i个数组包不包含和为j的子集
 */
class LC416 {
public:
    bool canPartition(vector<int>& nums) {
        int N = nums.size();
        int sum = 0;
        for (int& val : nums)
        {
            sum += val;
        }
        if (sum % 2) return false;
        vector<vector<bool>> dp(N+1, vector<bool>(sum/2+1, false));
        // 这里空集视作和为0 
        for (int i = 0; i <= N; ++i)
        {
            dp[i][0] = true;
        }
        // 状态转移
        for (int i = 1; i <= N; ++i)
        {
            for (int j = 1; j <= sum/2 ; ++j)
            {
                if (j - nums[i-1] < 0)
                {
                    dp[i][j] = dp[i-1][j];
                }
                else
                {
                    dp[i][j] = dp[i-1][j-nums[i-1]] || dp[i-1][j];
                }
            }
        }
        return dp[N][sum/2];

    }
};

/**
 * 观察 LC416 中 dp[i][j] 之和 dp[i-1][j] 或者 dp[i-1][j-nums[i-1]]有关
 * 所以可以进行状态压缩，把问题转化为 1d 的 dp 问题
 * 需要注意的是 这时需要从后向前遍历，这是为了防止正序遍历时 dp[j-nums[i-1]] 被其之前的操作更新为新的值
 */ 
class LC416_1d_dp {
public:
    bool canPartition(vector<int>& nums) {
        int N = nums.size();
        int sum = 0;
        for (int& val : nums)
        {
            sum += val;
        }
        if (sum % 2) return false;
        vector<bool> dp(sum/2+1, false);
        // 这里空集视作和为0 
        for (int i = 0; i <= N; ++i)
        {
            dp[0] = true;
        }
        // 状态转移
        for (int i = 1; i <= N; ++i)
        {
            for (int j = sum/2; j >= 1 ; --j)
            {
                if (j - nums[i-1] >= 0)
                {
                    dp[j] = dp[j-nums[i-1]] || dp[j];
                }
            }
        }
        return dp[sum/2];

    }
};

/**
 * 518. 零钱兑换 II
 * dp[i][j] 表示使用前i种凑成金额j的个数
 * 状态状态 dp[i][j] = dp[i][j-coins[i-1]] + dp[i-1][j] 当 j-coins[i-1] >= 0
 * 否则 dp[i][j] = dp[i-1][j]
 */ 
class LC518 {
public:
    int change(int amount, vector<int>& coins) {
        int len = coins.size();
        vector<vector<int>> dp(len+1, vector<int>(amount+1, 0));
        for (int i = 0; i <= len; ++i)
        {
            dp[i][0] = 1;
        }
        for (int i = 1; i <= len; ++i)
        {
            for (int j = 1; j <= amount; ++j)
            {
                if (j-coins[i-1] >= 0)
                {
                    dp[i][j] = dp[i][j-coins[i-1]] + dp[i-1][j];
                }
                else
                {
                    dp[i][j] = dp[i-1][j];
                }
            }
        }
        return dp[len][amount];

    }
};

/**
 * 
 */ 
class LC518_1d_dp {
public:
    int change(int amount, vector<int>& coins) {
        int len = coins.size();
        vector<int> dp(amount+1, 0);
        dp[0] = 1;
        for (int i = 0; i < len; ++i)
        {
            for (int j = 0; j <= amount; ++j)
            {
                if (j-coins[i] >= 0) dp[j] = dp[j] + dp[j-coins[i]];
            }
        }
        return dp[amount];

    }
};

/**
 * lc72. 编辑距离
 * dp[i][j] 表示 word1的前 i 个字母和 word2 的前 j 个字母之前的编辑距离
 * 考虑到每一步的操作（插入、删除、替换）。
 * 当前状态 dp[i][j] 可能和 dp[i-1][j]、 dp[i][j-1]、 dp[i-1][j-1]有关
 * 则有 dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+1) 
 * 当word1的第i个字符word1[i-1] != word2[j]时
 * 否则 dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]) 
 */ 


class LC72 {
public:
    int minDistance(string word1, string word2) {
        int m = word1.size(), n = word2.size();
        if (m * n == 0) return m + n;
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        // 边界初始化
        for (int i = 0; i < m+1; ++i) dp[i][0] = i;
        for (int j = 0; j < n+1; ++j) dp[0][j] = j;
        // 状态转移
        for (int i = 1; i < m+1; ++i)
        {
            for (int j = 1; j < n+1; ++j)
            {
                int left = dp[i-1][j]+1, down = dp[i][j-1]+1, left_down = dp[i-1][j-1];
                if (word1[i-1] != word2[j-1]) left_down++;
                dp[i][j] = min(left, min(down, left_down));
            }
        }
        return dp[m][n];
        
    }
};

class LC354 {
public:
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        if (envelopes.empty()) return 0;
        int m = envelopes.size();
        // 先按照 w 升序排序， w 相同情形再按照 h 降序排序
        sort(envelopes.begin(), envelopes.end(), [](const vector<int>& lhs,
        const vector<int>& rhs){
            if (lhs[0] < rhs[0])
            {
                return true;
            }
            else if (lhs[0] == rhs[0])
            {
                return lhs[1] > rhs[1];
            }
            else
            {
                return false;
            }
        });

        // 寻找排序完的 h 列的最长上升子列个数
        // dp[i] 包含 i 列的最长上升子序列个数
        int dp[m];
        fill(dp, dp + m, 1);
        for (int i = 1; i < m; ++i)
        {
            for (int j = 0; j < i; ++j)
            {
                if (envelopes[i][1] > envelopes[j][1])
                {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
        }
        return *max_element(dp, dp + m);
    }
};


/**
 * LC887. 鸡蛋掉落
 * 动态规划 + 二分搜索
 * 时间复杂度 O(K∗NlogN)，使用 hash 加入了记忆机制，理论上时间会减少
 * 因为题目中 1 <= K <= 100， 所以对每一对(K, N) 可以满足单射到 N*100+K
 * dp[K][N] 表示K 个鸡蛋，N层建筑情况下最少需要的步数。这里由于dp下表大于等于零，
 * 所以不会出现鸡蛋不够用的情况
 * 状态转移 ： dp(K,N)=1+ min(max(dp(K−1,X−1),dp(K,N−X))) 其中 1<=x<=N
 * 表示假设在第X层做一次尝试，要么鸡蛋摔碎，对应好状态（K-1, X-1）楼层小于X只剩K-1个鸡蛋
 * 如果鸡蛋没有摔碎，那么楼层高于x但小于等于N，剩余N个鸡蛋，对应状态(K, N-X)。考虑最坏情形，
 * 两者取较大者，再对所有的楼层X取小，这就建立了状态的转移
 * 对dp[K,N]的求解这里考虑到 dp(K−1,X−1)随X单增,dp(K,N−X)单减。使用二分查找求解
 */ 
class LC887_DP_BinarySearch {
public:
    unordered_map<int, int> meo;
    int dp(int K, int N)
    {
        if (meo.find(N*100+K) == meo.end())
        {
            int ans = 0;
            if (N == 0) ans = 0;
            else if (K == 1)
            {
                ans = N;
            }
            else
            {
                int l = 1, r = N;
                while (l+1 < r)
                {
                    int mid = (r-l)/2+l;
                    int f1 = dp(K-1, mid-1);
                    int f2 = dp(K, N-mid);
                    if (f1 > f2)
                    {
                        r = mid;
                    }
                    else if (f1 < f2)
                    {
                        l = mid;
                    }
                    else
                    {
                        l = mid;
                        r = mid;
                    }
                }
                ans = 1+min(max(dp(K-1, l-1), dp(K, N-l)), max(dp(K-1, r-1), dp(K, N-r)));
            }
            meo[N*100+K] = ans;
        }

        return meo[N*100+K];
    }
    int superEggDrop(int K, int N) {
        return dp(K, N);
    }
};

/**
 * LC887. 鸡蛋掉落
 * dp[i][j] 表示 做 T 次测试，有 K 个鸡蛋，能找到的最高答案
 * 则T <= N
 * 时间复杂度 O(K*N)
 */
class LC887_dp {
public:
    int superEggDrop(int K, int N) {
        if (N == 1) return 1;
        int ans = -1;
        vector<vector<int>> dp(N+1, vector<int>(K+1));
        for (int i = 1; i <= N; ++i)
        {
            dp[i][1] = i;
        }
        for (int j = 1; j <= K; ++j)
        {
            dp[1][j] = 1;
        }
        for (int i = 2; i <= N; ++i)
        {
            for (int j = 1; j <= K; ++j)
            {
                dp[i][j] = 1 + dp[i-1][j-1] + dp[i-1][j];
            }
            if (dp[i][K] >= N)
            {
                ans = i;
                break;
            }
        }
        return ans;
    }
};

class LC312 {
public:
    /**
     * 区间 DP
     * dp[i][j] 表示 i 到 j 编号的气球获得的最大收益
     * dp[l][r] = max(dp[l][r], dp[l][k - 1] + dp[k + 1][r] + nums[l - 1] * nums[k] * nums[r + 1])
     * 这里当 l < r dp[l][r] = 0
    */
    int maxCoins(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) return 0;
        if (n == 1) return nums[0];
        nums.insert(nums.begin(), 1);
        nums.push_back(1);
        vector<vector<int>> dp(n + 2, vector<int>(n + 2, 0));
        for (int len = 1; len <= n; ++len)
        {
            for (int l = 1, r = l + len - 1; r <= n; ++l, ++r)
            {
                for (int k = l; k <= r; ++k)
                {
                    dp[l][r] = max(dp[l][r], dp[l][k - 1] + dp[k + 1][r] + nums[l - 1] * nums[k] * nums[r + 1]);
                }
            }
        }
        return dp[1][n];
    }
};

/**
 * 198. 打家劫舍
 * 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，
 * 影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，
一夜之内能够偷窃到的最高金额。
*/
class LC198 {
public:
    /**
     * 动态规划 ： 
     * dp[i] 表示 前i间房屋的偷盗最高金额
     * 对第i间房屋可以选择偷盗或者不偷盗，
     * 前者 dp[i] = nums[i] + dp[i-2],
     * 后者 dp[i] = dp[i-1]
     * 状态转移方程 dp[i] = max(dp[i-2]+nums[i], dp[i-1])
    */
    int rob(vector<int>& nums) {
        int len = nums.size();
        if (len == 0) return 0;
        if (len == 1) return nums[0];
        vector<int> dp(len, 0);
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);
        for (int i = 2; i < len; ++i)
        {
            dp[i] = max(dp[i-2]+nums[i], dp[i-1]);
        }
        return dp[len-1];

    }
};

/**
 * 213. 打家劫舍 II
*/
class LC213 {
public:
    /**
     * DP rob 参考 198.打家劫舍
     * 主要是问题的转换 
     * 考虑将环形结构转化成链表结构
     * 相邻房屋最多只能偷盗一家，这里考虑从数组的开始和结束位置剪开
     * 有三种可能的情形 ：
     * 1.最优解包含 nums[0]不包含nums[end]
     * 2.最优解包含 nums[end] 不包含 nums[0]
     * 3. ~~ nums[0]和nums[end] 均不包含，但显然这种情形不是最优的
     * 对1、2情况参考198的题解取两者较大值即可
    */
    int robDP(vector<int>& nums) {
        int len = nums.size();
        if (len == 0) return 0;
        if (len == 1) return nums[0];
        vector<int> dp(len, 0);
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);
        for (int i = 2; i < len; ++i)
        {
            dp[i] = max(dp[i-2]+nums[i], dp[i-1]);
        }
        return dp[len-1];

    }
    int rob(vector<int>& nums) {
        int len = nums.size();
        if (len == 0) return 0;
        if (len == 1) return nums[0]; 
        vector<int> tmp1(nums.begin()+1, nums.end()); // -> nums[1:end]
        vector<int> tmp2(nums.begin(), nums.end()-1); // -> nums[:end-1]
        return max(robDP(tmp1), robDP(tmp2));

    }
};

/**
 * 337. 打家劫舍 III
 * 在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

    计算在不触动警报的情况下，小偷一晚能够盗取的最高金额
*/
class LC337 {
public:
    /**
     * 树形动态规划
     * 这里用「后序遍历」，这是因为：我们的逻辑是子结点陆续汇报信息给父结点，
     * 一层一层向上汇报，最后在根结点汇总值。
     * dp[node][j] ：这里 node 表示一个结点，以 node 为根结点的树，
        并且规定了 node 是否偷取能够获得的最大价值
        j = 0 表示 node 结点不偷取；
        j = 1 表示 node 结点偷取。
    */
    vector<int> dfsHelper(TreeNode* node)
    {
        vector<int> dp(2, 0);
        if (node == nullptr)
        {
            return dp;
        }
        vector<int> left = dfsHelper(node->left);
        vector<int> right = dfsHelper(node->right);
        dp[0] = max(left[0], left[1]) + max(right[0], right[1]);
        dp[1] = node->val + left[0] + right[0];
        return dp;
    }
    int rob(TreeNode* root) {
        vector<int>res = dfsHelper(root);
        return *max_element(res.begin(), res.end());

    }
};

/**
 * 494. 目标和
 * 给定一个非负整数数组，a1, a2, ..., an, 和一个目标数，S。
 * 现在你有两个符号 + 和 -。对于数组中的任意一个整数，你都可以从 + 或 -中选择一个符号添加在前面。

    返回可以使最终数组和为目标数 S 的所有添加符号的方法数。

*/
class LC494 {
public:
    /**
     * 回溯 超时
     * 时间复杂度 ： O(2^N)
    */
    void backTrack(vector<int>& nums, int start, int target, int& cnt)
    {
        if (start == nums.size())
        {
            if (target == 0) cnt++;
            return;
        }
        // 回溯部分
        target += nums[start];
        backTrack(nums, start+1, target, cnt);
        target -= nums[start];

        target -= nums[start];
        backTrack(nums, start+1, target, cnt);
        target += nums[start];
    }
    int findTargetSumWays_backTrack(vector<int>& nums, int S) {
        int cnt = 0;
        backTrack(nums, 0, S, cnt);
        return cnt;

    }
    /**
     * 动态规划
     * 加入记忆力机制。与回溯算法相比，相当于加入了剪枝
    */
    int dp(vector<int>& nums, int start, long rest, unordered_map<string, int>& memo)
    {
        if (start == nums.size())
        {
            if (rest == 0)
            {
                return 1;
            }
            return 0;
        }
        // 把它俩转成字符串才能作为哈希表的键
        string key = to_string(start) + "," + to_string(rest);
        int res;
        if (memo.find(key) != memo.end())
        {
            res = memo[key];
        }
        else
        {
            res = dp(nums, start+1, rest+nums[start], memo) + dp(nums, start+1, rest-nums[start], memo);
            memo[key] = res;
        } 
        return res;

    }
    int findTargetSumWays_dp(vector<int>& nums, int S) {
        int cnt = 0;
        unordered_map<string, int> memo;
        return dp(nums, 0, long(S), memo);
    }

    /**
     * 问题转化思想 + 动态规划
     * 假设 A 为 nums 中取加号， B 为 nums 中取减号的集合
     * 则 sum(A) - sum(B) = S
     * 又 sum(A) + sum(B) = sum(nums)
     * 所以 sum(A) = (S+sum(nums))/2
     * 问题转化为在 nums中选择若干个数使得 sum(A) = (S+sum(nums))/2
    */
    int findTargetSumWays(vector<int>& nums, int S) {
        int sum = 0;
        for (int& num : nums)
        {
            sum += num;
        }
        // sum(nums) > S 并且 (S+sum) 整除 2
        if (sum < S || (S+sum) % 2 == 1)
        {
            return 0;
        }
        return getSubsets(nums, (S+sum)/2);
        
    }
    /**
     * 动态规划求解 背包问题
     * dp[i][j]只在前 i+1 个物品中选择（0,1,2,...,n），若当前背包的容量为 j，
     * 则最多有 dp 种方法可以恰好装满背包
     * 状态转移 : nums[i] 要么选，要么不选。这取决于 j-nums[i] 是否大于等于0
     * 1、 j >= nums[i] dp[i][j] = dp[i-1][j-nums[i]] + dp[i-1][j]
     * 2、 j < nums[i] dp[i][j] = dp[i-1][j]
     * 边界：
     * dp[0][j] = 0, dp[0][0] = 1 
    */
    int getSubsets(vector<int>& nums, int target)
    {
        int len = nums.size();
        vector<vector<int>> dp(len+1, vector<int>(target+1, 0));
        dp[0][0] = 1;
        // for (int j = 1; j <= target; ++j) dp[0][j] = 0;

        for (int i = 1; i <= len; ++i)
        {
            for (int j = 0; j <= target; ++j)
            {
                if (j >= nums[i-1]) 
                {
                    dp[i][j] = dp[i-1][j-nums[i-1]] + dp[i-1][j];
                }
                else
                {
                    dp[i][j] = dp[i-1][j];
                }
            }
        }
        return dp[len][target];
    }
};

/**
 * 1143. 最长公共子序列
 * 给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度
*/
class LC1143 {
public:
    /**
     * 动态规划
     * dp[i][j] 表示 text1 前 i 个字符和 text2 前 j 个字符的最长公共子序列的长度
     * 默认的边界 dp[0][j] = dp[i][0] = 0
    */
    int longestCommonSubsequence(string text1, string text2) {
        int m = text1.size(), n = text2.size();
        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        for (int i = 1; i <= m; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                // text1的第 i 个字符是 text1[i-1]
                if (text1[i-1] == text2[j-1])
                {
                    dp[i][j] = 1 + dp[i-1][j-1];
                }
                else
                {
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }
        return dp[m][n];

    }
};

class LC1143 {
public:
    // 查找 word1 与 word2 的最长公共子序列
    // LC1143. 最长公共子序列
    // dp[i][j] 表示 word1 前 i 个字符及 word2 前 j 个字符的最长公共子序列
    int minDistance(string word1, string word2) {
        int n1 = word1.size(), n2 = word2.size();
        int dp[n1 + 1][n2 + 1];
        memset(dp, 0, sizeof(dp));

        for (int i = 1; i <= n1; ++i)
        {
            for (int j = 1; j <= n2; ++j)
            {
                if (word1[i - 1] == word2[j - 1])
                {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }
                else
                {
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j]);
                }
            }
        }

        return n1 + n2 - 2 * dp[n1][n2];
    }
};

class LC452 {
public:
    // 容易知道引爆的位置总是可以设置在 end 的位置
    // 根据 end 从小到大排序，然后对开始位置贪心策略
    int findMinArrowShots(vector<vector<int>>& points) {
        if (points.empty()) return 0;
        sort(points.begin(), points.end(), [](const vector<int>& lhs,
         const vector<int>& rhs){
             return lhs[1] < rhs[1];
         });

        int ans = 1, pos = points[0][1];
        for (const vector<int>& balloon : points)
        {
            if (balloon[0] > pos)
            {
                pos = balloon[1];
                ++ans;
            }
        }
        return ans;
    }
};

/**
 * 120. 三角形最小路径和
 * 倒序DP 
 * 辅助边界
*/
class LC120 {
public:
    /**
     * 辅助边界 m + 1 个 0
     * dp[i][j] 表示从最下层到 i 行 j 列的最小路径和
     * 状态转移 dp[i][j] = min(dp[i][j], min(dp[i + 1][j], dp[i + 1][j + 1])) + triangle[i][j]
    */
    int minimumTotal(vector<vector<int>>& triangle) {
        int m = triangle.size();
        if (m == 0) return 0;
        vector<vector<int>> dp(m + 1, vector<int>(m + 1, 0));

        for (int i = m - 1; i >= 0; --i)
        {
            for (int j = 0; j <= i; ++j)
            {
                dp[i][j] = min(dp[i + 1][j], dp[i + 1][j + 1]) + triangle[i][j];
            }
        }
        return dp[0][0];
    }
};



