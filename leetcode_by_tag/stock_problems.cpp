#include<bits/stdc++/h>
using namespace std;


/**
 * 121. 买卖股票的最佳时机
 * 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
    如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），
    设计一个算法来计算你所能获取的最大利润。
    注意：你不能在买入股票前卖出股票
*/
class LC121 {
public:
    /**
     * 贪心算法
     * 一次买入卖出一定要选择极差最大的两个时刻。
     * 而且买入一定要在卖出之前。
    */
    int maxProfit(vector<int>& prices) {
        if (prices.empty()) return 0;
        // min_val 记录当前时刻之前的最小值
        // max_val 计算极差，取最大值
        int max_val = 0, min_val = prices[0];
        for (int i = 1; i < prices.size(); ++i)
        {
            max_val = max(max_val, prices[i]-min_val);
            if (min_val > prices[i])
            {
                min_val = prices[i];
            }
        }
        return max_val;
    }
};

/**
 * 122. 买卖股票的最佳时机 II
 * 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
    设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
    注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）
*/
class LC122 {
public:
    /**
     * 贪心算法
     * 统计所有上升段
    */
    int maxProfit(vector<int>& prices) {
        int pro = 0;
        for (int i = 1; i < prices.size(); ++i)
        {
            if (prices[i] > prices[i-1])
            {
                pro += (prices[i] - prices[i-1]);
            }
        }
        return pro;
    }
};

/**
 * 123. 买卖股票的最佳时机 III
 * 给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
    设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
    注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）
*/

class LC123 {
public:
    /**
     * 动态规划 dp[i][j] 表示在第i天的最大利润
     * 其中，j = 0,1,2,3,4 分别表示无买入卖出、第一次买入、
     * 第一次卖出、第二次买入、第二次卖出
    */
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        if (n ==0) return 0;
        vector<vector<int>> dp(n, vector<int>(5, 0));
        // 边界初始化， 第一行和第一列
        // j = 0 的情形没有买入卖出最大利润是0
        
        for (int i = 0; i < n; ++i)
        {
            if (i == 0) 
            {
                dp[i][0] = 0;
                dp[i][1] = -prices[0]; //第一天第一次买入
                dp[i][2] = 0;
                dp[i][3] = INT_MIN; // 
                dp[i][4] = 0;
            }
            else
            {
                dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i]);
                dp[i][2] = max(dp[i-1][2], dp[i-1][1]+prices[i]);
                dp[i][3] = max(dp[i-1][3], dp[i-1][2]-prices[i]);
                dp[i][4] = max(dp[i-1][4], dp[i-1][3]+prices[i]);
            }
        }
        return max(0, max(dp[n-1][2], dp[n-1][4]));
    }
};

/**
 * 188. 买卖股票的最佳时机 IV
 * 给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。

    设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。

    注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）

*/
class LC188 {
public:
    // 无限次交易（贪心算法）
    int maxProfit_inf(vector<int>& prices)
    {
        int max_profit = 0;
        for(int i = 0; i < prices.size() - 1; i++)
        {
            if(prices[i + 1] > prices[i])
            // 只要上涨就进行买卖股票
                max_profit += (prices[i + 1] - prices[i]);
        }
        return max_profit;
    }
    /**
     * 动态规划
     * 如果 k 大于 n/2 就相当于买卖次数没有限制
     * 否则 dp[i][j] 表示在第i天的最大利润
     * 其中，j = 0,1,2,3,4,... 分别表示无买入卖出、第一次买入、
     * 第一次卖出、第二次买入、第二次卖出...
    */
    int maxProfit(int k, vector<int>& prices) {
        int n = prices.size();
        if (n == 0 || k == 0) return 0;
        if (k > n/2)
        {
            return maxProfit_inf(prices);
        }
        // 初始化 这里 -99999是一个很大的数代表当前时刻利润不可取
        vector<vector<int>> dp(n, vector<int>(2*k+1, -99999));
        // 边界初始化， 第一行和第一列
        // j = 0 的情形没有买入卖出最大利润是0
        for (int i = 0; i < n; ++i) 
        {
            dp[i][0] = 0;
        }
        for (int j = 0; j < 2*k+1; ++j)
        {
            dp[0][j] == -99999;
        }
        // 第一天买入或者卖出的利润
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for (int i = 1; i < n; ++i)
        {
            for (int j = 1; j <= k; ++j)
            {
                dp[i][2*j-1] = max(dp[i-1][2*j-1], dp[i-1][2*j-2]-prices[i]);
                dp[i][2*j] = max(dp[i-1][2*j], dp[i-1][2*j-1]+prices[i]);
            }
        }
        int max_val = 0;
        for (int i = 1; i <= k; ++i)
        {
            max_val = max(max_val, dp[n-1][2*i]);
        }
        return max_val;
    }
};
/**
 * 309. 最佳买卖股票时机含冷冻期
 * 给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。​

    设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

    你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
    卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
*/
class LC309 {
public:
    /**
     * 动态规划
     * dp[i][~]: 第 i 天的最大收益
     * dp[i][0]: 手上持有股票的最大收益
     * dp[i][1]: 手上不持有股票，并且处于冷冻期中的累计最大收益
     * dp[i][2]: 手上不持有股票，并且不在冷冻期中的累计最大收益
    */
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        if (n == 0)
        {
            return 0;
        }
        vector<vector<int>> dp(n, vector<int>(3, 0));
        dp[0][0] = -prices[0];
        for (int i = 1; i < n; ++i)
        {
            dp[i][0] = max(dp[i-1][0], dp[i-1][2]-prices[i]);
            dp[i][1] = dp[i-1][0]+prices[i];
            dp[i][2] = max(dp[i-1][2], dp[i-1][1]);
        }
        return max(dp[n-1][1], dp[n-1][2]);
    }
};

/**
 * 714. 买卖股票的最佳时机含手续费
 * 给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；非负整数 fee 代表了交易股票的手续费用。

    你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。

    返回获得利润的最大值。

    注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。
*/
class LC714 {
public:
    /**
     * 动态规划
     * dp[i][j] 代表第i天状态j时最大lir
     * 其中 j = 0 表示不持股 j = 1 表示持股
    */
    int maxProfit(vector<int>& prices, int fee) {
        int n = prices.size();
        if (n < 2) return 0;
        vector<vector<int>> dp(n, vector<int>(2, 0));
        dp[0][0] = 0;
        dp[0][1] = -prices[0]-fee;
        for (int i = 1; i < n; ++i)
        {
            dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i]);
            dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i]-fee);
        }
        return dp[n-1][0];

    }
};

