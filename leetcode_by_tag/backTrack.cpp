#include<bits/stdc++.h>
using namespace std;
typedef long long ll;

/**
 * Leetcode常见回溯算法实现
 * Easy/Middle 难度：
 * LC46.全排列 LC47. 全排列 II
 * 77.组合     78.子集 
 * LC22. 括号生成 93. 复原IP地址
 * Hard 难度：（需要剪枝或者说判断解是否可行）
 * 51. N皇后    37. 解数独
 * 39. 组合总和  40. 组合总和 II
 * 
 * LC17. 电话号码的字母组合
 * LC60. 第k个排列
 * 
 * 131. 分割回文串
 * 132. 分割回文串 II
*/


class LC39 {
public:
    vector<vector<int>> ans;
    void backTrack(int start, int target, vector<int>& candidates, vector<int> tmpAns, int tmpCnt)
    {
        if (tmpCnt == target) ans.push_back(tmpAns);
        for (int i = start; i < candidates.size(); ++i)
        {
            if (tmpCnt + candidates[i] > target) break;
            tmpAns.push_back(candidates[i]);
            backTrack(i, target, candidates, tmpAns, tmpCnt + candidates[i]);
            tmpAns.pop_back();
        }
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        backTrack(0, target, candidates, {}, 0);
        return ans;
    }
};

class LC40 {
public:
    vector<vector<int>> ans;
    void backTrack(int start, vector<int>& candidates, int target, vector<int> tmpAns, int tmpCnt)
    {
        if (tmpCnt == target) ans.push_back(tmpAns);

        for (int i = start; i < candidates.size(); ++i)
        {
            
            if (tmpCnt + candidates[i] > target) break;
            tmpAns.push_back(candidates[i]);
            backTrack(i + 1, candidates, target, tmpAns, tmpCnt + candidates[i]);
            tmpAns.pop_back();

            while (i + 1 < candidates.size() && candidates[i] == candidates[i + 1]) ++i;
        }
    }
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        backTrack(0, candidates, target, {}, 0);
        return ans;
    }
};

/**
 * 46. 全排列
 * 给定一个 没有重复 数字的序列，返回其所有可能的全排列
 * 搜索回溯
 */ 
class LC46 {
public:
    vector<vector<int>> paths;
    void backTrack(vector<int>& nums, int start)
    {
        if (nums.size() == start)
        {
            paths.push_back(nums);
            return;
        }
        for (int i = start; i < nums.size(); ++i)
        {
            swap(nums[i], nums[start]);
            backTrack(nums, start+1);
            swap(nums[i], nums[start]);
        }
    }
    void permute(vector<int>& nums) {
        backTrack(nums, 0);
    }
    void print_result(vector<int>& nums)
    {
        permute(nums);
        int m = paths.size(), n = paths[0].size();
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                cout << paths[i][j] << " ";
            }
            cout << endl;
        }
    }
};

class LC47 {
private:    
    vector<vector<int>> ans;   
public:
    void backTrack(vector<int>& nums, int currIdx)
    {
        if (currIdx == nums.size())
        {
            for(vector<int> an:ans) //结果保存并且剪枝
            {
                if (an==nums) return;
            }
            ans.push_back(nums);
            return;
        }

        for (int i = currIdx; i < nums.size(); ++i)
        {
            swap(nums[i], nums[currIdx]);
            backTrack(nums, currIdx + 1);
            swap(nums[i], nums[currIdx]);
            while (i + 1 < nums.size() && nums[i] == nums[i + 1])
            {
                ++i;
            }
        }
    }
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        if (nums.empty()) return ans;
        sort(nums.begin(), nums.end());
        backTrack(nums, 0);
        return ans;

    }
};

/**
 * 51. N皇后
 * PS ： 皇后可以攻击同一行、同一列、左上左下右上右下四个方向的任意单位
 * 回溯 + 规则判断（canPlace）
 */ 
class LC51 {
public:
    vector<string> genQueen(vector<int>& queenMap, int n)
    {
        
        vector<string> queen(n, string(n, '.'));
        for (int i = 0; i < n; ++i)
        {
            queen[i][queenMap[i]] = 'Q';
        }
        return queen;
    }
    bool canPlace(int k, vector<int>& queenMap) //判断 k 行是否可以放置 'Q'
    {
        for (int i = 0; i < k; ++i)
        {
            if (queenMap[i] == queenMap[k] ||
                abs(i-k) == abs(queenMap[i] - queenMap[k]) )
            {
                return false;
            }
        }
        return true;
    }
    // 当前行数k层的回溯
    void backTrack(int k, int n, vector<int>& queenMap, vector<vector<string>>& queenAns) 
    {
        if (k == n)
        {
            queenAns.push_back(genQueen(queenMap, n));
            return;
        }
        for (int i = 0; i < n; ++i)
        {
            /* 回溯部分 */
            queenMap[k] = i;
            if (canPlace(k, queenMap)) 
            {
                backTrack(k+1, n, queenMap, queenAns);
            }
        }
    }
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> queenAns;
        vector<int> queenMap(n, 0); 
        backTrack(0, n, queenMap, queenAns);
        return queenAns;
    }
};

/**
 * 37. 解数独
*/
class LC37 {
public:
    bool isValid(vector<vector<char>>& board, int i, int j, char ch)
    {
        for (int ix = 0; ix < 9; ++ix)
        {
            if (board[i][ix] == ch) return false;
            if (board[ix][j] == ch) return false;
            if (board[(i/3)*3+ ix/3][(j/3)*3+ ix%3] == ch) return false;
        }
        return true;
    }
    bool backTrack(vector<vector<char>>& board, int i, int j)
    {
        int m = board.size(), n = board[0].size();
        // 已经完成所有搜索
        // 自底向上依次返回
        if (i == m) 
        {
            return true;
        }
        // 已搜索至行尾
        // 开始新一行的搜索
        if (j == n)
        {
            return backTrack(board, i+1, 0);
        }
        // 当前点有初始值
        if (board[i][j] != '.')
        {
            return backTrack(board, i, j+1);
        }

        for (char ch = '1'; ch <= '9'; ++ch)
        {
            // 判断在(i, j)位置放入ch是否可行
            // 不可行则直接执行下一步
            if (!isValid(board, i, j, ch))
            {
                continue;
            }
            board[i][j] = ch;
            if (backTrack(board, i, j+1))
            {
                return true;
            }
            board[i][j] = '.';
        }
        return false;
    }
    void solveSudoku(vector<vector<char>>& board) {
        backTrack(board, 0, 0);
    }
};

int main()
{
    vector<int> nums46 = {1, 2, 3};
    LC46 lc46;
    lc46.print_result(nums46);

    string sol(8, '.');
    cout << sol << endl;
    
    system("pause");
    return 0;
}

/**
 * 78. 子集
 * 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）
*/
class LC78 {
public:
    vector<int> tmpSet;
    vector<vector<int>> sets;
    void backTrack(int start, vector<int>& nums)
    {
        sets.push_back(tmpSet);
        for (int i = start; i < nums.size(); i++)
        {
            tmpSet.push_back(nums[i]);
            backTrack(i+1, nums);
            tmpSet.pop_back();
        }
    }
    /**
     * 自底向上的递归
     */
    vector<vector<int>> recur(vector<int>& nums)
    {
        int len = nums.size();
        vector<vector<int>> ans = {{}};
        if (len == 0) return ans;
        // 最后一个元素先不考虑，求子集
        int back = nums.back();
        nums.pop_back();
        ans = recur(nums);
        int size = ans.size();
        // 在前边基础上加上包含最后一个元素的所有子集
        for (int i = 0; i < size; ++i)
        {
            ans.push_back(ans[i]);
            ans.back().push_back(back);
        }
        return ans;  
    }
    /**
     * 递归的另一种写法
    */
    vector<vector<int>> recur1(vector<int>& nums)
    {
        vector<vector<int>> ans = {{}};
        for (int i = 0; i < nums.size(); ++i)
        {
            vector<vector<int>> ansCopy = ans;
            int size = ans.size();
            for (int j = 0; j < size; ++j)
            {
                ansCopy[j].push_back(nums[i]);
                ans.push_back(ansCopy[j]);
            }
        }
        return ans;
    }
    vector<vector<int>> subsets(vector<int>& nums) {
        // base case，返回一个空集
        if (nums.empty()) return {{}};
        backTrack(0, nums);
        return sets;
    }
};

/**
 * 77. 组合
 * 给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合
*/
class LC77 {
public:
    void backTrack(int start, int k, int n, vector<int>& comb, vector<vector<int>>& ans)
    {
        if (comb.size() == k)
        {
            ans.push_back(comb);
        }
        for (int i = start; i <= n; ++i)
        {
            comb.push_back(i);
            backTrack(i+1, k, n, comb, ans);
            comb.pop_back();
        }
    }
    vector<vector<int>> combine(int n, int k) {
        vector<vector<int>> ans;
        if (n < k) return ans;
        vector<int> comb;
        backTrack(1, k, n, comb, ans);
        return ans;
    }
};

/**
 * 22. 括号生成
 * 数字 n 代表生成括号的对数，
 * 请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
*/
class LC22 {
public:
    void backTrack(vector<string>& ans, int Lremain, int Rremain, string str)
    {
        if (Lremain > Rremain)
        {
            return;
        }
        if (Lremain < 0 || Rremain < 0)
        {
            return;
        }
        if (Lremain == 0 && Rremain == 0)
        {
            ans.push_back(str);
        }
        str.push_back('('); // 选择
        backTrack(ans, Lremain-1, Rremain, str);
        str.pop_back(); // 撤销选择

        str.push_back(')'); // 选择
        backTrack(ans, Lremain, Rremain-1, str);
        str.pop_back(); // 撤销选择
    }
    vector<string> generateParenthesis(int n) {
        vector<string> ans;
        if (n == 0) return ans;
        string str;
        backTrack(ans, n, n, str);
        return ans;
    }
};

/**
 * 93. 复原IP地址
*/
class LC93 {
public:
    void backTrack(string s, vector<string>& ans, int split_num, string currStr, int pre, int curr)
    {
        if (split_num == 4)
        {
            if (curr == s.size() + 1)
            {
                ans.push_back(currStr.substr(0, currStr.size() - 1));
            }
            return;
        }
        for (int i = 1; i <= 3 && pre + i <= s.size() ; ++i)
        {
            string curr_split = s.substr(pre, i);
            if (stoi(curr_split) >= 0 && stoi(curr_split) <= 255 
            && to_string(stoi(curr_split)) == curr_split)
            {
                backTrack(s, ans, split_num+1, currStr + curr_split + ".",
                pre + i, pre + i + 1); 
            }
        }
    }
    vector<string> restoreIpAddresses(string s) {
        vector<string> ans;
        if (s.size() > 12)
        return ans;
        string currStr;
        backTrack(s, ans, 0, currStr, 0, 1);
        return ans;
    }
};


class LC17 {
public:
    vector<vector<char>> digit_map;
    vector<string> ans;
    void backTrack(int start, string& digits, string tmpAns)
    {
        if (start == digits.size())
        {
            ans.push_back(tmpAns);
            return;
        }
        int tnum = digits[start] - '0';
        for (char& c : digit_map[tnum])
        {
            if (c == '*') continue;
            tmpAns.push_back(c);
            backTrack(start + 1, digits, tmpAns);
            tmpAns.pop_back();
        }
        
    }
    vector<string> letterCombinations(string digits) {
        if (digits.empty()) return ans;
        digit_map.assign(10, vector<char>(4, '*'));
        digit_map[2] = {'a', 'b', 'c'};
        digit_map[3] = {'d', 'e', 'f'};
        digit_map[4] = {'g', 'h', 'i'};

        digit_map[5] = {'j', 'k', 'l'};
        digit_map[6] = {'m', 'n', 'o'};
        digit_map[7] = {'p', 'q', 'r', 's'};

        digit_map[8] = {'t', 'u', 'v'};
        digit_map[9] = {'w', 'x', 'y', 'z'};

        backTrack(0, digits, {});
        return ans;

    }
};

class LC39 {
private:
    vector<vector<int>> ans;
public:
    void backTrack(int currIdx, int target, vector<int>& candidates, vector<int>& tmpAns, int& tmpSum)
    {
        if (tmpSum == target)
        {
            ans.push_back(tmpAns);
            return;
        }
        for (int i = currIdx; i < candidates.size() && tmpSum + candidates[i] <= target; ++i)
        {
            tmpAns.push_back(candidates[i]);
            tmpSum += candidates[i];
            backTrack(i, target, candidates, tmpAns, tmpSum);
            tmpAns.pop_back();
            tmpSum -= candidates[i];
        }
        return;
    }
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        if (candidates.empty()) return ans;
        sort(candidates.begin(), candidates.end());
        if (target < candidates.front()) return ans;
        vector<int> tmpAns; int tmpSum = 0;
        backTrack(0, target, candidates, tmpAns, tmpSum);
        return ans;
    }
};

class LC40 {
private:
    vector<vector<int>> ans;
public:
    void backTrack(vector<int>& candidates, int& target, int currIdx, int currSum, vector<int>& currVec)
    {
        if (currSum == target)
        {
            ans.push_back(currVec);
            return;
        }
        for (int i = currIdx; i < candidates.size() && currSum + candidates[i] <= target; ++i)
        {
            currSum += candidates[i];
            currVec.push_back(candidates[i]);
            backTrack(candidates, target, i + 1, currSum, currVec);
            currSum -= candidates[i];
            currVec.pop_back();
            while (i + 1 < candidates.size() && candidates[i] == candidates[i + 1])
            {
                ++i;
            }
        }
        return;
    }
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        if (candidates.empty()) return ans;
        sort(candidates.begin(), candidates.end());
        if (target < candidates.front()) return ans;
        vector<int > currVec;
        backTrack(candidates, target, 0, 0, currVec);
        return ans;   
    }
};

class LC60 {
public:
    string getPermutation(int n, int k) {

        string ans;
        if (n < 1) return ans;
        vector<int> factorial(n, 1);
        factorial[0] = 1;
        for (int i = 1; i < n; ++i)
        {
            factorial[i] = factorial[i - 1] * i;
        }

        --k;
        vector<int> vis(n + 1, 0); // 1表示已访问，0未访问
        for (int i = 1; i <= n; ++i)
        {
            int order = k / factorial[n - i] + 1;

            for (int j = 1; j <= n; ++j)
            {
                if (vis[j] == 1) continue;

                order--;
                if (order == 0)
                {
                    ans += (j + '0');
                    vis[j] = 1;
                    break;
                }
            }

            k = k % factorial[n - i];
        }
        return ans;
        
    }
};


class LC131 {
public:
    int N;
    vector<vector<string>> res;
    bool Judge(string str)
    {
        int n = str.size();
        for (int i = 0; i < n / 2; ++i)
        {
            if (str[i] != str[n - 1 - i]) return false;
        }
        return true;
    }
    void backTrack(int start, string& s, vector<string> tmpVec)
    {
        if (start >= N)
        {
            res.push_back(tmpVec);
            return;
        }
        for (int t = 1; t + start <= N; ++t)
        {
            string ts = s.substr(start, t);
            if (Judge(ts))
            {
                tmpVec.push_back(ts);
                backTrack(t + start, s, tmpVec);
                tmpVec.pop_back();
            }
        }
    }
    vector<vector<string>> partition(string s) {
        N = s.size();
        backTrack(0, s, {});
        return res;
    }
};


class LC132 {
public:

    bool Judge(string tmps)
    {
        int tn = tmps.size();
        for (int i = 0; i < tn / 2; ++i)
        {
            if (tmps[i] != tmps[tn - 1 - i]) return false;
        }
        return true;
    }

    int minCut(string s) {
        int n = s.size();
        int dp[n];
        for (int i = 0; i < n; ++i)
        {
            if (Judge(s.substr(0, i + 1)))
            {
                dp[i] = 0;
            }
            else
            {
                dp[i] = 0x3f3f3f3f;
                for (int j = 0; j < i; ++j)
                {
                    if (Judge(s.substr(j + 1, i - j)))
                    {
                        dp[i] = min(dp[i], dp[j] + 1);
                    }
                }
            }
        }
        return dp[n - 1];
    }
};