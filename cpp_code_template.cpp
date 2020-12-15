#include<bits/stdc++.h>
using namespace std;
typedef long long ll;

// inf = 0x3f3f3f3f 很大的值（10^9量级） 
// 不是INT_MAX(2^32-1)，主要为了防止溢出整数
const int inf = 0x3f3f3f3f; 

/**
 * string to int\ long\ long long :
 * std::atoi, std::atol, std::atoll
 * c++11标准增加了全局函数std::to_string:

    string to_string (int val);

    string to_string (long val);

    string to_string (long long val);

    string to_string (unsigned val);

    string to_string (unsigned long val);

    string to_string (unsigned long long val);

    string to_string (float val);

    string to_string (double val);

    string to_string (long double val);
*/


/*
** 内置字符 函数
islower(char c) 是否为小写字母
isupper(char c) 是否为大写字母
isdigit(char c) 是否为数字
isalpha(char c) 是否为字母
isalnum(char c) 是否为字母或者数字
toupper(char c) 字母小转大
tolower(char c) 字母大转小
*/

/**
 * Definition for singly-linked list.
 */ 
struct ListNode 
{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};


// Definition for a binary tree node.
struct TreeNode 
{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};




#define compareByCustmFuc

/**
 * leetcode14 最长公共前缀
 * 输入: ["flower","flow","flight"]
 * 输出: "fl"
 */

#ifdef compareByCustmFuc
/**
 * 自定义 compare function 可以是结构体、non-static member function
 * 也可以使用 lambda 表达式
*/

// compare function 可以定义为结构体
struct {
  bool operator() (string& a, string& b) 
    {
        return a.size() < b.size();
    }
} compBystruct;

// 或者也可以是 non-static member function
bool compByMemberFunc (string& a, string& b)
{
    return a.size() < b.size();
}
class LC14 {
public:
    string longestCommonPrefix(vector<string>& strs) {
        string ans;
        if (strs.size() == 0) return ans;
        if (strs.size() == 1) return strs[0];
        // 根据 comp  从小到大排序
        // 这里 comp 即字符串长度大小 
        sort(strs.begin(), strs.end(), compByMemberFunc);
        // 也可以使用 lambda 表达式排序
        sort(strs.begin(), strs.end(), [](string& a, string& b){
            return a.size() < b.size();
        });
        for (int i = 0; i < strs[0].size(); ++i)
        {
            char tmp = strs[0][i];
            for (int j = 1; j < strs.size(); ++j)
            {
                if (strs[j][i] != tmp) return ans;
            }
            ans += tmp;
        }
        return ans;
    }
};

int getNum(int a)
{
    static int b = 1;
    if (1 != a)
    {
        b = a;
    }
    return b;
}

typedef struct Test
{
    char a:1;
    char b:2;
    char c:6;
}Test;

int main()
{
    // LC14 lc14;
    // vector <string> sample= {"flower", "flow", "flight"};
    // string ans = lc14.longestCommonPrefix(sample);
    // cout << "The results of test sample:" << ans << endl;

    char a = 121;
    for (int i = 0; i < 16; ++i)
    {
        a++;
    }
    printf("%d\n", a);
    // int a =getNum(5);
    // int b =getNum(1);
    // printf("%d %d\n", a, b);
    // cout << sizeof(Test);
    system("pause");
    return 0;
}
#endif


