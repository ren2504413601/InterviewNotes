#include<bits/stdc++/h>
using namespace std;

/**
 * 3. 无重复字符的最长子串
*/
class LC3 {
public:
    int lengthOfLongestSubstring(string s) {
        set<char> S;
        int l = 0, r = 0;
        int max_len = 0;
        while (r < s.size())
        {
            if (S.find(s[r]) == S.end())
            {
                S.insert(s[r]);
                ++r;
                max_len = max(max_len, r - l);
            }
            else
            {
                S.erase(s[l]);
                ++l;
            }
           
        }
        return max_len;
    }
};

/**
 * 239. 滑动窗口最大值
 * 给定一个数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。
 * 你只可以看到在滑动窗口内的 k 个数字。
*/
class LC239 {
public:
    /**
     * 使用单调队列（单减双向）维护
     * 队头元素是当前窗口最大值的index
     * 始终需要(i-k) < deq.front()。这说明队头元素在窗口内
     * 否则（ (i-k) == deq.front() ）需要弹出队头元素
    */
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int> ans;
        deque<int> deq; 
        for (int i = 0; i < n; ++i)
        {
            // 弹出所有比当前值小的队列元素
            while(!deq.empty() && nums[i] >= nums[deq.back()])
            {
                deq.pop_back();
            }

            if (!deq.empty() && (i-k) == deq.front())
            {
                deq.pop_front();
            }
            deq.push_back(i);
            if (i >= k-1)
            {
                ans.push_back(nums[deq.front()]);
            }
        }
        return ans;
    }
};

/**
 * 76. 最小覆盖子串
 * 给你一个字符串 S、一个字符串 T，请在字符串 S 里面找出：包含 T 所有字符的最小子串
*/
class LC76 {
public:
    /**
     * 滑动窗口
     * 对 S 的子串是否包含T这里使用 isValid 实现
     * 考虑使用 hasn table 计数， 然后判断 ori的计数个数
     * 是不是全大于等于 cnt 的计数
    */
    bool isValid(unordered_map<char, int>& ori, unordered_map<char, int>& cnt)
    {
        for (auto it = ori.begin(); it != ori.end(); ++it)
        {
            if (cnt[it->first] < it->second)
            {
                return false;
            }
        }
        return true;
    }
    string minWindow(string s, string t) {
        unordered_map<char, int> ori, cnt;
        for (char& c : t)
        {
            ++ori[c];
        }

        int l = 0, r = 0, ansL = -1, len = INT_MAX;
        while (r < s.size())
        {
            if (ori.find(s[r]) != ori.end())
            {
                ++cnt[s[r]];
            }
            while (isValid(ori, cnt) && l <= r)
            {
                if (r-l+1 < len)
                {
                    ansL = l;
                    len = r-l+1;
                }
                if (ori.find(s[l]) != ori.end())
                {
                    --cnt[s[l]];
                }
                ++l;
            }
            ++r;
        }
        if (ansL == -1) return "";
        return s.substr(ansL, len);
    }
};

/**
 * 567. 字符串的排列
 * 给定两个字符串 s1 和 s2，写一个函数来判断 s2 是否包含 s1 的排列。

    换句话说，第一个字符串的排列之一是第二个字符串的子串
*/
class LC567 {
public:
    /**
     * 滑动窗口 + Hansh Table(基于vector) 
     * 因为输入的字符串只包含小写字母，
     * 考虑char 转为 int  (s1[i] - 'a') 作为下表索引用vector计数
    */
    bool check(vector<int>& map1, vector<int>& map2)
    {
        for (int i = 0; i < 26; ++i)
        {
            if (map2[i] != map1[i])
            {
                return false;
            }
        }
        return true;
    }
    bool checkInclusion(string s1, string s2) {
        int len1 = s1.size(), len2 = s2.size();
        if (len2 < len1) return false;
        vector<int> map1(26, 0);
        vector<int> map2(26, 0);
        for (int i = 0; i < len1; ++i)
        {
            ++map1[s1[i] - 'a'];
            ++map2[s2[i] - 'a'];
        }
        for (int j = len1; j < len2; ++j)
        {
            if (check(map1, map2))
            {
                return true;
            }
            ++map2[s2[j] - 'a'];
            --map2[s2[j-len1] - 'a'];
        }
        return check(map1, map2);
    }
};

/**
 * 438. 找到字符串中所有字母异位词
 * 给定一个字符串 s 和一个非空字符串 p，找到 s 中所有是 p 的字母异位词的子串，
 * 返回这些子串的起始索引。

字符串只包含小写英文字母，并且字符串 s 和 p 的长度都不超过 20100。

说明：

字母异位词指字母相同，但排列不同的字符串。
不考虑答案输出的顺序。
*/
class LC438 {
public:
    /**
     * 滑动窗口 + Hansh Table(基于vector) 
    */
    bool check(vector<int>& map1, vector<int>& map2)
    {
        for (int i = 0; i < 26; ++i)
        {
            if (map1[i] < map2[i])
            {
                return false;
            }
        }
        return true;
    }
    vector<int> findAnagrams(string s, string p) {
        int len1= s.size(), len2= p.size();
        vector<int> ans;
        if (len2 > len1) return ans;
        vector<int> map1(26, 0), map2(26, 0);
        for (int i = 0; i < len2; ++i)
        {
            ++map2[p[i] - 'a'];
            ++map1[s[i] - 'a'];
        }
        for (int j = len2; j < len1; ++j)
        {
            if (check(map1, map2))
            {
                ans.push_back(j-len2);
            }
            ++map1[s[j] - 'a'];
            --map1[s[j-len2] - 'a'];
        }
        // 对最末尾子串的判断
        if (check(map1, map2))
        {
            ans.push_back(len1-len2);
        }
        return ans;

    }
};

/**
 * 424. 替换后的最长重复字符
 * 滑动窗口 
 * l 和 r 确定 左右边界，所以当前窗口的长度是 r - l + 1
 * 如果当前窗口大小 > max_len + k 则说明不能通过修改 k 个字符得到重复子串，需要移动左指针 l
 * 注意：每一次只考虑比之前更大的窗口
*/
class LC424 {
public:
    int characterReplacement(string s, int k) {
        vector<int> lowerCaseMap(26, 0);
        int l = 0, r = 0;
        int max_len = 0;
        while (r < s.size())
        {
            lowerCaseMap[s[r] - 'A']++;
            max_len = max(max_len, lowerCaseMap[s[r] - 'A']);
            if (r+1-l-k > max_len)
            {
                lowerCaseMap[s[l] - 'A']--;
                ++l;
            }
            ++r;
        }
        return s.size() - l;
    }
};