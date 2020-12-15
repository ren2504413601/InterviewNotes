#include<bits/stdc++.h>
using namespace std;
typedef long long ll;


class LC290 {
public:
    bool wordPattern(string pattern, string str) {
        vector<string> segs;
        string seg;
        for (char c:str) // 将str分割
        {
            if (c == ' ') 
            {
                segs.push_back(seg);
                seg.clear();
            }
            seg += c;
        }
        segs.push_back(seg);

        unordered_map<char, string> patternTostr;
        unordered_map<string, char> strTopattern;
        if (pattern.size() != segs.size()) return false;
        for (int i = 0; i < pattern.size(); ++i)
        {
            if (patternTostr.find(pattern[i]) == patternTostr.end())
            {
                patternTostr[pattern[i]] = segs[i];
            }
            else
            {
                if (patternTostr[pattern[i]] != segs[i]) return false;
            }
            if (strTopattern.find(segs[i]) == strTopattern.end())
            {
                strTopattern[segs[i]] = pattern[i];
            }
            else
            {
                if (strTopattern[segs[i]] != pattern[i]) return false;
            }
        }
        return true;
    }
};

void LC290Test()
{
    LC290 lc290;
    string s1 = "abba";
    string s2 = "dog cat cat dog";
    cout << lc290.wordPattern(s1, s2);
}

/**
 * 暴力回溯
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

void lc93_test()
{
    LC93 lc93;
    vector<string> ans = lc93.restoreIpAddresses("25525511135");
    for (string s : ans)
    {
        cout << s << endl;
    }
}

int main()
{
    lc93_test();
    system("pause");
    return 0;
}
