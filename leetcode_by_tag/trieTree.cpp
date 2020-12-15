#include<bits/stdc++.h>
using namespace std;

/**
 * LC208. 实现 Trie (前缀树)
 * 实现一个 Trie (前缀树)，包含 insert, search, 和 startsWith 这三个操作
*/
class LC208 {
private:
    bool isEnd;
    Trie* next[26];
public:
    /** Initialize your data structure here. */
    Trie() {
        isEnd = false;
        memset(next, 0, sizeof(next));
    }
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        Trie* node = this;
        for (char& c : word)
        {
            if (node -> next[c - 'a'] == NULL)
            {
                node -> next[c - 'a'] = new Trie();
            }
            node = node -> next[c - 'a'];
        }
        node -> isEnd = true;
    }
    
    /** Returns if the word is in the trie. */
    bool search(string word) {
        Trie* node = this;
        for (char& c : word)
        {
            if (node -> next[c - 'a'] == NULL)
            {
                return false;
            }
            node = node -> next[c - 'a'];
        }
        return node -> isEnd;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        Trie* node = this;
        for (char& c : prefix)
        {
            if (node -> next[c - 'a'] == NULL)
            {
                return false;
            }
            node = node -> next[c - 'a'];
        }
        return true;
    }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */

/**
 * LC212. 单词搜索 II
*/

class Trie
{
    public:
    Trie* next[26];
    string words;
    Trie()
    {
        words = "";
        memset(next, 0, sizeof(next));
    }
};
class LC212 {
public:
    void dfsHelper(vector<vector<char>>& board, Trie* node, int x, int y, vector<string>& ans)
    {
        int m = board.size(), n = board[0].size();
        char tmpc = board[x][y]; 
        if (tmpc == '.' || node -> next[tmpc - 'a'] == NULL) return;

        node = node -> next[tmpc - 'a'];

        if (node -> words != "")
        {
            ans.push_back(node ->words);
            node ->words = ""; // 去重， 很重要
        }

        board[x][y] = '.';
        if (x > 0) dfsHelper(board, node, x - 1, y, ans);
        if (x + 1 < m) dfsHelper(board, node, x + 1, y, ans);
        if (y > 0) dfsHelper(board, node, x, y - 1, ans);
        if (y + 1 < n) dfsHelper(board, node, x, y + 1, ans);
        board[x][y] = tmpc;
    }
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) 
    {
        vector<string> ans;
        int m = board.size();
        if (m == 0) return ans;
        int n = board[0].size();

        // 建 trie 树
        Trie* node = new Trie();
        for (string& s : words)
        {
            Trie* curr = node;
            for (char& c: s)
            {
                if (curr -> next[c - 'a'] == NULL)
                {
                    curr -> next[c - 'a'] = new Trie();
                }
                curr = curr -> next[c - 'a'];
            }
            curr -> words = s;
        }
        // dfs 
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                dfsHelper(board, node, i, j, ans);
            }
        }
        return ans;
    }
};