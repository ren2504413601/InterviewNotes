#include<bits/stdc++.h>

using namespace std;

/**
 * 42. 接雨水
 * 141. 环形链表
 * 142. 环形链表 II
 * 
 * 344. 反转字符串
 * 19. 删除链表的倒数第 N 个结点
*/

struct ListNode 
{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

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

class LC141 {
public:
    /**
     * set 存储链表
     * O(n) 空间复杂度
    */
    bool hasCycle(ListNode *head) {
        set<ListNode*> lset;
        while (head)
        {
            if (lset.find(head) != lset.end())
            {
                return true;
            }
            else
            {
                lset.insert(head);
            }
            head = head->next;
        }
        return false;
    }
    /**
     * 双指正（快慢指针）
    */
    bool hasCycle_(ListNode *head) {
        ListNode* q1 = head; // 慢指针
        ListNode* q2 = head; // 快指针
        while (q2 && q2->next)
        {
            q1 = q1->next;
            q2 = q2->next->next;
            if (q1 == q2) return true;
        }
        return false;
    }
};

class LC142 {
public:
    /**
     * 双指针
     * 快慢追及问题
     * q 快指针, p 慢指针
     * 假设整个链表包含 头结点到环入口(a个结点)， 环大小(b个节点)
     * 当 p q 第一次相遇时（p == q），
     * p 走 a 步。 q 走 a+nb。这里 n是整数
     * 并且 2a = a+nb => a = nb
     * 所以 p 走了 nb 步。只需让 p 再走 a步就可走到环的入口
     * 最后 p 走了 a+nb 步
     * 
    */
    ListNode *detectCycle(ListNode *head) {
        ListNode *p = head, *q = head;
        while (q && q -> next)
        {
            p = p -> next;
            q = q -> next -> next;
            if (p == q)
            {
                break; 
            }
        }
        if (q == nullptr || q -> next == nullptr) return nullptr; // no cycle
        q = head;
        while (p != q)
        {
            p = p -> next;
            q = q -> next;
        }
        return p;
        
    }
};

class LC344 {
public:
    void reverseString(vector<char>& s) {
        for (int i = 0; i < s.size() / 2; ++i)
        {
            swap(s[i], s[s.size() - 1 - i]);
        }
    }
};

class LC19 {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* first = head;
        ListNode* second = dummy;
        for (int i = 0; i < n; ++i) {
            first = first->next;
        }
        while (first) {
            first = first->next;
            second = second->next;
        }
        second->next = second->next->next;
        ListNode* ans = dummy->next;
        delete dummy;
        return ans;
    }
};