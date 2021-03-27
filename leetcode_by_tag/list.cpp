#include<bits/stdc++.h>
using namespace std;

/**
 * 力扣刷题总结之链表 https://leetcode-cn.com/circle/article/YGr54o/
 * 
 * 迭代法：
 * 剑指 Offer 24. 反转链表
 * 剑指 Offer 22. 链表中倒数第k个节点
 * 剑指 Offer 18. 删除链表的节点
 * 剑指 Offer 06. 从尾到头打印链表
 * 剑指 Offer 52. 两个链表的第一个公共节点
 * 21. 合并两个有序链表
 * 
 * 
 * 递归法：
 * 递归的思想相对迭代思想，稍微有点难以理解，
 * 处理的技巧是：不要跳进递归，而是利用明确的定义来实现算法逻辑
 * 24. 两两交换链表中的节点
 * 92. 反转链表 II
 * 148. 排序链表
 * 23. 合并K个升序链表
 * 25. K 个一组翻转链表
 * 234. 回文链表
*/

struct ListNode
{
    int val;
    ListNode* next;
    ListNode(int _val) : val(_val)
    {}
};

/**
 * 二路归并
 */
class LC21 {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* p = new ListNode(-1);
        ListNode* res = p;
        while (l1 && l2)
        {
            if (l1->val < l2->val)
            {
                p->next = l1;
                l1 = l1->next;
            }
            else
            {
                p->next = l2;
                l2 = l2->next;
            }
            p = p->next;
        }
        // while (l1)
        // {
        //     p->next = l1;
        //     l1 = l1->next;
        //     p = p->next;
        // }
        // while (l2)
        // {
        //     p->next = l2;
        //     l2 = l2->next;
        //     p = p->next;
        // }
        if (l1) p->next = l1;
        else p->next = l2;
        return res->next;
    }
};

/**
 * 翻转链表
*/
class JZOffer24 {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = NULL;
        ListNode* curr = head;
        ListNode* next;
        while (curr)
        {
            next = curr->next;

            curr->next = pre;
            pre = curr;
            curr = next;
        }
        return pre;
    }
};


class JZOffer22 {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        ListNode* p = head;
        ListNode* q = head;
        while (k--)
        {
            p = p->next;
        }
        while (p)
        {
            p = p->next;
            q = q-> next;
        }
        return q;

    }
};

class JZOffer18 {
public:
    ListNode* deleteNode(ListNode* head, int val) {
        if (!head) return NULL;
        ListNode *p = head, *q = head;
        if (p->val == val) return p->next;

        while (p->next)
        {
            q = p;
            p = p->next;
            if (p->val == val)
            {
                q->next = p->next;
                break;
            }
        }
        return head;
    }
};

class JZOffer06 {
public:
    vector<int> reversePrint(ListNode* head) {
        vector<int> res;
        while(head)
        {
            res.push_back(head -> val);
            head = head -> next;
        }
        reverse(res.begin(), res.end());
        return res;

    }
};

class JZOffer52 {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode *p = headA, *q = headB;
        while (1)
        {
            if (p == q) break;

            if (p == NULL) p = headB;
            else p = p->next;

            if (q == NULL) q = headA;
            else q = q->next;
        } 
        return p;
        
    }
};

class LC24 {
public:
    ListNode* swapPairs(ListNode* head) {
        if (head == NULL || head->next == NULL) return head;

        ListNode* newNode = head->next;
        head->next = swapPairs(newNode->next);
        newNode->next = head;
        return newNode;
    }
};



class LC148 {
public:
    ListNode* sortList(ListNode* head) {
        if (head == NULL || head->next == NULL) return head;
        ListNode *p = head->next, *q = head;
        
        while (p && p->next)
        {
            p = p->next->next;
            q = q->next;
        }

        ListNode* r = sortList(q->next);
        q->next = NULL;
        ListNode* l = sortList(head);
        ListNode* dump = new ListNode(-1);
        ListNode* sorted = dump;
        while (l && r)
        {
            if (l->val < r->val)
            {
                dump->next = l;
                l = l->next;
            }
            else
            {
                dump->next = r;
                r = r->next;
            }
            dump = dump->next;
        }

        if (l == NULL) dump->next = r;
        else dump->next = l;

        return sorted->next;
    }
};

class LC23 {
public:
    /**
     * 分治法
    */
    ListNode* mergeTwoLists(ListNode* la, ListNode* lb)
    {
        ListNode* head = new ListNode(-1);
        ListNode* dump = head;
        while (la && lb)
        {
            if (la -> val < lb -> val)
            {
                dump -> next = la;
                la = la -> next;
            }
            else
            {
                dump -> next = lb;
                lb = lb -> next;
            }
            dump = dump -> next;
        }
        if (la) dump -> next = la;
        else dump -> next = lb;
        return head -> next;
    }
    ListNode* splitMerge(vector<ListNode*>& lists, int l, int r)
    {
        if (l == r)
        {
            return lists[l];
        }
        if (l > r)
        {
            return nullptr;
        }
        int mid = (r - l) / 2 + l;
        return mergeTwoLists(splitMerge(lists, l, mid), splitMerge(lists, mid + 1, r));
    }
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        return splitMerge(lists, 0, lists.size() - 1); 
    }
};

class LC25 {
public:
    // 翻转一个子链表，并且返回新的头与尾
    pair<ListNode*, ListNode*> myReverse(ListNode* head, ListNode* tail) {
        ListNode* prev = tail->next;
        ListNode* p = head;
        while (prev != tail) {
            ListNode* nex = p->next;
            p->next = prev;
            prev = p;
            p = nex;
        }
        return {tail, head};
    }
     ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode* hair = new ListNode(0);
        hair->next = head;
        ListNode* pre = hair;

        while (head) {
            ListNode* tail = pre;
            // 查看剩余部分长度是否大于等于 k
            for (int i = 0; i < k; ++i) {
                tail = tail->next;
                if (!tail) {
                    return hair->next;
                }
            }
            ListNode* nex = tail->next;
            pair<ListNode*, ListNode*> result = myReverse(head, tail);
            head = result.first;
            tail = result.second;
            // 把子链表重新接回原链表
            pre->next = head;
            tail->next = nex;
            pre = tail;
            head = tail->next;
        }

        return hair->next;
    }
};

class LC25_recur {
public:
    /** 反转区间 [a, b) 的元素，注意是左闭右开 */
    ListNode* reverse(ListNode* a, ListNode* b) {
        ListNode *pre, *cur, *nxt;
        pre = NULL; cur = a; nxt = a;
        while (cur != b) {
            nxt = cur->next;
            // 逐个结点反转
            cur->next = pre;
            // 更新指针位置
            pre = cur;
            cur = nxt;
        }
        // 返回反转后的头结点
        return pre;
    }
    ListNode* reverseKGroup(ListNode* head, int k) {
        if (head == NULL) return NULL;
        // 区间 [a, b) 包含 k 个待反转元素
        ListNode *a, *b;
        a = b = head;
        for (int i = 0; i < k; i++) {
            // 不足 k 个，不需要反转，base case
            if (b == NULL) return head;
            b = b->next;
        }
        // 反转前 k 个元素
        ListNode* newHead = reverse(a, b);
        // 递归反转后续链表并连接起来
        a->next = reverseKGroup(b, k);
        return newHead;
    }
};

class LC92 {
public:
/**
 * https://labuladong.gitbook.io/algo/shu-ju-jie-gou-xi-lie/shou-ba-shou-shua-lian-biao-ti-mu-xun-lian-di-gui-si-wei/di-gui-fan-zhuan-lian-biao-de-yi-bu-fen
*/
    ListNode* successor = nullptr; // 后驱节点

    // 反转以 head 为起点的 n 个节点，返回新的头结点
    ListNode* reverseN(ListNode* head, int n) {
        if (n == 1) { 
            // 记录第 n + 1 个节点
            successor = head->next;
            return head;
        }
        // 以 head.next 为起点，需要反转前 n - 1 个节点
        ListNode* last = reverseN(head->next, n - 1);

        head->next->next = head;
        // 让反转之后的 head 节点和后面的节点连起来
        head->next = successor;
        return last;
    }
    
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        // base case
        if (left == 1) {
            return reverseN(head, right);
        }
        // 前进到反转的起点触发 base case
        head->next = reverseBetween(head->next, left - 1, right - 1);
        return head;
    }
};

class LC234 {
public:
    ListNode* reverseList(ListNode* node)
    {
        ListNode *p = NULL, *q = node, *r = new ListNode(-1);
        while (q)
        {
            r = q -> next;
            q -> next = p;
            p = q;
            q = r;
        }
        return p; 
    }
    bool isPalindrome(ListNode* head) {
        if (head == NULL) return true;
        ListNode *p = head, *q = head;
        while (q && q -> next)
        {
            p = p -> next;
            q = q -> next -> next;
        }

        q = reverseList(p);
        p = head;
        while (q)
        {
            if (p -> val != q -> val)
            {
                return false;
            }
            q = q -> next;
            p = p -> next;
        }
        return true;
    }
};