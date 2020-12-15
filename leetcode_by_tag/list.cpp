#include<bits/stdc++.h>
using namespace std;

struct ListNode
{
    int val;
    ListNode* next;
    ListNode(int _val) : val(_val)
    {}
};

/**
 * 翻转链表
*/
ListNode* reverseList(ListNode* node)
{
    ListNode *pre = NULL, *curr = node, *next = curr -> next;
    while (curr)
    {
        next = curr -> next;
        curr -> next = pre;
        pre = curr;
        curr = next;
    }
    return pre;
}
