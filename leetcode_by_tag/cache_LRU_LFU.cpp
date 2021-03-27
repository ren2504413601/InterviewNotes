#include<bits/stdc++.h>
using namespace std;

/**
 * 常见缓存算法
 * - LRU LRU (Least recently used) 最近最少使用，
 *      如果数据最近被访问过，那么将来被访问的几率也更高
 * - LFU (Least frequently used) 最不经常使用，
 *      如果一个数据在最近一段时间内使用次数很少，那么在将来一段时间内被使用的可能性也很小
 * - FIFO (Fist in first out) 先进先出，
 *      如果一个数据最先进入缓存中，则应该最早淘汰掉
*/

/**
 * 146. LRU缓存机制
 * LRUNode -- 双向链表
 * unordered_map<int, LRUNode*> -- Hash 双向链表 key -> LRUNode(key, val)
 * 当缓存容量已满，我们不仅仅要删除最后一个 LRUNode 节点，
 * 还要把 map 中映射到该节点的 key 同时删除，而这个 key 只能由 LRUNode 得到。
 * 如果 LRUNode 结构中只存储 val，那么我们就无法得知 key 是什么，就无法删除 map 中的键，造成错误
 * https://leetcode-cn.com/problems/lru-cache/solution/lru-ce-lue-xiang-jie-he-shi-xian-by-labuladong/
 */ 
struct LRUNode
{
    int key, value;
    LRUNode* pre;
    LRUNode* next;
    LRUNode(int _key, int _value) : key(_key), value(_value), pre(nullptr), next(nullptr){}
};

class LRUCache 
{
public:
    int size;
    LRUNode* head;
    LRUNode* tail;
    unordered_map<int, LRUNode*> umap; // Hash 双向链表 key -> LRUNode(key, val)
    LRUCache(int capacity)
    {
        size = capacity;
        head = nullptr;
        tail = nullptr;
    }
    
    int get(int key)
    {
        auto it = umap.find(key);
        if (it != umap.end())
        {
            LRUNode* p = it ->second;
            // 这里仅删除 双向链表 中的节点，不必删除哈希表中的 节点
            remove(p);
            setHead(p);
            return p -> value;
        }
        else
        {
            return -1;
        }
    }
    // 将key-value值存入缓冲区
    void put(int key, int value)
    {
        auto it = umap.find(key);
        if (it != umap.end()) // key 已存在
        {
            LRUNode* cur = it -> second;
            cur -> value = value;
            // 这里仅删除 双向链表 中的节点，不必删除哈希表中的 节点
            remove(cur);
            // 将节点插入到缓冲区的头部
            setHead(cur);
        }
        else
        {
            LRUNode* cur = new LRUNode(key, value);
            if (umap.size() >= size) // 缓存达到上限
            {
                auto ta = umap.find(tail->key);
                // 同时移除链表尾节点和 Hash table
                remove(tail);
                umap.erase(ta);
            }
            // 更新当前节点为头结点
            setHead(cur);
            umap[key] = cur;
        }
    }
    // 将当前节点设置为头结点
    void setHead(LRUNode* cur)
    {
        cur -> next = head;
        if (head != nullptr)
        {
            head -> pre = cur;
        }
        head = cur;
        if (tail == nullptr) // 说明原来链表为空
        {
            tail = head;
        }
    }
    // 删除当前节点
    void remove(LRUNode* cur)
    {
        if (cur == head)
        {
            head = head -> next;
        }
        else if (cur == tail)
        {
            tail = cur -> pre;
        }
        else
        {
            cur -> pre -> next = cur -> next;
            cur -> next -> pre = cur -> pre; 
        }
    }
};

void LRUTest()
{
    cout << "LRU test:" << endl;
    LRUCache cache = LRUCache(2);

    cache.put(1, 1);
    cache.put(2, 2);
    cout << cache.get(1) << endl;       // 返回  1
    cache.put(3, 3);    // 该操作会使得关键字 2 作废
    cout << cache.get(2) << endl;       // 返回 -1 (未找到)
    cache.put(4, 4);    // 该操作会使得关键字 1 作废
    cout << cache.get(1) << endl;       // 返回 -1 (未找到)
    cout << cache.get(3) << endl;       // 返回  3
    cout << cache.get(4) << endl;       // 返回  4
}

/**
 * 460. LFU缓存
 * LFUNode(int _cnt, int _time, int _key, int _value) -- 双向链表
 * unordered_map<int, LFUNode> -- Hash 双向链表 key -> LFUNode
 * https://leetcode-cn.com/problems/lfu-cache/solution/lfuhuan-cun-by-leetcode-solution/
*/
struct LFUNode 
{
    int cnt, time, key, value;

    LFUNode(int _cnt, int _time, int _key, int _value):cnt(_cnt), time(_time), key(_key), value(_value){}
    
    bool operator < (const LFUNode& rhs) const {
        return cnt == rhs.cnt ? time < rhs.time : cnt < rhs.cnt;
    }
};
class LFUCache 
{
    // 缓存容量，时间戳
    int capacity, time;
    unordered_map<int, LFUNode> key_table;
    set<LFUNode> S;
public:
    LFUCache(int _capacity) 
    {
        capacity = _capacity;
        time = 0;
        key_table.clear();
        S.clear();
    }
    
    int get(int key) 
    {
        if (capacity == 0) return -1;
        auto it = key_table.find(key);
        // 如果哈希表中没有键 key，返回 -1
        if (it == key_table.end()) return -1;
        // 从哈希表中得到旧的缓存
        LFUNode cache = it -> second;
        // 从平衡二叉树中删除旧的缓存
        S.erase(cache);
        // 将旧缓存更新
        cache.cnt += 1;
        cache.time = ++time;
        // 将新缓存重新放入哈希表和平衡二叉树中
        S.insert(cache);
        it -> second = cache;
        return cache.value;
    }
    
    void put(int key, int value) 
    {
        if (capacity == 0) return;
        auto it = key_table.find(key);
        if (it == key_table.end()) 
        {
            // 如果到达缓存容量上限
            if (key_table.size() == capacity) 
            {
                // 从哈希表和平衡二叉树中删除最近最少使用的缓存
                key_table.erase(S.begin() -> key);
                S.erase(S.begin());
            }
            // 创建新的缓存
            LFUNode cache = LFUNode(1, ++time, key, value);
            // 将新缓存放入哈希表和平衡二叉树中
            key_table.insert(make_pair(key, cache));
            S.insert(cache);
        }
        else 
        {
            // 这里和 get() 函数类似
            LFUNode cache = it -> second;
            S.erase(cache);
            cache.cnt += 1;
            cache.time = ++time;
            cache.value = value;
            S.insert(cache);
            it -> second = cache;
        }
    }
};

void LFUTest()
{
    cout << "LFU test:" << endl;
    LFUCache cache = LFUCache(2);

    cache.put(1, 1);
    cache.put(2, 2);
    cout << cache.get(1) << endl;       // 返回 1
    cache.put(3, 3);    // 去除 key 2
    cout << cache.get(2) << endl;       // 返回 -1 (未找到key 2)
    cout << cache.get(3) << endl;       // 返回 3
    cache.put(4, 4);    // 去除 key 1
    cout << cache.get(1) << endl;       // 返回 -1 (未找到 key 1)
    cout << cache.get(3) << endl;       // 返回 3
    cout << cache.get(4) << endl;       // 返回 4
}


int main()
{
    LRUTest();
    LFUTest();
    system("pause");
    return 0;
}