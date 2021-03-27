#include<bits/stdc++.h>
using namespace std;
typedef long long ll;

/**
 * 94. 二叉树的中序遍历
 * 102. 二叉树的层序遍历
 * 95. 不同的二叉搜索树 II
 * 96. 不同的二叉搜索树
 * 110. 平衡二叉树
 * 103. 二叉树的锯齿形层序遍历
 * 104. 二叉树的最大深度
 * 226. 翻转二叉树
 * 114. 二叉树展开为链表
 * 116. 填充每个节点的下一个右侧节点指针
 * 105. 从前序与中序遍历序列构造二叉树
 * 106. 从中序与后序遍历序列构造二叉树
 * 538. 把二叉搜索树转换为累加树
 * 1038. 把二叉搜索树转换为累加树
 * 654. 最大二叉树
 * 652. 寻找重复的子树
 * 230. 二叉搜索树中第K小的元素
 * 700. 二叉搜索树中的搜索
 * 701. 二叉搜索树中的插入操作
 * 98. 验证二叉搜索树  
 * 450. 删除二叉搜索树中的节点
 * 297. 二叉树的序列化与反序列化
 * 236. 二叉树的最近公共祖先
*/

 // Definition for a binary tree node.
 struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 };




/**
 * 使用队列结构实现层序遍历
 * 将 vector 存储的二叉树结构转换成标准的二叉树 
*/
TreeNode* genTreeByVector(vector<string>& array)
{
	queue<TreeNode*> que;
	
	if (array.empty() || array[0] == "null")
	return NULL;

	TreeNode* root = new TreeNode(stoi(array[0]));
	que.push(root);

	bool left_or_not = true;
	for (int i = 1; i < array.size(); ++i)
	{
		if (array[i] == "null")
		{
			if (left_or_not)
			{
				left_or_not = false;
				que.front()->left = NULL;
			}
			else
			{
				left_or_not = true;
				que.front()->right = NULL;
				que.pop();
			}
		}
		else
		{
			if (left_or_not)
			{
				left_or_not = false;

				TreeNode* tnode = new TreeNode(stoi(array[i]));
				que.front()->left = tnode;
				que.push(tnode);
			}
			else
			{
				left_or_not = true;

				TreeNode* tnode = new TreeNode(stoi(array[i]));
				que.front()->right = tnode;
				que.push(tnode);
				que.pop();
			}
		}
	}
	return root;
}

/**
 * 二叉树中序遍历
*/
class LC94 {
private:
    vector<int> ans;
public:
    void InorderTraversalHelper(TreeNode* root)
    {
        if (root)
        {
            InorderTraversalHelper(root->left);
            ans.push_back(root->val);
            InorderTraversalHelper(root->right);
        }
    }
    vector<int> inorderTraversal(TreeNode* root) {
        InorderTraversalHelper(root);
        return ans;
    }
	void sol_test()
	{
		// TreeNode* n1 = new TreeNode(1);
		// TreeNode* n2 = new TreeNode(2);
		// TreeNode* n3 = new TreeNode(3);
		// n1->left = NULL;
		// n1->right = n2;
		// n2->left = n3;
		vector<string> array = {"1", "null", "2", "3"};
		TreeNode* root = genTreeByVector(array);

		inorderTraversal(root);
		for (int& tv:ans) printf("%d\t", tv);
	}
};

/**
 * 二叉树层序遍历
*/
class LC102 {
public:
    vector<vector<int>> ans;
    void levelOrderTravel(TreeNode* node, int level)
    {
        if (level == ans.size()) ans.push_back({});
        ans[level].push_back(node->val);
        if (node->left) levelOrderTravel(node->left, level + 1);
        if (node->right) levelOrderTravel(node->right, level + 1);
    }
    vector<vector<int>> levelOrder(TreeNode* root) {
        if (!root) return ans;
        levelOrderTravel(root, 0);
        return ans;
        
    }
};

class LC95 {
public:
    vector<TreeNode*> genTree(int start, int end)
    {
        vector<TreeNode*> gTree;
        if (start > end) return {NULL};
        for (int i = start; i <= end; ++i)
        {
            vector<TreeNode*> ltree = genTree(start, i - 1);
            vector<TreeNode*> rtree = genTree(i + 1, end);
            for (TreeNode* l : ltree)
            for (TreeNode* r : rtree)
            {
                TreeNode* node = new TreeNode(i);
                node->left = l;
                node->right = r;
                gTree.push_back(node);
            }
        }
        return gTree;
    }
    vector<TreeNode*> generateTrees(int n) {
        vector<TreeNode*> node;
        if (n == 0) return node;
        return genTree(1, n);
    }
};

class LC96 {
public:
    int numTrees(int n) {
        if (n == 0 || n == 1) return 1;
        int Cantlan[n + 1];
        memset(Cantlan, 0, sizeof(Cantlan));
        Cantlan[0] = 1; Cantlan[1] = 1;
        
        for (int i = 2; i <= n; ++i)
        {
            for (int j = 1; j <= i; ++j)
            {
                Cantlan[i] += Cantlan[j - 1] * Cantlan[i - j];
            }
        }
        return Cantlan[n];

    }
};

class LC98 {
public:
    bool fun(struct TreeNode* root, long low, long high) {
        if (root == nullptr) return true;
        long num = root->val;
        if (num <= low || num >= high) return false;
        return fun(root->left, low, num) && fun(root->right, num, high);
    }
    bool isValidBST(struct TreeNode* root){
        return fun(root, LONG_MIN, LONG_MAX);
    }
};
 

/**
 * 自顶向下的递归
 */
class LC110_1 
{
public:
    int getHeight(TreeNode* node)
    {
        if (node == NULL)
        {
            return 0;
        }
        return 1+max(getHeight(node->left), getHeight(node->right));
    }
    bool isBalanced(TreeNode* root) {
        if (root == NULL) return true;
        return abs(getHeight(root->left)-getHeight(root->right))<2 &&
                isBalanced(root->left) &&
                isBalanced(root->right);
    }
};
/**
 * 自底向上的递归
 * 
 */
class LC110_2 
{
public:
    bool balancedHelper(TreeNode* node, int& depth)
	{
		if (node == NULL) // 空树视为平衡二叉树（深度为0）
		{
			depth = 0;
			return true;
		}
		// 这里使用 l, r分别记录当前node左右子树的深度
		// Note: 由于是自底向上的（深度使用的引用格式）
		// 所以最终得到的l和r是从最后一层节点一步步传上来的
		// 这与自顶向下相比减少了深度的重复计算
		int l = 0, r = 0;
		if (balancedHelper(node->left, l) && balancedHelper(node->right, r) && abs(l-r) < 2)
		{
			depth = 1+max(l, r);
			return true;
		}
		return false;
	}
    bool isBalanced(TreeNode* root) 
    {
        if (root == NULL) return true;
		// 初始的深度初始化为0
        int h = 0;
        return balancedHelper(root, h);
    }
};
/**
 * 使用队列结构生成 binary tree
 */  

TreeNode* generateBinaryTree(vector<string>& array)
{
    queue<TreeNode*> mem;

    if(array.empty() || array[0] == "NULL")
    {
        return NULL;
    }
    TreeNode* root = new TreeNode(stoi(array[0]));
    mem.push(root);

    bool left_or_not = true;
    for(int i = 1; i < array.size(); ++i)
    {
        if(array[i] != "null")
        {
            if(left_or_not == true)
            {
                left_or_not = false;

                TreeNode* tmpNode = new TreeNode(stoi(array[i]));
                mem.front()->left = tmpNode;
                mem.push(tmpNode);
            }
            else
            {
                left_or_not = true;

                TreeNode* tmpNode = new TreeNode(stoi(array[i]));
                mem.front()->right = tmpNode;
                mem.push(tmpNode);
                mem.pop();
            }
        }
        else
        {
            if(left_or_not == true)
            {
                left_or_not = false;

                mem.front()->left = NULL;
            }
            else
            {
                left_or_not = true;

                mem.front()->right = NULL;
                mem.pop();
            }
        }
    }

    return root;
}

void preCross(TreeNode* node, vector<int>& res)
{
    if (node == NULL) return;

    res.push_back(node->val);
    preCross(node->left, res);
    preCross(node->right, res);
}


class LC103_1 {
public:
    vector<vector<int>> ans;
    void helper(TreeNode* root, int level)
    {
        if (ans.size() == level) ans.push_back({});
        ans[level].push_back(root->val);
        if (root->left) helper(root->left, level + 1);
        if (root->right) helper(root->right, level + 1);
        
    }
    void reverseOddlevel()
    {
        for (int i = 1; i < ans.size(); i += 2)
        {
            vector<int> tmp;
            int tlen = ans[i].size();
            for (int j = 0; j < tlen; ++j)
            {
                tmp.push_back(ans[i][tlen - 1 - j]);
            }
            ans[i] = tmp;
        }
    }
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        if (!root) return ans;
        helper(root, 0);
        reverseOddlevel();
        return ans;
    }
};


class LC103_2 {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        if (!root) return ans;
        queue<TreeNode*> que;
        que.push(root);
        bool left_or_not = true;

        while (!que.empty())
        {
            int size = que.size();
            deque<int> tAns;
            while (size--)
            {
                TreeNode* tnode = que.front(); que.pop();
                if (left_or_not)
                {
                    tAns.push_back(tnode->val);
                }
                else
                {
                    tAns.push_front(tnode->val);
                }
                if (tnode->left) que.push(tnode->left);
                if (tnode->right) que.push(tnode->right);
            }
            left_or_not = !left_or_not;
            ans.push_back(vector<int> (tAns.begin(), tAns.end()));
        }

        return ans;
    }
};


class LC104 {
public:
    int maxD = 0;
    void dfs(TreeNode* root, int tdep)
    {
        maxD = max(maxD, tdep);
        if (root->left) dfs(root->left, tdep + 1);
        if (root->right) dfs(root->right, tdep + 1);
    }
    int maxDepth(TreeNode* root) {
        if (!root) return maxD;
        dfs(root, 1);
        return maxD;
    }
};

class LC105 {
public:
/*
 ** 递归
 ** 先从先序序列判断根节点，再由根节点确定左右子树
 ** 删除先序的根节点遍历下一位（这里通过pre_idx++实现）,
 ** 递归操作，直至找到叶子节点终止(此时叶子节点的左右区间是空集)
 
 ** 这里需要注意左右区间是左闭右开时,终止条件`left==right`.
 ** 左闭右闭情形，终止条件是`left>right`
 */
    vector<int> preorder;
    unordered_map<int, int> inorder_map;
    int preIdx = 0;
    TreeNode* helper(int l, int r)
    {
        if (l == r)
        {
            return nullptr;
        }
        int tval = preorder[preIdx];
        TreeNode* tnode = new TreeNode(tval);
        int inorder_idx = inorder_map[tval];
        preIdx++;
        tnode->left = helper(l, inorder_idx);
        tnode->right = helper(inorder_idx + 1, r);

        return tnode;
    }
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        this->preorder = preorder;
        for (int i = 0; i < inorder.size(); ++i) inorder_map[inorder[i]] = i;
        return helper(0, inorder.size());
    }
};

class LC106 {
/*
 ** 递归
 ** 先从后序序列逆向判断根节点，再由根节点确定左右子树
 ** 注意：这里应该先构建右子树，在构建左子树，负责会出现数组越界
 ** 这是因为post_idx在后序遍历数组往前移动（post_idx--）的时候，先指向右子节点的值
 
 ** 删除先序的根节点遍历下一位（这里通过pre_idx++实现）,
 ** 递归操作，直至找到叶子节点终止(此时叶子节点的左右区间是空集)
 
 ** 这里需要注意左右区间是左闭右开时,终止条件`left==right`.
 ** 左闭右闭情形，终止条件是`left>right`
 */
    vector<int> postorder;
    map<int,int> inorder_map;
    int post_idx;
public:
    TreeNode* treeBuild(int left,int right)
    {
        if(left==right) return NULL;
        TreeNode* root=new TreeNode(postorder[post_idx]);
        int inorder_idx=inorder_map[postorder[post_idx]];
        post_idx--;
        // 构建左右子树
        root->right=treeBuild(inorder_idx+1,right);
        root->left=treeBuild(left,inorder_idx);
        return root;
    }
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        this->postorder=postorder;
        post_idx=postorder.size()-1;
        for(int i=0;i<inorder.size();i++) inorder_map[inorder[i]]=i;
        return treeBuild(0,inorder.size());
    }
};


int main()
{
    // vector<string> arr = {
    //     "3", "9", "20", "null", "null", "15", "7"
    // };
    vector<string> arr = {
        "1", "2", "2", "3", "3", "null", "null", "4", "4"
    };
    cout<< "array size ="<<arr.size() << endl;
    TreeNode* root = generateBinaryTree(arr);
    vector<int> pre_cross;
    preCross(root, pre_cross);
    cout << "pre cross :";
    for (int& val:pre_cross)
    {
        cout << val << '\t';
    }
    
    LC110_1 sol;
    LC110_2 sol1;
    cout <<endl <<"is balance tree by sol1(from down to top):"<< sol.isBalanced(root);
    cout <<endl <<"is balance tree by sol2(from top to down):"<< sol1.isBalanced(root);
    system("pause");
	return 0;
}

class LC226 {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (root == nullptr) return nullptr;
        TreeNode* left = invertTree(root->left);
        TreeNode* right = invertTree(root->right);
        root->left = right;
        root->right = left;
        return root;
    }
};

class LC114 {
public:
    void flatten(TreeNode* root) {
        while (root)
        {
            if (root -> left == NULL)
            {
                root = root -> right;
            }
            else
            {
                TreeNode* pre = root -> left;
                while (pre -> right)
                {
                    pre = pre -> right;
                }
                pre -> right = root -> right;
                root -> right = root -> left;
                root -> left = NULL; 
                root = root -> right;
            }
        }
    }
};

class LC116 {
public:
    Node* connect(Node* root) {
        if(!root) return root;
        if(root->right) 
        {
            root->left->next=root->right;
            root->right->next=(root->next)? root->next->left:NULL;
        }
        connect(root->left);
        connect(root->right);
        return root;    
    }
};

class LC538 {
    int sum = 0;
public:
    /**
     * 右中左遍历
    */
    TreeNode* convertBST(TreeNode* root) {
        if (root)
        {
            convertBST(root -> right);
            sum += root -> val;
            root -> val = sum;
            convertBST(root -> left);
        }
        return root;
    }
};

class LC1038 {
    int sum = 0;
public:
    /**
     * 右中左遍历
    */
    TreeNode* bstToGst(TreeNode* root) {
        if (root)
        {
            bstToGst(root -> right);
            sum += root -> val;
            root -> val = sum;
            bstToGst(root -> left);
        }
        return root;
    }
};

class LC654 {
public:
    TreeNode* dfs(vector<int>& nums, int l, int r)
    {
        if (l == r)
        {
            return nullptr;
        }
        int maxIdx = l;
        int maxVal = INT_MIN;
        for (int i = l; i < r; ++i)
        {
            if (nums[i] > maxVal)
            {
                maxVal = nums[i];
                maxIdx = i;
            }
        }

        TreeNode* root = new TreeNode(maxVal);
        root->left = dfs(nums, l, maxIdx);
        root->right = dfs(nums, maxIdx + 1, r);
        return root;
    }
    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        return dfs(nums, 0, nums.size());
    }
};

class LC652 {
public:
/**
 * https://leetcode-cn.com/problems/find-duplicate-subtrees/solution/652er-cha-shu-de-xu-lie-hua-ji-xian-gen-oqac6/
 * 二叉树的序列化及先根后根遍历解法
*/
    unordered_map<string, int> umap;
    vector<TreeNode*> ans;
    string encode(TreeNode* root)
    {
        if (root == nullptr) return "#";
        string lcode = encode(root->left);
        string rcode = encode(root->right);
        string code = to_string(root->val) + "," + lcode + "," + rcode;
        umap[code]++;
        if (umap[code] == 2)
        {
            ans.push_back(root);
        }
        return code;
    }
    vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
        if (root == nullptr) return ans;
        encode(root);
        return ans;
    }
};

class LC230 {
public:
    int cnt = 0;
    int ans;
    void inorder(TreeNode* root, int k)
    {
        if (root == nullptr) return;
        inorder(root->left, k);
        ++cnt;
        if (cnt == k)
        {
            ans = root->val;
            return;
        }
        inorder(root->right, k);
    }
    int kthSmallest(TreeNode* root, int k) {
        inorder(root, k);
        return ans;
    }
};

class LC700 {
public:
    TreeNode* searchBST(TreeNode* root, int val) {
        if (root == nullptr) return nullptr;
        if (root->val == val) return root;
        TreeNode* ls = searchBST(root->left, val);
        TreeNode* rs = searchBST(root->right, val);
        
        if (ls) return ls;
        else if (rs) return rs;
        else return nullptr;
    }
};

class LC701 {
public:
/**
 * 总可以找到一个根节点完成添加
*/
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        if (root == nullptr) return new TreeNode(val);
        TreeNode* pos = root;
        while (pos)
        {
            if (pos->val < val) // 处于右子树
            {
                if (pos->right == nullptr)
                {
                    pos->right = new TreeNode(val);
                    break;
                }
                else
                {
                    pos = pos->right;
                }
            }
            else
            {
                if (pos->left == nullptr)
                {
                    pos->left = new TreeNode(val);
                    break;
                }
                else
                {
                    pos = pos->left;
                }
            }
        }
        return root;
    }
};

class LC450 {
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
        if(root == nullptr) return root;//第一种情况：没找到删除的节点，遍历到空节点直接返回
        if(root->val == key)
        {
            //第二种情况：左右孩子都为空（叶子节点），直接删除节点，返回NULL为根节点
            //第三种情况：其左孩子为空，右孩子不为空，删除节点，右孩子补位，返回右孩子为根节点
            if(root->left == nullptr) return root->right;
            //第四种情况：其右孩子为空，左孩子不为空，删除节点，左孩子补位，返回左孩子为根节点
            else if(root->right == nullptr) return root->left;
            //第五种情况：左右孩子节点都不为空，则将删除节点的左子树放到删除节点的右子树的最左面节点的左孩子的位置
            //并返回删除节点右孩子为新的根节点
            else{
                TreeNode* cur = root->right;//找右子树最左面的节点
                while(cur->left != NULL)
                {
                    cur = cur->left;
                }
                cur->left = root->left;//把要删除的节点左子树放在cur的左孩子的位置
                TreeNode* tmp = root;  //把root节点保存一下，下面来删除
                root = root->right;    //返回旧root的右孩子作为新root
                delete tmp;            //释放节点内存
                return root;
            }
        }
        if(root->val > key) root->left = deleteNode(root->left, key);
        if(root->val < key) root->right = deleteNode(root->right, key);
        return root;
    }
};

class LC297 {
public:
    string serialize(TreeNode* root) {
        if (!root) return "X";
        auto l = "(" + serialize(root->left) + ")";
        auto r = "(" + serialize(root->right) + ")";
        return  l + to_string(root->val) + r;
    }

    inline TreeNode* parseSubtree(const string &data, int &ptr) {
        ++ptr; // 跳过左括号
        auto subtree = parse(data, ptr);
        ++ptr; // 跳过右括号
        return subtree;
    }

    inline int parseInt(const string &data, int &ptr) {
        int x = 0, sgn = 1;
        if (!isdigit(data[ptr])) {
            sgn = -1;
            ++ptr;
        }
        while (isdigit(data[ptr])) {
            x = x * 10 + data[ptr++] - '0';
        }
        return x * sgn;
    }

    TreeNode* parse(const string &data, int &ptr) {
        if (data[ptr] == 'X') {
            ++ptr;
            return nullptr;
        }
        auto cur = new TreeNode(0);
        cur->left = parseSubtree(data, ptr);
        cur->val = parseInt(data, ptr);
        cur->right = parseSubtree(data, ptr);
        return cur;
    }

    TreeNode* deserialize(string data) {
        int ptr = 0;
        return parse(data, ptr);
    }
};

class LC236 {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (root == p || root == q)
        {
            return root;
        }

        if (root)
        {
            TreeNode* lnode = lowestCommonAncestor(root -> left, p, q);
            TreeNode* rnode = lowestCommonAncestor(root -> right, p, q);
            if (lnode && rnode)
            {
                return root;
            }
            else if (lnode == NULL)
            {
                return rnode;
            }
            else if (rnode == NULL)
            {
                return lnode;
            }
        }
        return NULL;
        
    }
};