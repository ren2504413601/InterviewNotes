#include<bits/stdc++.h>
using namespace std;
typedef long long ll;


 // Definition for a binary tree node.
 struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 };
 

/**
 * 自顶向下的递归
 */
class Solution1 
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
class Solution 
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
    
    Solution sol;
    Solution1 sol1;
    cout <<endl <<"is balance tree by sol(from down to top):"<< sol.isBalanced(root);
    cout <<endl <<"is balance tree by sol1(from top to down):"<< sol1.isBalanced(root);
    system("pause");
	return 0;
}