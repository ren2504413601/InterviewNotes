#include<bits/stdc++.h>
using namespace std;

/**
 * 224. 基本计算器
 * 实现一个基本的计算器来计算一个简单的字符串表达式的值。
 * 字符串表达式可以包含左括号 ( ，右括号 )，加号 + ，减号 -，非负整数和空格
*/

class LC224 {
public:
    /**
     * 栈和不反转字符串
     * 每当我们遇到 + 或 - 运算符时，我们首先将表达式求值到左边，
     * 然后将正负符号保存到下一次求值。
     * 如果字符是左括号 (，我们将迄今为止计算的结果和符号添加到栈上，然后重新开始进行计算，
     * 就像计算一个新的表达式一样。
     * 如果字符是右括号 )，则首先计算左侧的表达式。
     * 则产生的结果就是刚刚结束的子表达式的结果。如果栈顶部有符号，则将此结果与符号相乘
    */
    int calculate(string s) {
        int n = s.size();
        stack<int> stk;
        int sign = 1, operate = 0, result = 0;
        for (char c : s)
        {
            if (isdigit(c))
            {
                operate = operate * 10 + int(c - '0');
            }
            else if (c == '+')
            {
                result += sign * operate;
                sign = 1;
                operate = 0;
            }
            else if (c == '-')
            {
                result += sign * operate;
                sign = -1;
                operate = 0;
            }
            else if (c == '(')
            {
                stk.push(result);
                stk.push(sign);
                sign = 1;
                result = 0;
            }
            else if (c == ')')
            {
                result += sign * operate; // 括号中的结果
                int sig = stk.top();
                stk.pop();
                int preRes = stk.top(); // 括号外的结果
                stk.pop();
                result *= sig;
                result += preRes;
                operate = 0;
            }
        }
        return result + sign * operate;
    }
};

/**
 * 227. 基本计算器 II
 * 实现一个基本的计算器来计算一个简单的字符串表达式的值。
 * 字符串表达式仅包含非负整数，+， - ，*，/ 四种运算符和空格  。 整数除法仅保留整数部分。
*/
class LC227 {
public:
    /**
     * 一次遍历 + switch 判断
     * 分情况来处理遍历，num 表示当前的数字，curr_res 表示当前的结果，res 为最终的结果，
     * op 为操作符号，初始化为 '+'。
     * 当遇到数字的时候，将 num 自乘以 10 并加上这个数字，这是由于可能遇到多位数，
     * 所以每次要乘以 10。

     * 如果遇到运算符号，或者是最后一个位置的字符时，我们根据 上一个op 的值对 num 
     * 进行分别的加减乘除的处理，结果保存到 当前结果curr_res 中。
     * 然后再次判读如果 上一个op 是加或减，或者是最后一个位置的字符时，将 当前结果curRes 
     * 加到最终结果 res 中，并且 curRes 重置为0。
     * 最后将当前运算字符c赋值给 op（注意这里只有当时最后一个位置的字符时，
     * 才有可能不是运算符号，不过也无所谓，因为遍历已经结束），num 也要重置为0
    */
    int calculate(string s) {
        int n = s.size(), res = 0, curr_res = 0, num = 0;
        char op = '+';
        for (int i = 0; i < n; ++i)
        {
            char c = s[i];
            if (isdigit(c))
            {
                num = num * 10 + int(c - '0');
            }
            if (c == '+' || c == '-' || c == '*' || c == '/' || i == n-1)
            {
                switch(op)
                {
                    case '+':
                        curr_res += num;
                        break;
                    case '-':
                        curr_res -= num;
                        break;
                    case '*':
                        curr_res *= num;
                        break;
                    case '/':
                        curr_res /= num;
                        break;
                }
                if (c == '+' || c == '-' || i == n-1)
                {
                    res += curr_res;
                    curr_res = 0;
                }
                op = c;
                num = 0;
            }
        }
        return res;

    }
};