/**
 * 172. 阶乘后的零 
 * 50. Pow(x, n)
*/

class LC172 {
public:
    int trailingZeroes(int n) {
        int res = 0;
        for (int d = n; d / 5 > 0; d = d / 5)
        {
            res += d / 5;
        }
        return res;
    }
};

class LC50 {
public:
    double quickMul(double x, long long n)
    {
        double xw = x;
        double ans = 1;
        while (n)
        {
            if (n % 2 == 1)
            {
                ans *= xw;
            }
            xw *= xw;
            n /= 2;
        }
        return ans;
    }
    double myPow(double x, int n) {
        long long N = n;
        if (n >= 0) return quickMul(x, N);
        else return 1 / quickMul(x, -N);
    }
};
