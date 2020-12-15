#include<bits/stdc++.h>
using namespace std;

class RingQueue
{
public:
    RingQueue(int size = 10)
    {
        MAX_LEN = size;
        mFront = mRear = 0;
        mpQue = new int[MAX_LEN];
    }
    RingQueue(const RingQueue &src)
    {
        MAX_LEN = src.MAX_LEN;
        mFront = src.mFront;
        mRear = src.mRear;
        mpQue = new int[MAX_LEN];
        for (int i = mFront; i < mRear; i = (i + 1) % MAX_LEN)
        {
            mpQue[i] = src.mpQue[i];
        }
    }
    void operator=(const RingQueue &src)
    {
        cout << "operator=" << endl;
        // 先排除自赋值
        if (this == &src)
        return;
        // 先释放当前对象原来占用的外部资源
        delete[]mpQue;

        // 重新给当前对象分配外部资源空间并进行数据拷贝
        MAX_LEN = src.MAX_LEN;
        mFront = src.mFront;
        mRear = src.mRear;
        mpQue = new int[MAX_LEN];
        for (int i = mFront; i < mRear; i = (i + 1) % MAX_LEN)
        {
            mpQue[i] = src.mpQue[i];
        }
    }
    ~RingQueue()
    {
        delete []mpQue;
        mpQue = NULL;
    }
    int size()
    {
        return mRear - mFront;
    }
    void enqueue(int data)//入队
    {
        if (full())
        {
            int *pTmp = new int[2 * MAX_LEN];
            for (int i = mFront, j=0; i < mRear; i = (i + 1) % MAX_LEN,j++)
            {
                pTmp[j] = mpQue[i];
            }
            mFront = 0;
            mRear = MAX_LEN;
            MAX_LEN *= 2;
            delete []mpQue;
            mpQue = pTmp;
        }
        mpQue[mRear] = data;
        mRear = (mRear + 1) % MAX_LEN;
    }
    void dequeue()//出队
    {
        if (empty())
        return;
        mFront = (mFront + 1) % MAX_LEN;
    }
    int front()
    {
        if (empty())
        throw "queue is empty!";
        return mpQue[mFront];
    }
    int back()
    {
        if (empty())
        throw "queue is empty!";
        return mpQue[(mRear + MAX_LEN - 1) % MAX_LEN];
    }
    bool full()
    { 
        return (mRear + 1) % MAX_LEN == mFront; 
    }
    bool empty()
    {
        return mRear == mFront; 
    }
private:
    int *mpQue;
    int mFront;
    int mRear;
    int MAX_LEN;
};
int main()
{
    RingQueue* rq = new RingQueue(100);
    rq->enqueue(2);
    cout << rq->size();
    system("pause");
    return 0;
}

