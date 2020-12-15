/**
 * 单例模式
 * 作用：保证一个类只有一个实例，并提供一个访问它的全局访问点，
 * 使得系统中只有唯一的一个对象实例
 * 应用：常用于管理资源，如日志、线程池等
 * 单例模式的要点有三个：一是某个类只能有一个实例；二是它必须自行创建这个实例；
 * 三是它必须自行向整个系统提供这个实例
*/
#include<bits/stdc++.h>
using namespace std;

/**
 * https://blog.csdn.net/u010993820/article/details/80968933
*/
class Singleton_case1
{
public:
    static Singleton_case1& getInstance()
    {
        static Singleton_case1 instance;
        return instance;
    }
    void printTest()
    {
        cout<<"do something"<<endl;
    }
private:
    Singleton_case1(){}//防止外部调用构造创建对象
    Singleton_case1(Singleton_case1 const &singleton);//阻止拷贝创建对象
    Singleton_case1& operator=(Singleton_case1 const &singleton);//阻止赋值对象
};


/**
 * https://github.com/me115/design_patterns/tree/master/code/Singleton
*/
class Singleton_case2
{
public:
	virtual ~Singleton_case2();
	Singleton_case2 *m_Singleton;

	static Singleton_case2* getInstance();
	void singletonOperation();

private:
	static Singleton_case2 * instance;

	Singleton_case2();

};

Singleton_case2* Singleton_case2::instance = NULL;
Singleton_case2::Singleton_case2(){

}

Singleton_case2::~Singleton_case2(){
	delete instance;
}

Singleton_case2* Singleton_case2::getInstance(){
	if (instance == NULL)
	{
		instance = new Singleton_case2();
	}
	
	return  instance;
}


void Singleton_case2::singletonOperation(){
	cout << "singletonOperation" << endl;
}

int main()
{
    Singleton_case1& sg = Singleton_case1::getInstance();
    sg.printTest();
    // Singleton_case2 * sg = Singleton_case2::getInstance();
	// sg->singletonOperation();
    system("pause");
    return 0;
}